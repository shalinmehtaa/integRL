import json
import torch
import random
import argparse
from typing import List, Dict, Any
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from ..grader_helpers import r1_zero_reward_fn
from ..sft.evals import format_prompt, get_ground_truth
from ..sft.train_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
    init_vllm,
    load_policy_into_vllm_instance,
)


def read_sft_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = list()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if "prompt" in ex and "response" in ex:
                rows.append({"prompt": ex["prompt"], "response": ex["response"]})
    return rows


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = list()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def make_batches(n: int, batch_size: int, shuffle: bool = True, seed: int = 1709):
    idxs = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
    for i in range(0, n, batch_size):
        yield idxs[i: i + batch_size]


def ensure_pad_token(tokenizer: PreTrainedTokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def train_sft(
    model_id: str,
    train_path: str,
    val_path: str,
    policy_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    seed: int = 1709,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 5e-6,
    weight_decay: float = 0.0,
    epochs: int = 1,
    eval_every_steps: int = 200,
    eval_n_examples: int = 64,
    eval_batch_size: int = 16,
    eval_max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: List[str] | None = None,
    eval_prompt_path: str = "prompts/r1_zero.prompt",
):
    torch.manual_seed(seed)

    # Load tokenizer and policy model (HF) on policy_device
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ensure_pad_token(tokenizer)
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    policy.resize_token_embeddings(len(tokenizer))
    policy.to(policy_device)
    policy.train()

    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # vLLM evaluator on eval_device
    llm = init_vllm(model_id=model_id, device=eval_device, seed=seed)

    # Data
    train_rows = read_sft_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path else []

    # Load evaluation prompt template
    with open(eval_prompt_path, "r") as f:
        eval_prompt_template = f.read()

    global_step = 0
    for epoch in range(epochs):
        for batch_idxs in make_batches(len(train_rows), batch_size, shuffle=True, seed=seed + epoch):
            # Gradient accumulation group
            optimizer.zero_grad(set_to_none=True)

            # Prepare microbatch text
            micro_prompts: List[str] = [train_rows[i]["prompt"] for i in batch_idxs]
            micro_outputs: List[str] = [train_rows[i]["response"] for i in batch_idxs]

            # Tokenize and build masks
            toks = tokenize_prompt_and_output(micro_prompts, micro_outputs, tokenizer)
            input_ids = toks["input_ids"].to(policy_device)
            labels = toks["labels"].to(policy_device)
            response_mask = toks["response_mask"].to(policy_device)

            # Compute per-token log-probs from the policy
            scored = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = scored["log_probs"] # (B, T)

            # Normalize by count of response tokens for stable scale
            normalize_constant = float(response_mask.sum().item()) if response_mask.numel() > 0 else 1.0

            # One microbatch step (includes loss.backward())
            _, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=normalize_constant,
            )

            global_step += 1

            # Optimizer step after grad accumulation window
            if global_step % grad_accum_steps == 0:
                optimizer.step()
                policy.zero_grad(set_to_none=True)

            # Periodic evaluation with vLLM on eval_device
            if val_rows and (global_step % eval_every_steps == 0):
                # Sync policy weights into vLLM
                load_policy_into_vllm_instance(policy, llm)

                n_eval = min(eval_n_examples, len(val_rows))
                # Construct prompts via template and normalize GTs
                eval_prompts: List[str] = []
                eval_ground_truths: List[Any] = []
                for i in range(n_eval):
                    ex = val_rows[i]
                    prompt = format_prompt(eval_prompt_template, {"question": ex.get("question", "")})
                    gt = get_ground_truth(ex)  # handles '####' and normalization
                    eval_prompts.append(prompt)
                    eval_ground_truths.append(gt)

                results = log_generations(
                    generator=llm,
                    tokenizer=tokenizer,
                    prompts=eval_prompts,
                    ground_truths=eval_ground_truths,
                    reward_fn=r1_zero_reward_fn,
                    batch_size=eval_batch_size,
                    max_new_tokens=eval_max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    fast=True,
                    scoring_model=policy,
                )

                # Print a short summary
                if results and isinstance(results[-1], dict) and "aggregates" in results[-1]:
                    agg = results[-1]["aggregates"]
                    print(
                        f"[step {global_step}] "
                        f"loss={meta['loss'].item():.4f} "
                        f"avg_resp_len={agg['avg_response_length']:.2f} "
                        f"avg_token_entropy_resp={agg['avg_token_entropy_response']:.3f}"
                    )

        # End epoch
        print(f"Finished epoch {epoch+1}/{epochs}")

    # Final sync and return
    load_policy_into_vllm_instance(policy, llm)
    return policy, tokenizer, llm


def main():
    parser = argparse.ArgumentParser(description="SFT training script with vLLM evaluation")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-path", type=str, default="data/gsm8k/sft.jsonl")
    parser.add_argument("--val-path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--eval-prompt", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--eval-n-examples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", type=str, nargs="*", default=["</answer>"])
    args = parser.parse_args()

    train_sft(
        model_id=args.model_id,
        train_path=args.train_path,
        val_path=args.val_path,
        policy_device=args.policy_device,
        eval_device=args.eval_device,
        seed=args.seed,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        eval_every_steps=args.eval_every_steps,
        eval_n_examples=args.eval_n_examples,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        eval_prompt_path=args.eval_prompt,
    )


if __name__ == "__main__":
    main()
