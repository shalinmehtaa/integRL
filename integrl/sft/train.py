import os
import csv
import time
import json
import torch
import random
import argparse
from torch.optim import AdamW
from typing import List, Dict, Any
from torch.nn.utils import clip_grad_norm_
from ..grader_helpers import r1_zero_reward_fn
from ..sft.evals import format_prompt, get_ground_truth
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
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

def _save_model_and_tokenizer(policy: PreTrainedModel, tokenizer: PreTrainedTokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving model and tokenizer to: {out_dir}")
    policy.save_pretrained(save_directory=out_dir)
    tokenizer.save_pretrained(save_directory=out_dir)

# Logging helpers
HEADER = f"{'step':>8} | {'loss':>9} | {'ema':>9} | {'val':>9} | {'g_norm':>7} | {'tok/s':>10} | {'time':>8}"

def format_row(step: int, loss_item: float, ema: float, val_metric: float, grad_norm: float, tps: float, elapsed: float) -> str:
    return f"{step:8d} | {loss_item:9.4f} | {ema:9.4f} | {val_metric:9.4f} | {float(grad_norm):7.3f} | {tps:10,.0f} | {elapsed:8.2f}s"

def _compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        param_norm = torch.linalg.vector_norm(grad, ord=norm_type)
        total += float(param_norm.item() ** 2)
    return float(total ** 0.5)


def train_sft(
    model_id: str,
    train_path: str,
    val_path: str,
    policy_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    seed: int = 1709,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
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
    checkpoint_dir: str | None = None,
    checkpoint_every_steps: int = 0):
    torch.manual_seed(seed)

    # Metrics file (under checkpoint_dir if provided)
    metrics_dir = checkpoint_dir or os.getcwd()
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    metrics_writer = csv.writer(metrics_f)
    if write_header:
        metrics_writer.writerow(["step","train_loss","ema_loss","val_metric","grad_norm","tok_per_s","elapsed_s"])

    # Load tokenizer and policy model (HF) on policy_device
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2"
    )
    # policy.resize_token_embeddings(len(tokenizer))
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.to(policy_device)
    policy.train()

    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    # Initialize grads to zero
    optimizer.zero_grad(set_to_none=True)

    # vLLM evaluator on eval_device
    llm = init_vllm(model_id=model_id, device=eval_device, seed=seed)

    # Data
    train_rows = read_sft_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path else []

    # Load evaluation prompt template
    with open(eval_prompt_path, "r") as f:
        eval_prompt_template = f.read()

    # Logging state
    global_step = 0
    running_loss = None
    printed_header = False
    t0 = time.time()
    last_val_metric = float("nan")

    for epoch in range(epochs):
        for batch_idxs in make_batches(len(train_rows), batch_size, shuffle=True, seed=seed + epoch):

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

            # Per-step logging (train loss, EMA, grad norm, tok/s, elapsed)
            loss_item = float(meta["token_nll_mean"].item())
            running_loss = loss_item if running_loss is None else (0.95 * running_loss + 0.05 * loss_item)
            grad_norm_now = _compute_grad_norm(policy.parameters())
            elapsed = time.time() - t0
            toks_this_step = int(input_ids.numel())
            tps = toks_this_step / max(1e-6, elapsed)

            if not printed_header or (global_step % 200 == 0):
                print(HEADER)
                printed_header = True
            print(format_row(global_step, loss_item, running_loss, (last_val_metric if not torch.isnan(torch.tensor(last_val_metric)) else float("nan")), grad_norm_now, tps, elapsed))
            metrics_writer.writerow([global_step, loss_item, running_loss, (last_val_metric if not torch.isnan(torch.tensor(last_val_metric)) else float("nan")), float(grad_norm_now), tps, elapsed])
            metrics_f.flush()
            t0 = time.time()

            # Optimizer step after grad accumulation window
            if global_step % grad_accum_steps == 0:
                if max_grad_norm and max_grad_norm > 0:
                    clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                policy.zero_grad(set_to_none=True)

            # Periodic evaluation with vLLM on eval_device
            if val_rows and ((global_step % eval_every_steps == 0) or (global_step == 0)):
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

                # Compute validation metric (accuracy) and print short summary
                n_results = sum(1 for r in results if isinstance(r, dict) and "correct" in r)
                if n_results > 0:
                    correct = sum(1 for r in results if isinstance(r, dict) and r.get("correct") is True)
                    last_val_metric = correct / max(1, n_results)

                if results and isinstance(results[-1], dict) and "aggregates" in results[-1]:
                    agg = results[-1]["aggregates"]
                    print(
                        f"[step {global_step}] "
                        f"loss={meta['loss'].item():.4f} "
                        f"val_acc={last_val_metric:.4f} "
                        f"avg_resp_len={agg['avg_response_length']:.2f} "
                        f"avg_resp_token_entropy={agg['avg_token_entropy_response']:.3f}"
                    )

            # Periodic checkpointing
            if checkpoint_every_steps > 0 and checkpoint_dir and (global_step % checkpoint_every_steps == 0):
                ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step}")
                _save_model_and_tokenizer(policy, tokenizer, ckpt_path)

        print(f"Finished epoch {epoch+1}/{epochs}")

    # Final sync
    load_policy_into_vllm_instance(policy, llm)
    # Final save
    _save_model_and_tokenizer(policy, tokenizer, ckpt_path)
    metrics_f.close()
    return policy, tokenizer, llm


def main():
    parser = argparse.ArgumentParser(description="SFT training script with vLLM evaluation")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-path", type=str, default="integrl/data/gsm8k/sft.jsonl")
    parser.add_argument("--val-path", type=str, default="integrl/data/gsm8k/test.jsonl")
    parser.add_argument("--eval-prompt", type=str, default="integrl/prompts/r1_zero.prompt")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    
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
        max_grad_norm=args.max_grad_norm,
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
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_steps=args.checkpoint_every_steps)


if __name__ == "__main__":
    main()
