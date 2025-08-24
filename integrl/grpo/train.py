import os
import csv
import time
import json
import torch
import random
import argparse
from torch.optim import AdamW
from typing import List, Dict, Any, Tuple
from torch.nn.utils import clip_grad_norm_
from vllm import SamplingParams
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizer, 
    PreTrainedModel
)

from ..grader_helpers import r1_zero_reward_fn
from ..sft.evals import format_prompt, get_ground_truth
from ..sft.train_helpers import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
)
from .train_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = list()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


HEADER = f"{'step':>8} | {'loss':>9} | {'ema':>9} | {'trainR':>9} | {'g_norm':>7} | {'tok/s':>10} | {'time':>8}"

def format_row(step: int, loss_item: float, ema: float, train_reward: float, grad_norm: float, tps: float, elapsed: float) -> str:
    return f"{step:8d} | {loss_item:9.4f} | {ema:9.4f} | {train_reward:9.4f} | {float(grad_norm):7.3f} | {tps:10,.0f} | {elapsed:8.2f}s"


def _compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        param_norm = torch.linalg.vector_norm(grad, ord=norm_type)
        total += float(param_norm.item() ** 2)
    return float(total ** 0.5)


def _select_prompts_and_gts(
    data: List[Dict[str, Any]],
    prompt_template: str,
    n_prompts: int,
) -> Tuple[List[str], List[Any], List[str]]:
    """Return prompts (for generation), ground-truths, and question texts for logging."""
    if not data or n_prompts <= 0:
        return [], [], []
    idxs = random.sample(range(len(data)), k=min(n_prompts, len(data)))
    prompts: List[str] = []
    gts: List[Any] = []
    questions: List[str] = []
    for i in idxs:
        ex = data[i]
        if "question" not in ex:
            continue
        q = ex["question"]
        prompts.append(format_prompt(prompt_template, {"question": q}))
        gts.append(get_ground_truth(ex))
        questions.append(q)
    return prompts, gts, questions


def _repeat_by_group(xs: List[Any], group_size: int) -> List[Any]:
    out: List[Any] = []
    for x in xs:
        out.extend([x] * group_size)
    return out


def train_grpo(
    model_id: str,
    train_path: str,
    val_path: str | None = None,
    policy_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    seed: int = 1709,
    # GRPO hyperparams
    grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1, # on-policy default
    train_batch_size: int = 256, # on-policy default (== rollout_batch_size)
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline", # "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    max_grad_norm: float = 1.0,
    eval_every_steps: int = 10,
    eval_n_examples: int = 128,
    eval_batch_size: int = 32,
    eval_max_new_tokens: int = 1024,
    eval_temperature: float = 0.0,
    eval_top_p: float = 1.0,
    eval_stop: List[str] | None = None,
    eval_prompt_path: str = "integrl/prompts/r1_zero.prompt",
    checkpoint_dir: str | None = None,
    checkpoint_every_steps: int = 0):
    torch.manual_seed(seed)

    # Sanity asserts and constants
    assert train_batch_size % gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, \
        "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    # Metrics file
    metrics_dir = checkpoint_dir or os.getcwd()
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    metrics_writer = csv.writer(metrics_f)
    if write_header:
        metrics_writer.writerow(["step","train_loss","ema_loss","train_reward","grad_norm","tok_per_s","elapsed_s"])

    # Load tokenizer and policy model (HF) on policy_device
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2"
    )
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.to(policy_device)
    policy.train()

    optimizer = AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    optimizer.zero_grad(set_to_none=True)

    # vLLM generator on eval_device (can be same or different)
    llm = init_vllm(model_id=model_id, device=eval_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)
    # Ensure the generator starts with the current policy weights
    load_policy_into_vllm_instance(policy, llm)

    # Data
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path else []

    # Load evaluation prompt template
    with open(eval_prompt_path, "r") as f:
        prompt_template = f.read()

    # Logging state
    global_step = 0
    running_loss = None
    printed_header = False
    t0 = time.time()

    while global_step < grpo_steps:
        # Select prompts and repeat by group size
        # For simplicity, sample a fresh slice each step (could also shuffle/iterate)
        base_prompts, base_gts, _ = _select_prompts_and_gts(
            train_rows, prompt_template, n_prompts_per_rollout_batch
        )
        if not base_prompts:
            print("No valid training prompts found.")
            break

        prompts = _repeat_by_group(base_prompts, group_size)
        ground_truths = _repeat_by_group(base_gts, group_size)

        # Generate responses via vLLM
        sp = SamplingParams(
            temperature=sampling_temperature,
            top_p=1.0,
            max_tokens=sampling_max_tokens,
            stop=["</answer>"],
        )
        sp.include_stop_str_in_output = True
        outs = llm.generate(prompts, sp)
        responses: List[str] = []
        for out in outs:
            resp = out.outputs[0].text if out.outputs else ""
            responses.append(resp)

        # Tokenize prompt+response and compute policy log-probs
        toks = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = toks["input_ids"].to(policy_device)
        labels = toks["labels"].to(policy_device)
        response_mask = toks["response_mask"].to(policy_device)

        with torch.no_grad():
            scored = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs_full = scored["log_probs"]  # (B, T)

        # Rewards and advantages
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        advantages = advantages.to(policy_log_probs_full.device)
        raw_rewards = raw_rewards.to(policy_log_probs_full.device)

        # Disallow empty/too-short responses (count response tokens)
        if sampling_min_tokens > 0:
            resp_tokens_per_ex = response_mask.sum(dim=1)
            # Zero-out advantages for empty responses (conservative)
            zero_mask = (resp_tokens_per_ex < sampling_min_tokens)
            if zero_mask.any():
                advantages[zero_mask] = 0.0
                raw_rewards[zero_mask] = 0.0

        # If GRPO-Clip, compute and cache old_log_probs once per rollout batch
        old_log_probs_full = None
        if loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs_full = policy_log_probs_full.detach().clone()

        # Multiple epochs over the rollout batch (off-policy if > 1)
        for _ in range(epochs_per_rollout_batch):
            # Recompute current policy log probs for the batch (policy is changing across epochs)
            scored_now = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = scored_now["log_probs"]  # (B, T)

            # If on-policy GRPO-Clip is weakly setup, set old_log_probs from first epoch
            if loss_type == "grpo_clip" and old_log_probs_full is None:
                with torch.no_grad():
                    old_log_probs_full = policy_log_probs.detach().clone()

            # Microbatch over rollout batch
            for mb_start in range(0, rollout_batch_size, micro_train_batch_size):
                mb_end = min(mb_start + micro_train_batch_size, rollout_batch_size)

                mb_policy_log_probs = policy_log_probs[mb_start:mb_end]
                mb_response_mask = response_mask[mb_start:mb_end]
                mb_advantages = advantages[mb_start:mb_end].unsqueeze(-1)  # (b,1)
                mb_raw_rewards = raw_rewards[mb_start:mb_end].unsqueeze(-1) # (b,1)
                mb_old_log_probs = (
                    old_log_probs_full[mb_start:mb_end] if old_log_probs_full is not None else None
                )

                # One microbatch step (includes loss.backward())
                scaled_loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,  # type: ignore[arg-type]
                    raw_rewards=(mb_raw_rewards if loss_type == "no_baseline" else None),
                    advantages=(mb_advantages if loss_type != "no_baseline" else None),
                    old_log_probs=mb_old_log_probs,
                    cliprange=(cliprange if loss_type == "grpo_clip" else None),
                )

                global_step += 1

                # Optimizer step after grad accumulation
                if global_step % gradient_accumulation_steps == 0:
                    if max_grad_norm and max_grad_norm > 0:
                        clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()
                    policy.zero_grad(set_to_none=True)

                # Logging
                loss_item = float(meta["loss/scalar_unscaled"].item())
                running_loss = loss_item if running_loss is None else (0.95 * running_loss + 0.05 * loss_item)
                grad_norm_now = _compute_grad_norm(policy.parameters())
                elapsed = time.time() - t0
                toks_this_step = int(input_ids.numel())
                tps = toks_this_step / max(1e-6, elapsed)

                # Train reward (mean)
                train_reward = float(raw_rewards.mean().item())

                if not printed_header or (global_step % 200 == 0):
                    print(HEADER)
                    printed_header = True
                print(format_row(global_step, loss_item, (running_loss or 0.0), train_reward, grad_norm_now, tps, elapsed))
                metrics_writer.writerow([global_step, loss_item, (running_loss or 0.0), train_reward, float(grad_norm_now), tps, elapsed])
                metrics_f.flush()
                t0 = time.time()

                # Periodic evaluation (optional)
                if val_rows and ((global_step % eval_every_steps == 0) or (global_step == 0)):
                    # Sync policy weights into vLLM
                    load_policy_into_vllm_instance(policy, llm)

                    n_eval = min(eval_n_examples, len(val_rows))
                    eval_prompts: List[str] = []
                    eval_ground_truths: List[Any] = []
                    for i in range(n_eval):
                        ex = val_rows[i]
                        if "question" not in ex:
                            continue
                        prompt = format_prompt(prompt_template, {"question": ex.get("question", "")})
                        gt = get_ground_truth(ex)
                        eval_prompts.append(prompt)
                        eval_ground_truths.append(gt)

                    sp_eval = SamplingParams(
                        temperature=eval_temperature,
                        top_p=eval_top_p,
                        max_tokens=eval_max_new_tokens,
                        stop=(eval_stop or ["</answer>"]),
                    )
                    sp_eval.include_stop_str_in_output = True
                    eval_responses: List[str] = []
                    for i in range(0, len(eval_prompts), eval_batch_size):
                        batch_prompts = eval_prompts[i:i + eval_batch_size]
                        outs_eval = llm.generate(batch_prompts, sp_eval)
                        for out in outs_eval:
                            txt = out.outputs[0].text if out.outputs else ""
                            eval_responses.append(txt)

                    # Eval rewards (accuracy proxy)
                    eval_rewards: List[float] = []
                    fmt_r: List[float] = []
                    ans_r: List[float] = []
                    for resp, gt in zip(eval_responses, eval_ground_truths):
                        scores = r1_zero_reward_fn(resp, gt, fast=True)
                        eval_rewards.append(float(scores.get("reward", scores.get("total", 0.0))))
                        fmt_r.append(float(scores.get("format_reward", 0.0)))
                        ans_r.append(float(scores.get("answer_reward", 0.0)))

                    avg_eval = float(sum(eval_rewards) / max(1, len(eval_rewards)))
                    print(f"[eval step {global_step}] val_reward={avg_eval:.4f} fmt={sum(fmt_r)/max(1,len(fmt_r)):.3f} ans={sum(ans_r)/max(1,len(ans_r)):.3f}")

                # Periodic checkpointing
                if checkpoint_every_steps > 0 and checkpoint_dir and (global_step % checkpoint_every_steps == 0):
                    ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    print(f"Saving GRPO checkpoint to: {ckpt_path}")
                    policy.save_pretrained(save_directory=ckpt_path)
                    tokenizer.save_pretrained(save_directory=ckpt_path)

            # End microbatches over rollout batch
        
        # End epochs over rollout batch

        # After finishing updates for this rollout batch, optionally refresh vLLM weights
        load_policy_into_vllm_instance(policy, llm)

    metrics_f.close()
    return policy, tokenizer, llm


def main():
    parser = argparse.ArgumentParser(description="GRPO training loop with vLLM rollouts")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-path", type=str, default="integrl/data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", type=str, default="integrl/data/gsm8k/test.jsonl")
    parser.add_argument("--eval-prompt", type=str, default="integrl/prompts/r1_zero.prompt")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=1709)

    # GRPO hyperparams
    parser.add_argument("--grpo-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--grad-accum-steps", type=int, default=128)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--loss-type", type=str, default="reinforce_with_baseline", choices=[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip"
    ])
    parser.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Eval / logging
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--eval-n-examples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-stop", type=str, nargs="*", default=["</answer>"])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)

    args = parser.parse_args()

    train_grpo(
        model_id=args.model_id,
        train_path=args.train_path,
        val_path=args.val_path,
        policy_device=args.policy_device,
        eval_device=args.eval_device,
        seed=args.seed,
        grpo_steps=args.grpo_steps,
        learning_rate=args.lr,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gpu_memory_utilization=args.gpu_mem_util,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        cliprange=args.cliprange,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        eval_n_examples=args.eval_n_examples,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        eval_stop=args.eval_stop,
        eval_prompt_path=args.eval_prompt,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )


if __name__ == "__main__":
    main()
