import torch
import contextlib
from torch import Tensor
from unittest.mock import patch
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional, Any, Callable
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, Tensor]:
    """Tokenize prompt and output strings separately, concatenate per example."""
    assert len(prompt_strs) == len(output_strs), "prompt_strs and output_strs must be same length"
    batch_size = len(prompt_strs)

    # Batch tokenize without special tokens
    prompt_token_lists: List[List[int]] = tokenizer(
        prompt_strs, add_special_tokens=False
    ).input_ids
    output_token_lists: List[List[int]] = tokenizer(
        output_strs, add_special_tokens=False
    ).input_ids

    # Concatenate per-example and record lengths
    joined_token_lists: List[List[int]] = []
    prompt_lens: List[int] = []
    seq_lens: List[int] = []
    for p_ids, o_ids in zip(prompt_token_lists, output_token_lists):
        prompt_len = len(p_ids)
        joined = p_ids + o_ids
        joined_token_lists.append(joined)
        prompt_lens.append(prompt_len)
        seq_lens.append(len(joined))

    max_len = max(seq_lens) if seq_lens else 0

    # Determine pad token id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Build padded tensor of all ids: (B, max_len)
    all_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(joined_token_lists):
        if len(ids) > 0:
            all_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Build full response mask over all positions (B, max_len)
    full_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, (p_len, total_len) in enumerate(zip(prompt_lens, seq_lens)):
        if total_len > p_len:
            full_mask[i, p_len:total_len] = 1 # mark response tokens

    # Slice for input/labels and mask alignment: input_ids omits the final token; labels omit the first token
    input_ids = all_ids[:, :-1] if max_len > 0 else all_ids
    labels = all_ids[:, 1:] if max_len > 0 else all_ids
    response_mask = full_mask[:, 1:] if max_len > 0 else full_mask

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropy over the vocabulary for next-token predictions."""
    # Numerically stable: use log_softmax to get log-probs
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
    probs = torch.exp(log_probs)                   # (B, T, V)
    entropy = -(probs * log_probs).sum(dim=-1)     # (B, T)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: Optional[bool] = False) -> Dict[str, Tensor]:
    """Compute log probs from pretrained model logits and optionally compute entropy."""
    
    # Forward pass
    logits = model(input_ids, use_cache=False).logits  # (B, T, V)

    # Memory-efficient label log-probs: log p(y) = logits_y - logsumexp(logits)
    label_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    logsumexp = torch.logsumexp(logits, dim=-1)  # (B, T)
    log_probs_labels = label_logits - logsumexp  # (B, T)

    out: Dict[str, Tensor] = {"log_probs": log_probs_labels}

    if return_token_entropy:
        token_entropies = compute_entropy(logits) # Beware: may be memory-heavy for large batches
        out["token_entropy"] = token_entropies

    return out


def get_response_log_probs_microbatched(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    micro_batch_size: int,
    return_token_entropy: Optional[bool] = False,
    no_grad: bool = False
) -> Dict[str, Tensor]:
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    chunks = []
    ent_chunks = []
    with ctx:
        for start in range(0, input_ids.size(0), micro_batch_size):
            end = min(start + micro_batch_size, input_ids.size(0))
            scored = get_response_log_probs(
                model, input_ids[start:end], labels[start:end], return_token_entropy=return_token_entropy
            )
            chunks.append(scored["log_probs"])
            if return_token_entropy and "token_entropy" in scored:
                ent_chunks.append(scored["token_entropy"])
    out: Dict[str, Tensor] = {"log_probs": torch.cat(chunks, dim=0)}
    if return_token_entropy and ent_chunks:
        out["token_entropy"] = torch.cat(ent_chunks, dim=0)
    return out


def masked_normalize(
    log_probs: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: Optional[int] = None) -> Tensor:
    """ Normalize over dim (or all dims if None) of log_probs considering only elements where mask == 1."""
    m = mask.to(dtype=log_probs.dtype)
    masked = log_probs * m
    if dim is None:
        total = masked.sum()
    else:
        total = masked.sum(dim=dim)
    return total / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0) -> tuple[Tensor, Dict[str, Tensor]]:
    """Single micro-batch training step."""
    # Per-token NLL
    nll = -policy_log_probs # (B, T)

    # Masked sum + normalization via helper
    total_nll = masked_normalize(nll, response_mask, normalize_constant)

    # Scale for gradient accumulation
    loss = total_nll / float(gradient_accumulation_steps)

    loss.backward()

    num_resp_tokens = response_mask.sum().clamp_min(1)
    avg_nll_per_resp_token = (nll.detach() * response_mask.to(nll.dtype)).sum() / num_resp_tokens

    metadata: Dict[str, Tensor] = {
        "loss": loss.detach(),
        "token_nll_sum": (nll.detach() * response_mask.to(nll.dtype)).sum(),
        "token_nll_mean": avg_nll_per_resp_token,
        "response_tokens": num_resp_tokens.to(dtype=torch.float32),
        "grad_accumulation_steps": torch.tensor(float(gradient_accumulation_steps)),
    }
    return loss, metadata


def init_vllm(
    model_id: str, 
    device: str, 
    seed: int, 
    gpu_memory_utilization: float = 0.85):
    """Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy."""
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization)


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from: https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def log_generations(
    generator: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truths: List[Any],
    reward_fn: Callable[[str, Any, bool], Dict[str, float]],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    fast: bool = True,
    scoring_model: Optional[PreTrainedModel] = None,
) -> List[Dict[str, Any]]:
    """
    Generate responses for prompts using vLLM, score rewards, and compute response entropy/length stats.

    Returns per-example records with:
      prompt, response, ground_truth, rewards, avg_response_entropy, response_length, correct(bool).
    """
    assert len(prompts) == len(ground_truths), "prompts and ground_truths must align"
    n = len(prompts)
    results: List[Dict[str, Any]] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]
        batch_gts = ground_truths[start:end]

        # Generate responses via vLLM
        batch_responses: List[str] = []
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop or [],
        )
        sp.include_stop_str_in_output = True
        outs = generator.generate(batch_prompts, sp)
        for out in outs:
            text = out.outputs[0].text if out.outputs else ""
            batch_responses.append(text)

        # Rewards and correctness
        batch_rewards: List[Dict[str, float]] = []
        batch_correct: List[bool] = []
        for resp, gt in zip(batch_responses, batch_gts):
            r = reward_fn(resp, gt, fast=fast)
            batch_rewards.append(r)
            if "answer_reward" in r:
                correct = r["answer_reward"] > 0
            elif "total" in r:
                correct = r["total"] > 0
            else:
                correct = sum(r.values()) > 0
            batch_correct.append(bool(correct))

        # Response lengths (in tokens)
        resp_token_lists = tokenizer(batch_responses, add_special_tokens=False).input_ids
        resp_lens = [len(x) for x in resp_token_lists]

        # Average response token entropy (optional, needs scoring_model)
        avg_resp_entropy_per_ex: List[float] = [float("nan")] * len(batch_responses)
        if scoring_model is not None:
            scoring_model.eval()
            with torch.no_grad():
                toks = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
                input_ids = toks["input_ids"].to(next(scoring_model.parameters()).device)
                labels = toks["labels"].to(input_ids.device)
                resp_mask = toks["response_mask"].to(input_ids.device)

                scored = get_response_log_probs(
                    scoring_model, input_ids, labels, return_token_entropy=True
                )
                ent = scored["token_entropy"]  # (B, T)
                m = resp_mask.to(ent.dtype)
                per_sum = (ent * m).sum(dim=1)
                per_cnt = m.sum(dim=1).clamp_min(1)
                per_avg = per_sum / per_cnt
                avg_resp_entropy_per_ex = per_avg.detach().cpu().tolist()

        # Collate per-example records
        for i in range(len(batch_prompts)):
            results.append(
                {
                    "idx": start + i,
                    "prompt": batch_prompts[i],
                    "response": batch_responses[i],
                    "ground_truth": batch_gts[i],
                    "rewards": batch_rewards[i],
                    "avg_response_entropy": avg_resp_entropy_per_ex[i],
                    "response_length": resp_lens[i],
                    "correct": batch_correct[i],
                }
            )

    # Aggregate stats
    if results:
        ent_vals = [r["avg_response_entropy"] for r in results if isinstance(r["avg_response_entropy"], (int, float))]
        len_vals = [r["response_length"] for r in results]
        corr_len_vals = [r["response_length"] for r in results if r["correct"]]
        incorr_len_vals = [r["response_length"] for r in results if not r["correct"]]

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        aggregates = {
            "avg_token_entropy_response": _mean([x for x in ent_vals if not (isinstance(x, float) and torch.isnan(torch.tensor(x)))]),
            "avg_response_length": _mean(len_vals),
            "avg_response_length_correct": _mean(corr_len_vals),
            "avg_response_length_incorrect": _mean(incorr_len_vals),
        }
        results.append({"aggregates": aggregates})

    return results