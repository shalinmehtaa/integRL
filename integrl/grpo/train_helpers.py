import math
import torch
from typing import Callable, Dict, List, Tuple, Any, Literal


def _safe_call_reward_fn(
    reward_fn: Callable[..., Dict[str, float]],
    response: str,
    ground_truth: Any,
) -> Dict[str, float]:
    try:
        return reward_fn(response, ground_truth, fast=True)
    except TypeError:
        return reward_fn(response, ground_truth)


def _extract_total_reward(d: Dict[str, float]) -> float:
    """Prefer 'reward', else 'total', else sum of values."""
    if "reward" in d and isinstance(d["reward"], (int, float)):
        return float(d["reward"])
    if "total" in d and isinstance(d["total"], (int, float)):
        return float(d["total"])
    return float(sum(v for v in d.values() if isinstance(v, (int, float))))


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, Any], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[Any],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size."""
    assert len(rollout_responses) == len(repeated_ground_truths), \
        "rollout_responses and repeated_ground_truths must be the same length"
    assert group_size >= 1, "group_size must be >= 1"
    batch_size = len(rollout_responses)
    assert (batch_size % group_size == 0), \
        "Batch size must be divisible by group_size for grouping"

    # Score all responses
    total_rewards: List[float] = []
    fmt_rewards: List[float] = []
    ans_rewards: List[float] = []

    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = _safe_call_reward_fn(reward_fn, resp, gt)
        total_rewards.append(_extract_total_reward(scores))
        fmt_rewards.append(float(scores.get("format_reward", 0.0)))
        ans_rewards.append(float(scores.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(total_rewards, dtype=torch.float32)
    fmt_rewards_t = torch.tensor(fmt_rewards, dtype=torch.float32)
    ans_rewards_t = torch.tensor(ans_rewards, dtype=torch.float32)

    # Group into (num_groups, group_size)
    num_groups = batch_size // group_size
    rewards_g = raw_rewards.view(num_groups, group_size)

    # Per-group mean and std (population std)
    group_means = rewards_g.mean(dim=1, keepdim=True) # (num_groups, 1)
    group_stds = rewards_g.std(dim=1, unbiased=False, keepdim=True) # (num_groups, 1)

    # Normalize within groups
    centered = rewards_g - group_means # (num_groups, group_size)
    if normalize_by_std:
        denom = torch.clamp(group_stds, min=advantage_eps)
        advantages_g = centered / denom
    else:
        advantages_g = centered

    advantages = advantages_g.reshape(batch_size)

    # Metadata
    meta: Dict[str, float] = {
        "reward/mean": float(raw_rewards.mean().item()) if batch_size > 0 else math.nan,
        "reward/std": float(raw_rewards.std(unbiased=False).item()) if batch_size > 0 else math.nan,
        "reward/min": float(raw_rewards.min().item()) if batch_size > 0 else math.nan,
        "reward/max": float(raw_rewards.max().item()) if batch_size > 0 else math.nan,
        "format_reward/mean": float(fmt_rewards_t.mean().item()) if batch_size > 0 else math.nan,
        "answer_reward/mean": float(ans_rewards_t.mean().item()) if batch_size > 0 else math.nan,
        "groups/count": float(num_groups),
        "groups/mean_of_means": float(group_means.mean().item()) if num_groups > 0 else math.nan,
        "groups/mean_of_stds": float(group_stds.mean().item()) if num_groups > 0 else math.nan,
        "groups/std_min": float(group_stds.min().item()) if num_groups > 0 else math.nan,
        "groups/std_max": float(group_stds.max().item()) if num_groups > 0 else math.nan,
        "groups/std_near_zero_frac": float(
            (group_stds < advantage_eps).float().mean().item()
        )
        if num_groups > 0
        else math.nan,
        "advantages/mean": float(advantages.mean().item()) if batch_size > 0 else math.nan,
        "advantages/std": float(advantages.std(unbiased=False).item()) if batch_size > 0 else math.nan,
        "config/group_size": float(group_size),
        "config/advantage_eps": float(advantage_eps),
        "config/normalize_by_std": float(1.0 if normalize_by_std else 0.0),
    }

    return advantages, raw_rewards, meta


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token policy-gradient loss"""
    assert policy_log_probs.dim() == 2, "policy_log_probs must be (B, T)"
    B, T = policy_log_probs.shape

    A = raw_rewards_or_advantages
    if A.dim() == 1:
        A = A.unsqueeze(-1)  # (B, 1)
    assert A.shape[0] == B and A.shape[1] == 1, "advantages must be shape (B, 1) or (B,)"

    # Broadcast scalar per-sequence advantages over tokens
    A = A.to(policy_log_probs.dtype).expand_as(policy_log_probs)

    # Detach A to avoid backprop through reward computation
    loss = -(A.detach() * policy_log_probs)
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token GRPO-Clip loss."""
    assert policy_log_probs.dim() == 2 and old_log_probs.dim() == 2, "log_probs must be (B, T)"
    B, T = policy_log_probs.shape
    assert old_log_probs.shape == (B, T), "old_log_probs must match policy_log_probs shape"

    A = advantages
    if A.dim() == 1:
        A = A.unsqueeze(-1)  # (B, 1)
    assert A.shape == (B, 1), "advantages must have shape (B, 1) or (B,)"

    # Broadcast per-sequence A across tokens; keep gradients only through log-probs.
    A = A.to(policy_log_probs.dtype).expand_as(policy_log_probs)
    A_detached = A.detach()

    # Importance ratio per token
    ratio = torch.exp(policy_log_probs - old_log_probs) # (B, T)

    # Unclipped and clipped objectives: maximize these; loss will be negative
    unclipped_obj = ratio * A_detached
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_obj = clipped_ratio * A_detached

    obj = torch.minimum(unclipped_obj, clipped_obj)
    loss = -obj # (B, T)

    # Metadata for monitoring
    with torch.no_grad():
        outside_clip = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
        took_clipped = (clipped_obj < unclipped_obj)

        metadata = {
            "ppo/ratio_mean": ratio.mean(),
            "ppo/ratio_std": ratio.std(unbiased=False),
            "ppo/frac_outside_clip": outside_clip.float().mean(),
            "ppo/frac_took_clipped": took_clipped.float().mean(),
            "ppo/cliprange": torch.tensor(float(cliprange)),
            "advantages/mean": A_detached.mean(),
            "advantages/std": A_detached.std(unbiased=False),
        }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Convenience wrapper to compute per-token policy-gradient loss for:
      - "no_baseline": use raw rewards as advantages
      - "reinforce_with_baseline": use provided advantages (group-normalized)
      - "grpo_clip": GRPO clipped objective against frozen old policy
    """
    assert policy_log_probs.dim() == 2, "policy_log_probs must be (B, T)"
    B, T = policy_log_probs.shape

    def _as_col(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        A = _as_col(raw_rewards)
        assert A.shape[0] == B and A.shape[1] == 1, "raw_rewards must be (B,) or (B,1)"
        loss = compute_naive_policy_gradient_loss(A, policy_log_probs)
        with torch.no_grad():
            A_d = A.detach()
            meta = {
                "pg/variant_no_baseline": torch.tensor(1.0),
                "advantages/mean": A_d.mean().to(policy_log_probs.dtype),
                "advantages/std": A_d.std(unbiased=False).to(policy_log_probs.dtype),
            }
        return loss, meta

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        A = _as_col(advantages)
        assert A.shape[0] == B and A.shape[1] == 1, "advantages must be (B,) or (B,1)"
        loss = compute_naive_policy_gradient_loss(A, policy_log_probs)
        with torch.no_grad():
            A_d = A.detach()
            meta = {
                "pg/variant_reinforce_with_baseline": torch.tensor(1.0),
                "advantages/mean": A_d.mean().to(policy_log_probs.dtype),
                "advantages/std": A_d.std(unbiased=False).to(policy_log_probs.dtype),
            }
        return loss, meta

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        assert old_log_probs.shape == (B, T), "old_log_probs must be (B, T)"
        loss, meta = compute_grpo_clip_loss(_as_col(advantages), policy_log_probs, old_log_probs, float(cliprange))
        # Tag variant
        meta = {**meta, "pg/variant_grpo_clip": torch.tensor(1.0)}
        return loss, meta

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    m = mask.to(dtype=tensor.dtype)
    masked = tensor * m
    if dim is None:
        total = masked.sum()
        count = m.sum().clamp_min(1)
        return total / count
    else:
        total = masked.sum(dim=dim)
        count = m.sum(dim=dim).clamp_min(1)
        return total / count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute one GRPO micro-batch step:
      1) Compute per-token PG loss for the selected variant
      2) Masked mean over response tokens => per-example loss
      3) Mean over batch => scalar loss
      4) Scale by gradient_accumulation_steps and backprop
    """
    assert policy_log_probs.dim() == 2, "policy_log_probs must be (B, T)"
    assert response_mask.shape == policy_log_probs.shape, "response_mask must match (B, T)"
    B, T = policy_log_probs.shape

    # Per-token policy gradient loss (B, T)
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Masked mean across the sequence > per-sequence scalar (B,)
    per_ex_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # Mean over batch > scalar
    micro_loss = per_ex_loss.mean()

    # Scale for gradient accumulation and backprop
    scaled_loss = micro_loss / float(gradient_accumulation_steps)
    scaled_loss.backward()

    # Useful metadata
    with torch.no_grad():
        resp_tokens_per_ex = response_mask.to(dtype=torch.float32).sum(dim=1)
        metadata = {
            **meta,
            "loss/scalar_scaled": scaled_loss.detach(),
            "loss/scalar_unscaled": micro_loss.detach(),
            "loss/per_example_mean": per_ex_loss.detach().mean(),
            "tokens/response_total": response_mask.sum().to(dtype=torch.float32),
            "tokens/response_per_example_mean": resp_tokens_per_ex.mean(),
            "grad_accumulation_steps": torch.tensor(float(gradient_accumulation_steps)),
        }

    return scaled_loss, metadata