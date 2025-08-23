import time
import json
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from typing import Dict, Callable, List, Any, Optional

from ..grader_helpers import (
    r1_zero_reward_fn,
    question_only_reward_fn,
    normalize_final_answer,
)


def format_prompt(prompt: str, inputs: Dict[str, str]):
    prompt = prompt.format(question=inputs["question"])
    return prompt


def _extract_ground_truth(raw: Any) -> Any:
    """
    Convert raw dataset 'answer' into a normalized ground-truth string (or list of strings).
    Handles GSM8K-style '####' markers.
    """
    if isinstance(raw, (int, float)):
        raw = str(raw)
    if isinstance(raw, list):
        out = []
        for r in raw:
            s = str(r)
            if "####" in s:
                s = s.split("####")[-1].strip()
            s = normalize_final_answer(s)
            out.append(s)
        return out
    s = str(raw)
    if "####" in s:
        s = s.split("####")[-1].strip()
    s = normalize_final_answer(s)
    return s


def get_ground_truth(ex: Dict[str, Any]) -> Any:
    """    May return str or list[str] if multiple answers are accepted."""
    for key in ("answer", "ground_truth", "final_answer", "target", "label"):
        if key in ex and ex[key] is not None:
            return _extract_ground_truth(ex[key])
    return None


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any, bool], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[Any],
    eval_sampling_params: SamplingParams,
    fast: bool = True,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    assert len(prompts) == len(ground_truths), "prompts and ground_truths must align"
    n = len(prompts)
    results: List[Dict[str, Any]] = []
    for start in tqdm(range(0, n, batch_size), desc="Evaluating", leave=False):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]
        outputs = vllm_model.generate(batch_prompts, eval_sampling_params)
        for i, out in enumerate(outputs):
            idx = start + i
            response = out.outputs[0].text if out.outputs else ""
            gt = ground_truths[idx]
            scores = reward_fn(response, gt, fast=fast)
            results.append(
                {
                    "idx": idx,
                    "prompt": out.prompt,
                    "response": response,
                    "ground_truth": gt,
                    "rewards": scores,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation with DRGRPO grader")
    parser.add_argument("--dataset", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--model", type=str, required=True, help="HF or local model name for vLLM")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate (<= dataset size)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stop", type=str, nargs="*", default=["</answer>"])
    parser.add_argument("--reward-fn", type=str, default="r1_zero", choices=["r1_zero", "question_only"])
    parser.add_argument("--fast", action="store_true", help="Use fast grading (skips math_verify step)")
    parser.add_argument("--no-fast", dest="fast", action="store_false")
    parser.set_defaults(fast=True)
    parser.add_argument("--output", type=str, default=None, help="Path to JSONL file to save results")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt)

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    data: List[Dict[str, Any]] = []
    with open(dataset_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.limit is not None and args.limit > 0:
        data = data[: args.limit]

    # Prepare prompts and ground truths
    formatted_prompts: List[str] = []
    ground_truths: List[Any] = []
    questions: List[str] = []
    for ex in data:
        if "question" not in ex:
            # skip if no question
            continue
        questions.append(ex["question"])
        formatted_prompts.append(format_prompt(prompt_template, ex))
        ground_truths.append(get_ground_truth(ex))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
    )
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model=args.model, dtype=args.dtype)

    reward_fn = r1_zero_reward_fn if args.reward_fn == "r1_zero" else question_only_reward_fn

    results = evaluate_vllm(
        llm,
        reward_fn,
        formatted_prompts,
        ground_truths,
        sampling_params,
        fast=args.fast,
        batch_size=args.batch_size,
    )

    # Compute metrics
    total = len(results)
    if total == 0:
        print("No results produced.")
        return

    fmt_reward = sum(r["rewards"].get("format_reward", 0.0) for r in results) / total
    ans_reward  = sum(r["rewards"].get("answer_reward", 0.0) for r in results) / total
    tot_reward  = sum(r["rewards"].get("answer_reward", 0.0) for r in results) / total

    print(f"Total: {total} | Format reward: {fmt_reward:.3f} | Answer reward: {ans_reward:.3f} | Overall reward: {tot_reward:.3f}")

    # Attach question to each record (by idx alignment)
    for r in results:
        idx = r["idx"]
        # idx may not align if skipped malformed rows
        if 0 <= idx < len(questions):
            r["question"] = questions[idx]

    # Save results
    out_path = (
        Path(args.output)
        if args.output
        else Path("runs")
        / "evals"
        / f"{args.model.replace('/','__')}_{args.reward_fn}_{int(time.time())}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print a few examples
    print("Sample outputs:")
    for r in results[: min(3, len(results))]:
        print(f"Question: {r.get('question','')}")
        print(f"Ground truth: {r.get('ground_truth')}")
        print(f"Response: {r.get('response','')}")
        print(f"Rewards: {r.get('rewards')}")


if __name__ == "__main__":
    main()
