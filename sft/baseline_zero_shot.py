import json
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from typing import Dict, Callable, List


def format_prompt(prompt: str, inputs: Dict[str, str]):
    prompt = prompt.format(question=inputs["question"])
    return prompt


def evaluate_vllm(vllm_model: LLM,
                  reward_fn: Callable[[str, str], Dict[str, float]],
                  prompts: List[str],
                  eval_sampling_params: SamplingParams):
    pass


if __name__ == "__main__":
    validation_examples_path = "data/gsm8k/test.jsonl"
    prompt_path = "prompts/r1_zero.prompt"

    with open(prompt_path, "r") as f:
        prompt = f.read()

    data = list()
    with open(validation_examples_path, "r") as f:
        for line in tqdm(f):
            try:
                json_line = json.loads(line)
                data.append(json_line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")

    formatted_prompts = [format_prompt(prompt, data_sample) for data_sample in data[:10]]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"])
    
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype="float16")

    # Generate texts from the prompts.
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
