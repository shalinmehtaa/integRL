## integRL

Integrating math reasoning into LLMs with RL.

### Requirements
- **Python**: 3.11–3.12
- **GPU**: CUDA-capable GPU (used by vLLM and flash-attn)


### Data
GSM8K samples:
- `integrl/data/gsm8k/sft.jsonl` (SFT pairs)
- `integrl/data/gsm8k/train.jsonl`, `integrl/data/gsm8k/test.jsonl`

### Train (SFT)
```bash
python -m integrl.sft.train \
  --model-id Qwen/Qwen2.5-Math-1.5B \
  --train-path integrl/data/gsm8k/sft.jsonl \
  --val-path integrl/data/gsm8k/test.jsonl \
  --eval-prompt integrl/prompts/r1_zero.prompt \
  --epochs 1 \
  --checkpoint-dir integrl/checkpoints/my_run
```
