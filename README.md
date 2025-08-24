## integRL

Integrating math reasoning into LLMs with RL.

#### Data
``GSM8K Benchmark``:
- Train: `integrl/data/gsm8k/train.jsonl`
- Test: `integrl/data/gsm8k/test.jsonl`
- DeepSeek R1 reasoning traces for SFT: `integrl/data/gsm8k/sft.jsonl`

#### Train (Supervised Distillation)
```bash
python -m integrl.sft.train \
  --model-id Qwen/Qwen2.5-Math-1.5B \
  --train-path integrl/data/gsm8k/sft.jsonl \
  --val-path integrl/data/gsm8k/test.jsonl \
  --eval-prompt integrl/prompts/r1_zero.prompt \
  --epochs 1 \
  --checkpoint-dir integrl/checkpoints/my_run
```
