# Training smoke checks

This directory provides a small, inspectable bridge between the public agent runtime and a real optimizer step. It is a functional check, not a reproduction of the private multi-GPU experiment.

## Layers

### Dependency-free contract check

```bash
python -m training_smoke.validate
```

This validates the synthetic SFT records, runs the safe and unsafe candidates through the public runtime and verifier, and computes group-relative advantages. It does not load a language model.

### One-step SFT

```bash
python -m training_smoke.sft.train_one_step --device auto
```

The script loads a chat-template-compatible causal LM, masks the system and user prompt, computes loss only on the final assistant target, runs backward, and applies one AdamW update. Pass `--output-dir PATH` if you want to retain a checkpoint; otherwise the updated weights are discarded after the check.

### One-step agentic RL

```bash
python -m training_smoke.grpo.train_one_step --device auto
```

The script uses two fixed candidates for one customer prompt:

- a safe plan that gathers evidence, performs the refund dry run, and executes authorized side effects;
- an unsafe plan that attempts an unauthorized refund and makes an unsupported completion claim.

The existing runtime and verifier produce rewards for both trajectories. The script standardizes those rewards within the group, calculates a sequence-level clipped policy objective, runs backward, and applies one AdamW update.

## What this proves

- SFT records reach an assistant-only causal LM loss.
- Verifier outputs reach group-relative advantages.
- Both objectives produce a finite backward pass and optimizer step on a real model.
- Training code can use the same safety semantics as the runnable demo.

## What this does not prove

- online rollout generation or exploration;
- convergence or benchmark improvement;
- token-level production GRPO parity;
- reference-policy KL behavior;
- vLLM, FSDP, verl, or multi-node orchestration;
- reproduction of the full Qwen3-8B experiment.

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Override it with `--model` when using a local or smaller chat-template-compatible causal model.
