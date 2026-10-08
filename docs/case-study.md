# Case study: safer agent post-training for customer support

## The problem

Customer-support agents do more than generate text. They retrieve order and policy evidence, call operational tools, mutate financial state, recover from failures, and explain outcomes to customers. A fluent response is not enough: an agent can sound correct while issuing an unauthorized refund or promising an action that never happened.

I built the full-scale Agentic RL post-training system to improve task completion without treating safety failures as minor quality deductions.

## Full-scale experiment

The original experiment used Qwen3-8B, more than 30 after-sales tools, rollout-scoped sandboxes, eight-way online GRPO, and a hybrid verifier across verl, vLLM, FSDP, and multi-GPU infrastructure.

| Held-out metric | Before | After |
| --- | ---: | ---: |
| Task success | 8% | 93% |
| Unauthorized action rate | 28% | 3% |

These are results from the full experiment I built. The public repository does not claim to recreate those numbers with a 0.5B model.

## Key engineering decisions

### Make side effects observable

Every rollout receives an isolated namespace. Tool attempts, successful calls, blocked writes, refunds, and customer messages remain separately auditable. The verifier can therefore distinguish “the model tried to issue a refund” from “a refund was actually recorded.”

### Treat critical failures as caps

The verifier scores outcome, policy compliance, evidence, efficiency, and communication. Missing dry runs, unauthorized actions, duplicate writes, unsupported promises, and writes without evidence impose hard reward caps. A polished response cannot average away a financial safety failure.

### Keep the public reproduction honest

The full environment is too large and proprietary for a useful portfolio reproduction. The public version keeps the same conceptual interfaces while reducing scale:

| Full experiment | Public reproduction |
| --- | --- |
| Qwen3-8B | Qwen2.5-0.5B-Instruct |
| 30+ production tools | 5 synthetic representative tools |
| Production evaluation corpus | 96 synthetic training + 24 held-out cases |
| Eight-way asynchronous rollouts | Four model-generated completions per prompt |
| Multi-GPU online GRPO | Single-GPU TRL GRPO with LoRA |
| Hybrid fact-grounded verifier | Deterministic state- and policy-grounded verifier |

## What a reviewer can run

The repository has two paths:

1. A one-minute, zero-dependency path executes successful, retry, and unauthorized-action scenarios and runs the complete CPU test suite.
2. A Colab A100 notebook evaluates the base model, trains a LoRA SFT adapter, evaluates it, performs verifier-reward GRPO on model-generated completions, and evaluates the final adapter on the same untouched held-out split.

The public notebook reports valid tool calls, task completion, policy compliance, unauthorized actions, false promises, grounded responses, and mean reward. Public numbers are added only after a completed run; they remain separate from the full-scale results above.

## Production boundary

The public experiment demonstrates architecture, training interfaces, and evaluation discipline. It does not establish that a 0.5B model is ready for real customer traffic. A production deployment would still require representative policy coverage, adversarial evaluation, privacy controls, durable idempotency, monitoring, escalation paths, and human review for high-risk actions.
