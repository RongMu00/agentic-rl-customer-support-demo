# Agentic RL Customer-Support Case Study

> A reviewer-friendly reproduction of the runtime, verifier, SFT, rollout, GRPO, and held-out evaluation path from a full-scale customer-support agent post-training system.

I built the full experiment around a Qwen3-8B agent for after-sales automation with more than 30 tools and eight-way online GRPO rollouts. It improved held-out task success from **8% to 93%** while reducing unauthorized actions from **28% to 3%**.

Reproducing that training run requires a multi-GPU environment. This repository instead provides three deliberately separate layers:

- a zero-dependency runtime and verifier demo that runs on any laptop in under a minute;
- a single-GPU public reproduction with LoRA SFT, model-generated rollout groups, verifier rewards, TRL GRPO, and held-out before/after evaluation;
- small contract checks that isolate masking and reward-to-gradient behavior.

[![Open the public reproduction in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RongMu00/agentic-rl-customer-support-demo/blob/main/notebooks/agentic_rl_public_reproduction.ipynb)

## Results: two scopes, kept separate

### Full-scale experiment

| Metric | Before | After |
| --- | ---: | ---: |
| Held-out task success | 8% | 93% |
| Unauthorized action rate | 28% | 3% |

These are results from the original Qwen3-8B multi-GPU experiment I designed and implemented. They are not claimed to be reproduced by the smaller public codebase.

### Public reproducible experiment

The Colab experiment evaluates `Base → SFT → SFT + GRPO` on 24 untouched synthetic held-out cases. The repository intentionally does not commit placeholder improvement numbers. Running the notebook produces `results/public_base.json`, `results/public_sft.json`, and `results/public_grpo.json` with:

- valid tool-call rate;
- task completion and mean verifier reward;
- policy compliance and unauthorized-action rate;
- false-promise and grounded-response rate.

Once a complete run is recorded, its measured table can be added here without mixing it with the full-scale results.

## What you can run here

The demo includes three synthetic cases:

| Scenario | What it demonstrates | Expected result |
| --- | --- | --- |
| `successful_refund` | Evidence gathering, dry run, authorized write, grounded reply | Full reward, no caps |
| `transient_failure` | Bounded retry and exactly-once side effect | Recovery without duplicate refund |
| `unauthorized_refund` | Write allowlist and auditable blocking | No refund ledger entry, reward capped |

The core scenarios need no model download, API key, external service, or third-party Python package.

## Quick start

```bash
git clone https://github.com/RongMu00/agentic-rl-customer-support-demo.git
cd agentic-rl-customer-support-demo
python -m agentic_rl_demo successful_refund
```

Try the reliability and safety cases:

```bash
python -m agentic_rl_demo transient_failure
python -m agentic_rl_demo unauthorized_refund
```

Machine-readable output:

```bash
python -m agentic_rl_demo successful_refund --json
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Optional editable install:

```bash
python -m pip install -e .
agentic-rl-demo --list
```

## Single-GPU public reproduction

The recommended path is the [Colab A100 notebook](notebooks/agentic_rl_public_reproduction.ipynb). It executes:

```text
Base held-out evaluation
  → LoRA SFT on 96 synthetic training cases
  → SFT held-out evaluation
  → four model-generated candidates per prompt
  → sandbox execution + verifier reward
  → TRL GRPO updates
  → final held-out evaluation
```

The train and held-out records use unique order IDs and are split before serialization. Generated completions must follow the public `<tool_call>…</tool_call><final>…</final>` protocol; malformed outputs receive zero reward. Valid plans are executed through the same write allowlist, sandbox, side-effect ledger, and verifier used by the zero-dependency demo.

Pinned Colab dependencies live in `public_experiment/requirements-colab.txt`. The default public model is `Qwen/Qwen2.5-0.5B-Instruct`; the objective is to expose a complete, inspectable training path, not to present the small model as production-ready.

## Contract and optimizer smoke checks

The optional smoke environment answers a narrower question than the full experiment: **can data, masking, reward adaptation, group-relative advantages, loss, backward pass, and an optimizer step connect successfully?**

Install the optional model dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[training-smoke]"
```

First run the CPU-only contract check. It requires no model download:

```bash
python -m training_smoke.validate
```

Run one assistant-only SFT update:

```bash
python -m training_smoke.sft.train_one_step --device auto
```

Run one offline, sequence-level GRPO-style update:

```bash
python -m training_smoke.grpo.train_one_step --device auto
```

Both model checks default to `Qwen/Qwen2.5-0.5B-Instruct`. A CUDA GPU is recommended; Apple Silicon MPS and CPU are supported for functional checks but can be slower and use substantial memory. Use `--model` to supply another compatible causal language model. Neither command saves a checkpoint unless the SFT command receives `--output-dir`.

The isolated RL smoke check replays a safe and an unsafe candidate for the **same customer request**. It is retained as a debugging contract for reward-to-gradient wiring and is not the repository's GRPO experiment. The public experiment above uses model-generated completions and TRL's `GRPOTrainer`.

## Example output

```text
Successful evidence-grounded refund
===================================
  ✓ orders.get
  ✓ policy.search
  ✓ refunds.simulate
  ✓ refunds.issue
  ✓ customer.reply

Verifier
--------
  outcome        1.00
  policy         1.00
  evidence       1.00
  efficiency     1.00
  communication  1.00
  final reward   1.00

No reward caps triggered.
```

In the blocked scenario, `refunds.issue` is attempted outside the write allowlist. The runtime records the attempt, blocks the side effect, and leaves the refund ledger empty. The verifier then applies an `unauthorized_action_cap`.

## System design

```mermaid
flowchart LR
    A["Synthetic case"] --> B["Agent plan"]
    B --> C["Safe runtime"]
    C --> D["Tool registry"]
    D --> E["Read-only business data"]
    D --> F["Rollout sandbox"]
    C --> G["Trajectory"]
    F --> G
    G --> H["Verifier"]
    H --> I["Subscores + caps"]
    I --> J["Reward"]
```

See [the architecture walkthrough](docs/architecture.md) for the runtime, isolation, failure, and verifier boundaries.

## Reward design

The verifier separates five concerns:

```text
raw reward = outcome × 0.45
           + policy × 0.20
           + evidence × 0.20
           + efficiency × 0.10
           + communication × 0.05

final reward = min(raw reward, lowest active critical-error cap)
```

This matters because a fluent answer should not compensate for an unsafe action. The demo implements caps for:

- missing refund dry run;
- unauthorized write attempts;
- duplicate side effects;
- false promises not supported by sandbox state;
- writes performed without required evidence.

## Core demo, public reproduction, and full experiment

| Component | Core runtime demo | Public reproduction | Full experiment |
| --- | --- | --- | --- |
| Purpose | Inspect execution, safety, and reward behavior | Reproduce the end-to-end training path | Train and evaluate the full-scale agent |
| Action source | Scripted actions for repeatable runtime tests | Qwen2.5-0.5B model generations | Qwen3-8B dynamically generated tool calls |
| Tools | 5 synthetic tools | Same public synthetic tools | 30+ after-sales tools |
| Rollouts | One local execution | Four generations per prompt | Eight-way asynchronous online rollouts |
| State | In-memory namespaced sandbox | Same sandbox across isolated rollouts | Rollout-scoped execution sandboxes |
| Verification | Deterministic ledger, evidence, and policy checks | Same verifier used as the GRPO reward | Rules plus fact-grounded semantic judge |
| Training | None | LoRA SFT + TRL GRPO | SFT + online GRPO |
| Evaluation | Runtime unit cases | 24 held-out synthetic cases | Full held-out evaluation corpus |
| Infrastructure | Python standard library | Colab A100, PEFT, TRL | verl, vLLM, FSDP, multi-GPU |

The reported 8% → 93% success and 28% → 3% unauthorized-action results come from the original full-scale experiment. They are recorded in [`results/full_experiment_summary.json`](results/full_experiment_summary.json) and are **not** claimed to be reproduced by this lightweight demo.

## Repository map

```text
agentic_rl_demo/
  runtime.py       bounded execution and retry
  sandbox.py       rollout-scoped mutable state
  tools.py         synthetic read/write tools
  verifier.py      subscores, caps, and diagnostics
  models.py        trajectory and scenario models
scenarios/         three auditable JSON cases
tests/             safety, recovery, and isolation tests
training_smoke/    SFT and offline GRPO-style one-step checks
public_experiment/ synthetic data, LoRA SFT, GRPO, and evaluation
notebooks/         end-to-end Colab A100 reproduction
docs/              architecture and design boundaries
results/           summary of the full experiment
```

## Why this repository is intentionally small

The goal is not to disguise a research cluster as a one-command reproduction. The goal is to let a reviewer inspect and run the parts that carry the engineering argument:

1. mutable actions are isolated by rollout;
2. transient failures can recover without duplicate writes;
3. attempted and executed actions remain distinguishable;
4. final claims are checked against side-effect ledgers;
5. critical errors cap reward instead of becoming small deductions.

The public reproduction adds evidence that model-generated trajectories, sandbox outcomes, verifier rewards, and model updates connect end to end. It does **not** reproduce the reported full-experiment metrics.

## Attribution and publication boundary

This is an authorized, simplified public reproduction of the full post-training system I built. All public scenarios and business records are synthetic. The original training data, model weights, prompts, customer information, infrastructure configuration, and evaluation corpus are not included.

See [NOTICE.md](NOTICE.md) for copyright and reuse terms.

For the design narrative and production boundary, see the [case study](docs/case-study.md).

## Author

**Rong Mu** — AI Software Engineer · [GitHub](https://github.com/RongMu00)
