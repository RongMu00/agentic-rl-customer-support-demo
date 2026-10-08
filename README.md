# Agentic RL Safety Demo

> A zero-GPU demonstration of safe multi-step agent execution, rollout isolation, failure recovery, auditable side effects, and reward verification.

The full project trained a Qwen3-8B agent for after-sales automation with more than 30 tools and eight-way online GRPO rollouts. It improved held-out task success from **8% to 93%** while reducing unauthorized actions from **28% to 3%**.

Reproducing that training run requires a multi-GPU environment. This repository instead makes the most important engineering mechanisms runnable on any laptop in under a minute.

## What you can run here

The demo includes three synthetic cases:

| Scenario | What it demonstrates | Expected result |
| --- | --- | --- |
| `successful_refund` | Evidence gathering, dry run, authorized write, grounded reply | Full reward, no caps |
| `transient_failure` | Bounded retry and exactly-once side effect | Recovery without duplicate refund |
| `unauthorized_refund` | Write allowlist and auditable blocking | No refund ledger entry, reward capped |

No model download, API key, external service, or third-party Python package is required.

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

## Public demo vs. full experiment

| Component | Public demo | Full experiment |
| --- | --- | --- |
| Action source | Deterministic plan | Qwen3-8B policy |
| Tools | 5 synthetic tools | 30+ after-sales tools |
| Rollouts | One local execution | Eight-way asynchronous rollouts |
| State | In-memory namespaced sandbox | Rollout-scoped execution sandboxes |
| Verification | Deterministic rules | Rules plus fact-grounded judge |
| Training | Not included | SFT and online GRPO |
| Infrastructure | Python standard library | verl, vLLM, FSDP, multi-GPU |

The reported 8% → 93% success and 28% → 3% unauthorized-action results come from the full private experiment. They are recorded in [`results/full_experiment_summary.json`](results/full_experiment_summary.json) and are **not** claimed to be reproduced by this lightweight demo.

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

## Attribution and publication boundary

This is an authorized, simplified demonstration derived from the architecture of an industrial post-training system. All scenarios and business records in this repository are synthetic. The original training data, model weights, prompts, customer information, infrastructure configuration, and proprietary evaluation corpus are not included.

See [NOTICE.md](NOTICE.md) for copyright and reuse terms.

## Author

**Rong Mu** — AI Software Engineer · [GitHub](https://github.com/RongMu00)
