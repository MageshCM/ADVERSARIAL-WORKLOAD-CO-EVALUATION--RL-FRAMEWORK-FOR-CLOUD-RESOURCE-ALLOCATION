# AWC-RL – Adversarial Workload Co-Evolution for Reinforcement Learning

Reference prototype for the B.Tech project **“Adversarial Workload Co-Evolution – RL Framework for Cloud Resource
Allocation”** (Saveetha Engineering College, Project Work Phase I, Review 2).

An **Allocator Agent** (tabular Q-learning) that scales replicas is trained against an **Adversary Agent**
(cross-entropy-method search) that learns bounded, plausible workload traces that maximise the allocator's loss.
A **Curriculum Manager** raises workload difficulty only while the allocator copes, and a **Guardrail Layer**
filters unsafe actions. The **robustness gap** (adversarial loss − nominal loss) measures resilience.

> **Scope of the evidence.** Everything here runs in a *simulated* cloud (digital twin). The results are not production
> measurements and the allocator is tabular, not deep RL (planned for Phase II). See “Limitations”.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -v        # 19 tests
python train.py                       # 5 seeds, ~45 s on one CPU core -> results.json
python scripts/make_figures.py        # result plots -> docs/figures/
```

`python train.py --seeds 0 1 --out quick.json` runs a smaller experiment. Every random generator is seeded, so the
committed `results.json` is reproduced exactly (this repository was verified value-by-value against the numbers
reported in the project report, slides and IEEE paper).

## Demo

```bash
python demo.py                                # ~20 s: trains, attacks, then animates (opens a window)
python demo.py --save docs/figures/demo.gif   # same, saved as a GIF
```

![demo](docs/figures/demo.gif)

A step-by-step plan for recording a video is in [`docs/DEMO.md`](docs/DEMO.md).

## Results (5 seeds, mean per-step loss – lower is better)

| Method | Nominal | Adversarial (red-team) | Robustness gap | SLA-violating steps, adv. | Flash crowd (OOD) | Sawtooth (OOD) |
|---|---|---|---|---|---|---|
| HPA (60 % target) | 0.636 | 1.238 | 0.603 | 17.4 % | 0.731 | 0.650 |
| Static-12 | 0.651 | 2.301 | 1.650 | 77.5 % | 1.013 | 1.033 |
| Standard-RL (nominal traces only) | 0.596 | 2.003 | 1.407 | 64.0 % | 0.981 | 0.935 |
| **AWC-RL** | 0.660 | **1.217** | **0.557** | 19.4 % | 0.898 | 0.920 |

* Versus Standard-RL (same 3,000 training episodes): adversarial loss −39 %, robustness gap −60 %, at +11 % nominal loss.
* AWC-RL is **comparable to, not better than,** the tuned HPA rule; HPA is better on nominal and out-of-family traces.
* Gains on out-of-family workloads are small (8 % flash crowd, 2 % sawtooth).

## Repository layout

```
awc_rl/
  config.py      all constants / hyper-parameters (one frozen dataclass)
  workload.py    WorkloadGenerator: nominal, adversarial (6 parameters + difficulty), flash-crowd, sawtooth
  env.py         CloudEnv digital twin: 2-step provisioning delay, SLA, cost, 792-state discretisation
  guardrail.py   Guardrail: no scale-down under pressure, min replicas, quota (ready + pending)
  policy.py      Policy interface + run_episode (episode runner, separated from metric logic)
  allocator.py   Allocator Agent – tabular Q-learning
  adversary.py   Adversary Agent – cross-entropy method
  curriculum.py  CurriculumManager – difficulty follows the robustness gap
  baselines.py   HpaBaseline, StaticAllocation
  trainer.py     Trainer.train_standard / train_awc (co-evolution loop)
  metrics.py     evaluate_policy (nominal, red-team, out-of-family), robustness_gap
train.py         experiment driver (writes results.json)
demo.py          animated demo (train → attack → compare)
scripts/         make_figures.py
tests/           unit / integration tests (TC-01 … TC-10 of the report + regression tests)
docs/            figures (design diagrams + results), CODE_REVIEW.md
```

### Mapping to the project documents

| Document element | Code |
|---|---|
| Architecture: Allocator Agent / Adversary Agent / Curriculum Manager / Guardrail / Digital twin / Evaluation harness | `allocator.py` / `adversary.py` / `curriculum.py` / `guardrail.py` / `env.py` / `metrics.py` |
| Class diagram (Fig 4.8): CloudEnv, Policy, Allocator, HpaBaseline, AdversaryAgent, WorkloadGenerator, CurriculumManager, Trainer, Evaluator | same class names (`Adversary` = AdversaryAgent, `evaluate_policy` = Evaluator) |
| SRS FR-02, FR-07 | `env.py`, `allocator.py` |
| FR-03, FR-04 | `workload.py` (6 parameters, plausibility envelope) |
| FR-05, FR-06 | `trainer.py`, `curriculum.py` |
| FR-08 | `guardrail.py` |
| FR-09, FR-10, FR-11 | `metrics.py`, `train.py` (JSON log; DB schema in the report is a design only) |
| Table 7.1 TC-01 … TC-10 | `tests/test_awc_rl.py` (`test_tc01_…` … `test_tc10_…`) |
| Code review CR-01 … CR-08 | see `docs/CODE_REVIEW.md` |

Design diagrams (architecture, DFD, ER, class, sequence, flow chart, use case, UI wireframes) are in `docs/figures/`
(Graphviz sources in `docs/figures/src/`).

## Method in one paragraph

State = (demand pressure, trend, ready replicas, pending replicas) → 792 buckets; actions Δ ∈ {−2, −1, 0, +1, +3};
loss = 1.0·cost + 1.5·SLA-violation + 4.0·dropped-fraction. Training warm-starts on 600 nominal episodes, then runs
60 rounds of (CEM adversary update → curriculum update → 40 allocator episodes, 65 % adversarial incl. an archive of
past adversaries). Evaluation uses held-out nominal traces, a *fresh* red-team adversary at full difficulty
(24 candidates × 15 iterations) and two out-of-family shapes, over five seeds.

## Limitations

* Simulation only: constant provisioning delay, noise-free telemetry, synthetic traces.
* Tabular Q-learning; PPO/DDPG and a learned (neural) adversary are future work.
* The red-team adversary uses the same search method and parameter family as the training adversary (optimistic bias).
* Five seeds give a coarse variance estimate; the HPA set-point and loss weights influence the ranking.

## Roadmap (Phase II)

Deep-RL allocator/adversary · real production traces · richer state (GPU, memory, latency percentiles) ·
Kubernetes testbed with shadow-mode validation · web console and database back-end from the design chapter.

## Author

Magesh C M (212223220053) · Guide: Dr. Malathi K · Department of Information Technology, Saveetha Engineering College.
