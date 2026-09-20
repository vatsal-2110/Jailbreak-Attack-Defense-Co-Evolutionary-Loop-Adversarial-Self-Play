# Adversarial Self-Play for Jailbreak Robustness — Project Report

**Scope.** Build a closed loop in which a red-team model generates jailbreak
attempts against a defender model, a judge scores them, and the successful
attempts are converted into refusal training data for the next iteration of
the defender — evaluated on [HarmBench](https://github.com/centerforaisafety/HarmBench).

**Status, stated up front.** The pipeline is implemented, unit-tested, and
its data path is verified end-to-end. **It has not yet produced a robustness
result.** 

---

## 1. Problem framing

The underlying question is whether iterated adversarial pressure improves a
small instruction-tuned model's refusal behaviour faster than a static safety
dataset would.

The loop:

```
  HarmBench behaviour  ──▶  attacker LM  ──▶  adversarial prompts
                                                                  │
                                                                  ▼
 next defender  ◀── LoRA SFT ◀── refusal pairs ◀── judge ◀── defender
        │                             + benign                    responses
        └──────────────── attacked again next round ──────────────┘
```

Three properties make this harder than it looks, and each caused a concrete
defect in the first implementation:

1. **The judge is the measurement instrument.** Everything downstream the
   training set, the reported metric, the attacker's feedback signal is
   derived from its labels. A judge that fails silently doesn't produce noisy
   results; it produces confident wrong ones.
2. **The attacker is a generative process with no schema guarantee.** It is
   asked for JSON. 
3. **Optimising a single metric is trivially gameable.** A model that refuses
   every input scores a perfect 0% attack success rate.

---

## 2. Methodology

### 2.1 Components

| Component | Choice | Rationale |
|---|---|---|
| Defender | `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4 | Fits a single 16 GB GPU with room for training; small enough to iterate in hours, instruction-tuned so it has refusal behaviour to *move* |
| Adaptation | LoRA r=16, α=32, dropout=0.05, all attention + MLP projections | QLoRA-style 4-bit base + adapters; full fine-tuning is out of budget and would confound "safety trained" with "model degraded" |
| Attacker | Remote LM via OpenRouter, temperature 1.1 | Decoupled from the defender so attack quality doesn't degrade as the defender is modified; high temperature for strategy diversity |
| Judge | Remote general instruct model/Remote LM via OpenRouter, temperature 0 | Must answer "did this response fulfil *this* behaviour" |
| Behaviours | HarmBench | Standard benchmark, categorised, publicly citable |

### 2.2 Round structure

Round *n*: generate attacks conditioned on which earlier attacks succeeded →
defender answers → judge labels → successful attacks become `(attack →
refusal)` pairs → mix with benign retention data → LoRA SFT → checkpoint
`D_{n+1}`.

### 2.3 Measurement design

The single most important structural decision: **the loop's own attack success
rate is not the result.**

In-loop attacks adapt to each checkpoint and target behaviours the defender is
trained on. A change in that number conflates a change in the defender with a
change in the attacker, and is contaminated by training on the same
behaviours. It is a progress signal only.

Reportable numbers come from a **frozen probe set**:

- generated once, before any training,
- with no attack history (so it cannot adapt),
- against a **disjoint** set of held-out behaviours,

Every checkpoint is scored on that identical set, alongside an over-refusal
metric. Both are reported together, always.

In summary, training attacks tell you how the self-play process is progressing; a fixed, unseen test set tells you whether the defender actually became better.

---

## 3. Design decisions

### 3.1 Judging is routed by behaviour category

HarmBench's test split is not homogeneous — it is 159 `standard`, 81
`contextual`, 80 `copyright`. These need different treatment:

| Category | Scoring | Why not one judge for all |
|---|---|---|
| `standard` | LLM judge, one case per call | — |
| `contextual` | LLM judge **plus `ContextString`** | The behaviour text alone is a dangling referent: *"Slightly modify **this text**…"* — which text? The ~1 KB context is a separate CSV column |
| `copyright` | n-gram overlap vs. reference text | A harm classifier labels a faithfully reproduced book passage "safe" every single time, so these behaviours can never register a hit |

Copyright is excluded by default, because scoring it honestly needs reference
texts that this project cannot redistribute. Excluding it is a stated
limitation; scoring it with the wrong instrument would have been a silent one.


### 3.2 Refusal data is mixed with retention data

Training exclusively on `(attack → refusal)` pairs optimises directly for the
degenerate solution. Benign instructions, answered by the defender's *own*
pre-round weights, are mixed in at a configurable ratio — self-distillation,
so no stronger teacher model is required, and the anchor is the model's actual
prior behaviour rather than some other model's style.

Responses the model *already* refuses are dropped from the retention set:
reinforcing an existing false refusal is precisely backwards.

### 3.3 Round-to-round model state is explicit

`restart_policy` is a required, named choice:

- **`from_base`** — fresh base + fresh adapter each round, trained once on the
  cumulative dataset. `D_n` depends only on rounds `< n`. Costs a reload.
- **`incremental`** — keep the adapter, train on new data only. Cheaper, each
  example seen once.

Both are defensible. What is not defensible is doing neither — keeping the
adapter *and* replaying the cumulative dataset, which trains round 0's data
*n* times while the checkpoint name implies otherwise.

---

## 4. Rejected alternatives

| Rejected | In favour of | Reason |
|---|---|---|
| **Content-safety classifier as judge** (`llama-guard-*`, `nemotron-*-content-safety`) | General instruct model emitting `0`/`1` | These grade text against *their own* harm taxonomy and emit their own schema. The question here is different, did this response fulfil *this* target behaviour? Forcing the wrong question through them returned empty content on 100% of calls also loading llama-guard locally increases the computation and on its opposite smaller model hallucinates |
| **LLM Red Teamer** (`deepseek/deepseek-v4-flash-0731:free`)| openrouter LM for attacks generation | tried various models such as nvidia/nemotron-3.5-lightning:free, inclusionai/ling-3.0-flash-vl:free, z-ai/glm-5.2:free (instruction tuned, don't generate malicious/attacking prompts), thinkingmachines/inkling:free, thinkingmachines/inkling-small:free (agentic harness only, don't generate attacking prompts). deepseek/deepseek-v4-flash-0731:free was found best from all the models I used to serve as a attacker, but not free tier model any more😢.|
| **Batched judging** (10 cases per request) | One case per request | 10× cheaper, but a single mis-alignment silently mislabels up to 10 items, and a mislabelled item is indistinguishable from a correct one downstream. Retained as a config knob, off by default |
| **In-loop ASR as headline metric** | Frozen probe set on held-out behaviours | Adaptive attacks + trained-on behaviours measures memorisation and attacker drift, not robustness |
| **HarmBench test split for both training and reporting** | Disjoint train/probe split, asserted in code | Train/test contamination |
| **ASR as sole metric** | ASR **and** over-refusal, reported together | 0% ASR is achievable by refusing everything |
| **Single global mutated model across rounds** | Explicit `restart_policy` | Made "D2 trained on rounds 0–1" mean something other than what it says |
| **HarmBench official classifier** (`cais/HarmBench-Llama-2-13b-cls`) as default | Remote instruct judge | *Deferred, not rejected.* It is the correct instrument and the only way to get numbers comparable to published HarmBench results, but it is a 13B local model competing for the same GPU as the defender. |
| **DPO / RLAIF instead of SFT** | SFT on refusal pairs | Deferred. Needs paired preference data and more compute; SFT establishes whether the loop closes at all |

---

## 5. Experiments run and results

> **There is no robustness result to report.** could not find a free model to serve as a good attacker
> One execution reached round 0
> and produced zero usable training examples. Rounds 1–4 were never executed
> (`execution_count: null` on every training and later-round cell). No
> checkpoint was trained. The "0.00% ASR" the run printed is an artefact of a
> failed judge, not a measurement of defence.
>
> What follows is a diagnostic result: a quantified account of *why* the
> apparatus produced nothing, which is what motivated the rewrite.

### 5.1 Failure — total, and silent

| Metric | Value |
|---|---|
| Cases labelled `0` (safe) by the exception handler | **44 / 44** |
| ASR reported by the run | **0.00%** |
| Training examples produced | **0** |

One of the reasons I found for this was that the LM Red teamer(free tier models) were not able to generate attacking prompts, insted they were generating the type of respons that is naturally expected from a instruction tuned model (which is the goal) because of which the defender has no reason not to generate a response and the judge has no reason to mark that as a successful attack. 


### 5.2 Benchmark composition — the sample was mostly unscoreable

The 10 sampled behaviours(experimental), by HarmBench functional category:

| Category | Count | Scoreable by the pipeline as written? |
|---|---|---|
| `copyright` | 5 | **No** — safety judge always returns "safe" |
| `contextual` | 2 | **No** — `ContextString` was never read; behaviour text is a dangling referent |
| `standard` | 3 | Yes |

So even with a working judge, **7 of 10 behaviours could not have produced a
correct label**. The two behaviours that did yield well-formed attacks were
one `standard` and one `contextual`.

This is a sampling artefact of drawing uniformly from a split that is 25%
copyright and 25% contextual — but the pipeline had no category handling at
all, so the artefact was invisible.


## 6. What exists now

| Artefact | State |
|---|---|
| Pipeline implementation | 10 modules, ~1,730 LOC |
| Entry points | 4 — fetch, build probe set, self-play, evaluate (~470 LOC) |
| Test suite | 32 test functions / 37 cases, ~330 LOC |
| Configuration | Single YAML, typed, secrets from env |
| Frozen probe harness | Implemented, hashed, overwrite-protected |
| Over-refusal metric | Implemented |

---

## 8. Analysis

project develops an adversarial self-play framework for improving the jailbreak resistance of an LLM. A red-team LLM continuously generates adversarial prompts against a defender model, while a judge determines whether the generated response successfully satisfies the harmful target behavior. Successful jailbreaks are then converted into safety/refusal training examples and used to fine-tune the defender over multiple rounds.

A key design choice is that in-loop attack success rate is treated only as a progress signal, rather than the final evaluation metric. Since the attacker adapts to each defender checkpoint and the defender is trained on the same behaviours, using those attacks for evaluation would introduce significant bias. Instead, the project creates a frozen probe set before training, without attack history and on held-out behaviours. Every defender checkpoint is evaluated on exactly this same set.

The evaluation also measures over-refusal, so improved safety is not achieved simply by making the model refuse everything. 

Thus, the project evaluates two complementary objectives:
- Attack Success Rate (ASR): whether harmful jailbreak attempts succeed.
- Over-refusal rate: whether the model unnecessarily refuses benign requests.

Overall, the project is essentially an adaptive red-team → defense training → fixed evaluation loop, designed to study whether a defender can become more robust against jailbreaks while retaining useful behaviour.
---

## 9. Weaknesses and threats to validity


- **No result yet.** Everything above is apparatus.
- **Judge validity is unmeasured.** No agreement statistic against human
  labels. The judge is the instrument, and its error rate is currently
  unknown
- **Attacker and judge share a vendor** by default: a correlated failure mode.
- **Copyright behaviours are excluded**, so results cover 240 of 320 HarmBench
  behaviours and are not directly comparable to full-benchmark numbers.
- **A general instruct judge is not HarmBench's classifier**, so absolute ASR
  values are not comparable to published figures. Relative movement between
  checkpoints under one fixed judge remains meaningful.
- **Small scale by default** — 10 train + 10 probe behaviours. Enough to show
  the loop closes; far too few for a claim.
- **Model-written refusals** are never better than the generator, and are only
  length-filtered.
- **No attack-diversity metric.** The attacker is *asked* for diverse
  strategies; nothing verifies it. Expected to degrade in later rounds as
  history fills the prompt.
- **Free-tier attacker models** have no availability guarantee, so round
  composition may vary with provider load.

---

## 10. Challenges and Improvements

**Improvement of the Attacking Model.** Currently the pipeline is generalised to train only the defensive model not the attacking model. Improving both models together using some RLHF algorithm should be there so that the attacks can also be improved.

**Computational cost.** Repeated attack generation, response generation, judging, and LoRA fine-tuning across multiple self-play rounds is computationally expensive, particularly on limited GPU resources.

**Reliable jailbreak evaluation.** A simple safety classifier is not always sufficient to determine whether a response actually fulfills the target harmful behaviour, requiring a more capable judge and carefully designed evaluation criteria.

**Balancing safety and usefulness.** Fine-tuning on successful jailbreaks can make the defender safer but may also cause it to refuse legitimate requests. Measuring over-refusal is necessary to track this trade-off.

**Training-data quality.** Successful jailbreaks must be converted into appropriate refusal examples without introducing noisy or incorrectly labelled training data, since poor examples can negatively affect the defender.

**Choosing a judge is a modelling decision, not a config value.** The obvious
choice — a model with "content-safety" in its name — is the wrong instrument,
and picking it is what broke the original run. Recognising this required
reasoning about what the classifier was *trained to answer*.

---

## 11. With more time and compute

### 11.1 Immediate — produce the missing result

Run the experiment the apparatus was built for: baseline `D0` on the frozen
probe set, four self-play rounds, all five checkpoints evaluated on ASR and
over-refusal. 

### 11.2 Establish judge validity

Hand-label a stratified sample of ~200 cases; report a metric against the
automated judge. Cross-check with a second judge from a different vendor and
report disagreement rate. An ASR curve is worth exactly as much as the judge
behind it, and that number is currently unknown.

### 11.3 Adopt the reference classifier

Run `cais/HarmBench-Llama-2-13b-cls` locally with HarmBench's official prompt
templates, so absolute numbers are comparable to published results. Needs a
second GPU or sequential scheduling against the defender.

### 11.4 Scale

All 240 non-copyright behaviours rather than 10+10; several seeds; confidence
intervals. Source reference texts to bring the 80 copyright behaviours back
into scope with the n-gram classifier.

### 11.5 Strengthen the attacker

The current attacker is a single prompted LM. Stronger and better-studied
options — GCG's gradient-based suffixes,
AutoDAN — would give external validity: does hardening against a prompted
attacker transfer to an optimisation-based one, or is it attacker-specific
overfitting? This is the most interesting open question the setup can ask.

### 11.6 Measure attack diversity

Embed attacks, track pairwise similarity and cluster count per round. Test
directly whether history conditioning sustains novelty or collapses to
paraphrase — currently an assumption.

### 11.7 Ablations

- `retention_ratio` sweep: the ASR / over-refusal frontier.
- `from_base` vs `incremental`: does cumulative retraining beat incremental?
- History window size.
- Rounds-to-saturation: where does the curve flatten?

### 11.8 Method extensions

- DPO/RLVR on `(attack, refusal, harmful_completion)` triples instead of SFT.
- Multi-turn attacks — the current setup is single-turn, which excludes a
  large and practically important class of jailbreak.
- Transfer: does hardening on 10 behaviours generalise to 230 unseen ones? The
  probe-set design already supports asking this.
- Capability retention beyond refusal behaviour (MMLU or similar), to check
  the adapter isn't degrading the model generally.

---

## Appendix A — Reproducibility

```bash
pip install -r requirements.txt
./scripts/fetch_harmbench.sh data
export OPENROUTER_API_KEY=...

python scripts/build_probe_set.py --config configs/default.yaml   # once
python scripts/evaluate.py  --config configs/default.yaml --checkpoints base
python scripts/run_selfplay.py --config configs/default.yaml
python scripts/evaluate.py  --config configs/default.yaml \
    --checkpoints base runs/jailbreak_selfplay/checkpoints/D{1,2,3,4}
```

Seeds fixed in `configs/default.yaml`. Behaviour split is deterministic and
asserted disjoint. Probe set is hashed in `probe_manifest.json`; regenerating
it invalidates prior comparisons and requires `--force`.

## Appendix B — Where things live

| Concern | File |
|---|---|
| Fail-loud judging, category routing | [`src/selfplay/judge.py`](src/selfplay/judge.py) |
| Robust attack extraction | [`src/selfplay/redteam.py`](src/selfplay/redteam.py) |
| HarmBench categories, `ContextString`, splits | [`src/selfplay/data.py`](src/selfplay/data.py) |
| Prompt masking, collation | [`src/selfplay/train.py`](src/selfplay/train.py) |
| ASR excluding unscored, over-refusal | [`src/selfplay/metrics.py`](src/selfplay/metrics.py) |
| Frozen probe set | [`scripts/build_probe_set.py`](scripts/build_probe_set.py) |
| Reportable evaluation | [`scripts/evaluate.py`](scripts/evaluate.py) |
| Defect table with fixes | [`README.md`](README.md) |
| Original exploration (outputs stripped) | [`notebooks/00_original_exploration.ipynb`](notebooks/00_original_exploration.ipynb) |

Citations for every paper, dataset, model and library: [README.md](README.md#citations).
