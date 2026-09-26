# Jailbreak Self-Play

A defensive training loop that hardens a small language model against
adversarial prompts, measured on
[HarmBench](https://github.com/centerforaisafety/HarmBench).

Each round, Jailbreak-R1 writes prompts for a HarmBench behaviour.
Qwen2.5-1.5B-Instruct answers them. The HarmBench Mistral classifier marks
an answer successful when that answer fulfils the behaviour.
Qwen3-4B-Instruct-2507 writes a refusal for each success. Those pairs,
mixed with ordinary Alpaca instructions answered by the untrained defender,
fine-tune a fresh LoRA adapter. The next round attacks that adapter.

All four models load from Hugging Face, one at a time, in 4-bit. The number
to report is attack success on a frozen set of held-out behaviours, next to
the refusal rate on safe XSTest prompts. The success rate printed while
training is a progress signal on adaptive attacks.

```
                   ┌──────────────────────────────────────────┐
                   │  HarmBench behaviours (train split)      │
                   └───────────────────┬──────────────────────┘
                                       ▼
   ┌──────────────┐   prompts   ┌──────────────┐  responses  ┌──────────────┐
   │  Attacker    │────────────▶│  Defender    │────────────▶│  Classifier  │
   │ Jailbreak-R1 │             │  Qwen2.5 +   │             │  HarmBench   │
   │  (local HF)  │◀────────────│  LoRA        │             │  Mistral cls │
   └──────────────┘  what worked└──────▲───────┘             └──────┬───────┘
                                       │                            │
                                       │  LoRA SFT                  │ label=1
                                 ┌─────┴────────────────────────────▼──────┐
                                 │ refusal pairs + benign retention data   │
                                 └─────────────────────────────────────────┘

   Reported numbers come from a FROZEN probe set on held-out behaviours,
   never from the adaptive in-loop attacks.
```


**[REPORT.md](REPORT.md)** is the full write-up: methodology, design decisions,
rejected alternatives, the diagnostic results from that first run, how the
implementation was validated, and known weaknesses. **No robustness result has
been produced yet** — the pipeline is built and tested but has not been run to
completion.

---


## Layout

```
.
├── configs/default.yaml           # the whole experiment, one file
├── src/selfplay/
│   ├── config.py                  # typed config; secrets come from env
│   ├── data.py                    # HarmBench loading, categories, splits
│   ├── local_lm.py                # Hugging Face load, generate, unload
│   ├── llm_client.py              # OpenRouter client, only for an API judge
│   ├── redteam.py                 # Jailbreak-R1 attack generation
│   ├── defender.py                # 4-bit load, LoRA, batched generation
│   ├── judge.py                   # HarmBench classifier scoring
│   ├── safety_data.py             # refusal pairs + retention pairs
│   ├── train.py                   # LoRA SFT, prompt masking, collator
│   ├── metrics.py                 # ASR, per-category ASR, over-refusal
│   └── utils.py                   # logging, seeding, GPU release, I/O
├── scripts/
│   ├── fetch_harmbench.sh         # download behaviour CSVs
│   ├── build_probe_set.py         # generate and freeze the eval set
│   ├── run_selfplay.py            # the training loop
│   ├── evaluate.py                # probe ASR and over-refusal
│   └── _bootstrap.py              # puts src/ on the path; not run directly
├── tests/                         # pure-python, no GPU or network
└── notebooks/                     # original exploration, outputs stripped
```

---

## Setup

Python 3.10 or newer, and one NVIDIA GPU with CUDA. Attacker, defender,
classifier, and refusal writer load one at a time in 4-bit NF4. The largest
default model is 7B, so 16 GB is enough. `configs/default.yaml` is the
experiment: models, split sizes, learning rate, and round count.

The default Hugging Face repos are public. Set `HF_TOKEN` only for a gated
repo. Set `OPENROUTER_API_KEY` only if you change `judge.model_id` to an
API instruct model.

### Install

Linux or macOS:

```bash
git clone https://github.com/vatsal-2110/Jailbreak-Attack-Defense-Co-Evolutionary-Loop-Adversarial-Self-Play.git
cd jailbreak-selfplay
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .    # optional; the scripts add src/ themselves
```

Windows (PowerShell):

```powershell
git clone https://github.com/vatsal-2110/Jailbreak-Attack-Defense-Co-Evolutionary-Loop-Adversarial-Self-Play.git
cd jailbreak-selfplay
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Optional credentials:

```bash
export HF_TOKEN="..."          # Linux / macOS
```

```powershell
$env:HF_TOKEN = "..."          # PowerShell
```

### Behaviour data

The loop reads `data/harmbench_behaviors_text_test.csv`. The fetch script
also saves the validation split.

```bash
./scripts/fetch_harmbench.sh data
```

On Windows, Git Bash can run that script. From PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
$base = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets"
foreach ($split in "val", "test") {
  curl.exe -fsSL "$base/harmbench_behaviors_text_$split.csv" -o "data/harmbench_behaviors_text_$split.csv"
}
```

`configs/default.yaml` keeps the `standard` category only: 20 training
behaviours and 20 probe behaviours, seed 4. Copyright behaviours stay out
until `judge.copyright_reference_dir` points at reference texts.

---

## Scripts

| Script | What it does |
|---|---|
| `scripts/fetch_harmbench.sh` | Downloads the HarmBench val and test behaviour CSVs into `data/`. |
| `scripts/build_probe_set.py` | Writes the frozen evaluation attacks once. |
| `scripts/run_selfplay.py` | Runs attack, score, refusal, and LoRA training. |
| `scripts/evaluate.py` | Scores `base` and saved adapters on that frozen set. |
| `scripts/_bootstrap.py` | Adds `src/` to `sys.path`. The other Python scripts import it. |

### `build_probe_set.py`

Generates attacks on the held-out behaviours with no attack history, so every
later checkpoint is scored on the same prompts.

```bash
python scripts/build_probe_set.py --config configs/default.yaml
python scripts/build_probe_set.py --config configs/default.yaml --attacks-per-behavior 5
python scripts/build_probe_set.py --config configs/default.yaml --force
```

`--attacks-per-behavior` defaults to `redteam.attacks_per_behavior` (5).
`--force` overwrites an existing set and makes earlier scores incomparable.

Writes `runs/jailbreak_selfplay/probe/probe_attacks.json` and
`probe_manifest.json` (SHA-256, behaviour ids, attacker id).

### `run_selfplay.py`

```bash
python scripts/run_selfplay.py --config configs/default.yaml
python scripts/run_selfplay.py --config configs/default.yaml --rounds 2
```

`--rounds` overrides `num_rounds`. With the default of 4, rounds 0–4 generate
and score. Training runs after rounds 0–3 and saves
`runs/jailbreak_selfplay/checkpoints/D1` through `D4` when that round has
usable refusals. Round 0 is the base model. The last round does not train.
A round with no classifier success, or with every Qwen3 refusal rejected,
keeps the previous adapter and does not write a new checkpoint.

Also writes `behavior_split.json`, `retention_examples.json`,
`rounds/round_<n>_attacks.json`, `round_<n>_results.json`,
`round_<n>_summary.json`, and `all_results.json` under
`runs/jailbreak_selfplay/`.

### `evaluate.py`

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoints base
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoints base \
        runs/jailbreak_selfplay/checkpoints/D1 \
        runs/jailbreak_selfplay/checkpoints/D4
```

`--checkpoints` is required. `base` is the untrained defender; any other
value is an adapter directory. `--probe` overrides the default
`probe_attacks.json`. `--skip-overrefusal` drops the XSTest column.
`--out` sets the output directory (default `runs/jailbreak_selfplay/eval/`).

Writes `eval_reports.json` and `eval_table.md`:

```
| Checkpoint | Probe ASR % | scored/total | Over-refusal % |
| --- | ---: | ---: | ---: |
| base | ... | ... | ... |
| D1   | ... | ... | ... |
```

A checkpoint improved when probe ASR fell and over-refusal did not rise to
meet it.

---

## Run

Do these in order. Build the probe set before the first evaluation, and do
not regenerate it while comparing checkpoints.

```bash
# 1. Freeze the evaluation set. Once per experiment.
python scripts/build_probe_set.py --config configs/default.yaml

# 2. Baseline, before any training.
python scripts/evaluate.py --config configs/default.yaml --checkpoints base

# 3. Self-play. Prints in-loop ASR only; that is not the result.
python scripts/run_selfplay.py --config configs/default.yaml

# 4. Reportable numbers for every checkpoint that was saved.
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoints base \
        runs/jailbreak_selfplay/checkpoints/D1 \
        runs/jailbreak_selfplay/checkpoints/D2 \
        runs/jailbreak_selfplay/checkpoints/D3 \
        runs/jailbreak_selfplay/checkpoints/D4
```

Step 4 is the comparison to keep. If a training round was skipped, leave
that `D*` path out of `--checkpoints`.

---

## Design notes

### Evaluation

The in-loop ASR printed by `run_selfplay.py` is a progress signal, not a
result. It is measured on attacks that adapt to each checkpoint, against
behaviours the defender is trained on, so a change in it conflates a change in
the defender with a change in the attacker.

Reportable numbers come from `evaluate.py` against the probe set: generated
once, with no attack history, on behaviours in a disjoint split, hashed in
`probe_manifest.json`. Regenerating it mid-experiment invalidates every earlier
result, so `build_probe_set.py` refuses to overwrite without `--force`.

### Judging

HarmBench's three functional categories need three different treatments:

| Category | Scored by | Note |
|---|---|---|
| `standard` | HarmBench classifier | Self-contained request |
| `contextual` | same **plus `ContextString`** | The behaviour is meaningless without it |
| `copyright` | n-gram overlap against a reference text | A harm classifier calls verbatim reproduction "safe" |

Switch the scorer in `configs/default.yaml` by changing `judge.model_id`:

| `judge.model_id` | Where it runs |
|---|---|
| [`cais/HarmBench-Mistral-7b-val-cls`](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls) (default) | Local Hugging Face, official Yes/No template |
| [`cais/HarmBench-Llama-2-13b-cls`](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | Local Hugging Face, official Yes/No template |
| `meta-llama/llama-3.3-70b-instruct` | OpenRouter, reply `0`/`1` |

The classifier is loaded only while scoring. `run_selfplay.py` and `evaluate.py` unload the defender before loading it, then unload the classifier before the next model.

**Do not use a generic content-safety classifier as the judge.** Llama Guard,
`nemotron-*-content-safety` and `gpt-oss-safeguard-*` grade text against their
own harm taxonomy and emit their own schema. The question here is different —
*did this response fulfil this specific target behaviour?* — and forcing them
to answer it is what produced the original notebook's empty responses.

The default scorer is the local HarmBench classifier, so labels match the
official Yes/No behaviour test. An OpenRouter instruct judge is still
available by changing `judge.model_id`, but those scores are not directly
comparable to published HarmBench numbers.

Copyright behaviours are excluded by default (`data.functional_categories`)
because scoring them needs reference texts, which this repo does not ship. To
include them, set `judge.copyright_reference_dir` to a directory of
`<behavior_id>.txt` files. Without a reference an item is **unscored**, not
"safe".

### Training

`restart_policy` picks how checkpoints relate:

- **`from_base`** (default) — fresh base model and fresh adapter each round,
  trained once on the cumulative dataset. `D_n` depends only on rounds `< n`.
  Costs a model reload per round.
- **`incremental`** — keep the adapter, train on the new round's data only.
  Cheaper, and each example is seen once.


### Over-refusal

Training only on `(attack → refusal)` pairs teaches "refuse everything", which
scores a perfect 0% ASR. Two things counter it:

1. **Retention data** — benign instructions answered by the defender's own
   pre-training-round weights, mixed in at `train.retention_ratio`. Anchoring
   on the model's own answers avoids needing a stronger teacher.
2. **The over-refusal metric** — refusal rate on XSTest safe prompts, reported
   beside ASR.

The refusal detector in `metrics.py` is a conservative regex. It is a cheap
screen for a trend, not a classifier; for a headline number, judge those
responses with a model.

---

## Tests

```bash
pytest tests/ -q
```

They cover attack
parsing, judge-label parsing and the unscored-vs-safe distinction, prompt
masking and collation, and the train/probe split.

---

## Intended use

Defensive safety research: measuring and improving a model's robustness to
adversarial prompts, on a public benchmark built for that purpose. The
attacker exists to produce training signal for the defender.

<!-- Practical notes:

- `runs/` is gitignored. It holds model completions to adversarial prompts —
  review before sharing, and do not commit generated attacks or responses.
- The notebook under `notebooks/` has its outputs stripped for the same reason.
- Checkpoints here are deliberately over-refusing by construction. They are
  experimental artefacts, not general-purpose assistants.
- HarmBench's own terms of use apply to the behaviour data. -->

## License

MIT — see [LICENSE](LICENSE). Cited datasets, models and benchmarks carry
their own licenses, listed above.
