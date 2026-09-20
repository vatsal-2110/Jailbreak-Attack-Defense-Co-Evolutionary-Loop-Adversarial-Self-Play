# Jailbreak Self-Play

Iterative red-team / defender self-play for adversarial robustness fine-tuning,
evaluated on [HarmBench](https://github.com/centerforaisafety/HarmBench).

An attacker model generates adversarial prompts for a target behaviour, a
defender model answers them, a judge labels each answer, and the successful
attacks are turned into refusal training data for the next defender. Repeat.

```
                   ┌──────────────────────────────────────────┐
                   │  HarmBench behaviours (train split)      │
                   └───────────────────┬──────────────────────┘
                                       ▼
   ┌──────────────┐   prompts   ┌──────────────┐  responses  ┌──────────────┐
   │  Attacker    │────────────▶│  Defender    │────────────▶│  Judge       │
   │  (remote LM) │             │  Qwen2.5 +   │             │  (routed by  │
   │              │◀────────────│  LoRA        │             │   category)  │
   └──────────────┘  what worked└──────▲───────┘             └──────┬───────┘
                                       │                            │
                                       │  LoRA SFT                  │ label=1
                                 ┌─────┴────────────────────────────▼──────┐
                                 │ refusal pairs + benign retention data   │
                                 └─────────────────────────────────────────┘

   Reported numbers come from a FROZEN probe set on held-out behaviours,
   never from the adaptive in-loop attacks.
```

This repository is a rewrite of an exploratory notebook
([`notebooks/00_original_exploration.ipynb`](notebooks/00_original_exploration.ipynb)).
The original run produced no usable results; the section below documents what
was wrong and what changed, because several of the bugs are easy to reintroduce.

**[REPORT.md](REPORT.md)** is the full write-up: methodology, design decisions,
rejected alternatives, the diagnostic results from that first run, how the
implementation was validated, and known weaknesses. **No robustness result has
been produced yet** — the pipeline is built and tested but has not been run to
completion.

---

## What changed from the original notebook

The recorded notebook run reported **0.00% attack success rate**. That number
was a judge failure, not a defence. Specifically:

| # | Problem | Consequence | Fix |
|---|---------|-------------|-----|
| 1 | Judge returned `content=None`; the handler labelled the batch **safe** | 100% judge failure reported as 0% ASR | `LLMClient` raises on empty content; unscored items are `None` and excluded from the denominator, never counted as a defence ([`judge.py`](src/selfplay/judge.py), [`metrics.py`](src/selfplay/metrics.py)) |
| 2 | A content-safety classifier was used to judge behaviour fulfilment | Wrong question asked; empty/mismatched output | Judge is a general instruct model emitting `0`/`1`; classifiers are explicitly warned against ([Judging](#judging)) |
| 3 | JSON parse failure fell back to splitting raw text on newlines | 28 of 44 "attacks" were the attacker's scratchpad (`"Here's a thinking process:"`, `"**Constraints:**"`, a bare `"["`) | `extract_json_array` returns `None` rather than garbage ([`redteam.py`](src/selfplay/redteam.py)) |
| 4 | `ContextString` never read | Contextual behaviours ("Slightly modify **this text**…") had no referent | `Behavior.target_description()` splices context for attacker and judge ([`data.py`](src/selfplay/data.py)) |
| 5 | Copyright behaviours judged by a harm classifier | 5 of 10 sampled behaviours could never register a hit | Category routing; copyright scored by n-gram overlap, or left unscored |
| 6 | ASR compared across rounds using freshly adapted attacks | Defender change and attacker change conflated | Frozen probe set on held-out behaviours ([`build_probe_set.py`](scripts/build_probe_set.py)) |
| 7 | Behaviours drawn from the test split and trained on | No held-out evaluation | Disjoint train/probe split, asserted |
| 8 | One mutated global model across rounds | "D2 trained on rounds 0–1" was really D1 retrained on data it had seen | Explicit `restart_policy: from_base \| incremental` |
| 9 | `labels[:prompt_len] = [-100]*prompt_len` | Slice assignment *extends* the list when `prompt_len > len(labels)`; all-masked rows give NaN loss | Clamped; degenerate examples dropped ([`train.py`](src/selfplay/train.py)) |
| 10 | Trained only on refusals | A model that refuses everything scores 0% ASR | Benign retention data + over-refusal metric |
| 11 | `use_cache=False` pinned during generation | Generation several times slower than necessary | Toggled per phase via `generation_mode()` |
| 12 | Right-truncation of chat prompts | Cut the trailing assistant header off long prompts | `truncation_side="left"` |
| 13 | `apply_chat_template` relied on an implicit `return_dict` | Version-fragile; a missing attention mask breaks left-padded batches | Passed explicitly |

Regression tests in [`tests/`](tests/) pin items 1, 3, 4, 7, 9 and 10.

---

## Layout

```
.
├── configs/default.yaml           # the whole experiment, one file
├── src/selfplay/
│   ├── config.py                  # typed config; secrets come from env
│   ├── data.py                    # HarmBench loading, categories, splits
│   ├── llm_client.py              # OpenRouter client with explicit failures
│   ├── redteam.py                 # attacker + robust JSON extraction
│   ├── defender.py                # 4-bit load, LoRA, batched generation
│   ├── judge.py                   # category-routed scoring
│   ├── safety_data.py             # refusal pairs + retention pairs
│   ├── train.py                   # LoRA SFT, prompt masking, collator
│   ├── metrics.py                 # ASR, per-category ASR, over-refusal
│   └── utils.py                   # logging, seeding, I/O
├── scripts/
│   ├── fetch_harmbench.sh         # download behaviour CSVs
│   ├── build_probe_set.py         # generate + freeze the eval set (run once)
│   ├── run_selfplay.py            # the training loop
│   └── evaluate.py                # reportable numbers
├── tests/                         # pure-python, no GPU or network
└── notebooks/                     # original exploration, outputs stripped
```

---

## Install

```bash
git clone <your-repo-url> && cd jailbreak-selfplay
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # optional; scripts also run without it
./scripts/fetch_harmbench.sh data
```

Needs one CUDA GPU. The defaults (Qwen2.5-1.5B, 4-bit NF4, LoRA r=16) fit in
about 6 GB; a T4 or P100 is enough.

### Credentials

Never committed, read from the environment:

```bash
export OPENROUTER_API_KEY="..."   # attacker + judge
export HF_TOKEN="..."             # only for gated models
```

On Kaggle:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
```

---

## Quickstart

```bash
# 1. Freeze the evaluation set. Run ONCE per experiment.
python scripts/build_probe_set.py --config configs/default.yaml

# 2. Baseline, before any training.
python scripts/evaluate.py --config configs/default.yaml --checkpoints base

# 3. Self-play.
python scripts/run_selfplay.py --config configs/default.yaml

# 4. Reportable numbers for every checkpoint.
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoints base \
        runs/jailbreak_selfplay/checkpoints/D1 \
        runs/jailbreak_selfplay/checkpoints/D2 \
        runs/jailbreak_selfplay/checkpoints/D3 \
        runs/jailbreak_selfplay/checkpoints/D4
```

Step 4 writes `eval_table.md`:

```
| Checkpoint | Probe ASR % | scored/total | Over-refusal % |
| --- | ---: | ---: | ---: |
| base | ... | ... | ... |
| D1   | ... | ... | ... |
```

**Read both columns.** A checkpoint improved only if probe ASR fell *and*
over-refusal did not rise to meet it.

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
| `standard` | LLM judge | Self-contained request |
| `contextual` | LLM judge **plus `ContextString`** | The behaviour is meaningless without it |
| `copyright` | n-gram overlap against a reference text | A harm classifier calls verbatim reproduction "safe" |

**Do not use a content-safety classifier as the judge.** Llama Guard,
`nemotron-*-content-safety` and `gpt-oss-safeguard-*` grade text against their
own harm taxonomy and emit their own schema. The question here is different —
*did this response fulfil this specific target behaviour?* — and forcing them
to answer it is what produced the original notebook's empty responses.

The most faithful option is HarmBench's own classifier,
[`cais/HarmBench-Llama-2-13b-cls`](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls),
run locally with the official prompt templates. The OpenRouter default here is
a convenience; scores from a general instruct judge are not directly comparable
to published HarmBench numbers.

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

Mixing the two — keeping the adapter *and* replaying the cumulative dataset —
trains round 0's data `n` times and is what the original notebook did.

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

Pure Python — no GPU, no network, no model downloads. They cover attack
parsing, judge-label parsing and the unscored-vs-safe distinction, prompt
masking and collation, and the train/probe split.

---

## Known limitations

- **Small scale.** 10 train + 10 probe behaviours from HarmBench's 320 by
  default. Enough for a loop that works; not enough for a claim.
- **Single attacker, single judge.** No cross-model validation, so judge bias
  is unmeasured. A sample of hand-labelled cases is the cheapest check.
- **Model-written refusals** are never better than the generator that wrote
  them, and are only length-filtered.
- **The n-gram copyright check** approximates HarmBench's classifier; it
  catches verbatim reproduction and misses paraphrase.
- **No attack-diversity metric.** The attacker is *asked* for diverse
  strategies; nothing verifies it, and it drifts toward repetition in later
  rounds.
- **Judge and attacker are the same vendor** by default, which is a shared
  failure mode. Cross-vendor is safer.

---

## Citations

### Method

The loop is LM-generated red teaming (Perez et al.) with the attacker
conditioned on prior outcomes, hardening the target via SFT on AI-written
refusals (Bai et al.).

```bibtex
@inproceedings{perez2022redteaming,
  title     = {Red Teaming Language Models with Language Models},
  author    = {Perez, Ethan and Huang, Saffron and Song, Francis and Cai, Trevor
               and Ring, Roman and Aslanides, John and Glaese, Amelia and
               McAleese, Nat and Irving, Geoffrey},
  booktitle = {EMNLP},
  year      = {2022},
  eprint    = {2202.03286},
  archivePrefix = {arXiv}
}

@article{ganguli2022redteaming,
  title   = {Red Teaming Language Models to Reduce Harms: Methods, Scaling
             Behaviors, and Lessons Learned},
  author  = {Ganguli, Deep and Lovitt, Liane and Kernion, Jackson and others},
  journal = {arXiv preprint arXiv:2209.07858},
  year    = {2022}
}

@article{bai2022constitutional,
  title   = {Constitutional AI: Harmlessness from AI Feedback},
  author  = {Bai, Yuntao and Kadavath, Saurav and Kundu, Sandipan and others},
  journal = {arXiv preprint arXiv:2212.08073},
  year    = {2022}
}

@article{chao2023pair,
  title   = {Jailbreaking Black Box Large Language Models in Twenty Queries},
  author  = {Chao, Patrick and Robey, Alexander and Dobriban, Edgar and
             Hassani, Hamed and Pappas, George J. and Wong, Eric},
  journal = {arXiv preprint arXiv:2310.08419},
  year    = {2023}
}

@article{samvelyan2024rainbow,
  title   = {Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts},
  author  = {Samvelyan, Mikayel and Raparthy, Sharath Chandra and Lupu, Andrei and others},
  journal = {arXiv preprint arXiv:2402.16822},
  year    = {2024}
}
```

### Benchmark and evaluation data

```bibtex
@inproceedings{mazeika2024harmbench,
  title     = {HarmBench: A Standardized Evaluation Framework for Automated
               Red Teaming and Robust Refusal},
  author    = {Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and
               Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel
               and Basart, Steven and Li, Bo and Forsyth, David and
               Hendrycks, Dan},
  booktitle = {ICML},
  year      = {2024},
  eprint    = {2402.04249},
  archivePrefix = {arXiv}
}

@inproceedings{rottger2024xstest,
  title     = {XSTest: A Test Suite for Identifying Exaggerated Safety
               Behaviours in Large Language Models},
  author    = {R{\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and
               Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle = {NAACL},
  year      = {2024},
  eprint    = {2308.01263},
  archivePrefix = {arXiv}
}

@misc{taori2023alpaca,
  title  = {Stanford Alpaca: An Instruction-following LLaMA model},
  author = {Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and
            Dubois, Yann and Li, Xuechen and Guestrin, Carlos and
            Liang, Percy and Hashimoto, Tatsunori B.},
  year   = {2023},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}}
}
```

| Resource | Used for | License |
|---|---|---|
| [HarmBench](https://github.com/centerforaisafety/HarmBench) — `harmbench_behaviors_text_test.csv` | Target behaviours | MIT |
| [`cais/HarmBench-Llama-2-13b-cls`](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | Reference judge (optional) | See model card |
| [XSTest](https://github.com/paul-rottger/xstest) — `walledai/XSTest` | Over-refusal evaluation | CC-BY-4.0 |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) — `tatsu-lab/alpaca` | Benign retention prompts | CC-BY-NC-4.0 (non-commercial) |

> Alpaca is CC-BY-NC-4.0. Swap `data.retention_dataset` for a permissively
> licensed set if this is not a research project.

### Models

```bibtex
@article{qwen2024technicalreport,
  title   = {Qwen2.5 Technical Report},
  author  = {{Qwen Team}},
  journal = {arXiv preprint arXiv:2412.15115},
  year    = {2024}
}
```

- Defender: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0)
- Attacker / judge: served via [OpenRouter](https://openrouter.ai); see `configs/default.yaml`
- Llama Guard, referenced as a contrast in [Judging](#judging): Inan et al., arXiv:2312.06674

### Methods and libraries

```bibtex
@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and
               Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and
               Wang, Lu and Chen, Weizhu},
  booktitle = {ICLR},
  year      = {2022},
  eprint    = {2106.09685},
  archivePrefix = {arXiv}
}

@inproceedings{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and
               Zettlemoyer, Luke},
  booktitle = {NeurIPS},
  year      = {2023},
  eprint    = {2305.14314},
  archivePrefix = {arXiv}
}

@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and others},
  booktitle = {EMNLP: System Demonstrations},
  year      = {2020}
}
```

Also used: [PEFT](https://github.com/huggingface/peft),
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes),
[Datasets](https://github.com/huggingface/datasets).

---

## Intended use

Defensive safety research: measuring and improving a model's robustness to
adversarial prompts, on a public benchmark built for that purpose. The
attacker exists to produce training signal for the defender.

Practical notes:

- `runs/` is gitignored. It holds model completions to adversarial prompts —
  review before sharing, and do not commit generated attacks or responses.
- The notebook under `notebooks/` has its outputs stripped for the same reason.
- Checkpoints here are deliberately over-refusing by construction. They are
  experimental artefacts, not general-purpose assistants.
- HarmBench's own terms of use apply to the behaviour data.

## License

MIT — see [LICENSE](LICENSE). Cited datasets, models and benchmarks carry
their own licenses, listed above.
