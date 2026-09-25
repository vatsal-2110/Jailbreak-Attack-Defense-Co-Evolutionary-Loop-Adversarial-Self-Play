# Code graph

Static map of **jailbreak-selfplay**: package layout, import edges, runtime pipelines, types, and tests.

The experiment is a red-team / defender loop. Scripts orchestrate; `src/selfplay/` holds the library. Reportable numbers never come from in-loop ASR — they come from `scripts/evaluate.py` on a frozen probe set.

---

## 1. Repository layout

```
jailbreak-selfplay/
├── configs/default.yaml          # ExperimentConfig (no secrets)
├── src/selfplay/                 # installable package (pyproject: packages.find where=src)
│   ├── config.py
│   ├── data.py
│   ├── llm_client.py
│   ├── redteam.py
│   ├── defender.py
│   ├── judge.py
│   ├── safety_data.py
│   ├── train.py
│   ├── metrics.py
│   └── utils.py
├── scripts/
│   ├── _bootstrap.py             # prepends src/ onto sys.path
│   ├── fetch_harmbench.sh        # downloads HarmBench CSVs into data/
│   ├── build_probe_set.py
│   ├── run_selfplay.py
│   └── evaluate.py
├── tests/                        # pytest, pythonpath=src, no GPU/network
├── notebooks/                    # exploration (outputs stripped)
└── runs/                         # gitignored experiment artefacts
```

---

## 2. Module import graph

Arrows mean **imports**. Scripts depend on the library; library modules do not import scripts.

```mermaid
flowchart TB
  subgraph scripts["scripts/"]
    BP["_bootstrap.py"]
    FETCH["fetch_harmbench.sh"]
    BUILD["build_probe_set.py"]
    RUN["run_selfplay.py"]
    EVAL["evaluate.py"]
  end

  subgraph pkg["src/selfplay/"]
    CFG["config.py"]
    DATA["data.py"]
    LLM["llm_client.py"]
    RT["redteam.py"]
    DEF["defender.py"]
    JUDGE["judge.py"]
    SAFE["safety_data.py"]
    TRAIN["train.py"]
    MET["metrics.py"]
    UTIL["utils.py"]
  end

  BUILD --> BP
  RUN --> BP
  EVAL --> BP

  BUILD --> CFG
  BUILD --> DATA
  BUILD --> LLM
  BUILD --> RT
  BUILD --> UTIL

  RUN --> CFG
  RUN --> DATA
  RUN --> DEF
  RUN --> JUDGE
  RUN --> LLM
  RUN --> MET
  RUN --> RT
  RUN --> SAFE
  RUN --> TRAIN
  RUN --> UTIL

  EVAL --> CFG
  EVAL --> DATA
  EVAL --> DEF
  EVAL --> JUDGE
  EVAL --> LLM
  EVAL --> MET
  EVAL --> UTIL

  DATA --> UTIL
  LLM --> UTIL
  RT --> DATA
  RT --> LLM
  RT --> UTIL
  DEF --> CFG
  DEF --> UTIL
  JUDGE --> CFG
  JUDGE --> DATA
  JUDGE --> LLM
  JUDGE --> UTIL
  SAFE --> LLM
  SAFE --> MET
  SAFE --> UTIL
  TRAIN --> CFG
  TRAIN --> UTIL
  CFG -.->|"yaml load only"| YAML["configs/default.yaml"]
```

Shared leaf: **`utils.py`** (logging, seed, JSON/JSONL, directories). Config is typed dataclasses plus `load_config` / `require_env`.

---

## 3. Self-play runtime graph

`python scripts/run_selfplay.py --config configs/default.yaml`

```mermaid
flowchart TD
  START["main()"] --> LOADCFG["load_config / require_env OPENROUTER_API_KEY"]
  LOADCFG --> SPLIT["load_harmbench → filter_by_category → split_behaviors"]
  SPLIT --> CLIENT["LLMClient"]
  CLIENT --> GEN["RedTeamGenerator"]
  CLIENT --> BJ["BehaviorJudge"]
  SPLIT --> DEF0["load_defender"]
  DEF0 --> RET["load_retention_prompts → build_retention_dataset"]
  RET --> LORA["attach_lora"]

  LORA --> LOOP{"for round_idx in 0..num_rounds"}
  LOOP --> HIST["build_history(all_results)"]
  HIST --> ATT["generator.generate_round"]
  ATT --> DEFEND["run_attacks"]
  DEFEND --> SCORE["judge.score_all"]
  SCORE --> SUM["summarize_round"]
  SUM --> LAST{"round_idx == num_rounds?"}
  LAST -->|yes| DONE["write all_results.json"]
  LAST -->|no| SUCC["keep items with success == 1"]
  SUCC --> REF["build_refusal_dataset"]
  REF --> POL{"train.restart_policy"}
  POL -->|from_base| RESET["_reset_defender + cumulative refusals"]
  POL -->|incremental| KEEP["keep adapter + new refusals only"]
  RESET --> MIX["mix_datasets refusals + retention"]
  KEEP --> MIX
  MIX --> SFT["train_round → checkpoints/D{n}"]
  SFT --> LOOP
```

In-loop ASR is a **progress signal** on adaptive attacks against **train** behaviours. It is not the reported robustness number.

---

## 4. Probe freeze and evaluation graph

Two scripts. Probe is generated once; every checkpoint is scored on that same file.

```mermaid
flowchart LR
  subgraph freeze["build_probe_set.py"]
    HB["HarmBench CSV"] --> FILT["filter + split"]
    FILT --> PROBEB["probe behaviours only"]
    PROBEB --> RT["RedTeamGenerator.generate_round\nround_idx=-1, history=None"]
    RT --> PA["probe_attacks.json"]
    PA --> MAN["probe_manifest.json sha256"]
  end

  subgraph eval["evaluate.py"]
    PA2["probe_attacks.json"] --> CK["for each checkpoint"]
    CK --> LD["load_defender ± load_adapter"]
    LD --> RA["run_attacks"]
    RA --> JS["BehaviorJudge.score_all"]
    JS --> ASR["attack_success_rate + per_category_asr"]
    LD --> XS["load_overrefusal_prompts XSTest"]
    XS --> GR["generate_responses"]
    GR --> OR["overrefusal_rate"]
    ASR --> TAB["eval_table.md"]
    OR --> TAB
  end

  MAN -.->|"do not regenerate mid-experiment"| PA2
```

A checkpoint improved only if **probe ASR fell** and **over-refusal did not rise** to match it.

---

## 5. Type and class graph

```mermaid
classDiagram
  class ExperimentConfig {
    +name
    +output_root
    +num_rounds
    +seed
    +openrouter_base_url
    +DefenderConfig defender
    +LoraConfigSpec lora
    +RedTeamConfig redteam
    +JudgeConfig judge
    +TrainConfig train
    +DataConfig data
    +run_dir
    +checkpoint_dir
    +rounds_dir
    +probe_dir
  }

  class LLMClient {
    +complete()
  }
  class EmptyCompletionError

  class Behavior {
    +behavior_id
    +behavior
    +functional_category
    +context_string
    +target_description()
  }

  class AttackRecord {
    +to_dict()
  }

  class RedTeamGenerator {
    +generate_for_behavior()
    +generate_round()
  }

  class BehaviorJudge {
    +score_all()
  }
  class LLMJudge {
    +score()
  }
  class HarmBenchClsJudge {
    +score()
  }
  class CopyrightJudge {
    +score()
  }
  class JudgeOutcome {
    +label
    +detail
  }

  class SafetyCollator
  class ASRSummary {
    +to_dict()
  }

  ExperimentConfig --> DefenderConfig
  ExperimentConfig --> LoraConfigSpec
  ExperimentConfig --> RedTeamConfig
  ExperimentConfig --> JudgeConfig
  ExperimentConfig --> TrainConfig
  ExperimentConfig --> DataConfig

  RedTeamGenerator --> LLMClient
  RedTeamGenerator --> Behavior
  RedTeamGenerator --> AttackRecord

  BehaviorJudge --> LLMJudge
  BehaviorJudge --> HarmBenchClsJudge
  BehaviorJudge --> CopyrightJudge
  LLMJudge --> LLMClient
  LLMJudge --> JudgeOutcome
  HarmBenchClsJudge --> JudgeOutcome
  CopyrightJudge --> JudgeOutcome

  EmptyCompletionError <|-- RuntimeError
```

**Judge routing** (`BehaviorJudge.score_all`):

| `functional_category` | Scorer | Outcome if it cannot score |
|---|---|---|
| `standard`, `contextual` | `LLMJudge` (OpenRouter `0`/`1`) **or** `HarmBenchClsJudge` (local HF Yes/No) depending on `judge.model_id` | `success=None` (excluded from ASR) |
| `copyright` | `CopyrightJudge` (n-gram overlap vs `<behavior_id>.txt`) | `None` without a reference file |

Set `judge.model_id` to `cais/HarmBench-Mistral-7b-val-cls` (or `cais/HarmBench-Llama-2-13b-cls`) to load a local classifier instead of the API judge.

Constants in `judge.py`: `SAFE=0`, `UNSAFE=1`, `UNSCORED=None`.

---

## 6. Data and artefact flow

```mermaid
flowchart TB
  ENV["OPENROUTER_API_KEY / HF_TOKEN"] --> SCRIPTS["scripts"]
  YAML["configs/default.yaml"] --> CFG["load_config"]
  CSV["data/harmbench_behaviors_text_test.csv"] --> DATA["data.load_harmbench"]
  ALPACA["tatsu-lab/alpaca"] --> RETP["load_retention_prompts"]
  XSTEST["walledai/XSTest"] --> OVER["load_overrefusal_prompts"]
  HF["Hugging Face: Qwen2.5-1.5B-Instruct"] --> DEF["defender.load_defender"]
  OR["OpenRouter"] --> LLM["LLMClient"]

  DATA --> SPLIT["behavior_split.json"]
  LLM --> ATTACKS["rounds/round_N_attacks.json"]
  DEF --> RESULTS["rounds/round_N_results.json"]
  RESULTS --> SUM["rounds/round_N_summary.json"]
  RESULTS --> CK["checkpoints/Dn + train_examples.json"]
  RETP --> RETJSON["retention_examples.json"]

  LLM --> PROBE["probe/probe_attacks.json"]
  PROBE --> EVALOUT["eval reports + eval_table.md"]
```

`runs/` is gitignored: attacks, completions, and checkpoints.

---

## 7. Function-level call graph (library)

```mermaid
flowchart LR
  subgraph data_mod["data.py"]
    load_harmbench --> Behavior
    filter_by_category
    split_behaviors
    load_retention_prompts
    load_overrefusal_prompts
  end

  subgraph redteam_mod["redteam.py"]
    generate_round --> generate_for_behavior
    generate_for_behavior --> extract_json_array
    generate_round --> AttackRecord
    build_history
  end

  subgraph defender_mod["defender.py"]
    load_defender
    attach_lora
    load_adapter
    run_attacks --> generate_responses
    generate_responses --> generation_mode
  end

  subgraph train_mod["train.py"]
    train_round --> tokenize_dataset
    tokenize_dataset --> tokenize_example
    train_round --> SafetyCollator
  end

  subgraph safety_mod["safety_data.py"]
    build_refusal_dataset --> generate_refusal
    generate_refusal --> looks_like_refusal
    build_retention_dataset
    mix_datasets
  end

  subgraph metrics_mod["metrics.py"]
    summarize_round --> attack_success_rate
    attack_success_rate --> ASRSummary
    per_category_asr
    overrefusal_rate --> looks_like_refusal
  end
```

---

## 8. Tests → production

| Test file | Production symbols |
|---|---|
| `tests/test_data.py` | `Behavior`, `filter_by_category`, `split_behaviors` |
| `tests/test_redteam_parsing.py` | `extract_json_array`, `build_history` |
| `tests/test_judge.py` | `parse_single_label`, `CopyrightJudge`, `HarmBenchClsJudge` selection, HarmBench Yes/No parsing, `SAFE`/`UNSAFE`, ASR exclusion of unscored items |
| `tests/test_metrics.py` | `attack_success_rate`, `per_category_asr`, `looks_like_refusal`, `overrefusal_rate` |
| `tests/test_train_masking.py` | `tokenize_example`, `SafetyCollator`, `IGNORE_INDEX` |

Tests are pure Python: no GPU, no network. `pyproject.toml` sets `pythonpath = ["src"]`.

---

## 9. External systems

```mermaid
flowchart LR
  subgraph local["This repo"]
    PKG["src/selfplay"]
    SCR["scripts"]
  end

  SCR --> PKG
  PKG --> OR["OpenRouter chat completions\nattacker + optional API judge"]
  PKG --> TRANSFORMERS["transformers + bitsandbytes + PEFT\nQwen defender, 4-bit + LoRA\noptional local HarmBench classifier"]
  PKG --> HFDS["Hugging Face datasets\nAlpaca, XSTest"]
  PKG --> HB["HarmBench CSV on disk"]
```

Default IDs live in `configs/default.yaml`: defender `Qwen/Qwen2.5-1.5B-Instruct`, attacker `nvidia/nemotron-3.5-lightning:free`, judge `meta-llama/llama-3.3-70b-instruct`.
