# Adversarial Self-Play for Jailbreak Robustness — Project Report

**Scope.** Build a closed loop in which a red-team model generates jailbreak
attempts against a defender model, a judge scores them, and the successful
attempts are converted into refusal training data for the next iteration of
the defender — evaluated on [HarmBench](https://github.com/centerforaisafety/HarmBench).

---

## 1. Problem framing

The underlying question is whether iterated adversarial pressure improves a
small instruction-tuned model's refusal behaviour faster than a static safety
dataset would.

The loop:

```
  HarmBench behaviour  ──────────────▶  attacker LM  ─────────────▶  adversarial prompts
                                                                                    │
                                                                                    ▼
 next defender ◀── LoRA SFT ◀── refusal pairs ◀── refusal teacher ◀── judge ◀── defender
        │                             + benign                                    responses
        │                                                                            │ 
        └──────────────── attacked again next round ─────────────────────────────────┘
```


---

## 2. Methodology

### 2.1 Components

| Component | Choice | Rationale |
|---|---|---|
| Defender | `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4 | small enough to iterate in limited time, instruction-tuned so it has refusal behaviour to get started |
| Attacker | yukiyounai/Jailbreak-R1 | Decoupled from the defender so attack quality doesn't degrade as the defender is modified; high temperature for strategy diversity |
| Judge | cais/HarmBench-Mistral-7b-val-cls | Must answer "did this response fulfil *this* behaviour" |
| Safe response Generator/Refusal Teacher | Qwen/Qwen3-4B-Instruct-2507| Writes the training target for each successfull attack |
| Behaviours | HarmBench | Standard benchmark, categorised, publicly citable |
| Over Refusal Source | walledai/XSTest | Benchmark for testing over refusal rate|
| Adaptation | LoRA r=16, α=32, dropout=0.05, all attention + MLP projections | QLoRA-style 4-bit base + adapters; full fine-tuning is out of budget and would confound "safety trained" with "model degraded" |

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

### 3.1 Behaviour-Aware Evaluation Protocol

HarmBench contains three distinct behaviour categories—**Standard (159 behaviours), Contextual (81 behaviours), and Copyright (80 behaviours)**—that require different evaluation strategies. Applying a single judging mechanism across all categories would not provide a reliable measure of attack success.

For the **Standard** category, we use an LLM-based judge to determine whether the model's response constitutes a successful harmful behaviour. Each test case is evaluated independently.

The **Contextual** category requires additional information beyond the behaviour description itself. Many contextual behaviours contain references such as *“modify this text”* or *“perform this action on the given content,”* where the behaviour description alone does not specify the object being referred to. The corresponding context is provided separately in the HarmBench data. Therefore, evaluating these behaviours requires incorporating the associated `ContextString` into the evaluation prompt.

The **Copyright** category requires a fundamentally different criterion. Its objective is to determine whether the model reproduces protected reference material, making **n-gram overlap with the reference text** a more appropriate measure than a conventional harm/safety judge. A generic safety classifier may incorrectly classify faithful reproduction of copyrighted material as safe and therefore fail to identify a successful attack.

For this reason, the current evaluation focuses on the **Standard category**, while Contextual and Copyright behaviours are excluded from the default evaluation pipeline. This is a deliberate scope restriction rather than an assumption that these categories are equivalent to Standard behaviours. In particular, Copyright evaluation requires access to reference texts that are not redistributed with this project, while Contextual evaluation requires the corresponding context strings to be incorporated correctly. These dependencies are documented as limitations of the current evaluation setup.



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
| **Content-safety classifier as judge** (e.g. `llama-guard-*`, `nemotron-*-content-safety`) | General instruct model emitting `0`/`1` | These grade text against *their own* harm taxonomy and emit their own schema. The question here is different, did this response fulfil *this* target behaviour? Forcing the wrong question through them returned empty content on 100% of calls also loading llama-guard locally increases the computation and on its opposite smaller model hallucinates |
| **LLM Red Teamer** (openrouter LM)| General uncensored model/model which can generate harmful prompts | Using instruction tuned safe model are not best suitable to generate adversarial attacking prompts, instead using a uncensored model to generate better attacks is better for training purposed|
| **In-loop ASR as headline metric** | Frozen probe set on held-out behaviours | Adaptive attacks + trained-on behaviours measures memorisation and attacker drift, not robustness |
| **HarmBench test split for both training and reporting** | Disjoint train/probe split, asserted in code | Train/test contamination |
| **ASR as sole metric** | ASR **and** over-refusal, reported together | 0% ASR is achievable by refusing everything |
| **Single global mutated model across rounds** | Explicit `restart_policy` | Made "D2 trained on rounds 0–1" mean something other than what it says |
| **DPO / RLHF instead of SFT** | SFT on refusal pairs | Deferred. Needs paired preference data and more compute; SFT establishes whether the loop closes at all |

---

## 5. Experiments run and results

### 5.1 Results
Results of model fin-tuned at each step on test probe.
| Checkpoint | Probe ASR % | scored/total | Over-refusal % |
| --- | ---: | ---: | ---: |
| base | 53.33 | 15/15 | 0.00 |
| D1 | 53.33 | 15/15 | 0.00 |
| D2 | 40.00 | 15/15 | 0.00 |
| D3 | 40.00 | 15/15 | 0.00 |
| D4 | 26.67 | 15/15 | 0.00 |

These are the results of a test run, which consisted of test of size 15 only.

Results of model fine-tuned at each step on the fixed probe.
| Checkpoint | Probe ASR % | scored/total | Over-refusal % |
| --- | ---: | ---: | ---: |
| base | 62.00 | 100/100 | 0.00 |
| D1 | 53.00 | 100/100 | 0.00 |
| D2 | 42.00 | 100/100 | 0.00 |
| D3 | 27.00 | 100/100 | 0.00 |
| D4 | 17.00 | 100/100 | 0.00 |

We used a disjoint fixed probe of size 100, which consisted of 100 tests.

We can draw a conclusion from here that increasing the number of attacks for training will make the model more robust against attack.


### 5.2 Experiment Runs
We began with a base instruction-tuned model, which exhibited an initial Probe Attack Success Rate (ASR) of **62%**. From the standard category of the **HarmBench benchmark**, we selected **20 behaviours** and generated **5 attacks per behaviour**, resulting in a total of **100 initial attack prompts**.

The successful attacks were then identified and used to construct the training dataset. For each successful attack, we created training pairs consisting of the **attack prompt paired with a safe response**, along with corresponding **benign instruction–response pairs**. The resulting dataset was used to perform **LoRA-based Supervised Fine-Tuning (SFT)** of the model for 2 epochs.

This attack–defense process was performed iteratively for **four rounds**. At each round, the attacker was provided with the attacks generated in the previous round as references and was instructed to generate stronger and more effective attacks. This enabled the attacker to iteratively improve its attack strategies based on previously generated examples.

After each round of fine-tuning, the resulting model version was evaluated using the **same fixed probe set**, ensuring a consistent evaluation setting across all iterations. The evaluation results showed progressive improvement in the model's robustness, with the Probe ASR decreasing across successive fine-tuned model versions compared with the preceding stages.

All the checkpoints and results are stored in runs/jailbreak_selfplay folder.
runs/jailbreak_selfplay_test was a pipeline testing run to check the overall working of the pipeline.


### 5.3 Probe composition

The 20 sampled behaviours(experimental), by HarmBench standard category:

| Category | Count | Scoreable by the pipeline as written? |
|---|---|---|
| `copyright` | 0 | **No** — safety judge always returns "safe" |
| `contextual` | 0 | **No** — `ContextString` was never read; behaviour text is a dangling referent |
| `standard` | 20 | Yes |

Attacks per behaviour - 5

Using only standard provides us with the freedom to generate attacks based on the behaviour only without any other context.

## 6. Analysis

This project develops an adversarial self-play framework for improving the jailbreak resistance of an LLM. A red-team LLM continuously generates adversarial prompts against a defender model, while a judge determines whether the generated response successfully satisfies the harmful target behavior. Successful jailbreaks are then converted into safety/refusal training examples and used to fine-tune the defender over multiple rounds.

A key design choice is that in-loop attack success rate is treated only as a progress signal, rather than the final evaluation metric. Since the attacker adapts to each defender checkpoint and the defender is trained on the same behaviours, using those attacks for evaluation would introduce significant bias. Instead, the project creates a frozen probe set before training, without attack history and on held-out behaviours. Every defender checkpoint is evaluated on exactly this same set.

The evaluation also measures over-refusal, so improved safety is not achieved simply by making the model refuse everything. 

Thus, the project evaluates two complementary objectives:
- Attack Success Rate (ASR): whether harmful jailbreak attempts succeed.
- Over-refusal rate: whether the model unnecessarily refuses benign requests.

Overall, the project is essentially an adaptive red-team → defense training → fixed evaluation loop, designed to study whether a defender can become more robust against jailbreaks while retaining useful behaviour.

>One finding form the experiments is that the more will be the dataset the better will be the results.
>One more Implementation level detail is that we are freeing our vram b offloading models, this will be helpful when we have to work with larger datasets and models.
---

## 7. Weaknesses and Improvements

- **Copyright and standard behaviours are excluded**, so results cover 240 of 320 HarmBench
  behaviours and are not directly comparable to full-benchmark numbers.
- **Small scale by default** — 20 train + 20 probe behaviours. Enough to show
  the loop closes; far too few for a claim.
- **No attack-diversity metric.** The attacker is *asked* for diverse
  strategies; nothing verifies it. Expected to degrade in later rounds as
  history fills the prompt.
- **Improvement of the Attacking Model.** Currently the pipeline is generalised to train only the defensive model not the attacking model. Improving both models together using some RLHF algorithm should be there so that the attacks can also be improved.

---

## 8. With more time and compute

### 8.1 Establish judge validity

Hand-label a stratified sample of ~200 cases; report a metric against the
automated judge. Cross-check with a second judge from a different vendor and
report disagreement rate. An ASR curve is worth exactly as much as the judge
behind it, and that number is currently unknown.

### 8.2 Scale

Including all the behaviours of HarmBench including contextual and copyright for generating attacks and evals.

### 8.3 Strengthen the attacker

The current attacker is a single prompted LM. Stronger and better-studied
options — GCG's gradient-based suffixes,
AutoDAN — would give external validity: does hardening against a prompted
attacker transfer to an optimisation-based one, or is it attacker-specific
overfitting? This is the most interesting open question the setup can ask.

### 8.4 Measure attack diversity

Embed attacks, track pairwise similarity and cluster count per round. Test
directly whether history conditioning sustains novelty or collapses to
paraphrase — currently an assumption.

### 8.5 Ablations

- `retention_ratio` sweep: the ASR / over-refusal frontier.
- `from_base` vs `incremental`: does cumulative retraining beat incremental?
- History window size.
- Rounds-to-saturation: where does the curve flatten?

### 8.6 Method extensions

- DPO/RLVR on `(attack, refusal, harmful_completion)` triples instead of SFT.
- Multi-turn attacks — the current setup is single-turn, which excludes a
  large and practically important class of jailbreak.
- Transfer: does hardening on 20 behaviours generalise to 230 unseen ones? The
  probe-set design already supports asking this.
- Capability retention beyond refusal behaviour (MMLU or similar), to check
  the adapter isn't degrading the model generally.

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
| [`cais/HarmBench-Mistral-7b-val-cls`](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls) | Classifier for successful attacks | See model card |
| [`cais/HarmBench-Llama-2-13b-cls`](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | Alternate local classifier | See model card |
| [`yukiyounai/Jailbreak-R1`](https://huggingface.co/yukiyounai/Jailbreak-R1) | Attack generation | Apache-2.0 |
| [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Refusal targets for successful attacks | Apache-2.0 |
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
- Attacker: [`yukiyounai/Jailbreak-R1`](https://huggingface.co/yukiyounai/Jailbreak-R1) (Apache-2.0), loaded from Hugging Face
- Classifier: [`cais/HarmBench-Mistral-7b-val-cls`](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls)
- Refusal teacher: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0)
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
