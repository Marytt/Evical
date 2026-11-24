# EVICAL

**EVICAL: Evidence-Sensitive Confidence Estimation for Large Language Models in Medical Question Answering**

When a medical question loses the piece of evidence that decides its answer, a
reliable model should become less confident. EVICAL trains a lightweight
confidence head on the frozen hidden states of a language model so that its
confidence responds to missing evidence (NA: full context, NAP: decisive
evidence removed), while staying well calibrated on full-context questions.

## Highlights

- **No backbone fine-tuning.** The language model is run once to cache hidden
  states; only a small MLP head (input_dim → 256 → 128 → 1) is trained, on CPU or GPU.
- **Evidence-sensitive and calibrated.** Across seven 7–9B backbones, Graded EC
  opens a larger NA→NAP confidence gap than calibration-only training in 7/7
  backbones (mean 0.044 vs. 0.028) with comparable AUROC (0.704 vs. 0.705) and
  ECE (0.050 vs. 0.051).
- **Responds to decision-relevant evidence.** When removing evidence turns a
  correct answer into a wrong one, the Graded EC confidence drop is 0.101,
  versus 0.034 for all other pairs, larger in 7/7 backbones (Wilcoxon p = 0.016).
- **Beyond the model's own probabilities.** Training-free scores (max
  probability, margin, entropy) reach similar AUROC (0.695–0.699 vs. 0.700) but a
  much smaller evidence gap (0.011–0.018 vs. 0.043) and far higher ECE
  (0.27–0.31 vs. 0.05); the head has a larger gap and lower ECE in 7/7 backbones.
- **Transfers without refitting.** Heads trained on MedAbstain keep a positive
  evidence gap on MediQ in 7/7 backbones with no fitting on the target data
  (AUROC 0.712, ECE 0.060).

## Results

Main results on MedAbstain (MedQA), mean over 7 backbones and 3 head seeds:

| Objective | NA→NAP gap | Cohen's d_z | AUROC | ECE | UCER |
|---|--:|--:|--:|--:|--:|
| Brier-only | 0.028 | 0.23 | 0.705 | 0.051 | 0.010 |
| Fixed EC | 0.080 | 0.70 | 0.689 | 0.053 | 0.012 |
| **Graded EC** | **0.044** | 0.38 | 0.704 | **0.050** | **0.009** |

Backbones: Gemma-2-9B-IT, Llama-3.1-8B-Instruct, Llama-3-8B-Instruct,
Ministral-8B-Instruct-2410, Mistral-7B-Instruct-v0.3, Phi-4-Mini-Instruct,
Qwen2.5-7B-Instruct. Per-backbone tables are in `results/`.

## Installation

    git clone https://github.com/Marytt/Evical.git
    cd Evical
    pip install -r requirements.txt          # GPU runs (versions used for the paper)
    pip install -r requirements-cpu.txt      # tests, smoke test and table reproduction

Gated backbones need a Hugging Face token in the environment: `export HF_TOKEN=...`

## Quick check (no GPU, no data)

    python -m pytest -q tests               # loss and metric unit tests
    python scripts/smoke_test.py            # end-to-end pipeline on synthetic features

## Training and evaluation

    python scripts/run_pipeline.py --model Qwen2.5-7B-Instruct --ec_mode graded --seeds 42,43,44

- `--ec_mode graded` (Graded EC) or `--ec_mode fixed` (fixed margin); Brier-only is
  graded with `beta_ec: 0` in `configs/pipeline_config.yaml`.
- The split seed (train/validation split and layer selection) is `pipeline.seed`
  in the config (42 for the paper); head seeds are set with `--seeds`.
- The layer is chosen on the validation split from 0.6/0.7/0.8/0.9 of model depth.

## MediQ transfer

    python scripts/prepare_mediq.py --source data/mediq/all_dev_good.jsonl --out_dir data/mediq
    python scripts/run_transfer.py --model Qwen2.5-7B-Instruct \
        --source_run runs/Qwen2.5-7B-Instruct__graded --target_dataset mediq

## Analyses

| Script | Purpose |
|---|---|
| `scripts/reproduce_results.py` | rebuild the result tables from run records (`--check` compares with `results/`) |
| `scripts/correctness_transition.py` | confidence drop on correct→wrong pairs vs. all other pairs |
| `scripts/run_probability_baselines.py` | max probability, top-1/top-2 margin, normalized negative entropy |
| `scripts/build_length_control.py`, `run_length_control.py`, `analyze_length_control.py` | length-matched random-sentence deletion control |

The probability-baseline comparison uses a separate training run with split
seed 44 and head seed 42; every result file records its split seed, head seed(s),
selected layer, objective and dataset.

## Data

Benchmark text is not redistributed; see `docs/data_access.md`.

## License

Code: Apache-2.0 (`LICENSE`). Models and datasets remain under their own licenses (`THIRD_PARTY.md`).

## Citation

See `CITATION.cff`.
