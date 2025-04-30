# MusicGPT
> CMU 21-366 Course Project

A lightweight **PyTorch** implementation of a GPT‑style Transformer trained to compose symbolic music (MIDI) with the [REMI](https://github.com/GatechVIP/miditok) tokenization.

---

## ✨ Features

- **Single entry‑point** `main.py` orchestrates **training → sampling → evaluation**.
- **Config‑driven** (via [yacs CfgNode](https://github.com/rbgirshick/yacs)) — override any hyper‑parameter from the CLI.
- **Similarity guardrails** — BLEU + edit‑distance matching to catch plagiarism.
- **Weights & Biases** integration out‑of‑the‑box.

---

## 🚀 Quick Start

> **Prerequisites**: Python ≥ 3.9, CUDA‑capable GPU (optional but recommended).

```bash
# 1. Clone & install
$ git clone https://github.com/your‑handle/musicgpt.git
$ cd MusicGPT
$ pip install -r requirements.txt

# 2. Authenticate once with Weights & Biases
$ wandb login

# 3. Point to a folder (recursive OK) of *.mid or *.midi files
$ export DATA_DIR="/path/to/midi_dataset"

# 4. Run the full pipeline
$ python main.py \
    --data="$DATA_DIR" \
    --system.work_dir="./out/run1"
```
The command:
1. **Trains** the GPT.
2. **Generates** scratch pieces and continuations.
3. **Evaluates** them for originality.
4. Logs metrics + artifacts to W&B and `./out/run1`.

---

## ⚙️ Configuration at a Glance

All settings come from `get_config()` in `main.py` and can be overridden via *dot‑notation*:

| Section      | Flag                               | Default | Purpose                                 |
|--------------|--------------------------------------------|---------|-----------------------------------------|
| `system`     | `--system.seed=123`                        | 3407    | RNG seed                                |
|              | `--system.work_dir=./out/exp2`      | `./out` | Output root                             |
| `pipeline`   | `--pipeline.sample=False`                  | `True`  | Enable / disable stage                  |
| `model`      | `--model.n_layer=8`                       | 4      | Transformer depth                       |
|              | `--model.pretrained=/path/ckpt.pt`         | `None`  | Resume / fine‑tune                      |
| `gpt_trainer`| `--gpt_trainer.batch_size=32`              | 16      | Optimizer & schedule                    |
| `sample`     | `--sample.n_scratch=8`                     | 1       | Number of scratch pieces                |
|              | `--sample.seed_toks=256`                   | 512     | Prompt length for continuations         |
| `eval`       | `--eval.bleu_thr=0.75 --eval.edit_thr=0.06`| 0.80/0.05| Similarity thresholds                  |

Run configuration will be printed to terminal before any pipeline stage begins.
---

## 🏋️ Training Only

```bash
python main.py --data=$DATA_DIR \
               --pipeline.sample=False \
               --pipeline.evaluate=False
```
Checkpoints save every **200 iterations** to `<work_dir>/<model_name>.pt`.

Fine‑tune from a checkpoint:
```bash
python main.py --data=$DATA_DIR \
               --model.pretrained=/path/to/ckpt.pt
```

---

## 🎼 Music Generation Examples

```bash
# Scratch generation (no prompt)
python main.py --data=$DATA_DIR \
               --pipeline.train_gpt=False \
               --pipeline.evaluate=False \
               --model.pretrained=/path/to/model.pt \ 
               --sample.n_scratch=4

# Continuation (prompted)
python main.py --data=$DATA_DIR \
               --pipeline.train_gpt=False \
               --pipeline.evaluate=False \
               --model.pretrained=/path/to/model.pt \ 
               --sample.n_seed=3 --sample.seed_toks=256
```
Resulting `.mid` files are written to `<work_dir>`.

---

## 🧪 Evaluation Logic

A generated piece is flagged as *too similar* if:

```
BLEU ≥ eval.bleu_thr  AND  edit‑distance ≤ eval.edit_thr
```
Matching training snippets are dumped to `<work_dir>/eval/` for manual review.

---

## 📦 Dependencies

```text
pytorch >= 2.2
wandb
torch
numpy
einops
tqdm
miditok
```
Install with `pip install -r requirements.txt` or your preferred environment manager.
