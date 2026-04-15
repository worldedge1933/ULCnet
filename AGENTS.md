# AGENTS.md

## Project Summary

This repository is a local reproduction of:

`https://github.com/VamsiKrishna23092005/Ultra-Low-Complexity-Deep-Learning-Based-Noise-Suppression`

The project implements a two-stage speech enhancement pipeline:

- Stage 1: CRN estimates a magnitude mask from noisy STFT magnitude features.
- Stage 2: CNN estimates a complex mask to refine the enhanced complex spectrum.
- Dataset: VoiceBank+DEMAND, prepared into paired clean/noisy 16 kHz WAV files.

The current workspace has already been converted from a tiny placeholder Python project into a runnable reproduction project.

## Current State

Completed:

- Upstream source code was copied into `src/`.
- `pyproject.toml`, `.python-version`, and `uv.lock` were added for reproducible local setup.
- Package marker files were added under `src/dsp`, `src/features`, `src/models`, and `src/utils`.
- `src/utils/audio.py` was added to normalize audio arrays to mono float32.
- `src/train.py` was parameterized with CLI options for GPU/CPU training, batch size, epochs, max-items, max-steps-per-epoch, checkpoint directory, etc.
- `src/inference.py` now creates output directories automatically and handles mono input.
- `src/evaluate.py` now supports `--max-files` for quick evaluation runs.
- `data/download_voicebank.py` was added to download, verify, extract, resample, and organize the official VoiceBank+DEMAND dataset.
- `README.md` was updated with setup, data preparation, smoke test, inference, and evaluation commands.

Validated locally on CPU:

- Full VoiceBank+DEMAND 28-speaker data preparation completed.
- Training split contains 11572 clean/noisy pairs.
- Test split contains 824 clean/noisy pairs.
- Training smoke test completed and produced a checkpoint.
- Inference smoke test produced `outputs/p232_001_enhanced.wav`.
- Evaluation smoke test on 8 files ran successfully and produced SI-SNR output.

Important note:

- The current CPU checkpoint is only a smoke-test checkpoint, not a meaningful trained model.
- On the original machine, CUDA was not available.
- CPU benchmark with `batch-size=8`, 20 steps took about 44.93 seconds.
- Estimated CPU full training time was roughly 18-20 hours for 20 epochs, so proper training should continue on an NVIDIA GPU machine.

## Expected Dataset Layout

The training code expects this layout:

```text
data/
  voicebank/
    train/
      clean/
      noisy/
    test/
      clean/
      noisy/
```

The source archives are expected here if reusing local downloads:

```text
data/
  raw/
    voicebank_demand/
      clean_testset_wav.zip
      noisy_testset_wav.zip
      clean_trainset_28spk_wav.zip
      noisy_trainset_28spk_wav.zip
```

Known valid MD5 values:

```text
clean_testset_wav.zip          34EB1C0BA7EF667E9B966866C542FC16
noisy_testset_wav.zip          FB1B86CAA31E8BA5B506C0C64DA9AAB5
clean_trainset_28spk_wav.zip   D2D5A45EC32F8FCBF201BDE0447E20BA
noisy_trainset_28spk_wav.zip   1FCA9E8BAFB8CD069F6653C6D92F9E9C
```

Official dataset source:

`https://doi.org/10.7488/ds/2117`

## Setup On The GPU Machine

Recommended setup:

```powershell
uv sync --link-mode copy
```

Then confirm PyTorch sees CUDA:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

If `torch.cuda.is_available()` is `False` on the NVIDIA machine, install a CUDA-enabled PyTorch build before training. Use the official PyTorch selector for the machine's CUDA version:

`https://pytorch.org/get-started/locally/`

Do not assume the CPU-only wheel is enough.

## Data Preparation

If `data/voicebank` was copied from the old machine, verify counts:

```powershell
(Get-ChildItem data\voicebank\train\clean -Filter *.wav | Measure-Object).Count
(Get-ChildItem data\voicebank\train\noisy -Filter *.wav | Measure-Object).Count
(Get-ChildItem data\voicebank\test\clean -Filter *.wav | Measure-Object).Count
(Get-ChildItem data\voicebank\test\noisy -Filter *.wav | Measure-Object).Count
```

Expected:

```text
11572
11572
824
824
```

If only the ZIP archives were copied, prepare the dataset:

```powershell
uv run python data/download_voicebank.py
```

The script will use cached archives when MD5 checks pass, then extract, resample to 16 kHz, and write `data/voicebank`.

## Commands Already Verified

Training smoke test:

```powershell
uv run python -m src.train --epochs 1 --batch-size 1 --max-steps-per-epoch 2
```

Inference smoke test:

```powershell
uv run python -m src.inference -i data\voicebank\test\noisy\p232_001.wav -o outputs\p232_001_enhanced.wav -m checkpoints\model_epoch_1.pth
```

Evaluation smoke test:

```powershell
uv run python -m src.evaluate --model checkpoints\model_epoch_1.pth --max-files 8
```

## Recommended Next Steps On GPU

1. Verify CUDA is available.

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

2. Run a short GPU benchmark.

```powershell
Measure-Command { uv run python -m src.train --epochs 1 --batch-size 8 --max-steps-per-epoch 20 --checkpoint-dir checkpoints\gpu_benchmark }
```

3. Start full training.

```powershell
uv run python -m src.train --epochs 20 --batch-size 8 --checkpoint-dir checkpoints
```

4. Run inference on a test file.

```powershell
uv run python -m src.inference -i data\voicebank\test\noisy\p232_001.wav -o outputs\p232_001_enhanced_epoch20.wav -m checkpoints\model_epoch_20.pth
```

5. Run evaluation.

```powershell
uv run python -m src.evaluate --model checkpoints\model_epoch_20.pth
```

For a faster partial evaluation:

```powershell
uv run python -m src.evaluate --model checkpoints\model_epoch_20.pth --max-files 100
```

## Optional Metrics

PESQ and STOI are optional dependencies.

Install them with:

```powershell
uv sync --extra metrics --link-mode copy
```

If they are not installed, evaluation still prints SI-SNR.

## Files And Artifacts

Keep:

- `src/`
- `data/download_voicebank.py`
- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `README.md`
- `AGENTS.md`

Large artifacts:

- `data/` can be copied to avoid re-downloading and reprocessing the dataset.
- `checkpoints/` contains generated training checkpoints.
- `outputs/` contains generated enhanced audio.

Ignored/generated local artifacts:

- `.venv/`
- `*.egg-info/`
- `__pycache__/`
- `outputs/`
- `checkpoints/`
- raw and prepared dataset files under `data/`, except `data/download_voicebank.py`.

## Cautions For The Next Agent

- Do not treat `checkpoints/model_epoch_1.pth` as a useful trained model if it came from the CPU smoke run.
- The current model and training code are a faithful runnable reproduction scaffold, but not guaranteed to match the paper's reported SI-SNR without full training and careful hyperparameter validation.
- The official source audio is 48 kHz; this project prepares 16 kHz WAV because the model training path assumes 16 kHz and 3-second chunks.
- If moving the project by Git, remember that `.gitignore` excludes `data/`, `checkpoints/`, and `outputs/`; copy those directories separately if needed.
