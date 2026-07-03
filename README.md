# Krea2 LoRA Trainer Helper

A small local CLI for repeatable Krea 2 LoRA training projects. It wraps an existing `musubi-tuner` checkout and your local model files; it does not download, bundle, or commit models.

The intended workflow is RAW training with Krea 2 RAW, then using the resulting LoRA in ComfyUI with Krea 2 Turbo.

## Requirements

- WSL2 Ubuntu 24.04 or another Linux shell with Python 3.11+
- Existing `musubi-tuner` checkout at `~/src/musubi-tuner`
- Existing musubi virtual environment at `~/.venvs/musubi-krea2`
- Existing captioning virtual environment at `~/.venvs/vl-caption`
- Existing model files:
  - `~/ai_models/krea2/raw.safetensors`
  - `~/ai_models/qwen/split_files/vae/qwen_image_vae.safetensors`
  - `~/ai_models/qwen/text_encoders/qwen3vl_4b_bf16.safetensors`
- ComfyUI LoRA destination, defaulting to `/mnt/c/ComfyUI_windows_portable/ComfyUI/models/loras/krea2-jim`

## First-time setup

From this repo, optionally copy the editable config:

```bash
cp config.example.toml config.toml
```

Then validate the local paths:

```bash
python krea2_lora.py validate-env
```

From Windows PowerShell, use the WSL wrapper so `~/...` resolves inside Ubuntu:

```powershell
.\scripts\run-in-wsl.ps1 validate-env
```

If the ComfyUI LoRA destination is missing and its parent is writable:

```bash
python krea2_lora.py validate-env --create-comfy-dir
```

## Typical workflow

Create a project:

```bash
python krea2_lora.py init-project jagmoon
```

This creates:

```text
~/krea2_loras/jagmoon/
  images/
  cache/
  output/
  config/
    dataset.toml
    paths.env
    cache_latents.sh
    cache_text.sh
    train_krea2.sh
    copy_latest_to_comfy.sh
```

Place training images in `~/krea2_loras/jagmoon/images/`. Each image needs a matching `.txt` caption with the same basename.

Or import/link a prepared image folder:

```bash
python krea2_lora.py import-images jagmoon /mnt/c/Temp/JAG --mode symlink --trigger "jagmoon style"
python krea2_lora.py dataset-report jagmoon
```

Create caption stubs:

```bash
python krea2_lora.py create-caption-stubs jagmoon --trigger "jagmoon style"
```

Check the dataset:

```bash
python krea2_lora.py check-dataset jagmoon
```

Inspect the exact musubi commands without running them:

```bash
python krea2_lora.py cache-latents jagmoon --dry-run
python krea2_lora.py cache-text jagmoon --dry-run
python krea2_lora.py train jagmoon --dry-run
```

Run the workflow when ready:

```bash
python krea2_lora.py cache-latents jagmoon
python krea2_lora.py cache-text jagmoon
python krea2_lora.py train jagmoon
python krea2_lora.py copy-to-comfy jagmoon
```

Check project state:

```bash
python krea2_lora.py status jagmoon
```

## Commands

- `show-config`: print resolved path, dataset, and training settings
- `init-project PROJECT_NAME`: create project folders, `dataset.toml`, `paths.env`, and `train_krea2.sh`
- `validate-env`: check musubi, venvs, required model files, project root, and ComfyUI destination
- `check-dataset PROJECT_NAME`: ensure images exist and every image has a non-empty matching caption
- `import-images PROJECT_NAME SOURCE_DIR`: copy, symlink, or hardlink images into the project
- `dataset-report PROJECT_NAME`: summarize image size/counts, caption state, cache files, and outputs
- `create-caption-stubs PROJECT_NAME --trigger "...":` create or fill missing/empty captions
- `cache-latents PROJECT_NAME`: run `krea2_cache_latents.py`
- `cache-text PROJECT_NAME`: run `krea2_cache_text_encoder_outputs.py`
- `train PROJECT_NAME`: run the conservative Krea 2 RAW LoRA training command
- `copy-to-comfy PROJECT_NAME`: copy the latest `.safetensors` from project output to ComfyUI
- `status PROJECT_NAME`: summarize dataset, cache, and output state

The cache and train commands print the exact shell script before execution. Use `--dry-run` to inspect without running anything.

## Safety notes

Do not put model files, datasets, caches, outputs, or LoRA weights inside this repo. The `.gitignore` excludes common model, dataset, cache, output, venv, and log paths, but keep large files in the configured external folders.

This helper does not support cloud training and does not train on Krea 2 Turbo. It uses the RAW model path for training and leaves Turbo use to ComfyUI inference.

## Musubi reference

Krea 2 support in musubi-tuner is experimental. The generated commands follow the upstream Krea 2 documentation for latent caching, text encoder caching, and RAW LoRA training: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md>.
