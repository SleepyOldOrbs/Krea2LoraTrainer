# Krea2 LoRA Trainer Helper

A small local CLI for repeatable Krea 2 LoRA training projects. It wraps an existing `musubi-tuner` checkout and your local model files; it does not download, bundle, or commit models.

The intended workflow is RAW training with Krea 2 RAW, then using the resulting LoRA in ComfyUI with Krea 2 Turbo.

## Requirements

- WSL2 Ubuntu 24.04 or another Linux shell with Python 3.11+
- Existing `musubi-tuner` checkout at `~/src/musubi-tuner`
- Existing musubi virtual environment at `~/.venvs/musubi-krea2`
- Existing captioning virtual environment at `~/.venvs/vl-caption` with `pillow` and `transformers` for VL captions
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

If model files are missing, download the configured Hugging Face files:

```bash
python krea2_lora.py download-models
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

Generate missing captions with a vision-language model from the captioning venv:

```bash
python krea2_lora.py generate-captions jagmoon --trigger "jagmoon style"
```

By default, VL captioning uses `Salesforce/blip-image-captioning-base` and `--local-files-only`, so it only uses a model already present in the Hugging Face cache. Use `--allow-downloads` when you explicitly want Transformers to fetch the caption model into the local HF cache.

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

Or use the guided terminal UI:

```bash
python krea2_lora.py wizard jagmoon --source-dir /mnt/c/Temp/JAG --trigger "jagmoon style"
```

From Windows PowerShell:

```powershell
.\scripts\run-wizard-wsl.ps1 jagmoon --source-dir C:\Temp\JAG --trigger "jagmoon style"
```

## Local web app

Run the localhost dashboard from WSL:

```bash
python3 web_app.py --host 127.0.0.1 --port 8765
```

From Windows PowerShell:

```powershell
.\scripts\run-web-wsl.ps1
```

Then open:

```text
http://127.0.0.1:8765/
```

The web app is a local-only operational dashboard for the same helper commands. It can validate the environment, verify models, initialize projects, import images, generate VL captions, show dataset reports, dry-run cache/train commands, run cache steps, and copy the latest LoRA to ComfyUI. The training action is guarded by the backend and defaults to `--dry-run`.

The Project Control header includes a `Clear Project` button. It clears the form fields and local workflow ticks only; it does not delete project files.

The Run Log includes a summary box above the raw command output with a plain-language result for the last workflow action.

## Commands

- `show-config`: print resolved path, dataset, and training settings
- `download-models`: download or verify the configured Hugging Face model files
- `init-project PROJECT_NAME`: create project folders, `dataset.toml`, `paths.env`, and `train_krea2.sh`
- `validate-env`: check musubi, venvs, required model files, project root, and ComfyUI destination
- `check-dataset PROJECT_NAME`: ensure images exist and every image has a non-empty matching caption
- `import-images PROJECT_NAME SOURCE_DIR`: copy, symlink, or hardlink images into the project
- `dataset-report PROJECT_NAME`: summarize image size/counts, caption state, cache files, and outputs
- `wizard [PROJECT_NAME]`: open a guided terminal menu for setup, import, reports, dry-runs, caching, and copy-to-Comfy
- `create-caption-stubs PROJECT_NAME --trigger "...":` create or fill missing/empty captions
- `generate-captions PROJECT_NAME`: generate missing/empty captions with a vision-language model in the captioning venv
- `cache-latents PROJECT_NAME`: run `krea2_cache_latents.py`
- `cache-text PROJECT_NAME`: run `krea2_cache_text_encoder_outputs.py`
- `train PROJECT_NAME`: run the conservative Krea 2 RAW LoRA training command
- `copy-to-comfy PROJECT_NAME`: copy the latest `.safetensors` from project output to ComfyUI
- `status PROJECT_NAME`: summarize dataset, cache, and output state

The cache and train commands print the exact shell script before execution. Use `--dry-run` to inspect without running anything.

## Safety notes

Do not put model files, datasets, caches, outputs, or LoRA weights inside this repo. The `.gitignore` excludes common model, dataset, cache, output, venv, and log paths, but keep large files in the configured external folders.

This helper does not support cloud training and does not train on Krea 2 Turbo. It uses the RAW model path for training and leaves Turbo use to ComfyUI inference.

`download-models` skips existing non-empty target files unless `--force` is supplied. Keep the configured destinations outside this repo.

VL caption generation writes `.txt` sidecars next to project images. It does not store model files in this repo. Keep `captioning.local_files_only = true` or pass `--local-files-only` when you want to prevent Hugging Face downloads during captioning.

## Musubi reference

Krea 2 support in musubi-tuner is experimental. The generated commands follow the upstream Krea 2 documentation for latent caching, text encoder caching, and RAW LoRA training: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md>.
