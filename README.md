# Krea2 LoRA Trainer Helper

A small local CLI for repeatable Krea 2 LoRA training projects. It wraps an existing `musubi-tuner` checkout and your local model files. When requested, it can download models to configured external locations or the Hugging Face cache; it does not bundle or commit models.

The intended workflow is RAW training with Krea 2 RAW, then using the resulting LoRA in ComfyUI with Krea 2 Turbo.

## Requirements

- WSL2 Ubuntu 24.04 or another Linux shell with Python 3.11+
- Existing `musubi-tuner` checkout at `~/src/musubi-tuner`
- Existing musubi virtual environment at `~/.venvs/musubi-krea2`
- For the default high-quality caption backend, a Qwen-VL capable llama.cpp/KoboldCPP binary on `PATH` as `llama-qwen2vl-cli`, or set `captioning.llama_cli` in `config.toml`
- Optional Transformers captioning virtual environment at `~/.venvs/vl-caption` with `pillow` and `transformers` if using `captioning.backend = "transformers"`
- Existing model files:
  - `~/ai_models/krea2/raw.safetensors`
  - `~/ai_models/qwen/split_files/vae/qwen_image_vae.safetensors`
  - `~/ai_models/qwen/text_encoders/qwen3vl_4b_bf16.safetensors`
- Default caption model files, downloaded with `python krea2_lora.py download-models --model vl_caption`:
  - `~/ai_models/vl-caption/qwen2.5-vl-7b-captioner-relaxed-gguf/Qwen2.5-VL-7B-Captioner-Relaxed.Q6_K.gguf`
  - `~/ai_models/vl-caption/qwen2.5-vl-7b-captioner-relaxed-gguf/Qwen2.5-VL-7B-Captioner-Relaxed.mmproj-f16.gguf`
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

Run the test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

If model files are missing, download the configured Hugging Face files and the VL caption model:

```bash
python krea2_lora.py download-models
```

You can also target one model:

```bash
python krea2_lora.py download-models --model krea_raw
python krea2_lora.py download-models --model qwen_vae
python krea2_lora.py download-models --model qwen_text_encoder
python krea2_lora.py download-models --model vl_caption
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

Recursive imports preserve subfolders under the project `images/` directory, so duplicate filenames from different source folders do not overwrite each other.

Create caption stubs:

```bash
python krea2_lora.py create-caption-stubs jagmoon --trigger "jagmoon style"
```

Generate missing captions with the configured vision-language backend:

```bash
python krea2_lora.py generate-captions jagmoon --trigger "jagmoon style"
```

By default, captioning uses `qwen_gguf` with `mradermacher/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF`, the `Q6_K` language-model quant, the f16 vision projector, and `gpu_layers = 99` for llama.cpp builds that support GPU offload. This is the quality-oriented 16 GB VRAM preset: materially better captions than BLIP, without running the full f16 15.2 GB model.

To test another quant without editing `config.toml`, pass explicit filenames from the same GGUF repo:

```bash
python krea2_lora.py generate-captions jagmoon --model-file Qwen2.5-VL-7B-Captioner-Relaxed.Q6_K.gguf --mmproj-file Qwen2.5-VL-7B-Captioner-Relaxed.mmproj-f16.gguf
```

Use `--force` when replacing sparse or hand-written caption files:

```bash
python krea2_lora.py generate-captions jagmoon --trigger "jagmoon style" --force
```

Use `--allow-downloads` only when you want the helper to download missing configured caption files before captioning. Keep `captioning.local_files_only = true` or pass `--local-files-only` while curating datasets to avoid surprise downloads.

For large caption runs, start a persistent llama.cpp server once and pass its URL so the model is not reloaded for every image:

```bash
python krea2_lora.py generate-captions jagmoon --trigger "jagmoon style" --server-url http://172.22.112.1:8100
```

The legacy Transformers path is still available:

```bash
python krea2_lora.py generate-captions jagmoon --backend transformers --model Salesforce/blip-image-captioning-base
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

Plan multiple LoRA training variants by giving each run a safe output suffix and optional training overrides:

```bash
python krea2_lora.py train jagmoon --dry-run --run-name baseline-16
python krea2_lora.py train jagmoon --dry-run --run-name detail-32 --network-dim 32 --network-alpha 16
python krea2_lora.py train jagmoon --dry-run --run-name fast-8 --network-dim 8 --network-alpha 8 --max-train-epochs 8
```

Each variant writes a distinct musubi `--output_name`, so multiple LoRA outputs can live in the same project output folder.

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

The web app is a local-only operational dashboard for the same helper commands. It can validate the environment, download missing models, initialize projects, import images, generate VL captions, show dataset reports, dry-run cache/train commands, plan multiple training variants, run cache steps, and copy the latest or selected LoRA to ComfyUI. The caption controls expose the configured backend, model repo, GGUF model file, vision projector file, runtime executable, optional persistent server URL, token limit, GPU-layer offload, and Qwen prompt; the training action is guarded by the backend and defaults to `--dry-run`.

The server refuses non-loopback bind hosts by default. Keep `--host` on `127.0.0.1` or `localhost`; `--unsafe-host` is only for deliberate LAN exposure.

The Project Control header includes a `Clear Project` button. It clears the form fields and local workflow ticks only; it does not delete project files.

The Run Log includes a summary box above the raw command output with a plain-language result for the last workflow action.

The Training Variants panel creates dry-run commands for several LoRA run configurations. Use `Add Variant` for another planned run, adjust dim/alpha/epochs/LR/seed, then dry-run it to inspect the exact command. Real training still requires explicit backend confirmation; the UI keeps this conservative by default.

Dataset and project image lists use Windows-style natural ordering, so `image2.png` sorts before `image10.png`. LoRA outputs are listed newest-first, and each output row can be copied to the configured ComfyUI destination. Selected output copies are restricted to `.safetensors` files inside the current project's `output/` folder.

## Commands

- `show-config`: print resolved path, dataset, and training settings
- `download-models`: download or verify the configured Hugging Face model files and VL caption model
- `init-project PROJECT_NAME`: create project folders, `dataset.toml`, `paths.env`, and `train_krea2.sh`
- `validate-env`: check musubi, venvs, required model files, project root, and ComfyUI destination
- `check-dataset PROJECT_NAME`: ensure images exist and every image has a non-empty matching caption
- `import-images PROJECT_NAME SOURCE_DIR`: copy, symlink, or hardlink images into the project
- `dataset-report PROJECT_NAME`: summarize image size/counts, caption state, cache files, and outputs
- `wizard [PROJECT_NAME]`: open a guided terminal menu for setup, import, reports, dry-runs, caching, and copy-to-Comfy
- `create-caption-stubs PROJECT_NAME --trigger "...":` create or fill missing/empty captions
- `generate-captions PROJECT_NAME`: generate missing/empty captions with the configured vision-language backend; use `--force` to replace existing captions
- `cache-latents PROJECT_NAME`: run `krea2_cache_latents.py`
- `cache-text PROJECT_NAME`: run `krea2_cache_text_encoder_outputs.py`
- `train PROJECT_NAME`: run the conservative Krea 2 RAW LoRA training command; accepts safe run/output names and common per-run overrides
- `copy-to-comfy PROJECT_NAME`: copy the latest `.safetensors` from project output to ComfyUI
- `status PROJECT_NAME`: summarize dataset, cache, and output state

The cache and train commands print the exact shell script before execution. Use `--dry-run` to inspect without running anything.

## Safety notes

Do not put model files, datasets, caches, outputs, or LoRA weights inside this repo. The `.gitignore` excludes common model, dataset, cache, output, venv, and log paths, but keep large files in the configured external folders.

This helper does not support cloud training and does not train on Krea 2 Turbo. It uses the RAW model path for training and leaves Turbo use to ComfyUI inference.

`download-models` skips existing non-empty Krea/Qwen target files unless `--force` is supplied. With the default `qwen_gguf` caption backend, the VL caption language model and vision projector are downloaded into `paths.caption_models_dir`. With `captioning.backend = "transformers"`, the VL caption model is downloaded to the normal Hugging Face cache through the configured captioning venv. Keep all configured destinations outside this repo.

VL caption generation writes `.txt` sidecars next to project images. It does not store model files in this repo. Keep `captioning.local_files_only = true` or pass `--local-files-only` when you want to prevent Hugging Face downloads during captioning.

## WSL versus native Windows

This project defaults to WSL because the proven manual Krea 2 workflow used WSL2 Ubuntu paths, Linux virtual environments, Bash scripts, and musubi-tuner commands. The helper mirrors that known-good setup instead of introducing a second runtime surface.

The web app itself does not require WSL in principle. It is a Python HTTP server plus static HTML. The training backend currently assumes Linux-style paths, `venv/bin/activate`, `/mnt/c/...` Windows mounts, and `bash -lc` execution for cache/train commands.

Native Windows support is possible, but it would need explicit Windows command generation, `venv\\Scripts\\activate` handling, Windows path defaults, and validation against a Windows CUDA/PyTorch/musubi install. Until that is implemented and tested, WSL is the supported runtime.

## Musubi reference

Krea 2 support in musubi-tuner is experimental. The generated commands follow the upstream Krea 2 documentation for latent caching, text encoder caching, and RAW LoRA training: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md>.
