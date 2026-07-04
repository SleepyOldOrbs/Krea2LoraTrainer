#!/usr/bin/env python3
"""Local helper CLI for repeatable Krea 2 LoRA training with musubi-tuner."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import json
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only hit on Python < 3.11
    tomllib = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)")
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CAPTION_BACKENDS = {"qwen_gguf", "transformers"}
VLM_CAPTION_SCRIPT = r"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


def clean_text(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value.strip(" .")


def generated_text(result):
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return clean_text(first.get("generated_text") or first.get("caption") or "")
        return clean_text(first)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate image captions with a local vision-language model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trigger", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from PIL import Image
        from transformers import pipeline
    except Exception as exc:
        print(
            "error: captioning venv needs pillow and transformers installed "
            f"before VL captions can run: {exc}",
            file=sys.stderr,
        )
        return 2

    pipeline_args = {"model": args.model}
    if args.device == "cpu":
        pipeline_args["device"] = -1
    elif args.device == "cuda":
        pipeline_args["device"] = 0

    try:
        captioner = pipeline("image-to-text", **pipeline_args)
    except Exception as exc:
        print(f"error: could not load caption model {args.model!r}: {exc}", file=sys.stderr)
        return 2

    payload = json.load(sys.stdin)
    written = 0
    for item in payload["items"]:
        image_path = Path(item["image"])
        caption_path = Path(item["caption"])
        try:
            image = Image.open(image_path).convert("RGB")
            text = generated_text(captioner(image, max_new_tokens=args.max_new_tokens))
        except Exception as exc:
            print(f"error: failed to caption {image_path}: {exc}", file=sys.stderr)
            return 1

        trigger = clean_text(args.trigger)
        if trigger and text and trigger.lower() not in text.lower():
            text = f"{trigger}, {text}"
        elif trigger and not text:
            text = trigger

        if not text:
            print(f"error: model returned an empty caption for {image_path}", file=sys.stderr)
            return 1

        caption_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        written += 1
        print(f"captioned: {image_path.name} -> {caption_path.name}")

    print(f"VL caption generation wrote {written} captions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
HF_MODEL_DOWNLOAD_SCRIPT = r"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download or verify a Hugging Face model repository.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(
            "error: captioning venv needs huggingface_hub installed before VL model download can run: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    try:
        cache_path = snapshot_download(
            repo_id=args.model,
            local_files_only=args.local_files_only,
            force_download=args.force and not args.local_files_only,
        )
    except Exception as exc:
        mode = "verify local cache" if args.local_files_only else "download"
        print(f"error: could not {mode} model {args.model!r}: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] vl_caption: {args.model}")
    print(f"cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
HF_FILE_DOWNLOAD_SCRIPT = r"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download or verify a single Hugging Face repository file.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(
            "error: huggingface_hub is required to download caption GGUF files: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    try:
        path = hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            local_dir=args.local_dir,
            force_download=args.force,
        )
    except Exception as exc:
        print(f"error: could not download {args.repo}/{args.filename}: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] hf_file: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "paths": {
        "musubi_repo": "~/src/musubi-tuner",
        "musubi_venv": "~/.venvs/musubi-krea2",
        "captioning_venv": "~/.venvs/vl-caption",
        "caption_models_dir": "~/ai_models/vl-caption/qwen2.5-vl-7b-captioner-relaxed-gguf",
        "projects_root": "~/krea2_loras",
        "krea_raw": "~/ai_models/krea2/raw.safetensors",
        "qwen_vae": "~/ai_models/qwen/split_files/vae/qwen_image_vae.safetensors",
        "qwen_text_encoder": "~/ai_models/qwen/text_encoders/qwen3vl_4b_bf16.safetensors",
        "comfy_lora_dir": "/mnt/c/ComfyUI_windows_portable/ComfyUI/models/loras/krea2-jim",
    },
    "dataset": {
        "resolution": [768, 768],
        "batch_size": 1,
        "enable_bucket": True,
        "caption_extension": ".txt",
    },
    "training": {
        "output_name_template": "{project}_krea2_lora",
        "network_dim": 16,
        "network_alpha": 16,
        "max_train_epochs": 20,
        "save_every_n_epochs": 2,
        "learning_rate": "1e-4",
        "seed": 42,
        "max_data_loader_n_workers": 2,
        "blocks_to_swap": 20,
        "mixed_precision": "bf16",
        "optimizer_type": "adamw8bit",
        "discrete_flow_shift": "2.5",
        "text_cache_batch_size": 1,
    },
    "downloads": {
        "krea_raw_repo": "krea/Krea-2-Raw",
        "krea_raw_file": "raw.safetensors",
        "qwen_vae_repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "qwen_vae_file": "split_files/vae/qwen_image_vae.safetensors",
        "qwen_text_encoder_repo": "Comfy-Org/Qwen3-VL",
        "qwen_text_encoder_file": "text_encoders/qwen3vl_4b_bf16.safetensors",
    },
    "captioning": {
        "backend": "qwen_gguf",
        "model": "mradermacher/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF",
        "model_file": "Qwen2.5-VL-7B-Captioner-Relaxed.Q6_K.gguf",
        "mmproj_file": "Qwen2.5-VL-7B-Captioner-Relaxed.mmproj-f16.gguf",
        "llama_cli": "llama-qwen2vl-cli",
        "server_url": "",
        "prompt": (
            "Write one concise but detailed image caption for training a visual style model. "
            "Describe only visible content: subject, setting, medium, composition, lighting, colour palette, "
            "mood, texture, camera or painting qualities, and notable details. "
            "Do not use a title, label, or colon prefix. Do not mention AI, diffusion, LoRA, generated, "
            "or an artist name unless it appears as visible text. "
            "Return only the caption."
        ),
        "temperature": "0.2",
        "top_p": "0.9",
        "gpu_layers": 99,
        "max_new_tokens": 180,
        "local_files_only": True,
    },
}


@dataclass(frozen=True)
class AppConfig:
    paths: dict[str, Path]
    dataset: dict[str, Any]
    training: dict[str, Any]
    downloads: dict[str, Any]
    captioning: dict[str, Any]


@dataclass(frozen=True)
class ProjectPaths:
    name: str
    root: Path
    images: Path
    cache: Path
    output: Path
    config: Path
    dataset_toml: Path
    paths_env: Path
    cache_latents_script: Path
    cache_text_script: Path
    train_script: Path
    copy_to_comfy_script: Path


@dataclass
class CheckResult:
    level: str
    label: str
    message: str


def expand_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded)


def cli_path(value: str | os.PathLike[str]) -> Path:
    raw = str(value)
    match = WINDOWS_PATH_RE.match(raw)
    if os.name == "posix" and match:
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{tail}")
    return expand_path(raw)


def deep_merge(base: dict[str, dict[str, Any]], override: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {section: dict(values) for section, values in base.items()}
    for section, values in override.items():
        if isinstance(values, dict):
            merged.setdefault(section, {})
            merged[section].update(values)
        else:
            raise ValueError(f"Top-level config key '{section}' must be a table")
    return merged


def load_config(path: Path | None) -> AppConfig:
    selected = path or Path(os.environ.get("KREA2_LORA_CONFIG", "config.toml"))
    loaded: dict[str, Any] = {}
    if selected.exists():
        if tomllib is None:
            raise RuntimeError("Reading TOML config files requires Python 3.11 or newer")
        with selected.open("rb") as handle:
            loaded = tomllib.load(handle)
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {selected}")

    merged = deep_merge(DEFAULT_CONFIG, loaded)
    paths = {name: expand_path(value) for name, value in merged["paths"].items()}
    return AppConfig(
        paths=paths,
        dataset=merged["dataset"],
        training=merged["training"],
        downloads=merged["downloads"],
        captioning=merged["captioning"],
    )


def validate_project_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("Project name must not be empty")
    if Path(name).is_absolute() or "/" in name or "\\" in name:
        raise ValueError("Project name must be a simple folder name, not a path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Project name may contain letters, numbers, dots, underscores, and hyphens")
    return name


def project_paths(config: AppConfig, name: str) -> ProjectPaths:
    safe_name = validate_project_name(name)
    root = config.paths["projects_root"] / safe_name
    cfg = root / "config"
    return ProjectPaths(
        name=safe_name,
        root=root,
        images=root / "images",
        cache=root / "cache",
        output=root / "output",
        config=cfg,
        dataset_toml=cfg / "dataset.toml",
        paths_env=cfg / "paths.env",
        cache_latents_script=cfg / "cache_latents.sh",
        cache_text_script=cfg / "cache_text.sh",
        train_script=cfg / "train_krea2.sh",
        copy_to_comfy_script=cfg / "copy_latest_to_comfy.sh",
    )


def quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def windows_sort_key(value: str | os.PathLike[str]) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value).replace("\\", "/"))
    key: list[tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))
    return tuple(key)


def windows_sorted_paths(paths: list[Path] | tuple[Path, ...], root: Path | None = None) -> list[Path]:
    if root is None:
        return sorted(paths, key=lambda path: windows_sort_key(path.name))
    return sorted(paths, key=lambda path: windows_sort_key(path.relative_to(root).as_posix()))


def validate_simple_name(name: str, label: str) -> str:
    if not name or not SAFE_NAME_RE.fullmatch(name):
        raise ValueError(f"{label} may contain letters, numbers, dots, underscores, and hyphens")
    if "/" in name or "\\" in name or Path(name).is_absolute():
        raise ValueError(f"{label} must be a simple name, not a path")
    return name


def output_name(config: AppConfig, project: ProjectPaths, run_name: str = "", exact_output_name: str = "") -> str:
    if exact_output_name:
        return validate_simple_name(exact_output_name, "output name")
    template = str(config.training["output_name_template"])
    safe_run = validate_simple_name(run_name, "run name") if run_name else ""
    rendered = template.format(project=project.name, run=safe_run)
    if safe_run and "{run" not in template:
        rendered = f"{rendered}_{safe_run}"
    return validate_simple_name(rendered, "output name")


def training_settings(config: AppConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(config.training)
    if overrides:
        settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def caption_backend(config: AppConfig, override: str | None = None) -> str:
    backend = (override or str(config.captioning.get("backend", "transformers"))).strip().lower()
    aliases = {
        "hf": "transformers",
        "blip": "transformers",
        "vl": "transformers",
        "qwen": "qwen_gguf",
        "gguf": "qwen_gguf",
    }
    backend = aliases.get(backend, backend)
    if backend not in CAPTION_BACKENDS:
        raise ValueError(f"caption backend must be one of: {', '.join(sorted(CAPTION_BACKENDS))}")
    return backend


def caption_model_id(config: AppConfig, override: str | None = None) -> str:
    return (override or str(config.captioning["model"])).strip()


def caption_gguf_model_file(config: AppConfig, override: str | None = None) -> str:
    return (override or str(config.captioning["model_file"])).strip()


def caption_gguf_mmproj_file(config: AppConfig, override: str | None = None) -> str:
    return (override or str(config.captioning["mmproj_file"])).strip()


def caption_gguf_paths(
    config: AppConfig,
    model_file_override: str | None = None,
    mmproj_file_override: str | None = None,
) -> tuple[Path, Path]:
    model_dir = config.paths["caption_models_dir"]
    return (
        model_dir / caption_gguf_model_file(config, model_file_override),
        model_dir / caption_gguf_mmproj_file(config, mmproj_file_override),
    )


def caption_gguf_specs(
    config: AppConfig,
    repo_override: str | None = None,
    model_file_override: str | None = None,
    mmproj_file_override: str | None = None,
) -> list[dict[str, Any]]:
    repo = caption_model_id(config, repo_override)
    model_file = caption_gguf_model_file(config, model_file_override)
    mmproj_file = caption_gguf_mmproj_file(config, mmproj_file_override)
    model_path, mmproj_path = caption_gguf_paths(config, model_file, mmproj_file)
    return [
        {
            "name": "vl_caption_model",
            "repo": repo,
            "file": model_file,
            "target": model_path,
        },
        {
            "name": "vl_caption_mmproj",
            "repo": repo,
            "file": mmproj_file,
            "target": mmproj_path,
        },
    ]


def local_dir_for_hf_file(target: Path, hf_file: str) -> Path:
    parts = Path(hf_file).parts
    if len(parts) > len(target.parts):
        raise ValueError(f"Cannot derive local dir for {target} from {hf_file}")
    suffix = tuple(str(part) for part in target.parts[-len(parts) :])
    if suffix != parts:
        return target.parent
    local_dir = target
    for _ in parts:
        local_dir = local_dir.parent
    return local_dir


def model_specs(config: AppConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": "krea_raw",
            "repo": str(config.downloads["krea_raw_repo"]),
            "file": str(config.downloads["krea_raw_file"]),
            "target": config.paths["krea_raw"],
        },
        {
            "name": "qwen_vae",
            "repo": str(config.downloads["qwen_vae_repo"]),
            "file": str(config.downloads["qwen_vae_file"]),
            "target": config.paths["qwen_vae"],
        },
        {
            "name": "qwen_text_encoder",
            "repo": str(config.downloads["qwen_text_encoder_repo"]),
            "file": str(config.downloads["qwen_text_encoder_file"]),
            "target": config.paths["qwen_text_encoder"],
        },
    ]


def dataset_toml(config: AppConfig, project: ProjectPaths) -> str:
    resolution = config.dataset["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("dataset.resolution must be a two-item list")
    caption_extension = str(config.dataset["caption_extension"])
    batch_size = int(config.dataset["batch_size"])
    enable_bucket = str(bool(config.dataset["enable_bucket"])).lower()
    return f"""# Generated by krea2_lora.py. Edit if this project needs custom dataset settings.
[general]
caption_extension = "{caption_extension}"
batch_size = {batch_size}
enable_bucket = {enable_bucket}

[[datasets]]
image_directory = "{project.images}"
cache_directory = "{project.cache}"
resolution = [{int(resolution[0])}, {int(resolution[1])}]
caption_extension = "{caption_extension}"
batch_size = {batch_size}
enable_bucket = {enable_bucket}
"""


def paths_env(config: AppConfig, project: ProjectPaths) -> str:
    values = {
        "MUSUBI_REPO": config.paths["musubi_repo"],
        "MUSUBI_VENV": config.paths["musubi_venv"],
        "CAPTIONING_VENV": config.paths["captioning_venv"],
        "PROJECT": project.root,
        "KREA_RAW": config.paths["krea_raw"],
        "QWEN_VAE": config.paths["qwen_vae"],
        "QWEN_TEXT_ENCODER": config.paths["qwen_text_encoder"],
        "COMFY_LORA_DIR": config.paths["comfy_lora_dir"],
        "OUTPUT_NAME": output_name(config, project),
    }
    lines = ["# Generated by krea2_lora.py. Safe to inspect and edit."]
    for key, value in values.items():
        lines.append(f"export {key}={quote(value)}")
    return "\n".join(lines) + "\n"


def train_script(config: AppConfig, project: ProjectPaths) -> str:
    train = config.training
    return f"""#!/usr/bin/env bash
set -euo pipefail

source "{project.paths_env}"
source "$MUSUBI_VENV/bin/activate"
cd "$MUSUBI_REPO"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision {train["mixed_precision"]} \\
  src/musubi_tuner/krea2_train_network.py \\
  --dit "$KREA_RAW" \\
  --vae "$QWEN_VAE" \\
  --dataset_config "$PROJECT/config/dataset.toml" \\
  --sdpa \\
  --mixed_precision {train["mixed_precision"]} \\
  --timestep_sampling shift \\
  --weighting_scheme none \\
  --discrete_flow_shift {train["discrete_flow_shift"]} \\
  --optimizer_type {train["optimizer_type"]} \\
  --learning_rate {train["learning_rate"]} \\
  --gradient_checkpointing \\
  --max_data_loader_n_workers {int(train["max_data_loader_n_workers"])} \\
  --persistent_data_loader_workers \\
  --network_module networks.lora_krea2 \\
  --network_dim {int(train["network_dim"])} \\
  --network_alpha {int(train["network_alpha"])} \\
  --max_train_epochs {int(train["max_train_epochs"])} \\
  --save_every_n_epochs {int(train["save_every_n_epochs"])} \\
  --seed {int(train["seed"])} \\
  --output_dir "$PROJECT/output" \\
  --output_name "$OUTPUT_NAME" \\
  --fp8_base \\
  --fp8_scaled \\
  --blocks_to_swap {int(train["blocks_to_swap"])}
"""


def cache_latents_script(project: ProjectPaths) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

source "{project.paths_env}"
source "$MUSUBI_VENV/bin/activate"
cd "$MUSUBI_REPO"

python src/musubi_tuner/krea2_cache_latents.py \\
  --dataset_config "$PROJECT/config/dataset.toml" \\
  --vae "$QWEN_VAE"
"""


def cache_text_script(config: AppConfig, project: ProjectPaths) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

source "{project.paths_env}"
source "$MUSUBI_VENV/bin/activate"
cd "$MUSUBI_REPO"

python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \\
  --dataset_config "$PROJECT/config/dataset.toml" \\
  --text_encoder "$QWEN_TEXT_ENCODER" \\
  --batch_size {int(config.training["text_cache_batch_size"])}
"""


def copy_to_comfy_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/paths.env"
mkdir -p "$COMFY_LORA_DIR"

latest="$(find "$PROJECT/output" -maxdepth 1 -type f -name '*.safetensors' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${latest:-}" ]]; then
  echo "No .safetensors LoRA output found in $PROJECT/output" >&2
  exit 1
fi

cp -v "$latest" "$COMFY_LORA_DIR/"
"""


def write_text(path: Path, content: str, force: bool = False, executable: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    if executable and os.name == "posix":
        path.chmod(path.stat().st_mode | 0o755)
    return True


def command_cache_latents(config: AppConfig, project: ProjectPaths) -> str:
    args = [
        "python",
        "src/musubi_tuner/krea2_cache_latents.py",
        "--dataset_config",
        str(project.dataset_toml),
        "--vae",
        str(config.paths["qwen_vae"]),
    ]
    return activation_script(config) + "\n" + " ".join(quote(arg) for arg in args)


def command_cache_text(config: AppConfig, project: ProjectPaths) -> str:
    args = [
        "python",
        "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
        "--dataset_config",
        str(project.dataset_toml),
        "--text_encoder",
        str(config.paths["qwen_text_encoder"]),
        "--batch_size",
        str(int(config.training["text_cache_batch_size"])),
    ]
    return activation_script(config) + "\n" + " ".join(quote(arg) for arg in args)


def command_train(
    config: AppConfig,
    project: ProjectPaths,
    run_name: str = "",
    exact_output_name: str = "",
    training_overrides: dict[str, Any] | None = None,
) -> str:
    train = training_settings(config, training_overrides)
    args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        str(train["mixed_precision"]),
        "src/musubi_tuner/krea2_train_network.py",
        "--dit",
        str(config.paths["krea_raw"]),
        "--vae",
        str(config.paths["qwen_vae"]),
        "--dataset_config",
        str(project.dataset_toml),
        "--sdpa",
        "--mixed_precision",
        str(train["mixed_precision"]),
        "--timestep_sampling",
        "shift",
        "--weighting_scheme",
        "none",
        "--discrete_flow_shift",
        str(train["discrete_flow_shift"]),
        "--optimizer_type",
        str(train["optimizer_type"]),
        "--learning_rate",
        str(train["learning_rate"]),
        "--gradient_checkpointing",
        "--max_data_loader_n_workers",
        str(int(train["max_data_loader_n_workers"])),
        "--persistent_data_loader_workers",
        "--network_module",
        "networks.lora_krea2",
        "--network_dim",
        str(int(train["network_dim"])),
        "--network_alpha",
        str(int(train["network_alpha"])),
        "--max_train_epochs",
        str(int(train["max_train_epochs"])),
        "--save_every_n_epochs",
        str(int(train["save_every_n_epochs"])),
        "--seed",
        str(int(train["seed"])),
        "--output_dir",
        str(project.output),
        "--output_name",
        output_name(config, project, run_name, exact_output_name),
        "--fp8_base",
        "--fp8_scaled",
        "--blocks_to_swap",
        str(int(train["blocks_to_swap"])),
    ]
    return activation_script(config) + "\n" + " ".join(quote(arg) for arg in args)


def activation_script(config: AppConfig) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"source {quote(config.paths['musubi_venv'] / 'bin' / 'activate')}",
            f"cd {quote(config.paths['musubi_repo'])}",
        ]
    )


def print_script(script: str) -> None:
    print("Command script:")
    print(script)


def run_script(script: str, dry_run: bool) -> int:
    print_script(script)
    if dry_run:
        print("Dry run: command was not executed.")
        return 0
    return subprocess.run(["bash", "-lc", script], check=False).returncode


def nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while current != current.parent:
        if current.exists():
            return current
        current = current.parent
    return current if current.exists() else None


def can_create_dir(path: Path) -> bool:
    if path.exists():
        return path.is_dir()
    parent = nearest_existing_parent(path.parent)
    return bool(parent and parent.is_dir() and os.access(parent, os.W_OK))


def env_checks(config: AppConfig) -> list[CheckResult]:
    p = config.paths
    checks: list[CheckResult] = []

    def add(level: str, label: str, message: str) -> None:
        checks.append(CheckResult(level=level, label=label, message=message))

    if os.name != "posix":
        add("warn", "runtime", "This helper is intended to run inside WSL/Linux; Windows path expansion may be wrong.")

    musubi_repo = p["musubi_repo"]
    if musubi_repo.is_dir():
        add("ok", "musubi_repo", str(musubi_repo))
        for script in [
            "src/musubi_tuner/krea2_cache_latents.py",
            "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
            "src/musubi_tuner/krea2_train_network.py",
        ]:
            target = musubi_repo / script
            add("ok" if target.is_file() else "error", script, str(target))
    else:
        add("error", "musubi_repo", f"Missing directory: {musubi_repo}")

    try:
        backend = caption_backend(config)
        add("ok", "caption_backend", backend)
    except ValueError as exc:
        backend = "transformers"
        add("error", "caption_backend", str(exc))

    for key in ["musubi_venv"]:
        path = p[key]
        python_bin = path / "bin" / "python"
        if path.is_dir() and python_bin.is_file():
            add("ok", key, str(path))
        else:
            add("error", key, f"Expected venv with bin/python: {path}")

    caption_venv = p["captioning_venv"]
    caption_python = caption_venv / "bin" / "python"
    if backend == "transformers":
        if caption_venv.is_dir() and caption_python.is_file():
            add("ok", "captioning_venv", str(caption_venv))
        else:
            add("error", "captioning_venv", f"Expected venv with bin/python: {caption_venv}")
    elif caption_venv.is_dir() and caption_python.is_file():
        add("ok", "captioning_venv", f"{caption_venv} (available for transformers fallback)")
    else:
        add("warn", "captioning_venv", f"Not needed for qwen_gguf backend: {caption_venv}")

    if backend == "qwen_gguf":
        model_path, mmproj_path = caption_gguf_paths(config)
        add("ok" if model_path.is_file() else "warn", "vl_caption_model", str(model_path))
        add("ok" if mmproj_path.is_file() else "warn", "vl_caption_mmproj", str(mmproj_path))
        llama_cli = str(config.captioning.get("llama_cli", "llama-qwen2vl-cli"))
        resolved = resolve_executable(llama_cli)
        add("ok" if resolved else "warn", "llama_cli", resolved or f"Not found on PATH: {llama_cli}")

    for key in ["krea_raw", "qwen_vae", "qwen_text_encoder"]:
        path = p[key]
        add("ok" if path.is_file() else "error", key, str(path))

    projects_root = p["projects_root"]
    add("ok" if can_create_dir(projects_root) else "error", "projects_root", str(projects_root))

    comfy_dir = p["comfy_lora_dir"]
    if comfy_dir.is_dir():
        add("ok", "comfy_lora_dir", str(comfy_dir))
    elif can_create_dir(comfy_dir):
        add("warn", "comfy_lora_dir", f"Missing, but parent appears writable: {comfy_dir}")
    else:
        add("error", "comfy_lora_dir", f"Missing and parent is not writable: {comfy_dir}")
    return checks


def print_checks(checks: list[CheckResult]) -> int:
    for result in checks:
        print(f"[{result.level}] {result.label}: {result.message}")
    return 1 if any(result.level == "error" for result in checks) else 0


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return windows_sorted_paths(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        images_dir,
    )


def image_summary(images: list[Path], root: Path) -> dict[str, Any]:
    total_bytes = sum(path.stat().st_size for path in images)
    return {
        "root": str(root),
        "image_count": len(images),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "extensions": {
            ext: sum(1 for image in images if image.suffix.lower() == ext)
            for ext in sorted({image.suffix.lower() for image in images})
        },
    }


def dataset_report_data(project: ProjectPaths) -> dict[str, Any]:
    checks, images = dataset_checks(project)
    missing = [image.with_suffix(".txt") for image in images if not image.with_suffix(".txt").exists()]
    empty = [
        image.with_suffix(".txt")
        for image in images
        if image.with_suffix(".txt").exists()
        and not image.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore").strip()
    ]
    outputs = windows_sorted_paths(list(project.output.glob("*.safetensors"))) if project.output.is_dir() else []
    latest = max(outputs, key=lambda path: path.stat().st_mtime) if outputs else None
    lora_outputs = [
        {
            "name": output.name,
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "modified": output.stat().st_mtime,
        }
        for output in sorted(outputs, key=lambda path: (-path.stat().st_mtime, windows_sort_key(path.name)))
    ]
    return {
        "project": project.name,
        "project_root": str(project.root),
        "images": image_summary(images, project.images),
        "missing_caption_count": len(missing),
        "empty_caption_count": len(empty),
        "cache_file_count": sum(1 for item in project.cache.rglob("*") if item.is_file())
        if project.cache.is_dir()
        else 0,
        "lora_output_count": len(outputs),
        "latest_lora": str(latest) if latest else None,
        "lora_outputs": lora_outputs,
        "ok": not any(check.level == "error" for check in checks),
    }


def print_dataset_report(report: dict[str, Any]) -> None:
    print(f"Project: {report['project']}")
    print(f"Root: {report['project_root']}")
    print(f"Images: {report['images']['image_count']} ({report['images']['total_mb']} MB)")
    print(f"Extensions: {report['images']['extensions']}")
    print(f"Missing captions: {report['missing_caption_count']}")
    print(f"Empty captions: {report['empty_caption_count']}")
    print(f"Cache files: {report['cache_file_count']}")
    print(f"LoRA outputs: {report['lora_output_count']}")
    if report["latest_lora"]:
        print(f"Latest LoRA: {report['latest_lora']}")
    print(f"Dataset OK: {report['ok']}")


def dataset_checks(project: ProjectPaths) -> tuple[list[CheckResult], list[Path]]:
    checks: list[CheckResult] = []

    def add(level: str, label: str, message: str) -> None:
        checks.append(CheckResult(level=level, label=label, message=message))

    if not project.root.is_dir():
        add("error", "project", f"Missing project directory: {project.root}")
        return checks, []

    for label, path in [
        ("images", project.images),
        ("cache", project.cache),
        ("output", project.output),
        ("config", project.config),
    ]:
        add("ok" if path.is_dir() else "error", label, str(path))

    add("ok" if project.dataset_toml.is_file() else "error", "dataset_toml", str(project.dataset_toml))

    images = list_images(project.images)
    if images:
        add("ok", "image_count", str(len(images)))
    else:
        add("error", "image_count", f"No images found in {project.images}")

    missing: list[Path] = []
    empty: list[Path] = []
    for image in images:
        caption = image.with_suffix(".txt")
        if not caption.exists():
            missing.append(caption)
        elif not caption.read_text(encoding="utf-8", errors="ignore").strip():
            empty.append(caption)

    if missing:
        add("error", "missing_captions", f"{len(missing)} missing .txt captions")
        for caption in missing[:10]:
            add("error", "missing_caption", str(caption))
        if len(missing) > 10:
            add("error", "missing_caption", f"...and {len(missing) - 10} more")
    else:
        add("ok", "missing_captions", "0")

    if empty:
        add("error", "empty_captions", f"{len(empty)} empty .txt captions")
        for caption in empty[:10]:
            add("error", "empty_caption", str(caption))
        if len(empty) > 10:
            add("error", "empty_caption", f"...and {len(empty) - 10} more")
    else:
        add("ok", "empty_captions", "0")

    return checks, images


def create_project(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    for path in [project.images, project.cache, project.output, project.config]:
        path.mkdir(parents=True, exist_ok=True)

    wrote = [
        (project.dataset_toml, write_text(project.dataset_toml, dataset_toml(config, project), args.force)),
        (project.paths_env, write_text(project.paths_env, paths_env(config, project), args.force)),
        (
            project.cache_latents_script,
            write_text(project.cache_latents_script, cache_latents_script(project), args.force, executable=True),
        ),
        (
            project.cache_text_script,
            write_text(project.cache_text_script, cache_text_script(config, project), args.force, executable=True),
        ),
        (project.train_script, write_text(project.train_script, train_script(config, project), args.force, executable=True)),
        (
            project.copy_to_comfy_script,
            write_text(project.copy_to_comfy_script, copy_to_comfy_script(), args.force, executable=True),
        ),
    ]
    print(f"Project: {project.root}")
    for path, changed in wrote:
        action = "wrote" if changed else "kept"
        print(f"{action}: {path}")
    return 0


def show_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print("[paths]")
    for key, value in config.paths.items():
        print(f"{key} = {value}")
    print()
    print("[dataset]")
    for key, value in config.dataset.items():
        print(f"{key} = {value}")
    print()
    print("[training]")
    for key, value in config.training.items():
        print(f"{key} = {value}")
    print()
    print("[downloads]")
    for key, value in config.downloads.items():
        print(f"{key} = {value}")
    print()
    print("[captioning]")
    for key, value in config.captioning.items():
        print(f"{key} = {value}")
    return 0


def validate_env(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.create_comfy_dir:
        config.paths["comfy_lora_dir"].mkdir(parents=True, exist_ok=True)
    return print_checks(env_checks(config))


def download_models(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    specs = model_specs(config)
    include_vl_caption = args.model in {"all", "vl_caption"}
    if args.model not in {"all", "vl_caption"}:
        specs = [spec for spec in specs if spec["name"] == args.model]
    elif args.model == "vl_caption":
        specs = []

    rc = 0
    for spec in specs:
        target = Path(spec["target"])
        exists = target.is_file() and target.stat().st_size > 0
        size_mb = round(target.stat().st_size / 1024 / 1024, 2) if exists else 0
        label = str(spec["name"])
        if exists and not args.force:
            print(f"[ok] {label}: {target} ({size_mb} MB), skipping existing file")
            continue
        if args.verify_only:
            print(f"[missing] {label}: {target}")
            rc = 1
            continue
        if shutil.which("huggingface-cli") is None:
            print(
                "error: huggingface-cli was not found on PATH. Install huggingface_hub in the active environment.",
                file=sys.stderr,
            )
            return 2

        local_dir = local_dir_for_hf_file(target, str(spec["file"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "huggingface-cli",
            "download",
            str(spec["repo"]),
            str(spec["file"]),
            "--local-dir",
            str(local_dir),
        ]
        print("Download command:")
        print(" ".join(quote(part) for part in command))
        if args.dry_run:
            print("Dry run: model was not downloaded.")
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            rc = completed.returncode
        elif target.is_file():
            print(f"[ok] {label}: {target} ({round(target.stat().st_size / 1024 / 1024, 2)} MB)")
        else:
            print(f"[error] {label}: download command finished, but expected file is missing: {target}")
            rc = 1
    if include_vl_caption:
        vl_rc = download_vl_caption_model(config, args)
        if vl_rc:
            rc = vl_rc
    return rc


def download_vl_caption_model(config: AppConfig, args: argparse.Namespace) -> int:
    backend = caption_backend(config)
    if backend == "qwen_gguf":
        return download_vl_caption_gguf_model(config, args)
    return download_vl_caption_transformers_model(config, args)


def download_vl_caption_gguf_model(config: AppConfig, args: argparse.Namespace) -> int:
    specs = caption_gguf_specs(
        config,
        args.caption_model,
        getattr(args, "caption_model_file", None),
        getattr(args, "caption_mmproj_file", None),
    )
    rc = 0
    print("VL caption backend: qwen_gguf")
    python_bin = captioning_python(config)
    downloader_python = python_bin if python_bin.is_file() else Path(sys.executable)
    for spec in specs:
        target = Path(spec["target"])
        exists = target.is_file() and target.stat().st_size > 0
        size_mb = round(target.stat().st_size / 1024 / 1024, 2) if exists else 0
        label = str(spec["name"])
        if exists and not args.force:
            print(f"[ok] {label}: {target} ({size_mb} MB), skipping existing file")
            continue
        if args.verify_only:
            print(f"[missing] {label}: {target}")
            rc = 1
            continue
        local_dir = local_dir_for_hf_file(target, str(spec["file"]))
        command = [
            str(downloader_python),
            "-c",
            HF_FILE_DOWNLOAD_SCRIPT,
            "--repo",
            str(spec["repo"]),
            "--filename",
            str(spec["file"]),
            "--local-dir",
            str(local_dir),
        ]
        if args.force:
            command.append("--force")
        print("Download command:")
        display_command = [
            str(downloader_python),
            "-c",
            "<hf file download helper>",
            "--repo",
            str(spec["repo"]),
            "--filename",
            str(spec["file"]),
            "--local-dir",
            str(local_dir),
        ]
        if args.force:
            display_command.append("--force")
        print(" ".join(quote(part) for part in display_command))
        if args.dry_run:
            print("Dry run: VL caption file was not downloaded.")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode:
            rc = completed.returncode
        elif target.is_file():
            print(f"[ok] {label}: {target} ({round(target.stat().st_size / 1024 / 1024, 2)} MB)")
        else:
            print(f"[error] {label}: download command finished, but expected file is missing: {target}")
            rc = 1
    return rc


def download_vl_caption_transformers_model(config: AppConfig, args: argparse.Namespace) -> int:
    model = args.caption_model or str(config.captioning["model"])
    python_bin = captioning_python(config)
    command = [
        str(python_bin),
        "-c",
        HF_MODEL_DOWNLOAD_SCRIPT,
        "--model",
        model,
    ]
    if args.verify_only:
        command.append("--local-files-only")
    if args.force:
        command.append("--force")

    print(f"VL caption model: {model}")
    print("Download command:")
    display_command = [
        str(python_bin),
        "-c",
        "<hf model download helper>",
        "--model",
        model,
    ]
    if args.verify_only:
        display_command.append("--local-files-only")
    if args.force:
        display_command.append("--force")
    print(" ".join(quote(part) for part in display_command))
    if args.dry_run:
        print("Dry run: VL caption model was not downloaded.")
        return 0
    if not python_bin.is_file():
        print(f"Captioning venv python not found: {python_bin}")
        return 1

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def check_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    checks, _ = dataset_checks(project)
    return print_checks(checks)


def create_caption_stubs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    _, images = dataset_checks(project)
    if not images:
        print(f"No images found in {project.images}")
        return 1
    created = 0
    skipped = 0
    body = args.trigger.strip()
    if not body:
        print("Trigger must not be empty")
        return 1
    for image in images:
        caption = image.with_suffix(".txt")
        if caption.exists() and caption.read_text(encoding="utf-8", errors="ignore").strip():
            skipped += 1
            continue
        caption.write_text(body + "\n", encoding="utf-8", newline="\n")
        created += 1
    print(f"Created or filled {created} caption stubs. Skipped {skipped} non-empty captions.")
    return 0


def captioning_python(config: AppConfig) -> Path:
    return config.paths["captioning_venv"] / "bin" / "python"


def caption_generation_targets(project: ProjectPaths, force: bool, limit: int | None) -> tuple[list[dict[str, str]], int]:
    targets: list[dict[str, str]] = []
    skipped = 0
    for image in list_images(project.images):
        caption = image.with_suffix(".txt")
        existing = caption.exists() and caption.read_text(encoding="utf-8", errors="ignore").strip()
        if existing and not force:
            skipped += 1
            continue
        targets.append({"image": str(image), "caption": str(caption)})
        if limit is not None and len(targets) >= limit:
            break
    return targets, skipped


def clean_caption_text(value: object) -> str:
    text = ANSI_RE.sub("", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .\"'“”")


def caption_with_trigger(text: str, trigger: str) -> str:
    clean_trigger = clean_caption_text(trigger)
    clean_text = clean_caption_text(text)
    if clean_trigger and clean_text and clean_trigger.lower() not in clean_text.lower():
        return f"{clean_trigger}, {clean_text}"
    if clean_trigger and not clean_text:
        return clean_trigger
    return clean_text


def extract_qwen_caption(stdout: str, prompt: str) -> str:
    text = ANSI_RE.sub("", stdout or "")
    if prompt and prompt in text:
        text = text.split(prompt, 1)[-1]
    skip_prefixes = (
        "build:",
        "clip_",
        "common_",
        "encode_",
        "generate:",
        "ggml_",
        "llama_",
        "load_",
        "main:",
        "sampling:",
        "sampler",
        "system_info:",
        "mtmd_",
        "warn:",
        "usage:",
        "for normal use",
        "this is an experimental cli",
        "you are a helpful assistant",
    )
    lines: list[str] = []
    skip_template = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^\d+\.\d+\.\d+\.\d+\s+[A-Z]\s+", "", line).strip()
        if not line:
            skip_template = False
            continue
        lower = line.lower()
        if "chat template example:" in lower:
            skip_template = True
            continue
        if skip_template or lower.startswith(("<|im_start|>", "<|im_end|>")):
            continue
        if lower.startswith(skip_prefixes):
            continue
        line = re.sub(r"^(assistant|caption)\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        lines.append(line)
    return clean_caption_text(" ".join(lines))


def resolve_executable(command: str) -> str | None:
    value = command.strip()
    if not value:
        return None
    candidate = expand_path(value)
    if candidate.is_absolute() or "/" in value or "\\" in value:
        return str(candidate) if candidate.is_file() else None
    return shutil.which(value)


def print_caption_plan(
    backend: str,
    model: str,
    targets: list[dict[str, str]],
    skipped: int,
    local_files_only: bool,
) -> None:
    print(f"Caption backend: {backend}")
    print(f"Caption model: {model}")
    print(f"Caption targets: {len(targets)} images. Skipped {skipped} existing non-empty captions.")
    if local_files_only:
        print("Local cache only: enabled. No Hugging Face downloads will be attempted.")
    else:
        print("Local cache only: disabled. Missing configured caption files may be downloaded first.")


def run_transformers_caption_backend(
    config: AppConfig,
    args: argparse.Namespace,
    targets: list[dict[str, str]],
    model: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> int:
    python_bin = captioning_python(config)
    if not python_bin.is_file():
        print(f"Captioning venv python not found: {python_bin}")
        return 1

    command = [
        str(python_bin),
        "-c",
        VLM_CAPTION_SCRIPT,
        "--model",
        model,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        args.device,
    ]
    if args.trigger:
        command.extend(["--trigger", args.trigger.strip()])
    if local_files_only:
        command.append("--local-files-only")

    completed = subprocess.run(
        command,
        input=json.dumps({"items": targets}),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def run_qwen_gguf_caption_backend(
    config: AppConfig,
    args: argparse.Namespace,
    targets: list[dict[str, str]],
    model: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> int:
    server_url = (getattr(args, "server_url", "") or str(config.captioning.get("server_url", ""))).strip()
    if server_url:
        return run_qwen_server_caption_backend(config, args, targets, model, max_new_tokens, server_url)

    model_file = getattr(args, "model_file", None)
    mmproj_file = getattr(args, "mmproj_file", None)
    model_path, mmproj_path = caption_gguf_paths(config, model_file, mmproj_file)
    missing = [path for path in [model_path, mmproj_path] if not path.is_file()]
    if missing and not local_files_only:
        download_args = argparse.Namespace(
            caption_model=model,
            caption_model_file=model_file,
            caption_mmproj_file=mmproj_file,
            verify_only=False,
            force=False,
            dry_run=False,
        )
        rc = download_vl_caption_gguf_model(config, download_args)
        if rc:
            return rc
        missing = [path for path in [model_path, mmproj_path] if not path.is_file()]
    if missing:
        print("Missing Qwen GGUF caption files:")
        for path in missing:
            print(f"  {path}")
        print("Run: python krea2_lora.py download-models --model vl_caption")
        return 1

    llama_cli = getattr(args, "llama_cli", "") or str(config.captioning.get("llama_cli", "llama-qwen2vl-cli"))
    executable = resolve_executable(llama_cli)
    if executable is None:
        print(f"Qwen GGUF caption executable not found: {llama_cli}")
        print("Install a Qwen-VL capable llama.cpp/KoboldCPP build, or set captioning.llama_cli in config.toml.")
        return 1

    prompt = (
        getattr(args, "prompt", "")
        or str(config.captioning.get("prompt", "Describe this image for diffusion LoRA training."))
    ).strip()
    temperature = str(config.captioning.get("temperature", "")).strip()
    top_p = str(config.captioning.get("top_p", "")).strip()
    gpu_layers = getattr(args, "gpu_layers", None)
    if gpu_layers is None:
        gpu_layers = config.captioning.get("gpu_layers", "")
    gpu_layers_text = str(gpu_layers).strip()

    print(f"Qwen GGUF file: {model_path.name}")
    print(f"Qwen mmproj file: {mmproj_path.name}")
    print(f"Qwen executable: {executable}")
    if gpu_layers_text:
        print(f"Qwen GPU layers: {gpu_layers_text}")

    written = 0
    for item in targets:
        image_path = Path(item["image"])
        caption_path = Path(item["caption"])
        command = [
            executable,
            "-m",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "-p",
            prompt,
            "--image",
            str(image_path),
            "-n",
            str(max_new_tokens),
            "--no-perf",
        ]
        if temperature:
            command.extend(["--temp", temperature])
        if top_p:
            command.extend(["--top-p", top_p])
        if gpu_layers_text:
            command.extend(["-ngl", gpu_layers_text])

        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode:
            print(f"error: failed to caption {image_path}", file=sys.stderr)
            return completed.returncode

        text = caption_with_trigger(extract_qwen_caption(completed.stdout, prompt), args.trigger)
        if not text:
            print(f"error: model returned an empty caption for {image_path}", file=sys.stderr)
            return 1

        caption_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        written += 1
        print(f"captioned: {image_path.name} -> {caption_path.name}")

    print(f"VL caption generation wrote {written} captions.")
    return 0


def qwen_server_caption_payload(
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: str,
    top_p: str,
) -> dict[str, Any]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ],
        "max_tokens": max_new_tokens,
    }
    if temperature:
        payload["temperature"] = float(temperature)
    if top_p:
        payload["top_p"] = float(top_p)
    return payload


def run_qwen_server_caption_backend(
    config: AppConfig,
    args: argparse.Namespace,
    targets: list[dict[str, str]],
    model: str,
    max_new_tokens: int,
    server_url: str,
) -> int:
    prompt = (
        getattr(args, "prompt", "")
        or str(config.captioning.get("prompt", "Describe this image for diffusion LoRA training."))
    ).strip()
    temperature = str(config.captioning.get("temperature", "")).strip()
    top_p = str(config.captioning.get("top_p", "")).strip()
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"

    print(f"Qwen server URL: {server_url.rstrip('/')}")
    print(f"Qwen server model: {model}")
    written = 0
    for item in targets:
        image_path = Path(item["image"])
        caption_path = Path(item["caption"])
        payload = qwen_server_caption_payload(image_path, prompt, max_new_tokens, temperature, top_p)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"error: Qwen server rejected {image_path}: HTTP {exc.code}: {body}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"error: Qwen server request failed for {image_path}: {exc}", file=sys.stderr)
            return 1

        try:
            raw_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            print(f"error: Qwen server returned an unexpected response for {image_path}: {exc}", file=sys.stderr)
            return 1

        text = caption_with_trigger(raw_text, args.trigger)
        if not text:
            print(f"error: model returned an empty caption for {image_path}", file=sys.stderr)
            return 1
        caption_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        written += 1
        print(f"captioned: {image_path.name} -> {caption_path.name}")

    print(f"VL caption generation wrote {written} captions.")
    return 0


def run_caption_backend(
    config: AppConfig,
    args: argparse.Namespace,
    targets: list[dict[str, str]],
    skipped: int,
    model: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> int:
    try:
        backend = caption_backend(config, getattr(args, "backend", None))
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    print_caption_plan(backend, model, targets, skipped, local_files_only)
    if backend == "transformers":
        rc = run_transformers_caption_backend(config, args, targets, model, max_new_tokens, local_files_only)
    else:
        rc = run_qwen_gguf_caption_backend(config, args, targets, model, max_new_tokens, local_files_only)
    if rc:
        return rc
    print(f"Generated {len(targets)} captions. Skipped {skipped} existing non-empty captions.")
    return 0


def generate_captions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    if not project.images.is_dir():
        print(f"Missing image directory: {project.images}")
        return 1

    images = list_images(project.images)
    if not images:
        print(f"No supported images found in {project.images}")
        return 1

    model = caption_model_id(config, args.model)
    max_new_tokens = args.max_new_tokens or int(config.captioning["max_new_tokens"])
    local_files_only = (
        bool(config.captioning["local_files_only"]) if args.local_files_only is None else bool(args.local_files_only)
    )
    targets, skipped = caption_generation_targets(project, args.force, args.limit)
    if not targets:
        print(f"No captions need generation. Skipped {skipped} existing non-empty captions.")
        return 0

    try:
        backend = caption_backend(config, args.backend)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    if args.dry_run:
        print_caption_plan(backend, model, targets, skipped, local_files_only)
        for item in targets[:10]:
            print(f"would caption: {item['image']} -> {item['caption']}")
        if len(targets) > 10:
            print(f"...and {len(targets) - 10} more")
        print("Dry run: captions were not generated.")
        return 0

    return run_caption_backend(config, args, targets, skipped, model, max_new_tokens, local_files_only)


def import_images(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    source_dir = cli_path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}")
        return 1
    project.images.mkdir(parents=True, exist_ok=True)
    images = list_images(source_dir)
    if not images:
        print(f"No supported images found in {source_dir}")
        return 1

    imported = 0
    skipped = 0
    for source in images:
        dest = project.images / source.relative_to(source_dir)
        if dest.exists() and not args.force:
            skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if args.mode == "copy":
                shutil.copy2(source, dest)
            elif args.mode == "symlink":
                dest.symlink_to(source)
            elif args.mode == "hardlink":
                os.link(source, dest)
            else:  # pragma: no cover - argparse enforces choices
                raise ValueError(f"Unsupported import mode: {args.mode}")
            imported += 1

        if args.trigger:
            caption = dest.with_suffix(".txt")
            if args.force_caption or not caption.exists() or not caption.read_text(encoding="utf-8", errors="ignore").strip():
                caption.write_text(args.trigger.strip() + "\n", encoding="utf-8", newline="\n")

    print(f"Imported {imported} images from {source_dir} using {args.mode}. Skipped {skipped}.")
    if args.trigger:
        print("Caption stubs were created or filled from --trigger.")
    return 0


def dataset_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    report = dataset_report_data(project)
    print_dataset_report(report)
    if args.json:
        destination = cli_path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {destination}")
    return 0 if report["ok"] else 1


def prompt_text(label: str, default: str | None = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{label}{suffix}: ").strip()
        except EOFError:
            return default or ""
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print("A value is required.")


def wizard(args: argparse.Namespace) -> int:
    project_name = args.project_name or prompt_text("Project name", required=True)
    source_dir = args.source_dir or ""
    trigger = args.trigger or ""
    mode = args.mode

    print("Krea2 LoRA Helper")
    print(f"Project: {project_name}")
    while True:
        print()
        print("1. Validate environment")
        print("2. Init project")
        print("3. Import/link images")
        print("4. Create/fill caption stubs")
        print("5. Generate VL captions")
        print("6. Dataset report")
        print("7. Dry-run cache and train commands")
        print("8. Run latent cache")
        print("9. Run text cache")
        print("10. Copy latest LoRA to ComfyUI")
        print("0. Quit")
        try:
            choice = input("Select: ").strip()
        except EOFError:
            print()
            print("Input closed; exiting wizard.")
            return 0

        if choice == "0":
            return 0
        if choice == "1":
            validate_env(argparse.Namespace(config=args.config, create_comfy_dir=False))
        elif choice == "2":
            create_project(argparse.Namespace(config=args.config, project_name=project_name, force=False))
        elif choice == "3":
            source_dir = source_dir or prompt_text("Source image folder", required=True)
            if not trigger:
                trigger = prompt_text("Caption trigger", required=True)
            import_images(
                argparse.Namespace(
                    config=args.config,
                    project_name=project_name,
                    source_dir=source_dir,
                    mode=mode,
                    force=False,
                    trigger=trigger,
                    force_caption=False,
                )
            )
        elif choice == "4":
            trigger = trigger or prompt_text("Caption trigger", required=True)
            create_caption_stubs(argparse.Namespace(config=args.config, project_name=project_name, trigger=trigger))
        elif choice == "5":
            trigger = prompt_text("Caption trigger", default=trigger, required=False)
            generate_captions(
                argparse.Namespace(
                    config=args.config,
                    project_name=project_name,
                    backend=None,
                    model=None,
                    model_file=None,
                    mmproj_file=None,
                    trigger=trigger,
                    force=False,
                    limit=None,
                    max_new_tokens=None,
                    device="auto",
                    llama_cli=None,
                    server_url=None,
                    prompt=None,
                    gpu_layers=None,
                    local_files_only=None,
                    dry_run=False,
                )
            )
        elif choice == "6":
            dataset_report(argparse.Namespace(config=args.config, project_name=project_name, json=None))
        elif choice == "7":
            cache_latents(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
            cache_text(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
            train(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
        elif choice == "8":
            cache_latents(argparse.Namespace(config=args.config, project_name=project_name, dry_run=False, skip_checks=False))
        elif choice == "9":
            cache_text(argparse.Namespace(config=args.config, project_name=project_name, dry_run=False, skip_checks=False))
        elif choice == "10":
            copy_to_comfy(argparse.Namespace(config=args.config, project_name=project_name, file=None, dry_run=False))
        else:
            print("Unknown choice.")


def cache_latents(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    if not args.skip_checks:
        dataset_rc = print_checks(dataset_checks(project)[0])
        env_rc = print_checks(env_checks(config))
        if dataset_rc or env_rc:
            return 1
    return run_script(command_cache_latents(config, project), args.dry_run)


def cache_text(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    if not args.skip_checks:
        dataset_rc = print_checks(dataset_checks(project)[0])
        env_rc = print_checks(env_checks(config))
        if dataset_rc or env_rc:
            return 1
    return run_script(command_cache_text(config, project), args.dry_run)


def train_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    fields = {
        "network_dim": "network_dim",
        "network_alpha": "network_alpha",
        "max_train_epochs": "max_train_epochs",
        "save_every_n_epochs": "save_every_n_epochs",
        "learning_rate": "learning_rate",
        "seed": "seed",
        "blocks_to_swap": "blocks_to_swap",
    }
    overrides: dict[str, Any] = {}
    for attr, key in fields.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    if not args.skip_checks:
        dataset_rc = print_checks(dataset_checks(project)[0])
        env_rc = print_checks(env_checks(config))
        if dataset_rc or env_rc:
            return 1
    return run_script(
        command_train(
            config,
            project,
            run_name=getattr(args, "run_name", "") or "",
            exact_output_name=getattr(args, "output_name", "") or "",
            training_overrides=train_overrides_from_args(args),
        ),
        args.dry_run,
    )


def latest_lora(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("*.safetensors"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def copy_to_comfy(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    source = expand_path(args.file) if args.file else latest_lora(project.output)
    if source is None:
        print(f"No .safetensors LoRA output found in {project.output}")
        return 1
    if source.suffix.lower() != ".safetensors":
        print(f"Source file must be a .safetensors LoRA output: {source}")
        return 1
    if not source.is_file():
        print(f"Source file does not exist: {source}")
        return 1
    resolved_source = source.resolve()
    output_root = project.output.resolve()
    try:
        resolved_source.relative_to(output_root)
    except ValueError:
        print(f"Source file must be inside this project's output folder: {project.output}")
        return 1
    dest_dir = config.paths["comfy_lora_dir"]
    dest = dest_dir / resolved_source.name
    print(f"Copy: {resolved_source} -> {dest}")
    if args.dry_run:
        print("Dry run: file was not copied.")
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, dest)
    print(f"Copied: {dest}")
    return 0


def status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    print(f"Project: {project.root}")
    checks, images = dataset_checks(project)
    print_checks(checks)
    report = dataset_report_data(project)
    print(f"Image bytes: {report['images']['total_bytes']} ({report['images']['total_mb']} MB)")
    print(f"Cache files: {report['cache_file_count']}")
    print(f"LoRA outputs: {report['lora_output_count']}")
    if report["latest_lora"]:
        latest = Path(str(report["latest_lora"]))
        print(f"Latest LoRA: {latest}")
        print(f"Comfy target: {config.paths['comfy_lora_dir'] / latest.name}")
    if images and not report["lora_output_count"]:
        print("Next likely steps: cache-latents, cache-text, train")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krea2_lora.py",
        description="Local Krea 2 LoRA project helper around an existing musubi-tuner install.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml. Defaults to ./config.toml when present, otherwise built-in defaults.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-config", help="Print resolved config values.")
    show.set_defaults(func=show_config)

    downloads = sub.add_parser("download-models", help="Download or verify required Hugging Face model files.")
    downloads.add_argument(
        "--model",
        choices=["all", "krea_raw", "qwen_vae", "qwen_text_encoder", "vl_caption"],
        default="all",
    )
    downloads.add_argument("--caption-model", help="VL caption Hugging Face model. Defaults to config.captioning.model.")
    downloads.add_argument(
        "--caption-model-file",
        help="Qwen GGUF language-model filename. Defaults to config.captioning.model_file.",
    )
    downloads.add_argument(
        "--caption-mmproj-file",
        help="Qwen GGUF vision-projector filename. Defaults to config.captioning.mmproj_file.",
    )
    downloads.add_argument("--force", action="store_true", help="Download even when the target file already exists.")
    downloads.add_argument("--dry-run", action="store_true", help="Print download commands without running them.")
    downloads.add_argument("--verify-only", action="store_true", help="Only verify local files; do not call Hugging Face.")
    downloads.set_defaults(func=download_models)

    init = sub.add_parser("init-project", help="Create project folders and generated config files.")
    init.add_argument("project_name")
    init.add_argument("--force", action="store_true", help="Overwrite generated dataset/env/script files.")
    init.set_defaults(func=create_project)

    env = sub.add_parser("validate-env", help="Validate musubi, venvs, models, and output destinations.")
    env.add_argument("--create-comfy-dir", action="store_true", help="Create the ComfyUI LoRA directory if missing.")
    env.set_defaults(func=validate_env)

    dataset = sub.add_parser("check-dataset", help="Validate images and matching non-empty captions.")
    dataset.add_argument("project_name")
    dataset.set_defaults(func=check_dataset)

    importer = sub.add_parser("import-images", help="Import or link images from another folder into a project.")
    importer.add_argument("project_name")
    importer.add_argument("source_dir")
    importer.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    importer.add_argument("--force", action="store_true", help="Overwrite existing imported images.")
    importer.add_argument("--trigger", help="Create or fill captions with this trigger text while importing.")
    importer.add_argument("--force-caption", action="store_true", help="Overwrite existing captions when --trigger is set.")
    importer.set_defaults(func=import_images)

    report = sub.add_parser("dataset-report", help="Print a compact dataset/cache/output report.")
    report.add_argument("project_name")
    report.add_argument("--json", help="Optional path to write the report JSON.")
    report.set_defaults(func=dataset_report)

    wiz = sub.add_parser("wizard", help="Open a guided terminal menu for the common local workflow.")
    wiz.add_argument("project_name", nargs="?")
    wiz.add_argument("--source-dir", help="Prefill the image source folder for import.")
    wiz.add_argument("--trigger", help="Prefill the caption trigger.")
    wiz.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    wiz.set_defaults(func=wizard)

    stubs = sub.add_parser("create-caption-stubs", help="Create or fill missing/empty .txt captions.")
    stubs.add_argument("project_name")
    stubs.add_argument("--trigger", required=True, help="Caption stub text to write, usually the LoRA trigger phrase.")
    stubs.set_defaults(func=create_caption_stubs)

    vl_captions = sub.add_parser("generate-captions", help="Generate missing/empty captions with a VL model.")
    vl_captions.add_argument("project_name")
    vl_captions.add_argument("--backend", choices=sorted(CAPTION_BACKENDS), help="Caption backend. Defaults to config.")
    vl_captions.add_argument("--model", help="Caption model repo/id. Defaults to config.captioning.model.")
    vl_captions.add_argument("--model-file", help="Qwen GGUF language-model filename. Defaults to config.")
    vl_captions.add_argument("--mmproj-file", help="Qwen GGUF vision-projector filename. Defaults to config.")
    vl_captions.add_argument("--trigger", default="", help="Optional LoRA trigger phrase to prefix generated captions.")
    vl_captions.add_argument("--force", action="store_true", help="Overwrite existing non-empty captions.")
    vl_captions.add_argument("--limit", type=int, help="Maximum number of images to caption in this run.")
    vl_captions.add_argument("--max-new-tokens", type=int, help="Maximum caption tokens. Defaults to config.")
    vl_captions.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    vl_captions.add_argument("--llama-cli", help="Qwen GGUF executable override. Defaults to config.captioning.llama_cli.")
    vl_captions.add_argument("--server-url", help="Use a running llama.cpp server instead of spawning a CLI per image.")
    vl_captions.add_argument("--prompt", help="Qwen GGUF caption prompt override. Defaults to config.captioning.prompt.")
    vl_captions.add_argument("--gpu-layers", type=int, help="Qwen GGUF layers to offload to GPU. Defaults to config.")
    vl_captions.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=None,
        help="Only use models already present in the Hugging Face cache.",
    )
    vl_captions.add_argument(
        "--allow-downloads",
        dest="local_files_only",
        action="store_false",
        help="Allow the selected caption backend to download missing configured model files.",
    )
    vl_captions.add_argument("--dry-run", action="store_true", help="Show planned caption targets without running the VLM.")
    vl_captions.set_defaults(func=generate_captions)

    for name, help_text, handler in [
        ("cache-latents", "Run Krea2 latent caching via musubi-tuner.", cache_latents),
        ("cache-text", "Run Krea2 text encoder output caching via musubi-tuner.", cache_text),
        ("train", "Run the conservative Krea2 RAW LoRA training command.", train),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("project_name")
        command.add_argument("--dry-run", action="store_true", help="Print the command script without executing it.")
        command.add_argument("--skip-checks", action="store_true", help="Skip dataset/env preflight checks.")
        if name == "train":
            command.add_argument("--run-name", default="", help="Safe suffix for this training run output name.")
            command.add_argument("--output-name", default="", help="Exact safe musubi output name for this run.")
            command.add_argument("--network-dim", type=int, help="Override training.network_dim for this run.")
            command.add_argument("--network-alpha", type=int, help="Override training.network_alpha for this run.")
            command.add_argument("--max-train-epochs", type=int, help="Override training.max_train_epochs for this run.")
            command.add_argument(
                "--save-every-n-epochs",
                type=int,
                help="Override training.save_every_n_epochs for this run.",
            )
            command.add_argument("--learning-rate", help="Override training.learning_rate for this run.")
            command.add_argument("--seed", type=int, help="Override training.seed for this run.")
            command.add_argument("--blocks-to-swap", type=int, help="Override training.blocks_to_swap for this run.")
        command.set_defaults(func=handler)

    copy = sub.add_parser("copy-to-comfy", help="Copy the latest or selected LoRA to the ComfyUI LoRA folder.")
    copy.add_argument("project_name")
    copy.add_argument("--file", help="Specific .safetensors file to copy. Defaults to latest project output.")
    copy.add_argument("--dry-run", action="store_true", help="Print the copy action without copying.")
    copy.set_defaults(func=copy_to_comfy)

    stat = sub.add_parser("status", help="Summarize project dataset, cache, and LoRA output state.")
    stat.add_argument("project_name")
    stat.set_defaults(func=status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
