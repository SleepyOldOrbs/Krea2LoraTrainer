#!/usr/bin/env python3
"""Local helper CLI for repeatable Krea 2 LoRA training with musubi-tuner."""

from __future__ import annotations

import argparse
import os
import re
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only hit on Python < 3.11
    tomllib = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)")

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "paths": {
        "musubi_repo": "~/src/musubi-tuner",
        "musubi_venv": "~/.venvs/musubi-krea2",
        "captioning_venv": "~/.venvs/vl-caption",
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
}


@dataclass(frozen=True)
class AppConfig:
    paths: dict[str, Path]
    dataset: dict[str, Any]
    training: dict[str, Any]
    downloads: dict[str, Any]


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
    return AppConfig(paths=paths, dataset=merged["dataset"], training=merged["training"], downloads=merged["downloads"])


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


def output_name(config: AppConfig, project: ProjectPaths) -> str:
    template = str(config.training["output_name_template"])
    return template.format(project=project.name)


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


def command_train(config: AppConfig, project: ProjectPaths) -> str:
    train = config.training
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
        output_name(config, project),
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

    for key in ["musubi_venv", "captioning_venv"]:
        path = p[key]
        python_bin = path / "bin" / "python"
        if path.is_dir() and python_bin.is_file():
            add("ok", key, str(path))
        else:
            add("error", key, f"Expected venv with bin/python: {path}")

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
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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
    outputs = sorted(project.output.glob("*.safetensors")) if project.output.is_dir() else []
    latest = max(outputs, key=lambda path: path.stat().st_mtime) if outputs else None
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
    return 0


def validate_env(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.create_comfy_dir:
        config.paths["comfy_lora_dir"].mkdir(parents=True, exist_ok=True)
    return print_checks(env_checks(config))


def download_models(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    specs = model_specs(config)
    if args.model != "all":
        specs = [spec for spec in specs if spec["name"] == args.model]

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
    return rc


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
        dest = project.images / source.name
        if dest.exists() and not args.force:
            skipped += 1
        else:
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
        print("5. Dataset report")
        print("6. Dry-run cache and train commands")
        print("7. Run latent cache")
        print("8. Run text cache")
        print("9. Copy latest LoRA to ComfyUI")
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
            dataset_report(argparse.Namespace(config=args.config, project_name=project_name, json=None))
        elif choice == "6":
            cache_latents(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
            cache_text(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
            train(argparse.Namespace(config=args.config, project_name=project_name, dry_run=True, skip_checks=False))
        elif choice == "7":
            cache_latents(argparse.Namespace(config=args.config, project_name=project_name, dry_run=False, skip_checks=False))
        elif choice == "8":
            cache_text(argparse.Namespace(config=args.config, project_name=project_name, dry_run=False, skip_checks=False))
        elif choice == "9":
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


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    project = project_paths(config, args.project_name)
    if not args.skip_checks:
        dataset_rc = print_checks(dataset_checks(project)[0])
        env_rc = print_checks(env_checks(config))
        if dataset_rc or env_rc:
            return 1
    return run_script(command_train(config, project), args.dry_run)


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
    if not source.is_file():
        print(f"Source file does not exist: {source}")
        return 1
    dest_dir = config.paths["comfy_lora_dir"]
    dest = dest_dir / source.name
    print(f"Copy: {source} -> {dest}")
    if args.dry_run:
        print("Dry run: file was not copied.")
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
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
    downloads.add_argument("--model", choices=["all", "krea_raw", "qwen_vae", "qwen_text_encoder"], default="all")
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

    for name, help_text, handler in [
        ("cache-latents", "Run Krea2 latent caching via musubi-tuner.", cache_latents),
        ("cache-text", "Run Krea2 text encoder output caching via musubi-tuner.", cache_text),
        ("train", "Run the conservative Krea2 RAW LoRA training command.", train),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("project_name")
        command.add_argument("--dry-run", action="store_true", help="Print the command script without executing it.")
        command.add_argument("--skip-checks", action="store_true", help="Skip dataset/env preflight checks.")
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
