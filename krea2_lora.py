#!/usr/bin/env python3
"""Local helper CLI for repeatable Krea 2 LoRA training with musubi-tuner."""

from __future__ import annotations

import argparse
import os
import re
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
}


@dataclass(frozen=True)
class AppConfig:
    paths: dict[str, Path]
    dataset: dict[str, Any]
    training: dict[str, Any]


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
    train_script: Path


@dataclass
class CheckResult:
    level: str
    label: str
    message: str


def expand_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded)


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
    return AppConfig(paths=paths, dataset=merged["dataset"], training=merged["training"])


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
        train_script=cfg / "train_krea2.sh",
    )


def quote(value: str | os.PathLike[str]) -> str:
    return shlex.quote(str(value))


def output_name(config: AppConfig, project: ProjectPaths) -> str:
    template = str(config.training["output_name_template"])
    return template.format(project=project.name)


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
        (project.train_script, write_text(project.train_script, train_script(config, project), args.force, executable=True)),
    ]
    print(f"Project: {project.root}")
    for path, changed in wrote:
        action = "wrote" if changed else "kept"
        print(f"{action}: {path}")
    return 0


def validate_env(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.create_comfy_dir:
        config.paths["comfy_lora_dir"].mkdir(parents=True, exist_ok=True)
    return print_checks(env_checks(config))


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
    cache_count = sum(1 for item in project.cache.rglob("*") if item.is_file()) if project.cache.is_dir() else 0
    outputs = sorted(project.output.glob("*.safetensors")) if project.output.is_dir() else []
    print(f"Cache files: {cache_count}")
    print(f"LoRA outputs: {len(outputs)}")
    if outputs:
        latest = max(outputs, key=lambda path: path.stat().st_mtime)
        print(f"Latest LoRA: {latest}")
        print(f"Comfy target: {config.paths['comfy_lora_dir'] / latest.name}")
    if images and not outputs:
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
