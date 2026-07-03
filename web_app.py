#!/usr/bin/env python3
"""Localhost web UI for the Krea2 LoRA helper."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import krea2_lora


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


class WebState:
    def __init__(self, config: Path | None = None) -> None:
        self.config = config


def config_args(config: Path | None) -> list[str]:
    return ["--config", str(config)] if config else []


def text_value(payload: dict[str, object], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    return str(value).strip()


def bool_value(payload: dict[str, object], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    return bool(value)


def build_cli_args(payload: dict[str, object], config: Path | None = None) -> list[str]:
    action = text_value(payload, "action")
    project = text_value(payload, "project")
    args = [sys.executable, str(ROOT / "krea2_lora.py"), *config_args(config)]

    if action == "validate-env":
        args.append("validate-env")
        if bool_value(payload, "create_comfy_dir"):
            args.append("--create-comfy-dir")
    elif action == "download-models":
        args.append("download-models")
        caption_model = text_value(payload, "caption_model")
        if caption_model:
            args.extend(["--caption-model", caption_model])
    elif action == "init-project":
        require_project(project)
        args.extend(["init-project", project])
        if bool_value(payload, "force"):
            args.append("--force")
    elif action == "import-images":
        require_project(project)
        source_dir = text_value(payload, "source_dir")
        if not source_dir:
            raise ValueError("source_dir is required")
        mode = text_value(payload, "mode", "symlink")
        if mode not in {"copy", "symlink", "hardlink"}:
            raise ValueError("mode must be copy, symlink, or hardlink")
        args.extend(["import-images", project, source_dir, "--mode", mode])
        trigger = text_value(payload, "trigger")
        if trigger:
            args.extend(["--trigger", trigger])
        if bool_value(payload, "force"):
            args.append("--force")
        if bool_value(payload, "force_caption"):
            args.append("--force-caption")
    elif action == "create-caption-stubs":
        require_project(project)
        trigger = text_value(payload, "trigger")
        if not trigger:
            raise ValueError("trigger is required")
        args.extend(["create-caption-stubs", project, "--trigger", trigger])
    elif action == "generate-captions":
        require_project(project)
        args.extend(["generate-captions", project])
        caption_model = text_value(payload, "caption_model")
        if caption_model:
            args.extend(["--model", caption_model])
        trigger = text_value(payload, "trigger")
        if trigger:
            args.extend(["--trigger", trigger])
        if bool_value(payload, "force_caption"):
            args.append("--force")
        if bool_value(payload, "caption_local_only", True):
            args.append("--local-files-only")
        else:
            args.append("--allow-downloads")
    elif action == "dataset-report":
        require_project(project)
        args.extend(["dataset-report", project])
    elif action == "cache-latents":
        require_project(project)
        args.extend(["cache-latents", project])
        if bool_value(payload, "dry_run"):
            args.append("--dry-run")
    elif action == "cache-text":
        require_project(project)
        args.extend(["cache-text", project])
        if bool_value(payload, "dry_run"):
            args.append("--dry-run")
    elif action == "train":
        require_project(project)
        args.extend(["train", project])
        if not (bool_value(payload, "allow_train") and text_value(payload, "confirm") == "RUN_TRAINING"):
            args.append("--dry-run")
    elif action == "copy-to-comfy":
        require_project(project)
        args.extend(["copy-to-comfy", project])
        if bool_value(payload, "dry_run"):
            args.append("--dry-run")
    elif action == "status":
        require_project(project)
        args.extend(["status", project])
    else:
        raise ValueError(f"Unsupported action: {action}")
    return args


def require_project(project: str) -> None:
    if not project:
        raise ValueError("project is required")
    krea2_lora.validate_project_name(project)


def project_names(config: krea2_lora.AppConfig) -> list[str]:
    root = config.paths["projects_root"]
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def caption_status(caption: Path) -> tuple[str, str]:
    if not caption.exists():
        return "missing", ""
    text = caption.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return "empty", ""
    return "ready", text


def dataset_review_items(project: krea2_lora.ProjectPaths) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for image in krea2_lora.list_images(project.images):
        caption = image.with_suffix(".txt")
        status, text = caption_status(caption)
        relative = image.relative_to(project.images).as_posix()
        items.append(
            {
                "file_name": image.name,
                "relative_path": relative,
                "caption_file": caption.name,
                "caption": text,
                "caption_status": status,
                "size_bytes": image.stat().st_size,
            }
        )
    return items


def hf_model_cache_dir(model_id: str) -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "hub" / f"models--{model_id.replace('/', '--')}"


def hf_model_is_cached(model_id: str) -> bool:
    cache_dir = hf_model_cache_dir(model_id)
    snapshots = cache_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def model_inventory(config: krea2_lora.AppConfig) -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    for spec in krea2_lora.model_specs(config):
        target = Path(spec["target"])
        models.append(
            {
                "name": spec["name"],
                "label": {
                    "krea_raw": "Krea RAW",
                    "qwen_vae": "Qwen VAE",
                    "qwen_text_encoder": "Qwen text encoder",
                }.get(str(spec["name"]), str(spec["name"])),
                "repo": spec["repo"],
                "file": spec["file"],
                "path": str(target),
                "status": "installed" if target.is_file() else "missing",
                "size_bytes": target.stat().st_size if target.is_file() else None,
                "download_action": "download-models",
            }
        )

    caption_model = str(config.captioning["model"])
    cache_dir = hf_model_cache_dir(caption_model)
    models.append(
        {
            "name": "vl_caption",
            "label": "VL caption model",
            "repo": caption_model,
            "file": "Hugging Face snapshot cache",
            "path": str(cache_dir),
            "status": "installed" if hf_model_is_cached(caption_model) else "missing",
            "size_bytes": None,
            "download_action": "download-models",
        }
    )
    return models


def runtime_payload() -> dict[str, object]:
    proc_version = Path("/proc/version")
    version_text = proc_version.read_text(encoding="utf-8", errors="ignore").lower() if proc_version.is_file() else ""
    is_wsl = os.name == "posix" and ("microsoft" in version_text or "wsl" in version_text)
    is_linux = os.name == "posix"
    return {
        "platform": sys.platform,
        "os_name": os.name,
        "is_wsl": is_wsl,
        "label": "WSL/Linux" if is_wsl else ("Linux" if is_linux else "Native Windows"),
        "recommended": is_linux,
        "message": (
            "Recommended for musubi-tuner paths and shell scripts."
            if is_linux
            else "Windows is supported for the web shell, but generated musubi commands expect Linux-style paths."
        ),
    }


def safe_project_image(config: krea2_lora.AppConfig, project: str, image: str) -> Path:
    require_project(project)
    if not image:
        raise ValueError("image is required")
    paths = krea2_lora.project_paths(config, project)
    relative = Path(image.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image path must stay inside the project image folder")
    target = paths.images / relative
    if target.suffix.lower() not in krea2_lora.IMAGE_EXTENSIONS:
        raise ValueError("unsupported image type")
    return target


class Handler(BaseHTTPRequestHandler):
    state: WebState

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self.send_json(self.config_payload())
        elif parsed.path == "/api/report":
            query = parse_qs(parsed.query)
            project = query.get("project", [""])[0]
            self.send_json(self.report_payload(project))
        elif parsed.path == "/api/image":
            query = parse_qs(parsed.query)
            project = query.get("project", [""])[0]
            image = query.get("image", [""])[0]
            self.serve_project_image(project, image)
        elif parsed.path.startswith("/api/"):
            self.send_json({"error": "Unknown API route"}, HTTPStatus.NOT_FOUND)
        else:
            self.serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_json({"error": "Unknown API route"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            args = build_cli_args(payload, self.state.config)
            completed = subprocess.run(args, text=True, capture_output=True, check=False)
            self.send_json(
                {
                    "args": args[2:],
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def config_payload(self) -> dict[str, object]:
        config = krea2_lora.load_config(self.state.config)
        checks = [check.__dict__ for check in krea2_lora.env_checks(config)]
        return {
            "paths": {key: str(value) for key, value in config.paths.items()},
            "dataset": config.dataset,
            "training": config.training,
            "downloads": config.downloads,
            "captioning": config.captioning,
            "projects": project_names(config),
            "checks": checks,
            "models": model_inventory(config),
            "runtime": runtime_payload(),
        }

    def report_payload(self, project: str) -> dict[str, object]:
        require_project(project)
        config = krea2_lora.load_config(self.state.config)
        paths = krea2_lora.project_paths(config, project)
        report = krea2_lora.dataset_report_data(paths)
        report["review_items"] = dataset_review_items(paths)
        return report

    def serve_project_image(self, project: str, image: str) -> None:
        try:
            config = krea2_lora.load_config(self.state.config)
            target = safe_project_image(config, project, image)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Krea2 LoRA helper web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Keep this on localhost for normal use.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.toml path.")
    parser.add_argument("--open", action="store_true", help="Open the web app in the default browser.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    Handler.state = WebState(args.config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Krea2 LoRA helper web UI running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
