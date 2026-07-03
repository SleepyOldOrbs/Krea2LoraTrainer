#!/usr/bin/env python3
"""Localhost web UI for the Krea2 LoRA helper."""

from __future__ import annotations

import argparse
import json
import mimetypes
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
        }

    def report_payload(self, project: str) -> dict[str, object]:
        require_project(project)
        config = krea2_lora.load_config(self.state.config)
        paths = krea2_lora.project_paths(config, project)
        return krea2_lora.dataset_report_data(paths)

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
