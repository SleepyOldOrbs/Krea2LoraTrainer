import os
import sys
import tempfile
import unittest
from pathlib import Path

import krea2_lora
import web_app


def toml_path(path: Path) -> str:
    return str(path).replace("\\", "/")


class WebAppTests(unittest.TestCase):
    def test_build_train_args_defaults_to_dry_run(self) -> None:
        args = web_app.build_cli_args({"action": "train", "project": "demo"})
        self.assertEqual(args[:2], [sys.executable, str(Path(web_app.ROOT / "krea2_lora.py"))])
        self.assertIn("train", args)
        self.assertIn("--dry-run", args)

    def test_build_train_args_requires_explicit_confirmation_for_real_training(self) -> None:
        args = web_app.build_cli_args(
            {
                "action": "train",
                "project": "demo",
                "allow_train": True,
                "confirm": "RUN_TRAINING",
            }
        )
        self.assertIn("train", args)
        self.assertNotIn("--dry-run", args)

    def test_download_models_passes_caption_model(self) -> None:
        args = web_app.build_cli_args(
            {
                "action": "download-models",
                "caption_model": "Salesforce/blip-image-captioning-base",
            }
        )
        self.assertEqual(args[-3:], ["download-models", "--caption-model", "Salesforce/blip-image-captioning-base"])

    def test_download_models_can_target_one_model(self) -> None:
        args = web_app.build_cli_args({"action": "download-models", "model": "qwen_vae"})
        self.assertEqual(args[-3:], ["download-models", "--model", "qwen_vae"])

    def test_import_images_args_are_whitelisted(self) -> None:
        args = web_app.build_cli_args(
            {
                "action": "import-images",
                "project": "demo",
                "source_dir": "/mnt/c/Temp/JAG",
                "mode": "symlink",
                "trigger": "jagmoon style",
            }
        )
        self.assertEqual(
            args[-7:],
            ["import-images", "demo", "/mnt/c/Temp/JAG", "--mode", "symlink", "--trigger", "jagmoon style"],
        )

    def test_generate_captions_args_are_whitelisted(self) -> None:
        args = web_app.build_cli_args(
            {
                "action": "generate-captions",
                "project": "demo",
                "trigger": "jagmoon style",
                "caption_model": "Salesforce/blip-image-captioning-base",
                "caption_local_only": True,
            }
        )
        self.assertEqual(
            args[-7:],
            [
                "generate-captions",
                "demo",
                "--model",
                "Salesforce/blip-image-captioning-base",
                "--trigger",
                "jagmoon style",
                "--local-files-only",
            ],
        )

    def test_generate_captions_can_allow_downloads(self) -> None:
        args = web_app.build_cli_args(
            {
                "action": "generate-captions",
                "project": "demo",
                "caption_local_only": False,
            }
        )
        self.assertIn("--allow-downloads", args)
        self.assertNotIn("--local-files-only", args)

    def test_dataset_review_items_include_caption_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir()
            image = images / "sample.jpg"
            image.write_bytes(b"jpg")
            image.with_suffix(".txt").write_text("jagmoon style, generated caption\n", encoding="utf-8")
            project = type("Project", (), {"images": images})()

            items = web_app.dataset_review_items(project)

        self.assertEqual(items[0]["file_name"], "sample.jpg")
        self.assertEqual(items[0]["relative_path"], "sample.jpg")
        self.assertEqual(items[0]["caption_status"], "ready")
        self.assertIn("generated caption", str(items[0]["caption"]))

    def test_safe_project_image_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = krea2_lora.AppConfig(
                paths={
                    "projects_root": root / "projects",
                    "musubi_repo": root / "musubi",
                    "musubi_venv": root / "musubi-venv",
                    "captioning_venv": root / "caption-venv",
                    "krea_raw": root / "models" / "raw.safetensors",
                    "qwen_vae": root / "models" / "vae.safetensors",
                    "qwen_text_encoder": root / "models" / "text.safetensors",
                    "comfy_lora_dir": root / "comfy",
                },
                dataset={},
                training={},
                downloads={},
                captioning={},
            )
            project_images = root / "projects" / "demo" / "images"
            project_images.mkdir(parents=True)
            (project_images / "ok.png").write_bytes(b"png")

            self.assertEqual(web_app.safe_project_image(config, "demo", "ok.png"), project_images / "ok.png")
            with self.assertRaises(ValueError):
                web_app.safe_project_image(config, "demo", "../outside.png")

    def test_save_caption_payload_writes_matching_caption_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            projects = root / "projects"
            images = projects / "demo" / "images"
            images.mkdir(parents=True)
            (images / "sample.png").write_bytes(b"png")
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'projects_root = "{toml_path(projects)}"',
                        f'musubi_repo = "{toml_path(root / "musubi")}"',
                        f'musubi_venv = "{toml_path(root / "musubi-venv")}"',
                        f'captioning_venv = "{toml_path(root / "caption-venv")}"',
                        f'krea_raw = "{toml_path(root / "models" / "raw.safetensors")}"',
                        f'qwen_vae = "{toml_path(root / "models" / "vae.safetensors")}"',
                        f'qwen_text_encoder = "{toml_path(root / "models" / "text.safetensors")}"',
                        f'comfy_lora_dir = "{toml_path(root / "comfy")}"',
                    ]
                ),
                encoding="utf-8",
            )

            result = web_app.save_caption_payload(
                {"project": "demo", "image": "sample.png", "caption": " edited caption "},
                config_path,
            )

            self.assertEqual(result["saved"]["caption_status"], "ready")
            self.assertEqual(result["saved"]["caption"], "edited caption")
            self.assertEqual((images / "sample.txt").read_text(encoding="utf-8"), "edited caption\n")

    def test_save_caption_payload_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            projects = root / "projects"
            images = projects / "demo" / "images"
            images.mkdir(parents=True)
            (images / "sample.png").write_bytes(b"png")
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'projects_root = "{toml_path(projects)}"',
                        f'musubi_repo = "{toml_path(root / "musubi")}"',
                        f'musubi_venv = "{toml_path(root / "musubi-venv")}"',
                        f'captioning_venv = "{toml_path(root / "caption-venv")}"',
                        f'krea_raw = "{toml_path(root / "models" / "raw.safetensors")}"',
                        f'qwen_vae = "{toml_path(root / "models" / "vae.safetensors")}"',
                        f'qwen_text_encoder = "{toml_path(root / "models" / "text.safetensors")}"',
                        f'comfy_lora_dir = "{toml_path(root / "comfy")}"',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                web_app.save_caption_payload(
                    {"project": "demo", "image": "../outside.png", "caption": "bad"},
                    config_path,
                )

    def test_model_inventory_reports_file_and_vl_cache_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "models" / "krea2" / "raw.safetensors"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"raw")
            cache = root / "hf" / "hub" / "models--example--vl" / "snapshots" / "abc"
            cache.mkdir(parents=True)
            old_home = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = str(root / "hf")
            try:
                config = krea2_lora.AppConfig(
                    paths={
                        "projects_root": root / "projects",
                        "musubi_repo": root / "musubi",
                        "musubi_venv": root / "musubi-venv",
                        "captioning_venv": root / "caption-venv",
                        "krea_raw": raw,
                        "qwen_vae": root / "models" / "vae.safetensors",
                        "qwen_text_encoder": root / "models" / "text.safetensors",
                        "comfy_lora_dir": root / "comfy",
                    },
                    dataset={},
                    training={},
                    downloads={
                        "krea_raw_repo": "krea/Krea-2-Raw",
                        "krea_raw_file": "raw.safetensors",
                        "qwen_vae_repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                        "qwen_vae_file": "split_files/vae/qwen_image_vae.safetensors",
                        "qwen_text_encoder_repo": "Comfy-Org/Qwen3-VL",
                        "qwen_text_encoder_file": "text_encoders/qwen3vl_4b_bf16.safetensors",
                    },
                    captioning={"model": "example/vl"},
                )

                inventory = web_app.model_inventory(config)
            finally:
                if old_home is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = old_home

        statuses = {item["name"]: item["status"] for item in inventory}
        self.assertEqual(statuses["krea_raw"], "installed")
        self.assertEqual(statuses["qwen_vae"], "missing")
        self.assertEqual(statuses["vl_caption"], "installed")

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            web_app.build_cli_args({"action": "rm-rf", "project": "demo"})


if __name__ == "__main__":
    unittest.main()
