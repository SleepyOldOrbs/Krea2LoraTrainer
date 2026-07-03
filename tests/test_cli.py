import tempfile
import unittest
from pathlib import Path
from unittest import mock

import krea2_lora


def toml_path(path: Path) -> str:
    return str(path).replace("\\", "/")


class Krea2LoraCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.projects = self.root / "projects"
        self.musubi = self.root / "musubi-tuner"
        self.musubi_venv = self.root / "musubi-venv"
        self.caption_venv = self.root / "caption-venv"
        self.models = self.root / "models"
        self.comfy = self.root / "comfy" / "loras"

        for script in [
            "krea2_cache_latents.py",
            "krea2_cache_text_encoder_outputs.py",
            "krea2_train_network.py",
        ]:
            path = self.musubi / "src" / "musubi_tuner" / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test script\n", encoding="utf-8")

        for venv in [self.musubi_venv, self.caption_venv]:
            python_bin = venv / "bin" / "python"
            python_bin.parent.mkdir(parents=True, exist_ok=True)
            python_bin.write_text("", encoding="utf-8")

        self.krea_raw = self.models / "krea2" / "raw.safetensors"
        self.qwen_vae = self.models / "qwen" / "vae" / "qwen_image_vae.safetensors"
        self.qwen_text = self.models / "qwen" / "text" / "qwen3vl_4b_bf16.safetensors"
        for model in [self.krea_raw, self.qwen_vae, self.qwen_text]:
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_text("stub", encoding="utf-8")

        self.config = self.root / "config.toml"
        self.config.write_text(
            f"""
[paths]
musubi_repo = "{toml_path(self.musubi)}"
musubi_venv = "{toml_path(self.musubi_venv)}"
captioning_venv = "{toml_path(self.caption_venv)}"
projects_root = "{toml_path(self.projects)}"
krea_raw = "{toml_path(self.krea_raw)}"
qwen_vae = "{toml_path(self.qwen_vae)}"
qwen_text_encoder = "{toml_path(self.qwen_text)}"
comfy_lora_dir = "{toml_path(self.comfy)}"
""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *args: str) -> int:
        return krea2_lora.main(["--config", str(self.config), *args])

    def test_init_project_writes_expected_files(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        project = self.projects / "jagmoon"

        self.assertTrue((project / "images").is_dir())
        self.assertTrue((project / "cache").is_dir())
        self.assertTrue((project / "output").is_dir())

        dataset = (project / "config" / "dataset.toml").read_text(encoding="utf-8")
        self.assertIn("resolution = [768, 768]", dataset)
        self.assertIn('caption_extension = ".txt"', dataset)
        self.assertIn("enable_bucket = true", dataset)

        train = (project / "config" / "train_krea2.sh").read_text(encoding="utf-8")
        self.assertIn("krea2_train_network.py", train)
        self.assertIn("--network_dim 16", train)
        self.assertIn("--blocks_to_swap 20", train)

        cache_latents = (project / "config" / "cache_latents.sh").read_text(encoding="utf-8")
        self.assertIn("krea2_cache_latents.py", cache_latents)
        cache_text = (project / "config" / "cache_text.sh").read_text(encoding="utf-8")
        self.assertIn("krea2_cache_text_encoder_outputs.py", cache_text)
        copy_script = (project / "config" / "copy_latest_to_comfy.sh").read_text(encoding="utf-8")
        self.assertIn("COMFY_LORA_DIR", copy_script)

    def test_show_config_prints_resolved_settings(self) -> None:
        self.assertEqual(self.run_cli("show-config"), 0)

    def test_dataset_check_and_caption_stubs(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.png"
        image.write_bytes(b"not really an image")

        self.assertEqual(self.run_cli("check-dataset", "jagmoon"), 1)
        self.assertEqual(
            self.run_cli("create-caption-stubs", "jagmoon", "--trigger", "jagmoon style"),
            0,
        )
        self.assertEqual(self.run_cli("check-dataset", "jagmoon"), 0)
        self.assertEqual(image.with_suffix(".txt").read_text(encoding="utf-8"), "jagmoon style\n")

    def test_import_images_and_dataset_report(self) -> None:
        source = self.root / "source"
        source.mkdir()
        (source / "one.jpg").write_bytes(b"one")
        (source / "two.webp").write_bytes(b"two")

        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        self.assertEqual(
            self.run_cli(
                "import-images",
                "jagmoon",
                str(source),
                "--trigger",
                "jagmoon style",
            ),
            0,
        )
        self.assertEqual(self.run_cli("dataset-report", "jagmoon"), 0)
        self.assertEqual((self.projects / "jagmoon" / "images" / "one.txt").read_text(encoding="utf-8"), "jagmoon style\n")

    def test_cli_path_converts_windows_paths_under_posix(self) -> None:
        with mock.patch.object(krea2_lora.os, "name", "posix"):
            self.assertEqual(str(krea2_lora.cli_path(r"C:\Temp\JAG")), "/mnt/c/Temp/JAG")

    def test_train_dry_run_can_skip_checks(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        self.assertEqual(self.run_cli("train", "jagmoon", "--dry-run", "--skip-checks"), 0)

    def test_validate_env_accepts_fake_local_layout(self) -> None:
        self.assertEqual(self.run_cli("validate-env", "--create-comfy-dir"), 0)

    def test_download_models_verify_only_uses_existing_files(self) -> None:
        self.assertEqual(self.run_cli("download-models", "--verify-only"), 0)

    def test_download_models_skips_existing_without_hf_cli(self) -> None:
        with mock.patch.object(krea2_lora.shutil, "which", return_value=None):
            self.assertEqual(self.run_cli("download-models"), 0)

    def test_download_models_dry_run_for_missing_file(self) -> None:
        self.krea_raw.unlink()
        with mock.patch.object(krea2_lora.shutil, "which", return_value="/usr/bin/huggingface-cli"):
            self.assertEqual(self.run_cli("download-models", "--model", "krea_raw", "--dry-run"), 0)

    def test_local_dir_for_nested_hf_file(self) -> None:
        target = self.models / "qwen" / "split_files" / "vae" / "qwen_image_vae.safetensors"
        self.assertEqual(
            krea2_lora.local_dir_for_hf_file(target, "split_files/vae/qwen_image_vae.safetensors"),
            self.models / "qwen",
        )

    def test_project_name_rejects_paths(self) -> None:
        with self.assertRaises(ValueError):
            krea2_lora.validate_project_name("../bad")


if __name__ == "__main__":
    unittest.main()
