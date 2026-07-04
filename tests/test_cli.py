import json
import os
import subprocess
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
        self.caption_models = self.models / "caption"
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
caption_models_dir = "{toml_path(self.caption_models)}"
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

    def test_import_images_preserves_nested_source_paths(self) -> None:
        source = self.root / "source"
        (source / "alpha").mkdir(parents=True)
        (source / "beta").mkdir(parents=True)
        (source / "alpha" / "same.png").write_bytes(b"alpha")
        (source / "beta" / "same.png").write_bytes(b"beta")

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

        images = self.projects / "jagmoon" / "images"
        self.assertEqual((images / "alpha" / "same.png").read_bytes(), b"alpha")
        self.assertEqual((images / "beta" / "same.png").read_bytes(), b"beta")
        self.assertEqual((images / "alpha" / "same.txt").read_text(encoding="utf-8"), "jagmoon style\n")
        self.assertEqual((images / "beta" / "same.txt").read_text(encoding="utf-8"), "jagmoon style\n")
        self.assertEqual(self.run_cli("dataset-report", "jagmoon"), 0)

    def test_dataset_report_lists_lora_outputs_newest_first(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        output = self.projects / "jagmoon" / "output"
        old = output / "jagmoon_krea2_lora_2.safetensors"
        new = output / "jagmoon_krea2_lora_10.safetensors"
        old.write_bytes(b"old")
        new.write_bytes(b"new")
        os.utime(old, (100, 100))
        os.utime(new, (200, 200))
        project = krea2_lora.project_paths(krea2_lora.load_config(self.config), "jagmoon")

        report = krea2_lora.dataset_report_data(project)

        self.assertEqual([item["name"] for item in report["lora_outputs"]], [new.name, old.name])
        self.assertEqual(report["latest_lora"], str(new))

    def test_generate_captions_dry_run_does_not_write_files(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.png"
        image.write_bytes(b"not really an image")

        self.assertEqual(self.run_cli("generate-captions", "jagmoon", "--dry-run"), 0)
        self.assertFalse(image.with_suffix(".txt").exists())

    def test_generate_captions_invokes_captioning_venv(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.png"
        image.write_bytes(b"not really an image")

        def fake_run(command, input=None, text=None, capture_output=None, check=None):
            self.assertEqual(command[0], str(self.caption_venv / "bin" / "python"))
            self.assertIn("--local-files-only", command)
            payload = json.loads(input)
            Path(payload["items"][0]["caption"]).write_text("jagmoon style, generated caption\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "VL caption generation wrote 1 captions.\n", "")

        with mock.patch.object(krea2_lora.subprocess, "run", side_effect=fake_run):
            self.assertEqual(
                self.run_cli("generate-captions", "jagmoon", "--backend", "transformers", "--trigger", "jagmoon style"),
                0,
            )
        self.assertEqual(image.with_suffix(".txt").read_text(encoding="utf-8"), "jagmoon style, generated caption\n")

    def test_generate_captions_qwen_gguf_invokes_llama_cli(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.png"
        image.write_bytes(b"not really an image")
        model = self.caption_models / "Qwen2.5-VL-7B-Captioner-Relaxed.Q6_K.gguf"
        mmproj = self.caption_models / "Qwen2.5-VL-7B-Captioner-Relaxed.mmproj-f16.gguf"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"gguf")
        mmproj.write_bytes(b"mmproj")

        def fake_run(command, text=None, capture_output=None, check=None):
            self.assertEqual(command[0], "/usr/bin/llama-qwen2vl-cli")
            self.assertIn(str(model), command)
            self.assertIn(str(mmproj), command)
            self.assertIn(str(image), command)
            self.assertIn("-ngl", command)
            self.assertIn("99", command)
            return subprocess.CompletedProcess(command, 0, "assistant: a moody landscape painting with soft light\n", "")

        with (
            mock.patch.object(krea2_lora.shutil, "which", return_value="/usr/bin/llama-qwen2vl-cli"),
            mock.patch.object(krea2_lora.subprocess, "run", side_effect=fake_run),
        ):
            self.assertEqual(
                self.run_cli("generate-captions", "jagmoon", "--trigger", "jagmoon style"),
                0,
            )
        self.assertEqual(
            image.with_suffix(".txt").read_text(encoding="utf-8"),
            "jagmoon style, a moody landscape painting with soft light\n",
        )

    def test_generate_captions_qwen_gguf_accepts_file_overrides(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.png"
        image.write_bytes(b"not really an image")
        model = self.caption_models / "custom.Q6_K.gguf"
        mmproj = self.caption_models / "custom.mmproj-f16.gguf"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"gguf")
        mmproj.write_bytes(b"mmproj")

        def fake_run(command, text=None, capture_output=None, check=None):
            self.assertIn(str(model), command)
            self.assertIn(str(mmproj), command)
            return subprocess.CompletedProcess(command, 0, "caption: soft window light over an interior photograph\n", "")

        with (
            mock.patch.object(krea2_lora.shutil, "which", return_value="/usr/bin/llama-qwen2vl-cli"),
            mock.patch.object(krea2_lora.subprocess, "run", side_effect=fake_run),
        ):
            self.assertEqual(
                self.run_cli(
                    "generate-captions",
                    "jagmoon",
                    "--model-file",
                    model.name,
                    "--mmproj-file",
                    mmproj.name,
                ),
                0,
            )

        self.assertEqual(
            image.with_suffix(".txt").read_text(encoding="utf-8"),
            "soft window light over an interior photograph\n",
        )

    def test_generate_captions_qwen_server_uses_openai_style_endpoint(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        image = self.projects / "jagmoon" / "images" / "sample.jpg"
        image.write_bytes(b"not really an image")
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "a misty nocturnal street with glowing windows",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            requests.append((request, timeout))
            return FakeResponse()

        with mock.patch.object(krea2_lora.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertEqual(
                self.run_cli(
                    "generate-captions",
                    "jagmoon",
                    "--trigger",
                    "jagmoon style",
                    "--server-url",
                    "http://127.0.0.1:8100",
                ),
                0,
            )

        self.assertEqual(requests[0][0].full_url, "http://127.0.0.1:8100/v1/chat/completions")
        payload = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertEqual(payload["max_tokens"], 180)
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")
        self.assertEqual(
            image.with_suffix(".txt").read_text(encoding="utf-8"),
            "jagmoon style, a misty nocturnal street with glowing windows\n",
        )

    def test_extract_qwen_caption_ignores_llama_logs(self) -> None:
        stdout = """
0.04.169.695 I mtmd_cli_context: chat template example:
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant

0.04.172.267 W load_hparams: Qwen-VL models require at minimum 1024 image tokens
0.05.925.032 W       For normal use cases, please use the standard llama-cli
"a luminous river landscape under a pale cloudy sky."
"""

        self.assertEqual(
            krea2_lora.extract_qwen_caption(stdout, "Describe this image."),
            "a luminous river landscape under a pale cloudy sky",
        )

    def test_cli_path_converts_windows_paths_under_posix(self) -> None:
        with mock.patch.object(krea2_lora.os, "name", "posix"):
            self.assertEqual(str(krea2_lora.cli_path(r"C:\Temp\JAG")), "/mnt/c/Temp/JAG")

    def test_list_images_uses_windows_style_natural_sort(self) -> None:
        images = self.root / "sorted-images"
        images.mkdir()
        for name in ["image10.png", "image2.png", "Image1.png", "image11.png"]:
            (images / name).write_bytes(b"image")

        ordered = [path.name for path in krea2_lora.list_images(images)]

        self.assertEqual(ordered, ["Image1.png", "image2.png", "image10.png", "image11.png"])

    def test_train_dry_run_can_skip_checks(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        self.assertEqual(self.run_cli("train", "jagmoon", "--dry-run", "--skip-checks"), 0)

    def test_train_command_accepts_run_variant_overrides(self) -> None:
        config = krea2_lora.load_config(self.config)
        project = krea2_lora.project_paths(config, "jagmoon")

        script = krea2_lora.command_train(
            config,
            project,
            run_name="dim32",
            training_overrides={
                "network_dim": 32,
                "network_alpha": 16,
                "max_train_epochs": 12,
                "learning_rate": "5e-5",
                "seed": 123,
            },
        )

        self.assertIn("--network_dim 32", script)
        self.assertIn("--max_train_epochs 12", script)
        self.assertIn("--learning_rate 5e-5", script)
        self.assertIn("--seed 123", script)
        self.assertIn("--output_name jagmoon_krea2_lora_dim32", script)

    def test_validate_env_accepts_fake_local_layout(self) -> None:
        self.assertEqual(self.run_cli("validate-env", "--create-comfy-dir"), 0)

    def test_copy_to_comfy_accepts_selected_project_output_file(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        selected = self.projects / "jagmoon" / "output" / "dim32.safetensors"
        selected.write_bytes(b"lora")

        self.assertEqual(self.run_cli("copy-to-comfy", "jagmoon", "--file", str(selected), "--dry-run"), 0)

    def test_copy_to_comfy_rejects_selected_file_outside_project_output(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"not this project")

        self.assertEqual(self.run_cli("copy-to-comfy", "jagmoon", "--file", str(outside), "--dry-run"), 1)

    def test_copy_to_comfy_rejects_non_safetensors_selected_file(self) -> None:
        self.assertEqual(self.run_cli("init-project", "jagmoon"), 0)
        selected = self.projects / "jagmoon" / "output" / "notes.txt"
        selected.write_text("not a lora", encoding="utf-8")

        self.assertEqual(self.run_cli("copy-to-comfy", "jagmoon", "--file", str(selected), "--dry-run"), 1)

    def test_download_models_verify_only_uses_existing_files(self) -> None:
        self.assertEqual(self.run_cli("download-models", "--model", "krea_raw", "--verify-only"), 0)

    def test_download_models_skips_existing_without_hf_cli(self) -> None:
        with mock.patch.object(krea2_lora.shutil, "which", return_value=None):
            self.assertEqual(self.run_cli("download-models", "--model", "krea_raw"), 0)

    def test_download_models_dry_run_for_missing_file(self) -> None:
        self.krea_raw.unlink()
        with mock.patch.object(krea2_lora.shutil, "which", return_value="/usr/bin/huggingface-cli"):
            self.assertEqual(self.run_cli("download-models", "--model", "krea_raw", "--dry-run"), 0)

    def test_download_models_dry_run_includes_vl_caption_model(self) -> None:
        self.assertEqual(self.run_cli("download-models", "--model", "vl_caption", "--dry-run"), 0)

    def test_download_vl_caption_model_invokes_captioning_venv(self) -> None:
        self.config.write_text(
            self.config.read_text(encoding="utf-8")
            + '\n[captioning]\nbackend = "transformers"\nmodel = "example/vl-model"\n',
            encoding="utf-8",
        )

        def fake_run(command, text=None, capture_output=None, check=None):
            self.assertEqual(command[0], str(self.caption_venv / "bin" / "python"))
            self.assertIn("--model", command)
            self.assertIn("example/vl-model", command)
            self.assertIn("--local-files-only", command)
            return subprocess.CompletedProcess(command, 0, "[ok] vl_caption: example/vl-model\n", "")

        with mock.patch.object(krea2_lora.subprocess, "run", side_effect=fake_run):
            self.assertEqual(
                self.run_cli(
                    "download-models",
                    "--model",
                    "vl_caption",
                    "--caption-model",
                    "example/vl-model",
                    "--verify-only",
                ),
                0,
            )

    def test_download_vl_caption_gguf_downloads_model_and_mmproj(self) -> None:
        commands = []

        def fake_run(command, text=None, capture_output=None, check=None):
            commands.append(command)
            local_dir = Path(command[command.index("--local-dir") + 1])
            hf_file = command[command.index("--filename") + 1]
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / hf_file).write_bytes(b"model")
            return subprocess.CompletedProcess(command, 0, "[ok] hf_file\n", "")

        with mock.patch.object(krea2_lora.subprocess, "run", side_effect=fake_run):
            self.assertEqual(self.run_cli("download-models", "--model", "vl_caption"), 0)

        downloaded = [command[command.index("--filename") + 1] for command in commands]
        self.assertEqual(
            downloaded,
            [
                "Qwen2.5-VL-7B-Captioner-Relaxed.Q6_K.gguf",
                "Qwen2.5-VL-7B-Captioner-Relaxed.mmproj-f16.gguf",
            ],
        )

    def test_local_dir_for_nested_hf_file(self) -> None:
        target = self.models / "qwen" / "split_files" / "vae" / "qwen_image_vae.safetensors"
        self.assertEqual(
            krea2_lora.local_dir_for_hf_file(target, "split_files/vae/qwen_image_vae.safetensors"),
            self.models / "qwen",
        )

    def test_wizard_can_quit(self) -> None:
        with mock.patch("builtins.input", side_effect=["0"]):
            self.assertEqual(self.run_cli("wizard", "jagmoon"), 0)

    def test_wizard_handles_closed_input(self) -> None:
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(self.run_cli("wizard", "jagmoon"), 0)

    def test_project_name_rejects_paths(self) -> None:
        with self.assertRaises(ValueError):
            krea2_lora.validate_project_name("../bad")


if __name__ == "__main__":
    unittest.main()
