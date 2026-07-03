import sys
import unittest
from pathlib import Path

import web_app


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

    def test_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            web_app.build_cli_args({"action": "rm-rf", "project": "demo"})


if __name__ == "__main__":
    unittest.main()
