import os
import tempfile
import unittest
from unittest.mock import call, patch

import main


class TestMainWebBuild(unittest.TestCase):
    @patch.dict(os.environ, {"PROBINGRLM_SKIP_FRONTEND_BUILD": "1"}, clear=False)
    @patch("main.subprocess.run")
    def test_build_frontend_skips_when_env_flag_set(self, mock_run):
        main._build_frontend_assets()
        mock_run.assert_not_called()

    def test_build_frontend_fails_when_npm_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch("main.FRONTEND_DIR", tmp_dir),
                patch("main.shutil.which", return_value=None),
            ):
                with self.assertRaises(RuntimeError):
                    main._build_frontend_assets()

    def test_build_frontend_installs_when_node_modules_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = os.path.join(tmp_dir, "package-lock.json")
            with open(lock_path, "w", encoding="utf-8") as file_obj:
                file_obj.write("{}")

            with (
                patch("main.FRONTEND_DIR", tmp_dir),
                patch("main.shutil.which", return_value="/usr/bin/npm"),
                patch("main.subprocess.run") as mock_run,
            ):
                main._build_frontend_assets()
                self.assertEqual(
                    mock_run.call_args_list,
                    [
                        call(["/usr/bin/npm", "ci"], cwd=tmp_dir, check=True),
                        call(["/usr/bin/npm", "run", "build"], cwd=tmp_dir, check=True),
                    ],
                )

    def test_build_frontend_skips_install_when_node_modules_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "node_modules"), exist_ok=True)

            with (
                patch("main.FRONTEND_DIR", tmp_dir),
                patch("main.shutil.which", return_value="/usr/bin/npm"),
                patch("main.subprocess.run") as mock_run,
            ):
                main._build_frontend_assets()
                mock_run.assert_called_once_with(
                    ["/usr/bin/npm", "run", "build"], cwd=tmp_dir, check=True
                )


if __name__ == "__main__":
    unittest.main()
