from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyPinTests(unittest.TestCase):
    def test_python_version_is_pinned(self):
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.12.3")

    def test_cluster_ml_dependencies_are_pinned(self):
        requirements = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        expected = {
            "torch==2.7.1",
            "tabpfn-time-series==1.1.0",
            "tabpfn==8.0.3",
            "chronos-forecasting==2.2.2",
            "autogluon.timeseries==1.5.0",
        }
        self.assertTrue(expected.issubset(requirements))
        self.assertTrue(all("==" in requirement for requirement in requirements))

    def test_cluster_cuda_runtime_is_documented(self):
        requirements = (ROOT / "requirements.txt").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn("CUDA 12.6.3", requirements)
        self.assertIn("CUDA 12.6.3", readme)
        self.assertIn("https://download.pytorch.org/whl/cu126", readme)

    def test_timesfm_is_not_an_environment_dependency(self):
        requirements = (ROOT / "requirements.txt").read_text().lower()
        self.assertNotIn("timesfm", requirements)


if __name__ == "__main__":
    unittest.main()
