"""Tests for reproducibility verification utilities."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evaluation.reproducibility import (
    save_config_snapshot,
    validate_experiment_metadata,
    verify_training_seed,
)


class TestReproducibility(unittest.TestCase):
    """Test reproducibility verification functions."""

    def test_verify_training_seed_with_valid_metadata(self):
        """Test verify_training_seed with valid metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create metadata.json with seed
            metadata = {
                "experiment_id": "test-exp-001",
                "timestamp": "2026-02-02T23:00:00Z",
                "mode": "debate",
                "seed": 42,
                "data_paths": {"train": "/path/to/data"},
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Verify seed
            result = verify_training_seed(exp_dir, expected_seed=42)

            self.assertTrue(result["seed_match"], "Seed should match")
            self.assertEqual(result["configured_seed"], 42)
            self.assertEqual(result["metadata_seed"], 42)
            self.assertIsNone(result["wandb_seed"])
            self.assertEqual(len(result["warnings"]), 0)

    def test_verify_training_seed_mismatch(self):
        """Test verify_training_seed detects seed mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create metadata.json with wrong seed
            metadata = {
                "experiment_id": "test-exp-002",
                "seed": 99,  # Wrong seed
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Verify seed
            result = verify_training_seed(exp_dir, expected_seed=42)

            self.assertFalse(result["seed_match"], "Seed should not match")
            self.assertEqual(result["configured_seed"], 42)
            self.assertEqual(result["metadata_seed"], 99)

    def test_verify_training_seed_missing_metadata(self):
        """Test verify_training_seed handles missing metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Don't create metadata.json
            result = verify_training_seed(exp_dir, expected_seed=42)

            self.assertFalse(result["seed_match"], "Should fail without metadata")
            self.assertIn("metadata.json not found", result["warnings"][0])

    def test_verify_training_seed_with_wandb_mock(self):
        """Test verify_training_seed with mocked W&B."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create metadata.json with W&B info
            metadata = {
                "experiment_id": "test-exp-003",
                "seed": 42,
                "wandb_run_id": "abc123",
                "wandb_entity": "test-entity",
                "wandb_project": "test-project",
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Mock wandb module - need to patch the import inside the function
            mock_wandb = MagicMock()
            mock_api = MagicMock()
            mock_run = MagicMock()
            mock_run.config = {"seed": 42}
            mock_api.run.return_value = mock_run
            mock_wandb.Api.return_value = mock_api

            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                result = verify_training_seed(exp_dir, expected_seed=42)

                self.assertTrue(result["seed_match"], "Seed should match with W&B")
                self.assertEqual(result["wandb_seed"], 42)

    def test_save_config_snapshot(self):
        """Test save_config_snapshot creates snapshot directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()

            # Create test config files
            config1 = config_dir / "config1.yaml"
            config2 = config_dir / "config2.json"

            config1.write_text("test: config1")
            config2.write_text('{"test": "config2"}')

            # Save snapshot
            snapshot_dir = save_config_snapshot(
                exp_dir,
                config_files=[config1, config2],
                extra_metadata={"run_name": "test-run"},
            )

            # Verify snapshot directory exists
            self.assertTrue(snapshot_dir.exists())
            self.assertEqual(snapshot_dir.name, "config_snapshot")

            # Verify config files copied
            self.assertTrue((snapshot_dir / "config1.yaml").exists())
            self.assertTrue((snapshot_dir / "config2.json").exists())

            # Verify content matches
            self.assertEqual((snapshot_dir / "config1.yaml").read_text(), "test: config1")
            self.assertEqual((snapshot_dir / "config2.json").read_text(), '{"test": "config2"}')

            # Verify snapshot metadata exists
            metadata_path = snapshot_dir / "snapshot_metadata.json"
            self.assertTrue(metadata_path.exists())

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.assertIn("timestamp", metadata)
            self.assertIn("config_files", metadata)
            self.assertEqual(len(metadata["config_files"]), 2)
            self.assertIn("extra", metadata)
            self.assertEqual(metadata["extra"]["run_name"], "test-run")

    def test_save_config_snapshot_handles_missing_file(self):
        """Test save_config_snapshot handles missing config files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Reference a non-existent file
            nonexistent = Path(tmpdir) / "nonexistent.yaml"

            # Should not raise, just log warning
            snapshot_dir = save_config_snapshot(exp_dir, config_files=[nonexistent])

            # Snapshot directory should still be created
            self.assertTrue(snapshot_dir.exists())

            # Metadata should still be created
            metadata_path = snapshot_dir / "snapshot_metadata.json"
            self.assertTrue(metadata_path.exists())

    def test_validate_experiment_metadata_valid(self):
        """Test validate_experiment_metadata passes with valid metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create valid metadata.json
            metadata = {
                "experiment_id": "test-exp-004",
                "timestamp": "2026-02-02T23:00:00Z",
                "mode": "debate",
                "data_paths": {
                    "train": "gs://bucket/data/train.jsonl",  # Cloud path
                },
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            result = validate_experiment_metadata(exp_dir)

            self.assertTrue(result["valid"], "Metadata should be valid")
            self.assertEqual(len(result["errors"]), 0)

    def test_validate_experiment_metadata_missing_fields(self):
        """Test validate_experiment_metadata catches missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create metadata.json missing required fields
            metadata = {
                "experiment_id": "test-exp-005",
                # Missing: timestamp, mode, data_paths
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            result = validate_experiment_metadata(exp_dir)

            self.assertFalse(result["valid"], "Metadata should be invalid")
            self.assertGreater(len(result["errors"]), 0)
            self.assertIn("timestamp", " ".join(result["errors"]))
            self.assertIn("mode", " ".join(result["errors"]))
            self.assertIn("data_paths", " ".join(result["errors"]))

    def test_validate_experiment_metadata_missing_file(self):
        """Test validate_experiment_metadata handles missing metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Don't create metadata.json
            result = validate_experiment_metadata(exp_dir)

            self.assertFalse(result["valid"], "Should fail without metadata.json")
            self.assertIn("metadata.json not found", result["errors"][0])

    def test_validate_experiment_metadata_warns_on_missing_data_file(self):
        """Test validate_experiment_metadata warns about missing local data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create metadata.json with reference to non-existent local file
            metadata = {
                "experiment_id": "test-exp-006",
                "timestamp": "2026-02-02T23:00:00Z",
                "mode": "baseline",
                "data_paths": {
                    "train": "/nonexistent/path/to/data.jsonl",  # Local path that doesn't exist
                },
            }

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            result = validate_experiment_metadata(exp_dir)

            self.assertTrue(result["valid"], "Should be valid (warnings, not errors)")
            self.assertGreater(len(result["warnings"]), 0)
            self.assertIn("Data file not found", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
