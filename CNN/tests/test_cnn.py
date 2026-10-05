from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr02_cnn.common import ROOT, ModelArtifact, config, ids_hash, label, relative, v, write_json, write_table
from pr02_cnn.data import load_bundle, one_hot
from pr02_cnn.metrics import compute_metrics
from pr02_cnn.model import Log10Predictor, PromoterCNN
from pr02_cnn.predict import load_model, predict_rows, prediction_row, predict_sequences
from pr02_cnn.train import fit_cnn


class EncodingTests(unittest.TestCase):
    def test_channels_positions_and_dtype(self):
        x = one_hot("ACGT" * 12 + "AC")
        self.assertEqual(tuple(x.shape), (4, 50))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(torch.equal(x.sum(dim=0), torch.ones(50)))
        self.assertTrue(torch.equal(x[:, :4], torch.eye(4)))
        self.assertEqual(float(x[1, 49]), 1)

    def test_reject_legacy_and_ambiguous_sequences(self):
        for s in ["A" * 150, "A" * 49, "N" * 50, "a" * 50, "A" * 51]:
            with self.assertRaises(ValueError):
                one_hot(s)

    def test_model_shapes_and_log10_wrapper_gradients(self):
        cfg = config(ROOT / "CNN/configs/cnn_run_config.yaml")
        model = PromoterCNN(cfg["model"]).eval()
        x = one_hot("ACGT" * 12 + "AC").unsqueeze(0).requires_grad_()
        z = Log10Predictor(model, {"min_log10": 1., "max_log10": 5.})(x)
        z.sum().backward()
        self.assertEqual(tuple(z.shape), (1,))
        self.assertEqual(tuple(x.grad.shape), (1, 4, 50))
        with self.assertRaises(ValueError):
            model(x.transpose(1, 2))

    def test_rank_ties_negative_r2_and_undefined(self):
        m = compute_metrics([1, 2, 2, 4], [2, 3, 3, 1])
        self.assertAlmostEqual(m["spearman"][0], -1/3)
        self.assertLess(compute_metrics([1, 2], [10, 20])["r2"][0], 0)
        self.assertIsNone(compute_metrics([1, 1], [1, 2])["r2"][0])
        self.assertIsNone(compute_metrics([1, 2], [3, 3])["spearman"][0])

    def test_pooling_preserves_input_gradients_and_reload(self):
        cfg = config(ROOT / "CNN/configs/cnn_run_config.yaml")["model"]
        for pooling, length in [("flatten", 50), ("adaptive_avg", 10), ("adaptive_max", 5)]:
            with self.subTest(pooling=pooling):
                options = {**cfg, "pooling": pooling, "pooled_length": length}
                model = PromoterCNN(options).eval()
                x = one_hot("ACGT" * 12 + "AC").unsqueeze(0).requires_grad_()
                y = Log10Predictor(model, {"min_log10": 1., "max_log10": 5.})(x)
                y.sum().backward()
                self.assertEqual(tuple(x.grad.shape), (1, 4, 50))
                self.assertTrue(torch.isfinite(x.grad).all())
                restored = PromoterCNN(options).eval()
                restored.load_state_dict(model.state_dict())
                self.assertTrue(torch.equal(model(x), restored(x)))

    def test_invalid_pooling_rejected_and_legacy_weights_compatible(self):
        cfg = config(ROOT / "CNN/configs/cnn_run_config.yaml")["model"]
        old = PromoterCNN(cfg).eval()
        explicit = PromoterCNN({**cfg, "pooling": "flatten", "pooled_length": 50}).eval()
        explicit.load_state_dict(old.state_dict())
        x = one_hot("ACGT" * 12 + "AC").unsqueeze(0)
        self.assertTrue(torch.equal(old(x), explicit(x)))
        for extra in [{"pooling": "unknown"}, {"pooled_length": 0}, {"pooled_length": 51},
                      {"pooled_length": True}, {"pooling": "flatten", "pooled_length": 10}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                PromoterCNN({**cfg, **extra})

    def test_transform_extrapolation_and_overflow_failed_row(self):
        transform = {"min_log10": 2., "max_log10": 4., "transform_id": "test_transform"}
        self.assertEqual(label.normalise_strength(1e5, transform), 1.5)
        self.assertEqual(label.restore_prediction(-0.5, transform), (1., 10.))
        cfg = config(ROOT / "CNN/configs/cnn_run_config.yaml")
        meta = {"split_id": "test", "seed": 0, "transform": transform, "data_version": "synthetic"}
        row = prediction_row({"sample_id": "a", "strength": 10}, 1000., meta, cfg, ROOT / "CNN/dummy.pt")
        self.assertEqual(row["prediction_status"], "failed")
        self.assertIsNone(row["predicted_value"])
        self.assertIsNone(row["predicted_value_normalized"])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Task-local temporary directory; never touches frozen team data.
        base = ROOT / "work" / "cnn_tests"
        base.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=base)
        cls.folder = Path(cls.temp.name).resolve()
        assert cls.folder.is_relative_to(base.resolve())
        rng = np.random.default_rng(1729)
        rows, splits = [], []
        for i in range(30):
            sequence = "".join(rng.choice(list("ACGT"), 50))
            z = 1.0 + (i % 17) / 10
            sid = f"synthetic_{i:03d}"
            rows.append({"dataset_id": "course_ecoli50_strength", "sample_id": sid, "source_row": i+1,
                "sequence": sequence, "sequence_length": 50, "strength": 10**z, "target_log10": z,
                "annotation_status": "missing", "schema_version": "2.0.0", "data_version": "synthetic_cnn_test_v1"})
            splits.append({"dataset_id": "course_ecoli50_strength", "split_id": "synthetic_fixed", "sample_id": sid,
                "split": "train" if i < 20 else "val" if i < 26 else "test", "group_id": None,
                "split_strategy": "random", "seed": 1729, "schema_version": "2.0.0", "data_version": "synthetic_cnn_test_v1"})
        cls.rows, cls.splits = rows, splits
        cls.samples, cls.split_file = cls.folder / "samples.tsv", cls.folder / "splits.tsv"
        write_table(cls.samples, rows, delimiter="\t")
        write_table(cls.split_file, splits, delimiter="\t")
        cfg = config(ROOT / "CNN/configs/cnn_run_config.yaml")
        cfg.update(evidence_level="smoke", run_id="synthetic_cnn_test")
        cfg["data"].update(samples=relative(cls.samples), splits=relative(cls.split_file), split_id="synthetic_fixed", expected_samples=30, transform=None)
        cfg["training"].update(max_epochs=3, patience=2, batch_size=8, num_threads=2)
        cfg["model"].update(conv_channels=[4, 8], hidden_dim=8, dropout=0.0)
        cls.cfg = cfg
        cls.cfg_path = cls.folder / "config.yaml"
        cls.cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        _, groups, transform = load_bundle(cfg)
        cls.groups, cls.transform = groups, transform
        cls.transform_path = cls.folder / "label_transform.json"
        write_json(cls.transform_path, transform)
        for subset in ["train", "val", "test"]:
            write_table(cls.folder / f"{subset}.tsv", groups[subset], delimiter="\t")
        cls.model = fit_cnn(cls.folder / "train.tsv", cls.folder / "val.tsv", cls.transform_path, cls.cfg_path, cls.folder / "model")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_training_and_prediction_contract(self):
        bundle = predict_sequences(self.folder / "val.tsv", self.model, self.cfg_path, self.folder / "pred")
        p = v.load_table("predictions", bundle.predictions)
        self.assertEqual({r["sample_id"] for r in p}, {r["sample_id"] for r in self.groups["val"]})
        self.assertTrue(all(r["prediction_status"] == "ok" for r in p))
        meta = v.read_json(self.model.manifest)
        self.assertEqual(meta["fit_subsets"], ["train"])
        self.assertEqual(meta["selection_subsets"], ["val"])
        self.assertFalse(meta["test_used"])
        self.assertTrue(meta["smoke_validation"]["save_load_equal"])

    def test_data_order_is_not_an_identity(self):
        reverse_split = self.folder / "reversed_splits.tsv"
        write_table(reverse_split, self.splits[::-1], delimiter="\t")
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["splits"] = relative(reverse_split)
        _, groups, transform = load_bundle(cfg)
        self.assertEqual(transform, self.transform)
        self.assertEqual({r["sample_id"] for r in groups["train"]}, set(self.transform["train_ids"]))

    def test_pooled_training_checkpoint_and_prediction_contract(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["run_id"] = "synthetic_pooled_test"
        cfg["model"].update(pooling="adaptive_avg", pooled_length=10)
        path = self.folder / "pooled_config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        model = fit_cnn(self.folder / "train.tsv", self.folder / "val.tsv",
                        self.transform_path, path, self.folder / "pooled_model")
        bundle = predict_sequences(self.folder / "val.tsv", model, path, self.folder / "pooled_predictions")
        rows = v.load_table("predictions", bundle.predictions)
        self.assertEqual(len(rows), len(self.groups["val"]))
        self.assertTrue(all(r["prediction_status"] == "ok" for r in rows))
        restored, metadata = load_model(model)
        self.assertEqual(metadata["model_config"]["pooled_length"], 10)
        self.assertFalse(metadata["test_used"])
        self.assertTrue(metadata["smoke_validation"]["save_load_equal"])

    def test_train_only_transform_and_wrong_hash_rejected(self):
        bad = copy.deepcopy(self.transform)
        bad["train_ids"].append(self.groups["val"][0]["sample_id"])
        bad["train_ids_sha256"] = ids_hash(bad["train_ids"])
        bad_path = self.folder / "bad_transform.json"
        write_json(bad_path, bad)
        with self.assertRaises(v.ContractError):
            load_bundle(self.cfg, bad_path)
        bad = copy.deepcopy(self.transform)
        bad["samples_sha256"] = "0"*64
        write_json(bad_path, bad)
        with self.assertRaises(v.ContractError):
            load_bundle(self.cfg, bad_path)

    def test_failure_coverage_and_order(self):
        model, metadata = load_model(self.model)
        rows = copy.deepcopy(self.groups["val"])
        rows[1]["sequence"] = "N"*50
        predictions = predict_rows(rows, model, metadata, self.cfg, self.model.checkpoint)
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual([r["sample_id"] for r in predictions], [r["sample_id"] for r in rows])
        self.assertEqual(predictions[1]["prediction_status"], "failed")
        self.assertIsNone(predictions[1]["predicted_value"])

    def test_partial_prediction_bundle_preserves_failed_request(self):
        rows = copy.deepcopy(self.groups["val"])
        rows[2]["sequence"] = "N"*50
        path = self.folder / "malformed_val.tsv"
        write_table(path, rows, delimiter="\t")
        bundle = predict_sequences(path, self.model, self.cfg_path, self.folder / "partial")
        predictions = v.load_table("predictions", bundle.predictions)
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(predictions[2]["prediction_status"], "failed")
        self.assertEqual(v.read_json(bundle.run_manifest)["execution_status"], "partial")

    def test_unknown_and_duplicate_request_ids_rejected(self):
        for variant in ["unknown", "duplicate"]:
            rows = copy.deepcopy(self.groups["val"])
            rows[0]["sample_id"] = "unknown" if variant == "unknown" else rows[1]["sample_id"]
            path = self.folder / f"{variant}_val.tsv"
            write_table(path, rows, delimiter="\t")
            with self.assertRaises(v.ContractError):
                predict_sequences(path, self.model, self.cfg_path, self.folder / variant)

    def test_prediction_config_model_conflict_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["hidden_dim"] += 1
        path = self.folder / "wrong_model_config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with self.assertRaises(v.ContractError):
            predict_sequences(self.folder / "val.tsv", self.model, path, self.folder / "wrong_model_config")

    def test_checkpoint_tampering_and_output_overwrite_rejected(self):
        bad_checkpoint = self.folder / "tampered.pt"
        bad_checkpoint.write_bytes(self.model.checkpoint.read_bytes() + b"tamper")
        with self.assertRaises(v.ContractError):
            load_model(ModelArtifact(bad_checkpoint, self.model.manifest, self.model.config))
        with self.assertRaises(v.ContractError):
            fit_cnn(self.folder / "train.tsv", self.folder / "val.tsv", self.transform_path, self.cfg_path, self.folder / "model")

    def test_test_requires_explicit_frozen_plan_opt_in(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["prediction"]["subset"] = "test"
        path = self.folder / "test_config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with self.assertRaises(v.ContractError):
            predict_sequences(self.folder / "test.tsv", self.model, path, self.folder / "illegal_test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
