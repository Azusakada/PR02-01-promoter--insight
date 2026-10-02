"""PR02-01 proposed adapter contracts; Protocols only, not model implementations.

Python 3.10+. Filenames and signatures define handoff behavior; all concrete
implementations must validate sample IDs and versions before training.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal

Subset = Literal["train", "val", "test"]

@dataclass(frozen=True)
class DatasetBundle:
    samples: Path
    cleaning_log: Path
    manifest: Path

@dataclass(frozen=True)
class SplitBundle:
    split_manifest: Path
    split_config: Path

@dataclass(frozen=True)
class FeatureBundle:
    matrix: Path
    sample_index: Path
    vocabulary: Path | None
    manifest: Path

@dataclass(frozen=True)
class ModelArtifact:
    checkpoint: Path
    manifest: Path
    config: Path

@dataclass(frozen=True)
class PredictionBundle:
    predictions: Path
    qc: Path
    run_manifest: Path

@dataclass(frozen=True)
class AttributionBundle:
    long_table: Path
    tensors: Path
    qc: Path
    mapping: Path
    run_manifest: Path

@dataclass(frozen=True)
class AnalysisBundle:
    results: Path
    manifests: Path
    summary: Path

class DataModule(Protocol):
    def build_dataset(self, raw_dir: Path, config_path: Path, output_dir: Path) -> DatasetBundle: ...
    def build_splits(self, samples_path: Path, config_path: Path, output_dir: Path) -> SplitBundle: ...
    def fit_label_transform(self, samples_path: Path, splits_path: Path, *, split_id: str, output_path: Path) -> Path: ...
    def export_subset(self, samples_path: Path, splits_path: Path, transform_path: Path, *, split_id: str,
                      subset: Subset, output_path: Path) -> Path: ...

class FeatureModule(Protocol):
    def build_kmer_features(self, samples_path: Path, config_path: Path, output_dir: Path) -> FeatureBundle: ...
    def build_physicochemical_features(self, samples_path: Path, config_path: Path, output_dir: Path) -> FeatureBundle: ...
    def build_thermo_inputs(self, samples_path: Path, annotations_path: Path, config_path: Path, output_dir: Path) -> Path: ...

class MLModule(Protocol):
    def fit_regressor(self, train_table: Path, val_table: Path, feature_bundle: FeatureBundle,
                      transform_path: Path, config_path: Path, output_dir: Path) -> ModelArtifact: ...

class CNNModule(Protocol):
    def fit_cnn(self, train_table: Path, val_table: Path, transform_path: Path, config_path: Path,
                output_dir: Path) -> ModelArtifact: ...
    def compute_attributions(self, sequence_table: Path, model_artifact: ModelArtifact,
                             ig_config_path: Path, output_dir: Path) -> AttributionBundle: ...

class PredictorModule(Protocol):
    def predict_sequences(self, input_table: Path, model_artifact: ModelArtifact,
                          run_config_path: Path, output_dir: Path) -> PredictionBundle: ...

class ThermoModule(Protocol):
    def check_thermo_feasibility(self, samples_path: Path, config_path: Path, output_dir: Path) -> AnalysisBundle: ...
    def run_thermo(self, inputs_path: Path, run_config_path: Path, output_dir: Path) -> PredictionBundle: ...
    def fit_calibrator(self, train_predictions_path: Path, val_predictions_path: Path,
                       config_path: Path, output_dir: Path) -> ModelArtifact: ...

class AnalysisModule(Protocol):
    def build_annotations(self, samples_path: Path, evidence_config_path: Path, output_dir: Path) -> AnalysisBundle: ...
    def run_eda(self, samples_path: Path, splits_path: Path, annotations_path: Path, config_path: Path,
                output_dir: Path) -> AnalysisBundle: ...
    def build_mutations(self, parent_table: Path, attribution_table: Path,
                        config_path: Path, output_dir: Path) -> Path: ...
    def evaluate_mutations(self, parent_predictions: Path, mutant_predictions: Path,
                           mutation_table: Path, config_path: Path, output_dir: Path) -> AnalysisBundle: ...
    def evaluate_predictions(self, predictions_path: Path, requested_ids_path: Path,
                             config_path: Path, output_dir: Path) -> AnalysisBundle: ...
