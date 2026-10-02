from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import ids_hash, label, resolve, v

CHANNELS = "ACGT"


def one_hot(sequence: str) -> torch.Tensor:
    """Position i is the provided sequence's local 0-based position; no strand flip."""
    if not isinstance(sequence, str) or len(sequence) != 50 or any(base not in CHANNELS for base in sequence):
        raise ValueError("sequence必须是大写50bp A/C/G/T，不截取、不补齐、不猜方向")
    result = torch.zeros((4, 50), dtype=torch.float32)
    for position, base in enumerate(sequence):
        result[CHANNELS.index(base), position] = 1.0
    return result


class PromoterDataset(Dataset):
    def __init__(self, rows: list[dict], transform: dict):
        self.ids = [r["sample_id"] for r in rows]
        self.x = torch.stack([one_hot(r["sequence"]) for r in rows])
        self.y = torch.tensor([label.normalise_strength(r["strength"], transform) for r in rows], dtype=torch.float32)
        self.rows = rows

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.ids[index]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    import random
    random.seed(seed)


def make_loader(dataset, cfg, training=False):
    t = cfg["training"]
    generator = torch.Generator().manual_seed(t["seed"])
    return DataLoader(dataset, batch_size=t["batch_size"], shuffle=training, drop_last=False,
                      num_workers=t["num_workers"], generator=generator, worker_init_fn=seed_worker)


def load_bundle(cfg, transform_path=None):
    data = cfg["data"]
    samples_path, splits_path = resolve(data["samples"]), resolve(data["splits"])
    v.bundle_check(samples_path, splits_path)
    rows = v.load_table("dataset", samples_path)
    splits = v.load_table("splits", splits_path)
    if data.get("expected_samples") is not None:
        v.require(len(rows) == data["expected_samples"], "数据条数与配置不一致")
    selected = [r for r in splits if r["split_id"] == data["split_id"]]
    v.require(selected, "配置split_id不在固定划分文件中")
    by_split = {r["sample_id"]: r["split"] for r in selected}
    groups = {subset: [r for r in rows if by_split[r["sample_id"]] == subset] for subset in ["train", "val", "test"]}
    v.require(all(groups.values()), "train/val/test必须非空")
    # The supplied helper verifies full master SHA256 and sorted training IDs.
    transform_path = transform_path or data.get("transform")
    if transform_path:
        transform = v.read_json(resolve(transform_path))
        v.scale_check(transform, rows, splits, samples_path)
        v.require(transform["split_id"] == data["split_id"], "transform split_id不一致")
    else:
        transform = label.compute_label_transform(samples_path, splits_path, data["split_id"])
    if cfg["evidence_level"] == "preliminary":
        v.require("synthetic" not in rows[0]["data_version"].lower(), "合成数据不能写preliminary")
    return rows, groups, transform


def verify_subset(path: Path, expected: list[dict], name: str):
    rows = v.load_table("dataset", path)
    want = {r["sample_id"]: r for r in expected}
    got = {r["sample_id"]: r for r in rows}
    v.require(want == got, f"{name}表与冻结主表/split不一致，不能私自换行/标签/子集")
    return rows


def subset_identity(rows):
    ids = sorted(r["sample_id"] for r in rows)
    return {"ids": ids, "ids_sha256": ids_hash(ids), "n_samples": len(ids)}
