"""SKU-110K dataset: download, and load images + ground-truth boxes per split.

SKU-110K (Trax Retail, CVPR'19) is ~11.7k densely packed shelf photos with a single class
("object"), ~147 boxes per image on average. Splits: train 8,219 / val 588 / test 2,936.

How we use it:
  * ``test`` -- the leaderboard split. Runs sample a fixed, seeded subset so every
    approach/model is scored on exactly the same images.
  * ``val``  -- for iterating on prompts / parameters without touching test.
  * ``train`` -- available to approaches that need examples (few-shot, fine-tuning).

Layout after ``shelf-bench download``::

    data/SKU110K_fixed/images/{train,val,test}_*.jpg
    data/SKU110K_fixed/annotations/annotations_{train,val,test}.csv
"""

from __future__ import annotations

import csv
import io
import os
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
LOCAL_ROOT = "data/SKU110K_fixed"


def _default_root() -> str:
    """$SHELF_BENCH_DATA, else a local download if present, else the shared GCS copy."""
    if os.environ.get("SHELF_BENCH_DATA"):
        return os.environ["SHELF_BENCH_DATA"]
    if Path(LOCAL_ROOT, "annotations").is_dir():
        return LOCAL_ROOT
    from utils.llm import load_config

    return load_config().get("gcp", {}).get("data") or LOCAL_ROOT


# Local folder or gs:// URI. Cloud Run jobs set SHELF_BENCH_DATA=gs://<bucket>/SKU110K_fixed.
DEFAULT_ROOT = _default_root()
SPLITS = ("train", "val", "test")

# A box is (x1, y1, x2, y2) in absolute pixels of the original image.
Box = tuple[float, float, float, float]


@dataclass
class Sample:
    image_id: str  # file name, e.g. "test_0.jpg"
    path: str      # local path or gs:// URI
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


# ---- storage: local paths and gs:// URIs behave the same ----------------------------------------

@lru_cache(maxsize=1)
def _gcs():
    from google.cloud import storage

    return storage.Client()


def _split_gs(uri: str) -> tuple[str, str]:
    bucket, _, name = uri[len("gs://"):].partition("/")
    return bucket, name


def read_bytes(path: str) -> bytes:
    if str(path).startswith("gs://"):
        bucket, name = _split_gs(str(path))
        return _gcs().bucket(bucket).blob(name).download_as_bytes()
    return Path(path).read_bytes()


def join(root: str, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *parts]) if str(root).startswith("gs://") \
        else str(Path(root, *parts))


# ---- dataset ------------------------------------------------------------------------------------

def download(root: str = LOCAL_ROOT, keep_archive: bool = False) -> Path:
    """Download (resumable) and extract the full dataset (~12 GB archive)."""
    root = Path(root)
    if (root / "annotations" / "annotations_test.csv").exists():
        print(f"SKU-110K already present at {root}")
        return root
    archive = root.parent / "raw" / "SKU110K_fixed.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} -> {archive} (resumable, ~12 GB)")
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(archive), DATA_URL],
        check=True,
    )
    print(f"Extracting to {root.parent} ...")
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(root.parent)], check=True)
    if not keep_archive:
        archive.unlink()
    print(f"Done: {len(list((root / 'images').glob('*.jpg')))} images")
    return root


def upload(local_root: str, gcs_root: str, workers: int = 32) -> None:
    """Copy the extracted dataset to GCS. Resumable: files already there are skipped."""
    from google.cloud.storage import transfer_manager

    local_root = Path(local_root)
    bucket_name, prefix = _split_gs(gcs_root.rstrip("/"))
    bucket = _gcs().bucket(bucket_name)
    existing = {b.name for b in _gcs().list_blobs(bucket_name, prefix=prefix + "/")}
    files = [str(p.relative_to(local_root)) for p in local_root.rglob("*") if p.is_file()]
    todo = [f for f in files if f"{prefix}/{f}" not in existing]
    print(f"{len(files)} files, {len(files) - len(todo)} already in {gcs_root}, uploading {len(todo)}")
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        results = transfer_manager.upload_many_from_filenames(
            bucket, chunk, source_directory=str(local_root), blob_name_prefix=prefix + "/",
            max_workers=workers, worker_type=transfer_manager.THREAD,
        )
        failed = [f for f, r in zip(chunk, results, strict=True) if isinstance(r, Exception)]
        if failed:
            raise RuntimeError(f"{len(failed)} uploads failed, e.g. {failed[0]}: re-run to resume")
        print(f"  {min(i + 500, len(todo))}/{len(todo)}")
    print("Upload complete.")


@lru_cache(maxsize=8)
def load_split(split: str, root: str = DEFAULT_ROOT) -> dict[str, Sample]:
    """Parse ``annotations_<split>.csv`` into ``{image_id: Sample}``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = str(root)
    csv_path = join(root, "annotations", f"annotations_{split}.csv")
    try:
        text = read_bytes(csv_path).decode()
    except FileNotFoundError:
        raise FileNotFoundError(f"{csv_path} not found. Run `shelf-bench download` first.") from None
    samples: dict[str, Sample] = {}
    grouped: dict[str, list[Box]] = defaultdict(list)
    # columns: image_name, x1, y1, x2, y2, class, image_width, image_height
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 8:
            continue
        name = row[0]
        grouped[name].append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))
        if name not in samples:
            samples[name] = Sample(name, join(root, "images", name), int(row[6]), int(row[7]))
    for name, s in samples.items():
        s.boxes = grouped[name]
    return samples


def sample_images(
    split: str = "test", limit: int = 25, seed: int = 0, root: str = DEFAULT_ROOT
) -> list[Sample]:
    """A deterministic subset of a split (``limit <= 0`` means the whole split).

    Same (split, limit, seed) -> same images, so runs are directly comparable, whether the
    data is read locally or from GCS.
    """
    all_samples = sorted(load_split(split, str(root)).values(), key=lambda s: s.image_id)
    if not str(root).startswith("gs://"):
        all_samples = [s for s in all_samples if Path(s.path).exists()]
    if limit <= 0 or limit >= len(all_samples):
        return all_samples
    return random.Random(seed).sample(all_samples, limit)
