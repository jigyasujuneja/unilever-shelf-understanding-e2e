"""Datasets: SKU-110K shelf photos (boxes only) and a small labelled products set (see below).

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
    labels: list[dict] = field(default_factory=list)  # per box; empty for SKU-110K ("object")


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
    split: str = "test", limit: int = 25, seed: int = 0, root: str = DEFAULT_ROOT,
    name: str = "sku110k",
) -> list[Sample]:
    """A deterministic subset of a split (``limit <= 0`` means the whole split).

    Same (split, limit, seed) -> same images, so runs are directly comparable, whether the
    data is read locally or from GCS. ``name`` is the dataset: ``sku110k``, ``products``, ``rpc``
    or ``shelves`` (same layout as ``rpc``).
    """
    if name == "products" and split != "test":
        raise ValueError("the products dataset only has a test split")
    loaded = (load_products(str(root)) if name == "products" else
              load_rpc(split, str(root)) if name in ("rpc", "shelves") else load_split(split, str(root)))
    all_samples = sorted(loaded.values(), key=lambda s: s.image_id)
    if not str(root).startswith("gs://"):
        all_samples = [s for s in all_samples if Path(s.path).exists()]
    if limit <= 0 or limit >= len(all_samples):
        return all_samples
    return random.Random(seed).sample(all_samples, limit)


# ---- labelled products: product photos with brand / product / category labels ------------------
#
# 25 single-product photos plus a closed catalog of 184 products, from the Hugging Face dataset
# kierth/retail-products-philippines (human labels: brand, product name, size, category).
# Unlike SKU-110K these have real product labels, so classification / retrieval can be scored.
# They're catalog-style product photos, not shelf crops, and the brands are the Philippine
# market's (Knorr, Lady's Choice, Dove, Sunsilk, ... and non-Unilever items).
#
#     data/labeled_retail_benchmarks/labeled_fmcg_classification_benchmark.json
#     data/labeled_retail_benchmarks/images/labeled_sku_*.jpg

PRODUCTS_LOCAL = "data/labeled_retail_benchmarks"
PRODUCTS_JSON = "labeled_fmcg_classification_benchmark.json"
PRODUCT_FIELDS = ("product", "brand", "category")  # scored per image, "product" ranks


def products_root() -> str:
    """$SHELF_BENCH_PRODUCTS, else the copy in the repo, else the shared GCS copy."""
    if os.environ.get("SHELF_BENCH_PRODUCTS"):
        return os.environ["SHELF_BENCH_PRODUCTS"]
    if Path(PRODUCTS_LOCAL, PRODUCTS_JSON).exists():
        return PRODUCTS_LOCAL
    from utils.llm import load_config

    return load_config().get("gcp", {}).get("products") or PRODUCTS_LOCAL


@lru_cache(maxsize=4)
def _products_json(root: str) -> dict:
    import json

    return json.loads(read_bytes(join(root, PRODUCTS_JSON)))


def _product(row: dict) -> dict:
    return {"sku_id": row["id"], "product": row["product_name"], "brand": row["brand"],
            "category": row["category"], "size": row.get("extracted_size", "")}


def catalog(root: str | None = None, name: str = "products") -> dict[int, dict]:
    """The closed product catalog approaches choose from: ``{sku_id: product}``."""
    if name in ("rpc", "shelves"):
        return rpc_catalog(str(root or data_root(name)))
    rows = _products_json(str(root or products_root()))["full_labeled_fmcg_catalog"]
    return {r["id"]: _product(r) for r in rows}


@lru_cache(maxsize=4)
def load_products(root: str) -> dict[str, Sample]:
    """``{image_id: Sample}``; one whole-image box per photo, labelled with its catalog product."""
    from PIL import Image

    out: dict[str, Sample] = {}
    for r in _products_json(root)["downloaded_unilever_and_competitor_samples"]:
        name = Path(r["local_image_path"]).name
        path = join(root, "images", name)
        with Image.open(io.BytesIO(read_bytes(path))) as im:
            w, h = im.size
        out[name] = Sample(name, path, w, h, boxes=[(0.0, 0.0, float(w), float(h))],
                           labels=[_product(r)])
    return out


# ---- RPC: checkout photos with every product boxed and identified, plus reference photos ------
#
# Retail Product Checkout (Wei et al. 2019, arXiv:1901.07249; CC BY-NC-SA), from the Hugging Face
# copy benjamintli/retail-product-checkout. 200 products (Chinese supermarket items, 17
# categories). Two kinds of photos:
#   * train: each product alone on a turntable (~270 views per product) -> our reference gallery
#   * test / validation: several products on a checkout counter, every box labelled with its
#     product -> what approaches identify (Retrieval: given the boxes; End-to-end: find them too)
# Products have ids and categories but no names, so identification is image-to-image.
# ``shelf-bench rpc`` downloads a fixed subset (below) once and uploads it to ``gcp.rpc``:
#
#     data/rpc/rpc.json                    classes, gallery files, test / val images with boxes
#     data/rpc/gallery/<id>_<k>.jpg        REFS_PER_PRODUCT cropped reference views per product
#     data/rpc/images/{test,val}_*.jpg     checkout photos (longest side 1400 px)

RPC_LOCAL = "data/rpc"
RPC_JSON = "rpc.json"
RPC_HF = "benjamintli/retail-product-checkout"
RPC_FIELDS = ("product", "category")
RPC_SUBSETS = {"test": ("test", 100), "val": ("validation", 20)}  # our split -> (HF split, n)
REFS_PER_PRODUCT = 4


def data_root(name: str) -> str:
    """Where a dataset lives: $SHELF_BENCH_<NAME>, else a local copy, else its GCS copy."""
    if name == "sku110k":
        return DEFAULT_ROOT
    if name == "products":
        return products_root()
    local = SHELVES_LOCAL if name == "shelves" else RPC_LOCAL
    if os.environ.get(f"SHELF_BENCH_{name.upper()}"):
        return os.environ[f"SHELF_BENCH_{name.upper()}"]
    if Path(local, RPC_JSON).exists():
        return local
    from utils.llm import load_config

    return load_config().get("gcp", {}).get(name) or local


@lru_cache(maxsize=4)
def _rpc_json(root: str) -> dict:
    import json

    try:
        return json.loads(read_bytes(join(root, RPC_JSON)))
    except FileNotFoundError:
        raise FileNotFoundError(f"{join(root, RPC_JSON)} not found. Run `shelf-bench rpc` first.") from None


def rpc_catalog(root: str) -> dict[int, dict]:
    return {int(k): v for k, v in _rpc_json(root)["classes"].items()}


def rpc_gallery(root: str) -> dict[int, list[str]]:
    """``{product id: [reference photo path, ...]}``: the product alone, from several angles."""
    return {int(k): [join(root, f) for f in v] for k, v in _rpc_json(root)["gallery"].items()}


@lru_cache(maxsize=8)
def load_rpc(split: str, root: str) -> dict[str, Sample]:
    if split not in RPC_SUBSETS:
        raise ValueError(f"the rpc dataset has splits {tuple(RPC_SUBSETS)}, got {split!r}")
    cat = rpc_catalog(root)
    return {r["image"]: Sample(r["image"], join(root, "images", r["image"]), r["width"], r["height"],
                               boxes=[tuple(b) for b in r["boxes"]],
                               labels=[cat[c] for c in r["products"]])
            for r in _rpc_json(root)["splits"][split]}


RPC_CACHE = Path("data/rpc_cache")  # API pages, so an interrupted `shelf-bench rpc` resumes


def _hf(split: str, offset: int, length: int) -> dict:
    """One page of the Hugging Face dataset viewer API: ``rows`` (with signed image URLs).
    Cached on disk; rate limits (429) are waited out."""
    import json
    import time
    import urllib.error
    import urllib.request

    cached = RPC_CACHE / f"{split}_{offset}_{length}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    url = (f"https://datasets-server.huggingface.co/rows?dataset={RPC_HF}&config=default"
           f"&split={split}&offset={offset}&length={length}")
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 9:
                raise
            time.sleep(min(60, 5 * (attempt + 1)) if getattr(e, "code", None) == 429 else 2 ** attempt)
    RPC_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(body)
    return json.loads(body)


def _fetch_image(url: str):
    import time
    import urllib.error
    import urllib.request

    from PIL import Image

    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except (urllib.error.URLError, TimeoutError):
            if attempt == 7:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))


def _rpc_categories() -> dict[int, str]:
    """RPC product id -> category, from the dataset's class names ("<id>_<category>");
    Hugging Face label index i is RPC product id i + 1."""
    import json
    import urllib.request

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{RPC_HF}", timeout=60) as r:
        feats = json.loads(r.read())["cardData"]["dataset_info"]["features"]
    objects = next(f for f in feats if f["name"] == "objects")["struct"]
    names = next(f for f in objects if f["name"] == "category")["list"]["class_label"]["names"]
    return {int(i) + 1: n.split("_", 1)[1] for i, n in names.items()}


def prepare_rpc(out: str = RPC_LOCAL, seed: int = 0, log=print) -> Path:
    """Download the fixed RPC subset described above (~40 MB). Deterministic for a seed."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    out_p = Path(out)
    (out_p / "gallery").mkdir(parents=True, exist_ok=True)
    (out_p / "images").mkdir(parents=True, exist_ok=True)
    categories = _rpc_categories()

    # Reference gallery: the train photos are grouped by product (one turntable sequence each);
    # take REFS_PER_PRODUCT evenly spaced views of each product, cropped to its box.
    total = _hf("train", 0, 1)["num_rows_total"]
    log(f"Scanning {total} RPC reference photos for {REFS_PER_PRODUCT} views per product ...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [r["row"] for page in pool.map(lambda o: _hf("train", o, 100)["rows"], range(0, total, 100))
                for r in page]
    product = [int(r["objects"]["category"][0]) + 1 for r in rows]
    count: dict[int, int] = defaultdict(int)
    for c in product:
        count[c] += 1
    seen: dict[int, int] = defaultdict(int)
    picks = []  # (product, k, row)
    for c, row in zip(product, rows, strict=True):
        i, step = seen[c], max(1, count[c] // REFS_PER_PRODUCT)
        seen[c] += 1
        if i % step == 0 and i // step < REFS_PER_PRODUCT:
            picks.append((c, i // step, row))

    def ref(p) -> str:
        c, k, row = p
        name = f"gallery/{c}_{k}.jpg"
        if not (out_p / name).exists():
            x, y, w, h = row["objects"]["bbox"][0]
            im = _fetch_image(row["image"]["src"])
            im = im.crop((max(0, x - w * .05), max(0, y - h * .05),
                          min(im.width, x + w * 1.05), min(im.height, y + h * 1.05)))
            im.thumbnail((384, 384))
            im.save(out_p / name, quality=90)
        return name

    with ThreadPoolExecutor(max_workers=8) as pool:
        names = list(pool.map(ref, picks))
    gallery: dict[str, list[str]] = defaultdict(list)
    for (c, _, _), name in zip(picks, names, strict=True):  # rows are in turntable order
        gallery[str(c)].append(name)
    classes = {str(c): {"sku_id": c, "product": f"RPC #{c} ({categories[c]})", "category": categories[c]}
               for c in sorted(count)}
    log(f"  {len(names)} reference photos for {len(gallery)} products")

    # Checkout photos: a seeded subset of each split, boxes scaled with the image.
    splits: dict[str, list[dict]] = {}
    for ours, (hf_split, n) in RPC_SUBSETS.items():
        size = _hf(hf_split, 0, 1)["num_rows_total"]
        offsets = sorted(random.Random(seed).sample(range(size), n))

        def one(o: int, ours=ours, hf_split=hf_split) -> dict:
            row = _hf(hf_split, o, 1)["rows"][0]["row"]
            name = f"{ours}_{o:05d}.jpg"
            im = _fetch_image(row["image"]["src"])
            k = min(1.0, 1400 / max(im.size))
            if k < 1:
                im = im.resize((round(im.width * k), round(im.height * k)))
            im.save(out_p / "images" / name, quality=90)
            return {"image": name, "width": im.width, "height": im.height,
                    "boxes": [[round(x * k, 1), round(y * k, 1), round((x + w) * k, 1),
                               round((y + h) * k, 1)] for x, y, w, h in row["objects"]["bbox"]],
                    "products": [int(c) + 1 for c in row["objects"]["category"]]}

        with ThreadPoolExecutor(max_workers=8) as pool:
            splits[ours] = list(pool.map(one, offsets))
        log(f"  {ours}: {n} checkout photos, {sum(len(r['boxes']) for r in splits[ours])} products boxed")
    (out_p / RPC_JSON).write_text(json.dumps(
        {"source": f"huggingface.co/datasets/{RPC_HF} (CC BY-NC-SA 2.0)", "seed": seed,
         "classes": classes, "gallery": gallery, "splits": splits}, indent=1))
    log(f"Done: {out_p}")
    return out_p


# ---- shelves: HoloSelecta vending-machine shelves, every product boxed with its GTIN ---------
#
# HoloSelecta (Fuchs, Grundmann & Fleisch, IoT 2019; CC BY 4.0): 295 photos of Selecta vending
# machines in Zurich (phone and HoloLens), 10k boxes, 109 products (snacks, drinks), each labelled
# ``<name>__<size>__<GTIN>``. Real shelves with product identities, laid out in rows. There are no
# catalog photos, so the reference gallery is cropped from photos that are not scored.
# Photos come in bursts of the same machine seconds apart, so splits are made by session
# (photos less than SESSION_GAP_S apart), not by photo: a scored machine is never in the gallery.
# ``shelf-bench shelves RAW_DIR`` (RAW_DIR = the dataset's Drive folder: *.jpg + Pascal VOC *.xml)
# writes the same layout as RPC and uploads it to ``gcp.shelves``:
#
#     data/shelves/rpc.json                 classes, gallery files, test / val images with boxes
#     data/shelves/gallery/<id>_<k>.jpg     up to REFS_PER_PRODUCT crops per product
#     data/shelves/images/*.jpg             scored photos (longest side 1400 px)

SHELVES_LOCAL = "data/shelves"
SHELVES_SOURCE = "github.com/tobiagru/ObjectDetectionGroceryProducts (HoloSelecta, CC BY 4.0)"
SHELF_FIELDS = ("product",)  # no categories in the labels
SESSION_GAP_S = 300
SHELF_SPLIT_SHARE = {"test": 0.4, "val": 0.1}  # of photos; the rest is only used for the gallery


def _voc(path: Path) -> tuple[str, list[tuple[str, tuple[float, float, float, float]]]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        objs.append((o.findtext("name").strip(), tuple(float(bb.findtext(k))
                                                       for k in ("xmin", "ymin", "xmax", "ymax"))))
    return root.findtext("filename").strip(), objs


def _shelf_product(label: str, sku_id: int) -> dict:
    """``fuse_peach__50__5449000236623`` -> product "fuse peach 50", GTIN 5449000236623."""
    *name, gtin = label.split("__")
    words = " ".join(" ".join(name).replace("_", " ").split())
    return {"sku_id": sku_id, "product": words, "gtin": gtin}


def shelf_sessions(names: list[str]) -> list[list[str]]:
    """Group photos into bursts of the same machine. Names carry a time (``[IMG_]YYYYMMDD_HHMMSS``)
    or a camera counter (``DSC01627``, counted as a minute apart per step): a new session starts
    after a gap of more than SESSION_GAP_S."""
    import re
    from datetime import datetime

    def when(n: str) -> tuple[str, float]:
        if m := re.search(r"(\d{8}_\d{6})", n):
            return "time", datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
        if m := re.search(r"DSC(\d+)", n):
            return "dsc", int(m.group(1)) * 60.0
        raise ValueError(f"can't tell when {n} was taken")

    sessions: list[list[str]] = []
    last = None
    for (kind, t), n in sorted((when(n), n) for n in names):
        if last is None or kind != last[0] or t - last[1] > SESSION_GAP_S:
            sessions.append([])
        sessions[-1].append(n)
        last = (kind, t)
    return sessions


def prepare_shelves(raw: str, out: str = SHELVES_LOCAL, seed: int = 0, log=print) -> Path:
    """Build the shelves dataset from HoloSelecta's photos + VOC files. Deterministic for a seed."""
    import json

    from PIL import Image

    raw_p, out_p = Path(raw), Path(out)
    (out_p / "gallery").mkdir(parents=True, exist_ok=True)
    (out_p / "images").mkdir(parents=True, exist_ok=True)
    photos = {}
    for x in sorted(raw_p.glob("*.xml")):
        name, objs = _voc(x)
        if objs and (raw_p / name).exists():
            photos[name] = objs
    labels = sorted({lab for objs in photos.values() for lab, _ in objs})
    ids = {lab: i + 1 for i, lab in enumerate(labels)}

    sessions = shelf_sessions(list(photos))
    random.Random(seed).shuffle(sessions)
    split_of: dict[str, str] = {}
    quota = {s: share * len(photos) for s, share in SHELF_SPLIT_SHARE.items()}
    for sess in sessions:
        split = next((s for s in quota if quota[s] > 0), "gallery")
        quota[split] = quota.get(split, 0) - len(sess)
        split_of.update({n: split for n in sess})

    # Gallery: crops from gallery photos, spread over as many photos as possible.
    crops: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for n in sorted(photos):
        if split_of[n] == "gallery":
            seen = set()
            for lab, box in photos[n]:
                if lab not in seen:  # one crop per product per photo
                    seen.add(lab)
                    crops[lab].append((n, box))
    gallery: dict[str, list[str]] = {}
    for lab, cs in crops.items():
        step = max(1, len(cs) // REFS_PER_PRODUCT)
        files = []
        for k, (n, (x1, y1, x2, y2)) in enumerate(cs[::step][:REFS_PER_PRODUCT]):
            with Image.open(raw_p / n) as im:
                w, h = x2 - x1, y2 - y1
                c = im.convert("RGB").crop((max(0, x1 - w * .05), max(0, y1 - h * .05),
                                            min(im.width, x2 + w * .05), min(im.height, y2 + h * .05)))
            c.thumbnail((384, 384))
            f = f"gallery/{ids[lab]}_{k}.jpg"
            c.save(out_p / f, quality=90)
            files.append(f)
        gallery[str(ids[lab])] = files

    splits: dict[str, list[dict]] = {s: [] for s in SHELF_SPLIT_SHARE}
    for n in sorted(photos):
        s = split_of[n]
        if s == "gallery":
            continue
        with Image.open(raw_p / n) as im:
            im = im.convert("RGB")
            k = min(1.0, 1400 / max(im.size))
            if k < 1:
                im = im.resize((round(im.width * k), round(im.height * k)))
            im.save(out_p / "images" / Path(n).with_suffix(".jpg").name, quality=90)
            w, h = im.size
        splits[s].append({"image": Path(n).with_suffix(".jpg").name, "width": w, "height": h,
                          "boxes": [[round(v * k, 1) for v in b] for _, b in photos[n]],
                          "products": [ids[lab] for lab, _ in photos[n]]})
    classes = {str(i): _shelf_product(lab, i) for lab, i in ids.items()}
    no_ref = [lab for lab in labels if str(ids[lab]) not in gallery]
    (out_p / RPC_JSON).write_text(json.dumps(
        {"source": SHELVES_SOURCE, "seed": seed, "session_gap_s": SESSION_GAP_S,
         "classes": classes, "gallery": gallery, "splits": splits}, indent=1))
    log(f"{len(photos)} photos in {len(sessions)} sessions, {len(labels)} products; "
        f"gallery: {sum(map(len, gallery.values()))} crops for {len(gallery)} products "
        f"({len(no_ref)} products only appear in scored photos, so have no reference)")
    for s, rows in splits.items():
        log(f"  {s}: {len(rows)} photos, {sum(len(r['boxes']) for r in rows)} products boxed")
    return out_p
