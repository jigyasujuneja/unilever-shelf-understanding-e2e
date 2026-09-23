"""Contract tests that keep configuration, packaging and identifiers honest.

Each test here corresponds to a defect that shipped silently because nothing
asserted the invariant. They are intentionally cheap so they can run in the
default offline lane.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.run_ids import MAX_RUN_ID_LENGTH, build_run_id

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_config.yaml"


# --------------------------------------------------------------------------
# Config: the YAML must not silently contradict the Python defaults.
# --------------------------------------------------------------------------

# Keys where configs/default_config.yaml is *intended* to differ from
# BenchmarkConfig(). Anything not listed here is a drift bug: the YAML restates
# ~150 lines of Python defaults, so a divergence is almost always accidental.
#
# Add to this list only with a comment explaining why the divergence is correct.
ALLOWED_CONFIG_DIVERGENCES = {
    # The shipped config activates a curated subset of approaches for a demo run
    # rather than every registered approach.
    ("approaches",),
    # Planograms are not part of the shipped sample data.
    ("buckets", "planograms_bucket"),
    # Commented example endpoints (GEAP / Gemma / tuned SFT) that exist to be
    # copied and edited. They intentionally have no Python default.
    ("model_endpoints",),
}


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        out: dict[tuple[str, ...], object] = {}
        for key, sub in value.items():
            out.update(_flatten(sub, (*prefix, str(key))))
        return out
    return {prefix: value}


def _is_allowed(path: tuple[str, ...]) -> bool:
    return any(path[: len(allowed)] == allowed for allowed in ALLOWED_CONFIG_DIVERGENCES)


@pytest.mark.offline
def test_default_yaml_does_not_silently_diverge_from_python_defaults() -> None:
    """configs/default_config.yaml must only differ from BenchmarkConfig() on purpose.

    This is the test that would have caught two real cost bugs:
      * include_infrastructure_costs shipped `true` against a documented `false`,
        so every run folded modelled infrastructure into headline cost.
      * per_task_per_facing_usd shipped `{}`, which *replaced* rather than
        inherited the Python rates, halving the matching embed cost.
    """
    python_defaults = _flatten(BenchmarkConfig().model_dump())
    shipped = _flatten(BenchmarkConfig.from_yaml(str(DEFAULT_CONFIG)).model_dump())

    unexpected = {
        ".".join(path): (python_defaults.get(path), shipped_value)
        for path, shipped_value in shipped.items()
        if python_defaults.get(path) != shipped_value and not _is_allowed(path)
    }
    assert not unexpected, (
        "configs/default_config.yaml diverges from BenchmarkConfig() at these keys "
        "(python_default, yaml_value). Either fix the YAML or add the key to "
        f"ALLOWED_CONFIG_DIVERGENCES with a justification:\n{unexpected}"
    )


@pytest.mark.offline
def test_infrastructure_costs_are_excluded_by_default() -> None:
    """The documented default in docs/EVALUATION_PROTOCOL.md is `false`.

    Pinned separately from the drift test because this one value silently changes
    every headline cost number in the suite.
    """
    assert BenchmarkConfig().billing.include_infrastructure_costs is False
    shipped = BenchmarkConfig.from_yaml(str(DEFAULT_CONFIG))
    assert shipped.billing.include_infrastructure_costs is False


@pytest.mark.offline
def test_per_task_facing_rates_survive_yaml_load() -> None:
    """An empty YAML mapping must not wipe the documented per-task embed rates."""
    shipped = BenchmarkConfig.from_yaml(str(DEFAULT_CONFIG))
    rates = shipped.billing.embeddings_and_vision.per_task_per_facing_usd
    assert rates == BenchmarkConfig().billing.embeddings_and_vision.per_task_per_facing_usd
    # Spot-check the value that was being halved.
    assert rates["matching"] == pytest.approx(0.000050)


# --------------------------------------------------------------------------
# run_id: the join key for telemetry, billing labels and reports.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_run_ids_are_distinct_for_approaches_sharing_a_prefix() -> None:
    """These two approaches share their first 9 characters.

    The previous `approach[:9]` truncation gave them byte-identical run ids, so
    benchmarking both in one batch merged two different algorithms' rows.
    """
    a = build_run_id("batch", "cls", "two_stage_bbox_guided_nms", "flash")
    b = build_run_id("batch", "cls", "two_stage_physical_crop_per_facing", "flash")
    assert a != b


@pytest.mark.offline
@pytest.mark.parametrize(
    "model",
    ["flash", "gemini-3-flash-preview", "a-very-long-model-identifier-" * 4],
)
def test_run_ids_stay_distinct_and_label_safe_under_shortening(model: str) -> None:
    """Even when shortened to fit a GCP label, distinct inputs stay distinct."""
    approaches = [
        "two_stage_bbox_guided_nms",
        "two_stage_physical_crop_per_facing",
        "single_pass_full_shelf",
        "single_step_detect_classify_and_match",
        "class_agnostic_visual_embedding",
        "cloud_vision_visual_embedding",
        "configurable_multi_attribute_vlm",
        "open_vocab_brand_plus_catalog_resolver",
    ]
    ids = [build_run_id("batch-2024", "cls", a, model) for a in approaches]

    assert len(set(ids)) == len(ids), f"run_id collision among {ids}"
    for run_id in ids:
        assert len(run_id) <= MAX_RUN_ID_LENGTH, f"{run_id} exceeds the GCP label budget"
        assert re.fullmatch(r"[a-z0-9_-]+", run_id), f"{run_id} is not label-safe"


# --------------------------------------------------------------------------
# Packaging: the wheel must actually contain the bundled fixture.
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_wheel_contains_bundled_fixture_image(tmp_path: Path) -> None:
    """A non-editable install must be able to resolve the offline fixture.

    Without [tool.setuptools.package-data] the wheel shipped no `_fixtures/*.png`,
    so `pip install unilever-shelf-benchmark` produced a package whose entire
    offline story raised FileNotFoundError. `pip install -e .` hides this, which
    is why it survived: it is only observable through a real build.
    """
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable in this environment: {build.stderr[-300:]}")

    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "build produced no wheel"
    names = zipfile.ZipFile(wheels[0]).namelist()
    fixtures = [n for n in names if "_fixtures/" in n and n.endswith(".png")]
    assert fixtures, (
        "The built wheel contains no shelf_benchmark/_fixtures/*.png. "
        "Check [tool.setuptools.package-data] in pyproject.toml."
    )
    taxonomies = [n for n in names if n.endswith("_resources/taxonomy.yaml")]
    assert taxonomies, (
        "The built wheel contains no shelf_benchmark/_resources/taxonomy.yaml. "
        "Without it BenchmarkConfig() raises on a real install, because the taxonomy defaults "
        "are read from that file rather than duplicated as Python literals."
    )


# --------------------------------------------------------------------------
# Taxonomy: identical regardless of the process working directory.
# --------------------------------------------------------------------------

_PROBE = """
import json
from shelf_benchmark.config import BenchmarkConfig
from shelf_benchmark.tasks.classification import build_classification_prompt
t = BenchmarkConfig().taxonomy
print(json.dumps({
    "categories": t.categories,
    "subcategories": t.subcategories,
    "packaging_types": t.packaging_types,
    "pack_types": t.pack_types,
    "size_bucket_labels": t.size_bucket_labels,
    "prompt": build_classification_prompt(t),
}))
"""


def _taxonomy_from_cwd(cwd: Path) -> dict:
    """Load the default taxonomy in a fresh interpreter started from `cwd`."""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        # Keep the probe hermetic: never inherit a developer's GCP credentials.
        "GOOGLE_APPLICATION_CREDENTIALS": "",
    }
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"probe failed from {cwd}:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout)


@pytest.mark.offline
def test_default_taxonomy_is_identical_from_any_working_directory(tmp_path: Path) -> None:
    """The taxonomy must not depend on where the process was launched from.

    `TaxonomyConfig` used to resolve 'configs/taxonomy.yaml' against the CWD and, when that
    relative path missed, fall back to a *second* copy of the lists hardcoded in Python. The two
    copies had drifted, so running from the repo root gave 10/18/9 categories/subcategories/
    packaging types while running from anywhere else gave 10/14/7 -- a different 'Categories'
    universe, a different VLM prompt, and therefore incomparable benchmark numbers. Nothing failed.
    """
    from_root = _taxonomy_from_cwd(REPO_ROOT)
    from_tmp = _taxonomy_from_cwd(tmp_path)
    from_fs_root = _taxonomy_from_cwd(Path(os.sep))

    assert from_root == from_tmp == from_fs_root, (
        "The default taxonomy differs by working directory. It must be resolved relative to the "
        "installed package (config.PACKAGED_TAXONOMY_PATH), never relative to os.getcwd()."
    )
    # Guard against the degenerate 'everything is empty everywhere' way to pass the above.
    assert from_root["categories"] and from_root["subcategories"] and from_root["packaging_types"]


@pytest.mark.offline
def test_taxonomy_yaml_is_the_only_source_of_the_default_lists() -> None:
    """The bundled YAML -- not a Python literal -- defines the defaults."""
    from shelf_benchmark.config import PACKAGED_TAXONOMY_PATH, TaxonomyConfig

    assert PACKAGED_TAXONOMY_PATH.exists(), PACKAGED_TAXONOMY_PATH
    raw = yaml.safe_load(PACKAGED_TAXONOMY_PATH.read_text(encoding="utf-8"))
    tax = TaxonomyConfig()
    for key in ("categories", "subcategories", "packaging_types", "pack_types"):
        assert getattr(tax, key) == raw[key], f"{key} drifted from {PACKAGED_TAXONOMY_PATH}"
    assert tax.taxonomy_source == str(PACKAGED_TAXONOMY_PATH)


@pytest.mark.offline
def test_missing_taxonomy_override_is_an_error_not_a_silent_fallback(tmp_path: Path) -> None:
    """A typo'd taxonomy path must fail loudly instead of quietly loading a different taxonomy."""
    from shelf_benchmark.config import TaxonomyConfig

    with pytest.raises(FileNotFoundError, match="Taxonomy file not found"):
        TaxonomyConfig.from_yaml_or_defaults(tmp_path / "does_not_exist.yaml")


@pytest.mark.offline
def test_taxonomy_override_merges_onto_packaged_defaults(tmp_path: Path) -> None:
    """Keys omitted from an override file fall back to the bundled values."""
    from shelf_benchmark.config import TaxonomyConfig

    override = tmp_path / "my_taxonomy.yaml"
    override.write_text('categories: ["Only Mine"]\n', encoding="utf-8")

    tax = TaxonomyConfig.from_yaml_or_defaults(override)
    assert tax.categories == ["Only Mine"]
    assert tax.subcategories == TaxonomyConfig().subcategories, "omitted keys must inherit"
    assert tax.taxonomy_source == str(override.resolve())


@pytest.mark.offline
def test_relative_taxonomy_file_resolves_next_to_its_config(tmp_path: Path) -> None:
    """`taxonomy_file:` is relative to the config that declares it, not to the CWD."""
    (tmp_path / "side.yaml").write_text('categories: ["Sibling"]\n', encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text('taxonomy_file: "side.yaml"\n', encoding="utf-8")

    cfg = BenchmarkConfig.from_yaml(cfg_path)
    assert cfg.taxonomy.categories == ["Sibling"]


# --------------------------------------------------------------------------
# Billing: one field per quantity, one definition of "total cost".
# --------------------------------------------------------------------------


@pytest.mark.offline
def test_provisioned_throughput_rate_is_settable() -> None:
    """Changing the configured GSU rate must change the computed GSU cost.

    The engine used to read `getattr(pt, "hourly_rate_per_gsu_usd", None) or
    getattr(pt, "gsu_hourly_rate_usd", 22.0)`. On a Pydantic model the first name is always
    present and non-zero, so the second field -- which the config documented as *the* official
    committed-use rate, 8.1x lower -- could never take effect. The shape of the expression made
    it look like a considered fallback.
    """
    from shelf_benchmark.config import GCPBillingConfig, ModelPricing
    from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
    from shelf_benchmark.models import TokenUsageMetrics

    def gsu_cost(rate: float) -> float:
        cfg = GCPBillingConfig()
        cfg.provisioned_throughput.enabled = True
        cfg.provisioned_throughput.hourly_rate_per_gsu_usd = rate
        return GCPBillingAndCostEngine.compute_all_in_separated_gcp_cost(
            tokens=TokenUsageMetrics(),
            pricing=ModelPricing(),
            product_count=4,
            billing_cfg=cfg,
        ).vertex_ai_provisioned_throughput_usd

    cheap, dear = gsu_cost(2.70), gsu_cost(22.00)
    assert dear > cheap > 0.0
    # rel=1e-4, not tighter: both figures are stored rounded to 8 decimal places and the cheap one
    # is ~5e-6, so the last stored digit is worth ~1e-3 of the ratio.
    assert dear / cheap == pytest.approx(22.00 / 2.70, rel=1e-4)


@pytest.mark.offline
def test_accelerator_does_not_override_include_infrastructure_costs() -> None:
    """`include_infrastructure_costs: false` must hold even when an accelerator is configured.

    The gate used to be `include_infra_overhead or accel_per_sec > 0.0`, so `--accelerator tpu-v5e`
    silently folded modelled infrastructure into the total that the operator had explicitly asked
    to exclude -- making a CPU run and a TPU run non-comparable for reasons unrelated to hardware.
    """
    from shelf_benchmark.config import GCPBillingConfig, ModelPricing
    from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
    from shelf_benchmark.models import TokenUsageMetrics

    cfg = GCPBillingConfig()
    cfg.include_infrastructure_costs = False
    cfg.cloud_run.accelerator_type = "tpu-v5e"
    cfg.cloud_run.accelerator_count = 1

    cost = GCPBillingAndCostEngine.compute_all_in_separated_gcp_cost(
        tokens=TokenUsageMetrics(input_tokens=1000, output_tokens=100, total_tokens=1100),
        pricing=ModelPricing(),
        product_count=4,
        billing_cfg=cfg,
    )
    assert cost.cloud_run_compute_usd == 0.0
    assert cost.gcs_and_observability_usd == 0.0
    assert cost.includes_modelled_infrastructure is False


@pytest.mark.offline
@pytest.mark.parametrize("include_infra", [True, False])
def test_cost_per_1k_images_is_exactly_1000x_cost_per_image(include_infra: bool) -> None:
    """The reported totals must agree with each other under either infra setting.

    `cost_per_1k_images_usd` was assembled from its own mix of gated and un-gated components, so it
    always included infrastructure even when `cost_per_shelf_image_usd` excluded it. Two numbers in
    the same report disagreeing about what a run costs is worse than either being wrong.
    """
    from shelf_benchmark.config import GCPBillingConfig, ModelPricing
    from shelf_benchmark.evaluation.gcp_billing import GCPBillingAndCostEngine
    from shelf_benchmark.models import TokenUsageMetrics

    cfg = GCPBillingConfig()
    cfg.include_infrastructure_costs = include_infra

    cost = GCPBillingAndCostEngine.compute_all_in_separated_gcp_cost(
        tokens=TokenUsageMetrics(input_tokens=2000, thinking_tokens=500, output_tokens=300, total_tokens=2800),
        pricing=ModelPricing(),
        product_count=5,
        billing_cfg=cfg,
    )
    assert cost.cost_per_1k_images_usd == pytest.approx(cost.cost_per_shelf_image_usd * 1000.0, rel=1e-6)
    # A division cannot round-trip exactly at 8-decimal storage, so this asserts to the storage
    # granularity rather than to an exactness the format cannot express.
    assert cost.cost_per_product_usd == pytest.approx(cost.cost_per_shelf_image_usd / 5.0, abs=1e-8)


# --------------------------------------------------------------------------
# Doctests on the pure-function modules are part of the default lane.
# --------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.parametrize("module_name", ["shelf_benchmark.run_ids", "shelf_benchmark.text_normalization"])
def test_pure_module_doctests(module_name: str) -> None:
    """Executable examples in these modules must stay true.

    `pytest` does not collect doctests by default, so the examples in these docstrings were
    documentation that nothing checked. They are the cheapest specification available for two
    modules whose whole job is a deterministic string transform.
    """
    import doctest
    import importlib

    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {module_name}"
    assert result.attempted > 0, f"{module_name} has no doctests; this test would be vacuous"
