# Contributing

This repo benchmarks competing shelf-understanding approaches against each other. The numbers it
prints end up in decks. Treat every reported metric as a claim you have to defend.

## 1. Set up

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

`uv` works too if you prefer it: `uv venv && uv pip install -e '.[dev]'`.

Sanity check:

```bash
.venv/bin/pytest -q -m "not live"
.venv/bin/ruff check src tests
```

## 2. The two test lanes

| Marker | Needs GCP? | Runs in CI? | Use it for |
| --- | --- | --- | --- |
| `offline` | No | Yes | Parsing, scoring, geometry, config, report shaping |
| `live` | Yes, real credentials and quota | No | End-to-end calls against Vertex AI / GCS |

```python
import pytest

@pytest.mark.offline
def test_depth_nms_drops_occluded_facings():
    assert run_depth_nms(two_overlapping_facings) == 1
```

Markers are registered in `pyproject.toml` and `--strict-markers` is on, so a typo in a marker name
fails the run instead of quietly skipping your test.

Default local loop: `.venv/bin/pytest -q -m "not live"`. Only run `-m live` deliberately, and know
which project is being billed before you do.

## 3. Adding a new approach

Do not add a branch to an existing pipeline. Add a plugin:

```
src/shelf_benchmark/approaches/<your_approach_name>/plugin.py
```

`plugin.py` must expose either a `get_plugins()` factory returning
`BaseShelfApproachPlugin` instances, or a single concrete `BaseShelfApproachPlugin` subclass defined
in that module. The registry in
[`registry.py`](src/shelf_benchmark/approaches/registry.py) discovers it automatically on import.
See [`TEMPLATE_NEW_APPROACH.md`](src/shelf_benchmark/approaches/TEMPLATE_NEW_APPROACH.md) and the
existing `class_agnostic_visual_embedding` plugin for the shape.

Rules that matter:

- Your `approach_id` is what labels every row in the report. It must be unique and must describe
  what actually ran.
- A broken plugin raises on discovery by design. Set `SHELF_BENCH_TOLERANT_PLUGINS=1` only while you
  are mid-build locally, never in a committed script or in CI.
- Ship at least one `offline` test that constructs your plugin and asserts on its output shape with
  the network stubbed out.

## 4. Any change to a reported metric needs a test

If your diff can move a number in `reports/` -- accuracy, cost, latency accounting, token counts,
facing counts, dedupe behaviour -- the same diff must add or update a test that pins the new
behaviour with a concrete expected value. "I ran it and it looked right" is not reviewable, because
the next person cannot rerun your eyeball.

State the before and after in your PR description when a metric moves, and say why it moved.

## 5. What NOT to do

**Do not hardcode anything derived from `shelf-image.png`.** That file is one sample, not a test
fixture and not a spec. Constants like an expected facing count, a shelf-row split, a brand list, or
a bbox tuple copied out of a run against it will make your approach look accurate on the sample and
fall over on real data. Derive values, or put them in `configs/` and fixtures.

**Do not let a failure path degrade silently to a default.** These are all real and all forbidden:

```python
# Wrong: an unknown approach silently runs something else, still labelled as yours.
plugin = registry.get(approach_id) or DefaultApproach()

# Wrong: a parse failure becomes a confident zero that lands in the report.
try:
    confidence = float(response["confidence"])
except Exception:
    confidence = 0.0

# Wrong: an empty detection result reads as a genuinely empty shelf.
facings = detect(image) or []
```

Raise, or return an explicit `status` the reporting layer can surface. Use
`registry.require(approach_id)` rather than `registry.get()` plus a fallback. A loud failure costs
an afternoon; a quiet one costs a wrong conclusion in a customer deck.

**Do not commit generated output or credentials.** `reports/` and credential patterns are
gitignored. Keep it that way.

## 6. Before you open a PR

```bash
make check     # lint + typecheck + offline tests + wheel packaging check
```

or individually:

```bash
make lint       # ruff check .
make typecheck  # mypy
make test       # pytest -q -m "not live"
make wheel      # builds a wheel and asserts the bundled fixture is packaged
```

These run in CI (`.github/workflows/ci.yml`) on every push and pull request. Ruff, the offline test
lane, the wheel-install check and the Docker build are blocking. Mypy runs with
`continue-on-error: true` as a ratchet while annotations are backfilled -- do not add new type
errors, and fix ones you touch.

`make wheel` exists because `pip install -e .` cannot catch packaging bugs: the editable install
resolves files from the source tree, so a wheel missing `_fixtures/*.png` still "works" locally
while every non-editable install raises `FileNotFoundError`. CI installs a real wheel into a clean
virtualenv for the same reason.

Both `ruff check .` (with `ruff>=0.12.0`) and `mypy src/` are **100% clean (0 findings across all source files)**. Keep both lanes at zero findings in every PR. When multiple engineers share a single checkout or workstation, set `SHELF_BENCH_ISOLATE_RUNS=1` (or pass `--isolate-runs` / `reporting.isolate_runs: true`) so each run writes into its own `<output_dir>/<user>-<UTC timestamp>-<run_id>/` directory.

