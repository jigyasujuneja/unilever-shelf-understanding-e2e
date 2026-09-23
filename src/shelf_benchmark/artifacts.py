"""Durable artifact writing and the canonical per-run output directory layout.

Two problems are solved here, and they are related.

**Atomic writes.** Every report was previously produced by truncating the destination and
then streaming content into it. That is a lossy operation: a crash, a full disk, or two
concurrent runs pointed at the same ``output_dir`` leave a *syntactically valid but
truncated* file behind. A half-written ``benchmark_summary.json`` is not a parse error the
reader can detect -- ``row_level_report.csv`` truncated after the header is an empty result
set, and a leaderboard cut off mid-table is a shorter leaderboard. Downstream consumers
silently read the wrong numbers. The writers here stage into a temporary file **in the same
directory** (so ``os.replace`` is a same-filesystem rename, which POSIX guarantees is
atomic) and only then swap it into place. A reader therefore observes either the previous
complete file or the new complete file, never a prefix of either.

**Path ownership.** Callers used to rediscover artifacts by hardcoding relative paths, which
breaks the moment ``reporting.isolate_runs`` moves the run into a subdirectory. `RunArtifacts`
owns the layout: it is the single place an output path is constructed, and it emits a
``manifest.json`` describing what was written, so a consumer can discover the artifacts of a
run by reading one file instead of guessing filenames.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Dict, Iterator, List, Optional, Sequence

__all__ = [
    "ARTIFACT_SPECS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ArtifactSpec",
    "RunArtifacts",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "atomic_text_writer",
]

# Bumped whenever the *shape* of manifest.json changes. Readers must refuse a manifest whose
# major version they do not understand rather than silently misreading it.
MANIFEST_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------


def _staged_temp_path(dest: Path) -> Path:
    """Reserve a uniquely named temp file *beside* `dest`.

    Same-directory placement is load-bearing: `os.replace` is only atomic within one
    filesystem, so staging in `/tmp` and moving would degrade to a copy across a mount
    boundary and reintroduce exactly the torn-file window this module exists to close.
    The leading dot keeps the staging file out of naive `*.json` globs while it exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".part")
    os.close(fd)
    return Path(name)


@contextlib.contextmanager
def atomic_text_writer(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    newline: Optional[str] = None,
) -> Iterator[IO[str]]:
    """Yield a writable text handle whose content replaces `path` only on clean exit.

    Exists for the streaming writers (`csv.DictWriter` and friends) that need a file
    object rather than a finished string. If the body raises, the staging file is removed
    and `path` is left exactly as it was.
    """
    dest = Path(path)
    tmp = _staged_temp_path(dest)
    try:
        with open(tmp, "w", encoding=encoding, newline=newline) as handle:
            yield handle
            # Flush Python's buffer and force the bytes out to the device before the
            # rename. Without this the rename can land before the data does, which on a
            # crash yields a correctly named but empty file.
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` so that readers never observe a partial file."""
    dest = Path(path)
    with atomic_text_writer(dest, encoding=encoding) as handle:
        handle.write(text)
    return dest


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write `data` to `path` so that readers never observe a partial file."""
    dest = Path(path)
    tmp = _staged_temp_path(dest)
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return dest


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    default: Any = None,
) -> Path:
    """Serialize `payload` as JSON and write it atomically.

    Serialization happens *before* the destination is touched, so a payload containing a
    non-serializable value fails without having disturbed the existing file at all.
    """
    text = json.dumps(payload, indent=indent, default=default)
    return atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# Run output layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSpec:
    """Declares one artifact a run may produce: its stable key, filename and media type."""

    key: str
    relative_path: str
    media_type: str
    description: str


# The canonical layout of a run directory. `key` is the stable identifier callers use
# (and the key already present in the dict `BenchmarkReportGenerator.generate_all_reports`
# returns); `relative_path` is the only place the filename is spelled.
ARTIFACT_SPECS: Sequence[ArtifactSpec] = (
    ArtifactSpec(
        "row_level_csv",
        "row_level_report.csv",
        "text/csv",
        "One row per detected product facing, with scoring provenance.",
    ),
    ArtifactSpec(
        "row_level_json",
        "row_level_report.json",
        "application/json",
        "The row-level report as JSON, for programmatic consumers.",
    ),
    ArtifactSpec(
        "summary_csv",
        "benchmark_summary.csv",
        "text/csv",
        "One row per benchmark run (task x approach x model).",
    ),
    ArtifactSpec(
        "summary_json",
        "benchmark_summary.json",
        "application/json",
        "Run summaries, leaderboard and provenance in one payload.",
    ),
    ArtifactSpec(
        "leaderboard_csv",
        "leaderboard.csv",
        "text/csv",
        "Per (approach, model) aggregate metrics, ranked.",
    ),
    ArtifactSpec(
        "markdown_report",
        "benchmark_report.md",
        "text/markdown",
        "Human-readable benchmark report.",
    ),
    ArtifactSpec(
        "predictions_json",
        "predictions.json",
        "application/json",
        "Raw model output with no scoring, so the run can be re-scored later.",
    ),
    ArtifactSpec(
        "otel_log",
        "otel_logs.jsonl",
        "application/jsonl",
        "OpenTelemetry span records emitted during the run.",
    ),
    ArtifactSpec(
        "sft_dataset_jsonl",
        "tuning_data/shelf_sft_train.jsonl",
        "application/jsonl",
        "Vertex AI supervised fine-tuning dataset generated from this run.",
    ),
)

_SPECS_BY_KEY: Dict[str, ArtifactSpec] = {spec.key: spec for spec in ARTIFACT_SPECS}


class RunArtifacts:
    """Owns the output directory of a single run and the manifest describing it.

    Construct one per run and ask it for paths instead of joining strings. Nothing else
    should know that the summary is called ``benchmark_summary.json``; that fact lives in
    `ARTIFACT_SPECS` and nowhere else, so relocating or renaming an artifact is a one-line
    change rather than a search across the UI, the CLI and the reporting layer.

    `run_dir` is where files are written. `base_dir` is the root the manifest records paths
    relative to; with ``reporting.isolate_runs`` enabled the two differ, and the manifest
    stays portable because every recorded path is relative, never absolute.
    """

    schema_version = MANIFEST_SCHEMA_VERSION

    def __init__(
        self,
        run_dir: str | Path,
        *,
        base_dir: Optional[str | Path] = None,
        run_id: Optional[str] = None,
        generated_by: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.base_dir = Path(base_dir) if base_dir is not None else self.run_dir
        self.run_id = run_id
        self.generated_by = generated_by
        # Insertion-ordered so the manifest lists artifacts in the order they were produced.
        self._recorded: Dict[str, Dict[str, Any]] = {}

    # -- layout ---------------------------------------------------------

    @classmethod
    def spec_for(cls, key: str) -> ArtifactSpec:
        """Look up an artifact declaration, failing loudly on an unknown key.

        A typo'd key must not silently produce a file nobody reads.
        """
        try:
            return _SPECS_BY_KEY[key]
        except KeyError:
            known = ", ".join(sorted(_SPECS_BY_KEY))
            raise KeyError(
                f"Unknown artifact key '{key}'. Declare it in "
                f"shelf_benchmark.artifacts.ARTIFACT_SPECS first. Known keys: {known}."
            ) from None

    def path_for(self, key: str) -> Path:
        """Absolute path this run would write artifact `key` to."""
        return self.run_dir / self.spec_for(key).relative_path

    def relative_path_for(self, key: str) -> str:
        """POSIX-style path of artifact `key` relative to `base_dir`."""
        return self._relativize(self.path_for(key))

    def ensure_dir(self) -> Path:
        """Create the run directory, returning it."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def _relativize(self, path: Path) -> str:
        try:
            return path.relative_to(self.base_dir).as_posix()
        except ValueError:
            # Outside the run tree (an explicitly relocated otel log, say). Record the
            # absolute path rather than a misleading bare filename.
            return path.as_posix()

    # -- manifest -------------------------------------------------------

    def record(self, key: str, path: Optional[str | Path] = None, **extra: Any) -> Path:
        """Note that artifact `key` was produced, for inclusion in the manifest.

        `path` overrides the canonical location for artifacts a caller writes somewhere
        else (the OTel log, which `TelemetryConfig.otel_log_path` may point anywhere).
        Files that do not exist are still recorded, flagged ``exists: false``, so a
        consumer can tell "the run did not produce this" from "the run never heard of it".
        """
        spec = self.spec_for(key)
        target = Path(path) if path is not None else self.path_for(key)
        exists = target.is_file()
        entry: Dict[str, Any] = {
            "key": spec.key,
            "path": self._relativize(target),
            "media_type": spec.media_type,
            "description": spec.description,
            "exists": exists,
            "size_bytes": target.stat().st_size if exists else None,
        }
        entry.update(extra)
        self._recorded[key] = entry
        return target

    def entries(self) -> List[Dict[str, Any]]:
        """The recorded artifacts, in the order they were recorded."""
        return list(self._recorded.values())

    def manifest_payload(self) -> Dict[str, Any]:
        """The exact object `write_manifest` serializes."""
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generated_by_user": self.generated_by,
            "run_id": self.run_id,
            # Recorded so a consumer reading a manifest out of GCS can tell whether the
            # run was isolated into a subdirectory.
            "run_dir": self._relativize(self.run_dir) or ".",
            "artifacts": self.entries(),
        }

    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_FILENAME

    def write_manifest(self) -> Path:
        """Write `manifest.json` atomically and return its path."""
        return atomic_write_json(self.manifest_path(), self.manifest_payload())

    def as_path_mapping(self) -> Dict[str, str]:
        """`{key: absolute path}` for the artifacts recorded so far.

        Matches the dict shape `BenchmarkReportGenerator.generate_all_reports` already
        returns, so callers can migrate to `RunArtifacts` without changing their reads.
        """
        return {
            key: str((self.base_dir / entry["path"]).resolve())
            if not Path(entry["path"]).is_absolute()
            else entry["path"]
            for key, entry in self._recorded.items()
        }
