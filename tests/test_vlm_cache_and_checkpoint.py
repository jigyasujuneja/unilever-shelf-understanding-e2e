"""Unit tests for content-addressed VLM response disk cache and incremental row checkpointing (PR-17)."""

from __future__ import annotations

import json
from pathlib import Path

from utils.llm import LLMResult, Usage, _VLMResponseCache, vlm_cache_key


def test_vlm_cache_key_is_content_addressed_and_deterministic():
    k1 = vlm_cache_key(
        "gemini-3.5-flash-lite",
        b"jpeg-bytes-1",
        "Pick the panel",
        schema={"type": "object"},
        logprobs=6,
    )
    k2 = vlm_cache_key(
        "gemini-3.5-flash-lite",
        b"jpeg-bytes-1",
        "Pick the panel",
        schema={"type": "object"},
        logprobs=6,
    )
    k_diff = vlm_cache_key(
        "gemini-3.5-flash-lite",
        b"jpeg-bytes-2",
        "Pick the panel",
        schema={"type": "object"},
        logprobs=6,
    )
    assert len(k1) == 64
    assert k1 == k2
    assert k1 != k_diff


def test_vlm_response_cache_save_and_load_roundtrip(tmp_path: Path):
    key = vlm_cache_key("gemini-3.5-flash-lite", b"img", "prompt")
    res = LLMResult(
        data={"pack_size": "100g", "choice": 2},
        usage=Usage(input_tokens=300, output_tokens=12, thinking_tokens=0, calls=1, traffic={"priority": 1}),
        seconds=1.25,
        text='{"pack_size": "100g", "choice": 2}',
        meta={"contract_status": "ok", "contract_error": None},
    )
    _VLMResponseCache.save(tmp_path, key, res)
    loaded = _VLMResponseCache.load(tmp_path, key)
    assert loaded is not None
    assert loaded.data == {"pack_size": "100g", "choice": 2}
    assert loaded.usage.input_tokens == 300
    assert loaded.seconds == 0.0
    assert loaded.meta["cache_hit"] is True
    assert loaded.meta["cache_key"] == key


def test_runner_checkpoint_resume_skips_completed_rows(fake_root, tmp_path: Path, monkeypatch):
    from conftest import OracleLLM

    import runner

    ckpt_file = tmp_path / "ckpt.jsonl"
    monkeypatch.setenv("SHELF_BENCH_CHECKPOINT_FILE", str(ckpt_file))

    # First run writes completed rows to ckpt_file
    out1 = runner.run(
        "single_pass_dedup",
        "gemini-fake",
        split="test",
        limit=2,
        results_dir=tmp_path / "res1",
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    assert ckpt_file.is_file()
    lines = [json.loads(line) for line in ckpt_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert out1["images"] == 2

    # Second run resumes from ckpt_file
    out2 = runner.run(
        "single_pass_dedup",
        "gemini-fake",
        split="test",
        limit=2,
        results_dir=tmp_path / "res2",
        data_root=fake_root,
        llm=OracleLLM(),
        log=lambda *_: None,
    )
    assert out2["images"] == 2
    res2_rows = [
        json.loads(line)
        for line in (tmp_path / "res2" / out2["run_id"] / "images.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(r.get("resumed_from_checkpoint") is True for r in res2_rows)
