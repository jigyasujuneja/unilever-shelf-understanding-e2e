"""Residual InfoNCE Projection Head over ``gemini-embedding-2`` (``EXP-32``).

Learns a lightweight residual projection ``f(x) = unit(x + alpha * (relu(x @ W1) @ W2))``
strictly from reference gallery / training split views (zero test or eval sample access).
Sharpens cosine separation between same-brand sister variants while preserving pretrained
``gemini-embedding-2`` geometry via a small residual gate ``alpha`` and validation guard.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path(tempfile.gettempdir()) / "shelf_bench_emb_cache"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(1e-8, norms)


class ResidualInfoNCEHead:
    """Fast NumPy Residual InfoNCE adapter ``f(x) = norm(x + alpha * relu(x @ W1) @ W2)``."""

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 192,
        alpha: float = 0.22,
        temperature: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.seed = seed
        self.w1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.metrics: dict[str, float] = {}

    def is_fitted(self) -> bool:
        return self.w1 is not None and self.w2 is not None

    def transform_matrix(self, x: np.ndarray) -> np.ndarray:
        """Apply residual projection to ``(N, D)`` float32 matrix."""
        if not self.is_fitted() or x.size == 0:
            return _l2_normalize_rows(x) if x.ndim == 2 and x.size > 0 else x
        h = _relu(x @ self.w1)
        delta = h @ self.w2
        return _l2_normalize_rows(x + self.alpha * delta).astype(np.float32)

    def transform_one(self, vec: list[float] | np.ndarray) -> list[float]:
        """Apply residual projection to a single ``(D,)`` vector and return a list."""
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        out = self.transform_matrix(arr)[0]
        return out.tolist()

    @staticmethod
    def _eval_r1(anchors: np.ndarray, positives: np.ndarray, gallery: np.ndarray, gal_pids: np.ndarray, anc_pids: np.ndarray) -> float:
        if anchors.shape[0] == 0 or gallery.shape[0] == 0:
            return 0.0
        sims = anchors @ gallery.T
        top_idx = np.argmax(sims, axis=1)
        preds = gal_pids[top_idx]
        return float(np.mean(preds == anc_pids))

    def fit(
        self,
        index: list[tuple[int, str, list[float]]],
        catalog: dict[int, dict[str, Any]] | None = None,
        epochs: int = 35,
        lr: float = 0.08,
        batch_size: int = 128,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """Fit residual projection strictly on multi-view gallery vectors (zero test/eval access)."""
        if len(index) < 16:
            return {"status": 0.0, "base_val_r1": 0.0, "tuned_val_r1": 0.0}

        dim = len(index[0][2])
        self.dim = dim
        key_src = "|".join(f"{pid}:{path}" for pid, path, _ in sorted(index, key=lambda t: (t[0], t[1])))
        digest = hashlib.sha256(f"v1:{dim}:{self.hidden_dim}:{self.alpha}:{key_src}".encode()).hexdigest()[:20]
        cache_path = CACHE_DIR / f"residual_infonce_{digest}.pkl"
        if use_cache and cache_path.exists():
            try:
                saved = pickle.loads(cache_path.read_bytes())
                self.w1 = saved["w1"]
                self.w2 = saved["w2"]
                self.alpha = float(saved.get("alpha", self.alpha))
                self.metrics = dict(saved.get("metrics", {}))
                return self.metrics
            except Exception:
                pass

        by_pid: dict[int, list[np.ndarray]] = defaultdict(list)
        for pid, _, vec in index:
            by_pid[int(pid)].append(np.asarray(vec, dtype=np.float32))

        multi_pids = [pid for pid, views in sorted(by_pid.items()) if len(views) >= 2]
        if len(multi_pids) < 8:
            # Synthesize slight isotropic jitter pairs if gallery is single-view per SKU
            rng_syn = np.random.default_rng(self.seed)
            for pid, views in sorted(by_pid.items()):
                v0 = views[0]
                noise = rng_syn.normal(0.0, 0.025, size=v0.shape).astype(np.float32)
                v1 = v0 + noise
                v1 = v1 / max(1e-8, float(np.linalg.norm(v1)))
                by_pid[pid].append(v1)
            multi_pids = sorted(by_pid.keys())

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(multi_pids))
        n_val = max(4, int(round(0.15 * len(multi_pids))))
        val_pid_set = {multi_pids[i] for i in perm[:n_val]}
        train_pids = [multi_pids[i] for i in perm[n_val:]]
        if not train_pids:
            train_pids = multi_pids

        # Build anchor/positive matrices for train and held-out gallery validation
        tr_anc_list, tr_pos_list, tr_pid_list = [], [], []
        for pid in train_pids:
            views = by_pid[pid]
            for idx_v in range(len(views)):
                tr_anc_list.append(views[idx_v])
                tr_pos_list.append(views[(idx_v + 1) % len(views)])
                tr_pid_list.append(pid)

        vl_anc_list, vl_pos_list, vl_pid_list = [], [], []
        gal_vecs_list, gal_pids_list = [], []
        for pid, views in sorted(by_pid.items()):
            gal_vecs_list.append(views[0])
            gal_pids_list.append(pid)
            if pid in val_pid_set and len(views) >= 2:
                vl_anc_list.append(views[1])
                vl_pos_list.append(views[0])
                vl_pid_list.append(pid)

        tr_anc = _l2_normalize_rows(np.stack(tr_anc_list).astype(np.float32))
        tr_pos = _l2_normalize_rows(np.stack(tr_pos_list).astype(np.float32))
        tr_pids = np.asarray(tr_pid_list, dtype=np.int64)

        vl_anc = _l2_normalize_rows(np.stack(vl_anc_list).astype(np.float32)) if vl_anc_list else tr_anc[:16]
        vl_pids = np.asarray(vl_pid_list if vl_anc_list else tr_pid_list[:16], dtype=np.int64)
        gal_mat = _l2_normalize_rows(np.stack(gal_vecs_list).astype(np.float32))
        gal_pids = np.asarray(gal_pids_list, dtype=np.int64)

        base_val_r1 = self._eval_r1(vl_anc, vl_anc, gal_mat, gal_pids, vl_pids)

        # Initialize W1 with small random orthogonal-like weights and W2 at zero (identity start)
        w1 = (rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, self.hidden_dim))).astype(np.float32)
        w2 = np.zeros((self.hidden_dim, dim), dtype=np.float32)

        best_w1 = w1.copy()
        best_w2 = w2.copy()
        best_val_r1 = base_val_r1
        inv_tau = 1.0 / max(1e-3, self.temperature)
        n_train = tr_anc.shape[0]

        for _epoch in range(epochs):
            order = rng.permutation(n_train)
            for start in range(0, n_train, batch_size):
                batch_idx = order[start : start + batch_size]
                if len(batch_idx) < 4:
                    continue
                x_a = tr_anc[batch_idx]
                x_p = tr_pos[batch_idx]
                b_pids = tr_pids[batch_idx]

                # Forward pass on anchors and positives
                h_a = _relu(x_a @ w1)
                z_a_raw = x_a + self.alpha * (h_a @ w2)
                z_a = _l2_normalize_rows(z_a_raw)

                h_p = _relu(x_p @ w1)
                z_p_raw = x_p + self.alpha * (h_p @ w2)
                z_p = _l2_normalize_rows(z_p_raw)

                # Cosine logits (B, B) with same-PID false-negative mask
                logits = (z_a @ z_p.T) * inv_tau
                same_pid_mask = (b_pids[:, None] == b_pids[None, :]) & (
                    ~np.eye(len(batch_idx), dtype=bool)
                )
                logits = np.where(same_pid_mask, -1e9, logits)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_l = np.exp(logits)
                probs = exp_l / np.maximum(1e-8, np.sum(exp_l, axis=1, keepdims=True))

                # Gradient of InfoNCE loss w.r.t. z_a and z_p
                eye = np.eye(len(batch_idx), dtype=np.float32)
                d_logits = (probs - eye) * (inv_tau / len(batch_idx))
                d_za = d_logits @ z_p
                d_zp = d_logits.T @ z_a

                # Backprop through L2 normalization (approx projection orthogonal to z)
                d_za_raw = d_za - z_a * np.sum(d_za * z_a, axis=1, keepdims=True)
                d_zp_raw = d_zp - z_p * np.sum(d_zp * z_p, axis=1, keepdims=True)

                # Gradients for W2 and W1
                grad_w2 = self.alpha * (h_a.T @ d_za_raw + h_p.T @ d_zp_raw)
                d_ha = (d_za_raw @ w2.T) * (h_a > 0).astype(np.float32)
                d_hp = (d_zp_raw @ w2.T) * (h_p > 0).astype(np.float32)
                grad_w1 = self.alpha * (x_a.T @ d_ha + x_p.T @ d_hp)

                w1 -= lr * grad_w1
                w2 -= lr * grad_w2

            # Evaluate on held-out gallery views
            self.w1, self.w2 = w1, w2
            vl_proj = self.transform_matrix(vl_anc)
            gal_proj = self.transform_matrix(gal_mat)
            val_r1 = self._eval_r1(vl_proj, vl_proj, gal_proj, gal_pids, vl_pids)
            if val_r1 >= best_val_r1:
                best_val_r1 = val_r1
                best_w1 = w1.copy()
                best_w2 = w2.copy()

        # Validation safety guard: if residual head did not at least match base R@1, keep identity (w2=0)
        if best_val_r1 < base_val_r1:
            self.w1 = np.zeros((dim, self.hidden_dim), dtype=np.float32)
            self.w2 = np.zeros((self.hidden_dim, dim), dtype=np.float32)
            best_val_r1 = base_val_r1
        else:
            self.w1 = best_w1
            self.w2 = best_w2

        self.metrics = {
            "base_val_r1": round(base_val_r1, 4),
            "tuned_val_r1": round(best_val_r1, 4),
            "r1_gain": round(best_val_r1 - base_val_r1, 4),
            "train_pairs": float(n_train),
            "val_pairs": float(vl_anc.shape[0]),
        }
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(
                    pickle.dumps(
                        {"w1": self.w1, "w2": self.w2, "alpha": self.alpha, "metrics": self.metrics},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
            except Exception:
                pass
        return self.metrics
