"""Competing Architecture Tracks (Tracks A, B, C, D + GeminiDiffusion-as-Jev)."""

from shelf_e2e.tracks.base import BaseTrackPipeline
from shelf_e2e.tracks.track_a_cascading_vit import TrackACascadingViTPipeline
from shelf_e2e.tracks.track_b_e2e_vlm import TrackBEndToEndVLMPipeline
from shelf_e2e.tracks.track_c_tiered_hybrid import TrackCTieredHybridPipeline
from shelf_e2e.tracks.track_d_jev_routing import TrackDJevRoutingPipeline

__all__ = [
    "BaseTrackPipeline",
    "TrackACascadingViTPipeline",
    "TrackBEndToEndVLMPipeline",
    "TrackCTieredHybridPipeline",
    "TrackDJevRoutingPipeline",
]
