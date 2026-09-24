"""Competing Architecture Tracks (Tracks A, B1, B2, C, D1, D2 [mmastrac/djev], E [I-JEPA], F [PaliGemma-2-LoRA])."""

from shelf_e2e.tracks.base import BaseTrackPipeline
from shelf_e2e.tracks.track_a_cascading_vit import TrackACascadingViTPipeline
from shelf_e2e.tracks.track_b_e2e_vlm import TrackBEndToEndVLMPipeline
from shelf_e2e.tracks.track_c_tiered_hybrid import TrackCTieredHybridPipeline
from shelf_e2e.tracks.track_d_jev_routing import TrackDJevRoutingPipeline
from shelf_e2e.tracks.track_ef_modern_vision import (
    TrackB2TwoStageCropVLMPipeline,
    TrackEIJEPALatentWorldModelPipeline,
    TrackFPaliGemma2LoRAPipeline,
)

__all__ = [
    "BaseTrackPipeline",
    "TrackACascadingViTPipeline",
    "TrackBEndToEndVLMPipeline",
    "TrackB2TwoStageCropVLMPipeline",
    "TrackCTieredHybridPipeline",
    "TrackDJevRoutingPipeline",
    "TrackEIJEPALatentWorldModelPipeline",
    "TrackFPaliGemma2LoRAPipeline",
]
