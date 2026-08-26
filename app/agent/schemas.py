"""Validated contracts for vision boundaries, review decisions, and artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field




class TrashCandidate(BaseModel):
    location: str = Field(max_length=240)
    prompt_variants: list[str] = Field(min_length=2, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)


class TrashScreen(BaseModel):
    excessive_roadside_trash: bool
    pile_count: int = Field(ge=0)
    report: str = Field(max_length=600)
    uncertainty: bool
    candidates: list[TrashCandidate]


class ReviewDecision(BaseModel):
    image_id: str
    decision: Literal["approve", "no_subject", "exclude"]
    note: str = Field(default="", max_length=600)


class Observation(BaseModel):
    """A segmented trash instance in a sanitized street image."""

    detection_id: str
    image_id: str
    camera_lat: float
    camera_lon: float
    compass_angle: float | None = None
    captured_at: str | None = None

    mask_area_px: int = 0
    image_path: str | None = None
    overlay_path: str | None = None
    mask_path: str | None = None
    sam_prompt: str | None = None
    sam_score: float | None = None

    screen_confidence: float | None = None
    review_status: Literal["approved", "pending", "no_subject", "excluded"] = "approved"
    review_note: str = ""

    @property
    def accepted(self) -> bool:
        return self.review_status == "approved"


class CoverageCell(BaseModel):
    """Imagery availability metrics for an individual grid cell."""

    unit_id: str
    unit_name: str | None = None
    area_km2: float
    n_images: int
    images_per_km2: float
    imagery_available: bool
    is_blind_spot: bool = False


class CoverageAudit(BaseModel):
    """Aggregate imagery availability and density metrics across the AOI."""

    cells: list[CoverageCell]
    n_images: int
    median_images_per_km2: float
    blind_spot_threshold: float
    n_blind_spots: int
    fraction_area_blind: float
    n_units: int
    n_cells_with_imagery: int = 0
    interpretation: str = ""