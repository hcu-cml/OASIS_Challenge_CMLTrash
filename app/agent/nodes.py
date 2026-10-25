"""Backend nodes for the trash-first, privacy-gated roadside workflow."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
from langgraph.types import interrupt

from .. import citygml, db, geo, perception, sources
from .llm import Provider, jpeg_bytes
from .schemas import (
    CoverageAudit,
    CoverageCell,
    Observation,
    ReviewDecision,
    TrashScreen,
)
from .state import (
    AgentState,
    NodeReporter,
    RunCancelled,
    as_dump,
    dumps,
    load_observations,
)

log = logging.getLogger(__name__)
BLIND_SPOT_FRACTION = 0.25
CITYGML_FILENAME = "trash_citygml.gml"
GEOJSON_FILENAME = "trash_detections.geojson"


class Nodes:
    def __init__(
        self,
        run_id: str,
        provider: Provider,
        cfg: dict[str, Any],
        cancel_event: threading.Event | None = None,
    ):
        self.run_id = run_id
        self.provider = provider
        self.cfg = cfg
        self.cancel_event = cancel_event or threading.Event()
        self.report = NodeReporter(run_id, check_cancel=self._check_cancel)
        self.run_dir = Path(cfg["runs_dir"]) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._segmenter: perception.Segmenter | None = None

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled("run cancellation requested")

    def aoi(self, state: AgentState) -> dict:
        with self.report.step("aoi") as r:
            place = state.get("place") or self.cfg.get("place")
            drawn = state.get("bbox")
            if drawn:
                bbox = [float(value) for value in drawn]
                box_name = f"drawn area {bbox[1]:.4f},{bbox[0]:.4f} to {bbox[3]:.4f},{bbox[2]:.4f}"
                area = {"bbox": bbox, "geometry": geo.bbox_polygon(bbox), "display_name": box_name}
                place = place or box_name
            elif place:
                with r.tool("nominatim.search", f"q={place}"):
                    area = sources.geocode_area(place)
                bbox = area["bbox"]
            else:
                raise ValueError("Provide a place or draw a bounding box.")
            epsg = geo.utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            r.log(f"Area resolved to {area['display_name']}; working CRS EPSG:{epsg}")
            r.artifact("aoi", {"bbox": bbox, "geometry": area["geometry"], "name": area["display_name"]})
            return {
                "place": place,
                "bbox": bbox,
                "epsg": epsg,
                "aoi_geojson": area["geometry"] or {},
                "manifest": {"aoi": {
                    "query": place, "resolved": area["display_name"], "epsg": epsg,
                    "source": "drawn_bbox" if drawn else "nominatim", "bbox": bbox,
                }},
            }

    def coverage_audit(self, state: AgentState) -> dict:
        with self.report.step("coverage_audit") as r:
            bbox = state["bbox"]
            with r.tool("mapillary.search", f"limit={self.cfg['max_images']}"):
                images = sources.parse_images(sources.mapillary_search(
                    bbox, self.cfg.get("mapillary_token", ""), limit=self.cfg["max_images"],
                    check_cancel=self._check_cancel,
                ))
            polygons = _grid_polygons(bbox)
            cells: list[CoverageCell] = []
            for unit_id, name, polygon in polygons:
                self._check_cancel()
                area_km2 = _area_km2(polygon)
                if area_km2 <= 0:
                    continue
                n_images = _count_within(images, polygon)
                cells.append(CoverageCell(
                    unit_id=unit_id, unit_name=name, area_km2=round(area_km2, 4),
                    n_images=n_images, images_per_km2=round(n_images / area_km2, 2),
                    imagery_available=n_images > 0,
                ))
            # Blind spots are cells falling well below the median density.
            covered = [cell for cell in cells if cell.n_images > 0]
            densities = [cell.images_per_km2 for cell in covered]
            median = float(np.median(densities)) if densities else 0.0
            threshold = median * BLIND_SPOT_FRACTION
            for cell in cells:
                cell.is_blind_spot = not cell.imagery_available or cell.images_per_km2 < threshold
            blind = [cell for cell in cells if cell.is_blind_spot]
            blind_fraction = sum(cell.area_km2 for cell in blind) / max(
                sum(cell.area_km2 for cell in cells), 1e-9
            )
            if not covered:
                interpretation = (
                    "No street-level imagery is available anywhere in this area, so "
                    "nothing can be surveyed here."
                )
            else:
                interpretation = (
                    f"{len(images)} street-level images are available. "
                    f"{len(covered)} of {len(cells)} grid cells have imagery, and where "
                    f"imagery exists the typical density is {median:.0f} images per km²."
                )
            audit = CoverageAudit(
                cells=cells, n_images=len(images), median_images_per_km2=round(median, 2),
                blind_spot_threshold=round(threshold, 2), n_blind_spots=len(blind),
                fraction_area_blind=round(blind_fraction, 4), n_units=len(cells),
                n_cells_with_imagery=len(covered),
                interpretation=interpretation,
            )
            r.log(interpretation)
            r.artifact("coverage", as_dump(audit))
            r.artifact("images", [{"lat": image["lat"], "lon": image["lon"]} for image in images])
            return {
                "coverage": as_dump(audit),
                "images": images,
                "manifest": {"coverage_audit": {
                    "scope": "imagery_availability_only",
                    "n_images": len(images), "n_cells": len(cells),
                    "cells_with_imagery": len(covered),
                    "median_basis": "cells_with_imagery",
                    "blind_spot_fraction": BLIND_SPOT_FRACTION,
                }},
            }

    def fetch(self, state: AgentState) -> dict:
        with self.report.step("fetch") as r:
            images = state["images"]
            with r.tool("mapillary.thumbnails", f"{len(images)} images, {self.cfg['download_workers']} workers"):
                got = sources.download_thumbs(
                    images, str(self.run_dir / "images"),
                    workers=self.cfg["download_workers"],
                    check_cancel=self._check_cancel,
                )
            self._check_cancel()
            kept = [image for image in images if image.get("local_path")]
            r.log(f"Downloaded {got} of {len(images)} source images")
            if not kept:
                raise RuntimeError("No source imagery could be fetched.")
            return {"images": kept, "manifest": {"fetch": {"requested": len(images), "fetched": got}}}

    def blur_humans(self, state: AgentState) -> dict:
        """
        SAM3-only privacy pass: detect any person in each trash candidate,
        blur them, produce sanitized + overlay images. No VLM, no human gate.
        """
        with self.report.step("blur_humans") as r:
            segmenter = self._get_segmenter()
            source = {image["image_id"]: image for image in state.get("images", [])}
            candidate_ids = set(state.get("trash_candidate_ids", []))
            observations = load_observations(state)
            sanitized_dir = self.run_dir / "sanitized"
            overlay_dir = self.run_dir / "trash_overlays"
            sanitized_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)

            person_prompts = [
                "person walking", "person standing", "human body",
                "pedestrian", "person on bicycle",
            ]

            sanitized: list[dict[str, Any]] = []
            trash_rows = {
                row["image_id"]: dict(row) for row in state.get("trash_screens", [])
            }
            redacted = copied = 0
            total_human_masks = 0
            total = len(candidate_ids)

            for completed, image_id in enumerate(sorted(candidate_ids), start=1):
                self._check_cancel()
                image = source.get(image_id)
                if not image or not image.get("local_path"):
                    continue

                human_masks: list[np.ndarray] = []
                try:
                    with r.tool("sam3.segment", f"{image_id} · person detection"):
                        instances = segmenter.segment_cached(
                            image_id, image["local_path"], prompts=person_prompts,
                            namespace="privacy", category="human",
                            check_cancel=self._check_cancel,
                        )
                    human_masks = [inst.mask for inst in instances]
                except RunCancelled:
                    raise
                except Exception as exc:
                    r.log(f"{image_id}: SAM3 person detection failed: {exc}", level="warn")

                total_human_masks += len(human_masks)

                out = sanitized_dir / f"{image_id}.jpg"
                if human_masks:
                    with r.tool(
                        "pillow.redact_masks",
                        f"{image_id} · {len(human_masks)} person mask(s)",
                    ):
                        perception.redact_masks_cached(
                            image["local_path"], human_masks, str(out),
                            blur_radius=self.cfg["privacy_blur_radius"],
                            dilation_px=self.cfg["privacy_dilation_px"],
                        )
                    redacted += 1
                else:
                    shutil.copyfile(image["local_path"], out)
                    copied += 1

                # Point the record at the sanitized copy, never the original.
                clean = {
                    key: value for key, value in image.items()
                    if key not in ("local_path", "thumb_url")
                }
                clean.update({
                    "local_path": str(out),
                    "sanitized_url": self._url(out),
                })
                sanitized.append(clean)

                image_observations = [
                    item for item in observations if item.image_id == image_id
                ]
                overlay_path: Path | None = None
                if image_observations:
                    trash_masks = [
                        np.load(item.mask_path) for item in image_observations
                        if item.mask_path and Path(item.mask_path).exists()
                    ]
                    if trash_masks:
                        overlay_path = overlay_dir / f"{image_id}.jpg"
                        with r.tool(
                            "pillow.overlay",
                            f"{image_id} · {len(trash_masks)} trash mask(s)",
                        ):
                            perception.overlay_cached(str(out), trash_masks, str(overlay_path))
                    for item in image_observations:
                        item.image_path = str(out)
                        item.overlay_path = str(overlay_path) if overlay_path else None

                trash_row = trash_rows.get(image_id)
                if trash_row is not None:
                    trash_row["sanitized_url"] = self._url(out)
                    trash_row["green_overlay_path"] = str(overlay_path) if overlay_path else None
                    trash_row["green_overlay_url"] = self._url(overlay_path) if overlay_path else None

                r.screen_progress(image_id, completed, total)

            # Only observations backed by a file under sanitized/ may publish.
            sanitized_root = sanitized_dir.resolve()
            publishable: list[Observation] = []
            for item in observations:
                if item.image_id not in candidate_ids or not item.image_path:
                    continue
                image_path = Path(item.image_path).resolve()
                try:
                    image_path.relative_to(sanitized_root)
                except ValueError:
                    continue
                if image_path.is_file():
                    publishable.append(item)
            observations = publishable

            # Drop unredacted paths so they cannot leak downstream.
            for trash_row in trash_rows.values():
                trash_row.pop("image_path", None)

            stats = {
                "candidate_images": len(candidate_ids),
                "sanitized": len(sanitized), "redacted": redacted,
                "copied_no_humans": copied, "human_masks": total_human_masks,
                "sam3_person_prompts": person_prompts,
            }

            detection_stats = {
                **state.get("detection_stats", {}),
                "publishable_observations": len(observations),
            }
            r.log(
                f"Published {len(sanitized)} sanitized images "
                f"({redacted} blurred, {copied} clean); "
                f"{len(observations)} observations publishable; "
                f"{total_human_masks} human mask(s) segmented"
            )
            r.artifact("privacy_segmentation", stats)
            r.artifact("detections", dumps(observations))
            return {
                "sanitized_images": sanitized,
                "trash_screens": list(trash_rows.values()),
                "trash_review_queue": state.get("trash_review_queue", []),
                "observations": dumps(observations),
                "privacy_stats": stats, "detection_stats": detection_stats,
                "manifest": {"blur_humans": {
                    "method": "sam3_only",
                    "person_prompts": person_prompts,
                    "sanitized": len(sanitized), "redacted": redacted, "copied": copied,
                    "human_masks": total_human_masks,
                    "published_observations": len(observations),
                    "blur_radius": self.cfg["privacy_blur_radius"],
                    "dilation_px": self.cfg["privacy_dilation_px"],
                }},
            }

    def screen_trash(self, state: AgentState) -> dict:
        """
        One vision call per image, run concurrently.

        Each image is an independent question, so the calls are dispatched to a
        small thread pool instead of a serial loop. Results are written back by
        index, so the row order still matches the fetched image order and the
        node stays deterministic regardless of completion order.
        """
        with self.report.step("screen_trash") as r:
            vision = self.provider.get("vision")
            system = (
                "Inspect street-level imagery for excessive roadside trash. "
                "Count substantial uncontained piles or accumulations on a road edge, verge, "
                "footpath, alley, or open roadside ground. "
                "Do not count scattered light litter or a few isolated items. "
                "For each candidate provide a list of 2 to 4 distinct, concise text prompts "
                "suitable for segmenting that pile with an instance segmentation model."
            )
            user = "Assess this image for excessive roadside trash using the schema."
            images = state.get("images", [])
            total = len(images)
            workers = max(1, min(int(self.cfg["screen_workers"]), total or 1))
            budget = int(self.cfg["screen_max_tokens"])
            image_px = int(self.cfg["screen_image_px"])

            rows: list[dict[str, Any] | None] = [None] * total
            progress_lock = threading.Lock()
            completed = 0

            def screen_one(index: int, image: dict[str, Any]) -> None:
                nonlocal completed
                self._check_cancel()
                screen: TrashScreen | None = None
                error: str | None = None
                try:
                    screen = vision.structured(
                        system, user, TrashScreen,
                        images=[jpeg_bytes(image["local_path"], max_side=image_px)],
                        max_tokens=budget,
                    )
                    self._check_cancel()
                except RunCancelled:
                    raise
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                rows[index] = {
                    "image_id": image["image_id"], "image_path": image["local_path"],
                    "assessment": as_dump(screen) if screen else None,
                    "model_error": error,
                }
                if error:
                    r.log(f"{image['image_id']}: trash screen failed: {error}", level="warn")
                with progress_lock:
                    completed += 1
                    done = completed
                r.screen_progress(image["image_id"], done, total)

            with r.tool(
                f"{vision.model}.structured",
                f"screening {total} images",
            ):
                if workers == 1:
                    for index, image in enumerate(images):
                        screen_one(index, image)
                else:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [
                            pool.submit(screen_one, index, image)
                            for index, image in enumerate(images)
                        ]
                        # Surface the first real failure rather than a silent gap.
                        for future in futures:
                            future.result()

            screened = [row for row in rows if row is not None]
            r.log(
                f"Trash-screened {len(screened)} original images with {vision.model} "
                f"({workers} concurrent call(s), max_tokens={budget})"
            )
            return {
                "trash_screens": screened,
                "manifest": {"screen_trash": {
                    "model": vision.model, "sampling": vision.params,
                    "source": "original_fetched_images",
                    "images_attempted": len(screened),
                    "model_errors": sum(bool(row["model_error"]) for row in screened),
                    "criterion": "excessive_roadside_trash_not_scattered_litter",
                    "concurrency": workers,
                    "max_tokens": budget,
                    "image_max_side_px": image_px,
                }},
            }

    def segment_trash(self, state: AgentState) -> dict:
        with self.report.step("segment_trash") as r:
            segmenter = self._get_segmenter()
            vision_name = self._vision_label()
            image_meta = {image["image_id"]: image for image in state.get("images", [])}
            mask_dir = self.run_dir / "trash_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            observations: list[Observation] = []
            rows: list[dict[str, Any]] = []
            candidate_ids: list[str] = []
            low = self.cfg["review_confidence"]
            screens = state.get("trash_screens", [])
            total = len(screens)
            for completed, row in enumerate(screens, start=1):
                self._check_cancel()
                assessment = row.get("assessment")
                prompts = _candidate_prompts(assessment)
                instances: list[perception.Instance] = []
                sam_error: str | None = None
                if prompts:
                    quoted = ", ".join(f"'{prompt}'" for prompt in prompts)
                    r.log(
                        f"{row['image_id']}: {vision_name} supplied "
                        f"{len(prompts)} segmentation prompt(s): {quoted}"
                    )
                    try:
                        with r.tool(
                            "sam3.segment",
                            f"{row['image_id']} · {len(prompts)} {vision_name} prompt(s)",
                        ):
                            instances = segmenter.segment_cached(
                                row["image_id"], row["image_path"], prompts=prompts,
                                namespace="trash", category="excessive_roadside_trash",
                                check_cancel=self._check_cancel,
                            )
                        self._check_cancel()
                    except RunCancelled:
                        raise
                    except Exception as exc:
                        sam_error = f"{type(exc).__name__}: {exc}"
                elif assessment:
                    r.log(
                        f"{row['image_id']}: {vision_name} supplied no segmentation "
                        "prompts, so SAM3 was not run on this image",
                        level="warn",
                    )
                expected = int(assessment.get("pile_count", 0)) if assessment else 0
                found = len(instances)
                gemma_positive = bool(assessment and assessment["excessive_roadside_trash"])
                # Reviewer-facing prose, shown verbatim in the review modal.
                reasons: list[str] = []
                if row.get("model_error"):
                    reasons.append(f"{vision_name} failed to assess this image")
                if sam_error:
                    reasons.append("SAM3 failed to segment this image")
                if assessment and assessment["uncertainty"]:
                    reasons.append(f"{vision_name} was unsure about this image")
                if assessment and any(c["confidence"] < low for c in assessment["candidates"]):
                    reasons.append(f"{vision_name} had low confidence in at least one pile")
                if instances and any(
                    instance.score is None or instance.score < low for instance in instances
                ):
                    reasons.append("SAM3 confidence was low or unavailable for at least one mask")
                if expected != found:
                    reasons.append(
                        f"{vision_name} described {_piles(expected)} "
                        f"but SAM3 segmented {_piles(found)}"
                    )
                if gemma_positive and not instances:
                    reasons.append(
                        f"{vision_name} reported trash here but SAM3 found nothing to mask"
                    )
                needs_review = bool(instances) and bool(reasons)
                agreed_empty = not instances and not gemma_positive
                masks: list[dict[str, Any]] = []
                meta = image_meta.get(row["image_id"], {})
                for index, instance in enumerate(instances):
                    self._check_cancel()
                    detection_id = f"{row['image_id']}-{index}"
                    mask_path = mask_dir / f"{detection_id}.npy"
                    np.save(mask_path, instance.mask)
                    candidate = _candidate_for_prompt(assessment, instance.prompt)
                    status = "pending" if needs_review else (
                        "approved" if gemma_positive else "no_subject"
                    )
                    observations.append(Observation(
                        detection_id=detection_id, image_id=row["image_id"],
                        camera_lat=float(meta.get("lat", 0.0)), camera_lon=float(meta.get("lon", 0.0)),
                        compass_angle=meta.get("compass_angle"), captured_at=meta.get("captured_at"),
                        mask_area_px=int(instance.mask.sum()),
                        # Original candidate imagery is deliberately not published here.
                        image_path=None, overlay_path=None, mask_path=str(mask_path),
                        sam_prompt=instance.prompt, sam_score=instance.score,
                        screen_confidence=(candidate or {}).get("confidence"),
                        review_status=status,
                    ))
                    masks.append({"path": str(mask_path), "prompt": instance.prompt,
                                  "score": instance.score, "area_px": int(instance.mask.sum())})
                # Outside the mask loop so images without masks still define it.
                is_candidate = bool(instances)
                if is_candidate:
                    candidate_ids.append(row["image_id"])
                rows.append({
                    **row, "prompts": prompts, "sam_masks": masks, "sam_count": len(instances),
                    "sam_error": sam_error, "disagreement": needs_review,
                    "reason": "; ".join(reasons), "decision": None,
                    "is_candidate": is_candidate,
                    "notes": reasons,
                    "agreed_no_trash": agreed_empty,
                    "prompt_source": "vision_model_assessment",
                    "vision_pile_count": expected,
                })
                r.screen_progress(row["image_id"], completed, total)
            queue = [row["image_id"] for row in rows if row["disagreement"]]
            agreed_empty_images = sum(bool(row.get("agreed_no_trash")) for row in rows)
            noted_without_review = sum(
                bool(row.get("notes")) and not row["disagreement"] for row in rows
            )
            stats = {
                "screened_images": len(rows),
                "candidate_images": len(candidate_ids),
                "gemma_positive_images": sum(bool(row.get("assessment") and row["assessment"]["excessive_roadside_trash"]) for row in rows),
                "sam_positive_images": sum(bool(row["sam_count"]) for row in rows),
                "sam_masks": len(observations), "disagreements": len(queue),
                # Both models agreed on no trash, so no review was opened.
                "agreed_no_trash_images": agreed_empty_images,
                # Noted by the vision model but with no mask to decide on.
                "noted_without_review": noted_without_review,
                "model_errors": sum(bool(row.get("model_error") or row.get("sam_error")) for row in rows),
            }
            r.log(
                f"Saved {len(observations)} trash masks for {len(candidate_ids)} candidate "
                f"image(s); {agreed_empty_images} image(s) had no trash by both models"
            )
            if noted_without_review:
                r.log(
                    f"{noted_without_review} image(s) carried a note but no mask to "
                    "decide on, so no review was opened for them"
                )
            r.log(
                f"{len(queue)} image(s) need human review"
                if queue else "No image needs human review"
            )
            return {
                "trash_screens": rows, "trash_candidate_ids": candidate_ids,
                "trash_review_queue": queue, "observations": dumps(observations),
                "detection_stats": stats,
                "manifest": {"segment_trash": {
                    **stats, "namespace": "trash", "category": "excessive_roadside_trash",
                    "all_gemma_prompts_run": True, "raw_overlays_created": False,
                    "pre_redaction_detection_artifacts_emitted": False,
                    # Prompt provenance: every SAM3 prompt is model-generated,
                    # recorded per image, with no static fallback ensemble.
                    "prompt_source": "vision_model_assessment",
                    "prompt_fallback_ensemble": None,
                    "vision_model": vision_name,
                    "images_without_model_prompts": sum(
                        1 for row in rows if not row["prompts"]
                    ),
                    "prompt_sets": {row["image_id"]: row["prompts"] for row in rows},
                }},
            }

    def review_trash(self, state: AgentState) -> dict:
        with self.report.step("review_trash") as r:
            rows = {row["image_id"]: dict(row) for row in state.get("trash_screens", [])}
            observations = load_observations(state)
            queue = [
                image_id for image_id in state.get("trash_review_queue", [])
                if image_id in rows
                and not rows[image_id].get("privacy_excluded")
                and rows[image_id].get("sanitized_url")
            ]
            if not queue:
                r.log("No trash disagreements; review skipped")
                return {"observations": dumps(observations), "trash_reviews": []}
            vision_name = self._vision_label()
            payload = [{
                "image_id": image_id,
                "sanitized_base_url": rows[image_id]["sanitized_url"],
                "green_overlay_url": rows[image_id].get("green_overlay_url"),
                "reason": rows[image_id]["reason"],
                "gemma_assessment": rows[image_id].get("assessment"),
                "generated_prompts": rows[image_id].get("prompts", []),
                "sam_prompts": [mask.get("prompt") for mask in rows[image_id].get("sam_masks", [])],
                "sam_scores": [mask.get("score") for mask in rows[image_id].get("sam_masks", [])],
                "sam_count": rows[image_id].get("sam_count", 0),
                "vision_model": vision_name,
                "vision_pile_count": rows[image_id].get("vision_pile_count", 0),
                "prompt_source": rows[image_id].get("prompt_source"),
            } for image_id in queue if image_id in rows]
            decisions = interrupt({"kind": "trash_disagreement_review", "items": payload})
            self._check_cancel()
            applied = _decisions(decisions, queue)
            for image_id in queue:
                decision = applied.get(image_id, ReviewDecision(image_id=image_id, decision="exclude"))
                rows[image_id]["decision"] = decision.decision
                for observation in observations:
                    if observation.image_id == image_id:
                        observation.review_status = {
                            "approve": "approved", "no_subject": "no_subject", "exclude": "excluded",
                        }[decision.decision]
                        observation.review_note = decision.note
            reviews = [as_dump(decision) for decision in applied.values()]
            unresolved = len(queue) - len(applied)
            accepted = sum(observation.accepted for observation in observations)
            r.log(f"Applied {len(applied)} trash decisions; {accepted} observations accepted")
            r.artifact("detections", dumps(observations))
            return {
                "trash_screens": list(rows.values()), "observations": dumps(observations),
                "trash_reviews": reviews,
                "manifest": {"review_trash": {
                    "queued": len(queue), "decided": len(applied),
                    "unresolved_rejected": unresolved, "accepted_observations": accepted,
                }},
            }

    def export_citygml(self, state: AgentState) -> dict:
        """
        Publish the human-approved detections as CityGML 2.0.

        This sits after the review gate so nothing unreviewed reaches an
        interchange format. Other systems treat CityGML as authoritative, so only
        detections a person approved are written, and every feature records that
        its coordinate is the camera observation point rather than a surveyed
        position of the waste.
        """
        with self.report.step("export_citygml") as r:
            observations = state.get("observations", [])
            exportable = citygml.accepted_observations(observations)
            skipped = len(
                [o for o in observations if o.get("review_status") == "approved"]
            ) - len(exportable)
            out = self.run_dir / CITYGML_FILENAME

            with r.tool("citygml.write", f"{len(exportable)} approved detection(s)"):
                count = citygml.write(
                    str(out), observations, self.run_id,
                    place=state.get("place"), vision_model=self._vision_label(),
                )

            if skipped > 0:
                r.log(
                    f"{skipped} approved detection(s) had no usable camera coordinate "
                    "and were left out of the CityGML export",
                    level="warn",
                )
            r.log(
                f"Wrote CityGML {citygml.CITYGML_VERSION} with {count} feature(s) "
                f"to {CITYGML_FILENAME}"
                if count else
                f"Wrote an empty CityGML {citygml.CITYGML_VERSION} document: "
                "no detection was approved in this run"
            )
            # Same approved features as the CityGML, as GeoJSON for QGIS and web maps.
            geojson_out = self.run_dir / GEOJSON_FILENAME
            geojson_out.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [float(o["camera_lon"]), float(o["camera_lat"])]},
                    "properties": {key: o.get(key) for key in (
                        "detection_id", "image_id", "captured_at", "compass_angle",
                        "mask_area_px", "sam_prompt", "sam_score", "screen_confidence",
                        "review_status",
                    )} | {"position_source": "camera_gps_observation_point"},
                } for o in exportable],
            }, indent=2), encoding="utf-8")
            payload = {
                "url": self._url(out),
                "filename": CITYGML_FILENAME,
                "geojson_url": self._url(geojson_out),
                "geojson_filename": GEOJSON_FILENAME,
                "features": count,
                "skipped_without_coordinate": max(skipped, 0),
                "bytes": out.stat().st_size,
                "version": citygml.CITYGML_VERSION,
                "feature_type": "gen:GenericCityObject",
                "crs": citygml.SRS_NAME,
            }
            r.artifact("citygml", payload)
            return {
                "citygml": payload,
                "manifest": {"export_citygml": {
                    **payload,
                    "gate": "post_human_review",
                    "exported_review_status": "approved",
                    "position_source": "camera_gps_observation_point",
                    "position_is_surveyed_waste_location": False,
                }},
            }

    def report_node(self, state: AgentState) -> dict:
        with self.report.step("report") as r:
            observations = load_observations(state)
            audit = CoverageAudit.model_validate(state["coverage"])
            privacy = state.get("privacy_stats", {})
            detection = state.get("detection_stats", {})
            accepted = [item for item in observations if item.accepted]
            export = state.get("citygml", {})
            md = _brief(state, audit, privacy, detection, observations, export)
            (self.run_dir / "brief.md").write_text(md, encoding="utf-8")
            database = self._persist(r, state)
            manifest = {
                **state.get("manifest", {}),
                "database": database,
                "summary": {
                    "source_images": len(state.get("images", [])),
                    "trash_screened_original_images": detection.get("screened_images", 0),
                    "trash_candidate_images": detection.get("candidate_images", 0),
                    "sanitized_candidate_images": len(state.get("sanitized_images", [])),
                    "privacy": privacy,
                    "accepted_observations": len(accepted),
                    "citygml_features": export.get("features", 0),
                },
                "artifacts": ["brief.md", "manifest.json", CITYGML_FILENAME, GEOJSON_FILENAME],
                "reviews": {
                    "trash": state.get("trash_reviews", []),
                },
                "screening_evidence": {
                    "trash": _audit_rows(state.get("trash_screens", [])),
                },
                "llm": self.provider.manifest(),
            }
            (self.run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            r.log(f"Wrote brief and manifest to {self.run_dir}")
            r.artifact("report", {"markdown": md})
            return {"report_md": md}

    def _persist(self, r: NodeReporter, state: AgentState) -> dict[str, Any]:
        """
        Write the run to PostGIS when DATABASE_URL is set.

        The files on disk remain the record of the run, so a database failure is
        logged and noted in the manifest rather than failing the survey.
        """
        if not db.enabled():
            r.log(
                "PostGIS storage is off (DATABASE_URL not set): this run was not "
                "added to the time series",
                level="warn",
            )
            return {"enabled": False}
        try:
            with r.tool("postgis.persist_run", "runs, images, observations"):
                counts = db.persist_run({**state, "run_id": self.run_id}, vision_model=self._vision_label())
        except RunCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - persistence must not fail a run
            r.log(f"PostGIS write failed, artifacts are still on disk: {exc}", level="warn")
            return {"enabled": True, "persisted": False, "error": f"{type(exc).__name__}: {exc}"}
        r.log(
            f"Recorded in PostGIS: {counts['screened_images']} screened image(s) and "
            f"{counts['observations']} observation(s) "
            f"({counts['approved_observations']} approved), added to the time series"
        )
        return {"enabled": True, "persisted": True, **counts}

    def _vision_label(self) -> str:
        """Short model-family name used in reviewer-facing prose."""
        try:
            return _model_label(self.provider.get("vision").model)
        except Exception:  # noqa: BLE001 - naming must never fail a run
            return "the vision model"

    def _get_segmenter(self) -> perception.Segmenter:
        if self._segmenter is None:
            self._segmenter = perception.Segmenter(
                weights_dir=self.cfg["sam3_dir"], revision=self.cfg["sam3_revision"],
                confidence=self.cfg["confidence"], min_area_pct=self.cfg["min_area_pct"],
            )
        return self._segmenter

    def _url(self, path: str | Path) -> str:
        relative = Path(path).resolve().relative_to(self.run_dir.resolve())
        return f"/artifact/{self.run_id}/{relative.as_posix()}"





def _model_label(model_id: str) -> str:
    """Map a resolved model ID to a short family name a reviewer will recognise."""
    lowered = (model_id or "").lower()
    for needle, label in (
        ("gemma", "Gemma"), ("qwen", "Qwen"), ("llava", "LLaVA"),
        ("pixtral", "Pixtral"), ("intern", "InternVL"),
        ("molmo", "Molmo"), ("gpt", "GPT"),
    ):
        if needle in lowered:
            return label
    return "the vision model"


def _piles(count: int) -> str:
    """'no trash piles' / '1 trash pile' / '3 trash piles'."""
    if count == 0:
        return "no trash piles"
    return f"{count} trash pile" if count == 1 else f"{count} trash piles"


def _candidate_prompts(assessment: dict[str, Any] | None) -> list[str]:
    prompts: list[str] = []
    for candidate in (assessment or {}).get("candidates", []):
        for prompt in candidate.get("prompt_variants", []):
            clean = str(prompt).strip()
            if clean and clean not in prompts:
                prompts.append(clean)
    return prompts


def _candidate_for_prompt(
    assessment: dict[str, Any] | None, prompt: str
) -> dict[str, Any] | None:
    for candidate in (assessment or {}).get("candidates", []):
        if prompt in candidate.get("prompt_variants", []):
            return candidate
    return None


def _decisions(raw: Any, allowed_image_ids: list[str]) -> dict[str, ReviewDecision]:
    body = raw or {}
    values = body.get("decisions", []) if isinstance(body, dict) else []
    allowed = set(allowed_image_ids)
    out: dict[str, ReviewDecision] = {}
    for value in values:
        try:
            decision = ReviewDecision.model_validate(value)
        except Exception:
            continue
        if decision.image_id in allowed:
            out[decision.image_id] = decision
    return out


def _grid_polygons(bbox: list[float], steps: int = 4) -> list[tuple[str, str | None, Any]]:
    """Simple grid coverage cells. No Overpass needed."""
    from shapely.geometry import box

    west, south, east, north = bbox
    dx, dy = (east - west) / steps, (north - south) / steps
    return [
        (f"grid{i}{j}", f"grid cell {i},{j}",
         box(west + i * dx, south + j * dy, west + (i + 1) * dx, south + (j + 1) * dy))
        for i in range(steps) for j in range(steps)
    ]


def _area_km2(polygon: Any) -> float:
    from pyproj import Transformer
    from shapely.ops import transform

    centroid = polygon.centroid
    epsg = geo.utm_epsg(centroid.x, centroid.y)
    project = Transformer.from_crs(4326, epsg, always_xy=True).transform
    return transform(project, polygon).area / 1e6


def _count_within(points: list[dict[str, Any]], polygon: Any) -> int:
    from shapely.geometry import Point

    return sum(polygon.contains(Point(point["lon"], point["lat"])) for point in points)


def _brief(
    state: AgentState,
    audit: CoverageAudit,
    privacy: dict[str, Any],
    detection: dict[str, Any],
    observations: list[Observation],
    citygml_stats: dict[str, Any] | None = None,
) -> str:
    accepted = [item for item in observations if item.accepted]
    citygml_stats = citygml_stats or {}
    lines = [
        "# Roadside trash survey: run brief", "",
        f"**Area** {state.get('place')}  ",
        f"**Run** `{state.get('run_id')}`", "",
        "## Imagery availability", "", audit.interpretation, "",
        f"- Source images available: {audit.n_images}",
        f"- Audited coverage cells: {audit.n_units}",
        f"- Cells containing imagery: {audit.n_cells_with_imagery}",
        f"- Typical density where imagery exists: {audit.median_images_per_km2:g} images/km²", "",
        "## Trash detections", "",
        f"- Original images screened for trash: {detection.get('screened_images', 0)}",
        f"- Images the vision model called excessive roadside trash: {detection.get('gemma_positive_images', 0)}",
        f"- Images with SAM3 trash masks: {detection.get('sam_positive_images', 0)}",
        f"- Candidate images: {detection.get('candidate_images', 0)}",
        f"- SAM3 trash masks: {detection.get('sam_masks', 0)}",
        f"- Disagreements: {detection.get('disagreements', 0)}", "",
        "## Privacy (SAM3-only)", "",
        f"- Candidate images sanitized: {privacy.get('sanitized', 0)}",
        f"- Images with detected humans (blurred): {privacy.get('redacted', 0)}",
        f"- Images with no humans detected: {privacy.get('copied_no_humans', 0)}", "",
        "## Human-in-the-loop review", "",
        f"- Accepted observations after review: {len(accepted)}", "",
        "## CityGML export", "",
        f"- Approved detections exported: {citygml_stats.get('features', 0)}",
        f"- File: `{citygml_stats.get('filename', CITYGML_FILENAME)}` "
        f"(CityGML {citygml_stats.get('version', citygml.CITYGML_VERSION)}, "
        f"{citygml_stats.get('feature_type', 'gen:GenericCityObject')})", "",
        "## Coordinates", "",
        "Each exported detection is placed at the GPS position the camera recorded, "
        "which is the point the waste was observed from. This agent does not infer a "
        "position for the waste itself: doing so from a single uncalibrated "
        "street-level photograph required a depth estimate, a compass reading and an "
        "assumed field of view, and the result could not be validated. That inference "
        "was removed rather than reported with false precision.", "",
    ]
    return "\n".join(lines)


def _audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep decision evidence in the manifest without duplicating local file paths."""
    keys = (
        "image_id", "assessment", "model_error", "prompts", "sam_count",
        "sam_error", "reason", "notes", "agreed_no_trash", "vision_pile_count",
        "disagreement", "decision", "review_note",
    )
    return [{key: row.get(key) for key in keys if key in row} for row in rows]
