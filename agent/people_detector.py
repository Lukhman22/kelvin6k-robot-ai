"""
agent/people_detector.py - Dedicated Site People Detector for Camera 2.

Analyzes frames from dedicated Camera 2 to count personnel currently visible on site.
Uses Google Gemini Vision (config.GEMINI_MODEL via google-genai SDK) as the primary
multimodal detector to reliably detect workers with safety helmets, hi-vis vests,
and partial equipment occlusions. Falls back to local OpenCV HOG+NMS.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from agent import config
from agent.logging_setup import get_logger

log = get_logger("PEOPLE-DETECTOR")


@dataclass
class DetectedEntity:
    """Normalized structured detection entity for YOLO instance segmentation."""
    class_id: int
    class_name: str
    confidence: float
    box: list  # [x, y, width, height]
    width_px: int
    height_px: int
    mask: Optional[list] = None  # [[x, y], ...] contour polygon
    track_id: Optional[int] = None
    relation: Optional[str] = None  # e.g. "wearing" | "holding" | None
    person_track_id: Optional[int] = None  # track_id of associated person entity

    @property
    def bbox(self) -> list:
        return self.box

    @property
    def width(self) -> int:
        return self.width_px

    @property
    def height(self) -> int:
        return self.height_px

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class": self.class_name,
            "confidence": self.confidence,
            "box": self.box,
            "bbox": self.box,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width": self.width_px,
            "height": self.height_px,
            "mask": self.mask,
            "track_id": self.track_id,
            "relation": self.relation,
            "person_track_id": self.person_track_id,
        }


@dataclass
class PeopleCountResult:
    people_count: Optional[int]
    confidence: float = 0.0
    details: str = ""
    available: bool = True
    reason: str = ""
    detector: str = "yolo"  # "gemini" | "yolo" | "hog" | "mock" | "none"
    timestamp: float = field(default_factory=time.time)
    boxes: list = field(default_factory=list)  # list of ([x, y, w, h], confidence) for backwards-compatibility
    entities: list = field(default_factory=list)  # list of DetectedEntity
    supported_classes: list = field(default_factory=list)

    def entities_dict(self) -> List[Dict[str, Any]]:
        out = []
        for e in self.entities:
            if hasattr(e, "to_dict"):
                out.append(e.to_dict())
            elif isinstance(e, dict):
                out.append(e)
        return out


_detector_override: Optional[Callable[[bytes], Optional[PeopleCountResult]]] = None


def set_detector_override(fn: Optional[Callable[[bytes], Optional[PeopleCountResult]]]) -> None:
    """Allows tests or harnesses to supply a mock or predetermined detector result."""
    global _detector_override
    _detector_override = fn


class PeopleDetector:
    """Detects, tracks, and segments entities in site camera frames using Ultralytics YOLO segmentation."""

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        object_confidence_threshold: Optional[float] = None,
        eyewear_model_name: Optional[str] = None,
        eyewear_confidence_threshold: Optional[float] = None,
        eyewear_consecutive_frames: Optional[int] = None,
        eyewear_max_missed_frames: Optional[int] = None,
        fire_confidence_threshold: Optional[float] = None,
        lighter_confidence_threshold: Optional[float] = None,
    ):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.PEOPLE_DETECTOR_CONFIDENCE_THRESHOLD
        )
        self.object_confidence_threshold = (
            object_confidence_threshold
            if object_confidence_threshold is not None
            else getattr(config, "SITE_CAMERA_OBJECT_CONFIDENCE_THRESHOLD", 0.30)
        )
        self.model_name = model_name or getattr(config, "SITE_CAMERA_MODEL_NAME", "yolov8n-seg.pt")
        self.eyewear_model_name = (
            eyewear_model_name
            or getattr(config, "SITE_CAMERA_EYEWEAR_MODEL_NAME", "yolov8s-world.pt")
        )
        self.eyewear_confidence_threshold = (
            eyewear_confidence_threshold
            if eyewear_confidence_threshold is not None
            else getattr(config, "SITE_CAMERA_EYEWEAR_CONFIDENCE_THRESHOLD", 0.50)
        )
        self.eyewear_consecutive_frames = (
            eyewear_consecutive_frames
            if eyewear_consecutive_frames is not None
            else getattr(config, "SITE_CAMERA_EYEWEAR_CONSECUTIVE_FRAMES", 2)
        )
        self.eyewear_max_missed_frames = (
            eyewear_max_missed_frames
            if eyewear_max_missed_frames is not None
            else getattr(config, "SITE_CAMERA_EYEWEAR_MAX_MISSED_FRAMES", 1)
        )
        self.fire_confidence_threshold = (
            fire_confidence_threshold
            if fire_confidence_threshold is not None
            else getattr(config, "SITE_CAMERA_FIRE_CONFIDENCE_THRESHOLD", 0.25)
        )
        self.lighter_confidence_threshold = (
            lighter_confidence_threshold
            if lighter_confidence_threshold is not None
            else getattr(config, "SITE_CAMERA_LIGHTER_CONFIDENCE_THRESHOLD", 0.25)
        )
        self._model = None
        self._eyewear_model = None
        self._hog: Optional[cv2.HOGDescriptor] = None
        self._eyewear_tracks: List[Dict[str, Any]] = []
        self._fire_active: bool = False
        self.on_fire_alert: Optional[Callable[[str], None]] = None
        self.fire_alert_count: int = 0

    def reset_eyewear_tracks(self) -> None:
        """Clears eyewear temporal tracking history."""
        self._eyewear_tracks = []

    def reset_fire_alert(self) -> None:
        """Clears active fire alert state."""
        self._fire_active = False

    def _ensure_yolo(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def _ensure_eyewear_model(self):
        if self._eyewear_model is None:
            import os
            model_path = self.eyewear_model_name
            if os.path.exists(model_path):
                try:
                    from ultralytics import YOLO
                    m = YOLO(model_path)
                    m.set_classes(["eyewear", "glasses", "sunglasses", "safety glasses", "goggles", "person", "fire", "flame", "lighter"])
                    self._eyewear_model = m
                except Exception as exc:
                    log.warning("[PEOPLE-DETECTOR] Zero-shot/eyewear model could not be loaded: %s", exc)
                    return None
            else:
                log.warning("[PEOPLE-DETECTOR] Zero-shot/eyewear model not found at %s", model_path)
        return self._eyewear_model

    def get_supported_classes(self) -> List[str]:
        """Returns the list of object classes supported by the loaded YOLO model and PPE detector."""
        classes = []
        try:
            model = self._ensure_yolo()
            if hasattr(model, "names") and isinstance(model.names, dict):
                classes = list(model.names.values())
        except Exception:
            pass
        # Normalized detection architecture: PPE & accessories & safety objects
        for ppe in ("eyewear", "sunglasses", "safety glasses", "safety helmet", "gloves", "fire", "flame", "lighter"):
            if ppe not in classes:
                classes.append(ppe)
        return classes

    def is_class_supported(self, class_name: str) -> bool:
        """Returns True if the class is supported by the loaded YOLO model."""
        classes = {c.lower() for c in self.get_supported_classes()}
        c_lower = (class_name or "").strip().lower()
        if not c_lower:
            return False
        if c_lower in classes:
            return True
        if c_lower.endswith("s") and c_lower[:-1] in classes:
            return True
        if c_lower.endswith("es") and c_lower[:-2] in classes:
            return True
        return False

    @staticmethod
    def _compute_box_overlap(box1: List[int], box2: List[int]) -> Tuple[float, float]:
        """
        Computes (IoU, IoM) between two bounding boxes [x, y, w, h].
        IoU = Intersection over Union
        IoM = Intersection over Minimum Area (containment ratio)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        iw = max(0, xi2 - xi1)
        ih = max(0, yi2 - yi1)
        inter_area = iw * ih
        area1 = max(1, w1 * h1)
        area2 = max(1, w2 * h2)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        min_area = min(area1, area2)
        iom = inter_area / min_area if min_area > 0 else 0.0
        return iou, iom

    def _deduplicate_person_entities(
        self, entities: List[DetectedEntity]
    ) -> List[DetectedEntity]:
        """
        Deduplicates multiple person bounding boxes belonging to the same physical person.
        Suppresses sub-box/torso duplicates and shifted duplicate detections while:
        1. Preserving distinct real people (whose overlap is well below IoU 0.50 / IoM 0.65).
        2. Preserving all non-person objects (bottles, cell phones, chairs, etc.).
        3. Preserving tracking ID continuity.
        """
        non_persons = [e for e in entities if e.class_name.lower() != "person"]
        persons = [e for e in entities if e.class_name.lower() == "person"]

        if len(persons) <= 1:
            return entities

        # Sort candidates primarily by confidence descending
        sorted_persons = sorted(persons, key=lambda e: e.confidence, reverse=True)
        kept_persons: List[DetectedEntity] = []

        for candidate in sorted_persons:
            is_dup = False
            for i, existing in enumerate(kept_persons):
                iou, iom = self._compute_box_overlap(candidate.box, existing.box)
                # Overlap conditions for same physical person:
                # - High IoU (>= 0.50): duplicate shifted/redundant detection
                # - High Containment (IoM >= 0.65): upper body / torso sub-box duplicate
                if iou >= 0.50 or iom >= 0.65:
                    is_dup = True
                    area_cand = candidate.width_px * candidate.height_px
                    area_exist = existing.width_px * existing.height_px

                    # If candidate is significantly larger (full body) and has good confidence,
                    # prefer the larger box so the whole person is bounded.
                    if iom >= 0.65 and area_cand > 1.3 * area_exist and candidate.confidence >= 0.60:
                        # Inherit track_id
                        if existing.track_id is not None and candidate.track_id is None:
                            candidate.track_id = existing.track_id
                        elif existing.track_id is not None and candidate.track_id is not None:
                            candidate.track_id = min(existing.track_id, candidate.track_id)
                        kept_persons[i] = candidate
                    else:
                        # Keep existing; inherit track_id if candidate had one and existing didn't
                        if candidate.track_id is not None and existing.track_id is None:
                            existing.track_id = candidate.track_id
                        elif candidate.track_id is not None and existing.track_id is not None:
                            existing.track_id = min(existing.track_id, candidate.track_id)
                    break

            if not is_dup:
                kept_persons.append(candidate)

        return non_persons + kept_persons

    def _detect_eyewear(self, frame: np.ndarray) -> List[DetectedEntity]:
        """
        Zero-shot eyewear detection using YOLO-World.
        Detects visible eyewear (sunglasses, prescription glasses, reading glasses, safety glasses, goggles)
        and converts detections to separate DetectedEntity instances with class_name="eyewear".
        """
        if self._eyewear_model is False:
            return []
        if self._eyewear_model is None and hasattr(self._model, "_mock_return_value"):
            return []

        eyewear_entities: List[DetectedEntity] = []
        try:
            eyewear_model = self._ensure_eyewear_model()
            if eyewear_model is None:
                return []

            h, w = frame.shape[:2]
            results = eyewear_model.predict(
                source=frame,
                conf=self.eyewear_confidence_threshold,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )
            if not results or len(results) == 0 or results[0].boxes is None:
                return []

            r = results[0]
            eyewear_class_names = {"eyewear", "glasses", "sunglasses", "safety glasses", "goggles"}
            raw_eyewear_boxes = []

            for box in r.boxes:
                val_cls = box.cls[0]
                if hasattr(val_cls, "cpu"):
                    val_cls = val_cls.cpu()
                cls_id = int(val_cls)
                c_name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)

                if c_name.lower() in eyewear_class_names:
                    val_conf = box.conf[0]
                    if hasattr(val_conf, "cpu"):
                        val_conf = val_conf.cpu()
                    conf = float(val_conf)
                    if conf < self.eyewear_confidence_threshold:
                        continue

                    val_xyxy = box.xyxy[0]
                    if hasattr(val_xyxy, "cpu"):
                        val_xyxy = val_xyxy.cpu()
                    if hasattr(val_xyxy, "numpy"):
                        val_xyxy = val_xyxy.numpy()
                    xyxy = np.asarray(val_xyxy)

                    bx = max(0, min(w - 1, int(xyxy[0])))
                    by = max(0, min(h - 1, int(xyxy[1])))
                    bw = max(1, min(w - bx, int(xyxy[2] - xyxy[0])))
                    bh = max(1, min(h - by, int(xyxy[3] - xyxy[1])))
                    raw_eyewear_boxes.append((conf, [bx, by, bw, bh]))

            # Non-Maximum Suppression (NMS) across eyewear candidates
            if raw_eyewear_boxes:
                raw_eyewear_boxes.sort(key=lambda item: item[0], reverse=True)
                kept_boxes = []
                for conf, b in raw_eyewear_boxes:
                    overlap = False
                    for _, kept_b in kept_boxes:
                        iou, _ = self._compute_box_overlap(b, kept_b)
                        if iou > 0.40:
                            overlap = True
                            break
                    if not overlap:
                        kept_boxes.append((conf, b))

                for conf, b in kept_boxes:
                    bx, by, bw, bh = b
                    entity = DetectedEntity(
                        class_id=1,
                        class_name="eyewear",
                        confidence=round(conf, 2),
                        box=[bx, by, bw, bh],
                        width_px=bw,
                        height_px=bh,
                        mask=None,
                        track_id=None,
                    )
                    eyewear_entities.append(entity)
        except Exception as exc:
            log.warning("[PEOPLE-DETECTOR] Eyewear detection error: %s", exc)

        return eyewear_entities

    def _update_eyewear_temporal(
        self, raw_eyewear: List[DetectedEntity], persist: bool = True
    ) -> List[DetectedEntity]:
        """
        Applies simple temporal stability to eyewear detections:
        1. Eyewear must be detected in consecutive frames (>= eyewear_consecutive_frames)
           before being promoted to visible.
        2. Once visible, briefly tolerates missed frames (<= eyewear_max_missed_frames)
           to prevent bounding box flickering.
        3. Single-frame false detections are immediately dropped on the next frame.
        """
        if not persist or self.eyewear_consecutive_frames <= 1:
            if not persist:
                self._eyewear_tracks = []
            return raw_eyewear

        # Match current detections with existing active tracks
        matched_track_indices = set()
        matched_detection_indices = set()

        candidates = []
        for det_idx, det in enumerate(raw_eyewear):
            for trk_idx, trk in enumerate(self._eyewear_tracks):
                iou, iom = self._compute_box_overlap(det.box, trk["box"])
                if iou >= 0.20 or iom >= 0.25:
                    candidates.append((max(iou, iom), det_idx, trk_idx))

        # Sort candidate matches by overlap descending (greedy assignment)
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, det_idx, trk_idx in candidates:
            if det_idx in matched_detection_indices or trk_idx in matched_track_indices:
                continue
            matched_detection_indices.add(det_idx)
            matched_track_indices.add(trk_idx)

            det = raw_eyewear[det_idx]
            trk = self._eyewear_tracks[trk_idx]
            trk["box"] = det.box
            trk["confidence"] = det.confidence
            trk["entity"] = det
            trk["consecutive_hits"] += 1
            trk["missed_frames"] = 0

        # Handle unmatched active tracks (missed detections in this frame)
        updated_tracks = []
        for trk_idx, trk in enumerate(self._eyewear_tracks):
            if trk_idx in matched_track_indices:
                updated_tracks.append(trk)
            else:
                # If this track was already confirmed (consecutive hits met),
                # briefly tolerate a missed frame.
                if trk["consecutive_hits"] >= self.eyewear_consecutive_frames:
                    trk["missed_frames"] += 1
                    if trk["missed_frames"] <= self.eyewear_max_missed_frames:
                        updated_tracks.append(trk)
                # Else: unconfirmed track (hits < required) that missed a frame is dropped immediately

        # Handle unmatched detections (new eyewear candidate tracks)
        for det_idx, det in enumerate(raw_eyewear):
            if det_idx not in matched_detection_indices:
                updated_tracks.append({
                    "box": det.box,
                    "confidence": det.confidence,
                    "entity": det,
                    "consecutive_hits": 1,
                    "missed_frames": 0,
                })

        self._eyewear_tracks = updated_tracks

        # Return only tracks that satisfy consecutive hit requirements and haven't exceeded missed frame tolerance
        visible_entities: List[DetectedEntity] = []
        for trk in self._eyewear_tracks:
            if (
                trk["consecutive_hits"] >= self.eyewear_consecutive_frames
                and trk["missed_frames"] <= self.eyewear_max_missed_frames
            ):
                ent = trk["entity"]
                bx, by, bw, bh = trk["box"]
                stable_ent = DetectedEntity(
                    class_id=ent.class_id,
                    class_name="eyewear",
                    confidence=trk["confidence"],
                    box=[bx, by, bw, bh],
                    width_px=bw,
                    height_px=bh,
                    mask=ent.mask,
                    track_id=None,
                    relation=None,
                    person_track_id=None,
                )
                visible_entities.append(stable_ent)

        return visible_entities

    def _detect_fire_and_lighter(self, frame: np.ndarray) -> List[DetectedEntity]:
        """Detects fire, flame, and lighter using YOLO-World across near/medium/far distances."""
        entities: List[DetectedEntity] = []
        model = self._ensure_eyewear_model()
        if model is None or frame is None or frame.size == 0:
            return entities
        try:
            h, w = frame.shape[:2]
            min_conf = min(self.fire_confidence_threshold, self.lighter_confidence_threshold)
            results = model.predict(source=frame, conf=min_conf, iou=0.45, imgsz=640, verbose=False)
            if not results or not results[0].boxes:
                return entities
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0].cpu() if hasattr(box.cls[0], "cpu") else box.cls[0])
                c_name = (r.names.get(cls_id, "") if hasattr(r, "names") else "").lower().strip()
                conf = float(box.conf[0].cpu() if hasattr(box.conf[0], "cpu") else box.conf[0])
                if c_name in ("fire", "flame") and conf >= self.fire_confidence_threshold:
                    label = "fire"
                elif c_name == "lighter" and conf >= self.lighter_confidence_threshold:
                    label = "lighter"
                else:
                    continue

                xyxy = np.asarray(box.xyxy[0].cpu() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0])
                bx, by = max(0, min(w - 1, int(xyxy[0]))), max(0, min(h - 1, int(xyxy[1])))
                bw, bh = max(1, min(w - bx, int(xyxy[2] - xyxy[0]))), max(1, min(h - by, int(xyxy[3] - xyxy[1])))

                # Deduplicate overlapping fire & flame detections
                if label == "fire" and any(e.class_name == "fire" and self._compute_box_overlap([bx, by, bw, bh], e.box)[0] > 0.40 for e in entities):
                    continue

                entities.append(DetectedEntity(
                    class_id=cls_id, class_name=label, confidence=round(conf, 2),
                    box=[bx, by, bw, bh], width_px=bw, height_px=bh, mask=None, track_id=None
                ))
        except Exception as exc:
            log.warning("[PEOPLE-DETECTOR] Fire/lighter detection error: %s", exc)
        return entities

    def _trigger_fire_alert(self, message: str = "Warning. Fire detected on Camera 2.") -> None:
        """Dispatches spoken voice alert once per fire event via the TTS system."""
        log.warning("[FIRE-ALERT] %s", message)
        if self.on_fire_alert is not None:
            try:
                self.on_fire_alert(message)
                return
            except Exception as exc:
                log.warning("on_fire_alert callback error: %s", exc)
        try:
            from agent.voice import speak
            import threading
            threading.Thread(target=speak, args=(message,), daemon=True, name="fire-tts-alert").start()
        except Exception as exc:
            log.warning("Failed to invoke speak for fire alert: %s", exc)

    def detect_frame(self, frame: np.ndarray, persist: bool = True) -> PeopleCountResult:
        """
        Runs Ultralytics YOLO instance segmentation and tracking on an in-memory BGR frame.
        Detects and segments persons and all classes supported by the loaded model.
        """
        if frame is None or frame.size == 0:
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason="Invalid or empty camera frame.",
                detector="yolo",
            )

        try:
            model = self._ensure_yolo()
            # Run YOLO segmentation model with tracking enabled
            # Note: We do NOT pass classes=[0] so that all classes supported by the model
            # (persons, handheld items like bottles/phones, chairs, etc.) are detected & segmented.
            results = None
            try:
                results = model.track(
                    source=frame,
                    conf=self.object_confidence_threshold,
                    iou=0.50,
                    imgsz=640,
                    persist=persist,
                    verbose=False,
                )
            except Exception:
                results = model.predict(
                    source=frame,
                    conf=self.object_confidence_threshold,
                    iou=0.50,
                    imgsz=640,
                    verbose=False,
                )

            h, w = frame.shape[:2]
            detected_entities: List[DetectedEntity] = []
            detected_person_boxes = []
            supported_classes = (
                list(model.names.values())
                if hasattr(model, "names") and isinstance(model.names, dict)
                else []
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                r = results[0]
                masks_xy = r.masks.xy if (r.masks is not None and hasattr(r.masks, "xy")) else None

                for idx, box in enumerate(r.boxes):
                    val_xyxy = box.xyxy[0]
                    if hasattr(val_xyxy, "cpu"):
                        val_xyxy = val_xyxy.cpu()
                    if hasattr(val_xyxy, "numpy"):
                        val_xyxy = val_xyxy.numpy()
                    xyxy = np.asarray(val_xyxy)

                    val_conf = box.conf[0]
                    if hasattr(val_conf, "cpu"):
                        val_conf = val_conf.cpu()
                    conf = float(val_conf)

                    val_cls = box.cls[0]
                    if hasattr(val_cls, "cpu"):
                        val_cls = val_cls.cpu()
                    cls_id = int(val_cls)

                    class_name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
                    track_id = None
                    if hasattr(box, "id") and box.id is not None and len(box.id) > 0 and box.id[0] is not None:
                        val_id = box.id[0]
                        if hasattr(val_id, "cpu"):
                            val_id = val_id.cpu()
                        track_id = int(val_id)

                    bx = max(0, min(w - 1, int(xyxy[0])))
                    by = max(0, min(h - 1, int(xyxy[1])))
                    bw = max(1, min(w - bx, int(xyxy[2] - xyxy[0])))
                    bh = max(1, min(h - by, int(xyxy[3] - xyxy[1])))

                    mask_poly = None
                    if masks_xy is not None and idx < len(masks_xy):
                        poly_pts = masks_xy[idx]
                        if poly_pts is not None and len(poly_pts) > 0:
                            mask_poly = [[int(pt[0]), int(pt[1])] for pt in poly_pts]

                    entity = DetectedEntity(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=round(conf, 2),
                        box=[bx, by, bw, bh],
                        width_px=bw,
                        height_px=bh,
                        mask=mask_poly,
                        track_id=track_id,
                    )
                    detected_entities.append(entity)

            # Deduplicate person entities to prevent multiple detections / track IDs for same physical person
            detected_entities = self._deduplicate_person_entities(detected_entities)

            for e in detected_entities:
                if e.class_name.lower() == "person" and e.confidence >= self.confidence_threshold:
                    detected_person_boxes.append((e.box, e.confidence))

            # Count detected person instances correctly; deduplicate by track ID if tracking is available
            person_entities = [
                e for e in detected_entities
                if e.class_name.lower() == "person" and e.confidence >= self.confidence_threshold
            ]
            tracked_person_ids = {e.track_id for e in person_entities if e.track_id is not None}
            untracked_person_count = sum(1 for e in person_entities if e.track_id is None)
            people_count = len(tracked_person_ids) + untracked_person_count

            # Zero-shot eyewear detection via YOLO-World with temporal stability
            raw_eyewear = self._detect_eyewear(frame)
            eyewear_entities = self._update_eyewear_temporal(raw_eyewear, persist=persist)
            if eyewear_entities:
                detected_entities.extend(eyewear_entities)

            # Zero-shot fire and lighter detection via YOLO-World
            fire_lighter_entities = self._detect_fire_and_lighter(frame)
            if fire_lighter_entities:
                detected_entities.extend(fire_lighter_entities)

            for cl in ("eyewear", "fire", "lighter"):
                if cl not in supported_classes:
                    supported_classes.append(cl)

            # Average confidence or sensible default
            if detected_entities:
                avg_conf = float(np.mean([e.confidence for e in detected_entities]))
            else:
                avg_conf = 0.85

            class_summary = []
            entity_counts: Dict[str, int] = {}
            for e in detected_entities:
                entity_counts[e.class_name] = entity_counts.get(e.class_name, 0) + 1
            for cname, ccount in entity_counts.items():
                class_summary.append(f"{ccount} {cname}")

            details_str = ", ".join(class_summary) if class_summary else "none"

            # Voice alert trigger: once per fire event
            fire_present = any(e.class_name in ("fire", "flame") for e in detected_entities)
            if fire_present:
                if not self._fire_active:
                    self._fire_active = True
                    self.fire_alert_count += 1
                    self._trigger_fire_alert()
            else:
                self._fire_active = False

            return PeopleCountResult(
                people_count=people_count,
                confidence=round(avg_conf, 2),
                details=f"{people_count} person(s) detected via YOLO segmentation ({details_str})" if people_count else details_str,
                available=True,
                detector="yolo",
                boxes=detected_person_boxes,
                entities=detected_entities,
                supported_classes=supported_classes,
            )
        except Exception as exc:
            log.warning("[PEOPLE-DETECTOR] YOLO inference error: %s; trying local HOG fallback", exc)
            return self._detect_hog_frame(frame)

    def detect_people(self, jpeg_bytes: Optional[bytes]) -> PeopleCountResult:
        if not jpeg_bytes:
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason="No camera frame provided for analysis.",
                detector="none",
            )

        if _detector_override is not None:
            res = _detector_override(jpeg_bytes)
            if res is not None:
                return res

        # Primary: Gemini Vision via google-genai
        if config.GEMINI_API_KEY:
            try:
                gemini_res = self._detect_gemini(jpeg_bytes)
                if gemini_res is not None:
                    return gemini_res
            except Exception as exc:
                log.warning("[PEOPLE-DETECTOR] Gemini Vision call failed: %s; trying YOLO fallback", exc)

        # Fallback: Ultralytics YOLO person detection
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return PeopleCountResult(
                    people_count=None,
                    available=False,
                    reason="Could not decode JPEG frame.",
                    detector="yolo",
                )
            return self.detect_frame(frame)
        except Exception as exc:
            log.warning("[PEOPLE-DETECTOR] Local detection error: %s", exc)
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason=f"Detection error: {exc}",
                detector="yolo",
            )

    def _detect_gemini(self, jpeg_bytes: bytes) -> Optional[PeopleCountResult]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        prompt = (
            "You are an industrial site safety assistant for the Kelvin6K 3D concrete printing facility. "
            "Analyze this camera frame of the workshop/site area and count how many people (workers, "
            "operators, technicians, visitors) are currently visible. "
            "Include workers even if they are wearing hard hats, safety vests, seated, or partially occluded. "
            "Output JSON ONLY with this schema: "
            '{"people_count": <integer >= 0>, "confidence": <float 0.0-1.0>, "details": <short string describing detected locations or "none">}'
        )

        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=150,
            ),
        )

        text = (resp.text or "").strip()
        if not text:
            return None

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        count = int(data.get("people_count", 0))
        conf = float(data.get("confidence", 0.9))
        details = str(data.get("details", ""))

        return PeopleCountResult(
            people_count=count,
            confidence=conf,
            details=details,
            available=True,
            detector="gemini",
        )

    def _detect_hog_frame(self, frame: np.ndarray) -> PeopleCountResult:
        if cv2 is None:
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason="OpenCV (cv2) is not installed.",
                detector="none",
            )
        try:
            if self._hog is None:
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            h, w = frame.shape[:2]
            scale = 640.0 / max(w, 640)
            inv_scale = 1.0 / scale if scale > 0 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            boxes, weights = self._hog.detectMultiScale(
                small,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )

            if len(boxes) == 0:
                return PeopleCountResult(
                    people_count=0,
                    confidence=0.85,
                    details="none",
                    available=True,
                    detector="hog",
                    boxes=[],
                )

            rects = [[int(x), int(y), int(w_b), int(h_b)] for (x, y, w_b, h_b) in boxes]
            scores = [float(s[0]) if hasattr(s, "__len__") else float(s) for s in weights]
            indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.2, nms_threshold=0.4)

            detected_boxes = []
            if indices is not None:
                for idx in indices:
                    i = idx[0] if hasattr(idx, "__len__") else idx
                    rx, ry, rw, rh = rects[i]
                    bx = max(0, min(w - 1, int(rx * inv_scale)))
                    by = max(0, min(h - 1, int(ry * inv_scale)))
                    bw = max(10, min(w - bx, int(rw * inv_scale)))
                    bh = max(10, min(h - by, int(rh * inv_scale)))
                    s = scores[i]
                    c = round(max(0.4, min(0.99, s / 2.0 + 0.4)), 2)
                    detected_boxes.append(([bx, by, bw, bh], c))

            count = len(detected_boxes)
            avg_conf = float(np.mean(scores)) if scores else 0.7
            normalized_conf = max(0.4, min(0.9, avg_conf / 2.0 + 0.4))

            return PeopleCountResult(
                people_count=count,
                confidence=round(normalized_conf, 2),
                details=f"{count} person(s) detected via local HOG",
                available=True,
                detector="hog",
                boxes=detected_boxes,
            )
        except Exception as exc:
            log.warning("[PEOPLE-DETECTOR] Local HOG detection error: %s", exc)
            return PeopleCountResult(
                people_count=None,
                available=False,
                reason=f"Detection error: {exc}",
                detector="hog",
            )


def draw_people_overlay(
    frame: np.ndarray,
    result: PeopleCountResult,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Draws instance segmentation masks, bounding boxes, class labels with confidence
    and pixel dimensions (width_px x height_px), and live people count status banners on Camera 2.
    """
    if cv2 is None or frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    count = result.people_count or 0

    CLASS_COLORS = {
        "person": (0, 255, 0),        # Vibrant Green
        "bottle": (255, 191, 0),      # Deep Sky Blue
        "cell phone": (255, 0, 255),  # Magenta
        "backpack": (0, 215, 255),    # Gold / Yellow
        "cup": (255, 140, 0),         # Dark Orange
        "chair": (180, 105, 255),     # Pink
        "eyewear": (255, 230, 0),     # Cyan / Aqua
        "fire": (0, 69, 255),         # Red-Orange
        "flame": (0, 69, 255),        # Red-Orange
        "lighter": (0, 165, 255),     # Orange
    }

    def get_color(cname: str, cid: int = 0) -> Tuple[int, int, int]:
        cname_lower = cname.lower()
        if cname_lower in CLASS_COLORS:
            return CLASS_COLORS[cname_lower]
        palette = [
            (255, 128, 0), (0, 200, 255), (200, 0, 255), (0, 255, 200),
            (255, 255, 0), (255, 0, 128), (128, 255, 0), (0, 128, 255),
        ]
        return palette[(cid + sum(ord(ch) for ch in cname_lower)) % len(palette)]

    entities = getattr(result, "entities", [])

    if entities:
        # 1. Render segmentation masks first with alpha blending
        overlay = out.copy()
        has_mask = False
        for ent in entities:
            mask = getattr(ent, "mask", None) or (ent.get("mask") if isinstance(ent, dict) else None)
            if mask and len(mask) >= 3:
                cname = getattr(ent, "class_name", "") or (ent.get("class_name") if isinstance(ent, dict) else "object")
                cid = getattr(ent, "class_id", 0) if hasattr(ent, "class_id") else ent.get("class_id", 0)
                color = get_color(cname, cid)
                pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(out, [pts], True, color, 2)
                has_mask = True
        if has_mask:
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # 2. Render bounding boxes, corner brackets, and labels
        for ent in entities:
            box = getattr(ent, "box", None) or (ent.get("box") if isinstance(ent, dict) else None)
            if not box or len(box) < 4:
                continue
            x, y, bw, bh = box
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            bw = max(10, min(w - x, int(bw)))
            bh = max(10, min(h - y, int(bh)))

            cname = getattr(ent, "class_name", "") or (ent.get("class_name") if isinstance(ent, dict) else "object")
            cid = getattr(ent, "class_id", 0) if hasattr(ent, "class_id") else ent.get("class_id", 0)
            conf = float(getattr(ent, "confidence", 1.0) if hasattr(ent, "confidence") else ent.get("confidence", 1.0))
            track_id = getattr(ent, "track_id", None) if hasattr(ent, "track_id") else ent.get("track_id")

            color = get_color(cname, cid)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            corner_len = min(20, max(8, bw // 5), max(8, bh // 5))
            thick = 3
            cv2.line(out, (x, y), (x + corner_len, y), color, thick)
            cv2.line(out, (x, y), (x, y + corner_len), color, thick)
            cv2.line(out, (x + bw, y), (x + bw - corner_len, y), color, thick)
            cv2.line(out, (x + bw, y), (x + bw, y + corner_len), color, thick)
            cv2.line(out, (x, y + bh), (x + corner_len, y + bh), color, thick)
            cv2.line(out, (x, y + bh), (x, y + bh - corner_len), color, thick)
            cv2.line(out, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, thick)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, thick)

            id_str = f"#{track_id} " if track_id is not None else ""
            label = f"{id_str}{cname} {conf:.2f} ({bw}x{bh} px)"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = max(th + 8, y)
            cv2.rectangle(
                out, (x, label_y - th - 6), (x + tw + 8, label_y + baseline + 2), (20, 20, 25), cv2.FILLED
            )
            cv2.rectangle(
                out, (x, label_y - th - 6), (x + tw + 8, label_y + baseline + 2), color, 1
            )
            cv2.putText(
                out,
                label,
                (x + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    # Legacy / mock fallback: if result.boxes is set but entities is empty
    elif not entities and count > 0 and result.boxes:
        for box_item in result.boxes:
            box, conf = box_item[0], box_item[1]
            x, y, bw, bh = box
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            bw = max(10, min(w - x, int(bw)))
            bh = max(10, min(h - y, int(bh)))

            color = (0, 255, 0)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            corner_len = min(20, max(8, bw // 5), max(8, bh // 5))
            thick = 3
            cv2.line(out, (x, y), (x + corner_len, y), color, thick)
            cv2.line(out, (x, y), (x, y + corner_len), color, thick)
            cv2.line(out, (x + bw, y), (x + bw - corner_len, y), color, thick)
            cv2.line(out, (x + bw, y), (x + bw, y + corner_len), color, thick)
            cv2.line(out, (x, y + bh), (x + corner_len, y + bh), color, thick)
            cv2.line(out, (x, y + bh), (x, y + bh - corner_len), color, thick)
            cv2.line(out, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, thick)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, thick)

            label = f"Person {conf:.2f} ({bw}x{bh} px)"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(th + 8, y)
            cv2.rectangle(
                out, (x, label_y - th - 8), (x + tw + 10, label_y + baseline), (0, 160, 0), cv2.FILLED
            )
            cv2.putText(
                out,
                label,
                (x + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Status badge for people count (preserved from original implementation)
    if count > 0:
        badge_text = f"People Detected: {count}"
        (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(out, (16, 16), (36 + tw, 24 + th + baseline + 8), (20, 20, 25), cv2.FILLED)
        cv2.rectangle(out, (16, 16), (36 + tw, 24 + th + baseline + 8), (0, 255, 0), 2)
        cv2.circle(out, (32, 22 + th // 2), 6, (0, 255, 0), cv2.FILLED)
        cv2.putText(
            out,
            badge_text,
            (48, 20 + th + baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        badge_text = "People Detected: 0"
        warn_text = "WARNING: No Person Detected"
        (tw1, th1), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        (tw2, th2), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        box_w = max(tw1, tw2) + 50
        cv2.rectangle(out, (16, 16), (16 + box_w, 70), (20, 20, 28), cv2.FILLED)
        cv2.rectangle(out, (16, 16), (16 + box_w, 70), (0, 0, 255), 2)
        cv2.circle(out, (32, 35), 6, (0, 0, 255), cv2.FILLED)
        cv2.putText(out, badge_text, (48, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, warn_text, (48, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 1, cv2.LINE_AA)

    # Status badge for fire detection
    fire_count = sum(1 for e in entities if getattr(e, "class_name", "").lower() in ("fire", "flame"))
    if fire_count > 0:
        fire_badge_text = f"FIRE DETECTED: {fire_count}" if fire_count > 1 else "FIRE DETECTED"
        (ftw, fth), fbaseline = cv2.getTextSize(fire_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        fire_y = 80 if count == 0 else 66
        cv2.rectangle(out, (16, fire_y), (36 + ftw, fire_y + fth + fbaseline + 10), (20, 20, 28), cv2.FILLED)
        cv2.rectangle(out, (16, fire_y), (36 + ftw, fire_y + fth + fbaseline + 10), (0, 69, 255), 2)
        cv2.circle(out, (32, fire_y + fth // 2 + 5), 6, (0, 69, 255), cv2.FILLED)
        cv2.putText(
            out,
            fire_badge_text,
            (48, fire_y + fth + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 69, 255),
            2,
            cv2.LINE_AA,
        )

    return out
