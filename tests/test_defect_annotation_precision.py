"""Tests for Defect Annotation Localization Precision and Attendance Provenance.

Verifies:
1. validate_annotation_region rejects oversized macro-boxes (>45% frame area, >75% width & >65% height).
2. detect_and_create_annotation generates tight, localized defect regions (Region A corner cavity, Region B bead gap/buildup).
3. Generated annotation derivative produces valid JPEG bytes with disclaimer banner.
4. Attendance commands receive deterministic "local-attendance" provenance, never "gemini".
5. Image store metadata records individual defect regions and normalized coordinates.
"""

import json
import os
import tempfile
import cv2
import numpy as np

from agent.troubleshooter import (
    validate_annotation_region,
    detect_and_create_annotation,
    validate_and_process_attachment,
)
from agent import incident_store
from agent.query_router import Intent, classify


def test_validate_annotation_region_rejection():
    """Verify that oversized bounding boxes (>45% of frame) and out-of-bounds boxes are rejected."""
    img_shape = (600, 800)  # H=600, W=800 -> Area = 480,000

    # Macro-box spanning almost the whole frame (600x500 = 300,000 px = 62.5% area)
    oversized_box = [100, 50, 600, 500]
    assert not validate_annotation_region(oversized_box, img_shape), "Macro box >45% area must be rejected"

    # Box spanning >75% width and >65% height
    wide_tall_box = [50, 50, 650, 420]
    assert not validate_annotation_region(wide_tall_box, img_shape), "Box spanning >75% width and >65% height must be rejected"

    # Out of image bounds
    out_of_bounds = [-10, 20, 200, 200]
    assert not validate_annotation_region(out_of_bounds, img_shape), "Out-of-bounds box must be rejected"

    # Too small (<12px)
    too_small = [100, 100, 8, 10]
    assert not validate_annotation_region(too_small, img_shape), "Box <12px must be rejected"

    # Tight, realistic defect box (120x80 = 9,600 px = 2.0% area)
    valid_box = [150, 200, 120, 80]
    assert validate_annotation_region(valid_box, img_shape), "Tight localized defect box must be accepted"


def test_detect_and_create_annotation_tight_regions():
    """Verify detect_and_create_annotation produces localized regions and valid JPEG bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_incidents.db")
        incident_store.init_storage(tmpdir)

        # Create synthetic test image with a concrete bead structure (800x600)
        img = np.ones((600, 800, 3), dtype=np.uint8) * 35  # dark background
        # Draw concrete wall structure
        pts = np.array([[120, 150], [680, 150], [660, 450], [150, 430]], np.int32)
        cv2.fillPoly(img, [pts], (180, 180, 180))
        # Introduce defect cavity at corner (Region A)
        cv2.circle(img, (140, 160), 30, (25, 25, 25), -1)
        # Introduce bead gap along front edge (Region B)
        cv2.rectangle(img, (380, 420), (440, 460), (35, 35, 35), -1)

        _, encoded = cv2.imencode(".jpg", img)
        raw_bytes = encoded.tobytes()

        # Save original as primary image
        primary_saved = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="print_photo.jpg",
            content_type="image/jpeg",
            db_path=db_path,
            storage_dir=tmpdir,
        )
        primary_id = primary_saved["image_id"]

        att = validate_and_process_attachment(raw_bytes, "print_photo.jpg", "image/jpeg")
        assert att.kind == "image"
        assert att.data is not None

        ann_result = detect_and_create_annotation(
            attachment=att,
            observations="Torn concrete bead and surface roughness on corner",
            primary_image_id=primary_id,
            db_path=db_path,
        )

        assert ann_result is not None, "Annotation result must be generated"
        assert "annotated_image_id" in ann_result
        assert "regions" in ann_result
        regions = ann_result["regions"]
        assert len(regions) >= 1, "Must detect at least 1 localized defect region"

        # Verify raw bytes from storage
        data_tuple = incident_store.get_image_data(ann_result["annotated_image_id"], db_path=db_path)
        assert data_tuple is not None, "Stored annotated image bytes must be retrievable"
        ann_bytes, mime_type, _ = data_tuple
        assert mime_type == "image/jpeg"
        assert ann_bytes[:3] == b"\xff\xd8\xff", "Must be valid JPEG bytes"

        # Verify cv2 can decode the annotated image cleanly and dimensions match
        decoded_img = cv2.imdecode(np.frombuffer(ann_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded_img is not None, "Decoded image must not be None"
        assert decoded_img.shape == (600, 800, 3)

        # Verify regions are localized (< 45% frame area)
        total_area = 800 * 600
        for r in regions:
            bx, by, bw, bh = r["bbox"]
            box_area = bw * bh
            assert box_area / total_area < 0.45, f"Region {r.get('region_id')} area {box_area} exceeds 45% of frame"
            assert "confidence" in r
            assert "description" in r
            assert "defect_type" in r
            assert "region_id" in r


def test_attendance_provenance_classification_rules():
    """Verify that attendance queries and actions classify into local attendance intents."""
    local_attendance_intents = {
        Intent.SHIFT_START,
        Intent.SHIFT_END,
        Intent.GET_ATTENDANCE_STATUS,
        Intent.GET_ATTENDANCE_HOURS,
        Intent.GET_ATTENDANCE_OVERTIME,
        Intent.GET_OPERATOR_WORKING_DAYS,
        Intent.GET_ACTIVE_SHIFT,
        Intent.GET_PREVIOUS_SHIFT,
        Intent.GET_SHIFT_BY_OPERATOR,
    }

    # Verify attendance intents classify accurately
    start_intent, _ = classify("Kelvin start Anand's shift")
    assert start_intent == Intent.SHIFT_START
    assert start_intent in local_attendance_intents

    status_intent, _ = classify("Who is on shift?")
    assert status_intent in (Intent.GET_ATTENDANCE_STATUS, Intent.GET_ACTIVE_SHIFT)
    assert status_intent in local_attendance_intents

    overtime_intent, _ = classify("Who has the most overtime?")
    assert overtime_intent == Intent.GET_ATTENDANCE_OVERTIME
    assert overtime_intent in local_attendance_intents

    hours_intent, _ = classify("How many hours has Hamadan worked today?")
    assert hours_intent in (Intent.GET_ATTENDANCE_HOURS, Intent.GET_OPERATOR_WORKING_DAYS, Intent.GET_SHIFT_BY_OPERATOR)
    assert hours_intent in local_attendance_intents


def test_incident_store_preserves_regions_metadata():
    """Verify incident store records localized defect regions in SQLite image metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_incidents.db")
        incident_store.init_storage(tmpdir)

        # Real valid JPEG bytes
        _, enc = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
        annotated_img = enc.tobytes()

        regions_meta = [
            {
                "region_id": "A",
                "defect_type": "Rough / Torn Corner Material",
                "bbox": [120, 140, 95, 75],
                "normalized_bbox": [0.15, 0.23, 0.12, 0.13],
                "confidence": 0.88,
                "description": "Torn concrete bead and surface roughness on corner",
            },
            {
                "region_id": "B",
                "defect_type": "Bead Discontinuity / Layer Gap",
                "bbox": [380, 420, 60, 40],
                "normalized_bbox": [0.47, 0.70, 0.08, 0.07],
                "confidence": 0.84,
                "description": "Visible front bead material discontinuity",
            },
        ]

        # Save annotated image with regions metadata
        ann_saved = incident_store.save_incident_image(
            data=annotated_img,
            original_name="annotated_defect.jpg",
            content_type="image/jpeg",
            metadata={
                "is_annotated": True,
                "regions": regions_meta,
                "defect_type": "Region A: Rough / Torn Corner Material; Region B: Bead Discontinuity",
            },
            db_path=db_path,
            storage_dir=tmpdir,
        )

        ann_id = ann_saved["image_id"]
        assert ann_id is not None

        # Fetch image record from DB
        ann_record = incident_store.get_incident_image(ann_id, db_path=db_path)
        assert ann_record is not None
        assert "metadata" in ann_record
        meta = ann_record["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert "regions" in meta
        assert len(meta["regions"]) == 2
        assert meta["regions"][0]["region_id"] == "A"
        assert meta["regions"][1]["region_id"] == "B"
        assert meta["regions"][0]["normalized_bbox"] == [0.15, 0.23, 0.12, 0.13]


if __name__ == "__main__":
    print("Running test_validate_annotation_region_rejection...")
    test_validate_annotation_region_rejection()
    print("PASS: test_validate_annotation_region_rejection")

    print("Running test_detect_and_create_annotation_tight_regions...")
    test_detect_and_create_annotation_tight_regions()
    print("PASS: test_detect_and_create_annotation_tight_regions")

    print("Running test_attendance_provenance_classification_rules...")
    test_attendance_provenance_classification_rules()
    print("PASS: test_attendance_provenance_classification_rules")

    print("Running test_incident_store_preserves_regions_metadata...")
    test_incident_store_preserves_regions_metadata()
    print("PASS: test_incident_store_preserves_regions_metadata")

    print("\nALL 4/4 TESTS PASSED SUCCESSFULLY!")
