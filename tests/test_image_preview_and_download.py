"""
Tests for Kelvin6K Annotated-Image Preview and Download Functionality.

Validates:
1. TRACE & ROUTING:
   - Images are served with correct image MIME types (image/jpeg, image/png, image/webp).
   - Unknown image IDs return clean 404 JSON, NOT HTML.
   - Malicious/traversal IDs are rejected cleanly with 400/404, never exposing arbitrary files.
2. REAL IMAGE BYTES & MAGIC BYTES INTEGRITY:
   - Real binary bytes returned, matching magic bytes (b"\\xff\\xd8\\xff" for JPEG, b"\\x89PNG" for PNG).
   - Corrupted or non-image files on disk are detected via magic bytes and rejected with 500 error,
     guaranteeing HTML or arbitrary binary is never labeled or served as an image.
3. PREVIEW & DOWNLOAD HEADERS:
   - Inline preview request (download=False) returns Content-Disposition: inline with meaningful filename.
   - Download request via query parameter (download=True or download=1) returns Content-Disposition: attachment.
   - Dedicated download endpoint GET /api/incidents/images/{image_id}/download returns Content-Disposition: attachment.
   - Filenames are meaningful and accurately categorized:
     - Annotated images -> kelvin6k_annotation_<id>.<ext>
     - Reference images -> kelvin6k_reference_<id>.<ext>
     - Defect/upload images -> kelvin6k_defect_<id>.<ext>
4. SEPARATE ASSET PRESERVATION:
   - Original upload and AI annotation are distinct assets with unique IDs and content.
5. HEAD REQUEST SUPPORT:
   - HEAD /api/incidents/images/{image_id} returns 200 with matching headers.
"""

import os
import shutil
import tempfile
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from agent import config, db, incident_store
from api.server import app, _detect_image_format, _format_image_download_filename


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def image_env():
    tmp_dir = tempfile.mkdtemp(prefix="kelvin_img_test_")
    test_db = os.path.join(tmp_dir, "test_kelvin.db")
    img_dir = os.path.join(tmp_dir, "incident_images")
    os.makedirs(img_dir, exist_ok=True)

    orig_db = config.DATABASE_PATH
    orig_img_dir = config.INCIDENT_IMAGES_DIR

    config.DATABASE_PATH = test_db
    config.INCIDENT_IMAGES_DIR = img_dir

    db.init_db(test_db)
    with db.connect(test_db) as conn:
        incident_store._ensure_tables(conn)

    # 1. Generate original synthetic JPEG
    img_orig = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img_orig, (20, 20), (300, 220), (40, 180, 255), -1)
    cv2.putText(img_orig, "Defect Bead", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, enc_orig = cv2.imencode(".jpg", img_orig)
    orig_bytes = enc_orig.tobytes()

    # 2. Generate annotated copy (distinct pixel content with bounding box overlay)
    img_ann = img_orig.copy()
    cv2.rectangle(img_ann, (50, 50), (220, 170), (0, 140, 255), 2)
    cv2.putText(img_ann, "AI Observation: Bead Tearing", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    _, enc_ann = cv2.imencode(".jpg", img_ann)
    ann_bytes = enc_ann.tobytes()

    # 3. Generate component reference image (PNG format)
    img_png = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img_png, (40, 40), (280, 200), (0, 255, 120), -1)
    cv2.putText(img_png, "Nozzle Ref", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    _, enc_png = cv2.imencode(".png", img_png)
    png_bytes = enc_png.tobytes()

    # Save original defect upload
    defect_img = incident_store.save_incident_image(
        data=orig_bytes,
        original_name="operator_torn_bead.jpg",
        content_type="image/jpeg",
        metadata={"defect_type": "bead_tearing", "is_annotated": False},
        storage_dir=img_dir,
        db_path=test_db,
    )

    # Save annotated copy
    annotation_img = incident_store.save_incident_image(
        data=ann_bytes,
        original_name="annotated_operator_torn_bead.jpg",
        content_type="image/jpeg",
        metadata={
            "defect_type": "bead_tearing",
            "is_annotated": True,
            "parent_image_id": defect_img["image_id"],
        },
        storage_dir=img_dir,
        db_path=test_db,
    )

    # Save component reference image
    reference_img = incident_store.save_incident_image(
        data=png_bytes,
        original_name="ref_nozzle_assembly.png",
        content_type="image/png",
        metadata={
            "category": "nozzle",
            "is_reference": True,
            "component_name": "Standard Round Extrusion Nozzle",
        },
        storage_dir=img_dir,
        db_path=test_db,
    )

    yield {
        "db_path": test_db,
        "img_dir": img_dir,
        "orig_bytes": orig_bytes,
        "ann_bytes": ann_bytes,
        "png_bytes": png_bytes,
        "defect_img": defect_img,
        "annotation_img": annotation_img,
        "reference_img": reference_img,
    }

    config.DATABASE_PATH = orig_db
    config.INCIDENT_IMAGES_DIR = orig_img_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_01_detect_image_format_magic_bytes(image_env):
    """Verifies that format detection strictly checks magic bytes and rejects HTML/text."""
    # Genuine JPEG
    res_jpg = _detect_image_format(image_env["orig_bytes"])
    assert res_jpg is not None
    assert res_jpg[0] == "image/jpeg"
    assert res_jpg[1] == ".jpg"

    # Genuine PNG
    res_png = _detect_image_format(image_env["png_bytes"])
    assert res_png is not None
    assert res_png[0] == "image/png"
    assert res_png[1] == ".png"

    # HTML / text simulation -- MUST BE REJECTED
    html_bytes = b"<!DOCTYPE html><html><head><title>Kelvin6K</title></head><body>App</body></html>"
    assert _detect_image_format(html_bytes) is None

    # Arbitrary corrupted binary
    corrupted = b"\x00\x01\x02\x03\x04\x05"
    assert _detect_image_format(corrupted) is None


def test_02_format_image_download_filename(image_env):
    """Verifies meaningful filenames for annotated, reference, and defect images."""
    # Annotation
    ann_name = _format_image_download_filename("IMG-F94766D2-DE21", image_env["annotation_img"], ".jpg")
    assert ann_name == "kelvin6k_annotation_f94766d2-de21.jpg"

    # Reference
    ref_name = _format_image_download_filename("IMG-15E2FF42-CE89", image_env["reference_img"], ".png")
    assert ref_name == "kelvin6k_reference_15e2ff42-ce89.png"

    # Original Defect
    defect_name = _format_image_download_filename("IMG-B464F86D-5B24", image_env["defect_img"], ".jpg")
    assert defect_name == "kelvin6k_defect_b464f86d-5b24.jpg"


def test_03_get_image_inline_preview(client, image_env):
    """GET /api/incidents/images/{id} returns genuine image bytes with inline Content-Disposition."""
    resp = client.get(f"/api/incidents/images/{image_env['defect_img']['image_id']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
    assert "inline" in resp.headers["content-disposition"]
    assert "kelvin6k_defect_" in resp.headers["content-disposition"]
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_04_get_image_download_query_param(client, image_env):
    """GET /api/incidents/images/{id}?download=1 returns Content-Disposition: attachment."""
    resp = client.get(f"/api/incidents/images/{image_env['annotation_img']['image_id']}?download=1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert "attachment" in resp.headers["content-disposition"]
    assert "kelvin6k_annotation_" in resp.headers["content-disposition"]
    assert resp.content == image_env["ann_bytes"]


def test_05_dedicated_download_endpoint(client, image_env):
    """GET /api/incidents/images/{id}/download returns attachment header and real bytes."""
    resp = client.get(f"/api/incidents/images/{image_env['reference_img']['image_id']}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert "attachment" in resp.headers["content-disposition"]
    assert "kelvin6k_reference_" in resp.headers["content-disposition"]
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    assert resp.content == image_env["png_bytes"]


def test_06_head_request(client, image_env):
    """HEAD /api/incidents/images/{id} returns headers without body."""
    resp = client.head(f"/api/incidents/images/{image_env['defect_img']['image_id']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert "inline" in resp.headers["content-disposition"]


def test_07_unknown_image_id_returns_clean_404_json(client, image_env):
    """Unknown image ID returns HTTP 404 JSON detail, NEVER HTML."""
    resp = client.get("/api/incidents/images/IMG-NONEXISTENT-9999")
    assert resp.status_code == 404
    assert "application/json" in resp.headers["content-type"]
    data = resp.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_08_path_traversal_protection(client, image_env):
    """Path traversal attempt in image_id is rejected cleanly."""
    resp = client.get("/api/incidents/images/../../etc/passwd")
    assert resp.status_code in (400, 404)
    if resp.headers.get("content-type", "").startswith("application/json"):
        assert "detail" in resp.json()


def test_09_corrupted_image_file_returns_500_not_html(client, image_env):
    """If stored file is corrupted with HTML content, backend returns 500 JSON, never serving HTML as image."""
    corrupt_path = os.path.join(image_env["img_dir"], "corrupt_test.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"<!DOCTYPE html><html><body>Fake image</body></html>")

    corrupt_id = "IMG-CORRUPT-1234"
    with db.connect(image_env["db_path"]) as conn:
        incident_store._ensure_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO incident_images (
                image_id, file_path, original_name, content_hash,
                mime_type, size_bytes, width, height, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (corrupt_id, corrupt_path, "fake.jpg", "corrupt_hash_1234", "image/jpeg", 45, 100, 100, 1000.0, "{}"),
        )

    resp = client.get(f"/api/incidents/images/{corrupt_id}")
    assert resp.status_code == 500
    assert "application/json" in resp.headers["content-type"]
    assert "corrupted" in resp.json()["detail"].lower()


def test_10_separate_original_and_annotated_assets(client, image_env):
    """Ensures that original upload and annotated copy are stored and retrieved separately."""
    assert image_env["defect_img"]["image_id"] != image_env["annotation_img"]["image_id"]
    assert image_env["defect_img"]["content_hash"] != image_env["annotation_img"]["content_hash"]

    orig_resp = client.get(f"/api/incidents/images/{image_env['defect_img']['image_id']}")
    ann_resp = client.get(f"/api/incidents/images/{image_env['annotation_img']['image_id']}")

    assert orig_resp.status_code == 200
    assert ann_resp.status_code == 200
    assert "kelvin6k_defect_" in orig_resp.headers["content-disposition"]
    assert "kelvin6k_annotation_" in ann_resp.headers["content-disposition"]
    assert orig_resp.content == image_env["orig_bytes"]
    assert ann_resp.content == image_env["ann_bytes"]
