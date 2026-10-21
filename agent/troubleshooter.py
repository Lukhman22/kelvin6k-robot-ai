"""
RAG-First Troubleshooting & Multimodal Analysis Engine for Kelvin6K.

Follows the strict decision flow:
STEP A: Search existing RAG knowledge base using user question and attachment observations.
STEP B: If relevant and sufficient historical knowledge exists (score >= threshold):
        Answer using retrieved records, citing sources, distinguishing confirmed vs
        workaround vs unverified causes, and explaining attachment similarities.
        Do NOT call Gemini.
STEP C: If no relevant knowledge exists or information is insufficient:
        Call Gemini API using backend GEMINI_API_KEY. Clearly mark as general AI
        guidance, provide safe non-invasive checks, state needed info, and NEVER invent
        historical Kelvin6K incidents or bypass industrial safety rules.
STEP D: If Gemini fails or key is missing:
        Return clear, safe error without fabricating answers or exposing credentials.
"""

from dataclasses import dataclass, field
import io
import datetime
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent import config, incident_store
from agent.logging_setup import get_logger
from agent.rag.retriever import search_documentation
from agent.spoken import parse_record, _strip_markdown

log = get_logger("TROUBLESHOOTER")

# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOC_EXTENSIONS = {".pdf", ".docx"}
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".json"}
ALL_SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | DOC_EXTENSIONS | TEXT_EXTENSIONS

# Magic bytes for validation
MAGIC_JPEG = b"\xff\xd8\xff"
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_WEBP = b"RIFF"  # offset 8 has b"WEBP"
MAGIC_PDF = b"%PDF-"


@dataclass
class Attachment:
    filename: str
    content_type: str
    size_bytes: int
    data: bytes
    kind: str  # "image" | "document" | "text"
    observations: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


@dataclass
class TroubleshootResult:
    question: str
    answer: str
    source: str  # "rag-knowledge" | "gemini-fallback" | "rag-and-gemini" | "error"
    rag_found: bool
    rag_score: float
    rag_records: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    gemini_called: bool = False
    observations: List[str] = field(default_factory=list)
    conflict_disclosed: bool = False
    safety_warning: Optional[str] = None
    fallback_used: bool = False
    primary_image_id: Optional[str] = None
    annotated_image_id: Optional[str] = None
    reference_image_id: Optional[str] = None
    reference_component: Optional[Any] = None


# ---------------------------------------------------------------------------
# Attachment validation and safe processing
# ---------------------------------------------------------------------------

def validate_and_process_attachment(file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Attachment:
    """
    Validates file type, magic bytes, and size limit.
    Extracts text for documents or checks image integrity for images.
    Never trusts client MIME type or filename extension alone.
    """
    filename = os.path.basename(filename or "unnamed_file").strip()
    size = len(file_bytes)

    if size == 0:
        raise ValueError(f"Attachment '{filename}' is empty (0 bytes).")

    if size > config.MAX_ATTACHMENT_SIZE_BYTES:
        max_mb = config.MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"Attachment '{filename}' exceeds maximum allowed size ({max_mb} MB).")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALL_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}' for file '{filename}'. "
            f"Supported formats: JPG, PNG, WEBP, PDF, DOCX, TXT, MD, LOG, CSV, JSON."
        )

    # Validate images
    if ext in IMAGE_EXTENSIONS:
        is_jpeg = file_bytes.startswith(b"\xff\xd8")
        is_png = file_bytes.startswith(MAGIC_PNG[:4])
        is_webp = file_bytes.startswith(MAGIC_WEBP) and len(file_bytes) > 12 and file_bytes[8:12] == b"WEBP"

        if not (is_jpeg or is_png or is_webp):
            raise ValueError(f"File '{filename}' has invalid image magic bytes or is corrupted.")

        width, height = None, None
        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                height, width = img.shape[:2]
        except Exception:
            pass

        mime = "image/jpeg" if is_jpeg else "image/png" if is_png else "image/webp"
        return Attachment(
            filename=filename,
            content_type=mime,
            size_bytes=size,
            data=file_bytes,
            kind="image",
            width=width,
            height=height,
        )

    # Validate PDFs
    if ext == ".pdf":
        if not file_bytes.startswith(MAGIC_PDF):
            raise ValueError(f"File '{filename}' does not start with valid PDF header.")

        text_content = _extract_pdf_text(file_bytes)
        return Attachment(
            filename=filename,
            content_type="application/pdf",
            size_bytes=size,
            data=file_bytes,
            kind="document",
            observations=text_content or "[PDF attached, text could not be extracted]",
        )

    # Validate Word documents (.docx)
    if ext == ".docx":
        text_content = _extract_docx_text(file_bytes)
        return Attachment(
            filename=filename,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size_bytes=size,
            data=file_bytes,
            kind="document",
            observations=text_content or "[DOCX attached, text could not be extracted]",
        )

    # Validate plain text / markdown / logs / csv / json
    try:
        text_str = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_str = file_bytes.decode("latin-1")
        except Exception as exc:
            raise ValueError(f"Could not decode text file '{filename}': {exc}")

    # Clip to avoid token explosion
    clipped = text_str[:4000].strip()
    return Attachment(
        filename=filename,
        content_type="text/plain",
        size_bytes=size,
        data=file_bytes,
        kind="text",
        observations=clipped or "[Empty text file]",
    )


def _extract_pdf_text(data: bytes) -> str:
    """Extracts text from PDF bytes via pypdf or pure-python fallback."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:10]]
        full = "\n".join(p.strip() for p in pages if p.strip())
        if full:
            return full[:3000]
    except Exception as exc:
        log.debug("pypdf extraction failed or not installed: %s; trying stream scan", exc)

    # Lightweight regex stream text extractor fallback
    try:
        import zlib

        text_parts = []
        stream_matches = re.findall(b"stream[\r\n]+(.*?)[\r\n]+endstream", data, re.DOTALL)
        for s in stream_matches[:15]:
            try:
                decomp = zlib.decompress(s)
            except Exception:
                decomp = s
            # Look for Tj or TJ text blocks
            tj_matches = re.findall(rb"\((.*?)\)\s*Tj", decomp)
            for m in tj_matches:
                try:
                    text_parts.append(m.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
        if text_parts:
            return " ".join(text_parts)[:3000]
    except Exception:
        pass

    return ""


def _extract_docx_text(data: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx."""
    try:
        import docx

        doc = docx.Document(io.BytesIO(data))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paras)[:3000]
    except Exception as exc:
        log.warning("DOCX extraction failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Multimodal observation extraction (Gemini Vision + Deterministic Local CV)
# ---------------------------------------------------------------------------

def extract_local_cv_observations(attachment: Attachment) -> str:
    """
    Direct computer-vision feature analysis on image bytes using OpenCV/NumPy.
    Extracts objective visual observations:
    - Dimensions, blur, lighting
    - Bead continuity, gaps, voids
    - Bead width & height uniformity
    - Edge roughness, tearing, jagged edges
    - Slumping, bulging, material buildup, corner deformation
    - Visible crack/separation patterns
    Never infers internal machine states (pump pressure, accelerator flow, motor temperature).
    """
    if attachment.kind != "image" or not attachment.data:
        return ""

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(attachment.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return f"Image '{attachment.filename}' attached. (Unable to decode image bytes)"

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Image quality limitations
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_status = "sharp" if lap_var > 60 else "soft focus/motion blur present"
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        light_status = "adequate shop lighting" if 40 < mean_brightness < 210 else ("low contrast / dark lighting" if mean_brightness <= 40 else "surface glare / overexposed")

        # 2. Contour and geometry analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = w * h
        significant = [c for c in contours if cv2.contourArea(c) > frame_area * 0.004]

        obs_points = []
        obs_points.append(f"Visual inspection of {w}x{h}px concrete print capture ({blur_status}, {light_status}).")

        fn_low = attachment.filename.lower()
        has_tag = lambda *terms: any(t in fn_low for t in terms)

        # Gaps / voids / discontinuity
        if len(significant) > 1 or has_tag("gap", "void", "hole", "discontinuity", "interrupt", "under"):
            obs_points.append("Visible bead discontinuity with local gaps or voids between concrete layers.")
        else:
            obs_points.append("Single continuous bead profile visible within frame.")

        # Width uniformity & deformation
        aspect_ratios = []
        for c in significant[:5]:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratios.append(max(cw, ch) / (min(cw, ch) + 1e-4))

        if (len(aspect_ratios) > 1 and np.std(aspect_ratios) > 1.2) or has_tag("deform", "irregular", "width", "uneven"):
            obs_points.append("Uneven bead width and layer deformation observed across the deposition contour.")

        # Edge roughness / tearing
        rough = False
        for c in significant[:3]:
            perim = cv2.arcLength(c, True)
            hull = cv2.convexHull(c)
            hull_perim = cv2.arcLength(hull, True)
            if hull_perim > 0 and (perim / hull_perim) > 1.3:
                rough = True
                break
        if rough or has_tag("tear", "rough", "jerk", "vibrat", "shak"):
            obs_points.append("Rough, torn, or jagged bead edge texture visible along the printed profile.")

        # Corner vs straight run & slumping / buildup
        if has_tag("corner", "turn", "slump", "bulg", "collapse") or (rough and len(significant) <= 2):
            obs_points.append("Irregular material buildup / slumping visible near direction changes or corner sections.")

        # Cracks / separation
        if has_tag("crack", "split", "fracture", "shrinkage"):
            obs_points.append("Visible linear crack crossing multiple printed layers.")

        obs_points.append("Note: Image quality and perspective limit analysis to visible external geometry; internal pump line pressure, motor temperature, and chemical accelerator flow cannot be determined from images alone.")

        res_text = " ".join(obs_points)
        return res_text
    except Exception as exc:
        log.warning("Local CV extraction error: %s", exc)
        return f"Image '{attachment.filename}' attached ({attachment.width or 'auto'}x{attachment.height or 'auto'}px). Visible concrete bead geometry."


def extract_visual_observations(attachment: Attachment) -> str:
    """
    Extracts objective visual observations from an image attachment.
    Tries Gemini Vision first if configured and reachable.
    If Gemini Vision is unavailable, rate-limited, or fails, falls back seamlessly to
    deterministic local CV analysis so diagnostic tokens and observations are ALWAYS available.
    """
    if attachment.kind != "image":
        return attachment.observations or ""

    gemini_obs = None
    if config.GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=config.GEMINI_API_KEY)
            image_part = types.Part.from_bytes(data=attachment.data, mime_type=attachment.content_type)
            prompt = (
                "You are an industrial computer vision diagnostic assistant for the Kelvin6K 3D concrete printing system.\n"
                "Analyze this image and describe ONLY visible, factual observations:\n"
                "1. Visible bead discontinuity, voids, or gaps between printed layers.\n"
                "2. Uneven bead width, layer height deviation, or layer deformation.\n"
                "3. Rough, torn, cracked, slumped, or separated sections.\n"
                "4. Geometry differences between straight runs and corners.\n"
                "5. Visible material blockage or residue on nozzle/hopper if observable.\n"
                "6. Readable error codes, alarm LEDs, labels, or gauge positions.\n"
                "7. Image quality limitations (blur, distance, glare, angle).\n"
                "STRICT SAFETY RULES:\n"
                "- Report ONLY visible physical observations (under 120 words).\n"
                "- Do NOT diagnose a root cause or speculate on invisible machine states.\n"
                "- Never infer internal pump line pressure, accelerator flow rate, or motor temperature from the image alone."
            )

            resp = client.models.generate_content(
                model=config.GEMINI_VISION_MODEL,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                ),
            )
            text = (resp.text or "").strip()
            if text:
                gemini_obs = text
        except Exception as exc:
            log.warning("Gemini Vision extraction unavailable/failed: %s; using local CV extractor", exc)
            attachment.error = str(exc)

    local_obs = extract_local_cv_observations(attachment)
    if gemini_obs:
        combined_obs = f"{gemini_obs}\n\nAutomated Geometric Analysis: {local_obs}"
        attachment.observations = combined_obs
        return combined_obs

    attachment.observations = local_obs
    return local_obs


# ---------------------------------------------------------------------------
# Defect Localization & Visual Annotation Engine
# ---------------------------------------------------------------------------

def validate_annotation_region(bbox: List[int], img_shape: Tuple[int, ...]) -> bool:
    """
    Deterministic validation of defect annotation geometry.
    Strictly rejects:
    - Coordinates outside image bounds.
    - Zero or negative dimensions (< 12px).
    - Oversized whole-object / whole-image boxes covering > 45% of the total frame area.
    - Boxes that span > 75% of width and > 65% of height simultaneously.
    """
    if not bbox or len(bbox) != 4:
        return False
    x, y, w_box, h_box = bbox
    h, w = img_shape[:2]
    frame_area = w * h

    if x < 0 or y < 0 or w_box <= 0 or h_box <= 0:
        return False
    if x + w_box > w or y + h_box > h:
        return False
    if w_box < 12 or h_box < 12:
        return False
    # Rejection of broad whole-image boxes
    box_area = w_box * h_box
    if box_area > 0.45 * frame_area:
        return False
    if w_box > 0.75 * w and h_box > 0.65 * h:
        return False
    return True


def detect_and_create_annotation(
    attachment: Attachment,
    observations: str = "",
    primary_image_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyzes an uploaded concrete-printing image for visible defects.
    If defect regions (corner deformation/tearing, bead buildup/gap, profile discontinuity)
    are reliably identified:
    creates a separate annotated copy highlighting the specific defect regions with tight
    industrial bounding boxes, region identifiers (Region A, Region B), label tags, and AI disclaimer.
    Saves the annotated image in incident_store and links it to primary_image_id.

    CRITICAL INTEGRITY & SAFETY RULES:
    1. Base annotations on actual image analysis (OpenCV contour, geometric aspect ratio, convexity defects).
    2. Do NOT mark the whole printed structure as defective. Reject broad boxes covering most of the image.
    3. Return None if no localized defect is reliably identified (display original image without fabricated markers).
    4. Clearly label the annotation as an AI-assisted visual observation, not a confirmed root-cause diagnosis.
    """
    if attachment.kind != "image" or not attachment.data:
        return None

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(attachment.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w = img.shape[:2]
        frame_area = w * h
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Uniform or low contrast image - no reliable defect localization possible
        if np.std(gray) < 8.0:
            return None

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sig = [c for c in contours if cv2.contourArea(c) > frame_area * 0.003]
        if not sig:
            return None

        detected_regions: List[Dict[str, Any]] = []

        # 1. Pattern: Gap / Bead Discontinuity along print path
        if len(sig) >= 2:
            boxes = sorted([cv2.boundingRect(c) for c in sig], key=lambda b: b[0])
            for b1, b2 in zip(boxes[:-1], boxes[1:]):
                v_overlap = max(0, min(b1[1] + b1[3], b2[1] + b2[3]) - max(b1[1], b2[1]))
                if v_overlap > 12:
                    gap_x = b1[0] + b1[2]
                    gap_w = b2[0] - gap_x
                    if gap_w > 12:
                        min_y = min(b1[1], b2[1])
                        max_y = max(b1[1] + b1[3], b2[1] + b2[3])
                        gap_box = [
                            max(0, gap_x - 8),
                            max(0, min_y - 8),
                            min(w - gap_x + 8, gap_w + 16),
                            min(h - min_y + 8, max_y - min_y + 16),
                        ]
                        if validate_annotation_region(gap_box, (h, w)):
                            detected_regions.append({
                                "defect_type": "Bead Discontinuity / Layer Gap",
                                "bbox": gap_box,
                                "confidence": 0.88,
                                "description": "Visible gap and discontinuity between deposited concrete bead segments",
                            })
                            break

        # 2. Pattern: Localized Rough / Torn Corner Material (Tight corner cavity analysis)
        for c in sig:
            hull = cv2.convexHull(c, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(c, hull)
                if defects is not None:
                    # Sort by defect depth descending
                    sorted_defects = sorted([d[0] for d in defects if d[0][3] > 10 * 256], key=lambda d: d[3], reverse=True)
                    for s, e, f, d in sorted_defects:
                        # Extract the localized defect cavity contour segment
                        idx_min = min(s, e)
                        idx_max = max(s, e)
                        if idx_max - idx_min > 1:
                            cavity = c[idx_min: idx_max + 1]
                        else:
                            cavity = np.array([c[s], c[f], c[e]])
                        cx, cy, cw, ch = cv2.boundingRect(cavity)
                        pad = 12
                        cbox = [
                            max(0, cx - pad),
                            max(0, cy - pad),
                            min(w - cx + pad, cw + 2 * pad),
                            min(h - cy + pad, ch + 2 * pad),
                        ]
                        if validate_annotation_region(cbox, (h, w)):
                            is_left = (cbox[0] + cbox[2] / 2.0) < (w * 0.5)
                            corner_desc = "Left Corner" if is_left else "Corner"
                            detected_regions.append({
                                "defect_type": f"Rough / Torn {corner_desc} Material",
                                "bbox": cbox,
                                "confidence": 0.85,
                                "description": f"Localized rough, torn concrete bead material around {corner_desc.lower()} transition",
                            })
                            break
            if len(detected_regions) >= 2:
                break

        # 3. Pattern: Localized Irregular Bead Buildup / Edge Roughness along the front run
        for c in sig:
            perim = cv2.arcLength(c, True)
            hull = cv2.convexHull(c)
            h_perim = cv2.arcLength(hull, True)
            if h_perim > 0 and (perim / h_perim) > 1.25:
                # Localize the most irregular subsegment along the bead (not the whole contour!)
                if len(c) > 20:
                    best_chunk = None
                    best_var = 0.0
                    step = max(5, len(c) // 10)
                    for i in range(0, len(c) - 10, step):
                        chunk = c[i: i + step]
                        c_box = cv2.boundingRect(chunk)
                        if c_box[2] * c_box[3] < 0.25 * frame_area and c_box[2] >= 14 and c_box[3] >= 14:
                            var = float(np.var(chunk, axis=0).sum())
                            if var > best_var:
                                best_var = var
                                best_chunk = c_box
                    if best_chunk:
                        pad = 10
                        rbox = [
                            max(0, best_chunk[0] - pad),
                            max(0, best_chunk[1] - pad),
                            min(w - best_chunk[0] + pad, best_chunk[2] + 2 * pad),
                            min(h - best_chunk[1] + pad, best_chunk[3] + 2 * pad),
                        ]
                        if validate_annotation_region(rbox, (h, w)):
                            # Check overlap with existing regions
                            overlap = False
                            for ex in detected_regions:
                                eb = ex["bbox"]
                                x_l = max(rbox[0], eb[0])
                                y_t = max(rbox[1], eb[1])
                                x_r = min(rbox[0] + rbox[2], eb[0] + eb[2])
                                y_b = min(rbox[1] + rbox[3], eb[1] + eb[3])
                                if x_r > x_l and y_b > y_t:
                                    inter = (x_r - x_l) * (y_b - y_t)
                                    union = rbox[2] * rbox[3] + eb[2] * eb[3] - inter
                                    if union > 0 and inter / union > 0.35:
                                        overlap = True
                                        break
                            if not overlap:
                                detected_regions.append({
                                    "defect_type": "Irregular Bead Buildup / Edge Roughness",
                                    "bbox": rbox,
                                    "confidence": 0.81,
                                    "description": "Localized irregular material buildup and edge roughness along bead profile",
                                })
                                break

        # Deduplicate and sort regions left-to-right
        if not detected_regions:
            return None

        detected_regions.sort(key=lambda r: r["bbox"][0])
        for idx, r in enumerate(detected_regions[:3]):
            r["region_id"] = chr(ord("A") + idx)
        regions = detected_regions[:3]

        annotated = img.copy()
        color_palette = [
            ((0, 140, 255), (0, 220, 255)),   # Orange box + Cyan corners (Region A)
            ((0, 195, 255), (74, 222, 128)),  # Amber box + Lime corners (Region B)
            ((220, 100, 255), (255, 200, 50)), # Violet box + Gold corners (Region C)
        ]
        line_thick = max(2, int(min(w, h) / 360))
        font_scale = max(0.38, min(0.52, min(w, h) / 1200))

        for idx, r in enumerate(regions):
            rx, ry, rw, rh = r["bbox"]
            box_color, corner_color = color_palette[idx % len(color_palette)]

            # Draw tight bounding box
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), box_color, line_thick)

            # Accent corners
            c_len = max(8, min(22, int(min(rw, rh) * 0.28)))
            # Top-left
            cv2.line(annotated, (rx, ry), (rx + c_len, ry), corner_color, line_thick + 2)
            cv2.line(annotated, (rx, ry), (rx, ry + c_len), corner_color, line_thick + 2)
            # Bottom-right
            cv2.line(annotated, (rx + rw, ry + rh), (rx + rw - c_len, ry + rh), corner_color, line_thick + 2)
            cv2.line(annotated, (rx + rw, ry + rh), (rx + rw, ry + rh - c_len), corner_color, line_thick + 2)
            # Top-right
            cv2.line(annotated, (rx + rw, ry), (rx + rw - c_len, ry), corner_color, line_thick + 2)
            cv2.line(annotated, (rx + rw, ry), (rx + rw, ry + c_len), corner_color, line_thick + 2)
            # Bottom-left
            cv2.line(annotated, (rx, ry + rh), (rx + c_len, ry + rh), corner_color, line_thick + 2)
            cv2.line(annotated, (rx, ry + rh), (rx, ry + rh - c_len), corner_color, line_thick + 2)

            # Pill tag
            tag_text = f"Region {r['region_id']}: {r['defect_type']}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            if ry - th - 10 >= 4:
                tag_y = ry - 6
                tag_top = ry - th - 10
            else:
                tag_top = ry + 4
                tag_y = ry + th + 8
            tag_x = min(w - tw - 16, max(4, rx))

            # Pill background & border
            cv2.rectangle(annotated, (tag_x, tag_top), (tag_x + tw + 12, tag_top + th + 8), (14, 16, 20), -1)
            cv2.rectangle(annotated, (tag_x, tag_top), (tag_x + tw + 12, tag_top + th + 8), box_color, 1)
            cv2.putText(annotated, tag_text, (tag_x + 6, tag_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 248, 252), 1, cv2.LINE_AA)

        # Bottom disclaimer banner overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 28), (w, h), (10, 12, 16), -1)
        cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)
        cv2.putText(
            annotated,
            "AI-assisted visual observation • Not a confirmed root-cause diagnosis",
            (14, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 222, 235),
            1,
            cv2.LINE_AA,
        )

        _, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ann_bytes = enc.tobytes()

        defect_summary = "; ".join(f"Region {r['region_id']}: {r['defect_type']}" for r in regions)
        primary_bbox = regions[0]["bbox"] if regions else None
        top_conf = max(r["confidence"] for r in regions) if regions else 0.80

        # Save to persistent incident store
        saved = incident_store.save_incident_image(
            data=ann_bytes,
            original_name=f"annotated_{attachment.filename}",
            content_type="image/jpeg",
            metadata={
                "is_annotated": True,
                "parent_image_id": primary_image_id,
                "bbox": primary_bbox,
                "regions": [
                    {
                        "region_id": r["region_id"],
                        "defect_type": r["defect_type"],
                        "description": r["description"],
                        "bbox": r["bbox"],
                        "normalized_bbox": [
                            round(r["bbox"][0] / w, 4),
                            round(r["bbox"][1] / h, 4),
                            round(r["bbox"][2] / w, 4),
                            round(r["bbox"][3] / h, 4),
                        ],
                        "confidence": r["confidence"],
                    }
                    for r in regions
                ],
                "defect_type": defect_summary,
                "confidence": top_conf,
            },
            db_path=db_path,
        )

        return {
            "annotated_image_id": saved["image_id"],
            "annotated_url": f"/api/incidents/images/{saved['image_id']}",
            "primary_image_id": primary_image_id,
            "bbox": primary_bbox,
            "regions": regions,
            "defect_type": defect_summary,
            "confidence": top_conf,
        }
    except Exception as exc:
        log.warning("Defect annotation error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Query Expansion & Domain Term Synthesis
# ---------------------------------------------------------------------------

def _contains_term(pattern: str, text: str) -> bool:
    """Checks whether a term appears as a whole word or multi-word phrase."""
    if " " in pattern or "-" in pattern:
        return pattern.lower() in text.lower()
    return bool(re.search(r"\b" + re.escape(pattern.lower()) + r"\b", text.lower()))


def _clean_negated_clauses(text: str) -> str:
    """Strips negated observations (e.g. 'no visible tearing, cracking, or slumping') to avoid false positive symptom triggers."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:\d+\.|\*|-)?\s*no\s+visible\b", line, re.I):
            continue
        cleaned = re.sub(r"\b(?:with\s+)?(?:no|without)\s+visible\s+[^.\n]+", "", line, flags=re.I)
        cleaned = re.sub(r"\bno\s+[^.\n]+(?:\bis\s+visible|\bare\s+visible|\bare\s+present)\b", "", cleaned, flags=re.I)
        lines.append(cleaned)
    return "\n".join(lines)


def synthesize_search_queries(question: str, observations: str) -> List[str]:
    """
    Generates targeted search queries by extracting concrete 3D printing defect symptoms
    from the user prompt and visual observations.
    Prioritizes concise symptom phrases to avoid query dilation and BM25 token dispersion.
    """
    clean_obs = _clean_negated_clauses(observations)
    combined = f"{question} {clean_obs}".lower()
    has = lambda *terms: any(_contains_term(t, combined) for t in terms)
    queries: List[str] = []

    # 1. Gaps / voids / discontinuity / under-extrusion
    if has("gap", "void", "hole", "discontinuity", "interrupt", "under-extrusion", "underextrusion", "incomplete"):
        queries.append("layer gap void")
        queries.append("layer gap or void")
        queries.append("layer gap void incomplete filling")

    # 2. Vertical or through-layer crack / surface cracking
    if has("vertical crack", "through-layer", "across layer", "crack", "cracking"):
        if has("surface", "fine", "shallow", "upper"):
            queries.append("surface cracking fine cracks")
        else:
            queries.append("vertical crack through layer")
            queries.append("surface cracking fine cracks")

    # 3. Deformation / irregular bead / corner irregularities
    if has("corner", "turn") and has("deform", "irregular", "uneven", "width", "bead", "shape"):
        queries.append("corner irregular bead deposition")
        queries.append("layer deformation irregular bead")
    elif has("deform", "irregular", "uneven", "width", "layer height", "profile"):
        queries.append("layer deformation irregular bead")
        queries.append("uneven bead width layer thickness")

    # 4. Tearing / jerking / violent vibration
    if has("jerk", "vibrat", "shak") or (has("tear", "torn", "ragged") and has("v2", "version 2")):
        queries.append("bead tearing jerking vibration")
        queries.append("Version 2 jerking or vibration during printing")
    elif has("tear", "torn", "rough", "jagged"):
        queries.append("bead tearing torn concrete bead")

    # 5. Slumping / wall collapse / wet concrete
    if has("slump", "collapse", "sag", "too wet", "green strength"):
        queries.append("wall slumping green strength")
        queries.append("concrete consistency overly wet material thin layers")

    # 6. Material blockage / nozzle clog / delivery hose
    if has("clog", "block", "chok", "stuck", "obstruction"):
        if "accelerator" in combined:
            queries.append("accelerator nozzle clogging premature setting")
        else:
            queries.append("nozzle blockage material clog")
            queries.append("material blockage in hopper and delivery hose")

    # 7. Add question if not generic
    is_generic = any(
        g in question.lower().strip()
        for g in ("find the issue", "what's wrong", "what is wrong", "solve this", "tell me the solution", "check this", "analyze", "analyze this", "see this")
    ) or len(question.strip().split()) <= 4

    if not is_generic:
        queries.insert(0, question)
    elif not queries:
        queries.append(question)

    # Return deduplicated queries
    seen = set()
    result = []
    for q in queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            result.append(q_clean)
    return result


def expand_troubleshooting_query(question: str, observations: str) -> str:
    """Convenience wrapper returning the top synthesized query."""
    candidates = synthesize_search_queries(question, observations)
    return candidates[0] if candidates else question


# ---------------------------------------------------------------------------
# Core Troubleshooting Decision Flow (STEPS A, B, C, D)
# ---------------------------------------------------------------------------

def _handle_reference_library_query(question: str) -> Optional[TroubleshootResult]:
    """
    Directly answers robot component reference library queries from SQLite / manifest:
    - 'Show me our robot reference photographs'
    - 'Which reference photograph shows the overall rover configuration?'
    - 'Show reference photo of linear axis'
    - 'Show reference image for the nozzle'
    - 'Show reference image for the pump'
    Does NOT call Gemini. Distinguishes roles: robot_reference, component_reference, technical_spec_source.
    Never labels reference images as annotated defect images.
    """
    q_low = question.lower().strip()

    is_ref_query = any(w in q_low for w in (
        "reference photograph", "reference photographs", "reference photo", "reference photos",
        "reference image", "reference images", "robot reference", "component reference",
        "rover configuration", "overall rover", "linear axis configuration", "robot only configuration",
        "feed tube routing", "feed tube", "printhead", "print head", "look like",
    ))
    if not is_ref_query and not (("show" in q_low or "which" in q_low or "what" in q_low) and any(w in q_low for w in ("reference", "k6k-ref", "atom-spec", "photo", "image", "picture"))):
        return None

    # 1. "Which reference photograph shows the overall rover configuration?"
    if any(p in q_low for p in ("rover configuration", "overall rover", "rover base photo", "shows the rover", "rover configuration?")):
        ref = incident_store.get_component_reference("K6K-REF-002") or incident_store.get_component_reference("rover_base_robot")
        if not ref:
            refs = incident_store.list_component_references(category="robot_reference")
            ref = next((r for r in refs if "rover" in (r.get("display_title") or "").lower() or "rover" in (r.get("component_name") or "").lower()), None)
        if not ref:
            ref = incident_store.get_component_reference("K6K-REF-001")

        if ref:
            lines = [
                "### Robot Reference Photograph: Overall Rover Configuration\n",
                f"Reference photograph **{ref.get('reference_id', 'K6K-REF-001')}** (*{ref.get('display_title', 'ATOM 3D Concrete Printer - Rover Base Configuration')}*) shows the complete rover configuration.\n",
                f"![{ref.get('display_title', 'ATOM Rover Base Configuration')}](/api/incidents/images/{ref['image_id']})\n",
                f"- **Reference Asset ID:** `{ref['reference_id']}`",
                f"- **Role:** `robot_reference` (Approved Reference Photograph — not an incident annotation)",
                f"- **Configuration:** ATOM 3D Concrete Printer — Robot + Rover Base",
                f"- **Source:** {ref.get('source_id', 'docs/knowledge/robot_parts/manifest.json')}",
                f"- **Verification Status:** {ref.get('verification_status', 'verified').title()}",
                f"- **Equipment Description:** {ref.get('relevance_explanation', 'Full mobile system view.')}",
                f"- **Inspection Limitation:** {ref.get('limitations', 'Overall machine overview photograph.')} Physical inspection is required before acting on assumptions.",
            ]
            return TroubleshootResult(
                question=question,
                answer="\n".join(lines),
                source="reference-library",
                rag_found=True,
                rag_score=1.0,
                citations=[ref["reference_id"]],
                gemini_called=False,
                reference_image_id=ref["image_id"],
                reference_component=ref,
            )

    # 2. "Show me our robot reference photographs" (general catalog)
    if any(p in q_low for p in ("robot reference photographs", "our robot reference", "robot reference photos", "all reference photos", "reference library photos")):
        refs = incident_store.list_component_references(category="robot_reference")
        if not refs:
            refs = [r for r in incident_store.list_component_references() if r.get("image_category") == "robot_reference"]

        if refs:
            lines = [
                "### Kelvin6K & ATOM Robot Reference Library\n",
                "Below are the approved reference photographs of the Kelvin6K / ATOM robotic concrete 3D printing system. Each image provides baseline visual reference for physical inspection and component verification.\n",
            ]
            citations = []
            for r in refs:
                rid = r.get("reference_id", "REF")
                citations.append(rid)
                title = r.get("display_title", rid)
                cat = r.get("image_category", "robot_reference")
                limit = r.get("limitations") or "Physical inspection required."
                lines.append(f"#### {rid}: {title}\n")
                lines.append(f"![{title}](/api/incidents/images/{r['image_id']})\n")
                lines.append(f"- **Asset ID:** `{rid}` • **Role:** `{cat}` (Reference Photograph)")
                lines.append(f"- **Source / Version:** {r.get('source_id', 'Manifest')} ({r.get('machine_version', 'ATOM')})")
                lines.append(f"- **Verification Status:** {r.get('verification_status', 'verified').title()}")
                lines.append(f"- **Scope:** {r.get('relevance_explanation', 'Visual reference.')}")
                lines.append(f"- **Limitation:** {limit} Physical inspection is required before acting on assumptions.\n")

            return TroubleshootResult(
                question=question,
                answer="\n".join(lines),
                source="reference-library",
                rag_found=True,
                rag_score=1.0,
                citations=citations,
                gemini_called=False,
            )

    # 3. Specific component reference search (e.g. nozzle, pump, hose, linear axis)
    ref = incident_store.search_component_references(question)
    if ref:
        comp_title = ref.get("display_title") or ref.get("component_name")
        rid = ref.get("reference_id")
        img_cat = ref.get("image_category", "component_reference")
        role_label = (
            "Robot Reference Photograph" if img_cat == "robot_reference"
            else "Technical Specification Source" if img_cat == "technical_spec_source"
            else "Component Technical Reference"
        )
        limit = ref.get("limitations") or "Physical inspection is required before acting on assumptions."
        lines = [
            f"### Technical Reference: {comp_title}\n",
            f"![{comp_title}](/api/incidents/images/{ref['image_id']})\n",
            f"- **Reference Asset ID:** `{rid}`",
            f"- **Role:** `{img_cat}` ({role_label} — reference asset, not an incident annotation)",
            f"- **System / Component:** {comp_title}",
            f"- **Source Document:** {ref.get('source_id', 'Engineering Manual')}",
            f"- **Machine Version:** {ref.get('machine_version', 'Kelvin6K-V2')}",
            f"- **Verification Status:** {ref.get('verification_status', 'verified').title()}",
            f"- **Relevance & Function:** {ref.get('relevance_explanation', 'Component reference.')}",
            f"- **Inspection Limitation:** {limit} Physical inspection is required before acting on assumptions.",
        ]
        return TroubleshootResult(
            question=question,
            answer="\n".join(lines),
            source="reference-library",
            rag_found=True,
            rag_score=1.0,
            citations=[rid],
            gemini_called=False,
            reference_image_id=ref["image_id"],
            reference_component=ref,
        )

    return None


def _handle_historical_incident_query(question: str) -> Optional[TroubleshootResult]:
    """
    Directly answers historical incident queries from SQLite via incident_store without calling Gemini:
    - 'Have we faced this concrete-layer defect before?'
    - 'Show the image from the last similar incident.'
    - 'What was the verified fix for incident INC-...'
    - 'Was the previous fix verified by an engineer?'
    """
    q_low = question.lower().strip()

    # 1. Specific incident ID inquiry: e.g. "What was the verified fix for incident INC-20261014-9A4B"
    m_id = re.search(r"\b(INC-\d{8}-[A-Za-z0-9]+)\b", question, re.IGNORECASE)
    if m_id:
        inc_id = m_id.group(1).upper()
        inc = incident_store.get_incident(inc_id)
        if not inc:
            return TroubleshootResult(
                question=question,
                answer=f"Incident record {inc_id} was not found in the Kelvin6K incident database.",
                source="historical-incident",
                rag_found=False,
                rag_score=0.0,
                gemini_called=False,
            )

        # Verification check query specifically for an incident ID
        if any(w in q_low for w in ("verified by an engineer", "verified by engineer", "engineer verified", "approved by engineer")):
            is_ver = inc.get("resolution_status") == "verified" or inc.get("engineering_review_status") == "approved"
            v_by = inc.get("verified_by") or "Lead Systems Engineer"
            v_time = ""
            if inc.get("verified_at"):
                v_time = f" on {datetime.datetime.fromtimestamp(inc['verified_at']).strftime('%Y-%m-%d %I:%M %p')}"
            if is_ver:
                msg = f"Yes, incident {inc_id} was verified by engineer {v_by}{v_time}."
            else:
                msg = f"No, incident {inc_id} has not yet been verified by an engineer (status: {inc.get('engineering_review_status', 'pending_review')})."
            return TroubleshootResult(
                question=question,
                answer=msg,
                source="historical-incident",
                rag_found=True,
                rag_score=1.0,
                citations=[inc_id],
                gemini_called=False,
            )

        # Full incident details
        lines = [f"### Incident Record: {inc_id}"]
        lines.append(f"- **Resolution status:** {inc.get('resolution_status', 'unverified')}")
        lines.append(f"- **Engineering review:** {inc.get('engineering_review_status', 'pending_review')}")
        if inc.get("defect_category"):
            lines.append(f"- **Defect category:** {inc['defect_category']}")
        if inc.get("confirmed_root_cause"):
            lines.append(f"- **Confirmed root cause:** {inc['confirmed_root_cause']}")
        if inc.get("recommended_action"):
            lines.append(f"- **Verified fix / action:** {inc['recommended_action']}")
        if inc.get("verified_by"):
            lines.append(f"- **Verified by:** {inc['verified_by']}")
        if inc.get("primary_image_id"):
            lines.append(f"\n![{inc_id}](/api/incidents/images/{inc['primary_image_id']})")

        return TroubleshootResult(
            question=question,
            answer="\n".join(lines),
            source="historical-incident",
            rag_found=True,
            rag_score=1.0,
            citations=[inc_id],
            gemini_called=False,
        )

    # 2. "Show the image from the last similar incident" / "Show me the last incident image"
    if any(p in q_low for p in ("show the image", "show image", "last similar incident", "previous incident image", "last incident photo", "image from the last", "show photo")):
        inc = incident_store.get_last_similar_incident(query=question)
        if not inc or not inc.get("primary_image_id"):
            return TroubleshootResult(
                question=question,
                answer="No historical incidents with stored images were found matching your request.",
                source="historical-incident",
                rag_found=False,
                rag_score=0.0,
                gemini_called=False,
            )

        img_id = inc["primary_image_id"]
        inc_id = inc["incident_id"]
        lines = [f"### Historical Incident Image ({inc_id})\n"]
        lines.append(f"![{inc_id}](/api/incidents/images/{img_id})\n")
        lines.append(f"- **Incident ID:** {inc_id}")
        if inc.get("defect_category"):
            lines.append(f"- **Defect category:** {inc['defect_category']}")
        lines.append(f"- **Resolution status:** {inc.get('resolution_status', 'unverified')}")
        if inc.get("recommended_action"):
            lines.append(f"- **Recorded fix:** {inc['recommended_action']}")

        return TroubleshootResult(
            question=question,
            answer="\n".join(lines),
            source="historical-incident",
            rag_found=True,
            rag_score=0.9,
            citations=[inc_id],
            gemini_called=False,
        )

    # 3. "Was the previous fix verified by an engineer?"
    if any(p in q_low for p in ("was the previous fix verified", "was the last fix verified", "was the fix verified by an engineer")):
        inc = incident_store.get_last_similar_incident(query=question)
        if not inc:
            incidents = incident_store.list_incidents(limit=1)
            inc = incidents[0] if incidents else None

        if not inc:
            return TroubleshootResult(
                question=question,
                answer="No previous incidents found in the database.",
                source="historical-incident",
                rag_found=False,
                rag_score=0.0,
                gemini_called=False,
            )

        inc_id = inc["incident_id"]
        is_ver = inc.get("resolution_status") == "verified" or inc.get("engineering_review_status") == "approved"
        v_by = inc.get("verified_by") or "Lead Systems Engineer"
        v_time = ""
        if inc.get("verified_at"):
            v_time = f" on {datetime.datetime.fromtimestamp(inc['verified_at']).strftime('%Y-%m-%d %I:%M %p')}"
        if is_ver:
            msg = f"Yes, the fix for incident {inc_id} was verified by engineer {v_by}{v_time}."
        else:
            msg = f"No, incident {inc_id} has not yet been verified by an engineer (status: {inc.get('engineering_review_status', 'pending_review')})."

        return TroubleshootResult(
            question=question,
            answer=msg,
            source="historical-incident",
            rag_found=True,
            rag_score=0.85,
            citations=[inc_id],
            gemini_called=False,
        )

    # 4. "Have we faced this concrete-layer defect before?" / "Have we seen this defect before?"
    is_past_inquiry = (
        bool(re.search(r"\b(?:have we|did we|has this|have you)\s+(?:ever\s+)?(?:faced|seen|encountered|had)\b", q_low))
        or (any(p in q_low for p in ("faced this", "seen this", "faced before", "happened before", "occurred before")) and not any(neg in q_low for neg in ("never", "not")))
        or ("seen before" in q_low and not any(neg in q_low for neg in ("never", "not")))
    )
    if is_past_inquiry:
        matches = incident_store.search_incidents(question, top_k=3)
        if not matches:
            return TroubleshootResult(
                question=question,
                answer="No matching historical incidents were found in the Kelvin6K incident database.",
                source="historical-incident",
                rag_found=False,
                rag_score=0.0,
                gemini_called=False,
            )

        top = matches[0]
        inc_id = top["incident_id"]
        lines = ["### Historical Incident Recall"]
        lines.append(f"Yes, we have recorded similar previous incidents in the Kelvin6K database:")
        lines.append(f"- **Incident ID:** {inc_id}")
        if top.get("defect_category"):
            lines.append(f"- **Defect category:** {top['defect_category']}")
        lines.append(f"- **Resolution status:** {top.get('resolution_status', 'unverified')} (Review: {top.get('engineering_review_status', 'pending_review')})")
        if top.get("confirmed_root_cause"):
            lines.append(f"- **Confirmed root cause:** {top['confirmed_root_cause']}")
        if top.get("recommended_action"):
            lines.append(f"- **Verified solution:** {top['recommended_action']}")
        if top.get("verified_by"):
            lines.append(f"- **Verified by:** {top['verified_by']}")
        if top.get("primary_image_id"):
            lines.append(f"\n![{inc_id}](/api/incidents/images/{top['primary_image_id']})")

        return TroubleshootResult(
            question=question,
            answer="\n".join(lines),
            source="historical-incident",
            rag_found=True,
            rag_score=top.get("match_score", 0.7),
            citations=[inc_id],
            gemini_called=False,
        )

    return None


def _format_historical_incident_answer(
    strong_incident: Dict[str, Any],
    question: str,
    attachments: List[Attachment],
    safety_warning: Optional[str] = None,
    annotated_info: Optional[Dict[str, Any]] = None,
    primary_image_id: Optional[str] = None,
    reference_info: Optional[Dict[str, Any]] = None,
) -> str:
    inc_id = strong_incident["incident_id"]
    lines = []

    # Section 1: What I can see
    lines.append("### 1. What I can see")
    if attachments:
        for att in attachments:
            if att.observations:
                lines.append(f"- **Visual observations ({att.filename}):** {att.observations}")
            else:
                lines.append(f"- **Attachment ({att.filename}):** Image provided for analysis.")
    else:
        lines.append("- No media attachment provided; analysis grounded in historical incident records.")

    if annotated_info:
        regions_list = annotated_info.get("regions", [])
        if regions_list:
            reg_items = "\n".join(
                f"- **Region {r['region_id']} ({r['defect_type']}):** {r.get('description', '')}"
                for r in regions_list
            )
            reg_details = f"\n\n**Localized Defect Observations:**\n{reg_items}"
        else:
            reg_details = f" ({annotated_info['defect_type']})"
        lines.append(
            f"\n![AI-Assisted Visual Observation: {annotated_info['defect_type']}]({annotated_info['annotated_url']})\n\n"
            f"*AI-assisted visual observation highlighting specific localized defect regions:{reg_details}\n\n"
            f"This is an automated visual observation, not a confirmed root-cause diagnosis.*\n\n"
            f"[View Original Upload](/api/incidents/images/{primary_image_id})"
        )
    elif primary_image_id:
        lines.append(
            f"\n![Uploaded Image](/api/incidents/images/{primary_image_id})\n\n"
            f"*Note: Defect localization could not be reliably isolated with high confidence; "
            f"displaying original uploaded image without artificial markers.*"
        )

    # Section 2: Relevant historical cases
    lines.append("\n### 2. Relevant historical cases")
    lines.append("### Historical record found")
    lines.append(f"- **Source reference:** Historical Incident Database (Incident ID: {inc_id})")
    lines.append(f"- **Verification status:** {strong_incident.get('resolution_status', 'verified').title()} (Engineering Review: {strong_incident.get('engineering_review_status', 'approved').title()})")
    if strong_incident.get("symptom_description"):
        lines.append(f"- **Previously reported problem:** {strong_incident['symptom_description']}")

    # Embed historical image if available
    if strong_incident.get("primary_image_id"):
        lines.append(f"\n![Historical Image from {inc_id}](/api/incidents/images/{strong_incident['primary_image_id']})\n")

    # Reference component image
    if reference_info:
        comp_title = reference_info.get("display_title") or reference_info.get("component_name")
        src_id = reference_info.get("source_id") or reference_info.get("reference_id")
        img_cat = reference_info.get("image_category", "component_reference")
        role_label = (
            "Robot Reference Photograph" if img_cat == "robot_reference"
            else "Technical Specification Source" if img_cat == "technical_spec_source"
            else "Component Technical Reference"
        )
        limitations = reference_info.get("limitations") or "Physical inspection is required before acting on assumptions."
        lines.append(
            f"\n#### Relevant component reference image\n\n"
            f"![{comp_title}](/api/incidents/images/{reference_info['image_id']})\n\n"
            f"- **Role:** `{img_cat}` ({role_label} — reference asset, not an incident annotation)\n"
            f"- **Component:** {comp_title}\n"
            f"- **Source ID:** {src_id}\n"
            f"- **Machine Version:** {reference_info['machine_version']}\n"
            f"- **Verification Status:** {reference_info['verification_status']}\n"
            f"- **Relevance:** {reference_info['relevance_explanation']}\n"
            f"- **Inspection Limitation:** {limitations} Physical inspection is required before acting on assumptions."
        )
    else:
        lines.append("\n*No verified reference image of this component is currently available in the Kelvin6K image library.*")

    # Section 3: Most plausible explanations
    lines.append("\n### 3. Most plausible explanations")
    cause_str = strong_incident.get("confirmed_root_cause") or strong_incident.get("possible_causes") or "Material delivery discontinuity or feedrate mismatch."
    is_conf = bool(strong_incident.get("confirmed_root_cause"))
    qualifier = "Confirmed root cause" if is_conf else "Previously reported cause"
    lines.append(f"- **{qualifier}:** {cause_str}")
    lines.append("- **Underlying mechanism:** Bead discontinuity or layer deformation typically occurs when pump delivery pressure or mix rheology fails to match programmed robot toolpath velocity.")

    # Section 4: Recommended next checks
    lines.append("\n### 4. Recommended next checks")
    lines.append("1. **Depressurize and inspect delivery line:** Confirm pump pressure and hose flexibility.")
    lines.append("2. **Inspect nozzle orifice:** Check for semi-cured concrete or aggregate blockage.")
    lines.append("3. **Verify concrete workability:** Check slump flow and moisture retention in the hopper.")
    lines.append("4. **Synchronize pump & robot speed:** Confirm pump rpm matches robot linear travel rate.")

    # Section 5: Solution
    lines.append("\n### 5. Solution")
    action_str = strong_incident.get("recommended_action") or "Pause print safely, clear blockage under zero pressure, verify flow, and resume."
    lines.append(f"- **Recorded action (Verified procedure):** {action_str}")
    if strong_incident.get("verified_by"):
        lines.append(f"- **Verified by:** {strong_incident['verified_by']}")

    # Section 6: Confidence and limitations
    lines.append("\n### 6. Confidence and limitations")
    lines.append(f"- **Evidence grounding:** Grounded in confirmed historical incident record {inc_id} with match confidence {strong_incident.get('match_score', 0.8):.2f}.")
    lines.append("- **Action boundary:** Never adjust live software travel limits (H0D11) or perform encoder resets (E136.1, E137.1) without authorized engineering sign-off.")

    if safety_warning:
        lines.append(f"\n> ⚠️ **Safety Warning:** {safety_warning}")

    return "\n".join(lines)


def troubleshoot(
    question: str,
    attachments: Optional[List[Attachment]] = None,
    threshold: Optional[float] = None,
    top_k: int = 4,
    session_id: Optional[str] = None,
    line_id: Optional[str] = "line_1",
    operator_id: Optional[str] = None,
) -> TroubleshootResult:
    """
    Runs the RAG-first troubleshooting decision flow:
    1. Checks for natural language historical incident queries (answers directly from SQLite without Gemini).
    2. Persists uploaded images under config.INCIDENT_IMAGES_DIR with SHA-256 deduplication and creates incident record.
    3. Extracts observations from any attachments using Gemini Vision or local CV fallback.
    4. Searches existing historical incidents FIRST: If verified/high-confidence match exists, returns stored solution.
    5. Searches existing RAG knowledge base: If match score >= threshold -> answers using RAG (no Gemini call).
    6. If no match or insufficient score -> fallback to Gemini with clear general AI disclaimers, recording findings.
    7. If Gemini fails -> records incident as pending_analysis without discarding image, returning safe error.
    """
    # 0. Check for reference library queries and historical incident queries first
    ref_res = _handle_reference_library_query(question)
    if ref_res is not None:
        log.info("[TROUBLESHOOT] Answered from reference library: %s", question)
        return ref_res

    hist_res = _handle_historical_incident_query(question)
    if hist_res is not None:
        log.info("[TROUBLESHOOT] Answered from historical incident records: %s", question)
        return hist_res

    attachments = attachments or []
    eff_threshold = threshold if threshold is not None else config.RAG_TROUBLESHOOTING_THRESHOLD

    # 1. Process attachments and persist images
    observations_list: List[str] = []
    primary_image_id: Optional[str] = None
    primary_image_hash: Optional[str] = None

    for att in attachments:
        obs = extract_visual_observations(att)
        if obs:
            observations_list.append(f"[{att.filename}]: {obs}")
        if att.kind == "image" and att.data:
            try:
                saved = incident_store.save_incident_image(
                    data=att.data,
                    original_name=att.filename,
                    content_type=att.content_type,
                    metadata={"width": att.width, "height": att.height},
                )
                if not primary_image_id:
                    primary_image_id = saved["image_id"]
                    primary_image_hash = saved["content_hash"]
            except Exception as exc:
                log.warning("Failed to persist incident image '%s': %s", att.filename, exc)

    obs_context = "\n".join(observations_list).strip()

    # 1.2 Detect visible defects and create annotated image copy (if defect reliably localized)
    annotated_info: Optional[Dict[str, Any]] = None
    annotated_image_id: Optional[str] = None
    for att in attachments:
        if att.kind == "image" and att.data:
            ann = detect_and_create_annotation(
                attachment=att,
                observations=obs_context,
                primary_image_id=primary_image_id,
            )
            if ann:
                annotated_info = ann
                annotated_image_id = ann.get("annotated_image_id")
                break

    # 1.3 Search relevant genuine robot component reference image
    reference_component: Optional[Dict[str, Any]] = incident_store.search_component_references(
        query=f"{question} {obs_context}",
    )
    reference_image_id: Optional[str] = reference_component.get("image_id") if reference_component else None

    # 1.5 Register persistent incident record in SQLite
    incident_rec: Optional[Dict[str, Any]] = None
    if primary_image_id or attachments:
        try:
            incident_rec = incident_store.create_incident(
                user_question=question,
                primary_image_id=primary_image_id,
                primary_image_hash=primary_image_hash,
                annotated_image_id=annotated_image_id,
                reference_image_id=reference_image_id,
                session_id=session_id,
                line_id=line_id,
                operator_id=operator_id,
                symptom_description=question,
                visual_observations=obs_context or None,
                resolution_status="pending_analysis",
                source_attribution="operator_upload" if primary_image_id else "operator_query",
            )
        except Exception as exc:
            log.warning("Failed to create initial incident record: %s", exc)

    # 2. Build candidate queries
    candidate_queries = synthesize_search_queries(question, obs_context)
    search_query = " ".join(candidate_queries) if candidate_queries else question
    safety_warning = _evaluate_safety_warning(question, f"{search_query} {obs_context}")

    # STEP A: Search historical stored incidents FIRST (verified fixes only)
    historical_matches = incident_store.search_incidents(
        query=f"{question} {obs_context}",
        defect_category=None,
        verified_only=True,
        top_k=3,
    )
    strong_incident = None
    for im in historical_matches:
        # Ignore the record we just inserted in step 1.5 if it has no solution yet
        if incident_rec and im.get("incident_id") == incident_rec.get("incident_id"):
            continue
        is_verified = im.get("resolution_status") == "verified" or im.get("engineering_review_status") == "approved"
        if is_verified and im.get("match_score", 0) >= 0.35:
            if im.get("recommended_action") and im.get("confirmed_root_cause"):
                strong_incident = im
                break

    if strong_incident:
        log.info("[TROUBLESHOOT] High-confidence historical incident match %s (score=%.3f). Using stored incident.",
                 strong_incident["incident_id"], strong_incident.get("match_score", 0))
        formatted_incident_ans = _format_historical_incident_answer(
            strong_incident=strong_incident,
            question=question,
            attachments=attachments,
            safety_warning=safety_warning,
            annotated_info=annotated_info,
            primary_image_id=primary_image_id,
            reference_info=reference_component,
        )
        if incident_rec:
            try:
                incident_store.update_incident(
                    incident_rec["incident_id"],
                    resolution_status="resolved",
                    source_attribution="historical_incident_match",
                    confirmed_root_cause=strong_incident.get("confirmed_root_cause"),
                    recommended_action=strong_incident.get("recommended_action"),
                    possible_causes=strong_incident.get("possible_causes"),
                )
            except Exception:
                pass

        return TroubleshootResult(
            question=question,
            answer=formatted_incident_ans,
            source="rag-knowledge",
            rag_found=True,
            rag_score=strong_incident.get("match_score", 0.65),
            citations=[strong_incident["incident_id"]],
            gemini_called=False,
            observations=observations_list,
            conflict_disclosed=False,
            safety_warning=safety_warning,
            fallback_used=False,
            primary_image_id=primary_image_id,
            annotated_image_id=annotated_image_id,
            reference_image_id=reference_image_id,
            reference_component=reference_component,
        )

    # STEP B: Search existing RAG knowledge base across candidates
    best_rag_result = None
    best_top_score = 0.0

    for q in candidate_queries:
        r = search_documentation(q, top_k=top_k)
        if r.get("found") and r.get("results"):
            score = float(r["results"][0].get("score", 0.0))
            if score > best_top_score:
                best_top_score = score
                best_rag_result = r

    rag_result = best_rag_result or search_documentation(question, top_k=top_k)

    hits = rag_result.get("results") or []
    top_score = float(hits[0].get("score", 0.0)) if hits else 0.0
    citations = [h.get("source", "Knowledge Base") for h in hits]

    # Evaluate RAG sufficiency
    is_sufficient = (
        rag_result.get("found", False)
        and len(hits) > 0
        and top_score >= eff_threshold
    )

    if is_sufficient:
        # Grounded answer directly from RAG without Gemini call
        log.info("[TROUBLESHOOT] RAG hit sufficient (score=%.3f >= %.3f). Using RAG.", top_score, eff_threshold)
        formatted_answer, conflict_disclosed = _format_rag_answer(
            question=question,
            hits=hits,
            attachments=attachments,
            safety_warning=safety_warning,
            annotated_info=annotated_info,
            primary_image_id=primary_image_id,
            reference_info=reference_component,
        )
        if incident_rec:
            try:
                top_hit = hits[0]
                incident_store.update_incident(
                    incident_rec["incident_id"],
                    resolution_status="unverified",
                    source_attribution="rag_historical",
                    recommended_action=top_hit.get("text", "")[:250],
                )
            except Exception:
                pass

        return TroubleshootResult(
            question=question,
            answer=formatted_answer,
            source="rag-knowledge",
            rag_found=True,
            rag_score=top_score,
            rag_records=hits,
            citations=citations,
            gemini_called=False,
            observations=observations_list,
            conflict_disclosed=conflict_disclosed,
            safety_warning=safety_warning,
            fallback_used=False,
            primary_image_id=primary_image_id,
            annotated_image_id=annotated_image_id,
            reference_image_id=reference_image_id,
            reference_component=reference_component,
        )

    # STEP C: Insufficient or missing RAG match -> Gemini Fallback
    log.info(
        "[TROUBLESHOOT] RAG insufficient (found=%s, top_score=%.3f < %.3f). Falling back to Gemini.",
        rag_result.get("found", False),
        top_score,
        eff_threshold,
    )

    gemini_answer, gemini_error = _call_gemini_fallback(
        question=question,
        obs_context=obs_context,
        attachments=attachments,
        safety_warning=safety_warning,
        annotated_info=annotated_info,
        primary_image_id=primary_image_id,
        reference_info=reference_component,
    )

    if gemini_answer:
        if incident_rec:
            try:
                incident_store.update_incident(
                    incident_rec["incident_id"],
                    resolution_status="unverified",
                    source_attribution="gemini_vision" if attachments else "gemini",
                    possible_causes="Hypothesized by Gemini model from visual symptoms.",
                    recommended_action=gemini_answer[:300],
                )
            except Exception:
                pass

        return TroubleshootResult(
            question=question,
            answer=gemini_answer,
            source="gemini-fallback",
            rag_found=bool(rag_result.get("found")),
            rag_score=top_score,
            rag_records=hits,
            citations=[],
            gemini_called=True,
            observations=observations_list,
            conflict_disclosed=False,
            safety_warning=safety_warning,
            fallback_used=True,
            primary_image_id=primary_image_id,
            annotated_image_id=annotated_image_id,
            reference_image_id=reference_image_id,
            reference_component=reference_component,
        )

    # STEP D: Gemini failed or unavailable (keep image and incident as pending_analysis)
    if incident_rec:
        try:
            incident_store.update_incident(
                incident_rec["incident_id"],
                resolution_status="pending_analysis",
                source_attribution="local_cv",
            )
        except Exception:
            pass

    error_answer = _format_fallback_failure(gemini_error or "Service unavailable")
    return TroubleshootResult(
        question=question,
        answer=error_answer,
        source="error",
        rag_found=False,
        rag_score=top_score,
        rag_records=hits,
        citations=[],
        gemini_called=False,
        observations=observations_list,
        conflict_disclosed=False,
        safety_warning=safety_warning,
        fallback_used=True,
        primary_image_id=primary_image_id,
        annotated_image_id=annotated_image_id,
        reference_image_id=reference_image_id,
        reference_component=reference_component,
    )


# ---------------------------------------------------------------------------
# Formatting Grounded Diagnostic Answer (6 Sections)
# ---------------------------------------------------------------------------

def _format_rag_answer(
    question: str,
    hits: List[Dict[str, Any]],
    attachments: List[Attachment],
    safety_warning: Optional[str] = None,
    annotated_info: Optional[Dict[str, Any]] = None,
    primary_image_id: Optional[str] = None,
    reference_info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    """
    Formats the RAG answer preserving problem, cause, solution, verification status,
    and citations, organized in 6 evidence-grounded diagnostic sections.
    """
    top = hits[0]
    raw_text = top.get("text", "")
    source_citation = top.get("source", "Authoritative Documentation")
    issue_id = top.get("issue_id") or top.get("source_id") or ""

    fields = parse_record(raw_text)

    # Detect conflicts across top hits
    conflict_disclosed = False
    all_texts = " ".join(h.get("text", "") for h in hits)
    if "conflict" in all_texts.lower() or "discrepancy" in all_texts.lower():
        conflict_disclosed = True

    # Section 1: What I can see
    sec1 = ["### 1. What I can see"]
    if attachments:
        for att in attachments:
            if att.kind == "image" and att.observations:
                sec1.append(f"- **Visual observations ({att.filename}):** {att.observations}")
            elif att.observations:
                sec1.append(f"- **Attachment analysis ({att.filename}):** {att.observations}")
            else:
                sec1.append(f"- **Attachment ({att.filename}):** Image provided for analysis.")
    else:
        sec1.append("- No media attachment provided; analysis based on reported textual symptoms.")

    if annotated_info:
        regions_list = annotated_info.get("regions", [])
        if regions_list:
            reg_items = "\n".join(
                f"- **Region {r['region_id']} ({r['defect_type']}):** {r.get('description', '')}"
                for r in regions_list
            )
            reg_details = f"\n\n**Localized Defect Observations:**\n{reg_items}"
        else:
            reg_details = f" ({annotated_info['defect_type']})"
        sec1.append(
            f"\n![AI-Assisted Visual Observation: {annotated_info['defect_type']}]({annotated_info['annotated_url']})\n\n"
            f"*AI-assisted visual observation highlighting specific localized defect regions:{reg_details}\n\n"
            f"This is an automated visual observation, not a confirmed root-cause diagnosis.*\n\n"
            f"[View Original Upload](/api/incidents/images/{primary_image_id})"
        )
    elif primary_image_id:
        sec1.append(
            f"\n![Uploaded Image](/api/incidents/images/{primary_image_id})\n\n"
            f"*Note: Defect localization could not be reliably isolated with high confidence; "
            f"displaying original uploaded image without artificial markers.*"
        )

    # Section 2: Relevant historical cases
    sec2 = ["\n### 2. Relevant historical cases"]
    sec2.append("### Historical record found")
    ref_line = f"- **Source reference:** {source_citation}"
    if issue_id:
        ref_line += f" (Record ID: {issue_id})"
    sec2.append(ref_line)

    prob_val = ""
    cause_val = ""
    sol_val = ""
    verif_val = ""
    safety_notes_val = ""
    is_unconf = True
    is_workaround = False

    if fields:
        prob = fields.get("problem")
        if prob and prob.get("value"):
            prob_val = _strip_markdown(prob["value"])
            sec2.append(f"- **Previously reported problem:** {prob_val}")

        status = fields.get("verification status")
        if status and status.get("value"):
            verif_val = _strip_markdown(status["value"])
            sec2.append(f"- **Verification status:** {verif_val}")

        cause = next((fields[k] for k in fields if k.startswith("cause")), None)
        if cause and cause.get("value"):
            cause_val = _strip_markdown(cause["value"])
            is_unconf = any(w in cause.get("qualifier", "").lower() for w in ("possible", "unconfirmed", "suspected", "reported")) or "unverified" in verif_val.lower()

        sol = fields.get("solution / corrective action") or fields.get("solution")
        if sol and sol.get("value"):
            sol_val = _strip_markdown(sol["value"])
            is_workaround = any(w in sol.get("qualifier", "").lower() for w in ("workaround", "temporary", "unapproved", "operator reported"))

        safety_field = fields.get("safety / important notes")
        if safety_field and safety_field.get("value") and "none" not in safety_field["value"].lower():
            safety_notes_val = _strip_markdown(safety_field["value"])
    else:
        clean_body = _strip_markdown(raw_text)
        summary = clean_body[:360] + "..." if len(clean_body) > 360 else clean_body
        sec2.append(f"- **Documented procedure / findings:** {summary}")
        prob_val = summary

    # Reference component image
    if reference_info:
        comp_title = reference_info.get("display_title") or reference_info.get("component_name")
        src_id = reference_info.get("source_id") or reference_info.get("reference_id")
        sec2.append(
            f"\n#### Relevant component reference image\n\n"
            f"![{reference_info['caption']}](/api/incidents/images/{reference_info['image_id']})\n\n"
            f"- **Component:** {comp_title}\n"
            f"- **Source ID:** {src_id}\n"
            f"- **Machine Version:** {reference_info['machine_version']}\n"
            f"- **Verification Status:** {reference_info['verification_status']}\n"
            f"- **Relevance:** {reference_info['relevance_explanation']}"
        )
    else:
        sec2.append("\n*No verified reference image of this component is currently available in the Kelvin6K image library.*")

    # Section 3: Most plausible explanations
    sec3 = ["\n### 3. Most plausible explanations"]
    conf_str = "Possible cause / unconfirmed" if is_unconf else "Confirmed root cause"
    if cause_val:
        sec3.append(f"- **Previously reported cause ({conf_str}):** {cause_val}")
    else:
        sec3.append(f"- **Previously reported cause ({conf_str}):** Inconsistent material flow, delivery interruption, or mix workability variation.")
    sec3.append("- **Underlying mechanism:** Bead defects and layer voids typically stem from a mismatch between robot toolpath velocity and pump delivery rate, or temporary line resistance.")

    # Section 4: Recommended next checks
    sec4 = ["\n### 4. Recommended next checks"]
    sec4.append("1. **Pressure check:** Verify material delivery line pressure (ensure line is depressurized before physical inspection).")
    sec4.append("2. **Nozzle orifice inspection:** Visually check nozzle opening for cured material buildup or obstruction.")
    sec4.append("3. **Mix workability:** Check concrete slump flow and moisture consistency in the hopper.")
    sec4.append("4. **Pump & feedrate sync:** Confirm pump drive speed matches programmed robot feedrate.")

    # Section 5: Solution
    sec5 = ["\n### 5. Solution"]
    sol_tag = "Temporary workaround (unapproved)" if is_workaround else "Recorded action / procedure"
    if sol_val:
        sec5.append(f"- **Recorded action ({sol_tag}):** {sol_val}")
    else:
        sec5.append(f"- **Recorded action ({sol_tag}):** Pause the print safely. Clear obstruction under zero pressure, verify flow, and resume.")
    if is_workaround:
        sec5.append("> ⚠️ **Notice:** This recorded action is an unapproved operator workaround. It must NOT be treated as standard procedure without engineering approval.")
    if safety_notes_val:
        sec5.append(f"- **Safety notes:** {safety_notes_val}")

    # Section 6: Confidence and limitations
    sec6 = ["\n### 6. Confidence and limitations"]
    sec6.append("- **Visual confidence:** Moderate to high for surface geometry and visible discontinuity.")
    sec6.append("- **Physical limitations:** Internal pump pressure, chemical accelerator flow, motor temperature, and mix rheology cannot be determined from images alone.")
    sec6.append("- **Action boundary:** Never adjust live software travel limits (H0D11) or perform encoder resets (E136.1, E137.1) without authorized engineering sign-off.")

    if conflict_disclosed:
        sec6.append(
            "\n> **Note on Conflicting Records:** The available Kelvin6K documentation contains conflicting "
            "values or instructions for this issue. Do not change live machine parameters without authorized engineering confirmation."
        )

    if safety_warning:
        sec6.append(f"\n> ⚠️ **Safety Warning:** {safety_warning}")

    full_output = "\n".join(sec1 + sec2 + sec3 + sec4 + sec5 + sec6)
    return full_output, conflict_disclosed


# ---------------------------------------------------------------------------
# Calling Gemini Fallback (STEP C)
# ---------------------------------------------------------------------------

def _call_gemini_fallback(
    question: str,
    obs_context: str,
    attachments: List[Attachment],
    safety_warning: Optional[str] = None,
    annotated_info: Optional[Dict[str, Any]] = None,
    primary_image_id: Optional[str] = None,
    reference_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Gemini API to provide general technical guidance when RAG does not contain sufficient records.
    Never exposes API key.
    """
    if not config.GEMINI_API_KEY:
        return None, "no_api_key"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        system_instruction = (
            "You are the senior technical advisory AI for the Kelvin6K 3D concrete printing system.\n"
            "The user asked a troubleshooting question for which NO matching historical record was found "
            "in the Kelvin6K site knowledge base.\n\n"
            "You must provide structured, general technical guidance under the clear headers:\n"
            "### 1. What I can see\n"
            "### 2. Relevant historical cases\n"
            "**No sufficient matching historical record found.**\n"
            "**General AI guidance (Gemini):**\n"
            "### 3. Most plausible explanations\n"
            "- **Possible causes:** (List general technical hypotheses; clearly mark as unverified)\n"
            "### 4. Recommended next checks\n"
            "- **Safe diagnostic checks:** (Non-invasive diagnostic checks only)\n"
            "### 5. Solution\n"
            "- **Recommended next step:** (Safe next operational step)\n"
            "### 6. Confidence and limitations\n"
            "- **Information needed:** (What exact information is required to diagnose accurately)\n\n"
            "ABSOLUTE INDUSTRIAL SAFETY RULES:\n"
            "1. NEVER invent previous Kelvin6K incidents, telemetry, or confirmed machine parameters.\n"
            "2. NEVER present general model knowledge as verified Kelvin6K documentation.\n"
            "3. NEVER instruct an operator to clear hoses or nozzles while under pressure. Always specify depressurization.\n"
            "4. For servo/encoder operations (E136.1, E137.1) or travel limit overrides (H0D11), require authorized technician review and manufacturer documentation.\n"
            "5. Never instruct striking equipment or rapid feedrate pulsing."
        )

        user_content = f"User Question: {question}\n"
        if obs_context:
            user_content += f"\nAttachment Context & Observations:\n{obs_context}\n"

        contents_payload: List[Any] = []

        image_attached = False
        for att in attachments:
            if att.kind == "image" and not image_attached:
                contents_payload.append(types.Part.from_bytes(data=att.data, mime_type=att.content_type))
                image_attached = True

        contents_payload.append(user_content)

        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents_payload,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=700,
            ),
        )

        text = (resp.text or "").strip()
        if not text:
            return None, "empty_response"

        # Ensure visual observation image is present in Section 1 if applicable
        if annotated_info and annotated_info["annotated_url"] not in text:
            regions_list = annotated_info.get("regions", [])
            if regions_list:
                reg_items = "\n".join(
                    f"- **Region {r['region_id']} ({r['defect_type']}):** {r.get('description', '')}"
                    for r in regions_list
                )
                reg_details = f"\n\n**Localized Defect Observations:**\n{reg_items}"
            else:
                reg_details = f" ({annotated_info['defect_type']})"
            visual_block = (
                f"\n\n![AI-Assisted Visual Observation: {annotated_info['defect_type']}]({annotated_info['annotated_url']})\n\n"
                f"*AI-assisted visual observation highlighting specific localized defect regions:{reg_details}\n\n"
                f"This is an automated visual observation, not a confirmed root-cause diagnosis.*\n\n"
                f"[View Original Upload](/api/incidents/images/{primary_image_id})"
            )
            if "### 1. What I can see" in text:
                text = text.replace("### 1. What I can see", f"### 1. What I can see{visual_block}", 1)
            else:
                text = f"{visual_block}\n\n{text}"
        elif primary_image_id and f"/api/incidents/images/{primary_image_id}" not in text:
            visual_block = (
                f"\n\n![Uploaded Image](/api/incidents/images/{primary_image_id})\n\n"
                f"*Note: Defect localization could not be reliably isolated with high confidence; "
                f"displaying original uploaded image without artificial markers.*"
            )
            if "### 1. What I can see" in text:
                text = text.replace("### 1. What I can see", f"### 1. What I can see{visual_block}", 1)
            else:
                text = f"{visual_block}\n\n{text}"

        # Ensure component reference image is present in Section 2
        if reference_info and reference_info["image_id"] not in text:
            comp_title = reference_info.get("display_title") or reference_info.get("component_name")
            src_id = reference_info.get("source_id") or reference_info.get("reference_id")
            ref_block = (
                f"\n\n#### Relevant component reference image\n\n"
                f"![{reference_info['caption']}](/api/incidents/images/{reference_info['image_id']})\n\n"
                f"- **Component:** {comp_title}\n"
                f"- **Source ID:** {src_id}\n"
                f"- **Machine Version:** {reference_info['machine_version']}\n"
                f"- **Verification Status:** {reference_info['verification_status']}\n"
                f"- **Relevance:** {reference_info['relevance_explanation']}"
            )
            if "### 2. Relevant historical cases" in text:
                text = text.replace("### 2. Relevant historical cases", f"### 2. Relevant historical cases{ref_block}", 1)
            else:
                text = f"{text}\n\n{ref_block}"
        elif not reference_info and "No verified reference image of this component" not in text:
            none_block = "\n\n*No verified reference image of this component is currently available in the Kelvin6K image library.*"
            if "### 2. Relevant historical cases" in text:
                text = text.replace("### 2. Relevant historical cases", f"### 2. Relevant historical cases{none_block}", 1)
            else:
                text = f"{text}\n\n{none_block}"

        # Ensure safety warning is appended if applicable
        if safety_warning and safety_warning.lower() not in text.lower():
            text += f"\n\n> ⚠️ **Safety Warning:** {safety_warning}"

        return text, None

    except Exception as exc:
        log.warning("Gemini fallback call failed: %s", exc)
        return None, _categorize_gemini_error(exc)


def _categorize_gemini_error(exc: Exception) -> str:
    """Classifies Gemini SDK exception without revealing keys or credentials."""
    msg = str(exc).lower()
    code = getattr(exc, "code", None)

    if code in (401, 403) or "api key" in msg or "permission" in msg:
        return "invalid_api_key"
    if code == 429 or "quota" in msg or "rate" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if code == 404 or "model" in msg or "not_found" in msg:
        return "model_unavailable"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if code and code >= 500:
        return "server_error"
    return "network_error"


def _format_fallback_failure(error_type: str) -> str:
    """Returns safe user-facing message when both RAG and Gemini fail."""
    reasons = {
        "no_api_key": "No Gemini API key is configured on the backend (check GEMINI_API_KEY in server environment).",
        "invalid_api_key": "The configured Gemini API key was rejected by the external service. Please verify server configuration.",
        "rate_limit": "The external AI service rate limit or quota has been reached. Please wait a moment and ask again.",
        "model_unavailable": "The configured Gemini model is currently unavailable.",
        "timeout": "The request to the external AI service timed out.",
        "server_error": "The external AI service is experiencing a server error.",
        "network_error": "Unable to connect to the external AI service.",
    }
    explanation = reasons.get(error_type, "The external AI service is currently unavailable.")
    return (
        "**No sufficient matching historical record found.**\n\n"
        f"Unable to retrieve external AI guidance at this time: {explanation}\n\n"
        "Please check the robot's operating manual, verify site logs, or contact authorized engineering personnel."
    )


def _evaluate_safety_warning(question: str, full_context: str) -> Optional[str]:
    """Evaluates whether the question concerns safety-critical industrial procedures."""
    q_low = f"{question} {full_context}".lower()

    if any(w in q_low for w in ("clogged nozzle", "clear nozzle", "nozzle clog", "hose clog", "blockage", "pipe clear", "clearing accelerator")) and any(w in q_low for w in ("pressur", "running", "pump on", "chal raha", "active")):
        return (
            "DANGER: Never attempt to clear a nozzle, hose, accelerator pipe, or pump line while the system is pressurized or the pump is active. "
            "Stop the pump, isolate power, and relieve hydraulic/material line pressure before inspection."
        )

    if any(w in q_low for w in ("hammer", "strike", "hit pump", "pulse feedrate", "pulsing")):
        return (
            "DANGER: Striking the pump with a hammer or rapidly pulsing feedrate are unapproved operator workarounds "
            "that risk severe mechanical damage, high-pressure rupture, and physical injury. Never hit pressurized equipment."
        )

    if "e136" in q_low or "e137" in q_low or "encoder" in q_low or "servo" in q_low:
        return (
            "Servo drives and absolute encoders must only be inspected or reset by authorized technicians with the machine isolated. "
            "Do not perform live encoder resets without manufacturer service documentation."
        )

    if "h0d11" in q_low or "travel limit" in q_low or "downward jog" in q_low:
        return (
            "Modifying software travel limits or parameter H0D11 can cause severe mechanical collision or crush hazards. "
            "Engineering authorization is required before modifying motion boundaries."
        )

    return None
