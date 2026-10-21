"""
tests/test_site_camera.py - Tests for dedicated Camera 2 site people counting architecture.
"""

import time
import pytest
from unittest.mock import MagicMock

from agent import assistant, config, query_router
from agent.people_detector import PeopleCountResult, set_detector_override
from agent.services import machine_state as ms
from api.site_camera_service import SiteCameraService


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure detector override and site camera service binding are cleaned up between tests."""
    yield
    set_detector_override(None)
    ms.bind_site_camera_service(None)


def test_camera_isolation():
    """
    Verify Camera 1 and Camera 2 configuration and runtime isolation.
    Camera 1 uses CAMERA_INDEX (0) for metrology/printing.
    Camera 2 uses SITE_CAMERA_INDEX (1) dedicated to site safety/people counting.
    """
    assert config.CAMERA_INDEX == 0, "Camera 1 must remain on device index 0"
    assert config.SITE_CAMERA_INDEX == 1, "Camera 2 must be on dedicated device index 1"
    assert config.CAMERA_INDEX != config.SITE_CAMERA_INDEX, "Cameras 1 and 2 must use distinct device indices"

    service = SiteCameraService()
    assert service.index == config.SITE_CAMERA_INDEX
    assert service.source == config.SITE_CAMERA_SOURCE


def test_site_camera_offline_provenance():
    """
    When Camera 2 is offline or unavailable, the system must NEVER report 0 people.
    It must return available=False with UNAVAILABLE provenance and an informative reason.
    """
    # 1. Unbound service
    ms.bind_site_camera_service(None)
    data = ms.get_site_people_count()
    assert data["available"] is False
    assert data["provenance"] == "UNAVAILABLE"
    assert data["people_count"] is None
    assert "not bound" in data["reason"].lower() or "not initialized" in data["reason"].lower()

    # 2. Service initialized but offline/unavailable
    mock_service = MagicMock(spec=SiteCameraService)
    mock_service.status = "unavailable"
    mock_service.error = "Camera 2 device busy or disconnected"
    ms.bind_site_camera_service(mock_service)

    data = ms.get_site_people_count()
    assert data["available"] is False
    assert data["provenance"] == "UNAVAILABLE"
    assert data["people_count"] is None
    assert "unavailable" in data["reason"].lower()

    # 3. Router result and summary when Camera 2 is offline
    res = query_router.route("How many people are currently on site?")
    assert res.intent == query_router.Intent.GET_SITE_PEOPLE_COUNT
    assert res.available is False

    summary = query_router.summarise(res)
    assert "unavailable" in summary.lower()
    assert "0 people" not in summary.lower()
    assert "no people" not in summary.lower()


def test_site_camera_counts_and_summaries():
    """
    Verify accurate reporting and deterministic phrasing for 0, 1, and 2+ people counts.
    Also verifies low-confidence note generation.
    """
    mock_service = MagicMock(spec=SiteCameraService)
    mock_service.status = "live"
    ms.bind_site_camera_service(mock_service)

    # 1. Zero people detected
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=0,
        confidence=0.95,
        available=True,
        detector="gemini",
        details="none",
    )
    res_0 = query_router.route("How many people are currently on site?")
    assert res_0.intent == query_router.Intent.GET_SITE_PEOPLE_COUNT
    assert res_0.available is True
    assert res_0.facts["people_count"] == 0
    assert res_0.facts["provenance"] == "REAL"
    summary_0 = query_router.summarise(res_0)
    assert "No people are currently detected on site via Camera 2." in summary_0

    # 2. Single person detected
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=1,
        confidence=0.92,
        available=True,
        detector="gemini",
        details="1 operator near control panel",
    )
    res_1 = query_router.route("How many people are on site?")
    assert res_1.intent == query_router.Intent.GET_SITE_PEOPLE_COUNT
    assert res_1.available is True
    assert res_1.facts["people_count"] == 1
    summary_1 = query_router.summarise(res_1)
    assert "There is 1 person currently visible on site via Camera 2." in summary_1

    # 3. Multiple people (3) detected
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=3,
        confidence=0.88,
        available=True,
        detector="gemini",
        details="3 workers near printing bed",
    )
    res_3 = query_router.route("How many workers on site?")
    assert res_3.intent == query_router.Intent.GET_SITE_PEOPLE_COUNT
    assert res_3.available is True
    assert res_3.facts["people_count"] == 3
    summary_3 = query_router.summarise(res_3)
    assert "There are 3 people currently visible on site via Camera 2." in summary_3

    # 4. Low-confidence detection
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=2,
        confidence=0.50,  # Below threshold 0.65
        available=True,
        detector="gemini",
        details="shadowy figures",
    )
    res_low = query_router.route("How many people are visible?")
    assert res_low.intent == query_router.Intent.GET_SITE_PEOPLE_COUNT
    assert res_low.available is True
    summary_low = query_router.summarise(res_low)
    assert "There are 2 people currently visible on site via Camera 2." in summary_low
    assert "The count is uncertain (50% confidence)." in summary_low


def test_site_camera_caching():
    """
    Verify 2-second caching mechanism in SiteCameraService.
    Multiple queries within 2.0s should return the cached PeopleCountResult without re-calling detector.
    """
    call_counter = 0

    def mock_detector(jpeg):
        nonlocal call_counter
        call_counter += 1
        return PeopleCountResult(
            people_count=2,
            confidence=0.9,
            available=True,
            detector="mock",
        )

    set_detector_override(mock_detector)
    service = SiteCameraService()
    service.status = "live"
    service._jpeg = b"\xff\xd8fakejpeg\xff\xd9"
    service.cache_ttl = 2.0

    # First call: triggers detector
    res1 = service.get_people_count()
    assert res1.people_count == 2
    assert call_counter == 1

    # Second call immediately: must hit cache
    res2 = service.get_people_count()
    assert res2.people_count == 2
    assert call_counter == 1

    # Third call with force_refresh=True: bypasses cache
    res3 = service.get_people_count(force_refresh=True)
    assert res3.people_count == 2
    assert call_counter == 2


def test_query_router_bilingual_patterns():
    """
    Verify English, Hindi, and Hinglish query variations trigger Intent.GET_SITE_PEOPLE_COUNT.
    """
    queries = [
        # English
        "How many people are on site?",
        "How many people are currently on site?",
        "How many people are working on site?",
        "How many people are there right now?",
        "How many people on site?",
        "How many workers are on site?",
        "How many people are visible on camera?",
        "People count on site",
        "Site people count",
        "How many persons on site?",
        "How many people in the room?",
        # Hindi / Hinglish
        "Abhi site pe kitne log hain?",
        "Site par kitne log hain?",
        "Site pe kitne log hain?",
        "Camera me kitne log dikh rahe hain?",
        "Site par kitne workers hain?",
        "Kitne log site par hain?",
        "Site par kitne vyakti hain?",
    ]

    for q in queries:
        intent, hits = query_router.classify(q)
        assert intent == query_router.Intent.GET_SITE_PEOPLE_COUNT, (
            f"Expected GET_SITE_PEOPLE_COUNT for '{q}', got {intent} (hits: {hits})"
        )


def test_assistant_deterministic_resolution():
    """
    Verify that Kelvin assistant dispatches GET_SITE_PEOPLE_COUNT deterministically.
    """
    mock_service = MagicMock(spec=SiteCameraService)
    mock_service.status = "live"
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=4,
        confidence=0.91,
        available=True,
        detector="gemini",
    )
    ms.bind_site_camera_service(mock_service)

    bot = assistant.Assistant(use_llm=False)
    answer = bot.ask("How many people are currently on site?")
    assert "There are 4 people currently visible on site via Camera 2." in answer

    answer_hi = bot.ask("Abhi site pe kitne log hain?")
    assert "There are 4 people currently visible on site via Camera 2." in answer_hi


def test_yolo_person_detection_and_overlay():
    """
    Verify YOLO detection returns bounding boxes and draw_people_overlay generates green boxes & red warning.
    """
    import cv2
    import numpy as np
    from agent.people_detector import PeopleDetector, draw_people_overlay

    detector = PeopleDetector()
    # 1. Blank frame produces 0 detections and red warning
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    res_0 = detector.detect_frame(blank)
    assert res_0.people_count == 0
    assert len(res_0.boxes) == 0

    overlay_0 = draw_people_overlay(blank, res_0)
    assert overlay_0.shape == blank.shape
    # Check warning text area contains red pixels (BGR: [0, 0, 255] or high red)
    assert np.any(overlay_0[:, :, 2] > 200)

    # 2. Frame with simulated person detection produces green bounding box & label
    simulated_result = PeopleCountResult(
        people_count=1,
        confidence=0.92,
        details="1 person detected",
        available=True,
        detector="yolo",
        boxes=[([200, 100, 300, 500], 0.92)],
    )
    overlay_1 = draw_people_overlay(blank, simulated_result)
    assert overlay_1.shape == blank.shape
    # Check bounding box line contains green pixels (BGR: [0, 255, 0] or high green)
    assert np.any(overlay_1[:, :, 1] > 200)



def test_yolo_instance_segmentation_multiclass_and_held_objects():
    """
    Verify YOLO instance segmentation detects multiple classes, multiple instances,
    objects held by person (bottle, phone), and provides all per-entity fields:
    class_name, confidence, box: [x, y, w, h], width_px, height_px, mask, track_id.
    """
    from agent.people_detector import DetectedEntity, PeopleCountResult

    # Simulate segmentation detection with a person holding a bottle and a backpack nearby
    entity_person = DetectedEntity(
        class_id=0,
        class_name="person",
        confidence=0.94,
        box=[100, 150, 200, 450],
        width_px=200,
        height_px=450,
        mask=[[100, 150], [300, 150], [300, 600], [100, 600]],
        track_id=1,
    )
    entity_bottle = DetectedEntity(
        class_id=39,
        class_name="bottle",
        confidence=0.88,
        box=[180, 280, 50, 120],
        width_px=50,
        height_px=120,
        mask=[[180, 280], [230, 280], [230, 400], [180, 400]],
        track_id=2,
    )
    entity_backpack = DetectedEntity(
        class_id=24,
        class_name="backpack",
        confidence=0.85,
        box=[120, 220, 90, 140],
        width_px=90,
        height_px=140,
        mask=[[120, 220], [210, 220], [210, 360], [120, 360]],
        track_id=None,
    )

    result = PeopleCountResult(
        people_count=1,
        confidence=0.89,
        available=True,
        detector="yolo",
        entities=[entity_person, entity_bottle, entity_backpack],
        supported_classes=["person", "bottle", "backpack"],
    )

    # 1. Output schema validation
    dicts = result.entities_dict()
    assert len(dicts) == 3
    for d in dicts:
        assert "class_name" in d
        assert "confidence" in d
        assert "box" in d
        assert "width_px" in d
        assert "height_px" in d
        assert "mask" in d
        assert "track_id" in d
        assert d["width_px"] == d["box"][2]
        assert d["height_px"] == d["box"][3]

    # Verify held object is present
    classes = [d["class_name"] for d in dicts]
    assert "person" in classes
    assert "bottle" in classes
    assert "backpack" in classes

    # 2. Rendering on blank canvas
    import numpy as np
    from agent.people_detector import draw_people_overlay
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_people_overlay(blank, result)
    assert rendered.shape == blank.shape
    # Check that drawing wrote green (person) and non-zero pixels
    assert np.any(rendered[:, :, 1] > 200)


def test_tracking_deduplication_for_people_count():
    """
    Verify that when tracking IDs are available, the same tracked person
    is not counted multiple times, while distinct tracked persons are counted.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()
    mock_model = MagicMock()
    detector._model = mock_model

    # Simulate 3 person boxes, but two have the same track_id 1, and one has track_id 2
    mock_box1 = MagicMock()
    mock_box1.xyxy = [np.array([10, 10, 100, 200])]
    mock_box1.conf = [np.array(0.92)]
    mock_box1.cls = [np.array(0)]
    mock_box1.id = [np.array(1)]

    mock_box2 = MagicMock()
    mock_box2.xyxy = [np.array([12, 12, 98, 198])]  # duplicate/overlapping track 1
    mock_box2.conf = [np.array(0.90)]
    mock_box2.cls = [np.array(0)]
    mock_box2.id = [np.array(1)]

    mock_box3 = MagicMock()
    mock_box3.xyxy = [np.array([300, 50, 450, 300])]
    mock_box3.conf = [np.array(0.88)]
    mock_box3.cls = [np.array(0)]
    mock_box3.id = [np.array(2)]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box1, mock_box2, mock_box3]
    mock_res.masks = None
    mock_res.names = {0: "person"}

    mock_model.track.return_value = [mock_res]

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect_frame(blank)

    # 3 raw detections with 1 overlapping duplicate => duplicate suppressed so entities has 2 unique people
    assert res.people_count == 2
    assert len(res.entities) == 2
    track_ids = {e.track_id for e in res.entities}
    assert track_ids == {1, 2}


def test_unsupported_industrial_objects_policy():
    """
    Verify that unsupported industrial objects (pump, nozzle, accelerator, machine parts)
    are NOT hallucinated or mapped from generic detections (bottle, cup, chair).
    """
    from unittest.mock import MagicMock
    from agent.people_detector import DetectedEntity, PeopleCountResult
    from agent.services import machine_state as ms
    from agent import query_router
    from api.site_camera_service import SiteCameraService

    mock_service = MagicMock(spec=SiteCameraService)
    mock_service.status = "live"
    # Detections has a person and a generic bottle
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=1,
        confidence=0.91,
        available=True,
        detector="yolo",
        entities=[
            DetectedEntity(0, "person", 0.95, [100, 100, 200, 400], 200, 400, None, 1),
            DetectedEntity(39, "bottle", 0.87, [150, 200, 40, 100], 40, 100, None, 2),
        ],
        supported_classes=["person", "bottle", "cup", "chair"],
    )
    mock_service.detector = MagicMock()
    mock_service.detector.get_supported_classes.return_value = ["person", "bottle", "cup", "chair"]
    ms.bind_site_camera_service(mock_service)

    # 1. Query for visible objects
    res_objs = query_router.route("what objects are detected on camera 2?")
    assert res_objs.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_objs.available is True
    summary_objs = query_router.summarise(res_objs)
    assert "1 person" in summary_objs.lower()
    assert "1 bottle" in summary_objs.lower()
    # Ensure bottle was NOT renamed to pump or nozzle
    assert "pump" not in summary_objs.lower()
    assert "nozzle" not in summary_objs.lower()

    # 2. Query specifically for pump (unsupported class)
    res_pump = query_router.route("is the pump visible on camera 2?")
    summary_pump = query_router.summarise(res_pump)
    assert "does not support detecting 'pump'" in summary_pump.lower() or "unsupported" in summary_pump.lower()
    # Must NOT claim that the bottle is a pump
    assert "pump is currently detected" not in summary_pump.lower()


def test_site_camera_service_latest_detections():
    """
    Verify that SiteCameraService provides latest_detections() with normalized structured data.
    """
    from agent.people_detector import DetectedEntity, PeopleCountResult

    service = SiteCameraService()
    service.status = "live"
    service._cached_result = PeopleCountResult(
        people_count=2,
        confidence=0.92,
        available=True,
        detector="yolo",
        entities=[
            DetectedEntity(0, "person", 0.94, [10, 10, 100, 200], 100, 200, None, 1),
            DetectedEntity(0, "person", 0.90, [200, 20, 120, 220], 120, 220, None, 2),
        ],
        supported_classes=["person"],
    )

    dets = service.latest_detections()
    assert len(dets) == 2
    assert dets[0]["class_name"] == "person"
    assert dets[0]["width_px"] == 100
    assert dets[0]["height_px"] == 200
    assert dets[0]["track_id"] == 1


def test_decoupled_confidence_thresholds():
    """
    Verify that object detection uses 0.30 threshold so objects (e.g. close-up bottle)
    with confidence between 0.30 and 0.65 are detected, while person counting retains 0.65.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()
    assert detector.object_confidence_threshold == 0.30
    assert detector.confidence_threshold == 0.65

    # Mock YOLO model
    mock_model = MagicMock()
    detector._model = mock_model

    # Bottle with conf=0.45 (close-up/cropped bottle)
    box_bottle = MagicMock()
    box_bottle.xyxy = [np.array([50, 50, 200, 500])]
    box_bottle.conf = [0.45]
    box_bottle.cls = [39]
    box_bottle.id = None

    # Person with conf=0.40 (low-confidence person, below 0.65 safety threshold)
    box_low_person = MagicMock()
    box_low_person.xyxy = [np.array([300, 50, 450, 600])]
    box_low_person.conf = [0.40]
    box_low_person.cls = [0]
    box_low_person.id = None

    # Person with conf=0.88 (valid site worker)
    box_valid_person = MagicMock()
    box_valid_person.xyxy = [np.array([600, 50, 800, 650])]
    box_valid_person.conf = [0.88]
    box_valid_person.cls = [0]
    box_valid_person.id = [1]

    mock_result = MagicMock()
    mock_result.boxes = [box_bottle, box_low_person, box_valid_person]
    mock_result.names = {0: "person", 39: "bottle"}
    mock_result.masks = None
    mock_model.track.return_value = [mock_result]

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy_frame)

    # 1. model.track must have been called with conf=0.30
    mock_model.track.assert_called_once()
    _, kwargs = mock_model.track.call_args
    assert kwargs.get("conf") == 0.30

    # 2. All 3 entities detected above 0.30 are present in entities
    entity_names = [e.class_name for e in res.entities]
    assert entity_names == ["bottle", "person", "person"]
    bottle_entity = next(e for e in res.entities if e.class_name == "bottle")
    assert bottle_entity.confidence == 0.45

    # 3. People count strictly ignores the person with conf 0.40 (< 0.65)
    # Only the person with conf 0.88 is counted
    assert res.people_count == 1
    assert len(res.boxes) == 1
    assert res.boxes[0][1] == 0.88


def test_dynamic_class_support_glasses_unsupported():
    """
    Verify that unsupported objects like glasses are reported as unsupported_class
    and never guessed or mapped to unrelated classes.
    """
    from unittest.mock import MagicMock
    from agent.people_detector import DetectedEntity, PeopleCountResult
    from agent.services import machine_state as ms
    from agent import query_router

    mock_service = MagicMock()
    mock_service.status = "live"
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=1,
        confidence=0.92,
        available=True,
        detector="yolo",
        entities=[
            DetectedEntity(0, "person", 0.95, [100, 100, 200, 400], 200, 400, None, 1),
            DetectedEntity(39, "bottle", 0.87, [150, 200, 40, 100], 40, 100, None, 2),
        ],
        supported_classes=["person", "bottle", "cup", "chair"],
    )
    mock_service.detector = MagicMock()
    mock_service.detector.get_supported_classes.return_value = ["person", "bottle", "cup", "chair"]
    ms.bind_site_camera_service(mock_service)

    res_glasses = query_router.route("are glasses visible on camera 2?")
    assert res_glasses.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_glasses.facts["unsupported_class"] == "glasses"

    summary = query_router.summarise(res_glasses)
    assert "does not support detecting 'glasses'" in summary.lower() or "unsupported" in summary.lower()
    # Must never claim bottle or person is glasses
    assert "glasses is currently detected" not in summary.lower()


def test_dynamic_custom_yolo_classes_without_modifying_router():
    """
    Verify that future custom YOLO models automatically work with query_router.py
    without any hard-coded class lists.
    """
    from unittest.mock import MagicMock
    from agent.people_detector import DetectedEntity, PeopleCountResult
    from agent.services import machine_state as ms
    from agent import query_router

    custom_classes = ["bracket", "hydraulic_valve", "pipe", "nozzle"]

    mock_service = MagicMock()
    mock_service.status = "live"
    mock_service.get_people_count.return_value = PeopleCountResult(
        people_count=0,
        confidence=0.90,
        available=True,
        detector="yolo",
        entities=[
            DetectedEntity(2, "pipe", 0.85, [50, 60, 120, 300], 120, 300, None, 10),
        ],
        supported_classes=custom_classes,
    )
    mock_service.detector = MagicMock()
    mock_service.detector.get_supported_classes.return_value = custom_classes
    ms.bind_site_camera_service(mock_service)

    # 1. Supported custom class present in detections
    res_pipe = query_router.route("is pipe visible on camera 2?")
    assert res_pipe.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_pipe.facts["unsupported_class"] is None
    assert res_pipe.facts["target_class"] == "pipe"
    summary_pipe = query_router.summarise(res_pipe)
    assert "1 pipe" in summary_pipe.lower()

    # 2. Supported custom class NOT present in detections
    res_bracket = query_router.route("is bracket detected on camera 2?")
    assert res_bracket.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_bracket.facts["unsupported_class"] is None
    assert res_bracket.facts["target_class"] == "bracket"
    summary_bracket = query_router.summarise(res_bracket)
    assert "no brackets are currently detected" in summary_bracket.lower() or "no bracket" in summary_bracket.lower()

    # 3. Unsupported class on the custom model (e.g. gear)
    res_gear = query_router.route("is gear detected on camera 2?")
    assert res_gear.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_gear.facts["unsupported_class"] == "gear"
    summary_gear = query_router.summarise(res_gear)
    assert "does not support detecting 'gear'" in summary_gear.lower()

    # 4. Multi-word custom class matching (hydraulic valve -> hydraulic_valve)
    res_valve = query_router.route("is hydraulic valve visible on camera 2?")
    assert res_valve.intent == query_router.Intent.GET_SITE_DETECTIONS
    assert res_valve.facts["target_class"] == "hydraulic_valve"
    assert res_valve.facts["unsupported_class"] is None


def test_detected_entity_schema_compatibility():
    """
    Verify DetectedEntity returns class, confidence, bbox, width/height, mask and track_id
    as required by user specification, while retaining legacy aliases.
    """
    from agent.people_detector import DetectedEntity

    ent = DetectedEntity(
        class_id=39,
        class_name="bottle",
        confidence=0.88,
        box=[100, 150, 60, 120],
        width_px=60,
        height_px=120,
        mask=None,
        track_id=4,
    )

    # Property accessors
    assert ent.class_name == "bottle"
    assert ent.confidence == 0.88
    assert ent.bbox == [100, 150, 60, 120]
    assert ent.width == 60
    assert ent.height == 120
    assert ent.mask is None
    assert ent.track_id == 4

    # Dict representation
    d = ent.to_dict()
    assert d["class"] == "bottle"
    assert d["class_name"] == "bottle"
    assert d["confidence"] == 0.88
    assert d["bbox"] == [100, 150, 60, 120]
    assert d["box"] == [100, 150, 60, 120]
    assert d["width"] == 60
    assert d["width_px"] == 60
    assert d["height"] == 120
    assert d["height_px"] == 120
    assert d["mask"] is None
    assert d["track_id"] == 4


def test_overlapping_duplicate_person_suppression_different_track_ids():
    """
    Verify that when ByteTrack/YOLO assigns different track IDs to overlapping detections
    of the SAME physical person (e.g. full body + upper body/torso), the duplicate detection
    is suppressed, preventing multiple active person detections/track IDs for the same person,
    while non-person objects (bottle) remain unaffected.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()
    mock_model = MagicMock()
    detector._model = mock_model

    # Box 1: Full body person, conf=0.88, track_id=1
    box_full_person = MagicMock()
    box_full_person.xyxy = [np.array([100, 100, 300, 600])]
    box_full_person.conf = [0.88]
    box_full_person.cls = [0]
    box_full_person.id = [1]

    # Box 2: Upper body/torso duplicate, conf=0.72, track_id=2 (different track ID!)
    box_torso_person = MagicMock()
    box_torso_person.xyxy = [np.array([120, 100, 280, 350])]
    box_torso_person.conf = [0.72]
    box_torso_person.cls = [0]
    box_torso_person.id = [2]

    # Box 3: Handheld object (bottle), conf=0.85, track_id=3
    box_bottle = MagicMock()
    box_bottle.xyxy = [np.array([140, 250, 180, 350])]
    box_bottle.conf = [0.85]
    box_bottle.cls = [39]
    box_bottle.id = [3]

    mock_result = MagicMock()
    mock_result.boxes = [box_full_person, box_torso_person, box_bottle]
    mock_result.names = {0: "person", 39: "bottle"}
    mock_result.masks = None
    mock_model.track.return_value = [mock_result]

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy_frame)

    # 1. Exactly 1 person counted (not 2!)
    assert res.people_count == 1

    # 2. Exactly 2 entities in total: 1 deduplicated person + 1 bottle
    assert len(res.entities) == 2
    entity_classes = [e.class_name for e in res.entities]
    assert "person" in entity_classes
    assert "bottle" in entity_classes

    # 3. Only 1 person track ID exists
    person_entities = [e for e in res.entities if e.class_name == "person"]
    assert len(person_entities) == 1
    assert person_entities[0].box == [100, 100, 200, 500]  # Full body box preserved


def test_two_real_people_preserved_with_partial_overlap():
    """
    Verify that 2+ real people standing close together with partial overlap
    are both preserved and correctly counted.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()
    mock_model = MagicMock()
    detector._model = mock_model

    # Worker 1: [100, 100, 250, 600] -> w=150, h=500
    box_p1 = MagicMock()
    box_p1.xyxy = [np.array([100, 100, 250, 600])]
    box_p1.conf = [0.91]
    box_p1.cls = [0]
    box_p1.id = [1]

    # Worker 2: [200, 100, 350, 600] -> w=150, h=500 (shoulder overlap with Worker 1: 50px)
    box_p2 = MagicMock()
    box_p2.xyxy = [np.array([200, 100, 350, 600])]
    box_p2.conf = [0.89]
    box_p2.cls = [0]
    box_p2.id = [2]

    mock_result = MagicMock()
    mock_result.boxes = [box_p1, box_p2]
    mock_result.names = {0: "person"}
    mock_result.masks = None
    mock_model.track.return_value = [mock_result]

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy_frame)

    # Both real people must be preserved and counted
    assert res.people_count == 2
    assert len(res.entities) == 2
    track_ids = {e.track_id for e in res.entities}
    assert track_ids == {1, 2}


def test_zero_shot_eyewear_detection_separate_from_person():
    """
    Verify that YOLO-World zero-shot eyewear detection generates a separate 'eyewear'
    entity from 'person' without merging, without heuristics/landmarks, and without
    altering the people count.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector, draw_people_overlay

    detector = PeopleDetector()

    # Mock primary segmentation model (1 person)
    mock_seg = MagicMock()
    box_person = MagicMock()
    box_person.xyxy = [np.array([100, 100, 400, 600])]
    box_person.conf = [0.90]
    box_person.cls = [0]
    box_person.id = [1]
    res_seg = MagicMock()
    res_seg.boxes = [box_person]
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Mock YOLO-World eyewear model (1 eyewear/sunglasses)
    mock_eye = MagicMock()
    box_glasses = MagicMock()
    box_glasses.xyxy = [np.array([180, 150, 320, 200])]
    box_glasses.conf = [0.85]
    box_glasses.cls = [2]  # sunglasses in vocab
    res_eye = MagicMock()
    res_eye.boxes = [box_glasses]
    res_eye.names = {0: "eyewear", 1: "glasses", 2: "sunglasses", 3: "safety glasses", 4: "goggles", 5: "person"}
    mock_eye.predict.return_value = [res_eye]
    detector._eyewear_model = mock_eye

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Frame 1: first detection (not shown yet; requires consecutive frames)
    res_f1 = detector.detect_frame(dummy_frame)
    assert len(res_f1.entities) == 1
    assert res_f1.entities[0].class_name == "person"

    # Frame 2: consecutive detection confirmed and visible
    res = detector.detect_frame(dummy_frame)

    # 1. People count strictly reflects person (count = 1)
    assert res.people_count == 1

    # 2. Exactly 2 separate entities: person and eyewear
    assert len(res.entities) == 2
    entity_classes = [e.class_name for e in res.entities]
    assert entity_classes == ["person", "eyewear"]

    # 3. Eyewear entity properties
    eyewear_ent = next(e for e in res.entities if e.class_name == "eyewear")
    assert eyewear_ent.class_name == "eyewear"
    assert eyewear_ent.confidence == 0.85
    assert eyewear_ent.box == [180, 150, 140, 50]
    assert eyewear_ent.width_px == 140
    assert eyewear_ent.height_px == 50
    assert eyewear_ent.relation is None
    assert eyewear_ent.person_track_id is None

    # 4. Rendering does not throw and includes eyewear in overlay
    rendered = draw_people_overlay(dummy_frame, res)
    assert rendered.shape == dummy_frame.shape
    assert np.any(rendered[:, :, 0] > 200)


def test_eyewear_temporal_stability_and_missed_frame_tolerance():
    """
    Verify simple temporal stability:
    1. Eyewear must be detected in consecutive frames before showing.
    2. Tolerates 1 missed frame without dropping or flickering.
    3. Drops eyewear after 2 consecutive missed frames.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector(eyewear_consecutive_frames=2, eyewear_max_missed_frames=1)

    mock_seg = MagicMock()
    box_person = MagicMock()
    box_person.xyxy = [np.array([100, 100, 400, 600])]
    box_person.conf = [0.90]
    box_person.cls = [0]
    box_person.id = [1]
    res_seg = MagicMock()
    res_seg.boxes = [box_person]
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    mock_eye = MagicMock()
    box_glasses = MagicMock()
    box_glasses.xyxy = [np.array([180, 150, 320, 200])]
    box_glasses.conf = [0.85]
    box_glasses.cls = [0]
    res_eye = MagicMock()
    res_eye.boxes = [box_glasses]
    res_eye.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_eye]
    detector._eyewear_model = mock_eye

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Frame 1: Detection 1 -> Not shown yet (hits = 1 < 2)
    f1 = detector.detect_frame(dummy)
    assert [e.class_name for e in f1.entities] == ["person"]

    # Frame 2: Detection 2 -> Consecutive hits reached (hits = 2 >= 2) -> Shown!
    f2 = detector.detect_frame(dummy)
    assert [e.class_name for e in f2.entities] == ["person", "eyewear"]
    eye_f2 = next(e for e in f2.entities if e.class_name == "eyewear")
    assert eye_f2.box == [180, 150, 140, 50]

    # Frame 3: Missed frame from model! (0 raw eyewear detections)
    res_empty = MagicMock()
    res_empty.boxes = []
    res_empty.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_empty]

    # Frame 3: Tolerates 1 missed frame -> Eyewear box preserved, no flickering!
    f3 = detector.detect_frame(dummy)
    assert [e.class_name for e in f3.entities] == ["person", "eyewear"]
    eye_f3 = next(e for e in f3.entities if e.class_name == "eyewear")
    assert eye_f3.box == [180, 150, 140, 50]

    # Frame 4: Second consecutive missed frame -> Exceeds max_missed_frames (1) -> Dropped
    f4 = detector.detect_frame(dummy)
    assert [e.class_name for e in f4.entities] == ["person"]


def test_eyewear_low_confidence_filtering():
    """
    Verify that low-confidence eyewear detections below threshold (default 0.50)
    are filtered out and not shown.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector(eyewear_confidence_threshold=0.50, eyewear_consecutive_frames=1)

    mock_seg = MagicMock()
    res_seg = MagicMock()
    res_seg.boxes = []
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Low-confidence false detection (0.35 < 0.50 threshold)
    mock_eye = MagicMock()
    box_low = MagicMock()
    box_low.xyxy = [np.array([100, 100, 200, 150])]
    box_low.conf = [0.35]
    box_low.cls = [0]
    res_eye_low = MagicMock()
    res_eye_low.boxes = [box_low]
    res_eye_low.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_eye_low]
    detector._eyewear_model = mock_eye

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    res_low = detector.detect_frame(dummy)
    assert len(res_low.entities) == 0

    # High-confidence valid detection (0.75 >= 0.50 threshold)
    box_hi = MagicMock()
    box_hi.xyxy = [np.array([100, 100, 200, 150])]
    box_hi.conf = [0.75]
    box_hi.cls = [0]
    res_eye_hi = MagicMock()
    res_eye_hi.boxes = [box_hi]
    res_eye_hi.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_eye_hi]

    res_hi = detector.detect_frame(dummy)
    assert len(res_hi.entities) == 1
    assert res_hi.entities[0].class_name == "eyewear"
    assert res_hi.entities[0].confidence == 0.75


def test_eyewear_single_frame_false_positive_suppressed():
    """
    Verify that an isolated single-frame false positive detection is never shown
    and is immediately discarded on the next frame.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector(eyewear_consecutive_frames=2)

    mock_seg = MagicMock()
    res_seg = MagicMock()
    res_seg.boxes = []
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Single-frame spike
    mock_eye = MagicMock()
    box_spike = MagicMock()
    box_spike.xyxy = [np.array([300, 200, 420, 250])]
    box_spike.conf = [0.80]
    box_spike.cls = [0]
    res_spike = MagicMock()
    res_spike.boxes = [box_spike]
    res_spike.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_spike]
    detector._eyewear_model = mock_eye

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Frame 1: Spike appears -> suppressed (not shown)
    f1 = detector.detect_frame(dummy)
    assert len(f1.entities) == 0

    # Frame 2: Spike vanished -> raw detections empty
    res_empty = MagicMock()
    res_empty.boxes = []
    res_empty.names = {0: "eyewear"}
    mock_eye.predict.return_value = [res_empty]

    # Frame 2: Track was unconfirmed, so dropped immediately
    f2 = detector.detect_frame(dummy)
    assert len(f2.entities) == 0
    assert len(detector._eyewear_tracks) == 0


def test_fire_detection_without_lighter_multidistance():
    """
    Scenario: Fire/flame is detected without any lighter present.
    Verifies detection across near, medium, and far distances.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()

    # Mock base segmentation model (0 people)
    mock_seg = MagicMock()
    res_seg = MagicMock()
    res_seg.boxes = []
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Mock YOLO-World predictions for near, medium, far fire boxes
    # Far: small box [100, 100, 15, 20]
    # Medium: medium box [300, 200, 60, 80]
    # Near: large box [500, 150, 180, 240]
    mock_world = MagicMock()
    b_far = MagicMock()
    b_far.xyxy = [np.array([100, 100, 115, 120])]
    b_far.conf = [0.65]
    b_far.cls = [0]

    b_med = MagicMock()
    b_med.xyxy = [np.array([300, 200, 360, 280])]
    b_med.conf = [0.78]
    b_med.cls = [1]

    b_near = MagicMock()
    b_near.xyxy = [np.array([500, 150, 680, 390])]
    b_near.conf = [0.91]
    b_near.cls = [0]

    res_world = MagicMock()
    res_world.boxes = [b_far, b_med, b_near]
    res_world.names = {0: "fire", 1: "flame"}
    mock_world.predict.return_value = [res_world]
    detector._eyewear_model = mock_world

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy)

    fire_entities = [e for e in res.entities if e.class_name == "fire"]
    assert len(fire_entities) == 3
    assert all(e.class_name == "fire" for e in fire_entities)
    assert fire_entities[0].box == [100, 100, 15, 20]  # Far
    assert fire_entities[1].box == [300, 200, 60, 80]  # Medium
    assert fire_entities[2].box == [500, 150, 180, 240]  # Near
    # No lighter detected
    assert not any(e.class_name == "lighter" for e in res.entities)


def test_lighter_detection_without_flame_multidistance():
    """
    Scenario: Lighter is detected without any fire/flame present.
    Verifies detection across near, medium, and far distances.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector

    detector = PeopleDetector()

    mock_seg = MagicMock()
    res_seg = MagicMock()
    res_seg.boxes = []
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Far lighter [50, 50, 12, 25], Medium lighter [200, 200, 40, 80], Near lighter [400, 200, 100, 200]
    mock_world = MagicMock()
    b_far = MagicMock()
    b_far.xyxy = [np.array([50, 50, 62, 75])]
    b_far.conf = [0.60]
    b_far.cls = [0]

    b_med = MagicMock()
    b_med.xyxy = [np.array([200, 200, 240, 280])]
    b_med.conf = [0.72]
    b_med.cls = [0]

    b_near = MagicMock()
    b_near.xyxy = [np.array([400, 200, 500, 400])]
    b_near.conf = [0.85]
    b_near.cls = [0]

    res_world = MagicMock()
    res_world.boxes = [b_far, b_med, b_near]
    res_world.names = {0: "lighter"}
    mock_world.predict.return_value = [res_world]
    detector._eyewear_model = mock_world

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy)

    lighter_entities = [e for e in res.entities if e.class_name == "lighter"]
    assert len(lighter_entities) == 3
    assert lighter_entities[0].box == [50, 50, 12, 25]  # Far
    assert lighter_entities[1].box == [200, 200, 40, 80]  # Medium
    assert lighter_entities[2].box == [400, 200, 100, 200]  # Near
    # No fire detected
    assert not any(e.class_name == "fire" for e in res.entities)


def test_lighter_with_flame_simultaneous_detection():
    """
    Scenario: Lighter with flame is detected simultaneously.
    Verifies that lighter and fire are detected as separate bounding boxes,
    neither suppresses the other, and each retains its label and confidence.
    """
    from unittest.mock import MagicMock
    import numpy as np
    from agent.people_detector import PeopleDetector, draw_people_overlay

    detector = PeopleDetector()

    mock_seg = MagicMock()
    res_seg = MagicMock()
    res_seg.boxes = []
    res_seg.names = {0: "person"}
    res_seg.masks = None
    mock_seg.track.return_value = [res_seg]
    detector._model = mock_seg

    # Lighter body [300, 250, 50, 100], Flame on top [310, 200, 30, 50]
    mock_world = MagicMock()
    b_lighter = MagicMock()
    b_lighter.xyxy = [np.array([300, 250, 350, 350])]
    b_lighter.conf = [0.82]
    b_lighter.cls = [0]

    b_flame = MagicMock()
    b_flame.xyxy = [np.array([310, 200, 340, 250])]
    b_flame.conf = [0.79]
    b_flame.cls = [1]

    res_world = MagicMock()
    res_world.boxes = [b_lighter, b_flame]
    res_world.names = {0: "lighter", 1: "flame"}
    mock_world.predict.return_value = [res_world]
    detector._eyewear_model = mock_world

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_frame(dummy)

    # Both must be present
    classes = [e.class_name for e in res.entities]
    assert "lighter" in classes
    assert "fire" in classes

    ent_lighter = next(e for e in res.entities if e.class_name == "lighter")
    ent_fire = next(e for e in res.entities if e.class_name == "fire")

    assert ent_lighter.box == [300, 250, 50, 100]
    assert ent_lighter.confidence == 0.82
    assert ent_fire.box == [310, 200, 30, 50]
    assert ent_fire.confidence == 0.79

    # Verify overlay renders correctly with both boxes
    overlay = draw_people_overlay(dummy, res)
    assert overlay.shape == dummy.shape





