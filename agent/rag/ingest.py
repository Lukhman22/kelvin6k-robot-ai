"""
Document ingestion: load -> chunk -> (embed) -> store.

Supported: .txt, .md, .pdf (needs `pypdf`). Page numbers are preserved for PDFs
and Markdown headings become the `section` metadata, so answers can cite where
they came from.
"""

import json
import os
import re
import time
from typing import List, Optional, Tuple

from agent import config
from agent.logging_setup import get_logger
from agent.rag.embeddings import get_embedder
from agent.rag.store import Chunk, DocumentStore

log = get_logger("RAG")

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
JSONL_EXTENSIONS = {".jsonl"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS | JSONL_EXTENSIONS

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _doc_type(filename: str) -> str:
    """Coarse classification from the name, used as retrieval metadata."""
    lower = filename.lower()
    for keyword, label in (
        ("manual", "manual"),
        ("sop", "sop"),
        ("troubleshoot", "troubleshooting"),
        ("maintenance", "maintenance"),
        ("report", "report"),
        ("spec", "specification"),
        ("rule", "operational_rules"),
        ("operational", "operational_rules"),
        ("operator", "operator_knowledge"),
        ("manager", "manager_notes"),
        ("handwritten", "handwritten_notes"),
        ("fault", "fault_records"),
    ):
        if keyword in lower:
            return label
    return "document"


def load_text_file(path: str) -> List[Tuple[str, Optional[int]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [(fh.read(), None)]


def load_pdf(path: str) -> List[Tuple[str, Optional[int]]]:
    """Returns [(page_text, page_number)]. Requires pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion needs pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning("Could not extract page %d of %s: %s", number, path, exc)
            text = ""
        if text.strip():
            pages.append((text, number))
    return pages


def load_docx(path: str) -> List[Tuple[str, Optional[int]]]:
    """Extracts text and tables from DOCX and formats with markdown headings."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX ingestion needs python-docx. pip install python-docx") from exc
        
    doc = Document(path)
    text_blocks = []
    skip_section = False
    
    # Merge paragraphs and tables with markdown headings.
    # Exclude raw question lists (e.g. Section 6 'Agent Queries Supported')
    # so they do not pollute retrieval with answerless question phrases.
    for p in doc.paragraphs:
        if p.style.name.startswith('Heading'):
            level = p.style.name.replace('Heading', '').strip()
            level = int(level) if level.isdigit() else 1
            heading_text = p.text.strip()
            if "agent queries supported" in heading_text.lower() or "queries supported" in heading_text.lower():
                skip_section = True
                continue
            else:
                skip_section = False
            text_blocks.append(f"{'#' * level} {heading_text}")
        elif p.text.strip():
            if not skip_section:
                text_blocks.append(p.text.strip())
            
    for table in doc.tables:
        text_blocks.append("--- TABLE ---")
        for row in table.rows:
            text_blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
            
    full_text = "\n\n".join(text_blocks)
    return [(full_text, None)]


CONFLICT_ISSUES = {
    "K6K-CLEAN-001",
    "K6K-PAUSE-001",
    "K6K-LAYER-001",
    "K6K-CURE-001",
    "K6K-SPEED-001",
}

OPERATOR_QUESTION_MAP = {
    "K6K-CLEAN-001": {
        "english": "How frequently should the pump and hose be cleaned? What is the cleaning interval for pump and delivery hose? When to clean pump and hose? How often should the pump be washed out?",
        "hinglish": "Pump aur hose ki cleaning kitni der mein karni chahiye? Pump aur delivery hose kab clean karna chahiye? Cleaning interval kitna hai? Har wall ke baad cleaning zaroori hai kya?",
    },
    "K6K-HOSE-001": {
        "english": "How should the delivery hose between pump and hopper be positioned? Delivery hose positioning and routing. How to connect delivery hose without bending?",
        "hinglish": "Delivery hose ko kaise lagana chahiye? Pump se hopper tak hose ka connection kaisa hona chahiye? Hose bend hona chahiye ya nahi? Hose loose ya tight kaisa hona chahiye?",
    },
    "K6K-FLOW-001": {
        "english": "What should an operator do when concrete material flow suddenly drops or stops? Material flow sudden drop or interruption. Blockage in pump. Particle clog clearing. Can I hit the pump with a hammer? Can I pulse feedrate?",
        "hinglish": "Material flow achanak kam ho gaya, kya check karna chahiye? Flow achanak ruk gaya ya band ho gaya toh kya karein? Pump blockage kaise clear karein? Kya pump par hammer marna chahiye? Feedrate badhana chahiye kya?",
    },
    "K6K-SPEED-001": {
        "english": "What feedrates were reported for straight walls and complex curved paths? Feedrate guidance for straight versus curved paths. What speed for straight wall and curved path? Feedrate 100 120 150 170.",
        "hinglish": "Curved wall ke liye operator ne kaunsa feedrate range report kiya? Straight wall aur complex curved path ke liye feedrate kitna hona chahiye? Arm jerking rokne ke liye speed kitni karni hai?",
    },
    "K6K-LAYER-001": {
        "english": "How is layer bead thickness measured and corrected? Manual layer thickness measurement. Layer thickness above 45 mm. Is 40 mm or 50 mm correct?",
        "hinglish": "Layer 45 mm se zyada ho raha hai source mein kya observation hai? Layer thickness kaise naapna hai? Bead thickness check kaise karein? Tape measure se layer kaise check karein?",
    },
    "K6K-CURE-001": {
        "english": "How quickly must lower concrete layers harden to support subsequent layers? Layer green strength and wall slumping collapse. When do lower layers gain green strength? Wall slump risk.",
        "hinglish": "Lower layers ko kitni der mein hard hona chahiye? Wall slump ya collapse hone ka kya reason hai? Green strength kitne layer mein aani chahiye? Foundation soft hone par kya karein?",
    },
    "K6K-PAUSE-001": {
        "english": "What happens if printing is paused while accelerator mixed material remains in the hopper? Material stagnation during print pause. Does concrete harden in hopper after 5 minutes? Pause time extrusion.",
        "hinglish": "Pause ke dauran hopper mein material hard hone ke baare mein kya report hai? Print pause hone par accelerator mixed concrete kitni der mein set ho jata hai? 5 minute rule pause mein kya hai?",
    },
    "K6K-VISION-001": {
        "english": "Have shadows or shop lighting caused the camera system to stop the machine incorrectly? Camera false stops caused by shadows or lighting. Did lighting cause false stop?",
        "hinglish": "Kya camera lighting ki wajah se false stop report hua tha? Shadows ya roshni se camera machine ko rokta hai kya? Camera false positive lighting problem.",
    },
    "K6K-ENV-001": {
        "english": "Is there historical environmental and process data that could support analysis? Historical environmental and machine parameter notebook. Where is temperature and humidity history?",
        "hinglish": "Kya temperature aur humidity ka historical data available hai? Operator notebook mein kya environmental data note kiya gaya hai? Temperature humidity pump speed notebook.",
    },
    "K6-FAULT-E136.1": {
        "english": "What was reported as the cause of error E136.1? Error code E136.1 absolute encoder cable damage. How to resolve fault E136.1?",
        "hinglish": "Error code E136.1 ka kya reason hai? E136.1 alarm aane par kya check karna hai? Absolute encoder cable damage E136.1.",
    },
    "K6-FAULT-E137.1": {
        "english": "What was reported as the cause of error E137.1? Error code E137.1 robot servo position mismatch. Should encoder values be reset for E137.1?",
        "hinglish": "E137.1 ka reported explanation kya hai aur kya verify karna baaki hai? Error code E137.1 alarm kyun aata hai? Servo drive encoder reset E137.1.",
    },
    "K6-Z-001": {
        "english": "What was reported when the Z-axis could not move downward? Z-axis downward motion blocked by travel limit H0D11. Can parameter H0D11 be changed to jog Z-axis?",
        "hinglish": "H0D11 ke baare mein notes kya kehte hain? Kya yeh approved procedure hai? Z-axis neeche nahi ja raha downward motion lock ho gaya. Parameter H0D11 se manual jog.",
    },
    "K6-QUALITY-001": {
        "english": "What should be checked during the first few layers if concrete is not hardening sufficiently? Early layer accelerator check 50% hardness. What does 50 percent hardness mean?",
        "hinglish": "Pehle layers mein concrete hard nahi ho raha toh kya check karein? 50 percent hardness aur accelerator flow check. Early layer hardening check.",
    },
    "K6-SAFETY-001": {
        "english": "What precautions were reported to reduce wall damage or robot collision? Prevent wall contact and collision during printing. Can someone touch the wall during printing?",
        "hinglish": "Printing ke time wall ko touch karna ya collision se bachne ke liye kya safety rules hain? Robot collision se kaise bachein? Printing ke dauran wall ko chhu sakte hain kya?",
    },
}


def format_operator_record_to_markdown(record: dict) -> str:
    issue_id = record.get("issue_id", "K6K-UNKNOWN")
    title = record.get("title", "")
    category = record.get("category", "")
    status = record.get("status", "")
    source_type = record.get("source_type", "")

    # Problem / question mapping across schemas
    question = (
        record.get("symptoms_or_question")
        or record.get("reported_issue")
        or (record.get("question_variants")[0] if record.get("question_variants") else "")
        or record.get("title", "")
    )

    # Operator / manager instruction or action across schemas
    operator_ans = (
        record.get("operator_reported_answer")
        or record.get("reported_instruction")
        or record.get("reported_action")
        or record.get("reported_solution")
        or ""
    )

    # Causes mapping across schemas
    possible_causes = record.get("possible_causes", [])
    if not possible_causes and record.get("possible_cause_reported"):
        possible_causes = [record.get("possible_cause_reported")]

    confirmed_root = record.get("confirmed_root_cause")
    solution_type = record.get("solution_type", "")
    verification_req = record.get("verification_required", "")

    # Gaps / unknowns / missing info mapping
    conflicts = (
        record.get("conflicts_or_gaps")
        or record.get("missing_information")
        or record.get("unknowns")
        or []
    )
    safety = record.get("safety_notes", "")
    keywords = record.get("retrieval_keywords", [])
    source = record.get("source", "")
    confidence = record.get("confidence", "")
    language = record.get("language", "")

    is_conflict = (issue_id in CONFLICT_ISSUES) or ("conflict" in status.lower())

    if source_type == "manager_whatsapp_note" or "manager" in status.lower():
        source_label = "Manager WhatsApp Note (6 Sep 2026)"
        if "safety_review" in status.lower():
            sol_qualifier = "MANAGER REPORTED - SAFETY REVIEW REQUIRED"
            sol_text = (
                f"Reported manager instruction: '{operator_ans}'. "
                "Never clear or disconnect a pressurized line. "
                "Do not treat as an actionable operating procedure until authorized engineering confirmation."
            )
            verif_status = f"SAFETY REVIEW REQUIRED: {verification_req}"
        elif "clarification" in status.lower():
            sol_qualifier = "MANAGER REPORTED - NEEDS CLARIFICATION"
            sol_text = (
                f"Reported manager instruction: '{operator_ans}'. "
                "The note does not specify system, substance, or approved method. Clarify before action."
            )
            verif_status = f"NEEDS CLARIFICATION: {verification_req}"
        elif "engineering_confirmation" in status.lower():
            sol_qualifier = "MANAGER REPORTED - ENGINEERING CONFIRMATION REQUIRED"
            sol_text = (
                f"Reported manager instruction: '{operator_ans}'. "
                "Confirm approved specification and interval for active wall design/job."
            )
            verif_status = f"ENGINEERING CONFIRMATION REQUIRED: {verification_req}"
        elif "needs_limits" in status.lower():
            sol_qualifier = "MANAGER REPORTED - NEEDS LIMITS"
            sol_text = (
                f"Reported manager instruction: '{operator_ans}'. "
                "Do not apply unverified cooldown methods or invent temperature thresholds."
            )
            verif_status = f"NEEDS LIMITS: {verification_req}"
        elif "handover" in status.lower():
            sol_qualifier = "MANAGER REPORTED - OPERATIONAL HANDOVER"
            sol_text = f"Reported manager instruction: '{operator_ans}'. Confirm active print plan and handover records."
            verif_status = f"HANDOVER VERIFICATION: {verification_req}"
        else:
            sol_qualifier = "MANAGER REPORTED - OPERATIONAL NOTE"
            sol_text = f"Reported manager instruction: '{operator_ans}'."
            verif_status = f"MANAGER REPORTED: {verification_req}"

    elif source_type == "handwritten_notes_transcription" or "handwritten" in status.lower() or issue_id.startswith("K6K-HW-"):
        source_label = "Handwritten Fault Notes Transcription"
        sol_qualifier = "HANDWRITTEN NOTE - UNVERIFIED / NO CONFIRMED SOLUTION"
        if operator_ans:
            sol_text = f"Reported in handwritten fault notes: {operator_ans}. Root cause and repair procedure are not confirmed."
        else:
            sol_text = (
                "No verified solution supplied in handwritten notes. "
                "Requires physical inspection, diagnostic logs, and authorized engineering diagnosis."
            )
        verif_status = f"HANDWRITTEN REPORT - UNVERIFIED: {verification_req}"

    elif solution_type == "reported_workaround_unapproved" or "safety_review" in status.lower():
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "UNAPPROVED WORKAROUND - SAFETY REVIEW REQUIRED"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "THIS IS AN UNAPPROVED WORKAROUND AND MUST NOT BE RECOMMENDED AS AN OPERATING PROCEDURE. "
            "Do not strike equipment or rapidly pulse feedrate. "
            "An authorized engineer must confirm the approved blockage-clearing and diagnostic procedure."
        )
        verif_status = f"SAFETY REVIEW REQUIRED - UNAPPROVED WORKAROUND: {verification_req}"
    elif "requires_authorization" in status.lower() or solution_type == "machine_specific_procedure_requires_authorization":
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "OPERATOR REPORTED - REQUIRES AUTHORIZATION"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "DO NOT execute this procedure without an authorized technician and manufacturer documentation. "
            "This is a machine-specific procedure requiring authorization."
        )
        verif_status = f"REQUIRES AUTHORIZATION: {verification_req}"
    elif is_conflict and "engineering_approval" in status.lower():
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "CONFLICT - OPERATOR REPORTED - ENGINEERING APPROVAL REQUIRED"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "The available documentation contains conflicting values. "
            "Engineering approval is required before changing live machine parameters."
        )
        verif_status = f"CONFLICT - ENGINEERING APPROVAL REQUIRED: {verification_req}"
    elif is_conflict:
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "CONFLICT - OPERATOR REPORTED - MUST CONFIRM"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "The available documentation contains conflicting values. "
            "Confirm the approved standard with an authorized engineer before changing settings."
        )
        verif_status = f"CONFLICT - MUST CONFIRM: {verification_req}"
    elif "engineering_approval" in status.lower():
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "OPERATOR REPORTED - ENGINEERING APPROVAL REQUIRED"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "These values require engineering approval before changing live machine parameters."
        )
        verif_status = f"ENGINEERING APPROVAL REQUIRED: {verification_req}"
    elif solution_type == "observation_only" or "observation" in status.lower():
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "OPERATOR REPORTED - OBSERVATION ONLY"
        sol_text = (
            f"The operator reported: {operator_ans} "
            "This is an operator observation, not proof that issues cannot occur. "
            "Safeguards must remain active."
        )
        verif_status = f"OPERATOR OBSERVATION: {verification_req}"
    else:
        source_label = "Operator Audio Recording (Hindi)"
        sol_qualifier = "OPERATOR REPORTED - NEEDS VALIDATION"
        sol_text = f"The operator reported: {operator_ans} Confirm with approved operating documentation."
        verif_status = f"OPERATOR REPORTED - NEEDS VALIDATION: {verification_req}"

    if possible_causes:
        cause_qualifier = "POSSIBLE - NOT CONFIRMED"
        cause_text = "; ".join(possible_causes) + ". These causes are not confirmed."
    else:
        cause_qualifier = "NOT CONFIRMED"
        cause_text = "None reported; causal diagnosis not confirmed."

    confirmed_root_text = confirmed_root if confirmed_root else "None confirmed / not supplied in source records."

    safety_text = safety
    if "safety_review" in status.lower() or "unapproved" in solution_type:
        if "safety review required" not in safety_text.lower():
            safety_text += " Safety review required: unapproved workaround."
    elif "requires_authorization" in status.lower() or "machine_specific" in solution_type:
        if "requires authorization" not in safety_text.lower() and "authorization required" not in safety_text.lower():
            safety_text += " Requires authorization before carrying out this procedure."
    elif "engineering_approval" in status.lower():
        if "engineering approval" not in safety_text.lower():
            safety_text += " Engineering approval required before changing live parameters."

    q_data = OPERATOR_QUESTION_MAP.get(issue_id, {})
    eng_q = q_data.get("english", "")
    hing_q = q_data.get("hinglish", "")

    # For manager or other notes with question_variants
    variants = record.get("question_variants") or []
    if variants:
        if not eng_q:
            eng_q = "; ".join(variants)
        else:
            eng_q += " " + "; ".join(variants)

    if not eng_q:
        eng_q = question

    effective_source = source or source_label

    md = [
        f"## Issue: {issue_id} - {title}",
        "",
        f"**Problem:** {question}",
        f"**Solution / Corrective Action ({sol_qualifier}):** {sol_text}",
        f"**Cause / Possible Causes ({cause_qualifier}):** {cause_text}",
        f"**Confirmed Root Cause:** {confirmed_root_text}",
        f"**Safety / Important Notes:** {safety_text}",
        f"**Verification Status:** {verif_status}",
    ]
    if conflicts:
        md.append(f"**Conflicts / Gaps:** {'; '.join(conflicts)}")
    if keywords:
        md.append(f"**Keywords:** {', '.join(keywords)}")
    if eng_q:
        md.append(f"**English Questions:** {eng_q}")
    if hing_q:
        md.append(f"**Hinglish Questions:** {hing_q}")
    md.append(f"**Source:** {effective_source} | Confidence: {confidence} | Language: {language}")
    md.append("")

    return "\n".join(md)


def load_jsonl(path: str) -> List[Tuple[str, Optional[int]]]:
    """Extracts operator knowledge records from JSONL and maps them to structured Markdown."""
    pages: List[Tuple[str, Optional[int]]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                log.warning("Could not parse JSONL line %d in %s: %s", idx, path, exc)
                continue
            md = format_operator_record_to_markdown(record)
            pages.append((md, idx))
    return pages


def _section_for(text: str, offset: int) -> Optional[str]:
    """Nearest preceding Markdown heading, used as the section label."""
    last = None
    for match in _HEADING_RE.finditer(text):
        if match.start() > offset:
            break
        last = match.group(2)
    return last


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Splits into overlapping windows, preferring paragraph/sentence boundaries so
    chunks stay readable. Returns [(chunk_text, start_offset)].

    Markdown headings split FIRST: a manual section such as "## Delamination"
    should be its own chunk, otherwise a query for one topic retrieves the
    neighbouring section that happens to share a window.
    """
    size = chunk_size or config.RAG_CHUNK_SIZE
    step_back = overlap if overlap is not None else config.RAG_CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    # Section-aware split on markdown headings.
    headings = list(_HEADING_RE.finditer(text))
    if len(headings) > 1:
        sections: List[Tuple[str, int]] = []
        for index, match in enumerate(headings):
            start = match.start()
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section = text[start:end].strip()
            if not section:
                continue
            if len(section) <= size:
                sections.append((section, start))
            else:
                # Long section: window it, keeping offsets absolute.
                for piece, offset in _window(section, size, step_back):
                    sections.append((piece, start + offset))
        if sections:
            return sections

    if len(text) <= size:
        return [(text, 0)]
    return _window(text, size, step_back)


def _window(text: str, size: int, step_back: int) -> List[Tuple[str, int]]:
    """Overlapping windows that prefer paragraph/sentence boundaries."""

    chunks: List[Tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph break, then a sentence end, within the tail.
            window = text[start:end]
            for boundary in ("\n\n", ". ", "\n"):
                cut = window.rfind(boundary)
                if cut > size * 0.5:
                    end = start + cut + len(boundary)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start))
        if end >= len(text):
            break
        start = max(end - step_back, start + 1)
    return chunks


def ingest_file(path: str, store: DocumentStore) -> int:
    """Ingests one file into the store (replacing prior chunks for it)."""
    filename = os.path.basename(path)
    extension = os.path.splitext(path)[1].lower()

    if extension in PDF_EXTENSIONS:
        pages = load_pdf(path)
    elif extension in TEXT_EXTENSIONS:
        pages = load_text_file(path)
    elif extension in DOCX_EXTENSIONS:
        pages = load_docx(path)
    elif extension in JSONL_EXTENSIONS:
        pages = load_jsonl(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    removed = store.remove_file(filename)
    if removed:
        log.info("Replacing %d existing chunks for %s", removed, filename)

    doc_type = _doc_type(filename)
    now = time.time()
    new_chunks: List[Chunk] = []
    issue_ids: List[str] = []

    for page_text, page_number in pages:
        chunks_to_add = (
            [(page_text, 0)]
            if extension in JSONL_EXTENSIONS
            else chunk_text(page_text)
        )
        for index, (piece, offset) in enumerate(chunks_to_add):
            match = re.search(r"##\s*Issue:\s*([A-Za-z0-9\-.]+)", piece)
            if match:
                issue_ids.append(match.group(1))
            new_chunks.append(
                Chunk(
                    text=piece,
                    filename=filename,
                    doc_type=doc_type,
                    page=page_number,
                    section=_section_for(piece, 0) or _section_for(page_text, offset),
                    chunk_index=index,
                    ingested_at=now,
                )
            )

    if issue_ids:
        store.remove_issues(issue_ids)

    store.add(new_chunks)
    log.info("Ingested %s -> %d chunks (%s)", filename, len(new_chunks), doc_type)
    return len(new_chunks)


def embed_store(store: DocumentStore) -> None:
    """Attaches embeddings when a semantic embedder is available."""
    embedder = get_embedder()
    store.embedder_name = embedder.name
    store.is_semantic = embedder.is_semantic

    if not embedder.is_semantic:
        log.info("No embedding API configured -- index will use offline lexical search.")
        return

    vectors = embedder.embed_documents(store.texts)
    if vectors is None:
        log.warning("Embedding failed -- index falls back to lexical search.")
        store.is_semantic = False
        return
    for chunk, vector in zip(store.chunks, vectors):
        chunk.embedding = vector
    log.info("Embedded %d chunks with %s", len(store.chunks), embedder.name)


def ingest_path(path: str, index_dir: Optional[str] = None) -> dict:
    """
    Ingests a file or a whole directory (recursively) and saves the index.
    Returns a summary dict.
    """
    store = DocumentStore.load(index_dir)
    files: List[str] = []

    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            for name in names:
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, name))
    elif os.path.isfile(path):
        files = [path]
    else:
        return {"ok": False, "error": f"Path not found: {path}", "files": 0, "chunks": 0}

    if not files:
        return {
            "ok": False,
            "error": f"No supported documents in {path} "
                     f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})",
            "files": 0,
            "chunks": 0,
        }

    total = 0
    ingested = []
    errors = []
    for file_path in files:
        try:
            total += ingest_file(file_path, store)
            ingested.append(os.path.basename(file_path))
        except Exception as exc:
            errors.append(f"{os.path.basename(file_path)}: {exc}")
            log.warning("Failed to ingest %s: %s", file_path, exc)

    embed_store(store)
    index_path = store.save(index_dir)

    return {
        "ok": bool(ingested),
        "files": len(ingested),
        "chunks": total,
        "total_chunks_in_index": len(store),
        "ingested": ingested,
        "errors": errors,
        "index_path": index_path,
        "semantic": store.is_semantic,
    }
