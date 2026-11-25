import re
import logging
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pydantic import BaseModel
from config import (
    PPTS_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from models import SlideData

logger = logging.getLogger(__name__)

SMARTART_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
FOOTER_PLACEHOLDER_TYPES = {
    PP_PLACEHOLDER.FOOTER,
    PP_PLACEHOLDER.SLIDE_NUMBER,
    PP_PLACEHOLDER.DATE,
}
FOOTER_CONTENT_PATTERN = re.compile(
    r"^(?:"
    r"copyright(?:\s*[©(c)])?(?:\s+\d{4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|all rights reserved"
    r"|infineon technologies(?:\s+ag)?"
    r"|restricted"
    r"|confidential"
    r"|internal use only"
    r"|public"
    r")$",
    re.IGNORECASE
)

CONTACT_PATTERN = re.compile(
    r"(?:contact|owner|responsible)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE,
)


class Slide1Metadata(BaseModel):
    module_title: str = ""
    series_label: str = ""
    contact: str = ""


# ---------------------------------------------------------------------------
# Shape detection helpers
# ---------------------------------------------------------------------------

def _is_title_shape(shape) -> bool:
    return _get_placeholder_type(shape) == PP_PLACEHOLDER.TITLE


def _is_object_placeholder(shape) -> bool:
    return _get_placeholder_type(shape) == PP_PLACEHOLDER.OBJECT


def _is_footer_shape(shape) -> bool:
    placeholder_type = _get_placeholder_type(shape)
    return placeholder_type in FOOTER_PLACEHOLDER_TYPES


def _get_placeholder_type(shape):
    if not getattr(shape, "is_placeholder", False):
        return None
    try:
        return shape.placeholder_format.type
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug("Could not read placeholder type: %s", exc)
        return None


def _is_footer_position(shape, slide_height: int) -> bool:  # footer if it starts in the bottom 10% of the slide.
    top = getattr(shape, "top", None)
    if not isinstance(top, (int, float)):
        logger.debug("Shape has no numeric 'top'; skipping footer-position check")
        return False
    return top > slide_height * 0.90


def _is_footer_content(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    return all(
        FOOTER_CONTENT_PATTERN.match(line)
        or line.isdigit()
        or len(line) <= 2
        for line in lines
    )


def _contains_smartart(shape) -> bool:
    if shape.shape_type != MSO_SHAPE_TYPE.GROUP:
        return False
    element = getattr(shape, "_element", None)
    xml_str = getattr(element, "xml", "")
    return SMARTART_NAMESPACE in xml_str


# ---------------------------------------------------------------------------
# Per-slide extraction helpers
# ---------------------------------------------------------------------------

def _get_shape_inventory(shape) -> list[str]:
    labels = []
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        labels.append("image_placeholder")
    elif _is_object_placeholder(shape):
        # Many standard "content" placeholders are OBJECT type but contain plain text.
        has_text = getattr(shape, "has_text_frame", False) and shape.has_text_frame
        text = shape.text_frame.text.strip() if has_text else ""
        if text:
            labels.append("text_box")
        else:
            labels.append("image_placeholder")
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        if _contains_smartart(shape):
            labels.append("smart_art")
        else:
            labels.append("grouped_shapes")
    elif getattr(shape, "has_chart", False) and shape.has_chart:
        labels.append("chart")
    elif (getattr(shape, "has_text_frame", False)
          and shape.has_text_frame
          and not _is_title_shape(shape)):
        labels.append("text_box")
    else:
        labels.append("other")
    return labels


def _get_slide_shape_inventory(slide, slide_height: int) -> list[str]:
    labels = []
    for shape in slide.shapes:
        if _is_footer_shape(shape):
            continue
        if _is_footer_position(shape, slide_height):
            continue
        labels.extend(_get_shape_inventory(shape))
    seen = set()
    result = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _get_slide_title(slide) -> str:
    for shape in slide.shapes:
        if _is_title_shape(shape):
            return (
                shape.text_frame.text.strip()
                if shape.has_text_frame else ""
            )
    return ""


def _get_slide_body_text(slide, slide_height: int) -> str:
    parts = []
    for shape in slide.shapes:
        if (not getattr(shape, "has_text_frame", False)
                or not shape.has_text_frame):
            continue
        if _is_title_shape(shape):
            continue
        if _is_footer_shape(shape):
            continue
        if _is_footer_position(shape, slide_height):
            continue
        text = shape.text_frame.text.strip()
        if not text:
            continue
        if _is_footer_content(text):
            continue
        parts.append(text)
    return "\n".join(parts)


def _get_speaker_notes(slide) -> str:
    if (slide.has_notes_slide
            and slide.notes_slide
            and slide.notes_slide.notes_text_frame):
        return slide.notes_slide.notes_text_frame.text.strip()
    return ""


# ---------------------------------------------------------------------------
# Aspect ratio — compute correct render height per module
# ---------------------------------------------------------------------------

def _compute_render_height(prs: Presentation) -> int:
    """
    Compute the correct render height for this specific PPT file
    based on its actual slide dimensions.

    Uses IMAGE_WIDTH (max. possible in standard tier, from config) as the fixed width.
    Derives height from the actual slide aspect ratio so images
    are never stretched or letterboxed regardless of whether
    slides are 16:9, 4:3, or any other ratio.

    Prints a note if the computed height differs from IMAGE_HEIGHT
    in config so the discrepancy is visible in pipeline output.
    """

    actual_ratio  = prs.slide_width / prs.slide_height
    render_height = int(IMAGE_WIDTH / actual_ratio)

    expected_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    if abs(actual_ratio - expected_ratio) > 0.01:
        print(
            f"  Note: slide aspect ratio {actual_ratio:.4f} "
            f"— using computed IMAGE_HEIGHT={render_height} "
            f"(config has {IMAGE_HEIGHT})"
        )

    return render_height


# ---------------------------------------------------------------------------
# Deterministic metadata extraction from slide 1
# ---------------------------------------------------------------------------

def _clean_candidate_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" -:\t")
        if not line:
            continue
        if _is_footer_content(line):
            continue
        if line.lower().startswith(("slide ", "agenda", "outline")):
            continue
        lines.append(line)
    return lines


def _extract_contact(lines: list[str]) -> str:
    for line in lines:
        match = CONTACT_PATTERN.search(line)
        if not match:
            continue
        candidate = match.group(1).strip()
        candidate = candidate.split("<", 1)[0].strip()
        candidate = re.sub(r"\b\S+@\S+\b", "", candidate).strip()
        return candidate
    return ""


def extract_slide1_metadata(slide1: SlideData) -> Slide1Metadata:
    """
    Deterministically extract module title, series label, and contact
    from slide 1 text/notes.

    Heuristics:
    - module_title: explicit slide title if available; else longest
      informative body-text line.
    - series_label: nearest short line before module_title in body text.
    - contact: line containing contact/owner/responsible marker.
    """
    body_lines = _clean_candidate_lines(slide1.body_text)
    notes_lines = _clean_candidate_lines(slide1.speaker_notes)
    all_lines = body_lines + notes_lines

    module_title = slide1.title.strip()
    if not module_title:
        # Prefer richer lines for a specific module title.
        module_title = max(
            body_lines,
            key=lambda line: (len(line.split()), len(line)),
            default="",
        )

    series_label = ""
    if module_title and module_title in body_lines:
        title_idx = body_lines.index(module_title)
        for idx in range(title_idx - 1, -1, -1):
            candidate = body_lines[idx]
            if candidate == module_title:
                continue
            if 1 <= len(candidate.split()) <= 6 and len(candidate) <= 60:
                series_label = candidate
                break

    contact = _extract_contact(all_lines)

    if contact and module_title and contact in module_title:
        contact = ""

    return Slide1Metadata(
        module_title=module_title,
        series_label=series_label,
        contact=contact,
    )


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_pptx(module_id: str) -> tuple[list[SlideData], dict]:
    """
    Parse a PPTX file into SlideData objects and partial metadata.

    Returns:
        (slides, partial_metadata)
        partial_metadata keys:
          date          from PPT core properties
          render_height computed from actual slide aspect ratio
                                                used by slide_categorizer_step2.py instead of
                        fixed IMAGE_HEIGHT from config
    """
    pptx_path = Path(PPTS_DIR) / f"{module_id}.pptx"
    if not pptx_path.exists():
        raise FileNotFoundError(
            f"PPT file not found: {PPTS_DIR}/{module_id}.pptx"
        )

    prs = Presentation(str(pptx_path))

    # compute correct render height for this module's aspect ratio
    render_height = _compute_render_height(prs)

    # slide_height in EMUs used for footer position detection
    slide_height = prs.slide_height

    slides = []
    for idx, slide in enumerate(prs.slides, start=1):
        slides.append(
            SlideData(
                module_id=module_id,
                slide_number=idx,
                title=_get_slide_title(slide),
                body_text=_get_slide_body_text(slide, slide_height),
                speaker_notes=_get_speaker_notes(slide),
                shape_inventory=_get_slide_shape_inventory(
                    slide, slide_height
                ),
            )
        )

    partial_metadata = {
        "date": (
            str(prs.core_properties.created.date())
            if prs.core_properties.created else ""
        ),
        "render_height": render_height,
        # title, contact, series_label added by extract_slide1_metadata()
        # track, subtrack added by load_metadata() in main.py
    }

    return slides, partial_metadata


