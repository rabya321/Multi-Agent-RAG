import re
import shutil
import subprocess
import time
import logging
import zipfile
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from pptx import Presentation
from pptx.exc import PackageNotFoundError

from config import (
    PPTS_DIR,
    IMAGES_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    LIBREOFFICE_PATH,
)
from models import SlideData

logger = logging.getLogger(__name__)

NAV_TITLE_MATCH_MIN = 3
RENDER_SCALE_SAFETY_FACTOR = 0.9999   # avoids ceil() pushing 1568 -> 1569
LIBREOFFICE_TIMEOUT_SECONDS = 120
LIBREOFFICE_MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Navigation detection
# ---------------------------------------------------------------------------
NAVIGATION_TITLE_KEYWORDS = [
    "agenda",
    "table of contents",
    "contents",
    "outline",
    "topics",
    "in this module",
    "what we will cover",
    "overview",
    "abbreviations",
    "introduction",
    "summary",
    "training goals",
    "learning objectives",
    "target audience",
    "target group",
    "glossary",
    "future projects",
    "next steps",
    "recap",
    "references",
    "further reading",
    "quiz",
    "quizzes",
    "knowledge check",
    "exercise",
    "exercises",
]

# Whole-title match. Allows leading bullets/numbering ("1. Agenda", "• Agenda")
# and trailing punctuation ("Agenda:"), but NOT extra words
# ("Product Overview", "Introduction to IGBTs" no longer match).
_NAV_TITLE_REGEX = re.compile(
    r"^\W*(?:\d+[.)]?\s*)?(?:"
    + "|".join(
        re.escape(kw)
        for kw in sorted(NAVIGATION_TITLE_KEYWORDS, key=len, reverse=True)
    )
    + r")\W*$",
    re.IGNORECASE,
)


def _is_navigation_by_title(title: str, slide_number: int) -> bool:
    """
    Return True if slide should be categorized as NAVIGATION.
    Slide 1 is always NAVIGATION regardless of title.
    Other slides: the whole title must be a navigation keyword.
    """
    if slide_number == 1:
        return True
    if not title:
        return False
    normalized = " ".join(title.split())  # collapse whitespace / newlines
    return _NAV_TITLE_REGEX.match(normalized) is not None


def _body_matches_other_titles(
    body_text: str,
    all_titles: list[str],
) -> bool:
    """
    Return True if NAV_TITLE_MATCH_MIN or more distinct body lines match
    other slide titles (case-insensitive). Detects agenda slides.
    """
    title_set = {t.strip().casefold() for t in all_titles}
    matches = set()
    for line in body_text.splitlines():
        cleaned = line.strip().strip("-•*").strip().casefold()
        if cleaned and cleaned in title_set:
            matches.add(cleaned)
    return len(matches) >= NAV_TITLE_MATCH_MIN


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _libreoffice_available() -> bool:
    # Supports absolute executable paths and command names resolved via PATH.
    path = Path(LIBREOFFICE_PATH)
    if path.exists() and path.is_file():
        return True
    return shutil.which(LIBREOFFICE_PATH) is not None


def _visible_slide_numbers(module_id: str) -> list[int]:
    """1-based numbers of non-hidden slides, in order (= PDF page order)."""
    pptx_path = Path(PPTS_DIR) / f"{module_id}.pptx"
    prs = Presentation(str(pptx_path))
    return [
        i
        for i, slide in enumerate(prs.slides, start=1)
        if slide._element.get("show", "1") not in ("0", "false")
    ]


def _cleanup_stale_images(image_dir: Path, keep_slide_numbers: set[int]) -> None:
    """Delete slide_N.png files from earlier runs that are no longer needed."""
    if not image_dir.is_dir():
        return
    for png_path in image_dir.glob("slide_*.png"):
        match = re.fullmatch(r"slide_(\d+)\.png", png_path.name)
        if not match:
            continue
        if int(match.group(1)) not in keep_slide_numbers:
            try:
                png_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete stale image %s: %s", png_path, exc)


def _libreoffice_render(
    module_id: str,
    output_dir: Path,
    render_height: Optional[int] = None,
    page_to_slide_numbers: Optional[list[int]] = None,
    render_only_slide_numbers: Optional[set[int]] = None,
) -> dict[int, Path]:
    """
    Render PPTX to PNGs using LibreOffice (PPTX -> PDF) + pypdfium2 (PDF -> PNG).
    Returns {slide_number: png_path} for the slides actually rendered.

    render_height:         computed from the real slide aspect ratio by the
                           parser; falls back to IMAGE_HEIGHT from config.
    page_to_slide_numbers: PDF page i (0-based) -> slide number. Needed because
                           LibreOffice skips hidden slides.
    render_only_slide_numbers: if given, other pages are skipped.
    """
    pptx_path = Path(PPTS_DIR) / f"{module_id}.pptx"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".lo_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / f"{module_id}.pdf"
    pdf_path.unlink(missing_ok=True)  # never reuse a PDF from a crashed run
    try:
        converted = False
        last_error = "unknown error"
        for attempt in range(1, LIBREOFFICE_MAX_ATTEMPTS + 1):
            # Fresh profile per attempt: a hung soffice from a timeout can
            # keep the previous profile locked.
            profile_uri = (temp_dir / f"lo_profile_{attempt}").resolve().as_uri()
            try:
                result = subprocess.run(
                    [
                        LIBREOFFICE_PATH,
                        "--headless",
                        f"-env:UserInstallation={profile_uri}",
                        "--convert-to", "pdf",
                        "--outdir", str(temp_dir),
                        str(pptx_path),
                    ],
                    capture_output=True,
                    check=False,
                    timeout=LIBREOFFICE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                last_error = f"timed out after {LIBREOFFICE_TIMEOUT_SECONDS}s"
            else:
                if result.returncode == 0 and pdf_path.exists():
                    converted = True
                    break
                raw = result.stderr or result.stdout or b""
                last_error = (
                    raw.decode("utf-8", errors="replace").strip()
                    or f"exit code {result.returncode}"
                )
            if attempt < LIBREOFFICE_MAX_ATTEMPTS:
                logger.warning(
                    "LibreOffice conversion failed for %s (attempt %d): %s; retrying.",
                    module_id, attempt, last_error,
                )
        if not converted:
            raise RuntimeError(
                f"LibreOffice did not produce PDF for {module_id}: {last_error}"
            )
        return _pdf_to_pngs(
            pdf_path,
            output_dir,
            render_height,
            page_to_slide_numbers=page_to_slide_numbers,
            render_only_slide_numbers=render_only_slide_numbers,
        )
    finally:
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                break
            except OSError:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)


def _pdf_to_pngs(
    pdf_path: Path,
    output_dir: Path,
    render_height: Optional[int] = None,
    page_to_slide_numbers: Optional[list[int]] = None,
    render_only_slide_numbers: Optional[set[int]] = None,
) -> dict[int, Path]:
    """
    Render PDF pages to PNGs with pypdfium2. Returns {slide_number: png_path}.
    A page that fails to render is logged and skipped; the others are kept.
    """
    height = render_height or IMAGE_HEIGHT
    rendered: dict[int, Path] = {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(pdf)
        if (
            page_to_slide_numbers is not None
            and page_count != len(page_to_slide_numbers)
        ):
            raise RuntimeError(
                f"PDF page count ({page_count}) does not match expected "
                f"visible slide count ({len(page_to_slide_numbers)})."
            )
        for page_index in range(page_count):
            slide_number = (
                page_to_slide_numbers[page_index]
                if page_to_slide_numbers is not None
                else page_index + 1
            )
            if (
                render_only_slide_numbers is not None
                and slide_number not in render_only_slide_numbers
            ):
                continue
            page = pdf[page_index]
            try:
                scale_x = IMAGE_WIDTH / page.get_width()
                scale_y = height / page.get_height()
                scale = min(scale_x, scale_y) * RENDER_SCALE_SAFETY_FACTOR
                pil_image = page.render(scale=scale).to_pil()
                dest_path = output_dir / f"slide_{slide_number}.png"
                pil_image.save(dest_path, "PNG")
                rendered[slide_number] = dest_path
            except (pdfium.PdfiumError, OSError, ValueError) as exc:
                logger.warning(
                    "Failed to render slide %s from %s: %s",
                    slide_number, pdf_path.name, exc,
                )
            finally:
                page.close()
    finally:
        pdf.close()
    return rendered


# ---------------------------------------------------------------------------
# Main categorization function
# ---------------------------------------------------------------------------
def categorize_slides(
    slides: list[SlideData],
    render_height: Optional[int] = None,
) -> list[SlideData]:
    """
    Categorize slides as NAVIGATION, TEXT_ONLY, or IMAGE and render IMAGE
    slides to PNG.

    render_height: computed from the actual slide aspect ratio by
                   parse_pptx() (partial_metadata["render_height"]).
                   Falls back to IMAGE_HEIGHT from config if None.

    Pass 1 - NAVIGATION:
      Slide 1 always NAVIGATION.
      Whole-title keyword match -> NAVIGATION.
      Body lists 3+ other slide titles -> NAVIGATION.
    Pass 2 - TEXT_ONLY vs IMAGE:
      visual shapes present -> IMAGE, otherwise TEXT_ONLY.
    Pass 3 - Render IMAGE slides:
      LibreOffice -> PDF -> pypdfium2 -> PNG (hidden slides handled).
      Slides that fail to render are downgraded to TEXT_ONLY.
    """
    if not slides:
        return slides

    height = render_height or IMAGE_HEIGHT
    module_id = slides[0].module_id
    image_dir = Path(IMAGES_DIR) / module_id

    # Reset mutable fields so reruns on the same objects start clean.
    for slide in slides:
        slide.category = ""
        slide.image_path = ""

    all_titles = [s.title for s in slides if s.title.strip()]

    # ------------------------------------------------------------------
    # Pass 1 - NAVIGATION
    # ------------------------------------------------------------------
    for slide in slides:
        if (
            _is_navigation_by_title(slide.title, slide.slide_number)
            or _body_matches_other_titles(slide.body_text, all_titles)
        ):
            slide.category = "NAVIGATION"

    # ------------------------------------------------------------------
    # Pass 2 - TEXT_ONLY or IMAGE
    # ------------------------------------------------------------------
    visual_shapes = {
        "image_placeholder",
        "chart",
        "smart_art",
        "grouped_shapes",
    }
    image_slide_numbers: set[int] = set()
    for slide in slides:
        if slide.category == "NAVIGATION":
            continue
        if visual_shapes.intersection(slide.shape_inventory):
            slide.category = "IMAGE"
            image_slide_numbers.add(slide.slide_number)
        else:
            slide.category = "TEXT_ONLY"

    # Remove PNGs from earlier runs that are no longer needed
    # (also runs when there are zero IMAGE slides).
    _cleanup_stale_images(image_dir, image_slide_numbers)

    if not image_slide_numbers:
        return slides

    # ------------------------------------------------------------------
    # Pass 3 - Render IMAGE slides
    # ------------------------------------------------------------------
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered_map: dict[int, str] = {}

    if _libreoffice_available():
        try:
            visible = _visible_slide_numbers(module_id)
            render_targets = image_slide_numbers & set(visible)
            hidden_images = image_slide_numbers - set(visible)
            if hidden_images:
                logger.info(
                    "Hidden slides %s of %s will not be rendered.",
                    sorted(hidden_images), module_id,
                )
            if render_targets:
                rendered = _libreoffice_render(
                    module_id,
                    image_dir,
                    render_height=height,
                    page_to_slide_numbers=visible,
                    render_only_slide_numbers=render_targets,
                )
                rendered_map = {
                    num: str(path)
                    for num, path in rendered.items()
                    if num in image_slide_numbers
                }
        except (
            RuntimeError,
            ValueError,
            KeyError,
            OSError,
            subprocess.SubprocessError,
            pdfium.PdfiumError,
            PackageNotFoundError,
            zipfile.BadZipFile,
        ) as e:
            logger.warning("LibreOffice render failed for %s: %s", module_id, e)
            rendered_map = {}
    else:
        logger.warning(
            "LibreOffice unavailable (%s); IMAGE slides will be downgraded to TEXT_ONLY.",
            LIBREOFFICE_PATH,
        )

    # Apply paths, or downgrade on failure.
    downgraded = []
    for slide in slides:
        if slide.category != "IMAGE":
            continue
        if slide.slide_number in rendered_map:
            slide.image_path = rendered_map[slide.slide_number]
        else:
            slide.category = "TEXT_ONLY"
            slide.image_path = ""
            downgraded.append(slide.slide_number)
    if downgraded:
        logger.warning(
            "Downgraded %d IMAGE slides to TEXT_ONLY for module %s: %s",
            len(downgraded), module_id, downgraded,
        )

    return slides