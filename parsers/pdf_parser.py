"""
PDF parsing module for SmartPaper AI.

partition()           runs unstructured's hi_res layout model on a PDF
structure_elements()  turns its elements into text chunks, tables, figures
parse_pdf()           both, in one call

History, because this module has changed its mind and the reasons
matter:

The original notebook, and the first versions of this module, kept
ONLY NarrativeText elements as text and sent EVERY Image element to a
vision model. That mirrored the notebook, where nothing else was ever
embedded. It was wrong for real papers. On a 20-page IEEE Access
article the layout model produced:

- the ABSTRACT as a ListItem                       -> dropped
- 86 list items (contributions, protocols,
  criteria - a fifth of the paper's words)         -> dropped
- the paper title and 40 section headings          -> dropped
- 12 figure captions, 6 table captions             -> dropped
- "VOLUME 14, 2026" and the page number, per page  -> kept as passages
- 203 fragments with a median of 299 characters    -> each embedded alone
- 37 "figures", of which 20 were the journal's
  logo repeated on every page and 4 were author
  portraits                                        -> all sent to Gemini

So the paper overview was written from a DOI and an affiliation line,
answers were stitched from sentence fragments, and one upload spent
more vision requests than Gemini's free tier allows in a day.

What this module does now:

- Text is everything that is content - Title, NarrativeText, ListItem,
  uncategorized text, formulas - minus running headers/footers and
  page numbers (detected as short lines repeated across pages), grouped
  by section with unstructured's chunk_by_title into chunks of roughly
  config.CHUNK_TARGET_CHARS. A chunk knows the pages it spans.
- Captions ("FIGURE 5. ...", "TABLE 2. ...") are taken out of the text
  and attached to the figure or table they describe, along with a
  label ("Figure 5") that lets a question name it directly. Matching
  is by position on the page - the nearest caption within a small
  gap - because reading order is unreliable in a two-column layout.
- Figures are filtered. A crop that is tiny is dropped; one without a
  caption is dropped if it is empty of text (a portrait, an ornament),
  a wide strip (a masthead), or the same size on many pages (a logo).
  A crop WITH a caption is always kept: a paper's plots are often all
  exported at one size.
- The reference list (and whatever follows it, short of an appendix)
  is chunked separately and tagged section="references", so ordinary
  questions are not answered from other people's titles.
- Each figure carries the text the layout model read inside it
  ("ROC curve", axis labels, legend entries), which together with the
  caption is what gets indexed - see embeddings/embedder.py.

Two things are unchanged, deliberately:

- Images are found via isinstance(el, Image) and
  el.metadata.image_path, since unstructured saves cropped image files
  to disk (extract_image_block_to_payload=False) rather than embedding
  base64 in the element itself. Category-string matching on "Image"
  does NOT carry the file path.
- extract_image_dir has no default. unstructured names crops
  "figure-<page>-<n>.jpg", so two papers parsed into one shared
  directory overwrite each other's figures. The caller must pass a
  directory that is unique to this parse.
"""

import collections
import logging
import os
import re
import sys

from pypdf import PdfReader
from pypdf.errors import PyPdfError

if __name__ == "__main__":
    # Run as a script, sys.path[0] is parsers/ - add the repo root so
    # that `import config` works.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

logger = logging.getLogger("smartpaper.parser")


class PdfError(Exception):
    """The file cannot be parsed as a usable PDF. The message is user-facing."""


def count_pages(file_path: str) -> int:
    """
    Return the page count of a PDF, without running the (slow) layout
    model. Raises PdfError for files that are not readable PDFs or
    that are password-protected.
    """
    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            raise PdfError("This PDF is password-protected. Upload an unlocked copy.")
        return len(reader.pages)
    except PdfError:
        raise
    except (PyPdfError, OSError, ValueError, KeyError) as exc:
        logger.info("Rejected unreadable PDF: %s", exc)
        raise PdfError("This file could not be read as a PDF. It may be damaged.") from exc


# --- Captions and labels ----------------------------------------------------
# A caption starts with the word, a number, and then a period or colon:
# "FIGURE 5. Confusion matrix", "Fig. 3: Overview", "TABLE II. Results".
# The punctuation is what separates a caption from a sentence that
# merely begins "Table 4 summarizes the ...".
_CAPTION_RE = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?\.?|table)\s*(?P<number>\d+|[ivxlc]+)\s*[.:]",
    re.IGNORECASE,
)
_MAX_CAPTION_CHARS = 600

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _roman_to_int(text: str) -> int:
    total, previous = 0, 0
    for char in reversed(text.lower()):
        value = _ROMAN[char]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def caption_label(text: str) -> tuple[str, str] | None:
    """
    If text is a figure or table caption, return (kind, label) such as
    ("image", "Figure 5") or ("table", "Table 2"); otherwise None.
    """
    if not text or len(text) > _MAX_CAPTION_CHARS:
        return None
    match = _CAPTION_RE.match(text)
    if not match:
        return None
    number = match.group("number")
    value = int(number) if number.isdigit() else _roman_to_int(number)
    if match.group("kind").lower().startswith("fig"):
        return "image", f"Figure {value}"
    return "table", f"Table {value}"


# --- Helpers ----------------------------------------------------------------
_SKIPPED_TEXT_CATEGORIES = {"Image", "Table", "Header", "Footer", "PageNumber", "PageBreak"}

# "REFERENCES", "VIII. References", "Bibliography" - the heading after
# which the text is a list of other people's papers (and, in IEEE
# journals, the authors' biographies).
_REFERENCES_RE = re.compile(
    r"^\s*(?:[ivxlc\d]+[.)]?\s+)?(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE,
)
# One entry of a numbered reference list: "[16] X. Zhao, T. Wu, ...".
_REFERENCE_ENTRY_RE = re.compile(r"^\[\d{1,3}\]\s*\S")
# A heading that ends the reference list and returns to the paper itself.
_APPENDIX_RE = re.compile(r"^\s*(?:[ivxlc\d]+[.)]?\s+)?(appendix|appendices|supplementary)\b", re.IGNORECASE)


def _split_sections(body: list) -> list[tuple[str, list]]:
    """
    Split the body elements into runs of ("body" | "references", elements).

    The reference list is kept - a question may be about it - but it is
    tagged, so that ordinary questions are not answered from it. Its
    entries are dense with the paper's own keywords, and on a real
    paper they outranked the title page for "what is this paper about
    and who wrote it?".
    """
    # In a two-column layout the model can list the first reference
    # entries of a column BEFORE the "REFERENCES" heading. So an entry
    # that looks like one ("[16] X. Zhao, ...") on the heading's page
    # or later is a reference wherever it falls in the order.
    references_page = next(
        (_page_number(element) for _, element in body if _REFERENCES_RE.match(str(element).strip())),
        None,
    )

    runs: list[tuple[str, list]] = []
    section = "body"
    for _, element in body:
        text = str(element).strip()
        if _REFERENCES_RE.match(text):
            section = "references"
        elif section == "references" and element.category == "Title" and _APPENDIX_RE.match(text):
            section = "body"

        element_section = section
        if (
            section == "body"
            and references_page is not None
            and (_page_number(element) or 0) >= references_page
            and _REFERENCE_ENTRY_RE.match(text)
        ):
            element_section = "references"

        if runs and runs[-1][0] == element_section:
            runs[-1][1].append(element)
        else:
            runs.append((element_section, [element]))
    return runs


def _page_number(element):
    return getattr(element.metadata, "page_number", None)


def _normalise(text: str) -> str:
    """Collapse digits so 'VOLUME 14, 2026' and page numbers compare equal across pages."""
    return re.sub(r"\d+", "#", text.strip().lower())


def _find_boilerplate(elements) -> set[str]:
    """
    Short lines that recur on several pages: running heads, journal
    footers, page numbers. Returned in _normalise()d form.
    """
    pages_seen = collections.defaultdict(set)
    last_page = 1
    for element in elements:
        page = _page_number(element) or 1
        last_page = max(last_page, page)
        text = str(element).strip()
        if element.category in ("Image", "Table", "FigureCaption") or not text:
            continue
        # Captions are never boilerplate, however alike they look once
        # their numbers are masked ("FIGURE 5. Confusion matrix of ...").
        if len(text) >= config.BOILERPLATE_MAX_CHARS or caption_label(text) is not None:
            continue
        pages_seen[_normalise(text)].add(page)

    threshold = max(3, last_page // 4)
    return {text for text, pages in pages_seen.items() if len(pages) >= threshold}


def _image_size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - an unreadable crop is simply not a figure
        return None


def _box(element):
    """
    (x1, y1, x2, y2, page_height) of an element in the layout model's
    pixel space, or None if it carries no coordinates.
    """
    coordinates = getattr(element.metadata, "coordinates", None)
    points = getattr(coordinates, "points", None)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    system = getattr(coordinates, "system", None)
    page_height = getattr(system, "height", None) or max(ys)
    return min(xs), min(ys), max(xs), max(ys), page_height


def _gap(a, b) -> float:
    """Shortest distance between two boxes (0 if they touch or overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _claim_by_order(captions: list, kind: str, page, index: int, prefer_after: bool):
    """
    Fallback when elements carry no coordinates: give a figure/table
    the nearest unclaimed caption of its kind on its page by reading
    order. Figures prefer the caption that follows them; tables take
    whichever is nearest.
    """
    candidates = [c for c in captions if c["kind"] == kind and c["page"] == page and not c["claimed"]]
    if not candidates:
        return None
    if prefer_after:
        after = [c for c in candidates if c["index"] > index]
        chosen = min(after, key=lambda c: c["index"]) if after else max(candidates, key=lambda c: c["index"])
    else:
        chosen = min(candidates, key=lambda c: abs(c["index"] - index))
    chosen["claimed"] = True
    return chosen


def _assign_captions(items: list, captions: list, kind: str, prefer_after: bool) -> None:
    """
    Set item["caption"] (a caption dict or None) for every figure or
    table in items.

    Each item is {"index", "page", "box"}. Pairs on the same page are
    matched closest-first by the gap between their boxes, so a caption
    goes to the figure it sits under and a far-away logo gets none.
    Items without coordinates fall back to reading order.
    """
    for item in items:
        item["caption"] = None

    pairs = []
    for item_number, item in enumerate(items):
        if item["box"] is None:
            continue
        for caption in captions:
            if caption["kind"] != kind or caption["page"] != item["page"] or caption["box"] is None:
                continue
            gap = _gap(item["box"], caption["box"])
            if gap <= config.CAPTION_MAX_GAP_FRACTION * item["box"][4]:
                pairs.append((gap, item_number, caption["index"], caption))

    for _, item_number, _, caption in sorted(pairs, key=lambda pair: pair[:3]):
        item = items[item_number]
        if item["caption"] is None and not caption["claimed"]:
            caption["claimed"] = True
            item["caption"] = caption

    for item in items:
        if item["caption"] is None and item["box"] is None:
            item["caption"] = _claim_by_order(captions, kind, item["page"], item["index"], prefer_after)


# --- Structuring ------------------------------------------------------------
def structure_elements(elements) -> dict:
    """
    Turn unstructured's elements into the dict the embedding step
    consumes. Pure apart from reading image sizes from disk, so it can
    be tested without the layout model.

    Returns:
        dict with keys:
            text_chunks: list of {text, page_number, page_end, category,
                         section} - section-aware chunks; page_end is
                         the last page the chunk's text comes from;
                         section is "body" or "references"
            tables: list of {text, page_number, image_path, caption, label}
                    where text is "Table on page X: <caption> <cells>"
            images: list of {path, page_number, caption, label, ocr_text}
                    - real figures only
            images_skipped: how many image crops were dropped as
                            logos, ornaments or portraits
    """
    from unstructured.chunking.title import chunk_by_title
    from unstructured.documents.elements import Image as ImageElement

    elements = list(elements)
    boilerplate = _find_boilerplate(elements)

    # Pass 1: split the stream into captions, body text, tables, image candidates.
    captions = []
    body = []
    table_elements = []
    image_candidates = []

    for index, element in enumerate(elements):
        text = str(element).strip()
        category = element.category

        if category == "Table":
            table_elements.append((index, element))
            continue
        if isinstance(element, ImageElement):
            path = getattr(element.metadata, "image_path", None)
            if path and os.path.isfile(path):
                image_candidates.append((index, element, path, _image_size(path)))
            continue
        if category in _SKIPPED_TEXT_CATEGORIES or not text:
            continue

        labelled = caption_label(text)
        if labelled is None and category != "FigureCaption" and _normalise(text) in boilerplate:
            continue
        if labelled is not None:
            captions.append({
                "kind": labelled[0], "label": labelled[1], "text": text,
                "page": _page_number(element), "index": index, "claimed": False,
                "box": _box(element),
            })
            continue
        if category == "FigureCaption":
            # A caption the layout model recognised but that carries no
            # "Figure N." prefix: still a caption, just without a label.
            captions.append({
                "kind": "image", "label": None, "text": text,
                "page": _page_number(element), "index": index, "claimed": False,
                "box": _box(element),
            })
            continue

        body.append((index, element))

    # Pass 2: figures.
    figures = []
    skipped = 0
    for index, element, path, size in image_candidates:
        if size is None or min(size) < config.IMAGE_MIN_SIDE_PX:
            skipped += 1          # an icon, a rule, a logo strip
            continue
        figures.append({
            "index": index, "element": element, "path": path, "size": size,
            "page": _page_number(element), "box": _box(element),
        })
    _assign_captions(figures, captions, "image", prefer_after=True)

    # The remaining rules apply only to crops WITHOUT a caption. A
    # captioned crop is a figure, whatever its shape - and a paper's
    # plots are often all exported at one size, so "same size on many
    # pages" must not be held against them.
    size_pages = collections.defaultdict(set)
    for figure in figures:
        if figure["caption"] is None:
            size_pages[figure["size"]].add(figure["page"])

    images = []
    for figure in figures:
        caption = figure["caption"]
        size = figure["size"]
        ocr_text = str(figure["element"]).strip()
        if caption is None and (
            len(ocr_text) < config.IMAGE_MIN_OCR_CHARS                              # portrait, ornament
            or max(size) / min(size) > config.IMAGE_MAX_UNCAPTIONED_ASPECT          # masthead banner
            or len(size_pages[size]) >= config.IMAGE_REPEAT_PAGES                   # running logo
        ):
            skipped += 1
            continue
        images.append({
            "path": figure["path"],
            "page_number": figure["page"],
            "caption": caption["text"] if caption else "",
            "label": (caption["label"] if caption else None) or "",
            "ocr_text": ocr_text,
        })

    # Pass 3: tables.
    table_items = [
        {"index": index, "element": element, "page": _page_number(element), "box": _box(element)}
        for index, element in table_elements
    ]
    _assign_captions(table_items, captions, "table", prefer_after=False)

    tables = []
    for item in table_items:
        caption = item["caption"]
        caption_text = caption["text"] if caption else ""
        tables.append({
            "text": f"Table on page {item['page']}: {caption_text + ' ' if caption_text else ''}{str(item['element'])}",
            "page_number": item["page"],
            "image_path": getattr(item["element"].metadata, "image_path", None),
            "caption": caption_text,
            "label": (caption["label"] if caption else None) or "",
        })

    # A caption nobody claimed (its figure was not detected) is still
    # information about the paper: put it back into the text.
    body.extend((c["index"], elements[c["index"]]) for c in captions if not c["claimed"])
    body.sort(key=lambda pair: pair[0])

    # Pass 4: section-aware chunks, with the reference list kept apart.
    text_chunks = []
    for section, section_elements in _split_sections(body):
        chunks = chunk_by_title(
            section_elements,
            max_characters=config.CHUNK_MAX_CHARS,
            new_after_n_chars=config.CHUNK_TARGET_CHARS,
            combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER_CHARS,
            multipage_sections=True,
        )
        for chunk in chunks:
            text = str(chunk).strip()
            if not text:
                continue
            originals = getattr(chunk.metadata, "orig_elements", None) or []
            pages = sorted({p for p in (_page_number(o) for o in originals) if p is not None})
            first = pages[0] if pages else _page_number(chunk)
            text_chunks.append({
                "text": text,
                "page_number": first,
                "page_end": pages[-1] if pages else first,
                "category": chunk.category,
                "section": section,
            })

    logger.info(
        "Structured %d elements into %d text chunks, %d tables, %d figures (%d image crops skipped)",
        len(elements), len(text_chunks), len(tables), len(images), skipped,
    )
    return {
        "text_chunks": text_chunks,
        "tables": tables,
        "images": images,
        "images_skipped": skipped,
    }


def partition(file_path: str, extract_image_dir: str) -> list:
    """Run the hi_res layout model on a PDF and return unstructured's elements."""
    # Imported here rather than at module level: unstructured pulls in
    # torch and the layout-detection stack, which takes many seconds
    # to import. count_pages() and the app's startup should not pay
    # for that until a PDF is actually parsed.
    from unstructured.partition.pdf import partition_pdf

    return partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_images_in_pdf=True,
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=extract_image_dir,
    )


def parse_pdf(file_path: str, extract_image_dir: str) -> dict:
    """
    Parse a PDF into text chunks, tables, and figures.

    Args:
        file_path: path to the PDF file to parse.
        extract_image_dir: directory unstructured will save extracted
                            images (and table snapshots) into. Must be
                            unique to this parse - see module docstring.

    Returns:
        the dict described in structure_elements().
    """
    return structure_elements(partition(file_path, extract_image_dir))


if __name__ == "__main__":
    # quick manual smoke test: python parsers/pdf_parser.py path/to/paper.pdf
    import tempfile

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = parse_pdf(sys.argv[1], tmp_dir)
    print(f"Pages: {count_pages(sys.argv[1])}")
    print(f"Text chunks: {len(result['text_chunks'])}")
    print(f"Figures: {len(result['images'])} (skipped {result['images_skipped']} logos/ornaments)")
    print(f"Tables: {len(result['tables'])}")
