"""
Tests for parsers/pdf_parser.py: how the layout model's elements are
turned into text chunks, tables and figures.

The layout model itself is not run. structure_elements() is given
hand-built unstructured elements shaped like the ones a real 20-page
IEEE Access paper produced - the paper that showed the original
"NarrativeText only, every image is a figure" approach to be wrong.
"""

import base64
from pathlib import Path

import pytest
from unstructured.documents.coordinates import PixelSpace
from unstructured.documents.elements import (
    CoordinatesMetadata,
    ElementMetadata,
    FigureCaption,
    Header,
    Image,
    ListItem,
    NarrativeText,
    Table,
    Text,
    Title,
)

import config
from helpers import jpeg_base64
from parsers.pdf_parser import caption_label, structure_elements

LONG_OCR = "ROC curve True Positive Rate False Positive Rate LightGBM CatBoost"


PAGE = (2800, 3806)     # the layout model's pixel space for a letter-size page


def at(box):
    """Coordinates metadata for a box (x1, y1, x2, y2) on the page, or None."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return CoordinatesMetadata(
        points=((x1, y1), (x1, y2), (x2, y2), (x2, y1)),
        system=PixelSpace(width=PAGE[0], height=PAGE[1]),
    )


def el(cls, text, page, box=None):
    return cls(text=text, metadata=ElementMetadata(page_number=page, coordinates=at(box)))


def image(tmp_path, name, page, size=(900, 600), ocr=LONG_OCR, box=None):
    path = Path(tmp_path) / name
    path.write_bytes(base64.b64decode(jpeg_base64(*size)))
    return Image(text=ocr, metadata=ElementMetadata(page_number=page, image_path=str(path), coordinates=at(box)))


def table(tmp_path, name, page, cells):
    path = Path(tmp_path) / name
    path.write_bytes(base64.b64decode(jpeg_base64(600, 300)))
    return Table(text=cells, metadata=ElementMetadata(page_number=page, image_path=str(path)))


def paragraph(page, words=120, tag="body"):
    return el(NarrativeText, " ".join(f"{tag}{page}w{i}" for i in range(words)) + ".", page)


def all_text(result) -> str:
    return "\n".join(chunk["text"] for chunk in result["text_chunks"])


# --- what counts as content ---
def test_abstract_lists_and_headings_are_kept(tmp_path):
    """The layout model labels an IEEE abstract a ListItem; headings are Titles."""
    result = structure_elements([
        el(Title, "Early and Accurate Diagnosis of Ectopic Pregnancy", 1),
        el(ListItem, "ABSTRACT Ectopic pregnancy is a significant cause of maternal morbidity.", 1),
        el(Title, "I. INTRODUCTION", 1),
        paragraph(1),
        el(ListItem, "1. This study investigates tree-based ensemble models.", 2),
        el(Text, "where TP = True Positives and FP = False Positives.", 2),
    ])

    text = all_text(result)
    assert "Early and Accurate Diagnosis of Ectopic Pregnancy" in text
    assert "ABSTRACT Ectopic pregnancy" in text
    assert "I. INTRODUCTION" in text
    assert "1. This study investigates" in text
    assert "where TP = True Positives" in text


def test_the_paper_opens_with_its_title_and_abstract(tmp_path):
    """The overview is written from the first chunks - they must not be a DOI line."""
    result = structure_elements([
        el(NarrativeText, "Received 11 April 2026, accepted 1 May 2026.", 1),
        el(Title, "A SHAP-Based Explainable Machine Learning Approach", 1),
        el(ListItem, "ABSTRACT " + "Early diagnosis matters. " * 40, 1),
        el(Title, "I. INTRODUCTION", 1),
        paragraph(1, 200),
    ])

    opening = "\n".join(chunk["text"] for chunk in result["text_chunks"][:2])
    assert "A SHAP-Based Explainable Machine Learning Approach" in opening
    assert "ABSTRACT Early diagnosis matters." in opening


def test_running_headers_and_page_numbers_are_dropped(tmp_path):
    elements = []
    for page in range(1, 7):
        elements += [
            el(Header, "G. Kocun et al.: Early and Accurate Diagnosis of EP", page),
            paragraph(page),
            el(NarrativeText, f"VOLUME 14, 20{20 + page}", page),   # categorised as text by the model
            el(NarrativeText, str(80510 + page), page),
        ]

    text = all_text(structure_elements(elements))

    assert "VOLUME 14" not in text
    assert "G. Kocun et al." not in text
    assert "80511" not in text and "80516" not in text
    assert "body3w0" in text, "real paragraphs must survive"


def test_a_line_repeated_on_only_two_pages_is_not_treated_as_a_header(tmp_path):
    result = structure_elements([
        el(NarrativeText, "Attention is all you need.", 1), paragraph(1),
        el(NarrativeText, "Attention is all you need.", 2), paragraph(2),
        paragraph(3), paragraph(4),
    ])

    assert all_text(result).count("Attention is all you need.") == 2


# --- chunking ---
def test_fragments_are_grouped_into_section_sized_chunks(tmp_path):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    elements = [el(Title, "III. METHOD", 1)]
    elements += [
        el(NarrativeText, f"The {words[i % 8]} {words[(i // 8) % 8]} step {words[(i * 3) % 8]} refines the model.", 1 + i // 30)
        for i in range(120)
    ]

    result = structure_elements(elements)

    lengths = [len(chunk["text"]) for chunk in result["text_chunks"]]
    assert len(lengths) < 12, "120 one-line fragments must not become 120 chunks"
    assert max(lengths) <= config.CHUNK_MAX_CHARS
    assert min(lengths) > 200


def test_a_chunk_knows_every_page_it_spans(tmp_path):
    result = structure_elements([
        el(Title, "IV. RESULTS", 4),
        el(NarrativeText, "The section starts on page four. " * 8, 4),
        el(NarrativeText, "And continues on page five. " * 8, 5),
    ])

    chunk = result["text_chunks"][0]
    assert (chunk["page_number"], chunk["page_end"]) == (4, 5)


# --- captions ---
@pytest.mark.parametrize("text, expected", [
    ("FIGURE 5. Confusion matrix of the LightGBM model.", ("image", "Figure 5")),
    ("Fig. 3: Overview of the pipeline", ("image", "Figure 3")),
    ("Figure 1: The Transformer - model architecture.", ("image", "Figure 1")),
    ("TABLE 2. The dataset output labels.", ("table", "Table 2")),
    ("TABLE IV. Results", ("table", "Table 4")),
    # sentences that merely start with the word are not captions
    ("Table 4 summarizes the comparative performance.", None),
    ("Figure 9 presents the ROC curves.", None),
    ("Tables are hard.", None),
])
def test_captions_are_told_apart_from_sentences(text, expected):
    assert caption_label(text) == expected


def test_each_figure_gets_its_own_caption_even_when_they_interleave(tmp_path):
    """Two figures above their two captions, as on a real two-column page."""
    result = structure_elements([
        paragraph(13),
        image(tmp_path, "figure-13-1.jpg", 13, ocr="HistGradientBoosting Nested CV Confusion Matrix"),
        image(tmp_path, "figure-13-2.jpg", 13, ocr="LightGBM DART (Macro AUC = 0.941) CatBoost"),
        el(FigureCaption, "FIGURE 7. Confusion matrix of the Histogram-based Gradient Boosting model.", 13),
        el(FigureCaption, "FIGURE 9. ROC curves of the evaluated models.", 13),
        image(tmp_path, "figure-13-3.jpg", 13, ocr="MLP Nested CV Confusion Matrix (10-Fold)"),
        el(FigureCaption, "FIGURE 8. Confusion matrix of the MLP model.", 13),
    ])

    assert [(f["label"], f["ocr_text"][:12]) for f in result["images"]] == [
        ("Figure 7", "HistGradient"),
        ("Figure 9", "LightGBM DAR"),
        ("Figure 8", "MLP Nested C"),
    ]
    assert "FIGURE 7." not in all_text(result), "a caption belongs to its figure, not to the body text"


def test_a_caption_the_model_mislabelled_as_a_heading_still_reaches_its_figure(tmp_path):
    result = structure_elements([
        paragraph(15),
        image(tmp_path, "figure-15-1.jpg", 15),
        el(Title, "FIGURE 12. SHAP summary plot for the LightGBM DART model.", 15),
    ])

    assert result["images"][0]["label"] == "Figure 12"
    assert result["images"][0]["caption"].startswith("FIGURE 12. SHAP summary plot")


def test_table_captions_above_the_table_are_attached(tmp_path):
    result = structure_elements([
        paragraph(11),
        el(Title, "TABLE 2. The dataset output labels (dependent variables).", 11),
        table(tmp_path, "table-11-1.jpg", 11, "Output Label Value Ectopic pregnancy -2"),
        el(Title, "TABLE 3. Comparative performance of classification models.", 11),
        table(tmp_path, "table-11-2.jpg", 11, "Class Metric LGBM CatBoost 0.9762 0.9529"),
    ])

    assert [t["label"] for t in result["tables"]] == ["Table 2", "Table 3"]
    first = result["tables"][0]
    assert first["text"] == (
        "Table on page 11: TABLE 2. The dataset output labels (dependent variables). "
        "Output Label Value Ectopic pregnancy -2"
    )
    assert first["image_path"].endswith("table-11-1.jpg")


def test_a_caption_whose_figure_was_not_detected_stays_in_the_text(tmp_path):
    result = structure_elements([
        paragraph(6),
        el(FigureCaption, "FIGURE 4. MLP architecture.", 6),
        paragraph(6, tag="after"),
    ])

    assert result["images"] == []
    assert "FIGURE 4. MLP architecture." in all_text(result)


# --- which image crops are figures ---
def test_the_journal_logo_on_every_page_is_not_a_figure(tmp_path):
    elements = []
    for page in range(1, 6):
        elements.append(image(tmp_path, f"figure-{page}-1.jpg", page, size=(374, 64), ocr="IEEE Access"))
        elements.append(paragraph(page))
    elements.append(image(tmp_path, "figure-5-2.jpg", 5, size=(1000, 700)))
    elements.append(el(FigureCaption, "FIGURE 1. Histogram-based gradient boosting framework.", 5))

    result = structure_elements(elements)

    assert [f["label"] for f in result["images"]] == ["Figure 1"]
    assert result["images_skipped"] == 5


def test_a_masthead_cannot_steal_the_caption_of_the_figure_below_it(tmp_path):
    """
    In reading order the masthead comes first and the caption is the
    next one after it. On the page, the caption sits right under the
    real figure and far from the masthead - position decides.
    """
    result = structure_elements([
        image(tmp_path, "figure-1-1.jpg", 1, size=(444, 108),
              ocr="IEEE Access Multidisciplinary Rapid Review Open Access Journal", box=(2162, 88, 2606, 196)),
        paragraph(1),
        image(tmp_path, "figure-1-2.jpg", 1, size=(1000, 700), box=(1452, 1500, 2609, 2300)),
        el(FigureCaption, "FIGURE 1. Overview of the proposed framework.", 1, box=(1452, 2344, 2300, 2390)),
    ])

    assert len(result["images"]) == 1
    assert result["images"][0]["label"] == "Figure 1"
    assert result["images"][0]["ocr_text"] == LONG_OCR
    assert result["images_skipped"] == 1


def test_captions_are_matched_by_position_not_reading_order(tmp_path):
    """
    Two columns: the layout model lists both figures, then both
    captions - and lists the right-hand caption first.
    """
    left, right = (248, 321, 1277, 1218), (1452, 321, 2609, 1340)
    result = structure_elements([
        image(tmp_path, "figure-13-1.jpg", 13, ocr="HistGradientBoosting Nested CV Confusion Matrix", box=left),
        image(tmp_path, "figure-13-2.jpg", 13, ocr="LightGBM DART (Macro AUC = 0.941) CatBoost", box=right),
        el(FigureCaption, "FIGURE 9. ROC curves of the evaluated models.", 13, box=(1452, 1383, 2300, 1430)),
        el(FigureCaption, "FIGURE 7. Confusion matrix of the HistGB model.", 13, box=(248, 1262, 1277, 1310)),
    ])

    assert [(f["label"], f["ocr_text"][:12]) for f in result["images"]] == [
        ("Figure 7", "HistGradient"),
        ("Figure 9", "LightGBM DAR"),
    ]


def test_plots_exported_at_one_size_are_all_kept(tmp_path):
    """Same pixel size on many pages means 'logo' only for crops without a caption."""
    elements = []
    for page in range(1, 6):
        elements.append(paragraph(page))
        elements.append(image(tmp_path, f"figure-{page}-1.jpg", page, size=(1000, 700), box=(200, 1000, 1300, 1800)))
        elements.append(el(FigureCaption, f"FIGURE {page}. Confusion matrix of model {page}.", page, box=(200, 1840, 1300, 1890)))

    result = structure_elements(elements)

    assert [f["label"] for f in result["images"]] == [f"Figure {n}" for n in range(1, 6)]
    assert result["images_skipped"] == 0


def test_an_uncaptioned_image_repeated_on_every_page_is_a_logo(tmp_path):
    elements = []
    for page in range(1, 5):
        elements.append(image(tmp_path, f"figure-{page}-1.jpg", page, size=(300, 160), ocr="Journal of Something Important"))
        elements.append(paragraph(page))

    result = structure_elements(elements)

    assert result["images"] == []
    assert result["images_skipped"] == 4


def test_author_portraits_and_mastheads_are_not_figures(tmp_path):
    result = structure_elements([
        image(tmp_path, "figure-1-1.jpg", 1, size=(444, 108), ocr="IEEE Access Multidisciplinary Rapid Review Open Access Journal"),
        paragraph(1),
        image(tmp_path, "figure-20-1.jpg", 20, size=(386, 502), ocr=""),        # portrait: no caption, no text
        image(tmp_path, "figure-20-2.jpg", 20, size=(60, 60), ocr=LONG_OCR),    # icon: too small
        paragraph(20),
    ])

    assert result["images"] == []
    assert result["images_skipped"] == 3


def test_an_uncaptioned_chart_with_text_in_it_is_still_a_figure(tmp_path):
    result = structure_elements([paragraph(3), image(tmp_path, "figure-3-1.jpg", 3, size=(900, 700))])

    assert len(result["images"]) == 1
    assert result["images"][0]["label"] == ""
    assert result["images"][0]["ocr_text"] == LONG_OCR


def test_an_empty_document_yields_nothing():
    assert structure_elements([]) == {"text_chunks": [], "tables": [], "images": [], "images_skipped": 0}


# --- the reference list ---
def test_the_reference_list_is_chunked_apart_and_tagged(tmp_path):
    result = structure_elements([
        el(Title, "VII. CONCLUSION", 17),
        paragraph(17),
        el(Title, "REFERENCES", 18),
        el(ListItem, "[1] M. S. Kupka et al., Assisted reproductive technology in Europe, 2010.", 18),
        el(ListItem, "[2] T. Backman, Pregnancy during the use of levonorgestrel intrauterine system.", 18),
        el(NarrativeText, "GULTEKIN KOCUN received the M.D. degree from a university.", 20),
    ])

    sections = [(chunk["section"], chunk["text"][:14]) for chunk in result["text_chunks"]]
    assert sections == [("body", "VII. CONCLUSIO"), ("references", "REFERENCES\n\n[1")]
    assert "GULTEKIN KOCUN received" in result["text_chunks"][1]["text"], "author biographies follow the references"


def test_reference_entries_listed_before_their_heading_are_still_references(tmp_path):
    """Two columns: the layout model can emit the right column's entries first."""
    result = structure_elements([
        paragraph(18),
        el(ListItem, "[16] X. Zhao, T. Wu, and S. Zeng, Development and validation of a model.", 18),
        el(Title, "REFERENCES", 18),
        el(ListItem, "[1] M. S. Kupka et al., Assisted reproductive technology in Europe.", 18),
    ])

    body = "\n".join(c["text"] for c in result["text_chunks"] if c["section"] == "body")
    assert "[16] X. Zhao" not in body
    assert "body18w0" in body


def test_a_bracketed_number_earlier_in_the_paper_is_not_a_reference_entry(tmp_path):
    result = structure_elements([
        el(NarrativeText, "[4] reported that laparoscopy should be preferred whenever possible in such cases.", 6),
        paragraph(6),
        el(Title, "REFERENCES", 18),
        el(ListItem, "[1] M. S. Kupka et al., Assisted reproductive technology in Europe.", 18),
    ])

    body = "\n".join(c["text"] for c in result["text_chunks"] if c["section"] == "body")
    assert "[4] reported that laparoscopy" in body


def test_an_appendix_after_the_references_is_part_of_the_paper_again(tmp_path):
    result = structure_elements([
        paragraph(9),
        el(Title, "References", 10),
        el(ListItem, "[1] A. Vaswani et al., Attention is all you need, 2017.", 10),
        el(Title, "Appendix A", 11),
        el(NarrativeText, "Additional attention visualizations are shown here in detail for the reader.", 11),
    ])

    assert [c["section"] for c in result["text_chunks"]] == ["body", "references", "body"]
    assert "Additional attention visualizations" in result["text_chunks"][2]["text"]


def test_a_paper_without_a_reference_heading_is_all_body(tmp_path):
    result = structure_elements([paragraph(1), paragraph(2)])

    assert {c["section"] for c in result["text_chunks"]} == {"body"}
