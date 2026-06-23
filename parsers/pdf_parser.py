"""
PDF parsing module for SmartPaper AI.

Wraps unstructured.partition_pdf to extract text, images, and tables
from a research paper PDF into a structured dictionary.

Mirrors the categorization logic validated in the original notebook:
- Only NarrativeText elements are kept as text_chunks. The notebook's
  category loop also collects Title, Header, Text, ListItem, Footer,
  and FigureCaption, but create_multi_vector_retriever(...) is only
  ever called with NarrativeText as the texts argument - none of the
  other categories are embedded or used downstream. text_chunks is
  scoped to match exactly what the embedding step consumes.
- Table elements are kept by their string content (raw cell dump),
  tagged with the page number, matching the Day 1 guide's table-string
  format.
- Image file paths are pulled via isinstance(el, Image) and
  el.metadata.image_path, since unstructured saves cropped image files
  to disk (extract_image_block_to_payload=False) rather than embedding
  base64 in the element itself. Category-string matching on "Image"
  does NOT carry the file path - only the isinstance + metadata path
  does, which is why this module uses that approach exclusively for
  images rather than mirroring the (unused) category-loop variant from
  the notebook.
"""

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as ImageElement


def _build_text_chunk(element) -> dict:
    """Convert a NarrativeText element into a chunk dict with metadata."""
    page_number = getattr(element.metadata, "page_number", None)
    return {
        "text": str(element),
        "page_number": page_number,
        "category": element.category,
    }


def _build_table_string(element) -> str:
    """Convert a Table element into the page-tagged string format used downstream."""
    page_number = getattr(element.metadata, "page_number", None)
    return f"Table on page {page_number}: {str(element)}"


def parse_pdf(file_path: str, extract_image_dir: str = "extracted_data") -> dict:
    """
    Parse a PDF into text chunks, image file paths, and table strings.

    Args:
        file_path: path to the PDF file to parse.
        extract_image_dir: directory unstructured will save extracted
                            images (and table snapshots) into.

    Returns:
        dict with keys:
            text_chunks: list of {text, page_number, category} dicts,
                         built only from NarrativeText elements (this
                         is the only category the embedding step uses)
            images: list of image file paths (str), pulled from Image
                    elements via isinstance + metadata.image_path
            tables: list of table strings, formatted as
                    "Table on page X: <raw table content>"
    """
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_images_in_pdf=True,
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=extract_image_dir,
    )

    text_chunks = []
    images = []
    tables = []

    for element in elements:
        category = element.category

        if category == "Table":
            tables.append(_build_table_string(element))
        elif isinstance(element, ImageElement):
            image_path = getattr(element.metadata, "image_path", None)
            if image_path:
                images.append(image_path)
        elif category == "NarrativeText":
            text_chunks.append(_build_text_chunk(element))
        # Header, Footer, Title, Text, ListItem, FigureCaption are
        # categorized in the original notebook but never embedded or
        # consumed downstream, so they are intentionally skipped here.

    return {
        "text_chunks": text_chunks,
        "images": images,
        "tables": tables,
    }


if __name__ == "__main__":
    # quick manual smoke test: python parsers/pdf_parser.py path/to/paper.pdf
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1])
    print(f"Text chunks: {len(result['text_chunks'])}")
    print(f"Images: {len(result['images'])}")
    print(f"Tables: {len(result['tables'])}")