def extract_text_by_layout(page):
    """
    Tries to handle both single-column and two-column layouts.
    """
    # Try default extraction
    text = page.extract_text()
    if text and is_single_column(text):
        return text

    # If suspected two-column layout, try split logic
    return extract_two_column_text(page)

def is_single_column(text):
    """
    Heuristic to detect single-column text.
    You can refine this logic further.
    """
    lines = text.split("\n")
    left_margin_chars = [len(line) - len(line.lstrip()) for line in lines]
    avg_indent = sum(left_margin_chars) / len(left_margin_chars)
    return avg_indent < 10  # arbitrary threshold

def extract_two_column_text(page):
    """
    Splits the page vertically into two and extracts text from both halves.
    """
    width = page.width
    height = page.height
    mid_x = width / 2

    left_bbox = (0, 0, mid_x, height)
    right_bbox = (mid_x, 0, width, height)

    left_text = page.within_bbox(left_bbox).extract_text() or ""
    right_text = page.within_bbox(right_bbox).extract_text() or ""

    return left_text + "\n" + right_text
