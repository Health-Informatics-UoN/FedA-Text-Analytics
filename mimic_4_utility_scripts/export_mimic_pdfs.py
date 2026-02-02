import os
import re
import sys
import psycopg
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from xml.sax.saxutils import escape
from datetime import datetime

# --- Database connection settings (update) ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

OUTPUT_DIR = "mimic_discharge_pdfs"
BATCH_SIZE = 1000
LOG_PATH = "pdf_generation.log"

# remove control chars that are illegal in XML/HTML except newline and tab
INVALID_CTRL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"  # exclude \t (\x09), \n (\x0A), \r (\x0D) if you like
)


def sanitize_text_for_paragraph(raw_text: str) -> str:
    if raw_text is None:
        return ""
    # Remove problematic control chars
    cleaned = INVALID_CTRL_RE.sub("", raw_text)
    # Escape HTML special chars so ReportLab paragraph parser won't choke
    escaped = escape(cleaned)
    # Now turn newlines into real <br/> tags for line breaks in Paragraph
    with_breaks = escaped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return with_breaks


def safe_filename(name: str) -> str:
    # Make a filesystem-safe filename (basic)
    name = str(name)
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    return name


def write_plain_text_pdf(pdf_path: str, text: str):
    """
    Fallback PDF writer that writes raw lines using canvas (no HTML parsing).
    This is more robust but less pretty.
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin_x = 40
    margin_top = 40
    max_width = width - 2 * margin_x
    # Use a textobject for multiline text
    textobj = c.beginText(margin_x, height - margin_top)
    textobj.setFont("Helvetica", 10)
    # naive wrap: split on newline then wrap by approx chars per line
    # estimate chars per line based on font size (very rough)
    approx_char_per_line = int(max_width / 6.5)
    for paragraph in text.splitlines():
        if not paragraph:
            textobj.textLine("")  # keep blank lines
            continue
        start = 0
        while start < len(paragraph):
            chunk = paragraph[start : start + approx_char_per_line]
            textobj.textLine(chunk)
            start += approx_char_per_line
        # after each original newline, continue
    c.drawText(textobj)
    c.showPage()
    c.save()


def log(msg: str):
    now = datetime.utcnow().isoformat()
    line = f"{now} - {msg}\n"
    with open(LOG_PATH, "a", encoding="utf8") as f:
        f.write(line)
    print(line, end="")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor(name="discharge_stream") as cur:
                cur.itersize = BATCH_SIZE
                cur.execute("SELECT subject_id, note_seq, text FROM mimic_4.discharge;")

                styles = getSampleStyleSheet()
                text_style = styles["Normal"]

                count = 0
                for subject_id, note_seq, raw_text in cur:
                    try:
                        subject_dir = os.path.join(OUTPUT_DIR, str(subject_id))
                        os.makedirs(subject_dir, exist_ok=True)

                        safe_note_seq = safe_filename(note_seq)
                        pdf_path = os.path.join(subject_dir, f"{safe_note_seq}.pdf")

                        # prepare text for Paragraph
                        para_text = sanitize_text_for_paragraph(raw_text)

                        # Create PDF using Paragraph (HTML-lite)
                        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                        story = [Paragraph(para_text or "", text_style)]
                        doc.build(story)

                    except Exception as e:
                        # If Paragraph fails for this row, log the error and write a plain-text PDF fallback
                        log(
                            f"ERROR writing Paragraph PDF for subject_id={subject_id} note_seq={note_seq}: {e}"
                        )
                        try:
                            fallback_pdf = os.path.join(
                                subject_dir, f"{safe_note_seq}_fallback.pdf"
                            )
                            # Use original raw text (not escaped) for fallback so the content is preserved
                            write_plain_text_pdf(fallback_pdf, raw_text or "")
                            log(
                                f"WROTE fallback plain-text PDF for subject_id={subject_id} note_seq={note_seq} -> {fallback_pdf}"
                            )
                        except Exception as e2:
                            log(
                                f"CRITICAL: failed fallback for subject_id={subject_id} note_seq={note_seq}: {e2}"
                            )

                    count += 1
                    if count % 100 == 0:
                        log(f"Generated {count} PDFs so far...")

                log(f"Done. Generated/processed {count} rows.")

    except Exception as conn_e:
        log(f"Database connection/query failure: {conn_e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
