"""DOCX export with correct CJK fonts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from app.util import hhmmss
from summarization.chunking import speaker_label


def _set_cjk_font(document, latin: str = "Calibri", east_asia: str = "Microsoft YaHei") -> None:
    from docx.oxml.ns import qn

    style = document.styles["Normal"]
    style.font.name = latin
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)


def _add_markdown(document, markdown: str) -> None:
    """Render a small subset of Markdown into DOCX paragraphs/tables."""
    lines = (markdown or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                width = max(len(r) for r in rows)
                table = document.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                for r_idx, row in enumerate(rows):
                    for c_idx in range(width):
                        cell = table.cell(r_idx, c_idx)
                        cell.text = row[c_idx] if c_idx < len(row) else ""
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.font.size = None
                                if r_idx == 0:
                                    run.bold = True
            continue
        if line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("#### "):
            document.add_heading(line[5:].strip(), level=4)
        elif line.startswith(("- ", "* ")):
            text = line[2:].strip()
            p = document.add_paragraph(style="List Bullet")
            _add_inline(p, text)
        elif re.match(r"^\d+[.)]\s+", line):
            text = re.sub(r"^\d+[.)]\s+", "", line)
            p = document.add_paragraph(style="List Number")
            _add_inline(p, text)
        elif line.startswith("> "):
            p = document.add_paragraph()
            p.paragraph_format.left_indent = 24
            run = p.add_run(line[2:].strip())
            run.italic = True
        elif line.startswith("---"):
            document.add_paragraph("─" * 30)
        else:
            p = document.add_paragraph()
            _add_inline(p, line)
        i += 1


def _add_inline(paragraph, text: str) -> None:
    """Handle **bold** segments."""
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)


def build_docx(
    path: str | Path,
    *,
    title: str,
    meta_lines: Sequence[str],
    utterances: Sequence[dict[str, Any]] | None = None,
    names: dict[str, str] | None = None,
    summary_markdown: str | None = None,
    include_transcript: bool = True,
    include_summary: bool = True,
    duration_sec: float | None = None,
) -> Path:
    from docx import Document
    from docx.shared import Pt

    document = Document()
    _set_cjk_font(document)
    document.styles["Normal"].font.size = Pt(10.5)

    document.add_heading(title, level=0)
    for line in meta_lines:
        p = document.add_paragraph()
        p.add_run(line).italic = True

    if include_summary and summary_markdown:
        document.add_page_break()
        document.add_heading("结构化纪要", level=1)
        _add_markdown(document, summary_markdown)

    if include_transcript and utterances:
        document.add_page_break()
        document.add_heading("逐字稿", level=1)
        names = names or {}
        for u in utterances:
            p = document.add_paragraph()
            stamp = hhmmss(u.get("start"))
            speaker = speaker_label(u.get("speaker", ""), names)
            head = p.add_run(f"[{stamp}] {speaker}：")
            head.bold = True
            p.add_run(str(u.get("text", "")).strip())
            p.paragraph_format.space_after = Pt(6)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path
