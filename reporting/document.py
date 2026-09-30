"""One report, three formats.

Report content is built once as a list of blocks (headings, paragraphs,
bullet lists, tables, figures) and rendered to Markdown, a self-contained
HTML page (figures embedded), and a Word document. The three files can
therefore never disagree.
"""
from __future__ import annotations

import base64
import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


def fmt_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.3f}"
    return str(value)


@dataclass
class Block:
    kind: str                 # h1 | h2 | h3 | p | bullets | table | figure
    text: str = ""
    items: List[str] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    path: Path | None = None  # figure file


class Report:
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.blocks: List[Block] = []

    # ── building ────────────────────────────────────────────────────────
    def h1(self, text: str) -> None:
        self.blocks.append(Block("h1", text))

    def h2(self, text: str) -> None:
        self.blocks.append(Block("h2", text))

    def h3(self, text: str) -> None:
        self.blocks.append(Block("h3", text))

    def p(self, text: str) -> None:
        self.blocks.append(Block("p", text))

    def bullets(self, items: List[str]) -> None:
        if items:
            self.blocks.append(Block("bullets", items=list(items)))

    def table(self, frame: pd.DataFrame | None, caption: str = "") -> None:
        if frame is None or frame.empty:
            return
        if caption:
            self.p(caption)
        self.blocks.append(Block("table", frame=frame.reset_index(drop=True)))

    def figure(self, path: Path | None, caption: str) -> None:
        if path is not None and Path(path).is_file():
            self.blocks.append(Block("figure", text=caption, path=Path(path)))

    # ── Markdown ────────────────────────────────────────────────────────
    def to_markdown(self, out: Path) -> Path:
        lines = [f"# {self.title}", ""]
        if self.subtitle:
            lines += [self.subtitle, ""]
        for b in self.blocks:
            if b.kind == "h1":
                lines += [f"## {b.text}", ""]
            elif b.kind == "h2":
                lines += [f"### {b.text}", ""]
            elif b.kind == "h3":
                lines += [f"#### {b.text}", ""]
            elif b.kind == "p":
                lines += [b.text, ""]
            elif b.kind == "bullets":
                lines += [f"- {item}" for item in b.items] + [""]
            elif b.kind == "table":
                cols = list(b.frame.columns)
                lines.append("| " + " | ".join(map(str, cols)) + " |")
                lines.append("|" + "|".join(["---"] * len(cols)) + "|")
                for row in b.frame.itertuples(index=False):
                    cells = [fmt_value(v).replace("|", "\\|").replace("\n", " ") for v in row]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
            elif b.kind == "figure":
                rel = b.path.relative_to(out.parent).as_posix()
                lines += [f"![{b.text}]({rel})", "", f"*{b.text}*", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    # ── HTML (self-contained) ───────────────────────────────────────────
    def to_html(self, out: Path) -> Path:
        e = html.escape
        parts = [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{e(self.title)}</title><style>",
            "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;"
            "margin:24px auto;padding:0 16px;color:#1f2328;background:#fff;line-height:1.5}",
            "h1{border-bottom:2px solid #d0d7de;padding-bottom:6px}"
            "h2{margin-top:36px;border-bottom:1px solid #d0d7de;padding-bottom:4px}",
            ".sub{color:#57606a}.tbl{overflow-x:auto;margin:8px 0 16px}",
            "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d0d7de;"
            "padding:4px 8px;text-align:left;vertical-align:top}th{background:#f6f8fa}"
            "td.n{text-align:right;font-variant-numeric:tabular-nums}",
            "figure{margin:12px 0 24px}img{max-width:100%;border:1px solid #eee}"
            "figcaption{color:#57606a;font-size:13px}",
            "</style></head><body>",
            f"<h1>{e(self.title)}</h1>",
        ]
        if self.subtitle:
            parts.append(f"<p class='sub'>{e(self.subtitle)}</p>")
        for b in self.blocks:
            if b.kind in ("h1", "h2", "h3"):
                level = {"h1": 2, "h2": 3, "h3": 4}[b.kind]
                parts.append(f"<h{level}>{e(b.text)}</h{level}>")
            elif b.kind == "p":
                parts.append(f"<p>{e(b.text)}</p>")
            elif b.kind == "bullets":
                parts.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b.items) + "</ul>")
            elif b.kind == "table":
                head = "".join(f"<th>{e(str(c))}</th>" for c in b.frame.columns)
                rows = []
                for row in b.frame.itertuples(index=False):
                    cells = "".join(
                        f"<td class='n'>{e(fmt_value(v))}</td>" if isinstance(v, (int, float))
                        else f"<td>{e(fmt_value(v))}</td>"
                        for v in row
                    )
                    rows.append(f"<tr>{cells}</tr>")
                parts.append(f"<div class='tbl'><table><tr>{head}</tr>{''.join(rows)}</table></div>")
            elif b.kind == "figure":
                data = base64.b64encode(b.path.read_bytes()).decode()
                parts.append(
                    f"<figure><img alt='{e(b.text)}' src='data:image/png;base64,{data}'>"
                    f"<figcaption>{e(b.text)}</figcaption></figure>"
                )
        parts.append("</body></html>")
        out.write_text("".join(parts), encoding="utf-8")
        return out

    # ── Word ────────────────────────────────────────────────────────────
    def to_docx(self, out: Path) -> Path:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.shared import Inches, Pt

        doc = Document()
        section = doc.sections[0]
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(0.6))
        doc.styles["Normal"].font.size = Pt(10)

        doc.add_heading(self.title, level=0)
        if self.subtitle:
            doc.add_paragraph(self.subtitle)
        for b in self.blocks:
            if b.kind in ("h1", "h2", "h3"):
                doc.add_heading(b.text, level={"h1": 1, "h2": 2, "h3": 3}[b.kind])
            elif b.kind == "p":
                doc.add_paragraph(b.text)
            elif b.kind == "bullets":
                for item in b.items:
                    doc.add_paragraph(item, style="List Bullet")
            elif b.kind == "table":
                frame = b.frame
                table = doc.add_table(rows=1, cols=len(frame.columns))
                table.style = "Light Grid Accent 1"
                for cell, col in zip(table.rows[0].cells, frame.columns):
                    cell.text = str(col)
                for row in frame.itertuples(index=False):
                    cells = table.add_row().cells
                    for cell, value in zip(cells, row):
                        cell.text = fmt_value(value)
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Pt(8)
                doc.add_paragraph()
            elif b.kind == "figure":
                doc.add_picture(str(b.path), width=Inches(9.0))
                caption = doc.add_paragraph(b.text)
                caption.runs[0].italic = True
        doc.save(str(out))
        return out
