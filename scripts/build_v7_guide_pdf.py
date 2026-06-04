from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from urllib.parse import unquote


PAGE_W = 612
PAGE_H = 792
MARGIN_X = 56
MARGIN_TOP = 56
MARGIN_BOTTOM = 54
CONTENT_W = PAGE_W - (MARGIN_X * 2)


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text


def wrap_text(text: str, font_size: int, width: int) -> list[str]:
    avg = font_size * 0.52
    max_chars = max(18, int(width / avg))
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if len(trial) <= max_chars:
            line = trial
            continue
        if line:
            lines.append(line)
        while len(word) > max_chars:
            lines.append(word[:max_chars])
            word = word[max_chars:]
        line = word
    if line:
        lines.append(line)
    return lines or [""]


def read_png_rgb(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    width = height = bit_depth = color_type = None
    palette: list[tuple[int, int, int]] = []
    compressed = bytearray()
    while pos < len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk[0:4], "big")
            height = int.from_bytes(chunk[4:8], "big")
            bit_depth = chunk[8]
            color_type = chunk[9]
        elif chunk_type == b"PLTE":
            palette = [tuple(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in {0, 2, 3, 4, 6}:
        raise ValueError(f"Unsupported PNG format for PDF embedding: {path}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows: list[bytes] = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = bytearray(raw[offset:offset + stride])
        offset += stride
        for i in range(stride):
            left = scan[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 1:
                scan[i] = (scan[i] + left) & 0xFF
            elif filter_type == 2:
                scan[i] = (scan[i] + up) & 0xFF
            elif filter_type == 3:
                scan[i] = (scan[i] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                scan[i] = (scan[i] + predictor) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type}: {path}")
        rows.append(bytes(scan))
        prev = scan
    rgb = bytearray(width * height * 3)
    out = 0
    for row in rows:
        if color_type == 0:
            for value in row:
                rgb[out:out + 3] = bytes((value, value, value))
                out += 3
        elif color_type == 2:
            rgb[out:out + len(row)] = row
            out += len(row)
        elif color_type == 3:
            for value in row:
                color = palette[value] if value < len(palette) else (255, 255, 255)
                rgb[out:out + 3] = bytes(color)
                out += 3
        elif color_type == 4:
            for i in range(0, len(row), 2):
                value = row[i]
                alpha = row[i + 1] / 255
                comp = int((value * alpha) + (255 * (1 - alpha)))
                rgb[out:out + 3] = bytes((comp, comp, comp))
                out += 3
        elif color_type == 6:
            for i in range(0, len(row), 4):
                r, g, b, a = row[i:i + 4]
                alpha = a / 255
                rgb[out:out + 3] = bytes((
                    int((r * alpha) + (255 * (1 - alpha))),
                    int((g * alpha) + (255 * (1 - alpha))),
                    int((b * alpha) + (255 * (1 - alpha))),
                ))
                out += 3
    return width, height, bytes(rgb)


class PdfGuide:
    def __init__(self) -> None:
        self.pages: list[dict] = []
        self.anchors: dict[str, tuple[int, float]] = {}
        self.links: list[dict] = []
        self.new_page()

    @property
    def page(self) -> dict:
        return self.pages[-1]

    @property
    def y(self) -> float:
        return self.page["y"]

    @y.setter
    def y(self, value: float) -> None:
        self.page["y"] = value

    def new_page(self) -> None:
        self.pages.append({"ops": [], "y": PAGE_H - MARGIN_TOP, "images": {}})

    def ensure(self, height: float) -> None:
        if self.y - height < MARGIN_BOTTOM:
            self.new_page()

    def text(self, value: str, x: float, y: float, size: int = 11, font: str = "F1", color: str = "0 0 0") -> None:
        self.page["ops"].append(f"{color} rg BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({pdf_escape(value)}) Tj ET")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "0.75 0.78 0.82", width: float = 0.8) -> None:
        self.page["ops"].append(f"{color} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def box(self, x: float, y: float, w: float, h: float, stroke: str = "0.25 0.53 0.65", fill: str = "0.94 0.98 1") -> None:
        self.page["ops"].append(f"{fill} rg {stroke} RG 0.8 w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re B")

    def heading(self, text: str, level: int) -> None:
        if level == 1:
            size, gap_before, gap_after = 24, 0, 18
            color = "0.02 0.20 0.28"
        else:
            size, gap_before, gap_after = 16, 20, 10
            color = "0.03 0.32 0.42"
        self.ensure(gap_before + size + gap_after)
        self.y -= gap_before
        if level == 2:
            self.anchors[slugify(text)] = (len(self.pages) - 1, self.y + 4)
        self.text(text, MARGIN_X, self.y, size=size, font="F2", color=color)
        self.y -= size + gap_after
        if level == 1:
            self.line(MARGIN_X, self.y + 6, PAGE_W - MARGIN_X, self.y + 6, color="0.10 0.48 0.62", width=1.2)

    def paragraph(self, text: str, indent: float = 0, bullet: str | None = None) -> None:
        lines = wrap_text(text, 11, int(CONTENT_W - indent - (14 if bullet else 0)))
        line_h = 15
        self.ensure((len(lines) * line_h) + 7)
        x = MARGIN_X + indent
        if bullet:
            self.text(bullet, x, self.y, size=11, font="F2", color="0.03 0.32 0.42")
            x += 16
        for i, line in enumerate(lines):
            self.text(line, x, self.y - (i * line_h), size=11, color="0.10 0.12 0.14")
        self.y -= (len(lines) * line_h) + 7

    def toc_link(self, title: str, target: str) -> None:
        self.ensure(18)
        self.text(title, MARGIN_X + 12, self.y, size=11, color="0.02 0.38 0.52")
        self.links.append({
            "page": len(self.pages) - 1,
            "rect": [MARGIN_X + 10, self.y - 3, PAGE_W - MARGIN_X, self.y + 12],
            "target": target,
        })
        self.y -= 17

    def placeholder(self, text: str) -> None:
        lines = wrap_text(text, 10, int(CONTENT_W - 24))
        h = max(42, (len(lines) * 13) + 20)
        self.ensure(h + 10)
        self.box(MARGIN_X, self.y - h + 8, CONTENT_W, h)
        yy = self.y - 16
        for line in lines:
            self.text(line, MARGIN_X + 14, yy, size=10, font="F3", color="0.05 0.32 0.42")
            yy -= 13
        self.y -= h + 7

    def image(self, path: Path, title: str = "") -> None:
        try:
            w, h, _ = read_png_rgb(path)
        except Exception:
            self.placeholder(f"<image could not be embedded: {title or path.name}>")
            return
        display_w = CONTENT_W
        display_h = display_w * h / w
        max_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - 50
        if display_h > max_h:
            display_h = max_h
            display_w = display_h * w / h
        needed = display_h + 28
        self.ensure(needed)
        image_name = f"Im{len(self.page['images']) + 1}"
        self.page["images"][image_name] = str(path)
        x = MARGIN_X + ((CONTENT_W - display_w) / 2)
        y = self.y - display_h
        self.page["ops"].append(f"q {display_w:.2f} 0 0 {display_h:.2f} {x:.2f} {y:.2f} cm /{image_name} Do Q")
        self.y = y - 12
        if title:
            for line in wrap_text(title, 9, int(CONTENT_W)):
                self.text(line, MARGIN_X, self.y, size=9, font="F3", color="0.25 0.29 0.33")
                self.y -= 12
        self.y -= 6

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        col_count = max(len(row) for row in rows)
        col_w = CONTENT_W / col_count
        for index, row in enumerate(rows):
            row_lines = []
            max_lines = 1
            for cell in row:
                lines = wrap_text(cell.replace("`", ""), 9, int(col_w - 12))
                row_lines.append(lines)
                max_lines = max(max_lines, len(lines))
            h = (max_lines * 12) + 12
            self.ensure(h + 4)
            fill = "0.90 0.96 0.98" if index == 0 else "1 1 1"
            self.page["ops"].append(f"{fill} rg 0.82 0.86 0.89 RG 0.5 w {MARGIN_X:.2f} {self.y - h + 5:.2f} {CONTENT_W:.2f} {h:.2f} re B")
            for col, lines in enumerate(row_lines):
                x = MARGIN_X + (col * col_w) + 6
                yy = self.y - 9
                font = "F2" if index == 0 else "F1"
                for line in lines:
                    self.text(line, x, yy, size=9, font=font, color="0.10 0.12 0.14")
                    yy -= 12
            self.y -= h
        self.y -= 8


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [cell.strip() for cell in raw.split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def build(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    guide = PdfGuide()
    in_toc = False
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped:
            guide.y -= 4
            i += 1
            continue
        if stripped.startswith("# "):
            guide.heading(stripped[2:].strip(), 1)
            in_toc = False
        elif stripped.startswith("## "):
            title = stripped[3:].strip()
            guide.heading(title, 2)
            in_toc = title.lower() == "table of contents"
        elif stripped.startswith("<screenshot"):
            guide.placeholder(stripped)
        elif stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                image_title = match.group(1).strip()
                image_path = Path(unquote(match.group(2).strip()))
                if not image_path.is_absolute():
                    image_path = (md_path.parent / image_path).resolve()
                guide.image(image_path, image_title)
            else:
                guide.paragraph(stripped)
        elif stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            guide.table(rows)
            in_toc = False
            continue
        elif in_toc and stripped.startswith("- ["):
            match = re.match(r"- \[(.+?)\]\(#(.+?)\)", stripped)
            if match:
                guide.toc_link(match.group(1), match.group(2))
            else:
                guide.paragraph(stripped[2:], bullet="•")
        elif re.match(r"\d+\. ", stripped):
            guide.paragraph(re.sub(r"^\d+\. ", "", stripped), indent=8, bullet="•")
            in_toc = False
        elif stripped.startswith("- "):
            guide.paragraph(stripped[2:], indent=8, bullet="•")
            in_toc = False
        else:
            guide.paragraph(stripped)
            in_toc = False
        i += 1

    write_pdf(guide, pdf_path)


def write_pdf(guide: PdfGuide, out_path: Path) -> None:
    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        objects.append(obj.encode("latin-1") if isinstance(obj, str) else obj)
        return len(objects)

    catalog_id = add("")
    pages_id = add("")
    font_regular = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_italic = add("<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Oblique >>")

    page_ids = [add("") for _ in guide.pages]
    content_ids = [add("") for _ in guide.pages]
    annots_by_page: dict[int, list[int]] = {i: [] for i in range(len(guide.pages))}
    image_objects: dict[tuple[int, str], int] = {}

    for page_index, page in enumerate(guide.pages):
        for image_name, image_path in page.get("images", {}).items():
            w, h, rgb = read_png_rgb(Path(image_path))
            compressed = zlib.compress(rgb, 6)
            image_obj = (
                f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(compressed)} >>\n"
            ).encode("latin-1") + b"stream\n" + compressed + b"\nendstream"
            image_objects[(page_index, image_name)] = add(image_obj)

    for link in guide.links:
        target = guide.anchors.get(link["target"])
        if not target:
            continue
        target_page, target_y = target
        rect = " ".join(f"{v:.2f}" for v in link["rect"])
        annot = (
            f"<< /Type /Annot /Subtype /Link /Rect [{rect}] /Border [0 0 0] "
            f"/A << /S /GoTo /D [{page_ids[target_page]} 0 R /XYZ {MARGIN_X:.2f} {target_y:.2f} null] >> >>"
        )
        annots_by_page[link["page"]].append(add(annot))

    for idx, page in enumerate(guide.pages):
        stream = "\n".join(page["ops"]).encode("latin-1", "replace")
        objects[content_ids[idx] - 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        annot_part = ""
        if annots_by_page[idx]:
            annot_part = " /Annots [" + " ".join(f"{aid} 0 R" for aid in annots_by_page[idx]) + "]"
        image_resource = ""
        if page.get("images"):
            image_resource = " /XObject << " + " ".join(
                f"/{name} {image_objects[(idx, name)]} 0 R" for name in page["images"]
            ) + " >>"
        objects[page_ids[idx] - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}]"
            f" /Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R /F3 {font_italic} 0 R >>{image_resource} >>"
            f" /Contents {content_ids[idx]} 0 R{annot_part} >>"
        ).encode("latin-1")

    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {len(page_ids)} >>"
    ).encode("latin-1")
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R /PageMode /UseNone >>".encode("latin-1")

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode("latin-1")
        data += obj
        data += b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets[1:]:
        data += f"{offset:010d} 00000 n \n".encode("latin-1")
    data += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    out_path.write_bytes(data)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    md = root / "docs" / "V7_VIDEO_BUILDER_GUIDE.md"
    pdf = root / "docs" / "V7_VIDEO_BUILDER_GUIDE.pdf"
    if len(sys.argv) > 1:
        md = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        pdf = Path(sys.argv[2]).resolve()
    build(md, pdf)
    print(pdf)
