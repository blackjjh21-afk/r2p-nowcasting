#!/usr/bin/env python3
"""Render the public Figure 6 assets from the editable source PPTX.

The editable output moves the complete Direct row down by one-third of a box
height, expands panel (a), and translates panel (b) and the slide boundary by
the same amount. Palette and selected line-wrapping changes are also applied.
The static PNG/PDF mirror the adjusted native-object positions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scrub_office_metadata import scrub_office_file


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures/main"
SOURCE = OUTDIR / "fig6_architecture_source.pptx"
STEM = "fig6_architecture"
PPTX = OUTDIR / "fig6_architecture_editable.pptx"
PNG = OUTDIR / f"{STEM}.png"
PDF = OUTDIR / f"{STEM}.pdf"
CAPTION = OUTDIR / f"{STEM}_caption.txt"
MANIFEST = OUTDIR / f"{STEM}_audit_manifest.json"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS = {"a": NS_A, "p": NS_P}
ET.register_namespace("a", NS_A)
ET.register_namespace("p", NS_P)
ET.register_namespace(
    "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)

INK = "#25313C"
WHITE = "#FFFFFF"
PANEL_FILL = "#F7F8FA"
PANEL_EDGE = "#C8CFD6"
RADAR_FILL = "#DCECF7"
RADAR_EDGE = "#2878B5"
FIELD_FILL = "#E2F1EE"
PATCH_FILL = "#DDF3EE"
FIELD_EDGE = "#008F7A"
CONTEXT_FILL = "#FFF0D8"
CONTEXT_EDGE = "#D9822B"
DIRECT_RED = "#D81B3A"
DIRECT_RED_FILL = "#F7D1D8"
# The attention/decoder modules use a desaturated blue-gray that remains
# distinct from both the brighter radar blue and the red Direct-R2P route.
MODULE_FILL = "#E5E7EE"
MODULE_EDGE = "#59647F"
OUTPUT_EDGE = "#000000"

# Historical targets are supervision rather than a forward-computation input.
# A white box retains the user's unshaded target treatment; a saturated purple
# dashed outline gives the historical supervision a clear visual role.
HISTORICAL_FILL = WHITE
HISTORICAL_EDGE = "#7B5EA7"

HISTORICAL_SHAPES = {
    "Field historical gauge targets",
    "Direct historical gauge targets",
}
HISTORICAL_CONNECTOR_IDS = {"19", "20"}
MODULE_SHAPES = {
    "Query conditioned attention",
    "Cross lead decoder",
}
OUTPUT_SHAPES = {
    "Field RN60 output",
    "Direct RN60 output",
    "Direct detail output",
}
MODULE_CONNECTOR_IDS = {"55", "57"}
WRAPS = {
    "Future HSR supervision": ["Future HSR", "targets", "(exPreCast only)"],
    "Query conditioned attention": ["Query-", "conditioned", "attention"],
    "Field gauge context": ["Issuance-time", "gauge context"],
    "Direct gauge context": ["Issuance-time", "gauge context"],
    "Field historical gauge targets": ["Historical gauge", "targets"],
    "Direct historical gauge targets": ["Historical gauge", "targets"],
}

# Every architecture box is 720,000 EMU high. Moving the Direct route by
# 240,000 EMU therefore creates the requested one-third-box clearance from the
# field-first supervision boxes. Panel (b) and the slide boundary move by the
# same amount so all pre-existing outer margins remain unchanged.
EMU_PER_INCH = 914_400
LAYOUT_SHIFT_EMU = 240_000
LAYOUT_SHIFT = LAYOUT_SHIFT_EMU / EMU_PER_INCH
PANEL_A_BACKGROUND_ID = "3"
DIRECT_LAYOUT_IDS = {
    "19",  # historical-target elbow connector
    "25",  # lane label
    "26",  # radar history
    "27",  # Direct R2P
    "28",  # Gauge RN60
    "29",  # radar-to-R2P connector
    "30",  # R2P-to-RN60 connector
    "31",  # issuance-time gauge context
    "33",  # historical gauge targets
    "78",  # gauge-context elbow connector
}
PANEL_B_LAYOUT_IDS = {
    "36", "37", "38", "39", "40", "41", "42", "43", "44", "45",
    "46", "47", "53", "54", "55", "56", "57", "81", "85",
}
STATIC_HEIGHT = 10.60 + LAYOUT_SHIFT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def set_srgb(parent: ET.Element, colour: str) -> None:
    for child in list(parent):
        parent.remove(child)
    ET.SubElement(parent, q(NS_A, "srgbClr"), {"val": colour.lstrip("#")})


def replace_paragraphs(shape: ET.Element, lines: list[str]) -> None:
    body = shape.find("p:txBody", NS)
    if body is None:
        raise RuntimeError("shape lacks a text body")
    paragraphs = body.findall("a:p", NS)
    if not paragraphs:
        raise RuntimeError("shape lacks a source paragraph")
    template = paragraphs[0]
    for paragraph in paragraphs:
        body.remove(paragraph)
    for line in lines:
        paragraph = copy.deepcopy(template)
        text_nodes = paragraph.findall(".//a:t", NS)
        if not text_nodes:
            raise RuntimeError("paragraph lacks a text run")
        text_nodes[0].text = line
        for extra in text_nodes[1:]:
            extra.text = ""
        body.append(paragraph)


def translate_shape_y(element: ET.Element, identifier: str) -> None:
    transform = element.find("p:spPr/a:xfrm", NS)
    if transform is None:
        raise RuntimeError(f"layout object {identifier} lacks a transform")
    offset = transform.find("a:off", NS)
    if offset is None or offset.get("y") is None:
        raise RuntimeError(f"layout object {identifier} lacks a y offset")
    offset.set("y", str(int(offset.get("y", "0")) + LAYOUT_SHIFT_EMU))


def modify_presentation(xml: bytes) -> bytes:
    root = ET.fromstring(xml)
    slide_size = root.find("p:sldSz", NS)
    if slide_size is None or slide_size.get("cy") is None:
        raise RuntimeError("presentation lacks a slide height")
    slide_size.set("cy", str(int(slide_size.get("cy", "0")) + LAYOUT_SHIFT_EMU))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def modify_slide(xml: bytes) -> bytes:
    root = ET.fromstring(xml)
    found_shapes: set[str] = set()
    found_connectors: set[str] = set()
    found_output_shapes: set[str] = set()
    found_layout_objects: set[str] = set()
    found_panel_a_background = False
    tree = root.find(".//p:spTree", NS)
    if tree is None:
        raise RuntimeError("slide has no shape tree")
    for element in tree:
        properties = element.find(".//p:cNvPr", NS)
        if properties is None:
            continue
        identifier = properties.get("id", "")
        name = properties.get("name", "")
        if identifier in DIRECT_LAYOUT_IDS | PANEL_B_LAYOUT_IDS:
            translate_shape_y(element, identifier)
            found_layout_objects.add(identifier)
        if identifier == PANEL_A_BACKGROUND_ID:
            transform = element.find("p:spPr/a:xfrm", NS)
            extent = transform.find("a:ext", NS) if transform is not None else None
            if extent is None or extent.get("cy") is None:
                raise RuntimeError("panel (a) background lacks a height extent")
            extent.set("cy", str(int(extent.get("cy", "0")) + LAYOUT_SHIFT_EMU))
            found_panel_a_background = True
        if name in WRAPS:
            replace_paragraphs(element, WRAPS[name])
            found_shapes.add(name)
        if name in HISTORICAL_SHAPES:
            shape_properties = element.find("p:spPr", NS)
            if shape_properties is None:
                raise RuntimeError(f"{name} lacks shape properties")
            fill = shape_properties.find("a:solidFill", NS)
            line_fill = shape_properties.find("a:ln/a:solidFill", NS)
            if fill is None or line_fill is None:
                raise RuntimeError(f"{name} lacks solid fill or line fill")
            set_srgb(fill, HISTORICAL_FILL)
            set_srgb(line_fill, HISTORICAL_EDGE)
        if name in MODULE_SHAPES:
            shape_properties = element.find("p:spPr", NS)
            if shape_properties is None:
                raise RuntimeError(f"{name} lacks shape properties")
            fill = shape_properties.find("a:solidFill", NS)
            line_fill = shape_properties.find("a:ln/a:solidFill", NS)
            if fill is None or line_fill is None:
                raise RuntimeError(f"{name} lacks solid fill or line fill")
            set_srgb(fill, MODULE_FILL)
            set_srgb(line_fill, MODULE_EDGE)
        if name in OUTPUT_SHAPES:
            line_fill = element.find("p:spPr/a:ln/a:solidFill", NS)
            if line_fill is None:
                raise RuntimeError(f"{name} lacks a line fill")
            set_srgb(line_fill, OUTPUT_EDGE)
            found_output_shapes.add(name)
        if identifier in HISTORICAL_CONNECTOR_IDS:
            line_fill = element.find("p:spPr/a:ln/a:solidFill", NS)
            if line_fill is None:
                raise RuntimeError(f"connector {identifier} lacks a line fill")
            set_srgb(line_fill, HISTORICAL_EDGE)
            found_connectors.add(identifier)
        if identifier in MODULE_CONNECTOR_IDS:
            line_fill = element.find("p:spPr/a:ln/a:solidFill", NS)
            if line_fill is None:
                raise RuntimeError(f"connector {identifier} lacks a line fill")
            set_srgb(line_fill, MODULE_EDGE)
    if found_shapes != set(WRAPS):
        raise RuntimeError(f"missing expected text shapes: {set(WRAPS) - found_shapes}")
    if found_connectors != HISTORICAL_CONNECTOR_IDS:
        raise RuntimeError(
            f"missing historical supervision connectors: "
            f"{HISTORICAL_CONNECTOR_IDS - found_connectors}"
        )
    if found_output_shapes != OUTPUT_SHAPES:
        raise RuntimeError(
            f"missing output shapes: {OUTPUT_SHAPES - found_output_shapes}"
        )
    expected_layout_objects = DIRECT_LAYOUT_IDS | PANEL_B_LAYOUT_IDS
    if found_layout_objects != expected_layout_objects:
        raise RuntimeError(
            f"missing layout objects: {expected_layout_objects - found_layout_objects}"
        )
    if not found_panel_a_background:
        raise RuntimeError("missing panel (a) background")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_editable_pptx() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    temporary = PPTX.with_suffix(PPTX.suffix + ".partial")
    with zipfile.ZipFile(SOURCE) as source, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "ppt/slides/slide1.xml":
                content = modify_slide(content)
            elif info.filename == "ppt/presentation.xml":
                content = modify_presentation(content)
            target.writestr(info, content)
    temporary.replace(PPTX)
    scrub_office_file(PPTX)


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str,
    edge: str,
    fontsize: float = 11.0,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.075",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.3,
            linestyle=(0, (4, 2.4)) if dashed else "-",
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        linespacing=1.06,
        zorder=4,
    )


def panel(ax, x: float, y: float, w: float, h: float, label: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            facecolor=PANEL_FILL,
            edgecolor=PANEL_EDGE,
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(x + 0.20, y + 0.31, label, ha="left", va="center", fontsize=16,
            fontweight="bold", color=INK)
    ax.text(x + 0.70, y + 0.31, title, ha="left", va="center", fontsize=16,
            fontweight="bold", color=INK)


def arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str,
    dashed: bool = False,
    linewidth: float = 1.45,
) -> None:
    style = (0, (4, 2.4)) if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], color=color,
            linewidth=linewidth, linestyle=style, solid_capstyle="round", zorder=1
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=11.5,
            linewidth=linewidth, linestyle=style, color=color, shrinkA=0,
            shrinkB=0, zorder=2
        )
    )


def shifted_y(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x, y + LAYOUT_SHIFT) for x, y in points]


def render_static() -> None:
    fig, ax = plt.subplots(figsize=(18, STATIC_HEIGHT))
    ax.set_xlim(0, 18)
    ax.set_ylim(STATIC_HEIGHT, 0)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    panel(ax, 1.283, 0.576, 15.501, 5.118 + LAYOUT_SHIFT, "(a)",
          "Forecast routes and scope of gauge supervision")
    ax.text(2.573, 1.886, "Field-first", ha="right", va="center", fontsize=14,
            fontweight="bold", color=INK)
    rounded_box(ax, 2.783, 1.456, 1.575, .787, "Radar\nhistory",
                fill=RADAR_FILL, edge=RADAR_EDGE)
    rounded_box(ax, 5.736, 1.456, 1.575, .787, "pySTEPS\nor exPreCast",
                fill=FIELD_FILL, edge=FIELD_EDGE)
    rounded_box(ax, 8.689, 1.456, 1.575, .787, "Forecast HSR",
                fill=FIELD_FILL, edge=FIELD_EDGE)
    rounded_box(ax, 11.641, 1.456, 1.575, .787, "CNN",
                fill=PATCH_FILL, edge=FIELD_EDGE)
    rounded_box(ax, 14.594, 1.456, 1.575, .787, "Gauge RN60",
                fill=PANEL_FILL, edge=OUTPUT_EDGE)
    arrow(ax, [(4.358, 1.850), (5.736, 1.850)], color=RADAR_EDGE)
    arrow(ax, [(7.311, 1.850), (8.689, 1.850)], color=FIELD_EDGE)
    arrow(ax, [(10.264, 1.850), (11.641, 1.850)], color=FIELD_EDGE)
    arrow(ax, [(13.216, 1.850), (14.594, 1.850)], color=FIELD_EDGE)

    rounded_box(ax, 5.736, 2.480, 1.575, .787,
                "Future HSR\ntargets\n(exPreCast only)", fill=WHITE,
                edge=FIELD_EDGE, fontsize=10.2, dashed=True)
    arrow(ax, [(6.523, 2.480), (6.523, 2.243)], color=FIELD_EDGE, dashed=True)

    rounded_box(ax, 10.460, 2.638, 1.575, .787, "Historical gauge\ntargets",
                fill=HISTORICAL_FILL, edge=HISTORICAL_EDGE, fontsize=10.5,
                dashed=True)
    rounded_box(ax, 12.822, 2.638, 1.575, .787, "Issuance-time\ngauge context",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE, fontsize=10.5)
    # Use symmetric, visibly long elbow connectors.  The previous raster
    # reconstruction attached adjacent box edges and reduced both horizontal
    # runs to ~0.2 units, even though the editable PPTX used full elbows.
    arrow(ax, [(11.2475, 2.638), (11.2475, 2.4405), (12.3105, 2.4405),
               (12.3105, 2.243)], color=HISTORICAL_EDGE, dashed=True,
          linewidth=1.55)
    arrow(ax, [(13.6095, 2.638), (13.6095, 2.4405), (12.5475, 2.4405),
               (12.5475, 2.243)], color=CONTEXT_EDGE, linewidth=1.55)

    ax.text(2.573, 3.856 + LAYOUT_SHIFT, "Direct", ha="right", va="center",
            fontsize=14,
            fontweight="bold", color=INK)
    rounded_box(ax, 2.783, 3.426 + LAYOUT_SHIFT, 1.575, .787, "Radar\nhistory",
                fill=RADAR_FILL, edge=RADAR_EDGE)
    rounded_box(ax, 8.689, 3.425 + LAYOUT_SHIFT, 1.575, .787, "Direct R2P",
                fill=DIRECT_RED_FILL, edge=DIRECT_RED)
    rounded_box(ax, 14.594, 3.426 + LAYOUT_SHIFT, 1.575, .787, "Gauge RN60",
                fill=PANEL_FILL, edge=OUTPUT_EDGE)
    arrow(ax, shifted_y([(4.358, 3.819), (8.689, 3.819)]), color=RADAR_EDGE)
    arrow(ax, shifted_y([(10.264, 3.819), (14.594, 3.819)]),
          color=DIRECT_RED)

    rounded_box(ax, 7.507, 4.606 + LAYOUT_SHIFT, 1.575, .787,
                "Historical gauge\ntargets",
                fill=HISTORICAL_FILL, edge=HISTORICAL_EDGE, fontsize=10.5,
                dashed=True)
    rounded_box(ax, 9.870, 4.606 + LAYOUT_SHIFT, 1.575, .787,
                "Issuance-time\ngauge context",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE, fontsize=10.5)
    arrow(ax, shifted_y([(8.2945, 4.606), (8.2945, 4.409),
                         (9.3585, 4.409), (9.3585, 4.212)]),
          color=HISTORICAL_EDGE, dashed=True,
          linewidth=1.55)
    arrow(ax, shifted_y([(10.6575, 4.606), (10.6575, 4.409),
                         (9.5945, 4.409), (9.5945, 4.212)]),
          color=CONTEXT_EDGE, linewidth=1.55)

    panel(ax, 1.283, 5.843 + LAYOUT_SHIFT, 15.501, 4.104, "(b)",
          "Direct R2P architecture")
    rounded_box(ax, 2.783, 6.732 + LAYOUT_SHIFT, 1.575, .787,
                "Radar\nhistory",
                fill=RADAR_FILL, edge=RADAR_EDGE)
    rounded_box(ax, 5.145, 6.732 + LAYOUT_SHIFT, 1.575, .787,
                "Conv stem\n+ ConvGRU",
                fill=RADAR_FILL, edge=RADAR_EDGE)
    rounded_box(ax, 7.507, 6.732 + LAYOUT_SHIFT, 1.575, .787,
                "Multiscale\npyramid",
                fill=RADAR_FILL, edge=RADAR_EDGE)
    arrow(ax, shifted_y([(4.358, 7.126), (5.145, 7.126)]),
          color=RADAR_EDGE)
    arrow(ax, shifted_y([(6.720, 7.126), (7.507, 7.126)]),
          color=RADAR_EDGE)

    rounded_box(ax, 2.783, 8.701 + LAYOUT_SHIFT, 1.575, .787,
                "Issuance-time\ngauge context",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE, fontsize=10.5)
    rounded_box(ax, 5.145, 8.701 + LAYOUT_SHIFT, 1.575, .787,
                "Station\nencoder",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE)
    arrow(ax, shifted_y([(4.358, 9.095), (5.145, 9.095)]),
          color=CONTEXT_EDGE)

    rounded_box(ax, 9.870, 7.717 + LAYOUT_SHIFT, 1.575, .787,
                "Query-\nconditioned\nattention",
                fill=MODULE_FILL, edge=MODULE_EDGE, fontsize=10.2)
    rounded_box(ax, 12.232, 7.717 + LAYOUT_SHIFT, 1.575, .787,
                "Cross-lead\ndecoder",
                fill=MODULE_FILL, edge=MODULE_EDGE)
    rounded_box(ax, 14.594, 7.717 + LAYOUT_SHIFT, 1.575, .787,
                "Gauge\nRN60",
                fill=PANEL_FILL, edge=OUTPUT_EDGE)
    arrow(ax, shifted_y([(9.082, 7.126), (9.476, 7.126),
                         (9.476, 8.110), (9.870, 8.110)]),
          color=RADAR_EDGE)
    arrow(ax, shifted_y([(6.720, 9.095), (10.657, 9.095),
                         (10.657, 8.504)]),
          color=CONTEXT_EDGE)
    ax.text(8.710, 8.935 + LAYOUT_SHIFT, "+ Location and Lead", ha="center",
            va="center", fontsize=11.5, fontweight="bold", color=CONTEXT_EDGE)
    arrow(ax, shifted_y([(11.445, 8.110), (12.232, 8.110)]),
          color=MODULE_EDGE)
    arrow(ax, shifted_y([(13.807, 8.110), (14.594, 8.110)]),
          color=MODULE_EDGE)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(PNG, dpi=300, facecolor=WHITE)
    fig.savefig(PDF, facecolor=WHITE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def verify_pptx() -> dict[str, object]:
    with zipfile.ZipFile(PPTX) as archive:
        root = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    positions: dict[str, dict[str, str]] = {}
    colours: dict[str, str] = {}
    outlines: dict[str, str] = {}
    texts: dict[str, list[str]] = {}
    shape_tree = root.find(".//p:spTree", NS)
    if shape_tree is None:
        raise RuntimeError("final PPTX lacks a shape tree")
    for element in shape_tree:
        properties = element.find(".//p:cNvPr", NS)
        if properties is None:
            continue
        name = properties.get("name", "")
        if name in set(WRAPS) | HISTORICAL_SHAPES | MODULE_SHAPES | OUTPUT_SHAPES:
            transform = element.find("p:spPr/a:xfrm", NS)
            if transform is not None:
                offset = transform.find("a:off", NS)
                extent = transform.find("a:ext", NS)
                positions[name] = {**(offset.attrib if offset is not None else {}),
                                   **(extent.attrib if extent is not None else {})}
            texts[name] = [node.text or "" for node in element.findall(".//a:t", NS)]
            fill = element.find("p:spPr/a:solidFill/a:srgbClr", NS)
            if fill is not None:
                colours[name] = fill.get("val", "")
            outline = element.find("p:spPr/a:ln/a:solidFill/a:srgbClr", NS)
            if outline is not None:
                outlines[name] = outline.get("val", "")
    return {
        "positions_emu": positions,
        "fills": colours,
        "outlines": outlines,
        "wrapped_text": texts,
    }


def main() -> None:
    write_editable_pptx()
    render_static()
    caption = (
        "Figure 6. Forecast routes for gauge-referenced point accumulation. "
        "Blue denotes radar inputs and encoding, green the field-first route, "
        "orange issuance-time gauge context and query conditioning, red the "
        "Direct R2P route, blue-gray the attention and decoder modules, and "
        "white boxes with purple dashed outlines historical gauge targets; "
        "dashed green denotes future-HSR supervision of exPreCast. Solid "
        "connectors show forward computation."
    )
    CAPTION.write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "schema": "figure6-jh2-final-v1",
        "status": "complete",
        "source_pptx": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "source_modified": False,
        "layout_adjustment": {
            "shift_emu": LAYOUT_SHIFT_EMU,
            "shift_inches": LAYOUT_SHIFT,
            "direct_row_translated": True,
            "panel_a_height_expanded": True,
            "panel_b_translated": True,
            "slide_height_expanded": True,
        },
        "historical_supervision_palette": {
            "fill": HISTORICAL_FILL,
            "edge_and_arrow": HISTORICAL_EDGE,
        },
        "direct_module_palette": {
            "fill": MODULE_FILL,
            "edge_and_arrow": MODULE_EDGE,
        },
        "direct_r2p_palette": {"fill": DIRECT_RED_FILL, "edge_and_arrow": DIRECT_RED},
        "output_palette": {"fill": PANEL_FILL, "edge": OUTPUT_EDGE},
        "pptx_verification": verify_pptx(),
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (PPTX, PNG, PDF, CAPTION)
        },
        "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                   "sha256": sha256(Path(__file__).resolve())},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(PPTX)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
