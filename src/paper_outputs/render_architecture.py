"""Render the final Figure 6 architecture directly with Matplotlib.

The fixed box coordinates, labels, palette and connectors are the static
manuscript geometry. Rendering needs no Office source files or processing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
# outline gives the historical supervision a clear visual role.
HISTORICAL_FILL = WHITE
HISTORICAL_EDGE = "#7B5EA7"

# Fixed vertical clearance in inches, preserving the final panel geometry.
LAYOUT_SHIFT = 240_000 / 914_400
STATIC_HEIGHT = 10.60 + LAYOUT_SHIFT


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


def _render_static(output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    PNG = output / "figure6_route_architecture.png"
    PDF = output / "figure6_route_architecture.pdf"
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
                edge=FIELD_EDGE, fontsize=10.2)
    arrow(ax, [(6.523, 2.480), (6.523, 2.243)], color=FIELD_EDGE)

    rounded_box(ax, 10.460, 2.638, 1.575, .787, "Historical gauge\ntargets",
                fill=HISTORICAL_FILL, edge=HISTORICAL_EDGE, fontsize=10.5)
    rounded_box(ax, 12.822, 2.638, 1.575, .787, "Issuance-time\ngauge context",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE, fontsize=10.5)
    # Preserve the symmetric, visibly long elbow connectors.
    arrow(ax, [(11.2475, 2.638), (11.2475, 2.4405), (12.3105, 2.4405),
               (12.3105, 2.243)], color=HISTORICAL_EDGE,
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
                fill=HISTORICAL_FILL, edge=HISTORICAL_EDGE, fontsize=10.5)
    rounded_box(ax, 9.870, 4.606 + LAYOUT_SHIFT, 1.575, .787,
                "Issuance-time\ngauge context",
                fill=CONTEXT_FILL, edge=CONTEXT_EDGE, fontsize=10.5)
    arrow(ax, shifted_y([(8.2945, 4.606), (8.2945, 4.409),
                         (9.3585, 4.409), (9.3585, 4.212)]),
          color=HISTORICAL_EDGE,
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
                "Gauge RN60",
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
    return PNG, PDF


def render(output: Path) -> tuple[Path, Path]:
    """Write Figure 6 PNG/PDF independently of other figures' style settings."""
    # The former separate process used Matplotlib defaults. Preserve those
    # settings even when called after other publication figures in-process.
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        return _render_static(Path(output))
