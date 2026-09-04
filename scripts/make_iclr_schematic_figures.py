#!/usr/bin/env python3
"""Render the two schematic figures used by the ICLR-style manuscript."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # The optional QA helper is supplied through PYTHONPATH.
    require_matplotlib_panel_alignment = None


NAVY = "#123B63"
BLUE = "#1F5A92"
MID_BLUE = "#4F8DC9"
CYAN = "#63A7C7"
PALE_BLUE = "#DCEAF7"
VERY_PALE_BLUE = "#F4F8FC"
INK = "#243442"
MID_GREY = "#687785"
LIGHT_GREY = "#D9E0E6"
VERY_LIGHT_GREY = "#F3F5F7"
WARNING = "#C97856"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": WHITE,
        }
    )


def prepare_axis(ax: Axes, panel: str, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_gid(panel)
    ax.text(
        -0.01,
        1.015,
        panel,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.06,
        1.015,
        title,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def rounded_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = VERY_PALE_BLUE,
    edgecolor: str = MID_BLUE,
    textcolor: str = INK,
    fontsize: float = 6.6,
    linewidth: float = 0.9,
    radius: float = 0.025,
    fontweight: str = "normal",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=fontweight,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID_BLUE,
    linewidth: float = 1.15,
    style: str = "-|>",
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=1,
    )


def draw_person(ax: Axes, x: float, y: float, scale: float, color: str = MID_BLUE) -> None:
    ax.add_patch(Circle((x, y + 0.047 * scale), 0.025 * scale, fc=WHITE, ec=color, lw=0.9))
    ax.plot(
        [x, x],
        [y + 0.020 * scale, y - 0.040 * scale],
        color=color,
        lw=1.0,
        solid_capstyle="round",
    )
    ax.plot(
        [x - 0.038 * scale, x, x + 0.038 * scale],
        [y - 0.005 * scale, y + 0.008 * scale, y - 0.005 * scale],
        color=color,
        lw=1.0,
        solid_capstyle="round",
    )
    ax.plot(
        [x - 0.030 * scale, x, x + 0.030 * scale],
        [y - 0.092 * scale, y - 0.040 * scale, y - 0.092 * scale],
        color=color,
        lw=1.0,
        solid_capstyle="round",
    )


def draw_head(
    ax: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    *,
    dense: bool,
) -> None:
    x, y = center
    ax.add_patch(Ellipse((x, y), width, height, fc=WHITE, ec=NAVY, lw=1.15, zorder=2))
    ax.add_patch(
        Arc(
            (x, y + height * 0.01),
            width * 0.42,
            height * 0.21,
            theta1=205,
            theta2=335,
            color=MID_GREY,
            lw=0.75,
        )
    )
    ax.plot([x, x], [y + height * 0.39, y + height * 0.47], color=NAVY, lw=0.9)
    ax.plot(
        [x - width * 0.04, x, x + width * 0.04],
        [y + height * 0.41, y + height * 0.47, y + height * 0.41],
        color=NAVY,
        lw=0.9,
    )
    if dense:
        angles = np.linspace(0, 2 * math.pi, 18, endpoint=False)
        for angle_value in angles:
            px = x + width * 0.39 * math.cos(angle_value)
            py = y + height * 0.38 * math.sin(angle_value)
            ax.add_patch(
                Circle((px, py), width * 0.013, fc=LIGHT_GREY, ec=MID_GREY, lw=0.45, zorder=3)
            )
    fp_y = y + height * 0.25
    for label, px in (("Fp1", x - width * 0.16), ("Fp2", x + width * 0.16)):
        ax.add_patch(Circle((px, fp_y), width * 0.034, fc=MID_BLUE, ec=WHITE, lw=0.8, zorder=4))
        ax.text(
            px,
            fp_y - height * 0.11,
            label,
            ha="center",
            va="top",
            fontsize=6.1,
            color=NAVY,
            fontweight="bold",
        )


def draw_waveform(ax: Axes, x0: float, x1: float, y: float, amp: float, color: str) -> None:
    t = np.linspace(0, 1, 180)
    signal = (
        0.53 * np.sin(2 * np.pi * 5.2 * t)
        + 0.27 * np.sin(2 * np.pi * 11.0 * t + 0.5)
        + 0.12 * np.sin(2 * np.pi * 19.0 * t + 1.1)
    )
    ax.plot(x0 + (x1 - x0) * t, y + amp * signal, color=color, lw=0.85, clip_on=False)


def figure_one() -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.90)

    ax = axes[0]
    prepare_axis(ax, "a", "Laboratory evidence")
    draw_head(ax, (0.50, 0.62), 0.43, 0.43, dense=True)
    for px in (0.16, 0.31, 0.69, 0.84):
        draw_person(ax, px, 0.34, 0.82, color=CYAN)
    ax.text(
        0.50,
        0.24,
        "Multi-participant public EEG",
        ha="center",
        va="center",
        fontsize=6.7,
        color=INK,
    )
    arrow(ax, (0.50, 0.20), (0.50, 0.13), color=MID_BLUE)
    rounded_box(
        ax,
        (0.11, 0.035),
        0.78,
        0.075,
        "Retain Fp1/Fp2 before training",
        facecolor=PALE_BLUE,
        edgecolor=BLUE,
        fontsize=6.5,
        fontweight="bold",
    )

    ax = axes[1]
    prepare_axis(ax, "b", "Wearable constraints")
    draw_head(ax, (0.50, 0.68), 0.36, 0.34, dense=False)
    rounded_box(
        ax,
        (0.05, 0.39),
        0.41,
        0.105,
        "Only two\nfrontal channels",
        facecolor=PALE_BLUE,
        fontsize=6.3,
    )
    rounded_box(
        ax, (0.54, 0.39), 0.41, 0.105, "Previously unseen\nusers", facecolor=PALE_BLUE, fontsize=6.3
    )
    rounded_box(
        ax,
        (0.05, 0.19),
        0.41,
        0.105,
        "Blink and motion\ncontamination",
        edgecolor=WARNING,
        facecolor="#FAEEE8",
        fontsize=6.3,
    )
    rounded_box(
        ax,
        (0.54, 0.19),
        0.41,
        0.105,
        "Reference and\nhardware shift",
        edgecolor=WARNING,
        facecolor="#FAEEE8",
        fontsize=6.3,
    )
    ax.text(
        0.50,
        0.075,
        "Convenience does not guarantee transferability",
        ha="center",
        va="center",
        fontsize=6.5,
        color=NAVY,
        fontweight="bold",
    )

    ax = axes[2]
    prepare_axis(ax, "c", "Question and evidence")
    rounded_box(
        ax,
        (0.05, 0.77),
        0.90,
        0.13,
        "Can a two-channel model generalize\nto a new participant?",
        facecolor=PALE_BLUE,
        edgecolor=BLUE,
        fontsize=6.8,
        fontweight="bold",
    )
    rounded_box(ax, (0.09, 0.59), 0.37, 0.09, "Zero calibration?", fontsize=6.3)
    rounded_box(ax, (0.54, 0.59), 0.37, 0.09, "Low-shot update?", fontsize=6.3)
    arrow(ax, (0.50, 0.77), (0.50, 0.70), color=MID_BLUE)
    ax.text(
        0.50,
        0.50,
        "Evidence ladder",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MID_GREY,
        fontweight="bold",
    )
    evidence = [
        (0.37, "COG-BCI MATB\n26-fold strict LOSO", BLUE),
        (0.22, "OpenNeuro ds007169\nExternal replication", MID_BLUE),
        (0.07, "Wearable session\nDevice diagnostic", CYAN),
    ]
    for index, (py, label, color) in enumerate(evidence):
        rounded_box(
            ax, (0.16, py), 0.68, 0.095, label, facecolor=WHITE, edgecolor=color, fontsize=6.2
        )
        if index < len(evidence) - 1:
            arrow(ax, (0.50, py), (0.50, py - 0.045), color=LIGHT_GREY, linewidth=0.9)

    return fig


def draw_stage_box(
    ax: Axes,
    center_x: float,
    width: float,
    title: str,
    detail: str,
    shape: str,
    color: str,
) -> None:
    x0 = center_x - width / 2
    rounded_box(
        ax,
        (x0, 0.31),
        width,
        0.45,
        "",
        facecolor=color,
        edgecolor=BLUE,
        linewidth=0.85,
        radius=0.018,
    )
    ax.text(
        center_x, 0.64, title, ha="center", va="center", fontsize=6.4, color=NAVY, fontweight="bold"
    )
    ax.text(
        center_x, 0.49, detail, ha="center", va="center", fontsize=5.9, color=INK, linespacing=1.12
    )
    ax.text(center_x, 0.36, shape, ha="center", va="center", fontsize=5.7, color=MID_GREY)


def figure_two() -> Figure:
    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(3, 1, height_ratios=(0.95, 1.75, 1.35), hspace=0.23)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.955)

    ax = axes[0]
    prepare_axis(ax, "a", "Signal preparation")
    stages = [
        ("Fp1/Fp2", "Raw EEG"),
        ("1–40 Hz", "Band-pass"),
        ("250 Hz", "Resample"),
        ("4 s / 2 s", "Window / stride"),
        ("Per channel", "Window z-score"),
        ("2 × 1000", "Model input"),
    ]
    centers = np.linspace(0.075, 0.925, len(stages))
    width = 0.125
    for index, ((top, bottom), center) in enumerate(zip(stages, centers, strict=True)):
        rounded_box(
            ax,
            (center - width / 2, 0.30),
            width,
            0.43,
            "",
            facecolor=VERY_PALE_BLUE,
            edgecolor=MID_BLUE,
            radius=0.016,
        )
        if index == 0:
            draw_waveform(ax, center - 0.045, center + 0.045, 0.56, 0.055, BLUE)
        else:
            ax.text(
                center,
                0.58,
                top,
                ha="center",
                va="center",
                fontsize=6.4,
                color=NAVY,
                fontweight="bold",
            )
        ax.text(center, 0.39, bottom, ha="center", va="center", fontsize=5.9, color=INK)
        if index < len(stages) - 1:
            arrow(
                ax,
                (center + width / 2 + 0.006, 0.515),
                (centers[index + 1] - width / 2 - 0.006, 0.515),
                color=CYAN,
                linewidth=1.0,
            )

    ax = axes[1]
    prepare_axis(ax, "b", "FrontEEGNet architecture")
    widths = [0.11, 0.13, 0.14, 0.11, 0.16, 0.13, 0.13]
    gap = 0.012
    centers: list[float] = []
    cursor = 0.009
    for width_value in widths:
        centers.append(cursor + width_value / 2)
        cursor += width_value + gap
    specs = [
        ("Input", "Fp1/Fp2", "B × 1 × 2 × 1000", VERY_LIGHT_GREY),
        ("Temporal", "Conv 1 × 125\n8 filters + BN", "B × 8 × 2 × 1000", PALE_BLUE),
        ("Spatial", "Depthwise 2 × 1\nD = 2 + BN + ELU", "B × 16 × 1 × 1000", "#CFE2F3"),
        ("Pool 1", "AvgPool 1 × 4\nDropout 0.5", "B × 16 × 1 × 250", VERY_PALE_BLUE),
        ("Separable", "Depthwise 1 × 31\nPointwise 1 × 1\nBN + ELU", "B × 16 × 1 × 250", "#BFD8EE"),
        ("Readout", "AvgPool 1 × 8\nDropout + GAP", "B × 16", VERY_PALE_BLUE),
        ("Classifier", "Linear 16 → 3", "Easy / Medium /\nDifficult", "#A9CBE7"),
    ]
    for index, (center, width_value, spec) in enumerate(zip(centers, widths, specs, strict=True)):
        title, detail, shape, color = spec
        draw_stage_box(ax, center, width_value, title, detail, shape, color)
        if index < len(specs) - 1:
            arrow(
                ax,
                (center + width_value / 2 + 0.006, 0.535),
                (centers[index + 1] - widths[index + 1] / 2 - 0.006, 0.535),
                color=MID_BLUE,
                linewidth=1.0,
            )
    ax.text(
        0.50,
        0.13,
        "1,915 trainable parameters  •  compact EEGNet configuration for two frontal channels",
        ha="center",
        va="center",
        fontsize=6.4,
        color=NAVY,
        fontweight="bold",
    )

    ax = axes[2]
    prepare_axis(ax, "c", "Target-adaptation scopes")
    method_labels = ["Source-only", "Head-only", "Partial", "Full"]
    parameter_labels = ["0", "51", "803", "1,915"]
    component_labels = ["Temporal", "Spatial", "Separable", "Head"]
    component_centers = [0.38, 0.54, 0.70, 0.86]
    row_centers = [0.72, 0.54, 0.36, 0.18]
    updated = {
        "Source-only": set(),
        "Head-only": {"Head"},
        "Partial": {"Separable", "Head"},
        "Full": set(component_labels),
    }
    for component, center in zip(component_labels, component_centers, strict=True):
        ax.text(
            center,
            0.88,
            component,
            ha="center",
            va="center",
            fontsize=6.1,
            color=NAVY,
            fontweight="bold",
        )
    ax.text(0.08, 0.88, "Mode", ha="left", va="center", fontsize=6.1, color=NAVY, fontweight="bold")
    ax.text(
        0.25,
        0.88,
        "Updated\nparameters",
        ha="center",
        va="center",
        fontsize=6.1,
        color=NAVY,
        fontweight="bold",
    )
    for method, params, py in zip(method_labels, parameter_labels, row_centers, strict=True):
        ax.text(
            0.08,
            py,
            method,
            ha="left",
            va="center",
            fontsize=6.3,
            color=INK,
            fontweight="bold" if method == "Source-only" else "normal",
        )
        ax.text(0.25, py, params, ha="center", va="center", fontsize=6.3, color=MID_GREY)
        for component, center in zip(component_labels, component_centers, strict=True):
            is_updated = component in updated[method]
            rounded_box(
                ax,
                (center - 0.058, py - 0.043),
                0.116,
                0.086,
                "Update" if is_updated else "Frozen",
                facecolor=MID_BLUE if is_updated else VERY_LIGHT_GREY,
                edgecolor=BLUE if is_updated else LIGHT_GREY,
                textcolor=WHITE if is_updated else MID_GREY,
                fontsize=5.7,
                linewidth=0.8,
                radius=0.014,
            )
    ax.text(
        0.50,
        0.045,
        "BN statistics remain fixed in Head-only and Partial",
        ha="center",
        va="center",
        fontsize=5.8,
        color=MID_GREY,
    )

    return fig


def run_alignment_qa(fig: Figure, stem: Path, panel_ids: list[str]) -> None:
    if require_matplotlib_panel_alignment is None:
        raise RuntimeError(
            "Panel-alignment QA is unavailable. Add the nature-figure scripts directory "
            "to PYTHONPATH before rendering."
        )
    require_matplotlib_panel_alignment(
        fig,
        axes=fig.axes,
        panel_ids=panel_ids,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def export_figure(fig: Figure, stem: Path, panel_ids: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    run_alignment_qa(fig, stem, panel_ids)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    output = Path(args.output_directory)
    export_figure(figure_one(), output / "figure1_motivation", ["a", "b", "c"])
    export_figure(figure_two(), output / "figure2_fronteegnet", ["a", "b", "c"])


if __name__ == "__main__":
    main()
