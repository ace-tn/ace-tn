"""Shared drawing primitives for tensor network diagrams.

All diagrams use black-and-white rectangles with 90-degree bond lines.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "source", "png")


def new_fig(width=6, height=4):
    """Create a new figure with white background and equal aspect."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def draw_tensor(ax, x, y, w, h, label, fontsize=11, lw=1.5, zorder=3):
    """Draw a labeled rectangle tensor at centre (x, y)."""
    rect = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="square,pad=0",
        edgecolor="black", facecolor="white", linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontfamily="serif", math_fontfamily="cm", zorder=zorder + 1)


def draw_bond(ax, start, end, double=False, lw=1.5, dashed=False, zorder=2):
    """Draw a bond line from start (x,y) to end (x,y).

    If double=True, draw two parallel lines to indicate a D^2 bond.
    Use *zorder* to control layering relative to tensor boxes (zorder 3).
    """
    x0, y0 = start
    x1, y1 = end
    style = ":" if dashed else "-"

    if not double:
        ax.plot([x0, x1], [y0, y1], style, color="black", linewidth=lw,
                solid_capstyle="butt", zorder=zorder)
    else:
        # Offset perpendicular to the bond direction
        dx, dy = x1 - x0, y1 - y0
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return
        nx, ny = -dy / length, dx / length
        off = 0.04
        ax.plot([x0 + nx * off, x1 + nx * off],
                [y0 + ny * off, y1 + ny * off],
                style, color="black", linewidth=lw, solid_capstyle="butt",
                zorder=zorder)
        ax.plot([x0 - nx * off, x1 - nx * off],
                [y0 - ny * off, y1 - ny * off],
                style, color="black", linewidth=lw, solid_capstyle="butt",
                zorder=zorder)


def draw_label(ax, x, y, text, fontsize=10, ha="center", va="center"):
    """Place a text label."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontfamily="serif", math_fontfamily="cm")


def save(fig, name, tight=True):
    """Save figure to OUTPUT_DIR/name."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight" if tight else None,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {path}")
