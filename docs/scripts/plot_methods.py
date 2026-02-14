"""Generate tensor diagrams for the Methods (iPEPS) section."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tn_draw import new_fig, draw_tensor, draw_bond, draw_label, save

# ── Tensor sizes ──────────────────────────────────────────────
TW, TH = 0.9, 0.7       # site tensor
SW, SH = 0.5, 0.5       # small tensor (corner, double-layer)
LEG = 0.6                # bond stub length


def site_tensor():
    """Rank-5 site tensor A_i."""
    fig, ax = new_fig(4, 4)
    cx, cy = 0, 0
    draw_tensor(ax, cx, cy, TW, TH, r"$A_i$")

    # bond stubs: l, u, r, d
    draw_bond(ax, (cx - TW/2, cy), (cx - TW/2 - LEG, cy))
    draw_label(ax, cx - TW/2 - LEG - 0.15, cy, r"$l$", fontsize=12)

    draw_bond(ax, (cx, cy + TH/2), (cx, cy + TH/2 + LEG))
    draw_label(ax, cx, cy + TH/2 + LEG + 0.15, r"$u$", fontsize=12)

    draw_bond(ax, (cx + TW/2, cy), (cx + TW/2 + LEG, cy))
    draw_label(ax, cx + TW/2 + LEG + 0.15, cy, r"$r$", fontsize=12)

    draw_bond(ax, (cx, cy - TH/2), (cx, cy - TH/2 - LEG))
    draw_label(ax, cx, cy - TH/2 - LEG - 0.15, r"$d$", fontsize=12)

    # physical index stub (angled 45 degrees to avoid overlap with d)
    px, py = cx + TW/4, cy - TH/2
    draw_bond(ax, (px, py), (px + LEG * 0.5, py - LEG * 0.5))
    draw_label(ax, px + LEG * 0.5 + 0.12, py - LEG * 0.5 - 0.12, r"$s$", fontsize=12)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    save(fig, "site_tensor.png")


def double_layer():
    """Double-layer tensor: A_i contracted with A*_i."""
    fig, ax = new_fig(8, 4.5)

    h_off = 0.3     # horizontal offset for perspective
    th08 = TH * 0.8

    # ── Left: ket / bra ──
    lx = -2.0
    ky, by = 0.7, -0.7

    # Ket layer (shifted right)
    kx = lx + h_off
    draw_tensor(ax, kx, ky, TW, th08, r"$A_i$")
    draw_bond(ax, (kx - TW/2, ky), (kx - TW/2 - LEG * 0.7, ky))
    draw_label(ax, kx - TW/2 - LEG * 0.7 - 0.15, ky, r"$l$", fontsize=10)
    draw_bond(ax, (kx + TW/2, ky), (kx + TW/2 + LEG * 0.7, ky))
    draw_label(ax, kx + TW/2 + LEG * 0.7 + 0.15, ky, r"$r$", fontsize=10)
    draw_bond(ax, (kx, ky + th08/2), (kx, ky + th08/2 + LEG * 0.5))
    draw_label(ax, kx + 0.15, ky + th08/2 + LEG * 0.5 + 0.1, r"$u$", fontsize=10)
    draw_bond(ax, (kx, ky - th08/2), (kx, ky - th08/2 - LEG * 0.4))
    draw_label(ax, kx + 0.15, ky - th08/2 - LEG * 0.4 - 0.1, r"$d$", fontsize=10)

    # Bra layer (shifted left)
    bx = lx - h_off
    draw_tensor(ax, bx, by, TW, th08, r"$A^*_i$", fontsize=10)
    draw_bond(ax, (bx - TW/2, by), (bx - TW/2 - LEG * 0.7, by))
    draw_label(ax, bx - TW/2 - LEG * 0.7 - 0.18, by, r"$l'$", fontsize=10)
    draw_bond(ax, (bx + TW/2, by), (bx + TW/2 + LEG * 0.7, by))
    draw_label(ax, bx + TW/2 + LEG * 0.7 + 0.18, by, r"$r'$", fontsize=10)
    draw_bond(ax, (bx, by - th08/2), (bx, by - th08/2 - LEG * 0.5))
    draw_label(ax, bx + 0.18, by - th08/2 - LEG * 0.5 - 0.1, r"$d'$", fontsize=10)
    draw_bond(ax, (bx, by + th08/2), (bx, by + th08/2 + LEG * 0.4))
    draw_label(ax, bx + 0.18, by + th08/2 + LEG * 0.4 + 0.1, r"$u'$", fontsize=10)

    # Physical contraction (angled line between ket and bra)
    s_kx = kx - TW/4          # depart from left quarter of ket bottom
    s_bx = bx + TW/4          # arrive at right quarter of bra top
    draw_bond(ax, (s_kx, ky - th08/2), (s_bx, by + th08/2))
    mid_sx = (s_kx + s_bx) / 2
    mid_sy = (ky - th08/2 + by + th08/2) / 2
    draw_label(ax, mid_sx + 0.2, mid_sy, r"$s_i$", fontsize=9)

    # ── Equals sign ──
    draw_label(ax, 0.4, 0, r"$=$", fontsize=18)

    # ── Right: double-layer tensor ──
    rx = 2.8
    dw, dh = 0.9, 0.7
    draw_tensor(ax, rx, 0, dw, dh, r"$a_i$")
    draw_bond(ax, (rx - dw/2, 0), (rx - dw/2 - LEG * 0.7, 0), double=True)
    draw_bond(ax, (rx + dw/2, 0), (rx + dw/2 + LEG * 0.7, 0), double=True)
    draw_bond(ax, (rx, dh/2), (rx, dh/2 + LEG * 0.5), double=True)
    draw_bond(ax, (rx, -dh/2), (rx, -dh/2 - LEG * 0.5), double=True)

    ax.set_xlim(-4.2, 4.8)
    ax.set_ylim(-2.2, 2.2)
    save(fig, "double_layer.png")


def boundary_environment():
    """3x3 CTM environment grid."""
    fig, ax = new_fig(7, 7)
    sp = 1.8  # spacing
    tw, th = 0.85, 0.55
    fs = 10

    xy = "_{(x,y)}"
    labels = [
        [f"$C^1{xy}$", f"$E^1{xy}$", f"$C^2{xy}$"],
        [f"$E^4{xy}$", f"$a{xy}$",   f"$E^2{xy}$"],
        [f"$C^4{xy}$", f"$E^3{xy}$", f"$C^3{xy}$"],
    ]

    for row in range(3):
        for col in range(3):
            x = (col - 1) * sp
            y = (1 - row) * sp
            draw_tensor(ax, x, y, tw, th, labels[row][col], fontsize=fs)

    # Horizontal bonds
    for row in range(3):
        for col in range(2):
            x0 = (col - 1) * sp + tw / 2
            x1 = (col) * sp - tw / 2
            y = (1 - row) * sp
            is_double = (row == 1)  # middle row has D^2 bonds
            draw_bond(ax, (x0, y), (x1, y), double=is_double)

    # Vertical bonds
    for row in range(2):
        for col in range(3):
            x = (col - 1) * sp
            y0 = (1 - row) * sp - th / 2
            y1 = (-row) * sp + th / 2
            is_double = (col == 1)  # middle column has D^2 bonds
            draw_bond(ax, (x, y0), (x, y1), double=is_double)

    ax.set_xlim(-sp * 1.4, sp * 1.4)
    ax.set_ylim(-sp * 1.5, sp * 1.5)
    save(fig, "boundary_environment.png")


if __name__ == "__main__":
    site_tensor()
    double_layer()
    boundary_environment()
    print("plot_methods.py: done")
