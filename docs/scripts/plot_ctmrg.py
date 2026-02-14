"""Generate tensor diagrams for the CTMRG section."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tn_draw import new_fig, draw_tensor, draw_bond, draw_label, save

TW, TH = 0.8, 0.6
SP = 1.6  # grid spacing

# Coordinate subscript shorthands (reused across all CTMRG diagrams)
_xy   = "_{(x,y)}"
_x1y  = "_{(x\\!+\\!1,y)}"
_xy1  = "_{(x,y\\!-\\!1)}"
_x1y1 = "_{(x\\!+\\!1,y\\!-\\!1)}"


def _draw_env_grid(ax, labels, rows, cols, x0=0, y0=0,
                   row_doubles=None, col_doubles=None,
                   tw=TW, th=TH, sp=SP, fontsize=11):
    """Draw a grid of tensors with bonds.

    row_doubles: set of row indices where horizontal bonds are double.
    col_doubles: set of col indices where vertical bonds are double.
    """
    if row_doubles is None:
        row_doubles = set()
    if col_doubles is None:
        col_doubles = set()

    centres = {}
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * sp
            y = y0 - r * sp
            centres[(r, c)] = (x, y)
            draw_tensor(ax, x, y, tw, th, labels[r][c], fontsize=fontsize)

    # Horizontal bonds
    for r in range(rows):
        for c in range(cols - 1):
            x0b = centres[(r, c)][0] + tw / 2
            x1b = centres[(r, c + 1)][0] - tw / 2
            yb = centres[(r, c)][1]
            draw_bond(ax, (x0b, yb), (x1b, yb), double=(r in row_doubles))

    # Vertical bonds
    for r in range(rows - 1):
        for c in range(cols):
            yb0 = centres[(r, c)][1] - th / 2
            yb1 = centres[(r + 1, c)][1] + th / 2
            xb = centres[(r, c)][0]
            draw_bond(ax, (xb, yb0), (xb, yb1), double=(c in col_doubles))

    return centres


def ctmrg_up_move():
    """Full environment for an up-move."""
    fig, ax = new_fig(7, 9)
    sp, tw, th, fs = 1.8, 0.85, 0.55, 10
    labels = [
        [f"$C^1{_xy}$",  f"$E^1{_xy}$",  f"$C^2{_xy}$"],
        [f"$E^4{_xy}$",  f"$a'{_xy}$",   f"$E^2{_xy}$"],
        [f"$E^4{_xy1}$", f"$a{_xy1}$",   f"$E^2{_xy1}$"],
        [f"$C^4{_xy1}$", f"$E^3{_xy1}$", f"$C^3{_xy1}$"],
    ]
    _draw_env_grid(ax, labels, 4, 3, x0=-sp, y0=sp*1.5,
                   row_doubles={1, 2}, col_doubles={1},
                   tw=tw, th=th, sp=sp, fontsize=fs)

    ax.set_xlim(-sp - tw - 0.3, sp + tw + 0.5)
    ax.set_ylim(-sp*2 - th, sp*2 + th)
    save(fig, "ctmrg_up_move.png")


def ctmrg_absorption():
    """Absorption: before (2 rows) ==> after (1 row)."""
    fig, ax = new_fig(12, 4.5)
    sp, tw, th, fs = 1.8, 0.85, 0.55, 10

    # ── Left: before ──
    lx = -3.5
    labels_before = [
        [f"$C^1{_xy}$",  f"$E^1{_xy}$",  f"$C^2{_xy}$"],
        [f"$E^4{_xy}$",  f"$a'{_xy}$",   f"$E^2{_xy}$"],
    ]
    _draw_env_grid(ax, labels_before, 2, 3, x0=lx - sp, y0=sp*0.5,
                   row_doubles={1}, col_doubles={1},
                   tw=tw, th=th, sp=sp, fontsize=fs)

    # ── Arrow ──
    ax.annotate("", xy=(1.2, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # ── Right: after ──
    rx = 3.5
    labels_after = [
        [f"$C'^1{_xy1}$", f"$E'^1{_xy1}$", f"$C'^2{_xy1}$"],
    ]
    _draw_env_grid(ax, labels_after, 1, 3, x0=rx - sp, y0=0,
                   tw=tw, th=th, sp=sp, fontsize=fs)

    ax.set_xlim(lx - sp - tw - 0.3, rx + sp + tw + 0.3)
    ax.set_ylim(-sp - th, sp + th)
    save(fig, "ctmrg_absorption.png")


def ctmrg_quarter_tensors():
    """Quarter tensors Q1, Q2, Q3, Q4 with coordinate subscripts and stubs."""
    fig, ax = new_fig(10, 7)
    sp = 1.6
    tw, th = 0.85, 0.5
    fs = 10
    stub = 0.3  # external stub length

    def draw_quarter(ax, x0, y0, labels_q, title, row_d, col_d, stubs):
        """Draw one 2x2 quarter with grid bonds and external stubs.

        stubs: list of (row, col, direction, double) for external bond stubs.
        """
        centres = _draw_env_grid(ax, labels_q, 2, 2, x0=x0, y0=y0,
                                 row_doubles=row_d, col_doubles=col_d,
                                 tw=tw, th=th, sp=sp, fontsize=fs)
        # title above
        cx = x0 + sp / 2
        cy = y0 + th / 2 + 0.35
        draw_label(ax, cx, cy, title, fontsize=13)
        # external stubs
        for r, c, direction, dbl in stubs:
            sx, sy = centres[(r, c)]
            if direction == "right":
                draw_bond(ax, (sx + tw/2, sy), (sx + tw/2 + stub, sy), double=dbl)
            elif direction == "left":
                draw_bond(ax, (sx - tw/2, sy), (sx - tw/2 - stub, sy), double=dbl)
            elif direction == "up":
                draw_bond(ax, (sx, sy + th/2), (sx, sy + th/2 + stub), double=dbl)
            elif direction == "down":
                draw_bond(ax, (sx, sy - th/2), (sx, sy - th/2 - stub), double=dbl)

    # Q1 (top-left): C1, E1 / E4, a  at (x,y)
    draw_quarter(ax, -4.5, 1.8,
                 [[f"$C^1{_xy}$", f"$E^1{_xy}$"],
                  [f"$E^4{_xy}$", f"$a{_xy}$"]],
                 "$Q_1$", row_d={1}, col_d={1},
                 stubs=[(0, 1, "right", False),    # E^1 → χ
                        (1, 0, "down",  False),    # E^4 → χ
                        (1, 1, "right", True),     # a → D²
                        (1, 1, "down",  True)])    # a → D²

    # Q2 (top-right): E1, C2 / a, E2  at (x+1,y)
    draw_quarter(ax, -0.5, 1.8,
                 [[f"$E^1{_x1y}$", f"$C^2{_x1y}$"],
                  [f"$a{_x1y}$",   f"$E^2{_x1y}$"]],
                 "$Q_2$", row_d={1}, col_d={0},
                 stubs=[(0, 0, "left",  False),    # E^1 → χ
                        (1, 1, "down",  False),    # E^2 → χ
                        (1, 0, "left",  True),     # a → D²
                        (1, 0, "down",  True)])    # a → D²

    # Q4 (bottom-left): E4, a / C4, E3  at (x,y-1)
    draw_quarter(ax, -4.5, -1.8,
                 [[f"$E^4{_xy1}$", f"$a{_xy1}$"],
                  [f"$C^4{_xy1}$", f"$E^3{_xy1}$"]],
                 "$Q_4$", row_d={0}, col_d={1},
                 stubs=[(0, 0, "up",    False),    # E^4 → χ
                        (1, 1, "right", False),    # E^3 → χ
                        (0, 1, "right", True),     # a → D²
                        (0, 1, "up",    True)])    # a → D²

    # Q3 (bottom-right): a, E2 / E3, C3  at (x+1,y-1)
    draw_quarter(ax, -0.5, -1.8,
                 [[f"$a{_x1y1}$",   f"$E^2{_x1y1}$"],
                  [f"$E^3{_x1y1}$", f"$C^3{_x1y1}$"]],
                 "$Q_3$", row_d={0}, col_d={0},
                 stubs=[(0, 1, "up",   False),     # E^2 → χ
                        (1, 0, "left", False),     # E^3 → χ
                        (0, 0, "left", True),      # a → D²
                        (0, 0, "up",   True)])     # a → D²

    ax.set_xlim(-5.5, 2.8)
    ax.set_ylim(-4.0, 3.5)
    save(fig, "ctmrg_quarter_tensors.png")


def ctmrg_projector():
    """Full environment split into R1 (left) and R2 (right) with SVD line."""
    fig, ax = new_fig(10, 9)
    sp = 1.7
    tw, th = 0.85, 0.5

    labels = [
        [f"$C^1{_xy}$",  f"$E^1{_xy}$",  f"$E^1{_x1y}$",  f"$C^2{_x1y}$"],
        [f"$E^4{_xy}$",  f"$a{_xy}$",    f"$a{_x1y}$",    f"$E^2{_x1y}$"],
        [f"$E^4{_xy1}$", f"$a{_xy1}$",   f"$a{_x1y1}$",   f"$E^2{_x1y1}$"],
        [f"$C^4{_xy1}$", f"$E^3{_xy1}$", f"$E^3{_x1y1}$", f"$C^3{_x1y1}$"],
    ]
    centres = _draw_env_grid(ax, labels, 4, 4, x0=-1.5*sp, y0=1.5*sp,
                             row_doubles={1, 2}, col_doubles={1, 2},
                             tw=tw, th=th, sp=sp, fontsize=10)

    # Dashed vertical SVD line between columns 1 and 2
    xmid = (centres[(0, 1)][0] + centres[(0, 2)][0]) / 2
    ytop = centres[(0, 0)][1] + th/2 + 0.4
    ybot = centres[(3, 0)][1] - th/2 - 0.4
    ax.plot([xmid, xmid], [ytop, ybot], "--", color="black", linewidth=1.2)

    # R1, R2 labels
    xleft = (centres[(0, 0)][0] + centres[(0, 1)][0]) / 2
    xright = (centres[(0, 2)][0] + centres[(0, 3)][0]) / 2
    draw_label(ax, xleft, ybot - 0.35, "$R_1$", fontsize=13)
    draw_label(ax, xright, ybot - 0.35, "$R_2$", fontsize=13)

    ax.set_xlim(-sp*2 - tw, sp*2 + tw)
    ax.set_ylim(ybot - 0.8, ytop + 0.3)
    save(fig, "ctmrg_projector.png")


if __name__ == "__main__":
    ctmrg_up_move()
    ctmrg_absorption()
    ctmrg_quarter_tensors()
    ctmrg_projector()
    print("plot_ctmrg.py: done")
