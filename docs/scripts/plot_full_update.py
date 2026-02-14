"""Generate tensor diagrams for the Full Update section."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tn_draw import new_fig, draw_tensor, draw_bond, draw_label, save

TW, TH = 0.9, 0.7
SW, SH = 0.65, 0.55   # reduced / small tensors
LEG = 0.5


def fu_two_sites():
    """Two site tensors A_i and A_j connected by a bond."""
    fig, ax = new_fig(7, 3.5)
    gap = 1.2  # half distance between tensor centres

    for sign, label in [(-1, "$A_i$"), (1, "$A_j$")]:
        cx = sign * gap
        draw_tensor(ax, cx, 0, TW, TH, label)
        # outer horizontal leg
        draw_bond(ax, (cx + sign * TW/2, 0), (cx + sign * (TW/2 + LEG), 0))
        # vertical legs
        draw_bond(ax, (cx, TH/2), (cx, TH/2 + LEG))
        draw_bond(ax, (cx, -TH/2), (cx, -TH/2 - LEG))
        # physical leg (angled toward the other tensor)
        px = cx - sign * TW/4
        draw_bond(ax, (px, -TH/2), (px - sign * LEG * 0.5, -TH/2 - LEG * 0.5))

    # connecting bond
    draw_bond(ax, (-gap + TW/2, 0), (gap - TW/2, 0))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.0, 1.8)
    save(fig, "fu_two_sites.png")


def fu_qr_decomposition():
    """QR decomposition: A^Q_i, a^R_i --- a^R_j, A^Q_j."""
    fig, ax = new_fig(9, 3.5)

    # Positions (left to right)
    qx_l, rx_l, rx_r, qx_r = -2.8, -1.5, 1.5, 2.8
    y = 0

    # A^Q_i
    draw_tensor(ax, qx_l, y, TW, TH, "$A^Q_i$", fontsize=10)
    draw_bond(ax, (qx_l - TW/2, y), (qx_l - TW/2 - LEG, y))
    draw_bond(ax, (qx_l, TH/2), (qx_l, TH/2 + LEG))
    draw_bond(ax, (qx_l, -TH/2), (qx_l, -TH/2 - LEG))

    # a^R_i
    draw_tensor(ax, rx_l, y, SW, SH, "$a^R_i$", fontsize=10)
    draw_bond(ax, (qx_l + TW/2, y), (rx_l - SW/2, y))
    # physical leg (angled toward centre)
    draw_bond(ax, (rx_l, -SH/2), (rx_l + LEG * 0.4, -SH/2 - LEG * 0.4))

    # a^R_j
    draw_tensor(ax, rx_r, y, SW, SH, "$a^R_j$", fontsize=10)
    draw_bond(ax, (rx_l + SW/2, y), (rx_r - SW/2, y))
    # physical leg (angled toward centre)
    draw_bond(ax, (rx_r, -SH/2), (rx_r - LEG * 0.4, -SH/2 - LEG * 0.4))

    # A^Q_j
    draw_tensor(ax, qx_r, y, TW, TH, "$A^Q_j$", fontsize=10)
    draw_bond(ax, (rx_r + SW/2, y), (qx_r - TW/2, y))
    draw_bond(ax, (qx_r + TW/2, y), (qx_r + TW/2 + LEG, y))
    draw_bond(ax, (qx_r, TH/2), (qx_r, TH/2 + LEG))
    draw_bond(ax, (qx_r, -TH/2), (qx_r, -TH/2 - LEG))

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-2.0, 1.8)
    save(fig, "fu_qr_decomposition.png")


def fu_gate_application():
    """Gate applied to reduced tensors."""
    fig, ax = new_fig(7, 4)

    rx_l, rx_r = -1.2, 1.2
    gy = -1.2
    y = 0

    # a^R_i
    draw_tensor(ax, rx_l, y, SW, SH, "$a^R_i$", fontsize=10)
    draw_bond(ax, (rx_l - SW/2, y), (rx_l - SW/2 - LEG, y))

    # a^R_j
    draw_tensor(ax, rx_r, y, SW, SH, "$a^R_j$", fontsize=10)
    draw_bond(ax, (rx_r + SW/2, y), (rx_r + SW/2 + LEG, y))

    # connecting bond
    draw_bond(ax, (rx_l + SW/2, y), (rx_r - SW/2, y))

    # physical legs down to gate
    draw_bond(ax, (rx_l, -SH/2), (rx_l, gy + SH/2))
    draw_bond(ax, (rx_r, -SH/2), (rx_r, gy + SH/2))

    # gate
    gw = rx_r - rx_l + SW
    draw_tensor(ax, 0, gy, gw, SH, "$g_{ij}$", fontsize=11)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 1.5)
    save(fig, "fu_gate_application.png")


def fu_norm_tensor():
    """Norm tensor N_ij for a horizontal bond.

    From build_norm_tensor (k=0, s1=i left, s2=j right):
      Right half (site i): C^2_i, E^2_i, E^1_i, C^3_i, E^3_i
      Left half  (site j): C^1_j, E^1_j, E^4_j, C^4_j, E^3_j

    Diagram layout (3 rows x 4 cols):
      Row 0: C^2_i  E^1_i  E^1_j  C^1_j
      Row 1: E^2_i  A^Q_i  A^Q_j  E^4_j   (ket up, bra down with gap)
      Row 2: C^3_i  E^3_i  E^3_j  C^4_j

    The connecting bonds between A^Q_i--A^Q_j are OPEN (stubs with a gap),
    giving the norm tensor its rank-4 structure.
    Bonds from E to the bra layer are dashed where they pass through the ket layer.
    """
    fig, ax = new_fig(10, 10)
    sp = 1.8       # grid spacing
    tw, th = 0.75, 0.5
    stw, sth = 0.8, 0.5   # site tensor size
    fs = 9
    ket_off = 0.55   # vertical offset from row centre
    h_off = 0.18     # horizontal perspective offset (ket inward, bra outward)
    stub = 0.25       # length of open-bond stubs

    # -- Grid centres for the 3-row x 4-col boundary --
    def pos(r, c):
        return (-1.5 * sp + c * sp, sp - r * sp)

    # Row 0 (top): corners & top edges
    top_labels = [r"$C^2_i$", r"$E^1_i$", r"$E^1_j$", r"$C^1_j$"]
    top_centres = {}
    for c, lab in enumerate(top_labels):
        x, y = pos(0, c)
        top_centres[c] = (x, y)
        draw_tensor(ax, x, y, tw, th, lab, fontsize=fs)

    # Row 2 (bottom): corners & bottom edges
    bot_labels = [r"$C^3_i$", r"$E^3_i$", r"$E^3_j$", r"$C^4_j$"]
    bot_centres = {}
    for c, lab in enumerate(bot_labels):
        x, y = pos(2, c)
        bot_centres[c] = (x, y)
        draw_tensor(ax, x, y, tw, th, lab, fontsize=fs)

    # Row 1 (middle): side edges
    side_labels = [r"$E^2_i$", r"$E^4_j$"]
    side_cols = [0, 3]
    side_centres = {}
    for c, lab in zip(side_cols, side_labels):
        x, y = pos(1, c)
        side_centres[c] = (x, y)
        draw_tensor(ax, x, y, tw, th, lab, fontsize=fs)

    # Row 1 (middle): ket site tensors (offset up AND inward)
    ket_labels = [r"$A^Q_i$", r"$A^Q_j$"]
    ket_cols = [1, 2]
    ket_centres = {}
    inward = {1: h_off, 2: -h_off}  # col 1 shifts right, col 2 shifts left
    for c, lab in zip(ket_cols, ket_labels):
        x, y_base = pos(1, c)
        x += inward[c]
        y = y_base + ket_off
        ket_centres[c] = (x, y)
        draw_tensor(ax, x, y, stw, sth, lab, fontsize=fs)

    # Row 1 (middle): bra site tensors (offset down AND outward)
    bra_labels = [r"$A^{Q*}_i$", r"$A^{Q*}_j$"]
    bra_centres = {}
    outward = {1: -h_off, 2: h_off}  # col 1 shifts left, col 2 shifts right
    for c, lab in zip(ket_cols, bra_labels):
        x, y_base = pos(1, c)
        x += outward[c]
        y = y_base - ket_off
        bra_centres[c] = (x, y)
        draw_tensor(ax, x, y, stw, sth, lab, fontsize=fs, lw=1.0)

    # ── Horizontal bonds ──
    # Top row
    for c in range(3):
        x0 = top_centres[c][0] + tw/2
        x1 = top_centres[c + 1][0] - tw/2
        yb = top_centres[c][1]
        draw_bond(ax, (x0, yb), (x1, yb))

    # Bottom row
    for c in range(3):
        x0 = bot_centres[c][0] + tw/2
        x1 = bot_centres[c + 1][0] - tw/2
        yb = bot_centres[c][1]
        draw_bond(ax, (x0, yb), (x1, yb))

    # Middle row: side edge → ket and bra (angled lines for perspective)
    # Left side edge to ket_i / bra_i
    ex_r = side_centres[0][0] + tw/2
    ey = side_centres[0][1]
    draw_bond(ax, (ex_r, ey + ket_off * 0.4),
              (ket_centres[1][0] - stw/2, ket_centres[1][1]))
    draw_bond(ax, (ex_r, ey - ket_off * 0.4),
              (bra_centres[1][0] - stw/2, bra_centres[1][1]))

    # Right side edge to ket_j / bra_j
    ex_l = side_centres[3][0] - tw/2
    ey = side_centres[3][1]
    draw_bond(ax, (ex_l, ey + ket_off * 0.4),
              (ket_centres[2][0] + stw/2, ket_centres[2][1]))
    draw_bond(ax, (ex_l, ey - ket_off * 0.4),
              (bra_centres[2][0] + stw/2, bra_centres[2][1]))

    # Ket open bonds: stubs toward centre, NOT connected (rank-4 open indices)
    ky = ket_centres[1][1]
    draw_bond(ax, (ket_centres[1][0] + stw/2, ky),
              (ket_centres[1][0] + stw/2 + stub, ky))
    draw_bond(ax, (ket_centres[2][0] - stw/2, ky),
              (ket_centres[2][0] - stw/2 - stub, ky))

    # Bra open bonds: stubs toward centre, NOT connected
    by = bra_centres[1][1]
    draw_bond(ax, (bra_centres[1][0] + stw/2, by),
              (bra_centres[1][0] + stw/2 + stub, by))
    draw_bond(ax, (bra_centres[2][0] - stw/2, by),
              (bra_centres[2][0] - stw/2 - stub, by))

    # ── Vertical bonds ──
    # Column 0: C^2_i → E^2_i → C^3_i  (chi bonds)
    draw_bond(ax, (top_centres[0][0], top_centres[0][1] - th/2),
              (side_centres[0][0], side_centres[0][1] + th/2))
    draw_bond(ax, (side_centres[0][0], side_centres[0][1] - th/2),
              (bot_centres[0][0], bot_centres[0][1] + th/2))

    # Column 3: C^1_j → E^4_j → C^4_j  (chi bonds)
    draw_bond(ax, (top_centres[3][0], top_centres[3][1] - th/2),
              (side_centres[3][0], side_centres[3][1] + th/2))
    draw_bond(ax, (side_centres[3][0], side_centres[3][1] - th/2),
              (bot_centres[3][0], bot_centres[3][1] + th/2))

    # Columns 1,2: four SEPARATE diagonal bonds per column.
    # E has independent bonds to ket and bra — they are NOT connected.
    #
    # Layering via zorder:
    #   Tensor boxes sit at zorder 3, labels at zorder 4.
    #   E→bra bonds at zorder 2 (default) → ket boxes cover them → "underneath"
    #   ket→E^3 bonds at zorder 5 → they render OVER bra boxes → "passing over"
    #   E→ket and bra→E^3 are direct connections (default zorder).

    for c in ket_cols:
        ex = top_centres[c][0]         # E^1 / E^3 column x (grid centre)
        kx = ket_centres[c][0]         # ket x (shifted inward)
        bx = bra_centres[c][0]         # bra x (shifted outward)
        ket_top = ket_centres[c][1] + sth/2
        ket_bot = ket_centres[c][1] - sth/2
        bra_top = bra_centres[c][1] + sth/2
        bra_bot = bra_centres[c][1] - sth/2
        e_top_bot = top_centres[c][1] - th/2
        e_bot_top = bot_centres[c][1] + th/2

        # Bond 1: E^1 → ket (solid, direct)
        draw_bond(ax, (ex, e_top_bot), (kx, ket_top))

        # Bond 2: E^1 → bra (solid line, hidden behind ket box; dashed through ket)
        draw_bond(ax, (ex, e_top_bot), (bx, bra_top))          # solid at zorder 2
        # Dashed overlay where the line passes through the ket box (zorder 3.5)
        def _lx(x1, y1, x2, y2, y):
            return x1 + (y - y1) / (y2 - y1) * (x2 - x1)
        xkt = _lx(ex, e_top_bot, bx, bra_top, ket_top)
        xkb = _lx(ex, e_top_bot, bx, bra_top, ket_bot)
        draw_bond(ax, (xkt, ket_top), (xkb, ket_bot), dashed=True, zorder=3.5)

        # Bond 3: ket → E^3 (solid straight line, zorder 5 → drawn OVER bra box)
        draw_bond(ax, (kx, ket_bot), (ex, e_bot_top), zorder=5)

        # Bond 4: bra → E^3 (solid, direct)
        draw_bond(ax, (bx, bra_bot), (ex, e_bot_top))

    ax.set_xlim(-2.7 * sp, 2.7 * sp)
    ax.set_ylim(-1.6 * sp, 1.7 * sp)
    save(fig, "fu_norm_tensor.png")


if __name__ == "__main__":
    fu_two_sites()
    fu_qr_decomposition()
    fu_gate_application()
    fu_norm_tensor()
    print("plot_full_update.py: done")
