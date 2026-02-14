"""Generate tensor diagrams for the Simple Update section."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tn_draw import new_fig, draw_tensor, draw_bond, draw_label, save

TW, TH = 0.9, 0.7       # site tensor
SW, SH = 0.65, 0.55      # reduced tensor
LW, LH = 0.35, 0.35      # lambda tensor (small box)
LEG = 0.5


def _draw_lambda(ax, x, y, label=r"$\lambda$"):
    """Draw a small lambda box."""
    draw_tensor(ax, x, y, LW, LH, label, fontsize=8, lw=1.2)


def su_lambda_network():
    """2D view of two sites with lambda boxes on all bonds."""
    fig, ax = new_fig(7, 5)

    gap = 2.0  # distance between site centres
    sites = [(-gap/2, 0), (gap/2, 0)]
    signs = [-1, 1]  # left, right

    for (cx, cy), sign in zip(sites, signs):
        draw_tensor(ax, cx, cy, TW, TH, "$A$")

        # lambda on top
        lam_y = cy + TH/2 + LH/2 + 0.15
        draw_bond(ax, (cx, cy + TH/2), (cx, lam_y - LH/2))
        _draw_lambda(ax, cx, lam_y)
        draw_bond(ax, (cx, lam_y + LH/2), (cx, lam_y + LH/2 + 0.3))

        # lambda on bottom
        lam_y = cy - TH/2 - LH/2 - 0.15
        draw_bond(ax, (cx, cy - TH/2), (cx, lam_y + LH/2))
        _draw_lambda(ax, cx, lam_y)
        draw_bond(ax, (cx, lam_y - LH/2), (cx, lam_y - LH/2 - 0.3))

        # physical leg (angled toward the other tensor)
        px = cx - sign * TW/4
        draw_bond(ax, (px, -TH/2), (px - sign * LEG * 0.4, -TH/2 - LEG * 0.4))

    # lambda on left of left site
    lx = sites[0][0]
    lam_x = lx - TW/2 - LW/2 - 0.15
    draw_bond(ax, (lx - TW/2, 0), (lam_x + LW/2, 0))
    _draw_lambda(ax, lam_x, 0)
    draw_bond(ax, (lam_x - LW/2, 0), (lam_x - LW/2 - 0.3, 0))

    # lambda between sites
    mid_x = 0
    lam_x = mid_x
    draw_bond(ax, (sites[0][0] + TW/2, 0), (lam_x - LW/2, 0))
    _draw_lambda(ax, lam_x, 0)
    draw_bond(ax, (lam_x + LW/2, 0), (sites[1][0] - TW/2, 0))

    # lambda on right of right site
    rx = sites[1][0]
    lam_x = rx + TW/2 + LW/2 + 0.15
    draw_bond(ax, (rx + TW/2, 0), (lam_x - LW/2, 0))
    _draw_lambda(ax, lam_x, 0)
    draw_bond(ax, (lam_x + LW/2, 0), (lam_x + LW/2 + 0.3, 0))

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.2, 2.2)
    save(fig, "su_lambda_network.png")


def su_absorb_before():
    """Two sites with lambda boxes on surrounding bonds, bare connecting bond."""
    fig, ax = new_fig(7, 5)
    gap = 2.2

    for sign, label in [(-1, "$A_i$"), (1, "$A_j$")]:
        cx = sign * gap / 2
        draw_tensor(ax, cx, 0, TW, TH, label)

        # top lambda
        ly = TH/2 + LH/2 + 0.15
        draw_bond(ax, (cx, TH/2), (cx, ly - LH/2))
        _draw_lambda(ax, cx, ly)
        draw_bond(ax, (cx, ly + LH/2), (cx, ly + LH/2 + 0.3))

        # bottom lambda
        ly = -TH/2 - LH/2 - 0.15
        draw_bond(ax, (cx, -TH/2), (cx, ly + LH/2))
        _draw_lambda(ax, cx, ly)
        draw_bond(ax, (cx, ly - LH/2), (cx, ly - LH/2 - 0.3))

        # outer horizontal lambda
        lam_x = cx + sign * (TW/2 + LW/2 + 0.15)
        draw_bond(ax, (cx + sign * TW/2, 0), (lam_x - sign * LW/2, 0))
        _draw_lambda(ax, lam_x, 0)
        draw_bond(ax, (lam_x + sign * LW/2, 0),
                  (lam_x + sign * (LW/2 + 0.3), 0))

        # physical leg (angled toward the other tensor)
        px = cx - sign * TW/4
        draw_bond(ax, (px, -TH/2), (px - sign * LEG * 0.4, -TH/2 - LEG * 0.4))

    # bare connecting bond (no lambda)
    draw_bond(ax, (-gap/2 + TW/2, 0), (gap/2 - TW/2, 0))

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.2)
    save(fig, "su_absorb_before.png")


def su_absorb_after():
    """Two sites after lambda absorption — bare legs, no lambda boxes."""
    fig, ax = new_fig(7, 3.5)
    gap = 2.2

    for sign, label in [(-1, "$T_i$"), (1, "$T_j$")]:
        cx = sign * gap / 2
        draw_tensor(ax, cx, 0, TW, TH, label)
        draw_bond(ax, (cx + sign * TW/2, 0), (cx + sign * (TW/2 + LEG), 0))
        draw_bond(ax, (cx, TH/2), (cx, TH/2 + LEG))
        draw_bond(ax, (cx, -TH/2), (cx, -TH/2 - LEG))
        # physical leg (angled toward the other tensor)
        px = cx - sign * TW/4
        draw_bond(ax, (px, -TH/2), (px - sign * LEG * 0.4, -TH/2 - LEG * 0.4))

    draw_bond(ax, (-gap/2 + TW/2, 0), (gap/2 - TW/2, 0))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.8, 1.8)
    save(fig, "su_absorb_after.png")


def su_gate_svd():
    """Gate applied to reduced tensors (same as full update gate diagram)."""
    fig, ax = new_fig(7, 4)
    rx_l, rx_r = -1.2, 1.2
    gy = -1.2
    y = 0

    draw_tensor(ax, rx_l, y, SW, SH, "$a^R_i$", fontsize=10)
    draw_bond(ax, (rx_l - SW/2, y), (rx_l - SW/2 - LEG, y))

    draw_tensor(ax, rx_r, y, SW, SH, "$a^R_j$", fontsize=10)
    draw_bond(ax, (rx_r + SW/2, y), (rx_r + SW/2 + LEG, y))

    draw_bond(ax, (rx_l + SW/2, y), (rx_r - SW/2, y))
    draw_bond(ax, (rx_l, -SH/2), (rx_l, gy + SH/2))
    draw_bond(ax, (rx_r, -SH/2), (rx_r, gy + SH/2))

    gw = rx_r - rx_l + SW
    draw_tensor(ax, 0, gy, gw, SH, "$g_{ij}$", fontsize=11)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 1.5)
    save(fig, "su_gate_svd.png")


def su_svd_result():
    """SVD result: U -- lambda -- V*."""
    fig, ax = new_fig(7, 3)
    ux, lx, vx = -1.5, 0, 1.5
    y = 0

    draw_tensor(ax, ux, y, SW, SH, "$U$", fontsize=12)
    draw_bond(ax, (ux - SW/2, y), (ux - SW/2 - LEG, y))
    draw_bond(ax, (ux + SW/2, y), (lx - LW/2, y))

    _draw_lambda(ax, lx, y, r"$\Sigma$")
    draw_bond(ax, (lx + LW/2, y), (vx - SW/2, y))

    draw_tensor(ax, vx, y, SW, SH, "$V^\\dagger$", fontsize=10)
    draw_bond(ax, (vx + SW/2, y), (vx + SW/2 + LEG, y))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.2, 1.2)
    save(fig, "su_svd_result.png")


if __name__ == "__main__":
    su_lambda_network()
    su_absorb_before()
    su_absorb_after()
    su_gate_svd()
    su_svd_result()
    print("plot_simple_update.py: done")
