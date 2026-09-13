r"""Startup splash art — a mountain-range motif, lavalamp-style: shown once
at session start, single amber color, above the input.

Built by defining key vertices as (row, col) coordinates and drawing each
segment column-by-column: for every column between the two endpoints,
interpolate the row height and place exactly one character there. This is
the key difference from a raw per-pixel Bresenham walk — stepping through
every grid pixel produces little staircase clusters wherever a line is
steeper than 1 row/column, because several pixels land in the same column.
Placing one char per column instead gives a single clean diagonal stroke.
A dash pattern is applied over that column sequence (not raw pixels), so
gaps line up with the visual line instead of chopping it into fragments.
"""

HEIGHT = 20
WIDTH = 70


def _draw_segment(grid, r0, c0, r1, c1, dash=(4, 1), min_len_for_dash=8):
    """Draw one straight segment from (r0, c0) to (r1, c1). Steps along
    whichever axis (row or column) spans more cells, interpolating the
    other coordinate — a segment steeper than 1 row/column (common for tall
    peaks) would otherwise skip rows entirely and leave gaps. Short
    segments are drawn solid, since dashing only a few cells just produces
    gaps rather than a readable dotted line."""
    on, off = dash
    cycle = on + off
    dc = c1 - c0
    dr = r1 - r0
    n = max(abs(dr), abs(dc))

    if n == 0:
        grid[r0][c0] = "."
        return

    if dc == 0:
        ch = "|"
    elif dr == 0:
        ch = "_"
    else:
        ch = "\\" if (dr > 0) == (dc > 0) else "/"

    solid = n < min_len_for_dash
    for i in range(n + 1):
        if not solid and (i % cycle) >= on:
            continue
        t = i / n
        r = round(r0 + t * dr)
        c = round(c0 + t * dc)
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            grid[r][c] = ch


def _draw_polyline(grid, vertices, dash=(4, 1)):
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        _draw_segment(grid, r0, c0, r1, c1, dash=dash)


def _mark_vertices(grid, vertices, peak_indices=None):
    """Stamp 'o' at true peaks, '.' at other listed joints — a final pass
    so they're never overwritten by segment drawing."""
    peak_indices = peak_indices or set()
    for i, (r, c) in enumerate(vertices):
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            grid[r][c] = "o" if i in peak_indices else "."


def _make_grid(height, width, fill=" "):
    return [[fill for _ in range(width)] for _ in range(height)]


def build_splash() -> str:
    grid = _make_grid(HEIGHT, WIDTH)
    baseline = HEIGHT - 1

    peaks = {
        "left": [(baseline, 2), (12, 12), (baseline, 22)],
        "second": [(baseline, 8), (5, 25), (baseline, 42)],
        # "center" has its left edge removed (occluded by "second", the
        # peak in front of it), so its only vertex besides the peak itself
        # is its right foot — no left-edge crossing through "second".
        "center": [(1, 32), (baseline, 42)],
        "right": [(baseline, 42), (9, 52), (baseline, 62)],
        "small_right": [(baseline, 50), (13, 56), (baseline, 62)],
    }
    ground = [(baseline, 0), (baseline, WIDTH - 1)]

    _draw_polyline(grid, ground, dash=(4, 1))
    for verts in peaks.values():
        _draw_polyline(grid, verts, dash=(4, 1))

    for name, verts in peaks.items():
        _mark_vertices(grid, verts, peak_indices={0} if name == "center" else {1})

    return "\n".join("".join(row) for row in grid)
