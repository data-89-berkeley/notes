"""
Utility functions and visualization class for the constrained optimization demo.
Used by constrained_optimization.ipynb

Students look for the maximum of f(x1, x2) on the constraint set
X = {(x1, x2) : g(x1, x2) = level} by reading level sets and gradients, then
test their answer against the geometry behind the Lagrange multiplier equation:
at a constrained maximum the point lies on X and grad f is parallel to grad g.

Clicking on the plot needs plotly's FigureWidget, which needs the `anywidget`
package. Without it the demo still runs, and the point is placed with sliders.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display

try:
    # Ships with matplotlib 3.6+, and traces contours about 10x faster than a figure.
    import contourpy
except ImportError:
    contourpy = None
    from matplotlib.figure import Figure


# Colors: everything about f is blue, everything about g is orange.
F_COLOR = "#2166ac"
G_COLOR = "#d55e00"
POINT_COLOR = "#1a1a1a"
# Pale single-hue ramp so the heatmap stays behind the lines drawn on top of it.
F_HEATMAP_SCALE = [[0.0, "#ffffff"], [1.0, "#b9d3ea"]]

# Plot window and grid. Clicks snap to the grid, so its spacing (0.025) must stay
# well below POSITION_TOL.
WINDOW = (-3.0, 3.0)
GRID_SIZE = 241
GRID = np.linspace(WINDOW[0], WINDOW[1], GRID_SIZE)
X1_GRID, X2_GRID = np.meshgrid(GRID, GRID)

# Check tolerances.
POSITION_TOL = 0.06   # distance from the point to X, in plot units
PARALLEL_TOL = 0.10   # sine of the angle between grad f and grad g (about 6 degrees)
CLOSE_FACTOR = 3.0    # within this multiple of a tolerance counts as "Almost!"
VALUE_TOL = 0.02      # share of the range of f on X allowed below the best maximum

# Lengths of the marks drawn at the selected point, in plot units. The marks for
# f are longer so they still show behind those for g when the two line up, which
# is exactly what happens at a solution.
F_ARROW_LENGTH, G_ARROW_LENGTH = 1.2, 0.75
F_TANGENT_HALF_LENGTH, G_TANGENT_HALF_LENGTH = 0.8, 0.45

QUIVER_POINTS = 13
RANDOM = "Random"
RANDOM_ATTEMPTS = 60
LEVEL_SCAN_STEP = 0.1  # spacing of the levels tested when bounding the level slider

ScalarField = Callable[[np.ndarray, np.ndarray], np.ndarray]


# Function libraries (dropdown label -> function).

F_FUNCTIONS: dict[str, ScalarField] = {
    "Hill: −x₁² − x₂²": lambda x, y: -(x**2 + y**2),
    "Offset hill: −(x₁−1)² − 2(x₂−½)²": lambda x, y: -((x - 1) ** 2) - 2 * (y - 0.5) ** 2,
    "Plane: x₁ + 2x₂": lambda x, y: x + 2 * y,
    "Saddle: x₁x₂": lambda x, y: x * y,
    "Two peaks": lambda x, y: (
        2.0 * np.exp(-((x - 1.2) ** 2 + (y - 1.0) ** 2) / 1.2)
        + 1.4 * np.exp(-((x + 1.4) ** 2 + (y + 1.0) ** 2) / 1.2)
    ),
    "Waves: sin(x₁) + cos(x₂)": lambda x, y: np.sin(x) + np.cos(y),
}

G_FUNCTIONS: dict[str, ScalarField] = {
    "Circle: x₁² + x₂² − 4": lambda x, y: x**2 + y**2 - 4,
    "Ellipse: x₁²/6 + x₂²/2 − 1": lambda x, y: x**2 / 6 + y**2 / 2 - 1,
    "Line: x₁ + x₂ − 1": lambda x, y: x + y - 1,
    "Parabola: x₂ − x₁² + 1": lambda x, y: y - x**2 + 1,
    "Hyperbola: x₁x₂ − 1": lambda x, y: x * y - 1,
    "Wave: x₂ − sin(1.5x₁)": lambda x, y: y - np.sin(1.5 * x),
}


def _rotated(x, y, center, angle):
    """Coordinates of (x, y) in a frame centered at `center`, rotated by `angle`."""
    dx, dy = x - center[0], y - center[1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy


def random_objective(rng: np.random.Generator) -> ScalarField:
    """Draw a random objective: a tilted hill, a plane, two peaks, or a saddle."""
    kind = rng.choice(["hill", "plane", "peaks", "saddle"])
    angle = rng.uniform(0, 2 * np.pi)
    if kind == "hill":
        center = rng.uniform(-1.5, 1.5, 2)
        a, b = rng.uniform(0.4, 1.5, 2)

        def f(x, y):
            u, v = _rotated(x, y, center, angle)
            return -(a * u**2 + b * v**2)

    elif kind == "plane":
        def f(x, y):
            return np.cos(angle) * x + np.sin(angle) * y

    elif kind == "peaks":
        first = 1.4 * np.array([np.cos(angle), np.sin(angle)]) + rng.uniform(-0.3, 0.3, 2)
        second = -1.4 * np.array([np.cos(angle), np.sin(angle)]) + rng.uniform(-0.3, 0.3, 2)
        height = rng.uniform(0.5, 0.85)

        def f(x, y):
            return 2.0 * (
                np.exp(-((x - first[0]) ** 2 + (y - first[1]) ** 2) / 1.2)
                + height * np.exp(-((x - second[0]) ** 2 + (y - second[1]) ** 2) / 1.2)
            )

    else:
        center = rng.uniform(-0.7, 0.7, 2)

        def f(x, y):
            u, v = _rotated(x, y, center, angle)
            return u * v

    return f


def random_constraint(rng: np.random.Generator) -> ScalarField:
    """Draw a random constraint whose zero level set crosses the plot window."""
    kind = rng.choice(["ellipse", "line", "parabola"])
    angle = rng.uniform(0, 2 * np.pi)
    center = rng.uniform(-0.7, 0.7, 2)
    if kind == "ellipse":
        a, b = rng.uniform(1.3, 2.2, 2)

        def g(x, y):
            u, v = _rotated(x, y, center, angle)
            return (u / a) ** 2 + (v / b) ** 2 - 1

    elif kind == "line":
        def g(x, y):
            u, _ = _rotated(x, y, center, angle)
            return u

    else:
        curvature = rng.uniform(0.3, 0.8)

        def g(x, y):
            u, v = _rotated(x, y, center, angle)
            return v - curvature * u**2

    return g


# Calculus and geometry on the plot window.

def gradient(func: ScalarField, point, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of `func` at `point`."""
    x, y = float(point[0]), float(point[1])
    return np.array([
        (func(x + h, y) - func(x - h, y)) / (2 * h),
        (func(x, y + h) - func(x, y - h)) / (2 * h),
    ], dtype=float)


def level_curves(z_grid: np.ndarray, level: float) -> list[np.ndarray]:
    """Polylines of the level set {z = level} on the plot window."""
    z_min, z_max = float(z_grid.min()), float(z_grid.max())
    if not z_min < level < z_max:
        return []
    if contourpy is not None:
        segments = contourpy.contour_generator(GRID, GRID, z_grid).lines(level)
    else:
        segments = Figure().add_subplot().contour(GRID, GRID, z_grid, levels=[level]).allsegs[0]
    return [np.asarray(s, dtype=float) for s in segments if len(s) > 1]


def _drop_repeated_vertices(curve: np.ndarray) -> np.ndarray:
    """Contours through grid nodes repeat vertices, which rounding turns into false peaks."""
    closed = np.allclose(curve[0], curve[-1])
    keep = np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-3]
    curve = curve[keep]
    if closed:
        curve[-1] = curve[0]
    return curve


def constraint_curves(g: ScalarField, level: float) -> list[np.ndarray]:
    """
    Polylines of X = {g = level}, with vertices moved onto X by Newton steps.
    Raw contour vertices sit slightly off the curve, and that error would show up
    as false local maxima when f is evaluated along them.
    """
    curves = [_drop_repeated_vertices(c) for c in level_curves(g(X1_GRID, X2_GRID), level)]
    h = 1e-5
    for curve in curves:
        for _ in range(3):
            x, y = curve[:, 0], curve[:, 1]
            grad_x = (g(x + h, y) - g(x - h, y)) / (2 * h)
            grad_y = (g(x, y + h) - g(x, y - h)) / (2 * h)
            norm_sq = np.maximum(grad_x**2 + grad_y**2, 1e-8)
            step = (g(x, y) - level) / norm_sq
            curve[:, 0] -= step * grad_x
            curve[:, 1] -= step * grad_y
    return curves


def _on_window_edge(points: np.ndarray) -> np.ndarray:
    margin = 2 * (GRID[1] - GRID[0])
    return np.any(np.abs(points) >= WINDOW[1] - margin, axis=1)


def constrained_maxima(f: ScalarField, curves: list[np.ndarray]) -> list[np.ndarray]:
    """
    Local maxima of f along the curves, ignoring points where a curve leaves the
    window: f may simply keep growing past the edge, so those are not maxima.
    """
    maxima = []
    for curve in curves:
        values = f(curve[:, 0], curve[:, 1])
        closed = np.allclose(curve[0], curve[-1])
        if closed:
            points, values = curve[:-1], values[:-1]
            before, after = np.roll(values, 1), np.roll(values, -1)
            interior = np.ones(len(values), dtype=bool)
        else:
            points = curve
            before = np.r_[np.inf, values[:-1]]
            after = np.r_[values[1:], np.inf]
            interior = np.ones(len(values), dtype=bool)
            interior[[0, -1]] = False
        is_peak = (values > before) & (values >= after) & interior & ~_on_window_edge(points)
        maxima.extend(points[is_peak])
    return maxima


def _values_on_curves(f: ScalarField, curves: list[np.ndarray]) -> np.ndarray:
    samples = np.vstack(curves)
    return f(samples[:, 0], samples[:, 1])


def maximum_on_window_edge(f: ScalarField, curves: list[np.ndarray]) -> bool:
    """
    True when the highest value of f on X within the window sits on the window's
    edge rather than at a maximum inside it. The window then acts as an extra
    constraint, and the point the plot suggests is not a true constrained maximum.
    """
    if not curves:
        return False
    values = _values_on_curves(f, curves)
    spread = float(values.max() - values.min())
    if spread < 1e-6:
        return False
    maxima = constrained_maxima(f, curves)
    best_inside = max((float(f(p[0], p[1])) for p in maxima), default=-np.inf)
    return best_inside < float(values.max()) - VALUE_TOL * spread


def has_clear_maximum(f: ScalarField, curves: list[np.ndarray]) -> bool:
    """True when the highest value of f on the curves is a maximum inside the window."""
    if not curves:
        return False
    values = _values_on_curves(f, curves)
    spread = float(values.max() - values.min())
    return spread > 1e-6 and not maximum_on_window_edge(f, curves)


def problem_is_usable(f: ScalarField, g: ScalarField, level: float) -> bool:
    """
    True when X is non-empty in the window and the maximum of f on X is not on
    the window's edge. Unlike has_clear_maximum, this allows f constant on X,
    where every point of X is a maximum.
    """
    curves = constraint_curves(g, level)
    return bool(curves) and not maximum_on_window_edge(f, curves)


@lru_cache(maxsize=None)
def named_pair_usable(f_name: str, g_name: str) -> bool:
    """Whether a dropdown pair works at level 0, where every new problem starts."""
    return problem_is_usable(F_FUNCTIONS[f_name], G_FUNCTIONS[g_name], 0.0)


def usable_level_range(f: ScalarField, g: ScalarField, z_g: np.ndarray) -> tuple[float, float]:
    """
    The interval of levels around 0, in steps of LEVEL_SCAN_STEP, where the
    problem stays usable. Bounding the level slider by it means dragging can
    never reach a problem whose maximum is on the window's edge.
    """
    bounds = []
    for direction, limit in ((-1, float(z_g.min())), (1, float(z_g.max()))):
        level = 0.0
        while True:
            step = round(level + direction * LEVEL_SCAN_STEP, 2)
            if (step - limit) * direction >= 0 or not problem_is_usable(f, g, step):
                break
            level = step
        bounds.append(level)
    return bounds[0], bounds[1]


# The "check if feasible" and "check if maximal" tests.

@dataclass
class FeasibilityResult:
    """Outcome of testing a proposed point against the Lagrange conditions."""
    in_constraint: bool
    parallel: bool
    distance: float       # distance from the point to X (inf when X is empty)
    sine: float           # size of the part of grad f perpendicular to grad g
    message: str
    verdict: str          # "Feasible!", "Almost!", or "Try again!"

    @property
    def feasible(self) -> bool:
        return self.verdict == "Feasible!"


@dataclass
class MaximalityResult:
    """Outcome of testing a feasible point for being the constrained maximum."""
    is_maximum: bool
    message: str
    verdict: str          # "Correct!" or "Almost!"


def _distance_to_curves(point: np.ndarray, curves: list[np.ndarray]) -> float:
    if not curves:
        return float("inf")
    return float(np.min(np.linalg.norm(np.vstack(curves) - point, axis=1)))


def _perpendicular_share(grad_f, grad_g, typical_grad_f: float) -> float:
    """
    Size of the projection of grad f perpendicular to grad g, relative to grad f.
    This equals the sine of the angle between them. `typical_grad_f` keeps the
    ratio stable where grad f nearly vanishes and its direction is mostly noise.
    """
    norm_g = np.linalg.norm(grad_g)
    if norm_g < 1e-8:
        return float("inf")
    unit_g = grad_g / norm_g
    perpendicular = grad_f - (grad_f @ unit_g) * unit_g
    scale = max(float(np.linalg.norm(grad_f)), 0.25 * typical_grad_f, 1e-12)
    return float(np.linalg.norm(perpendicular) / scale)


def _typical_gradient_norm(z_grid: np.ndarray) -> float:
    spacing = GRID[1] - GRID[0]
    d_x2, d_x1 = np.gradient(z_grid, spacing)
    return float(np.median(np.hypot(d_x1, d_x2)))


def check_feasibility(f, g, z_f, curves, point) -> FeasibilityResult:
    """
    Test whether `point` meets the Lagrange conditions for: maximize f subject
    to g = level. Primal feasibility: the point lies on X. Dual feasibility:
    grad f is parallel to grad g there.

    `curves` are the polylines of X and `z_f` is f on the grid. These conditions
    also hold at constrained minima and saddles; `check_maximal` tells them apart.
    """
    point = np.asarray(point, dtype=float)
    distance = _distance_to_curves(point, curves)
    sine = _perpendicular_share(
        gradient(f, point), gradient(g, point), _typical_gradient_norm(z_f)
    )
    in_constraint = distance <= POSITION_TOL
    # Near a sharp peak grad f turns too fast for any clickable point to pass the
    # angle test, so a point within POSITION_TOL of a true maximum also passes.
    maxima = constrained_maxima(f, curves)
    near_maximum = any(np.linalg.norm(p - point) <= POSITION_TOL for p in maxima)
    parallel = sine <= PARALLEL_TOL or near_maximum

    if not curves:
        return FeasibilityResult(False, parallel, distance, sine,
                                 "𝒳 is empty in this window. Try another level.", "Try again!")

    if not (in_constraint and parallel):
        close = distance <= CLOSE_FACTOR * POSITION_TOL and sine <= CLOSE_FACTOR * PARALLEL_TOL
        return FeasibilityResult(in_constraint, parallel, distance, sine, "",
                                 "Almost!" if close else "Try again!")

    return FeasibilityResult(True, True, distance, sine, "", "Feasible!")


def check_maximal(f, curves, point) -> MaximalityResult:
    """
    Test whether a feasible `point` is the constrained maximum: its value of f
    must match the highest constrained maximum of f on X within VALUE_TOL.
    """
    point = np.asarray(point, dtype=float)
    samples = np.vstack(curves)
    values = f(samples[:, 0], samples[:, 1])
    spread = float(values.max() - values.min())
    if spread < 1e-6:
        return MaximalityResult(True, "f is constant on 𝒳, so every point of 𝒳 is a maximum.",
                                "Correct!")

    maxima = constrained_maxima(f, curves)
    if not maxima:
        return MaximalityResult(
            False, "This point is stationary, but f has no maximum on 𝒳 in this window.", "Almost!"
        )

    best = max(float(f(p[0], p[1])) for p in maxima)
    if float(f(point[0], point[1])) >= best - VALUE_TOL * spread:
        return MaximalityResult(True, "", "Correct!")
    return MaximalityResult(False, "This point is stationary, but f is larger elsewhere on 𝒳.",
                            "Almost!")


# Drawing helpers.

def _arrow_path(tail, direction, length, head_share=0.25, head_angle=0.45):
    """x and y lists for an arrow, with None breaks between its strokes."""
    tail = np.asarray(tail, dtype=float)
    tip = tail + length * direction
    xs, ys = [tail[0], tip[0], None], [tail[1], tip[1], None]
    head = head_share * length
    for sign in (1, -1):
        cos_a, sin_a = np.cos(sign * head_angle), np.sin(sign * head_angle)
        back = -np.array([
            cos_a * direction[0] - sin_a * direction[1],
            sin_a * direction[0] + cos_a * direction[1],
        ])
        barb = tip + head * back
        xs += [tip[0], barb[0], None]
        ys += [tip[1], barb[1], None]
    return xs, ys


def _quiver_paths(func: ScalarField):
    """
    Arrows of the gradient field on a coarse grid. Lengths use the square root of
    the gradient size, so weak regions stay visible next to steep ones.
    """
    nodes = np.linspace(WINDOW[0] + 0.25, WINDOW[1] - 0.25, QUIVER_POINTS)
    spacing = nodes[1] - nodes[0]
    tails = [(a, b) for a in nodes for b in nodes]
    grads = [gradient(func, tail) for tail in tails]
    norms = [float(np.linalg.norm(grad)) for grad in grads]
    largest = max(norms)
    xs, ys = [], []
    if largest < 1e-9:
        return xs, ys
    for tail, grad, norm in zip(tails, grads, norms):
        if norm < 1e-9:
            continue
        length = 0.85 * spacing * np.sqrt(norm / largest)
        arrow_x, arrow_y = _arrow_path(tail, grad / norm, length, head_share=0.32)
        xs += arrow_x
        ys += arrow_y
    return xs, ys


def _contour_spec(z_grid: np.ndarray, count: int = 12) -> dict:
    """Evenly spaced levels between the 2nd and 98th percentiles of the grid."""
    low, high = np.percentile(z_grid, [2, 98])
    size = max((high - low) / count, 1e-6)
    return dict(start=float(low), end=float(high), size=float(size), coloring="lines", showlabels=False)


# Trace positions in the figure.
(HEATMAP, F_LEVELS, G_LEVELS, F_FIELD, G_FIELD, CONSTRAINT,
 F_TANGENT, G_TANGENT, F_ARROW, G_ARROW, POINT) = range(11)


class ConstrainedOptimizationDemo:
    """Interactive practice for solving constrained optimization problems graphically."""

    def __init__(self, seed: Optional[int] = None, require_maximum: bool = True):
        self.rng = np.random.default_rng(seed)
        self.require_maximum = require_maximum
        self.point = np.array([0.0, 0.0])
        self._syncing_point = False
        self._setting_level_bounds = False
        self._filtering_constraints = False
        self._feasibility: Optional[FeasibilityResult] = None
        self._maximality: Optional[MaximalityResult] = None
        # True while the maximum on X lies on the window's edge (see _update_blocked).
        self._blocked = False

        self._create_widgets()
        self._create_figure()
        self._choose_functions()
        self._setup_callbacks()
        self._update_fields()
        self._update_point()

    # Setup

    def _create_widgets(self):
        style = {"description_width": "62px"}
        wide = widgets.Layout(width="330px")

        self.f_dropdown = widgets.Dropdown(
            options=[RANDOM, *F_FUNCTIONS], value=RANDOM,
            description="f (objective)", style={"description_width": "86px"}, layout=wide,
        )
        self.g_dropdown = widgets.Dropdown(
            options=[RANDOM, *G_FUNCTIONS], value=RANDOM,
            description="g (constraint)", style={"description_width": "86px"}, layout=wide,
        )
        self.new_random_button = widgets.Button(
            description="New random problem", icon="refresh", layout=widgets.Layout(width="330px"),
        )
        # Each new problem resets this to 0 and bounds it to the levels where the
        # maximum stays inside the window (see _set_level_bounds).
        self.level_slider = widgets.FloatSlider(
            value=0.0, min=-4.0, max=4.0, step=0.01, description="g level",
            readout_format=".2f", style=style, layout=wide,
        )

        def checkbox():
            return widgets.Checkbox(value=False, indent=False, description="",
                                    layout=widgets.Layout(width="auto"))

        self.f_levels_box, self.f_field_box = checkbox(), checkbox()
        self.g_levels_box, self.g_field_box = checkbox(), checkbox()

        self.x1_slider = widgets.FloatSlider(
            value=0.0, min=WINDOW[0], max=WINDOW[1], step=0.01, description="x₁",
            readout_format=".2f", style=style, layout=wide,
        )
        self.x2_slider = widgets.FloatSlider(
            value=0.0, min=WINDOW[0], max=WINDOW[1], step=0.01, description="x₂",
            readout_format=".2f", style=style, layout=wide,
        )

        self.feasible_button = widgets.Button(
            description="Check if Feasible", button_style="primary",
            layout=widgets.Layout(width="163px"),
        )
        # Greyed out until the point passes the feasibility checks.
        self.maximal_button = widgets.Button(
            description="Check if Maximal", button_style="primary", disabled=True,
            layout=widgets.Layout(width="163px"),
        )
        self.result_html = widgets.HTML(value="", layout=widgets.Layout(width="330px"))
        self.plot_output = widgets.Output()

    def _create_figure(self):
        empty = dict(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False)
        flat = lambda color: [[0.0, color], [1.0, color]]
        # x0/dx rather than coordinate arrays: FigureWidget drops numpy x and y
        # arrays set at construction, which makes clicks report cell indices.
        spacing = float(GRID[1] - GRID[0])
        grid = dict(x0=WINDOW[0], dx=spacing, y0=WINDOW[0], dy=spacing)
        traces = [
            go.Heatmap(
                **grid, colorscale=F_HEATMAP_SCALE, showscale=False,
                hovertemplate="x₁ = %{x:.2f}<br>x₂ = %{y:.2f}<extra></extra>",
            ),
            go.Contour(**grid, showscale=False, hoverinfo="skip", visible=False,
                       colorscale=flat(F_COLOR), line=dict(width=1.2)),
            go.Contour(**grid, showscale=False, hoverinfo="skip", visible=False,
                       colorscale=flat(G_COLOR), line=dict(width=1.0, dash="dot")),
            go.Scatter(**empty, line=dict(color=F_COLOR, width=1.2)),
            go.Scatter(**empty, line=dict(color=G_COLOR, width=1.2)),
            go.Scatter(**empty, line=dict(color=G_COLOR, width=3.5)),
            go.Scatter(**empty, line=dict(color=F_COLOR, width=4)),
            go.Scatter(**empty, line=dict(color=G_COLOR, width=4)),
            go.Scatter(**empty, line=dict(color=F_COLOR, width=3.5)),
            go.Scatter(**empty, line=dict(color=G_COLOR, width=3.5)),
            go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip", showlegend=False,
                       marker=dict(size=12, color=POINT_COLOR, line=dict(color="white", width=2))),
        ]
        axis = dict(
            range=list(WINDOW), fixedrange=True, showgrid=False, zeroline=False,
            ticks="outside", tickcolor="#999999", linecolor="#999999", mirror=True,
            constrain="domain",
        )
        layout = go.Layout(
            width=520, height=510, margin=dict(l=50, r=10, t=10, b=45),
            plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
            hovermode="closest", font=dict(size=14, color="#333333"),
            xaxis=dict(axis, title="x₁"),
            # Equal scales, so perpendicular lines look perpendicular on screen.
            yaxis=dict(axis, title="x₂", scaleanchor="x", scaleratio=1),
            annotations=[
                dict(text="𝒳", x=0, y=0, showarrow=False, visible=False,
                     font=dict(size=24, color="#333333"),
                     bgcolor="rgba(255,255,255,0.85)", borderpad=1),
                dict(text="", xref="paper", yref="paper", x=0.5, y=0.97, yanchor="top",
                     showarrow=False, visible=False, font=dict(size=24, color="#1a1a1a"),
                     bgcolor="white", bordercolor="#333333", borderwidth=2, borderpad=10),
            ],
        )
        try:
            self.fig = go.FigureWidget(data=traces, layout=layout)
            self.clickable = True
        except ImportError:
            self.fig = go.Figure(data=traces, layout=layout)
            self.clickable = False
            # Each redraw replaces the whole plot here, so wait for slider release.
            self.level_slider.continuous_update = False
            self.x1_slider.continuous_update = False
            self.x2_slider.continuous_update = False

    def _setup_callbacks(self):
        self.f_dropdown.observe(self._on_objective_change, names="value")
        self.g_dropdown.observe(self._on_function_change, names="value")
        self.new_random_button.on_click(self._on_new_random)
        self.level_slider.observe(self._on_level_change, names="value")
        for box, partner in (
            (self.f_levels_box, self.f_field_box),
            (self.f_field_box, self.f_levels_box),
            (self.g_levels_box, self.g_field_box),
            (self.g_field_box, self.g_levels_box),
        ):
            box.observe(self._exclusive_handler(partner), names="value")
        self.x1_slider.observe(self._on_point_slider, names="value")
        self.x2_slider.observe(self._on_point_slider, names="value")
        self.feasible_button.on_click(self._on_check_feasible)
        self.maximal_button.on_click(self._on_check_maximal)
        if self.clickable:
            self.fig.data[HEATMAP].on_click(self._on_plot_click)

    # Problem state

    def _pick(self, dropdown, library, draw) -> ScalarField:
        return draw(self.rng) if dropdown.value == RANDOM else library[dropdown.value]

    def _choose_functions(self):
        """
        Set f and g at level 0, redrawing random choices until the maximum of f on
        X lies inside the window. Dropdown pairs need no redraw: the g dropdown
        only lists constraints that work with the chosen f.
        """
        for _ in range(RANDOM_ATTEMPTS):
            f = self._pick(self.f_dropdown, F_FUNCTIONS, random_objective)
            g = self._pick(self.g_dropdown, G_FUNCTIONS, random_constraint)
            if has_clear_maximum(f, constraint_curves(g, 0.0)):
                break
        self.f, self.g = f, g
        self.z_f = f(X1_GRID, X2_GRID)
        self.z_g = g(X1_GRID, X2_GRID)
        self._set_level_bounds()
        self.new_random_button.disabled = RANDOM not in (self.f_dropdown.value, self.g_dropdown.value)

    def _set_level_bounds(self):
        """Reset the level to 0 and limit the slider to the levels where the problem is usable."""
        low, high = usable_level_range(self.f, self.g, self.z_g)
        slider = self.level_slider
        # These changes would fire the level observer; the caller redraws anyway.
        self._setting_level_bounds = True
        try:
            # Every range contains 0, so moving there first keeps the value inside
            # the bounds; widen before narrowing so min never passes max.
            slider.value = 0.0
            slider.min = min(slider.min, low)
            slider.max = high
            slider.min = low
        finally:
            self._setting_level_bounds = False

    def _filter_constraints(self):
        """List only the constraints that give the chosen f a maximum inside the window."""
        f_name = self.f_dropdown.value
        names = [name for name in G_FUNCTIONS
                 if f_name == RANDOM or named_pair_usable(f_name, name)]
        options = [RANDOM, *names]
        if list(self.g_dropdown.options) == options:
            return
        current = self.g_dropdown.value
        self._filtering_constraints = True
        try:
            self.g_dropdown.options = options
            # A constraint that no longer fits switches to Random, which redraws one that does.
            self.g_dropdown.value = current if current in options else RANDOM
        finally:
            self._filtering_constraints = False

    # Drawing

    def _show(self):
        """Without FigureWidget the figure is static, so display it again."""
        if not self.clickable:
            with self.plot_output:
                clear_output(wait=True)
                display(self.fig)

    def _update_fields(self):
        """Redraw everything that depends on f, g, the level, or the checkboxes."""
        low, high = np.percentile(self.z_f, [2, 98])
        field_f = _quiver_paths(self.f) if self.f_field_box.value else ([], [])
        field_g = _quiver_paths(self.g) if self.g_field_box.value else ([], [])

        with self.fig.batch_update():
            data = self.fig.data
            data[HEATMAP].update(z=self.z_f, zmin=float(low), zmax=float(high))
            data[F_LEVELS].update(z=self.z_f, contours=_contour_spec(self.z_f),
                                  visible=self.f_levels_box.value)
            data[G_LEVELS].update(z=self.z_g, contours=_contour_spec(self.z_g),
                                  visible=self.g_levels_box.value)
            data[F_FIELD].update(x=field_f[0], y=field_f[1])
            data[G_FIELD].update(x=field_g[0], y=field_g[1])
        self._update_constraint()

    def _update_constraint(self):
        """Redraw the constraint set X, the only part of the plot that depends on the level."""
        self._curves = constraint_curves(self.g, self.level_slider.value)
        x_path, y_path = [], []
        for curve in self._curves:
            x_path += [*curve[:, 0], None]
            y_path += [*curve[:, 1], None]
        with self.fig.batch_update():
            self.fig.data[CONSTRAINT].update(x=x_path, y=y_path)
            self._place_constraint_label()
        self._update_blocked()

    def _update_blocked(self):
        """
        Lock the controls below "New random problem" while the maximum of f on X
        lies on the window's edge: that point only looks maximal because the
        window cuts X off. f, g and the level stay usable to leave this state.
        """
        self._blocked = maximum_on_window_edge(self.f, self._curves)
        for control in (self.f_levels_box, self.f_field_box, self.g_levels_box,
                        self.g_field_box, self.x1_slider, self.x2_slider, self.feasible_button):
            control.disabled = self._blocked
        feasible = self._feasibility is not None and self._feasibility.feasible
        self.maximal_button.disabled = self._blocked or not feasible
        if self._blocked:
            # Smaller than the check results so the longer text fits inside the plot.
            self._set_popup("<b>Maxima Outside Display Window.<br>Generate a New Problem.</b>",
                            "#b3261e", size=20)

    def _place_constraint_label(self):
        label = self.fig.layout.annotations[0]
        if not self._curves:
            label.visible = False
            return
        longest = max(self._curves, key=len)
        anchor = longest[len(longest) // 4]
        # Sit just beside the curve, on the side that points back into the window.
        step = longest[min(len(longest) // 4 + 1, len(longest) - 1)] - longest[len(longest) // 4 - 1]
        normal = np.array([-step[1], step[0]]) / max(np.linalg.norm(step), 1e-9)
        if anchor @ normal > 0:
            normal = -normal
        spot = anchor + 0.28 * normal
        label.update(x=float(spot[0]), y=float(spot[1]), visible=True)

    def _update_point(self):
        """Redraw the selected point and the marks attached to it."""
        marks = {F_TANGENT: ([], []), G_TANGENT: ([], []), F_ARROW: ([], []), G_ARROW: ([], [])}
        for func, levels_box, field_box, tangent, arrow, half_length, arrow_length in (
            (self.f, self.f_levels_box, self.f_field_box, F_TANGENT, F_ARROW,
             F_TANGENT_HALF_LENGTH, F_ARROW_LENGTH),
            (self.g, self.g_levels_box, self.g_field_box, G_TANGENT, G_ARROW,
             G_TANGENT_HALF_LENGTH, G_ARROW_LENGTH),
        ):
            grad = gradient(func, self.point)
            norm = np.linalg.norm(grad)
            if norm < 1e-8:
                continue
            direction = grad / norm
            if levels_box.value:
                along = half_length * np.array([-direction[1], direction[0]])
                ends = np.array([self.point - along, self.point + along])
                marks[tangent] = (ends[:, 0], ends[:, 1])
            if field_box.value:
                marks[arrow] = _arrow_path(self.point, direction, arrow_length)

        with self.fig.batch_update():
            for index, (xs, ys) in marks.items():
                self.fig.data[index].update(x=xs, y=ys)
            self.fig.data[POINT].update(x=[self.point[0]], y=[self.point[1]])

    def _set_popup(self, text: str = "", color: str = "#333333", size: int = 24):
        popup = self.fig.layout.annotations[1]
        popup.update(text=text, visible=bool(text), bordercolor=color, font_size=size)

    def _clear_result(self):
        self._feasibility = None
        self._maximality = None
        self.maximal_button.disabled = True
        self.result_html.value = ""
        self._set_popup()

    def _shown_mark(self, levels_box, field_box) -> Optional[str]:
        if levels_box.value:
            return "levels"
        if field_box.value:
            return "field"
        return None

    def _dual_feasibility_text(self, parallel: bool) -> str:
        """Describe dual feasibility in terms of the marks shown at the point."""
        f_mark = self._shown_mark(self.f_levels_box, self.f_field_box)
        g_mark = self._shown_mark(self.g_levels_box, self.g_field_box)
        is_ = "is" if parallel else "is not"
        are = "are" if parallel else "are not"
        if f_mark == g_mark == "levels":
            text = f"Level sets {are} parallel"
        elif (f_mark, g_mark) == ("levels", "field"):
            text = f"∇g {is_} perpendicular to the level set of f"
        elif (f_mark, g_mark) == ("field", "levels"):
            text = f"∇f {is_} perpendicular to the level set of g"
        else:
            text = f"Gradients {are} parallel"
        return ("✅ " if parallel else "❌ ") + text

    def _render_result(self):
        """Show the latest check results, worded to match the current display choices."""
        feasibility = self._feasibility
        if feasibility is None:
            return
        primal = "✅ In 𝒳" if feasibility.in_constraint else "❌ Not in 𝒳"
        dual = self._dual_feasibility_text(feasibility.parallel)
        message = self._maximality.message if self._maximality else feasibility.message
        note = f"<div style='margin-top:6px'>{message}</div>" if message else ""
        self.result_html.value = (
            "<div style='line-height:1.6'>"
            f"<div><b>Primal Feasibility:</b> {primal}</div>"
            f"<div><b>Dual Feasibility:</b> {dual}</div>{note}</div>"
        )
        verdict = self._maximality.verdict if self._maximality else feasibility.verdict
        colors = {"Correct!": "#2e7d32", "Feasible!": "#2e7d32",
                  "Almost!": "#b26a00", "Try again!": "#b3261e"}
        self._set_popup(f"<b>{verdict}</b>", colors[verdict])

    # Callbacks

    def _on_objective_change(self, change):
        self._filter_constraints()
        self._on_function_change(change)

    def _on_function_change(self, change):
        if self._filtering_constraints:
            return
        self._choose_functions()
        self._redraw_all()

    def _on_new_random(self, button):
        self._choose_functions()
        self._redraw_all()

    def _on_level_change(self, change):
        if self._setting_level_bounds:
            return
        # Only X moves with the level, so redraw just that while the slider is dragged.
        self._clear_result()
        self._update_constraint()
        self._show()

    def _redraw_all(self):
        self._clear_result()
        self._update_fields()
        self._update_point()
        self._show()

    def _exclusive_handler(self, partner):
        def handler(change):
            if change["new"] and partner.value:
                # Unticking the partner fires this handler again, with new=False.
                partner.value = False
            # The display choice does not change the answer, so keep any result
            # and reword it to match what is now shown.
            self._update_fields()
            self._update_point()
            self._render_result()
            self._show()
        return handler

    def _on_plot_click(self, trace, points, state):
        if self._blocked or not points.xs:
            return
        self._move_point(points.xs[0], points.ys[0])

    def _on_point_slider(self, change):
        if not self._syncing_point:
            self._move_point(self.x1_slider.value, self.x2_slider.value)

    def _move_point(self, x1, x2):
        self.point = np.array([round(float(x1), 2), round(float(x2), 2)])
        self._syncing_point = True
        self.x1_slider.value, self.x2_slider.value = self.point
        self._syncing_point = False
        self._clear_result()
        self._update_point()
        self._show()

    def _on_check_feasible(self, button):
        self._feasibility = check_feasibility(self.f, self.g, self.z_f, self._curves, self.point)
        self._maximality = None
        self.maximal_button.disabled = not self._feasibility.feasible
        self._render_result()
        self._show()

    def _on_check_maximal(self, button):
        if self._feasibility is None or not self._feasibility.feasible:
            return
        self._maximality = check_maximal(self.f, self._curves, self.point)
        self._render_result()
        self._show()

    # Display

    def _checkbox_grid(self):
        def label(html):
            return widgets.HTML(html, layout=widgets.Layout(width="auto"))

        cells = [
            label(""), label("<b>Level sets</b>"), label("<b>Gradient field</b>"),
            label(f"<b style='color:{F_COLOR}'>f</b>"), self.f_levels_box, self.f_field_box,
            label(f"<b style='color:{G_COLOR}'>g</b>"), self.g_levels_box, self.g_field_box,
        ]
        return widgets.GridBox(cells, layout=widgets.Layout(
            grid_template_columns="40px 110px 130px", width="330px", align_items="center",
        ))

    def _check_buttons(self):
        if not self.require_maximum:
            self.feasible_button.layout.width = "330px"
            return self.feasible_button
        return widgets.HBox([self.feasible_button, self.maximal_button],
                            layout=widgets.Layout(width="330px", justify_content="space-between"))

    def display(self):
        """Display the complete interface."""
        if self.clickable:
            how_to = "Click the plot to place your point, then use the sliders to fine-tune it."
            plot = self.fig
        else:
            how_to = ("Use the x₁ and x₂ sliders to place your point. "
                      "(Clicking on the plot needs the <code>anywidget</code> package.)")
            plot = self.plot_output
            self._show()

        key = (
            f"<div style='font-size:13px;color:#444;line-height:1.5'>"
            f"<span style='color:{F_COLOR}'>■</span> f, on a background where darker means larger<br>"
            f"<span style='color:{G_COLOR}'>■</span> g, with the constraint set 𝒳 drawn in bold"
            f"</div>"
        )
        controls = widgets.VBox(
            [
                self.f_dropdown, self.g_dropdown, self.level_slider, self.new_random_button,
                widgets.HTML("<b>Show</b>"), self._checkbox_grid(),
                widgets.HTML("<b>Your point</b>"), self.x1_slider, self.x2_slider,
                self._check_buttons(), self.result_html, widgets.HTML(key),
            ],
            layout=widgets.Layout(width="340px", min_width="340px", padding="0 8px 0 0"),
        )
        header = widgets.HTML(f"<b>Find the maximum of f on 𝒳 = {{x : g(x) = level}}.</b> {how_to}")
        display(widgets.VBox([
            header,
            # No wrapping: in a narrow notebook this scrolls sideways, so the controls
            # stay beside the plot rather than dropping below it.
            widgets.HBox([controls, plot], layout=widgets.Layout(flex_flow="row nowrap", overflow="auto")),
        ]))


def show_constrained_optimization(seed: Optional[int] = None, require_maximum: bool = True):
    """
    Display the constrained optimization demo.

    seed: fixes the random problems, for example so a class sees the same one.
    require_maximum: when True, a feasible point unlocks a "Check if Maximal"
        button, which earns "Correct!" only at the constrained maximum. When
        False, that button is hidden and passing the two Lagrange checks ends
        at "Feasible!", so constrained minima and saddles also pass.
    """
    demo = ConstrainedOptimizationDemo(seed=seed, require_maximum=require_maximum)
    demo.display()
    return demo
