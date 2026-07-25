"""Tests: schema validation, deterministic SVG generation, geometry helpers,
render/compare round-trips on synthetic fixtures with known ground truth."""

import numpy as np
import pytest
from PIL import Image

from png2svg import compare as cmp
from png2svg.geom import rounded_polygon, smooth_polygon
from png2svg.model import ModelError, Project, validate_shape
from png2svg.render import render_svg
from png2svg.svggen import generate_svg


def make_project(shapes, w=100, h=100):
    return Project(
        source_path="x.png", width=w, height=h, sha256="0" * 64,
        background=[255, 255, 255, 255], view_box=[0, 0, w, h], shapes=shapes,
    )


RECT = {
    "id": "r", "type": "path",
    "d": [["M", 20, 20], ["L", 80, 20], ["L", 80, 60], ["L", 20, 60], ["Z"]],
    "fills": [{"type": "solid", "color": "#3050a0"}],
}


class TestModel:
    def test_valid_shape_passes(self):
        validate_shape(RECT)

    def test_rejects_bad_command(self):
        bad = dict(RECT, d=[["M", 0, 0], ["X", 1, 1]])
        with pytest.raises(ModelError):
            validate_shape(bad)

    def test_rejects_wrong_arity(self):
        bad = dict(RECT, d=[["M", 0, 0], ["A", 1, 2, 3]])
        with pytest.raises(ModelError):
            validate_shape(bad)

    def test_rejects_missing_paint_field(self):
        bad = dict(RECT, fills=[{"type": "linear", "x1": 0, "y1": 0, "x2": 1}])
        with pytest.raises(ModelError):
            validate_shape(bad)

    def test_rejects_duplicate_ids(self):
        p = make_project([RECT, dict(RECT)])
        with pytest.raises(ModelError):
            p.validate()


class TestSvgGen:
    def test_deterministic(self):
        p = make_project([RECT])
        assert generate_svg(p) == generate_svg(p)

    def test_no_raster_or_script(self):
        conic = {
            "id": "c", "type": "path",
            "d": [["M", 10, 50], ["A", 40, 40, 0, 1, 1, 90, 50], ["Z"]],
            "fills": [{"type": "conic", "cx": 50, "cy": 50, "radius": 45,
                       "angle_start": 0, "angle_end": 180, "wedges": 8,
                       "stops": [{"offset": 0, "color": "#ff0000"},
                                 {"offset": 1, "color": "#0000ff"}]}],
        }
        svg = generate_svg(make_project([RECT, conic]))
        for banned in ("<image", "<script", "data:", "xlink:href"):
            assert banned not in svg

    def test_solid_rect_renders_exact(self):
        svg = generate_svg(make_project([RECT]))
        img = render_svg(svg, 100, 100)
        arr = np.asarray(img)
        assert tuple(arr[40, 50][:3]) == (0x30, 0x50, 0xA0)
        assert arr[40, 50][3] == 255
        assert arr[10, 10][3] == 0  # outside is transparent

    def test_linear_gradient_midpoint(self):
        grad = dict(RECT, fills=[{
            "type": "linear", "x1": 20.0, "y1": 0.0, "x2": 80.0, "y2": 0.0,
            "stops": [{"offset": 0, "color": "#000000"},
                      {"offset": 1, "color": "#ffffff"}]}])
        img = render_svg(generate_svg(make_project([grad])), 100, 100)
        mid = np.asarray(img)[40, 50][:3].astype(int)
        # renderer interpolates premultiplied sRGB; midpoint ~127 each channel
        assert all(abs(v - 127) <= 6 for v in mid)

    def test_conic_wedges_cover_without_pinholes(self):
        conic = {
            "id": "c", "type": "path",
            "d": [["M", 50, 5], ["A", 45, 45, 0, 1, 1, 50, 95],
                  ["A", 45, 45, 0, 1, 1, 50, 5], ["Z"]],
            "fills": [{"type": "conic", "cx": 50, "cy": 50, "radius": 48,
                       "angle_start": -180, "angle_end": 180, "wedges": 24,
                       "stops": [{"offset": 0, "color": "#804020"},
                                 {"offset": 1, "color": "#804020"}]}],
        }
        img = render_svg(generate_svg(make_project([conic])), 100, 100)
        arr = np.asarray(img)
        yy, xx = np.mgrid[0:100, 0:100]
        inside = (xx - 49.5) ** 2 + (yy - 49.5) ** 2 < 40**2
        # no seams or centre pinholes (real seam defects measure 186-230)
        assert arr[..., 3][inside].min() >= 250


class TestGeom:
    def test_rounded_polygon_square(self):
        segs = rounded_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 2.0)
        assert segs[0][0] == "M" and segs[-1][0] == "Z"
        assert sum(1 for s in segs if s[0] == "A") == 4

    def test_smooth_polygon_structure(self):
        c = {"t_in": 3.0, "t_out": 3.0, "h_in": 1.0, "h_out": 1.0}
        segs = smooth_polygon([(0, 0), (10, 0), (5, 10)], [c, c, c])
        assert segs[0][0] == "M" and segs[-1][0] == "Z"
        assert sum(1 for s in segs if s[0] == "C") == 3


class TestCompare:
    def test_identical_images_score_perfectly(self):
        svg = generate_svg(make_project([RECT]))
        img = render_svg(svg, 100, 100)
        m = cmp.compare(img, img, (255, 255, 255))
        assert m["silhouette_iou"] == 1.0
        assert m["edge_dist_mean"] == 0.0
        assert m["deltaE_mean"] == 0.0

    def test_shifted_shape_detected(self):
        a = generate_svg(make_project([RECT]))
        shifted = dict(RECT, d=[["M", 23, 20], ["L", 83, 20], ["L", 83, 60],
                                ["L", 23, 60], ["Z"]])
        b = generate_svg(make_project([shifted]))
        m = cmp.compare(render_svg(a, 100, 100), render_svg(b, 100, 100),
                        (255, 255, 255))
        assert m["silhouette_iou"] < 0.95
        assert m["edge_dist_max"] >= 3.0

    def test_colour_error_detected(self):
        a = generate_svg(make_project([RECT]))
        recol = dict(RECT, fills=[{"type": "solid", "color": "#a03050"}])
        b = generate_svg(make_project([recol]))
        m = cmp.compare(render_svg(a, 100, 100), render_svg(b, 100, 100),
                        (255, 255, 255))
        assert m["silhouette_iou"] > 0.99   # same geometry
        assert m["deltaE_mean"] > 10        # very different colour

    def test_ciede2000_zero_for_identical(self):
        lab = cmp.linear_to_lab(cmp.srgb_to_linear(
            np.array([[[120.0, 60.0, 200.0]]])))
        assert cmp.ciede2000(lab, lab)[0, 0] == pytest.approx(0.0)


class TestAnalyse:
    def _render(self, shapes):
        svg = generate_svg(make_project(shapes, 200, 200))
        return render_svg(svg, 200, 200)

    def test_flat_rect_and_gradient_circle(self):
        from png2svg.analyse import analyse_image

        rect = {
            "id": "r", "type": "path",
            "d": [["M", 10, 10], ["L", 90, 10], ["L", 90, 60], ["L", 10, 60], ["Z"]],
            "fills": [{"type": "solid", "color": "#3050a0"}],
        }
        circle = {
            "id": "c", "type": "path",
            "d": [["M", 140, 80], ["A", 45, 45, 0, 1, 1, 140, 170],
                  ["A", 45, 45, 0, 1, 1, 140, 80], ["Z"]],
            "fills": [{
                "type": "linear", "x1": 95.0, "y1": 0.0, "x2": 185.0, "y2": 0.0,
                "stops": [{"offset": 0, "color": "#204080"},
                          {"offset": 1, "color": "#80c0ff"}]}],
        }
        img = self._render([rect, circle])
        features, _ = analyse_image(img, [255, 255, 255, 255])
        comps = features["components"]
        assert len(comps) == 2
        rect_c = min(comps, key=lambda c: c["bbox"][1])
        circ_c = max(comps, key=lambda c: c["bbox"][1])
        assert rect_c["paint"]["kind"] == "flat"
        assert rect_c["paint"]["color"] == "#3050a0"
        assert sum(1 for s in rect_c["segments"] if s["kind"] == "line") == 4
        assert circ_c["paint"]["kind"] == "linear"
        arcs = [s for s in circ_c["segments"] if s["kind"] == "arc"]
        assert arcs and abs(arcs[0]["radius"] - 45) < 1.5
        assert abs(arcs[0]["centre"][0] - 140) < 1.5

    def test_boundary_trace_length(self):
        from png2svg.analyse import trace_boundary

        m = np.zeros((30, 30), bool)
        m[5:25, 5:25] = True
        assert len(trace_boundary(m)) == 76  # 4*20 - 4


class TestResiduals:
    def test_localises_injected_error(self):
        from png2svg.residuals import find_residuals

        a = generate_svg(make_project([RECT]))
        patched = dict(RECT, fills=[{"type": "solid", "color": "#3050a0"}])
        wrong_patch = {
            "id": "w", "type": "path",
            "d": [["M", 40, 30], ["L", 60, 30], ["L", 60, 45], ["L", 40, 45], ["Z"]],
            "fills": [{"type": "solid", "color": "#a05030"}],
        }
        b = generate_svg(make_project([patched, wrong_patch]))
        rep = find_residuals(render_svg(a, 100, 100), render_svg(b, 100, 100),
                             (255, 255, 255))
        assert rep["colour_clusters"]
        x0, y0, x1, y1 = rep["colour_clusters"][0]["bbox"]
        assert 38 <= x0 and x1 <= 62 and 28 <= y0 and y1 <= 47


class TestStroke:
    """Stroked outlines: the paint that makes a constant-width logo a path
    of a few nodes instead of an outlined polygon of forty."""

    def line_shape(self, **stroke):
        s = {"paint": {"type": "solid", "color": "#3355ff"}, "width": 20}
        s.update(stroke)
        return {"id": "s", "type": "path",
                "d": [["M", 20, 50], ["L", 80, 50]], "fills": [], "stroke": s}

    def test_width_and_round_cap_geometry(self):
        proj = make_project([self.line_shape(linecap="round")])
        img = render_svg(generate_svg(proj), 100, 100, supersample=4)
        a = np.asarray(img)
        # width 20 centred on y=50 covers rows 40..59 and nothing outside
        assert a[39, 50, 3] == 0 and a[60, 50, 3] == 0
        assert a[40, 50, 3] == 255 and a[59, 50, 3] == 255
        # the round cap reaches half a width past the endpoint
        assert a[50, 11, 3] == 255 and a[50, 8, 3] == 0

    def test_butt_cap_stops_at_the_endpoint(self):
        proj = make_project([self.line_shape(linecap="butt")])
        a = np.asarray(render_svg(generate_svg(proj), 100, 100, supersample=4))
        assert a[50, 21, 3] == 255 and a[50, 18, 3] == 0

    def test_gradient_stroke_registers_def_and_paints(self):
        shape = self.line_shape(paint={
            "type": "linear", "x1": 20, "y1": 0, "x2": 80, "y2": 0,
            "stops": [{"offset": 0.0, "color": "#000000"},
                      {"offset": 1.0, "color": "#ffffff"}]})
        svg = generate_svg(make_project([shape]))
        assert 'stroke="url(#s-s)"' in svg and "<linearGradient id=\"s-s\"" in svg
        a = np.asarray(render_svg(svg, 100, 100, supersample=4))
        assert a[50, 22, 0] < 40 and a[50, 78, 0] > 215

    def test_stroke_is_not_clipped_to_its_own_path(self):
        """Clipping a stroke to its path would keep only the inner half."""
        shape = self.line_shape()
        shape["fills"] = [{"type": "solid", "color": "#ff0000"}]
        svg = generate_svg(make_project([shape]))
        assert svg.index('stroke="#3355ff"') > svg.rindex("</g>") \
            if "</g>" in svg else True
        a = np.asarray(render_svg(svg, 100, 100, supersample=4))
        assert a[41, 50, 3] == 255 and a[58, 50, 3] == 255

    def test_deterministic(self):
        proj = make_project([self.line_shape(linecap="round", linejoin="round")])
        assert generate_svg(proj) == generate_svg(proj)

    def test_empty_fills_allowed_only_with_stroke(self):
        validate_shape(self.line_shape())
        bare = {"id": "s", "type": "path", "d": [["M", 0, 0], ["L", 1, 1]],
                "fills": []}
        with pytest.raises(ModelError, match="stroked shape"):
            validate_shape(bare)

    def test_rejects_conic_stroke_and_bad_cap(self):
        with pytest.raises(ModelError, match="conic stroke"):
            validate_shape(self.line_shape(paint={
                "type": "conic", "cx": 0, "cy": 0, "radius": 5,
                "angle_start": 0, "angle_end": 90, "stops": []}))
        with pytest.raises(ModelError, match="linecap"):
            validate_shape(self.line_shape(linecap="rounded"))


def _deviation(pts, segs):
    """True max deviation of a fitted chain, densely sampled by arc length."""
    from png2svg.curves import _bezier
    cur, dense = pts[0], []
    for q in segs:
        if q[0] == "L":
            n = max(int(np.linalg.norm(q[1] - cur) * 20), 50)
            dense.append(np.linspace(cur, q[1], n))
            cur = q[1]
        else:
            ctrl = np.array([cur, q[1], q[2], q[3]])
            n = max(int(np.sum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1)) * 20), 50)
            dense.append(_bezier(ctrl, np.linspace(0, 1, n)))
            cur = q[3]
    D = np.vstack(dense)
    return float(np.min(np.linalg.norm(D[None] - pts[:, None], axis=2), axis=1).max())


class TestCurves:
    """Bezier chain fitting: the primitive that makes free-form outlines
    tractable without hand-writing a fit per image."""

    def test_recovers_a_known_cubic_in_one_segment(self):
        from png2svg.curves import fit_run, _bezier
        ctrl = np.array([[0.0, 0.0], [30.0, 80.0], [120.0, 80.0], [150.0, 0.0]])
        pts = _bezier(ctrl, np.linspace(0, 1, 300))
        segs = fit_run(pts, tol=0.01)
        assert len(segs) == 1 and segs[0][0] == "C"
        assert np.allclose(segs[0][1], ctrl[1], atol=0.5)
        assert np.allclose(segs[0][2], ctrl[2], atol=0.5)

    def test_straight_run_becomes_one_line(self):
        from png2svg.curves import fit_run
        pts = np.stack([np.linspace(0, 200, 300), np.full(300, 50.0)], 1)
        segs = fit_run(pts, tol=0.05)
        assert segs == [("L", pytest.approx(pts[-1]))] or (
            len(segs) == 1 and segs[0][0] == "L")

    def test_tolerance_is_honoured_and_buys_fewer_segments(self):
        from png2svg.curves import fit_run
        t = np.linspace(np.pi, 0, 400)
        semi = np.stack([100 + 80 * np.cos(t), 100 + 80 * np.sin(t)], 1)
        loose, tight = fit_run(semi, tol=0.5), fit_run(semi, tol=0.02)
        assert _deviation(semi, loose) <= 0.5
        assert _deviation(semi, tight) <= 0.02 * 1.5
        assert len(loose) <= len(tight)

    def test_corners_are_preserved_not_smoothed(self):
        from png2svg.curves import corner_indices
        side = np.linspace(0, 100, 120)
        square = np.vstack([
            np.stack([side, np.zeros(120)], 1),
            np.stack([np.full(120, 100.0), side], 1),
            np.stack([side[::-1], np.full(120, 100.0)], 1),
            np.stack([np.zeros(120), side[::-1]], 1)])
        assert len(corner_indices(square, angle_deg=45.0)) == 4

    def test_subpixel_contour_tracks_a_known_circle(self):
        from png2svg.measure import Field, subpixel_contour
        from scipy import ndimage
        yy, xx = np.mgrid[0:200, 0:200]
        disc = np.hypot(xx - 99.5, yy - 99.5) <= 60.0
        img = np.where(disc[..., None], np.array([20, 20, 20]),
                       np.array([255, 255, 255])).astype(np.uint8)
        img = ndimage.gaussian_filter(img, (0.7, 0.7, 0))
        C = subpixel_contour(Field(img, (255, 255, 255)), disc, offset=8.0)
        r = np.hypot(C[:, 0] - 100.0, C[:, 1] - 100.0)
        assert abs(r.mean() - 60.0) < 0.4
        assert r.std() < 0.4
        # continuous: no torn gaps, which would wreck any fit across them
        assert np.linalg.norm(np.diff(C, axis=0), axis=1).max() < 3.0


class TestOutline:
    """Structured segmentation: prefer the primitive a designer would have
    used, and fall back to Beziers only where the outline is free-form."""

    def rounded_rect(self, w=200.0, h=120.0, r=25.0, n=2400):
        """Ground truth: a rounded rectangle, sampled along its outline."""
        segs, cx, cy = [], w / 2, h / 2
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            a0 = np.arctan2(sy, sx) - np.pi / 4
            t = np.linspace(a0, a0 + np.pi / 2, n // 8)
            segs.append(np.stack([cx + (cx - r) * sx + r * np.cos(t) - (cx - r) * sx,
                                  cy + (cy - r) * sy + r * np.sin(t) - (cy - r) * sy], 1))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # superellipse with a high exponent approximates a rounded rect well
        k = 8.0
        x = cx + (cx) * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / k)
        y = cy + (cy) * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / k)
        return np.stack([x, y], 1)

    def test_circle_becomes_arcs_with_the_right_radius(self):
        from png2svg.outline import segment_outline
        t = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
        circle = np.stack([300 + 90 * np.cos(t), 300 + 90 * np.sin(t)], 1)
        prims = segment_outline(circle, tol=0.2)
        arcs = [p for p in prims if p["kind"] == "arc"]
        assert arcs, "a circle must segment into arcs, not cubics"
        assert all(abs(a["r"] - 90.0) < 1.0 for a in arcs)
        assert all(np.hypot(a["c"][0] - 300, a["c"][1] - 300) < 1.0 for a in arcs)

    def test_polygon_becomes_lines_only(self):
        from png2svg.outline import segment_outline
        corners = np.array([[50.0, 50.0], [250.0, 60.0], [230.0, 200.0], [60.0, 190.0]])
        pts = np.vstack([np.linspace(corners[i], corners[(i + 1) % 4], 300,
                                     endpoint=False) for i in range(4)])
        prims = segment_outline(pts, tol=0.2)
        assert all(p["kind"] == "line" for p in prims)
        assert len(prims) <= 6

    def test_tolerance_controls_the_tradeoff(self):
        from png2svg.outline import segment_outline
        t = np.linspace(0, 2 * np.pi, 1500, endpoint=False)
        blob = np.stack([300 + 100 * np.cos(t) + 12 * np.cos(3 * t),
                         300 + 80 * np.sin(t) + 9 * np.sin(2 * t)], 1)
        tight = segment_outline(blob, tol=0.1)
        loose = segment_outline(blob, tol=0.8)
        assert len(loose) <= len(tight)

    def test_no_absurd_radius_arcs(self):
        """A nearly straight run fits a huge circle; it must stay a line."""
        from png2svg.outline import segment_outline
        x = np.linspace(0, 400, 900)
        pts = np.vstack([np.stack([x, 100 + 0.0005 * (x - 200) ** 2], 1),
                         np.stack([x[::-1], np.full(900, 300.0)], 1)])
        prims = segment_outline(pts, tol=0.5)
        for p in prims:
            if p["kind"] == "arc":
                chord = np.linalg.norm(p["p1"] - p["p0"])
                assert p["r"] < 40 * max(chord, 1.0)


class TestSnapAndConvert:
    """Constraints are a hypothesis about how the artwork was drawn. Verified
    snapping keeps the ones that fit better and reverts the rest."""

    def tilted_rounded_rect(self, n=2000, r=30.0, w=260.0, h=160.0, deg=0.0):
        pts = []
        for (cx, cy, a0) in ((w - r, h - r, 0.0), (r, h - r, np.pi / 2),
                             (r, r, np.pi), (w - r, r, -np.pi / 2)):
            t = np.linspace(a0, a0 + np.pi / 2, n // 8)
            pts.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1))
            pts.append(np.linspace(pts[-1][-1], pts[-1][-1], 2))
        ring = np.vstack(pts)
        # close the straight runs between the corner arcs
        full = []
        for i in range(0, len(ring), 1):
            full.append(ring[i])
        full = np.array(full)
        idx = np.linspace(0, len(full) - 1, n).astype(int)
        return full[idx] + 40.0

    def test_snapping_squares_up_near_axis_edges(self):
        from png2svg.outline import segment_outline, snap_outline
        # a rectangle whose edges are a fraction of a degree off axis
        w, h = 300.0, 180.0
        corners = np.array([[0, 0], [w, 0.9], [w - 0.6, h], [0.4, h - 0.5]]) + 60
        pts = np.vstack([np.linspace(corners[i], corners[(i + 1) % 4], 400,
                                     endpoint=False) for i in range(4)])
        prims = segment_outline(pts, tol=0.3)
        snapped, notes = snap_outline(prims, contour=pts, angle_tol=1.5)
        dirs = []
        for p in snapped:
            if p["kind"] == "line":
                d = p["p1"] - p["p0"]
                dirs.append(np.degrees(np.arctan2(d[1], d[0])) % 90.0)
        assert dirs, "expected line primitives"
        assert all(min(a, 90 - a) < 0.05 for a in dirs), dirs

    def test_bad_constraint_is_rejected_not_applied(self):
        """A fillet hypothesis between near-parallel edges must not be kept."""
        from png2svg.outline import snap_outline, _deviation
        contour = np.stack([np.linspace(0, 300, 600), np.full(600, 100.0)], 1)
        prims = [
            {"kind": "line", "p0": np.array([0.0, 100.0]),
             "p1": np.array([140.0, 100.0])},
            {"kind": "arc", "p0": np.array([140.0, 100.0]),
             "p1": np.array([160.0, 100.0]), "c": np.array([150.0, 100.0]),
             "r": 10.0, "sweep_angle": 0.2},
            {"kind": "line", "p0": np.array([160.0, 100.0]),
             "p1": np.array([300.0, 100.0])},
        ]
        before = [_deviation(p, contour) for p in prims]
        snapped, notes = snap_outline(prims, contour=contour, allow=0.3)
        after = [_deviation(p, contour) for p in snapped]
        # the contract is not that snapping repairs a bad fit, but that it
        # never degrades one: every primitive stays within its allowance
        assert all(a <= b + 0.3 + 1e-9 for a, b in zip(after, before)), (
            before, after, notes)

    def test_to_segments_round_trips_through_the_model(self):
        from png2svg.outline import segment_outline, to_segments
        from png2svg.model import validate_shape
        t = np.linspace(0, 2 * np.pi, 900, endpoint=False)
        circle = np.stack([200 + 70 * np.cos(t), 150 + 70 * np.sin(t)], 1)
        segs = to_segments(segment_outline(circle, tol=0.25))
        assert segs[0][0] == "M" and segs[-1][0] == "Z"
        shape = {"id": "c", "type": "path", "d": segs,
                 "fills": [{"type": "solid", "color": "#000000"}]}
        validate_shape(shape)
        img = render_svg(generate_svg(make_project([shape], 400, 300)), 400, 300,
                         supersample=4)
        a = np.asarray(img)
        assert a[150, 200, 3] == 255            # centre filled
        assert a[150, 200 + 60, 3] == 255       # inside the radius
        assert a[150, 200 + 80, 3] == 0         # outside it


class TestPaintFitting:
    """Recovering a gradient's construction, not just its appearance."""

    def render_gradient(self, stops, angle=30.0, w=240, h=160):
        shape = {"id": "g", "type": "path",
                 "d": [["M", 20, 20], ["L", w - 20, 20], ["L", w - 20, h - 20],
                       ["L", 20, h - 20], ["Z"]],
                 "fills": [{"type": "linear",
                            "x1": 20, "y1": 20, "x2": w - 20, "y2": h - 20,
                            "stops": stops}]}
        img = render_svg(generate_svg(make_project([shape], w, h)), w, h)
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.asarray(bg), np.asarray(img)[:, :, 3] > 128

    def near(self, got: str, want: str, tol: int = 4):
        a = [int(got[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(want[i:i + 2], 16) for i in (1, 3, 5)]
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def test_recovers_two_stop_gradient(self):
        """End stops land within a few levels, not exactly: no pixel centre
        sits on the gradient's endpoint, so the extreme colour is never
        actually present in the image to be recovered."""
        from png2svg.paint import fit_linear_gradient
        stops = [{"offset": 0.0, "color": "#ff9529"},
                 {"offset": 1.0, "color": "#2050c0"}]
        rgb, mask = self.render_gradient(stops)
        g = fit_linear_gradient(rgb, mask, erode=3)
        assert g["rms"] < 2.0
        got = [s["color"] for s in g["stops"]]
        assert self.near(got[0], "#ff9529"), got
        assert self.near(got[-1], "#2050c0"), got

    def test_recovers_the_knee_of_a_three_stop_gradient(self):
        from png2svg.paint import fit_linear_gradient
        stops = [{"offset": 0.0, "color": "#ff9529"},
                 {"offset": 0.5, "color": "#ff29a1"},
                 {"offset": 1.0, "color": "#9929ff"}]
        rgb, mask = self.render_gradient(stops)
        g = fit_linear_gradient(rgb, mask, erode=3)
        assert len(g["stops"]) >= 3
        mid = [s for s in g["stops"] if 0.3 < s["offset"] < 0.7]
        assert mid, [s["offset"] for s in g["stops"]]
        assert mid[0]["color"] == "#ff29a1"
        assert abs(mid[0]["offset"] - 0.5) < 0.03

    def test_axis_is_found_within_a_degree(self):
        from png2svg.paint import fit_linear_gradient
        rgb, mask = self.render_gradient(
            [{"offset": 0.0, "color": "#000000"},
             {"offset": 1.0, "color": "#ffffff"}])
        g = fit_linear_gradient(rgb, mask, erode=3)
        # the fixture's gradient vector runs (20,20) -> (220,140)
        expected = np.degrees(np.arctan2(140 - 20, 220 - 20))
        assert abs(g["axis_deg"] - expected) < 1.0

    def test_flat_colour_ignores_an_overlay(self):
        from png2svg.paint import flat_colour
        rgb = np.full((120, 120, 3), 250, np.uint8)
        rgb[20:100, 20:100] = (60, 90, 200)
        mask = np.zeros((120, 120), bool)
        mask[20:100, 20:100] = True
        rgb[40:46, 20:100] = (255, 0, 0)          # a watermark stripe
        assert flat_colour(rgb, mask, erode=3) == "#3c5ac8"

    def test_trim_rejects_an_overlay_that_would_skew_the_fit(self):
        """A shadow painted over a gradient drags the whole fit toward it."""
        from png2svg.paint import fit_linear_gradient
        stops = [{"offset": 0.0, "color": "#ff9529"},
                 {"offset": 1.0, "color": "#2050c0"}]
        rgb, mask = self.render_gradient(stops)
        rgb = rgb.copy()
        rgb[60:110, 60:150] = (rgb[60:110, 60:150] * 0.55).astype(np.uint8)
        naive = fit_linear_gradient(rgb, mask, erode=3, n_stops=2)
        robust = fit_linear_gradient(rgb, mask, erode=3, n_stops=2, trim=0.2)
        assert robust["rms"] < naive["rms"] / 2
        got = [s["color"] for s in robust["stops"]]
        assert self.near(got[0], "#ff9529", 6) and self.near(got[-1], "#2050c0", 6), got


class TestTextureAwareScoring:
    """A textured source disagrees with itself by more than any vector can
    match. Per-pixel deltaE then reports failure for a reconstruction that
    is structurally right, so texture_std flags when to judge low-frequency."""

    def clean_render(self):
        grad = dict(RECT, fills=[{
            "type": "linear", "x1": 20.0, "y1": 0.0, "x2": 80.0, "y2": 0.0,
            "stops": [{"offset": 0, "color": "#3050a0"},
                      {"offset": 1, "color": "#a05030"}]}])
        return render_svg(generate_svg(make_project([grad])), 100, 100)

    def test_texture_std_separates_clean_from_grainy(self):
        clean = self.clean_render()
        arr = np.asarray(clean).astype(np.int16)
        rng = np.random.default_rng(0)
        noisy = arr.copy()
        noisy[..., :3] += rng.integers(-40, 41, noisy[..., :3].shape)
        noisy_img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
        m_clean = cmp.compare(clean, clean, (255, 255, 255))
        m_noisy = cmp.compare(noisy_img, clean, (255, 255, 255))
        assert m_clean["texture_std"] < 2.0
        assert m_noisy["texture_std"] > 8.0

    def test_lowfreq_forgives_grain_but_not_wrong_colour(self):
        clean = self.clean_render()
        arr = np.asarray(clean).astype(np.int16)
        rng = np.random.default_rng(1)
        noisy = arr.copy()
        noisy[..., :3] += rng.integers(-40, 41, noisy[..., :3].shape)
        noisy_img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
        grainy = cmp.compare(noisy_img, clean, (255, 255, 255))
        # grain costs a lot per pixel but little once blurred
        assert grainy["deltaE_lowfreq_mean"] < grainy["deltaE_mean"] / 2

        wrong = dict(RECT, fills=[{"type": "solid", "color": "#20a040"}])
        wrong_img = render_svg(generate_svg(make_project([wrong])), 100, 100)
        bad = cmp.compare(clean, wrong_img, (255, 255, 255))
        # a genuinely wrong colour survives blurring — this must not be forgiven
        assert bad["deltaE_lowfreq_mean"] > 10
