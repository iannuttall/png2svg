"""Tests: schema validation, deterministic SVG generation, geometry helpers,
render/compare round-trips on synthetic fixtures with known ground truth."""

import numpy as np
import pytest
from PIL import Image

from png2svg import compare as cmp
from png2svg import primitives as prim
from png2svg.geom import path_bounds, rounded_polygon, smooth_polygon
from png2svg.model import ModelError, Project, validate_shape
from png2svg.render import render_svg
from png2svg.svggen import generate_svg, svg_stats, tight_view_box


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

    def test_rejects_id_that_cannot_be_an_svg_animation_target(self):
        bad = dict(RECT, id="1 bad id")
        with pytest.raises(ModelError, match="XML-safe"):
            validate_shape(bad)


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

    def test_conic_wedges_share_one_ramp_definition(self):
        conic = {
            "id": "c", "type": "path", "d": RECT["d"],
            "fills": [{"type": "conic", "cx": 50, "cy": 50, "radius": 45,
                       "angle_start": 0, "angle_end": 180, "wedges": 8,
                       "stops": [{"offset": 0, "color": "#ff0000"},
                                 {"offset": 1, "color": "#0000ff"}]}],
        }
        svg = generate_svg(make_project([conic]))
        assert svg.count("<stop ") == 2
        assert svg.count('href="#_g0r"') == 8

    def test_export_profiles_preserve_pixels_and_change_structure(self):
        project = make_project([RECT])
        semantic = generate_svg(project, profile="semantic")
        compact = generate_svg(project, profile="compact")
        animation = generate_svg(project, profile="animation")
        assert '<path id="r"' in semantic
        assert 'id="r"' not in compact
        assert '<g id="r">' in animation
        assert len(compact) < len(semantic)
        a = np.asarray(render_svg(semantic, 100, 100))
        b = np.asarray(render_svg(compact, 100, 100))
        c = np.asarray(render_svg(animation, 100, 100))
        assert np.array_equal(a, b) and np.array_equal(a, c)
        assert svg_stats(compact)["paths"] == 1

    def test_animation_id_wraps_a_whole_fill_stack_and_stroke_once(self):
        shape = dict(
            RECT,
            fills=[
                {"type": "solid", "color": "#3050a0"},
                {"type": "solid", "color": "#ffffff", "opacity": 0.2,
                 "rect": [20, 20, 60, 20]},
            ],
            stroke={
                "paint": {"type": "solid", "color": "#101010"},
                "width": 2,
            },
        )
        svg = generate_svg(make_project([shape]), profile="animation")
        assert svg.count('id="r"') == 1
        assert svg.index('<g id="r">') < svg.index("clip-path")
        assert svg.index('stroke="#101010"') < svg.rindex("</g>")

    def test_internal_definition_ids_cannot_collide_with_shape_ids(self):
        import xml.etree.ElementTree as ET

        gradient = dict(
            RECT,
            id="a",
            fills=[{
                "type": "linear", "x1": 0, "y1": 0, "x2": 100, "y2": 0,
                "stops": [{"offset": 0, "color": "#000000"},
                          {"offset": 1, "color": "#ffffff"}],
            }],
        )
        second = dict(RECT, id="a-p0")
        root = ET.fromstring(generate_svg(make_project([gradient, second])))
        ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
        assert len(ids) == len(set(ids))
        assert {"a", "a-p0", "_g0"} <= set(ids)

    def test_halo_checker_accepts_a_fully_opaque_svg(self):
        from png2svg.cli import _halo_free

        rgba = np.full((20, 20, 4), 255, np.uint8)
        rgba[..., :3] = (30, 60, 90)
        assert _halo_free(rgba, (255, 255, 255), bright=True)
        assert _halo_free(rgba, (0, 0, 0), bright=False)


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

    def test_path_bounds_solves_curve_extrema(self):
        quadratic = [["M", 0, 0], ["Q", 50, 100, 100, 0]]
        assert np.allclose(path_bounds(quadratic), (0, 0, 100, 50))
        cubic = [["M", 0, 0], ["C", 0, 90, 100, 90, 100, 0]]
        assert np.allclose(path_bounds(cubic), (0, 0, 100, 67.5))

    def test_path_bounds_solves_arc_extrema(self):
        circle = [
            ["M", 50, 0],
            ["A", 50, 50, 0, 0, 1, 50, 100],
            ["A", 50, 50, 0, 0, 1, 50, 0],
            ["Z"],
        ]
        assert np.allclose(path_bounds(circle), (0, 0, 100, 100), atol=1e-10)

        phi = np.radians(30)
        p0 = (100 + 40 * np.cos(phi), 100 + 40 * np.sin(phi))
        p1 = (100 - 40 * np.cos(phi), 100 - 40 * np.sin(phi))
        rotated = [
            ["M", *p0], ["A", 40, 20, 30, 0, 1, *p1],
            ["A", 40, 20, 30, 0, 1, *p0], ["Z"],
        ]
        dx = np.hypot(40 * np.cos(phi), 20 * np.sin(phi))
        dy = np.hypot(40 * np.sin(phi), 20 * np.cos(phi))
        assert np.allclose(path_bounds(rotated),
                           (100 - dx, 100 - dy, 100 + dx, 100 + dy))

    def test_tight_viewbox_uses_model_paths_not_raster_sampling(self):
        assert np.allclose(
            tight_view_box(make_project([RECT]), padding=2),
            [18, 18, 64, 44],
        )


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

    def test_foreground_mask_does_not_cut_a_channel_at_a_colour_seam(self):
        rgb = np.full((80, 80, 3), 255, np.uint8)
        rgb[10:70, 10:40] = (235, 110, 220)
        rgb[10:70, 40:70] = (45, 55, 180)
        rgb[10:70, 39] = (190, 96, 210)
        rgb[10:70, 40] = (90, 67, 190)
        mask = cmp.foreground_mask(Image.fromarray(rgb), (255, 255, 255))
        assert mask[30, 20] and mask[30, 39] and mask[30, 40] and mask[30, 60]
        assert not mask[5, 40] and not mask[75, 40]


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
        assert 'stroke="url(#_g0)"' in svg
        assert '<linearGradient id="_g0"' in svg
        a = np.asarray(render_svg(svg, 100, 100, supersample=4))
        assert a[50, 22, 0] < 40 and a[50, 78, 0] > 215

    def test_stroke_is_not_clipped_to_its_own_path(self):
        """Clipping a stroke to its path would keep only the inner half."""
        shape = self.line_shape()
        shape["fills"] = [{"type": "solid", "color": "#ff0000"}]
        svg = generate_svg(make_project([shape]))
        assert "clip-path" not in svg
        assert svg.index('stroke="#3355ff"') > svg.index('fill="#ff0000"')
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

    def test_subpixel_contour_stops_before_a_nearby_component(self):
        from png2svg.measure import Field, subpixel_contour

        img = np.full((64, 64, 3), 255, np.uint8)
        img[10:54, 8:24] = 20
        img[10:54, 31:47] = 20
        mask = np.zeros((64, 64), bool)
        mask[10:54, 8:24] = True
        contour = subpixel_contour(Field(img, (255, 255, 255)), mask, offset=12)
        jumps = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
        assert jumps.max() < 2.0
        assert np.allclose(contour.min(axis=0), (8, 10), atol=0.25)
        assert np.allclose(contour.max(axis=0), (24, 54), atol=0.25)

    def test_subpixel_contour_normalises_a_low_contrast_edge(self):
        from png2svg.measure import Field, subpixel_contour

        img = np.full((60, 60, 3), 255, np.uint8)
        img[12:48, 14:46] = 230
        mask = np.zeros((60, 60), bool)
        mask[12:48, 14:46] = True
        contour = subpixel_contour(Field(img, (255, 255, 255)), mask)
        assert np.allclose(contour.min(axis=0), (14, 12), atol=0.25)
        assert np.allclose(contour.max(axis=0), (46, 48), atol=0.25)

    def test_edge_samples_choose_background_connected_to_the_edge(self):
        from png2svg.measure import Field, edge_samples

        img = np.full((70, 70, 3), 255, np.uint8)
        img[15:55, 10:60] = 30
        img[2:9, 10:60] = 30
        points = edge_samples(
            Field(img, (255, 255, 255)),
            (15, 15),
            (55, 15),
            offset=14,
            count=12,
        )
        assert len(points) == 12
        assert np.max(np.abs(points[:, 1] - 15)) < 0.1

    def test_edge_point_finds_the_first_side_of_a_thin_stroke(self):
        from png2svg.measure import Field, edge_point
        from scipy import ndimage

        img = np.full((40, 40, 3), 255, np.uint8)
        img[15:18, 5:35] = 20
        img = ndimage.gaussian_filter(img, (0.6, 0.6, 0))
        point = edge_point(Field(img, (255, 255, 255)), (20, 10), (0, 1), 15)
        assert point is not None
        assert point[0] == pytest.approx(20.5)
        assert point[1] == pytest.approx(15.0, abs=0.1)

    def test_edge_point_skips_a_short_resampling_ring_before_the_edge(self):
        from png2svg.measure import Field, edge_point

        # The small bump crosses the contrast threshold for one sample but is
        # followed by clean background. The sustained ramp is the real edge.
        values = np.zeros(60, dtype=np.uint8)
        values[21:24] = 8
        values[25:31] = [15, 35, 65, 95, 120, 140]
        values[31:] = 140
        img = np.zeros((60, 40, 3), dtype=np.uint8)
        img[:, 5:35] = values[:, None, None]
        point = edge_point(Field(img, (0, 0, 0)), (20, 0), (0, 1), 50)
        assert point is not None
        assert point[1] == pytest.approx(27.67, abs=0.15)


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


class TestFillRule:
    """A real hole in a compound path — the only cut that survives being
    recoloured or composited onto something other than its own background."""

    def donut(self, rule):
        return {"id": "d", "type": "path", "fill_rule": rule,
                "d": [["M", 10, 50], ["A", 40, 40, 0, 1, 1, 90, 50],
                      ["A", 40, 40, 0, 1, 1, 10, 50], ["Z"],
                      ["M", 30, 50], ["A", 20, 20, 0, 1, 1, 70, 50],
                      ["A", 20, 20, 0, 1, 1, 30, 50], ["Z"]],
                "fills": [{"type": "solid", "color": "#000000"}]}

    def test_evenodd_cuts_a_hole_that_same_winding_would_not(self):
        even = render_svg(generate_svg(make_project([self.donut("evenodd")])), 100, 100)
        non = render_svg(generate_svg(make_project([self.donut("nonzero")])), 100, 100)
        # both subpaths wind the same way, so only evenodd leaves a hole
        assert np.asarray(even)[50, 50, 3] == 0
        assert np.asarray(non)[50, 50, 3] == 255
        # the ring itself is filled either way
        assert np.asarray(even)[50, 15, 3] == 255

    def test_attribute_only_emitted_when_needed(self):
        assert 'fill-rule="evenodd"' in generate_svg(
            make_project([self.donut("evenodd")]))
        assert "fill-rule" not in generate_svg(make_project([self.donut("nonzero")]))

    def test_rejects_an_unknown_rule(self):
        with pytest.raises(ModelError, match="fill_rule"):
            validate_shape(self.donut("winding"))


class TestPrimitives:
    """Ground truth: synthesise a union whose parameters are known, take its
    boundary the way `measure` would, and check the fit recovers the numbers.

    The apparatus is the risk here, not the fitter (conventions.md 12), so the
    sampling test comes first: it asserts the SDF is zero on an analytically
    known boundary, with no rasterisation in the loop.
    """

    @staticmethod
    def sheared_rect(xl, xr, ytop, ybot, s, cy):
        """Parallelogram with horizontal top/bottom, clockwise from top-left.
        xl/xr are the slant lines' x at y = cy."""
        return [(xl + (ytop - cy) * s, ytop), (xr + (ytop - cy) * s, ytop),
                (xr + (ybot - cy) * s, ybot), (xl + (ybot - cy) * s, ybot)]

    def test_sdf_is_zero_on_an_analytic_boundary(self):
        # a rounded square, boundary written down in closed form: flats and
        # arcs both, so no rasterisation anywhere in this assertion
        r = 6.0
        verts = [(20, 20), (80, 20), (80, 80), (20, 80)]
        pts = [(26 + t * 48, 20.0) for t in np.linspace(0, 1, 40)]
        pts += [(74 + r * np.sin(a), 26 - r * np.cos(a))
                for a in np.linspace(0, np.pi / 2, 40)]
        pts += [(20.0, 26 + t * 48) for t in np.linspace(0, 1, 40)]
        d = prim.rounded_convex_sdf(np.array(pts), verts, r)
        assert np.abs(d).max() < 1e-9, np.abs(d).max()

    def test_generic_construction_helpers_preserve_convex_geometry(self):
        rect = prim.rectangle(10, 20, 40, 12)
        oriented = prim.oriented_rectangle((30, 26), 40, 12, 0)
        assert np.allclose(rect, oriented)
        clipped = prim.clip_halfplane(rect, (2, 0), 70)
        assert clipped[:, 0].max() == pytest.approx(35)
        assert prim.paths([(clipped, [2, 0, 2, 0])])[-1] == ["Z"]
        through_vertices = prim.clip_halfplane(rect, (1, 0), 50)
        edges = np.roll(through_vertices, -1, axis=0) - through_vertices
        assert np.all(np.linalg.norm(edges, axis=1) > 0)

    def test_sdf_sign_is_inside_negative(self):
        verts = [(10, 10), (60, 10), (60, 40), (10, 40)]     # clockwise, y down
        ccw = verts[::-1]
        for v in (verts, ccw):                               # winding must not matter
            d = prim.rounded_convex_sdf(np.array([[35.0, 25.0], [35.0, 5.0]]), v, 4.0)
            assert d[0] < 0 and d[1] > 0, (v, d)
            assert abs(d[0] + 15.0) < 1e-9                   # 15px from the top edge

    def test_erode_gives_the_fillet_centres(self):
        core = prim.erode_convex([(0, 0), (40, 0), (40, 30), (0, 30)], 5.0)
        assert np.allclose(np.sort(core, axis=0),
                           np.sort([[5, 5], [35, 5], [35, 25], [5, 25]], axis=0))

    def test_radius_too_large_is_rejected_not_silently_wrong(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        prim.erode_convex(square, 4.0)                       # fits
        with pytest.raises(prim.PrimitiveError, match="too large"):
            prim.erode_convex(square, 6.0)                   # inside out

    def test_collinear_edges_are_rejected(self):
        with pytest.raises(prim.PrimitiveError, match="parallel"):
            prim.erode_convex([(0, 0), (10, 0), (20, 0), (10, 10)], 3.0)

    def test_union_recovers_eight_known_parameters(self):
        # three parallelograms, 180-degree symmetric: the arrangement the
        # swipe-s mark turned out to be. width and offset are consequences.
        truth = np.array([128.4, 60.0, 50.0, 30.0, 16.0, 25.0, 6.0, 1.7])

        def build(p):
            ang, cx, cy, a, b, g, k, r = p
            s = np.cos(np.radians(ang)) / np.sin(np.radians(ang))
            rects = [(cx - a, cx - a + 2 * b, cy - g, cy + k),
                     (cx - b, cx + b, cy - g, cy + g),
                     (cx + a - 2 * b, cx + a, cy - k, cy + g)]
            return [(self.sheared_rect(xl, xr, yt, yb, s, cy), r)
                    for xl, xr, yt, yb in rects]

        # sample the true boundary by marching a fine grid and keeping the
        # near-zero crossings -- independent of the fitter's own machinery
        ys, xs = np.mgrid[0:100:0.25, 0:120:0.25]
        P = np.stack([xs.ravel(), ys.ravel()], 1)
        d = prim.union_sdf(P, build(truth))
        contour = P[np.abs(d) < 0.02]
        assert len(contour) > 500

        start = truth + np.array([0.9, -2.0, 1.5, 3.0, -2.0, 2.5, 1.0, 1.2])
        fit = prim.fit_union(contour, build, start)
        assert fit.mean < 0.01, fit.summary()
        assert np.allclose(fit.params, truth, atol=0.02), fit.params

    def test_trim_survives_a_rounded_spike(self):
        """A sharp union tip rasterises rounded; trim must absorb it rather
        than let it bend every parameter."""
        truth = np.array([40.0, 40.0, 20.0, 5.0])

        def build(p):
            x0, y0, size, r = p
            return [([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size),
                      (x0, y0 + size)], r)]

        ys, xs = np.mgrid[0:100:0.25, 0:100:0.25]
        P = np.stack([xs.ravel(), ys.ravel()], 1)
        contour = P[np.abs(prim.union_sdf(P, build(truth))) < 0.02]
        # 3% of points pulled 1.5px inward, as a rounded tip would read
        spoilt = contour.copy()
        spoilt[::33] += 1.5
        clean = prim.fit_union(spoilt, build, truth + 2.0)
        trimmed = prim.fit_union(spoilt, build, truth + 2.0, trim=0.06, passes=3)
        assert np.abs(trimmed.params - truth).max() < np.abs(clean.params - truth).max()
        assert np.allclose(trimmed.params, truth, atol=0.05), trimmed.params

    def test_raster_and_ink_bounds_agree_with_the_geometry(self):
        shapes = [([(10, 10), (60, 10), (60, 40), (10, 40)], 4.0)]
        m = prim.raster(shapes, 80, 60)
        assert m[25, 35] and not m[5, 35] and not m[25, 70]
        # the corner pixel is outside the r=4 fillet (centre (14,14): pixel
        # centre (10.5,10.5) is 4.95px away), its neighbour is inside (3.54px)
        assert not m[10, 10] and m[11, 11]
        # exact, not sampled: the bbox of a rounded convex polygon is its
        # core's bbox grown by r, so these are equalities not tolerances
        assert np.allclose(prim.ink_bounds(shapes), (10.0, 10.0, 60.0, 40.0))

    def test_ink_bounds_catches_an_extreme_point_mid_arc(self):
        # a slanted parallelogram's leftmost point is on the bottom-left
        # fillet, not at any vertex -- the case that tempts you to sample
        verts = [(40, 0), (90, 0), (60, 40), (10, 40)]
        r = 8.0
        x0, y0, x1, y1 = prim.ink_bounds([(verts, r)])
        core = prim.erode_convex(verts, r)
        assert abs(x0 - (core[:, 0].min() - r)) < 1e-12
        assert np.allclose((y0, y1), (0.0, 40.0))   # flats: tangent to top/bottom
        # and nothing in the union pokes outside the reported box
        ys, xs = np.mgrid[-20:60:0.5, -20:110:0.5]
        P = np.stack([xs.ravel(), ys.ravel()], 1)
        hit = P[prim.union_sdf(P, [(verts, r)]) <= 0]
        assert hit[:, 0].min() >= x0 - 1e-9 and hit[:, 0].max() <= x1 + 1e-9
        assert hit[:, 1].min() >= y0 - 1e-9 and hit[:, 1].max() <= y1 + 1e-9

    def test_mixed_corner_radii_share_one_geometry_across_all_operations(self):
        verts = [(0, 0), (40, 0), (40, 30), (0, 30)]
        radii = [2.0, 4.0, 6.0, 8.0]
        boundary = np.array([
            (2, 0), (36, 0), (40, 4), (40, 24),
            (34, 30), (8, 30), (0, 22), (0, 2),
        ], float)
        assert np.abs(prim.rounded_convex_sdf(boundary, verts, radii)).max() < 1e-9
        sign = prim.rounded_convex_sdf(
            np.array([(20, 15), (0.1, 0.1), (39.9, 0.1)]), verts, radii
        )
        assert sign[0] < 0 and sign[1] > 0 and sign[2] > 0
        reversed_sign = prim.rounded_convex_sdf(
            np.array([(20, 15), (0.1, 0.1), (39.9, 0.1)]),
            verts[::-1],
            radii[::-1],
        )
        assert np.allclose(sign, reversed_sign)
        assert prim.paths([(verts, radii)]) == rounded_polygon(verts, radii)
        assert np.allclose(prim.ink_bounds([(verts, radii)]), (0, 0, 40, 30))
        mask = prim.raster([(verts, radii)], 45, 35)
        assert mask[15, 20] and not mask[0, 0] and not mask[0, 39]

        angle = np.linspace(np.pi, 1.5 * np.pi, 30)
        known_arc = np.stack([2 + 2 * np.cos(angle), 2 + 2 * np.sin(angle)], 1)

        def build(p):
            return [(verts, [p[0], 4.0, 6.0, 8.0])]

        fit = prim.fit_union(known_arc, build, [3.5], bounds=([0], [10]))
        assert fit.params[0] == pytest.approx(2.0, abs=1e-7)

    def test_fit_union_reports_original_worst_point_coordinates(self):
        verts = [(10, 10), (60, 10), (60, 50), (10, 50)]

        def build(p):
            return [(verts, [p[0], 4.0, 8.0, 2.0])]

        contour = np.array([
            (15, 10), (56, 10), (60, 14), (60, 42),
            (52, 50), (12, 50), (10, 48), (10, 15),
        ], float)
        fit = prim.fit_union(contour, build, [6.0])
        index, point, residual = fit.worst_points(1)[0]
        kept_position = np.flatnonzero(fit.kept).tolist().index(index)
        assert point == tuple(contour[index])
        assert residual == fit.residual[kept_position]

    def test_deterministic(self):
        shapes = [([(5, 5), (50, 5), (50, 45), (5, 45)], 7.0)]
        P = np.array([[10.0, 12.0], [30.0, 25.0], [49.0, 44.0]])
        a = prim.union_sdf(P, shapes)
        b = prim.union_sdf(P, shapes)
        assert a.tobytes() == b.tobytes()


class TestSharedRamp:
    def test_recovers_one_ramp_from_two_partly_hidden_copies(self):
        """Two copies of a shape, each exposing a different slice of the same
        ramp. Neither slice alone pins the end stops; together they must."""
        from png2svg.paint import fit_shared_ramp
        c0, c1 = np.array([250.0, 120.0, 220.0]), np.array([250.0, 40.0, 250.0])
        img = np.zeros((200, 60, 3), np.uint8)
        m_a = np.zeros((200, 60), bool)
        m_b = np.zeros((200, 60), bool)
        # copy A spans y 0..100 but only y 10..55 is visible;
        # copy B spans y 100..200 but only y 150..190 is visible
        for y in range(200):
            for span, lo, hi, m in ((( 0, 100), 10,  55, m_a),
                                    ((100, 200), 150, 190, m_b)):
                t0, t1 = span
                if lo <= y < hi:
                    u = (y + 0.5 - t0) / (t1 - t0)
                    img[y, 12:48] = np.round(c0 + (c1 - c0) * u)
                    m[y, 12:48] = True
        out = fit_shared_ramp(img, [(m_a, 0, 100), (m_b, 100, 200)], erode=3)
        assert out["rms"] < 1.0, out
        assert out["stops"][0]["color"] == "#fa78dc", out["stops"]
        assert out["stops"][1]["color"] == "#fa28fa", out["stops"]

    def test_raises_rather_than_returning_junk_when_nothing_survives(self):
        from png2svg.paint import fit_shared_ramp
        img = np.zeros((20, 20, 3), np.uint8)
        with pytest.raises(ValueError, match="erode"):
            fit_shared_ramp(img, [(np.zeros((20, 20), bool), 0, 20)])

    def test_maps_one_global_ramp_across_reversed_and_forward_pieces(self):
        from png2svg.paint import map_ramp

        ramp = [(0.0, "#000000"), (0.25, "#202020"),
                (0.75, "#e0e0e0"), (1.0, "#ffffff")]
        paint = {
            "type": "linear", "x1": 0, "y1": 0, "x2": 100, "y2": 0,
            "stops": [{"offset": 0, "color": "#ff0000"},
                      {"offset": 0.5, "color": "#00ff00"},
                      {"offset": 1, "color": "#0000ff"}],
        }
        forward = map_ramp(paint, 0.1, 0.6, ramp)
        reverse = map_ramp(paint, 0.9, 0.6, ramp)
        assert [s["offset"] for s in forward["stops"]] == [0.0, 0.3, 1.0]
        assert forward["stops"][-1]["color"] == reverse["stops"][-1]["color"]
        assert paint["stops"][0]["color"] == "#ff0000"

        top = map_ramp(paint, 0.24, 0.0, ramp)
        turn = map_ramp(paint, 0.24, 0.45, ramp)
        middle = map_ramp(paint, 0.45, 0.64, ramp)
        assert top["stops"][0]["color"] == turn["stops"][0]["color"]
        assert turn["stops"][-1]["color"] == middle["stops"][0]["color"]

    def test_map_ramp_preserves_a_hard_stop(self):
        from png2svg.paint import ramp_segment

        ramp = [
            (0.0, "#000000"),
            (0.5, "#000000"),
            (0.5, "#ffffff"),
            (1.0, "#ffffff"),
        ]
        stops = ramp_segment(ramp, 0.25, 0.75)
        assert [(s["offset"], s["color"]) for s in stops] == [
            (0.0, "#000000"),
            (0.5, "#000000"),
            (0.5, "#ffffff"),
            (1.0, "#ffffff"),
        ]

        before = ramp_segment(ramp, 0.25, 0.5)
        after = ramp_segment(ramp, 0.5, 0.75)
        assert before[-1]["color"] == "#000000"
        assert after[0]["color"] == "#ffffff"

        reverse_before = ramp_segment(ramp, 0.75, 0.5)
        reverse_after = ramp_segment(ramp, 0.5, 0.25)
        assert reverse_before[-1]["color"] == "#ffffff"
        assert reverse_after[0]["color"] == "#000000"


class TestStructureDetection:
    """`analyse` should say "this is one shape repeated" when it is, and stay
    quiet when it is not. A false positive here sends the agent down the wrong
    route, so the negative cases matter more than the positive one."""

    @staticmethod
    def render(shapes, w=260, h=200):
        proj = make_project(shapes, w, h)
        return render_svg(generate_svg(proj), w, h)

    @staticmethod
    def rect_path(x, y, w, h, r=8.0):
        from png2svg.geom import rounded_polygon
        return rounded_polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], r)

    def analyse(self, img):
        from png2svg.analyse import analyse_image
        feats, _ = analyse_image(img, (255, 255, 255, 255))
        return feats["components"][0]["structure"]

    def test_finds_the_offset_of_three_repeated_bars(self):
        # staggered, so the union actually has structure -- three bars at the
        # same y would just merge into one rectangle and there'd be nothing
        # for any detector to find
        d = []
        for i in range(3):                     # at a size the tracer resolves
            d += self.rect_path(40 + 80 * i, 60 + 60 * i, 140, 160, 16.0)
        st = self.analyse(self.render([
            {"id": "s", "type": "path", "d": d,
             "fills": [{"type": "solid", "color": "#204080"}]}], 520, 400))
        assert st["repeated_spacings"], st
        assert st["hint"] and "OVERLAPPING PRIMITIVES" in st["hint"], st["hint"]

    def test_flags_an_L_shape_as_overlapping_primitives(self):
        # an L IS two overlapping rects, so routing it to 3b is correct
        d = self.rect_path(40, 30, 60, 140) + self.rect_path(40, 110, 170, 60)
        st = self.analyse(self.render([
            {"id": "s", "type": "path", "d": d,
             "fills": [{"type": "solid", "color": "#204080"}]}]))
        assert st["repeated_spacings"], st
        assert "OVERLAPPING PRIMITIVES" in (st["hint"] or ""), st["hint"]

    def test_quiet_on_a_single_rounded_rect(self):
        st = self.analyse(self.render([
            {"id": "s", "type": "path", "d": self.rect_path(40, 40, 170, 110),
             "fills": [{"type": "solid", "color": "#204080"}]}]))
        # a lone rect IS symmetric, and saying so is true and harmless -- but
        # it must not claim the shape is a repeat of something
        assert not st["repeated_spacings"], st["repeated_spacings"]
        assert not st["crossing_corners"], st["crossing_corners"]
        assert "OVERLAPPING PRIMITIVES" not in (st["hint"] or ""), st["hint"]

    def test_quiet_on_a_single_blob(self):
        # a lone ellipse: no straight edges to space out, nothing to repeat.
        # This is the case that must NOT be sent to the primitives route.
        st = self.analyse(self.render([
            {"id": "s", "type": "path",
             "d": [["M", 40, 100], ["A", 90, 60, 0, 1, 1, 220, 100],
                   ["A", 90, 60, 0, 1, 1, 40, 100], ["Z"]],
             "fills": [{"type": "solid", "color": "#204080"}]}]))
        assert not st["repeated_spacings"], st["repeated_spacings"]
        assert not st["crossing_corners"], st["crossing_corners"]
        assert "OVERLAPPING PRIMITIVES" not in (st["hint"] or ""), st["hint"]

    def test_reports_symmetry_and_edge_directions(self):
        st = self.analyse(self.render([
            {"id": "s", "type": "path", "d": self.rect_path(40, 40, 170, 110),
             "fills": [{"type": "solid", "color": "#204080"}]}]))
        assert "rot180" in st["symmetry"] and st["symmetry"]["rot180"]["iou"] > 0.99
        angles = sorted(round(d["angle_deg"]) for d in st["edge_directions"])
        assert angles == [0, 90], st["edge_directions"]

    def test_flags_corners_far_tighter_than_the_usual_radius(self):
        from png2svg.analyse import crossing_corners
        seg = lambda r, x: {"kind": "corner", "p0": [x, 10.0], "p1": [x, 12.0],
                            "arc_radius": r}
        # five designed 17px fillets and two 2.5px crossings
        got = crossing_corners([seg(17.0, 1), seg(17.4, 2), seg(16.9, 3),
                                seg(17.1, 4), seg(17.2, 5),
                                seg(2.5, 6), seg(2.4, 7)])
        assert [c["radius"] for c in got] == [2.5, 2.4], got
        assert all(c["typical_radius"] == 17.0 for c in got), got
        # and it stays silent when every corner agrees
        assert crossing_corners([seg(17.0, 1), seg(17.2, 2), seg(16.8, 3)]) == []

    def test_deterministic(self):
        img = self.render([{"id": "s", "type": "path",
                            "d": self.rect_path(40, 40, 170, 110),
                            "fills": [{"type": "solid", "color": "#204080"}]}])
        import json as _json
        assert _json.dumps(self.analyse(img)) == _json.dumps(self.analyse(img))
