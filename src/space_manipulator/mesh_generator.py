"""Parametric 3D CAD mesh generator (binary STL in millimeters) for MuJoCo.

Generates 13 binary STL files inside ``models/meshes/`` in **millimeters (mm)**
so they can be loaded into ``models/space_manipulator.xml`` with
``scale="0.001 0.001 0.001"``, matching FreeCAD's standard millimeter STL export.

All part geometries are expressed in their respective MuJoCo body-local frames
so joint axes, sensor frames, and end-effector/target sites align exactly with
the physics definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_MESH_DIR: Path = (
    Path(__file__).resolve().parents[2] / "models" / "meshes"
)

MESH_FILENAMES: tuple[str, ...] = (
    "sat_bus_frame.stl",
    "sat_mli_foil.stl",
    "sat_rcs_thrusters.stl",
    "solar_array_frame.stl",
    "solar_array_cells.stl",
    "arm_link1_housing.stl",
    "arm_link1_accents.stl",
    "arm_link2_housing.stl",
    "arm_link2_accents.stl",
    "arm_link3_wrist.stl",
    "arm_gripper.stl",
    "target_sat_body.stl",
    "target_grapple_fixture.stl",
)


def _rot_x(angle_rad: float) -> NDArray[np.float64]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def _rot_y(angle_rad: float) -> NDArray[np.float64]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _rot_z(angle_rad: float) -> NDArray[np.float64]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def _translate(x: float, y: float, z: float) -> NDArray[np.float64]:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def _frame_from_z_axis(
    origin: Sequence[float], z_dir: Sequence[float]
) -> NDArray[np.float64]:
    """Build a 4x4 transform whose local +Z axis aligns with ``z_dir``."""
    z = np.asarray(z_dir, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    if norm < 1e-12:
        raise ValueError("Direction vector cannot be zero.")
    z /= norm

    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= float(np.linalg.norm(x))
    y = np.cross(z, x)

    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = np.asarray(origin, dtype=np.float64)
    return m


def _octagonal_polygon(half_w: float, half_h: float, chamfer: float) -> list[tuple[float, float]]:
    """Return CCW 2D vertices of a chamfered rectangle (octagon) centered at (0, 0)."""
    c = min(chamfer, half_w * 0.95, half_h * 0.95)
    return [
        (half_w - c, -half_h),
        (half_w, -half_h + c),
        (half_w, half_h - c),
        (half_w - c, half_h),
        (-half_w + c, half_h),
        (-half_w, half_h - c),
        (-half_w, -half_h + c),
        (-half_w + c, -half_h),
    ]


def _regular_polygon(radius: float, sides: int, phase_rad: float = 0.0) -> list[tuple[float, float]]:
    """Return CCW 2D vertices of a regular polygon centered at (0, 0)."""
    return [
        (
            radius * math.cos(phase_rad + 2.0 * math.pi * i / sides),
            radius * math.sin(phase_rad + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


@dataclass
class MeshStats:
    """Summary metrics for a generated STL mesh."""

    path: Path
    triangle_count: int
    bounds_min_mm: tuple[float, float, float]
    bounds_max_mm: tuple[float, float, float]


class MeshBuilder:
    """Accumulator of 3D triangles with outward CCW normals and binary STL export."""

    def __init__(self) -> None:
        self._tris: list[NDArray[np.float64]] = []

    def add_triangle(
        self,
        v0: Sequence[float],
        v1: Sequence[float],
        v2: Sequence[float],
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        tri = np.array([v0, v1, v2], dtype=np.float64)
        if transform is not None:
            tri = tri @ transform[:3, :3].T + transform[:3, 3]
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(np.linalg.norm(cross)) > 1e-9:
            self._tris.append(tri)

    def add_quad(
        self,
        v0: Sequence[float],
        v1: Sequence[float],
        v2: Sequence[float],
        v3: Sequence[float],
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a CCW quad (v0 -> v1 -> v2 -> v3) split into two triangles."""
        self.add_triangle(v0, v1, v2, transform)
        self.add_triangle(v0, v2, v3, transform)

    def add_box(
        self,
        center: Sequence[float],
        size: Sequence[float],
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a rectangular box of full dimensions ``size`` centered at ``center``."""
        cx, cy, cz = center
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        t = _translate(cx, cy, cz)
        if transform is not None:
            t = transform @ t
        pts = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        faces = [
            (0, 3, 2, 1),  # -Z
            (4, 5, 6, 7),  # +Z
            (0, 1, 5, 4),  # -Y
            (2, 3, 7, 6),  # +Y
            (1, 2, 6, 5),  # +X
            (3, 0, 4, 7),  # -X
        ]
        for a, b, c, d in faces:
            self.add_quad(pts[a], pts[b], pts[c], pts[d], t)

    def add_prism(
        self,
        poly_bottom: Sequence[tuple[float, float]],
        z0: float,
        z1: float,
        poly_top: Sequence[tuple[float, float]] | None = None,
        transform: NDArray[np.float64] | None = None,
        cap_bottom: bool = True,
        cap_top: bool = True,
    ) -> None:
        """Extrude or loft between two CCW 2D polygons along local Z."""
        n = len(poly_bottom)
        if poly_top is None:
            poly_top = poly_bottom
        if len(poly_top) != n or n < 3:
            raise ValueError("Polygons must have equal length >= 3.")

        for i in range(n):
            j = (i + 1) % n
            b0 = (poly_bottom[i][0], poly_bottom[i][1], z0)
            b1 = (poly_bottom[j][0], poly_bottom[j][1], z0)
            t1 = (poly_top[j][0], poly_top[j][1], z1)
            t0 = (poly_top[i][0], poly_top[i][1], z1)
            self.add_quad(b0, b1, t1, t0, transform)

        if cap_bottom:
            cb = (
                sum(p[0] for p in poly_bottom) / n,
                sum(p[1] for p in poly_bottom) / n,
                z0,
            )
            for i in range(n):
                j = (i + 1) % n
                b0 = (poly_bottom[i][0], poly_bottom[i][1], z0)
                b1 = (poly_bottom[j][0], poly_bottom[j][1], z0)
                self.add_triangle(cb, b1, b0, transform)

        if cap_top:
            ct = (
                sum(p[0] for p in poly_top) / n,
                sum(p[1] for p in poly_top) / n,
                z1,
            )
            for i in range(n):
                j = (i + 1) % n
                t0 = (poly_top[i][0], poly_top[i][1], z1)
                t1 = (poly_top[j][0], poly_top[j][1], z1)
                self.add_triangle(ct, t0, t1, transform)

    def add_chamfered_box(
        self,
        center: Sequence[float],
        size: Sequence[float],
        chamfer: float,
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add an octagonal-cross-section box extruded along Z with beveled caps."""
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        c = min(chamfer, hx * 0.8, hy * 0.8, hz * 0.4)
        outer = _octagonal_polygon(hx, hy, c)
        inner = _octagonal_polygon(hx - c * 0.55, hy - c * 0.55, c * 0.65)

        t = _translate(*center)
        if transform is not None:
            t = transform @ t

        self.add_prism(inner, -hz, -hz + c * 0.55, outer, t, cap_bottom=True, cap_top=False)
        self.add_prism(outer, -hz + c * 0.55, hz - c * 0.55, outer, t, cap_bottom=False, cap_top=False)
        self.add_prism(outer, hz - c * 0.55, hz, inner, t, cap_bottom=False, cap_top=True)

    def add_frustum(
        self,
        r0: float,
        r1: float,
        z0: float,
        z1: float,
        segments: int = 24,
        transform: NDArray[np.float64] | None = None,
        cap_bottom: bool = True,
        cap_top: bool = True,
    ) -> None:
        """Add a circular cylinder, cone, or frustum along local Z."""
        if r0 <= 0.0 and r1 <= 0.0:
            return
        angles = [2.0 * math.pi * i / segments for i in range(segments)]
        ring0 = [(r0 * math.cos(a), r0 * math.sin(a), z0) for a in angles]
        ring1 = [(r1 * math.cos(a), r1 * math.sin(a), z1) for a in angles]

        for i in range(segments):
            j = (i + 1) % segments
            if r0 > 1e-6 and r1 > 1e-6:
                self.add_quad(ring0[i], ring0[j], ring1[j], ring1[i], transform)
            elif r0 > 1e-6:
                self.add_triangle(ring0[i], ring0[j], (0.0, 0.0, z1), transform)
            else:
                self.add_triangle((0.0, 0.0, z0), ring1[j], ring1[i], transform)

        if cap_bottom and r0 > 1e-6:
            c0 = (0.0, 0.0, z0)
            for i in range(segments):
                j = (i + 1) % segments
                self.add_triangle(c0, ring0[j], ring0[i], transform)

        if cap_top and r1 > 1e-6:
            c1 = (0.0, 0.0, z1)
            for i in range(segments):
                j = (i + 1) % segments
                self.add_triangle(c1, ring1[i], ring1[j], transform)

    def add_hollow_ring(
        self,
        r_inner: float,
        r_outer: float,
        z0: float,
        z1: float,
        segments: int = 28,
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a closed annular ring/flange along local Z."""
        angles = [2.0 * math.pi * i / segments for i in range(segments)]
        in0 = [(r_inner * math.cos(a), r_inner * math.sin(a), z0) for a in angles]
        in1 = [(r_inner * math.cos(a), r_inner * math.sin(a), z1) for a in angles]
        out0 = [(r_outer * math.cos(a), r_outer * math.sin(a), z0) for a in angles]
        out1 = [(r_outer * math.cos(a), r_outer * math.sin(a), z1) for a in angles]

        for i in range(segments):
            j = (i + 1) % segments
            # Outer wall (+radial)
            self.add_quad(out0[i], out0[j], out1[j], out1[i], transform)
            # Inner wall (-radial)
            self.add_quad(in0[j], in0[i], in1[i], in1[j], transform)
            # Bottom face (-Z)
            self.add_quad(in0[i], in0[j], out0[j], out0[i], transform)
            # Top face (+Z)
            self.add_quad(out1[i], out1[j], in1[j], in1[i], transform)

    def add_strut(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        radius: float,
        segments: int = 10,
    ) -> None:
        """Add a cylindrical/polygonal strut connecting 3D points ``p0`` and ``p1``."""
        p0_arr = np.asarray(p0, dtype=np.float64)
        p1_arr = np.asarray(p1, dtype=np.float64)
        diff = p1_arr - p0_arr
        length = float(np.linalg.norm(diff))
        if length < 1e-6:
            return
        t = _frame_from_z_axis(p0_arr, diff)
        self.add_frustum(radius, radius, 0.0, length, segments=segments, transform=t)

    def add_conduit(
        self,
        points: Sequence[Sequence[float]],
        radius: float,
        segments: int = 10,
    ) -> None:
        """Sweep a tubular cable/conduit along a polyline."""
        for k in range(len(points) - 1):
            self.add_strut(points[k], points[k + 1], radius, segments=segments)
        for p in points[1:-1]:
            self.add_frustum(
                radius * 1.12,
                radius * 1.12,
                -radius * 0.7,
                radius * 0.7,
                segments=segments,
                transform=_translate(*p),
            )

    def add_parabolic_dish(
        self,
        radius: float,
        depth: float,
        thickness: float,
        segments: int = 28,
        rings: int = 5,
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a thickened parabolic dish opening toward +Z."""
        angles = [2.0 * math.pi * i / segments for i in range(segments)]
        r_vals = [radius * k / rings for k in range(rings + 1)]

        def _z_front(r: float) -> float:
            return depth * (r / radius) ** 2

        for k in range(rings):
            r_in, r_out = r_vals[k], r_vals[k + 1]
            zf_in, zf_out = _z_front(r_in), _z_front(r_out)
            zb_in, zb_out = zf_in - thickness, zf_out - thickness

            for i in range(segments):
                j = (i + 1) % segments
                ca, sa = math.cos(angles[i]), math.sin(angles[i])
                cb, sb = math.cos(angles[j]), math.sin(angles[j])

                if k == 0:
                    self.add_triangle(
                        (0.0, 0.0, 0.0),
                        (r_out * ca, r_out * sa, zf_out),
                        (r_out * cb, r_out * sb, zf_out),
                        transform,
                    )
                    self.add_triangle(
                        (0.0, 0.0, -thickness),
                        (r_out * cb, r_out * sb, zb_out),
                        (r_out * ca, r_out * sa, zb_out),
                        transform,
                    )
                else:
                    self.add_quad(
                        (r_in * ca, r_in * sa, zf_in),
                        (r_out * ca, r_out * sa, zf_out),
                        (r_out * cb, r_out * sb, zf_out),
                        (r_in * cb, r_in * sb, zf_in),
                        transform,
                    )
                    self.add_quad(
                        (r_in * cb, r_in * sb, zb_in),
                        (r_out * cb, r_out * sb, zb_out),
                        (r_out * ca, r_out * sa, zb_out),
                        (r_in * ca, r_in * sa, zb_in),
                        transform,
                    )

        # Outer rim closing front to back
        zf_edge = depth
        zb_edge = depth - thickness
        for i in range(segments):
            j = (i + 1) % segments
            ca, sa = math.cos(angles[i]), math.sin(angles[i])
            cb, sb = math.cos(angles[j]), math.sin(angles[j])
            self.add_quad(
                (radius * ca, radius * sa, zb_edge),
                (radius * ca, radius * sa, zf_edge),
                (radius * cb, radius * sb, zf_edge),
                (radius * cb, radius * sb, zb_edge),
                transform,
            )

    def to_array(self) -> NDArray[np.float64]:
        if not self._tris:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return np.stack(self._tris, axis=0)

    def write_binary_stl(self, path: Path | str) -> MeshStats:
        """Write triangles to a binary STL file (millimeters)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tris = self.to_array()
        n_tris = int(tris.shape[0])
        if n_tris == 0:
            raise ValueError(f"Cannot write empty mesh to {out_path}.")

        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.maximum(norms, 1e-12))

        header = b"Binary STL generated by mujoco-space-manipulator (units: mm)"
        header = header.ljust(80, b"\x00")[:80]

        with out_path.open("wb") as fh:
            fh.write(header)
            fh.write(struct.pack("<I", n_tris))
            for idx in range(n_tris):
                nx, ny, nz = normals[idx]
                p0x, p0y, p0z = v0[idx]
                p1x, p1y, p1z = v1[idx]
                p2x, p2y, p2z = v2[idx]
                fh.write(
                    struct.pack(
                        "<12fH",
                        float(nx),
                        float(ny),
                        float(nz),
                        float(p0x),
                        float(p0y),
                        float(p0z),
                        float(p1x),
                        float(p1y),
                        float(p1z),
                        float(p2x),
                        float(p2y),
                        float(p2z),
                        0,
                    )
                )

        flat = tris.reshape(-1, 3)
        bmin = tuple(float(v) for v in np.min(flat, axis=0))
        bmax = tuple(float(v) for v in np.max(flat, axis=0))
        return MeshStats(
            path=out_path,
            triangle_count=n_tris,
            bounds_min_mm=(bmin[0], bmin[1], bmin[2]),
            bounds_max_mm=(bmax[0], bmax[1], bmax[2]),
        )


def build_sat_bus_frame() -> MeshBuilder:
    """Octagonal chamfered spacecraft bus (~600 mm), +X adapter ring, HGA dish, star trackers."""
    mb = MeshBuilder()
    # Core octagonal structural chassis (along Z, 600x600x600 mm with 85 mm corner chamfers)
    mb.add_chamfered_box((0.0, 0.0, 0.0), (600.0, 600.0, 600.0), chamfer=85.0)

    # Top and bottom structural rim decks
    oct_rim = _octagonal_polygon(306.0, 306.0, 86.0)
    mb.add_prism(oct_rim, 286.0, 304.0)
    mb.add_prism(oct_rim, -304.0, -286.0)

    # Structural vertical corner ribs (exposed between MLI panels)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            mb.add_box((sx * 255.0, sy * 255.0, 0.0), (24.0, 24.0, 590.0))

    # +X front manipulator mounting adapter ring (x = 300..330 mm, meets link1 at x=330 mm)
    t_x = _translate(300.0, 0.0, 0.0) @ _rot_y(math.pi / 2.0)
    mb.add_hollow_ring(76.0, 145.0, 0.0, 14.0, segments=32, transform=t_x)
    mb.add_frustum(86.0, 64.0, 10.0, 30.0, segments=28, transform=t_x)
    for k in range(8):
        ang = 2.0 * math.pi * k / 8.0
        by, bz = 115.0 * math.cos(ang), 115.0 * math.sin(ang)
        mb.add_frustum(
            9.0,
            9.0,
            0.0,
            18.0,
            segments=10,
            transform=_translate(300.0, by, bz) @ _rot_y(math.pi / 2.0),
        )

    # High-Gain Parabolic Antenna (HGA) on +Z top deck
    mast_base = (-115.0, -70.0, 300.0)
    mb.add_frustum(26.0, 20.0, 300.0, 415.0, segments=16, transform=_translate(-115.0, -70.0, 0.0))
    mb.add_box((-115.0, -70.0, 422.0), (46.0, 56.0, 36.0))
    dish_t = _translate(-100.0, -55.0, 445.0) @ _rot_y(0.45) @ _rot_x(0.32)
    mb.add_parabolic_dish(165.0, 44.0, 7.0, segments=32, rings=6, transform=dish_t)
    # Feed horn & tripod struts
    horn_tip = dish_t @ np.array([0.0, 0.0, 125.0, 1.0])
    mb.add_frustum(11.0, 18.0, 95.0, 132.0, segments=14, transform=dish_t)
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        rim_local = np.array([138.0 * math.cos(ang), 138.0 * math.sin(ang), 30.0, 1.0])
        rim_world = dish_t @ rim_local
        mb.add_strut(rim_world[:3], horn_tip[:3], radius=3.5, segments=8)

    # Dual baffled Star Trackers on +Z top deck
    for sy in (-1.0, 1.0):
        st_t = _translate(135.0, sy * 145.0, 304.0) @ _rot_x(-sy * 0.42) @ _rot_y(0.28)
        mb.add_box((0.0, 0.0, 18.0), (54.0, 54.0, 36.0), transform=st_t)
        mb.add_frustum(22.0, 34.0, 36.0, 96.0, segments=18, transform=st_t)

    # Secondary S-band telemetry whip mast on +Z
    mb.add_frustum(14.0, 9.0, 300.0, 395.0, segments=12, transform=_translate(-185.0, 155.0, 0.0))
    mb.add_frustum(16.0, 16.0, 395.0, 415.0, segments=14, transform=_translate(-185.0, 155.0, 0.0))
    return mb


def build_sat_mli_foil() -> MeshBuilder:
    """Faceted gold MLI thermal blankets fitted onto the octagonal bus bays."""
    mb = MeshBuilder()

    def _add_quilted_panel(t_face: NDArray[np.float64], w: float, h: float, nx: int = 3, ny: int = 3) -> None:
        cw, ch = w / nx, h / ny
        for ix in range(nx):
            for iy in range(ny):
                cx = -w * 0.5 + (ix + 0.5) * cw
                cy = -h * 0.5 + (iy + 0.5) * ch
                # Slight alternating puff relief for realistic MLI crinkle reflections
                relief = 7.0 + 2.5 * ((ix + iy) % 2)
                cell_t = t_face @ _translate(cx, cy, 0.0)
                poly_base = [
                    (-cw * 0.48, -ch * 0.48),
                    (cw * 0.48, -ch * 0.48),
                    (cw * 0.48, ch * 0.48),
                    (-cw * 0.48, ch * 0.48),
                ]
                poly_peak = [
                    (-cw * 0.34, -ch * 0.34),
                    (cw * 0.34, -ch * 0.34),
                    (cw * 0.34, ch * 0.34),
                    (-cw * 0.34, ch * 0.34),
                ]
                mb.add_prism(poly_base, 0.0, relief, poly_peak, transform=cell_t)

    # -X back face
    _add_quilted_panel(_translate(-300.0, 0.0, 0.0) @ _rot_y(-math.pi / 2.0), 420.0, 420.0)
    # +X face around adapter ring (4 perimeter MLI blankets)
    for y0, z0, bw, bh in [
        (0.0, 195.0, 380.0, 95.0),
        (0.0, -195.0, 380.0, 95.0),
        (195.0, 0.0, 95.0, 270.0),
        (-195.0, 0.0, 95.0, 270.0),
    ]:
        _add_quilted_panel(
            _translate(300.0, y0, z0) @ _rot_y(math.pi / 2.0), bh, bw, nx=2, ny=2
        )

    # +Y and -Y faces (leaving central SADA circular hub clear)
    for sy in (-1.0, 1.0):
        rot = _rot_x(-sy * math.pi / 2.0)
        for ox in (-155.0, 155.0):
            _add_quilted_panel(_translate(ox, sy * 300.0, 0.0) @ rot, 130.0, 400.0, nx=2, ny=3)
        for oz in (-165.0, 165.0):
            _add_quilted_panel(_translate(0.0, sy * 300.0, oz) @ rot, 160.0, 105.0, nx=2, ny=1)

    # +Z and -Z deck blankets
    _add_quilted_panel(_translate(45.0, 0.0, -300.0) @ _rot_x(math.pi), 340.0, 360.0)
    _add_quilted_panel(_translate(15.0, 40.0, 300.0), 210.0, 210.0, nx=2, ny=2)

    # 4 diagonal chamfered faces (45 degrees in XY; face midpoint at (257.5, 257.5) -> r = 364.16 mm)
    diag_dist = 257.5 * math.sqrt(2.0) + 0.5
    for k in range(4):
        ang = math.pi / 4.0 + k * math.pi / 2.0
        dx = diag_dist * math.cos(ang)
        dy = diag_dist * math.sin(ang)
        t_diag = _translate(dx, dy, 0.0) @ _rot_z(ang) @ _rot_y(math.pi / 2.0)
        _add_quilted_panel(t_diag, 440.0, 92.0, nx=3, ny=1)

    return mb


def build_sat_rcs_thrusters() -> MeshBuilder:
    """8 corner RCS thruster pods with 3 conical de Laval nozzles each (24 nozzles total)."""
    mb = MeshBuilder()
    nozzle_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                cx, cy, cz = sx * 262.0, sy * 262.0, sz * 274.0
                mb.add_chamfered_box((cx, cy, cz), (56.0, 56.0, 56.0), chamfer=13.0)
                signs = np.array([sx, sy, sz], dtype=np.float64)
                for base_dir in nozzle_dirs:
                    out_dir = base_dir * signs
                    t_noz = _frame_from_z_axis((cx, cy, cz), out_dir)
                    # Throat collar + diverging conical de Laval bell
                    mb.add_frustum(14.0, 10.0, 24.0, 34.0, segments=16, transform=t_noz)
                    mb.add_frustum(10.0, 20.0, 34.0, 64.0, segments=18, transform=t_noz)
                    mb.add_hollow_ring(17.0, 21.0, 61.0, 64.0, segments=18, transform=t_noz)
    return mb


def build_solar_array_frame() -> MeshBuilder:
    """Solar array drive hubs, Y-yoke trusses, and rigid wing frames along +Y and -Y."""
    mb = MeshBuilder()
    for sy in (-1.0, 1.0):
        # SADA rotary drum emerging from +/-Y bus wall
        t_hub = _translate(0.0, sy * 298.0, 0.0) @ _rot_x(-sy * math.pi / 2.0)
        mb.add_frustum(56.0, 48.0, 0.0, 42.0, segments=20, transform=t_hub)
        mb.add_frustum(34.0, 34.0, 42.0, 62.0, segments=18, transform=t_hub)

        # Y-shaped yoke truss connecting hub (y=360) to inner panel spar (y=450)
        p_hub = (0.0, sy * 355.0, 0.0)
        p_left = (-175.0, sy * 450.0, 0.0)
        p_right = (175.0, sy * 450.0, 0.0)
        mb.add_strut(p_hub, p_left, radius=11.0, segments=10)
        mb.add_strut(p_hub, p_right, radius=11.0, segments=10)
        mb.add_strut(p_hub, (0.0, sy * 450.0, 0.0), radius=9.0, segments=10)

        # Rigid rectangular perimeter frame & cross-ribs (|y| = 445..1425, centered ~935 mm)
        y_in, y_out = sy * 445.0, sy * 1425.0
        y_mid = 0.5 * (y_in + y_out)
        wing_len = abs(y_out - y_in)
        # Side rails at x = +/- 225 mm
        for sx in (-1.0, 1.0):
            mb.add_box((sx * 225.0, y_mid, 0.0), (16.0, wing_len, 20.0))
        # Central structural spine
        mb.add_box((0.0, y_mid, 0.0), (14.0, wing_len, 22.0))
        # Crossbars (5 ribs dividing into 4 bays)
        for k in range(5):
            yk = y_in + (y_out - y_in) * (k / 4.0)
            mb.add_box((0.0, yk, 0.0), (466.0, 16.0, 20.0))
    return mb


def build_solar_array_cells() -> MeshBuilder:
    """Segmented photovoltaic solar cell blankets on both +Y and -Y wings."""
    mb = MeshBuilder()
    for sy in (-1.0, 1.0):
        y_in = sy * 455.0
        step_y = sy * 240.0
        for row in range(4):
            yc = y_in + (row + 0.5) * step_y
            for col_x in (-112.0, 112.0):
                # Raised solar cell module on both +Z and -Z faces
                mb.add_box((col_x, yc, 0.0), (202.0, 222.0, 24.0))
                # Subtle stepped cell border on top & bottom
                mb.add_box((col_x, yc, 13.5), (186.0, 206.0, 3.0))
                mb.add_box((col_x, yc, -13.5), (186.0, 206.0, 3.0))
    return mb


def build_arm_link1_housing() -> MeshBuilder:
    """Joint 1 Z-axis motor drum at (0,0,0) + structural bracket & clevis to Joint 2 (90,0,0)."""
    mb = MeshBuilder()
    # Joint 1 vertical actuator drum (Z axis)
    mb.add_frustum(62.0, 62.0, -56.0, 56.0, segments=28)
    mb.add_frustum(50.0, 62.0, 56.0, 68.0, segments=28)
    mb.add_frustum(62.0, 50.0, -68.0, -56.0, segments=28)
    # Mounting flange to base_sat +X adapter (at x=0..62, z=0)
    mb.add_chamfered_box((22.0, 0.0, 0.0), (68.0, 96.0, 88.0), chamfer=16.0)
    # Dual-cheek clevis yoke supporting Joint 2 along Y axis at x=90 mm
    for sy in (-1.0, 1.0):
        mb.add_chamfered_box((66.0, sy * 56.0, 0.0), (88.0, 20.0, 94.0), chamfer=10.0)
        mb.add_frustum(
            48.0,
            48.0,
            -11.0,
            11.0,
            segments=24,
            transform=_translate(90.0, sy * 56.0, 0.0) @ _rot_x(math.pi / 2.0),
        )
    return mb


def build_arm_link1_accents() -> MeshBuilder:
    """Thermal rings, bearing caps, and external umbilical conduit for Link 1."""
    mb = MeshBuilder()
    # Thermal rings on Joint 1 drum
    mb.add_hollow_ring(60.0, 66.0, 40.0, 50.0, segments=28)
    mb.add_hollow_ring(60.0, 66.0, -50.0, -40.0, segments=28)
    # Joint 2 outer encoder caps at (90, +/-67, 0)
    for sy in (-1.0, 1.0):
        mb.add_frustum(
            30.0,
            24.0,
            0.0,
            10.0,
            segments=20,
            transform=_translate(90.0, sy * 66.0, 0.0) @ _rot_x(-sy * math.pi / 2.0),
        )
    # External umbilical conduit
    mb.add_conduit(
        [(0.0, 64.0, -32.0), (45.0, 72.0, -28.0), (82.0, 68.0, -14.0)],
        radius=5.5,
    )
    mb.add_conduit(
        [(0.0, -64.0, 32.0), (45.0, -72.0, 28.0), (82.0, -68.0, 14.0)],
        radius=5.5,
    )
    return mb


def build_arm_link2_housing() -> MeshBuilder:
    """Joint 2 Y-axis drum at (0,0,0), tapered octagonal boom to Joint 3 at (550,0,0)."""
    mb = MeshBuilder()
    # Joint 2 actuator drum along Y at (0,0,0), fits cleanly inside link1 clevis (y in [-44, 44])
    t_j2 = _rot_x(math.pi / 2.0)
    mb.add_frustum(54.0, 54.0, -44.0, 44.0, segments=28, transform=t_j2)
    mb.add_chamfered_box((42.0, 0.0, 0.0), (76.0, 82.0, 86.0), chamfer=16.0)

    # Tapered octagonal boom from x=75 mm to x=485 mm along +X
    t_x = _rot_y(math.pi / 2.0)
    oct_start = _octagonal_polygon(43.0, 43.0, 14.0)
    oct_end = _octagonal_polygon(37.0, 37.0, 12.0)
    mb.add_prism(oct_start, 75.0, 485.0, poly_top=oct_end, transform=t_x)

    # Dual-cheek clevis yoke at x=485..550 mm supporting Joint 3 at (550, 0, 0)
    mb.add_chamfered_box((485.0, 0.0, 0.0), (44.0, 96.0, 78.0), chamfer=12.0)
    for sy in (-1.0, 1.0):
        mb.add_chamfered_box((526.0, sy * 46.0, 0.0), (78.0, 18.0, 78.0), chamfer=9.0)
        mb.add_frustum(
            42.0,
            42.0,
            -9.0,
            9.0,
            segments=24,
            transform=_translate(550.0, sy * 46.0, 0.0) @ _rot_x(math.pi / 2.0),
        )
    return mb


def build_arm_link2_accents() -> MeshBuilder:
    """Anodized flanges, stiffener collars, and external harness conduits on Link 2."""
    mb = MeshBuilder()
    t_x = _rot_y(math.pi / 2.0)
    for x_pos, r_in, r_out in [
        (82.0, 42.0, 50.0),
        (210.0, 40.0, 48.0),
        (345.0, 38.0, 46.0),
        (468.0, 36.5, 45.0),
    ]:
        mb.add_hollow_ring(r_in, r_out, x_pos, x_pos + 16.0, segments=24, transform=t_x)

    # Joint 2 & Joint 3 encoder caps
    for sy in (-1.0, 1.0):
        mb.add_frustum(
            26.0,
            20.0,
            0.0,
            8.0,
            segments=20,
            transform=_translate(550.0, sy * 55.0, 0.0) @ _rot_x(-sy * math.pi / 2.0),
        )

    # Dual external cable harnesses along top shoulders of boom
    for sy in (-1.0, 1.0):
        mb.add_conduit(
            [
                (55.0, sy * 24.0, 45.0),
                (140.0, sy * 20.0, 46.0),
                (345.0, sy * 18.0, 43.0),
                (485.0, sy * 20.0, 42.0),
            ],
            radius=4.8,
        )
    return mb


def build_arm_link3_wrist() -> MeshBuilder:
    """Joint 3 Y-axis drum at (0,0,0), forearm boom to x=515 mm, and wrist camera."""
    mb = MeshBuilder()
    t_j3 = _rot_x(math.pi / 2.0)
    # Fits inside link2's clevis (y in [-36, 36])
    mb.add_frustum(46.0, 46.0, -35.0, 35.0, segments=26, transform=t_j3)
    mb.add_chamfered_box((36.0, 0.0, 0.0), (64.0, 66.0, 74.0), chamfer=14.0)

    # Tapered octagonal forearm boom (x = 65..510 mm)
    t_x = _rot_y(math.pi / 2.0)
    oct_start = _octagonal_polygon(36.0, 36.0, 11.0)
    oct_end = _octagonal_polygon(32.0, 32.0, 10.0)
    mb.add_prism(oct_start, 65.0, 510.0, poly_top=oct_end, transform=t_x)

    # Reinforcing wrist rings
    for x_pos in (165.0, 325.0, 475.0):
        mb.add_hollow_ring(33.0, 41.0, x_pos, x_pos + 14.0, segments=22, transform=t_x)

    # Wrist inspection camera & spotlight pod mounted on +Z near x=435..495 mm
    mb.add_box((460.0, 0.0, 38.0), (34.0, 32.0, 20.0))
    mb.add_chamfered_box((465.0, 0.0, 58.0), (64.0, 44.0, 30.0), chamfer=7.0)
    # Optical lens barrel pointing +X toward ee_site
    mb.add_frustum(
        13.0,
        15.0,
        495.0,
        518.0,
        segments=18,
        transform=_translate(0.0, 0.0, 58.0) @ _rot_y(math.pi / 2.0),
    )
    return mb


def build_arm_gripper() -> MeshBuilder:
    """3-finger chamfered orbital gripper + capture ring centered around ee_site (600,0,0)."""
    mb = MeshBuilder()
    t_x = _rot_y(math.pi / 2.0)

    # Wrist capture collar / base flange (x = 510..548 mm)
    mb.add_frustum(36.0, 52.0, 510.0, 528.0, segments=28, transform=t_x)
    mb.add_hollow_ring(26.0, 56.0, 528.0, 550.0, segments=28, transform=t_x)
    # Alignment capture ring floating cleanly around wrist throat
    mb.add_hollow_ring(46.0, 54.0, 562.0, 572.0, segments=28, transform=t_x)

    # 3 chamfered orbital gripper fingers spaced 120 deg apart around +X, spanning ee_site at x=600 mm
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0 + math.pi / 6.0
        t_finger = _rot_x(ang)
        # Base knuckle at r=48 mm, x=546..580 mm
        mb.add_chamfered_box((564.0, 0.0, 46.0), (40.0, 22.0, 18.0), chamfer=4.0, transform=t_finger)
        # Angled gripping jaw converging from r=46 to r=26 mm across ee_site (x=600 mm -> 628 mm)
        jaw_t = t_finger @ _translate(580.0, 0.0, 46.0) @ _rot_y(0.28)
        mb.add_chamfered_box((24.0, 0.0, 0.0), (54.0, 18.0, 13.0), chamfer=3.5, transform=jaw_t)
        # Inner friction pad
        mb.add_box((26.0, 0.0, -7.5), (34.0, 14.0, 4.0), transform=jaw_t)

    return mb


def build_target_sat_body() -> MeshBuilder:
    """Hexagonal target microsatellite body behind grapple fixture (+X), with solar fins & antenna."""
    mb = MeshBuilder()
    t_x = _rot_y(math.pi / 2.0)
    hex_poly = _regular_polygon(105.0, 6, phase_rad=math.pi / 6.0)
    hex_rear = _regular_polygon(86.0, 6, phase_rad=math.pi / 6.0)

    # Hexagonal main hull from x = +52 mm to x = +235 mm
    mb.add_prism(hex_poly, 52.0, 235.0, transform=t_x)
    # Rear tapered adapter cone x = +235..+262 mm
    mb.add_prism(hex_poly, 235.0, 262.0, poly_top=hex_rear, transform=t_x)
    # Front bulkhead bezel at x = 44..56 mm
    mb.add_frustum(74.0, 98.0, 44.0, 56.0, segments=20, transform=t_x)

    # Two compact deployable solar fins along +/-Y
    for sy in (-1.0, 1.0):
        mb.add_strut((145.0, sy * 92.0, 0.0), (145.0, sy * 125.0, 0.0), radius=8.0, segments=8)
        mb.add_chamfered_box((145.0, sy * 215.0, 0.0), (105.0, 180.0, 10.0), chamfer=3.0)

    # Top telemetry dish / whip antenna (+Z)
    mb.add_frustum(10.0, 6.0, 90.0, 165.0, segments=12, transform=_translate(175.0, 0.0, 0.0))
    mb.add_frustum(22.0, 26.0, 165.0, 175.0, segments=16, transform=_translate(175.0, 0.0, 0.0))
    return mb


def build_target_grapple_fixture() -> MeshBuilder:
    """Conical alignment dish + grapple pin centered at target_site (0,0,0), facing -X."""
    mb = MeshBuilder()
    t_x = _rot_y(math.pi / 2.0)

    # Conical capture alignment plate facing -X (x = 20..48 mm)
    mb.add_frustum(68.0, 32.0, 20.0, 48.0, segments=24, transform=t_x)
    mb.add_hollow_ring(56.0, 72.0, 16.0, 22.0, segments=24, transform=t_x)

    # Central grapple shaft & spherical/flanged capture knob centered at target_site (0,0,0)
    mb.add_frustum(9.0, 9.0, -8.0, 36.0, segments=18, transform=t_x)
    mb.add_frustum(16.0, 16.0, -14.0, -2.0, segments=20, transform=t_x)
    mb.add_frustum(8.0, 16.0, -20.0, -14.0, segments=20, transform=t_x)

    # 3 radial alignment target tabs on the fixture rim
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        mb.add_box((22.0, 0.0, 44.0), (16.0, 10.0, 28.0), transform=_rot_x(ang))
    return mb


MESH_BUILDERS = {
    "sat_bus_frame.stl": build_sat_bus_frame,
    "sat_mli_foil.stl": build_sat_mli_foil,
    "sat_rcs_thrusters.stl": build_sat_rcs_thrusters,
    "solar_array_frame.stl": build_solar_array_frame,
    "solar_array_cells.stl": build_solar_array_cells,
    "arm_link1_housing.stl": build_arm_link1_housing,
    "arm_link1_accents.stl": build_arm_link1_accents,
    "arm_link2_housing.stl": build_arm_link2_housing,
    "arm_link2_accents.stl": build_arm_link2_accents,
    "arm_link3_wrist.stl": build_arm_link3_wrist,
    "arm_gripper.stl": build_arm_gripper,
    "target_sat_body.stl": build_target_sat_body,
    "target_grapple_fixture.stl": build_target_grapple_fixture,
}


def generate_all_meshes(output_dir: Path | str | None = None) -> dict[str, MeshStats]:
    """Generate all 13 binary STL files (in mm) and return their mesh statistics."""
    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_MESH_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, MeshStats] = {}
    for filename in MESH_FILENAMES:
        builder = MESH_BUILDERS[filename]()
        stats = builder.write_binary_stl(target_dir / filename)
        results[filename] = stats
    return results


def main() -> None:
    stats_map = generate_all_meshes()
    total_tris = sum(s.triangle_count for s in stats_map.values())
    print(f"Generated {len(stats_map)} STL meshes in {DEFAULT_MESH_DIR} ({total_tris} triangles):")
    for name, s in stats_map.items():
        bmin = ", ".join(f"{v:.1f}" for v in s.bounds_min_mm)
        bmax = ", ".join(f"{v:.1f}" for v in s.bounds_max_mm)
        print(f"  - {name:28s}: {s.triangle_count:5d} tris | mm bounds [{bmin}] -> [{bmax}]")


if __name__ == "__main__":
    main()
