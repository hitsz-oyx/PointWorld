"""Unit tests for the geometric primitives in
``tools/libero/scene_geometry.py`` and ``tools/libero/export_clip.py``.

Run with::

    conda activate pointworld-env   # numpy only; no mujoco required
    python tools/libero/tests/test_scene_geometry.py

These tests cover the two regressions flagged in ``docs/指导.md`` as
P0-1 (backproject_depth formula) and P0-2 (raw element segmentation).
Both previous implementations happened to be correct for an identity
camera, so we explicitly use a non-identity extrinsic / non-trivial
mocked seg map to make sure the bugs are caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the package importable when running this file from anywhere.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root (so "tools" works)
sys.path.insert(0, str(_HERE.parent.parent))         # tools/ (so "libero" works)
sys.path.insert(0, str(_HERE.parent))                # tools/libero (so flat module names work)

from tools.libero.scene_geometry import (  # noqa: E402
    backproject_depth,
    get_per_pixel_bodies,
)

# ----------------------------------------------------------------------------
# Test 1: backproject_depth with a non-identity camera.
# ----------------------------------------------------------------------------

def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)


def test_backproject_depth_non_identity_camera() -> None:
    """Round-trip reprojection must hold for a non-identity extrinsic.

    A unit test that only used ``T_c_w = I`` (the previous
    implementation) would not have caught the sign-of-translation bug
    described in docs/指导.md §P0-1.

    Convention: ``T_c_w`` (input to backproject_depth) is the
    world-to-camera transform, i.e. ``p_cam = T_c_w @ p_world``.
    """
    # Camera pose in world: rotated by Y then X, then offset.
    # This is ``T_w_c`` (camera-to-world). The function takes
    # ``T_c_w = inv(T_w_c)``.
    R_w_c = _rotation_y(0.4) @ _rotation_x(-0.2)
    t_w_c = np.array([0.3, -0.1, 0.7], dtype=np.float32)
    T_w_c = np.eye(4, dtype=np.float32)
    T_w_c[:3, :3] = R_w_c
    T_w_c[:3, 3] = t_w_c
    T_c_w = np.linalg.inv(T_w_c)

    # Pinhole intrinsics. Image size 8x8, focal ~6, principal point at
    # the center -- large enough to exercise the backprojection math.
    H, W = 8, 8
    K = np.array([
        [6.0, 0.0, 3.5],
        [0.0, 6.0, 3.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Build a depth image where every pixel has the same depth, then
    # verify backprojection -> re-projection round trip yields that depth.
    depth = np.full((H, W), 1.25, dtype=np.float32)
    pts_world = backproject_depth(depth, K, T_c_w)  # (H, W, 3)

    # Re-project the world points back into the camera frame using the
    # same T_c_w we passed in.
    pts_world_flat = pts_world.reshape(-1, 3)
    pts_cam_h = np.concatenate(
        [pts_world_flat, np.ones((pts_world_flat.shape[0], 1), dtype=np.float32)],
        axis=-1,
    )
    pts_cam_flat = (T_c_w @ pts_cam_h.T).T[:, :3]
    reproj_depth = pts_cam_flat[:, 2]  # z-component = camera-frame depth

    assert reproj_depth.shape == (H * W,)
    assert np.allclose(reproj_depth, 1.25, atol=1e-5), (
        f"reprojection depth mismatch: max err = "
        f"{np.max(np.abs(reproj_depth - 1.25))}"
    )

    # Sanity: world points are not all at the origin (the bug being
    # guarded against collapsed the camera origin to the world origin
    # when the sign of t was wrong).
    assert np.linalg.norm(pts_world - pts_world[0, 0]) > 0.5, (
        "backprojected world points are not differentiated -- the "
        "extrinsic translation was likely ignored."
    )


# ----------------------------------------------------------------------------
# Test 2: get_per_pixel_bodies uses raw geom_id (no off-by-one).
# ----------------------------------------------------------------------------

class _MockModel:
    """Minimal stand-in for ``env.sim.model`` to drive get_per_pixel_bodies.

    We just need ``nbody``, ``ngeom``, ``geom_bodyid`` and ``body(i).name``.
    The model below is hand-crafted so that:
      - geom 0 belongs to body 0 (world)
      - geom 1 belongs to body 1 (table)
      - geom 2 belongs to body 2 (akita_black_bowl)
    so the test can verify the raw-geom-id mapping directly.
    """
    nbody = 3
    ngeom = 3
    geom_bodyid = np.array([0, 1, 2], dtype=np.int32)

    def body(self, i: int):
        names = ["world", "table", "akita_black_bowl"]
        return _MockBody(names[i].encode("utf-8"))

class _MockBody:
    def __init__(self, name: bytes) -> None:
        self.name = name


class _MockSim:
    model = _MockModel()


class _MockEnv:
    sim = _MockSim()


def test_get_per_pixel_bodies_raw_geom_id() -> None:
    """The element seg map is the raw geom id, not geom id + 1.

    We feed the function a seg map of:
        [[-1,  1,  1],
         [ 1,  2,  2],
         [ 2,  2, -1]]
    where -1 = no hit (out-of-range sentinel), 1 = geom 1 (body 1 =
    table), 2 = geom 2 (body 2 = akita_black_bowl). robosuite's
    element seg returns -1 for sky / off-the-mesh pixels, so we
    follow that convention.

    The previous (buggy) implementation did ``seg - 1`` and then looked
    up ``geom_bodyid``, which would map 1 -> 0, 2 -> 1, giving every
    bowl pixel the body of the table.
    """
    seg = np.array(
        [[-1, 1, 1],
         [1, 2, 2],
         [2, 2, -1]],
        dtype=np.int32,
    )[..., None]  # (H, W, 1) as produced by robosuite

    body_id, body_name = get_per_pixel_bodies(_MockEnv(), seg)

    # Expected: body_id == (geom_id + 1) for valid geoms, == 0 for no-hit.
    expected_id = np.array(
        [[0, 2, 2],
         [2, 3, 3],
         [3, 3, 0]],
        dtype=np.int32,
    )
    assert np.array_equal(body_id, expected_id), (
        f"body_id mismatch.\nGot:      {body_id}\nExpected: {expected_id}"
    )

    # body_name must use the body of geom=seg, not geom=seg-1.
    assert body_name[0, 1] == "table", (
        f"seg=1 should map to body 'table', got {body_name[0, 1]!r}"
    )
    assert body_name[1, 1] == "akita_black_bowl", (
        f"seg=2 should map to body 'akita_black_bowl', got {body_name[1, 1]!r}"
    )
    # No-hit pixels get an empty string.
    assert body_name[0, 0] == "" and body_name[2, 2] == ""


# ----------------------------------------------------------------------------
# Test 3: get_gripper_body_names excludes wrist (link6) and includes hand (link7).
# ----------------------------------------------------------------------------

def test_get_gripper_body_names_includes_hand_excludes_wrist() -> None:
    """``get_gripper_body_names`` must NOT include ``robot0_link6`` (wrist)
    but MUST include ``robot0_link7`` (hand) and the two fingers.

    The previous implementation hardcoded ``("robot0_link6", "robot0_link7")``
    as a "fallback" so the resulting ``robot_flows`` cloud would have
    enough surface area. That made the LIBERO adapter emit a
    "wrist + hand" cloud that does not match PointWorld's
    training-time "gripper-only" action geometry (see docs/指导.md
    §P1-3). The new implementation only includes a body iff its name
    matches the gripper keywords OR it owns a mesh whose name matches
    the gripper keywords.

    Layout in this test:
      * world
      * robot0_link6 (wrist)     -- 1 mesh geom with mesh "wrist_0"
      * robot0_link7 (hand)      -- 1 mesh geom with mesh "hand_0"
      * robot0_leftfinger        -- 1 cylinder geom (no mesh)
      * robot0_rightfinger       -- 1 cylinder geom (no mesh)

    The test runs in ``pointworld-env`` which has no ``mujoco`` package
    installed, so we inject a tiny stub module exposing only the
    two enum values ``get_gripper_body_names`` reads.
    """
    import sys
    import types

    # Inject a minimal ``mujoco`` stub so we can import
    # ``get_gripper_body_names`` without installing the real package.
    # The enum values match mujoco 2.3.x, which is what LIBERO
    # 0.1.0 ships.
    mujoco_stub = types.ModuleType("mujoco")
    mujoco_stub.mjtGeom = types.SimpleNamespace(mjGEOM_MESH=7)
    mujoco_stub.mjtObj = types.SimpleNamespace(mjOBJ_MESH=5)
    sys.modules.setdefault("mujoco", mujoco_stub)

    from tools.libero import scene_geometry as sg

    body_names = {
        0: "world",
        1: "robot0_link6",
        2: "robot0_link7",
        3: "robot0_leftfinger",
        4: "robot0_rightfinger",
    }
    _geom_bodyid = np.array([1, 2, 3, 4], dtype=np.int32)
    _geom_type = np.array([7, 7, 5, 5], dtype=np.int32)  # MESH, MESH, CYL, CYL
    _geom_dataid = np.array([0, 1, -1, -1], dtype=np.int32)
    _body_geomnum = np.array([0, 1, 1, 1, 1], dtype=np.int32)
    mesh_name_for_id = {0: "wrist_0", 1: "hand_0"}

    class _MockBody:
        def __init__(self, name):
            self.name = name.encode("utf-8") if isinstance(name, str) else name

    class _MockModel:
        nbody = 5
        ngeom = 4
        geom_bodyid = _geom_bodyid
        geom_type = _geom_type
        geom_dataid = _geom_dataid
        body_geomnum = _body_geomnum

        def body(self, i):
            return _MockBody(body_names[i])

        def body_name2id(self, name):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            for bid, bname in body_names.items():
                if bname == name:
                    return bid
            raise ValueError(name)

        def id2name(self, obj_type, obj_id):
            assert obj_type == 5  # mjOBJ_MESH
            return mesh_name_for_id[obj_id]

    class _MockSim:
        model = _MockModel()

    class _MockEnv:
        sim = _MockSim()

    # Monkey-patch ``list_robot_body_names`` to use the mocked model.
    original_list = sg.list_robot_body_names
    sg.list_robot_body_names = lambda env: [
        body_names[i] for i in range(env.sim.model.nbody)
        if body_names[i].startswith("robot0_")
    ]
    try:
        result = sg.get_gripper_body_names(_MockEnv())
    finally:
        sg.list_robot_body_names = original_list

    # Hand and the two fingers must be present.
    assert "robot0_link7" in result, f"hand must be included; got {result}"
    assert "robot0_leftfinger" in result, f"left finger must be included; got {result}"
    assert "robot0_rightfinger" in result, f"right finger must be included; got {result}"
    # The wrist must NOT be present (we have a finger body, so the
    # EEF-proximity fallback is not used; only the name/mesh match
    # is active, which excludes link6).
    assert "robot0_link6" not in result, (
        f"robot0_link6 (wrist) must NOT be a gripper body; got {result}"
    )


def test_get_gripper_body_names_eef_proximity_fallback() -> None:
    """``get_gripper_body_names`` must fall back to the EEF-proximity
    heuristic on the actual LIBERO scene layout, where the gripper
    meshes don't carry gripper keywords (they're named
    ``robot0_link7_vis_*`` etc.).

    On the LIBERO ``libero_spatial`` BDDL scene, ``get_gripper_body_names``
    must return ``['robot0_link6', 'robot0_link7']`` -- the wrist +
    hand cluster, which is the last 2 robot bodies with visible geoms.
    We do NOT want it to pull in the entire arm (link0..link5).
    """
    import sys
    import types

    mujoco_stub = types.ModuleType("mujoco")
    mujoco_stub.mjtGeom = types.SimpleNamespace(mjGEOM_MESH=7)
    mujoco_stub.mjtObj = types.SimpleNamespace(mjOBJ_MESH=5)
    sys.modules.setdefault("mujoco", mujoco_stub)

    from tools.libero import scene_geometry as sg

    # The full LIBERO / Panda layout. No gripper keywords in any body
    # name OR mesh name, so passes 1 + 2 find nothing. The
    # EEF-proximity fallback then takes the last 2 robot bodies with
    # visible geoms (link6 + link7) and includes them.
    body_names = {
        0: "world",
        1: "robot0_base",
        2: "robot0_link0",   # arm
        3: "robot0_link1",   # arm
        4: "robot0_link2",   # arm
        5: "robot0_link3",   # arm
        6: "robot0_link4",   # arm
        7: "robot0_link5",   # arm (forearm) -- STOP here
        8: "robot0_link6",   # gripper (wrist + fingers)
        9: "robot0_link7",   # gripper (hand)
        10: "robot0_right_hand",  # empty ref frame
        11: "gripper0_eef",  # EEF ref frame
    }
    _body_geomnum = np.array([0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0], dtype=np.int32)
    # All robot bodies have 2 mesh geoms named "robot0_link*_vis_*"
    # (no gripper keywords).
    n_geoms = int(_body_geomnum.sum())
    _geom_bodyid = []
    _geom_type = []
    _geom_dataid = []
    for bid in range(len(body_names)):
        n = int(_body_geomnum[bid])
        for gi in range(n):
            _geom_bodyid.append(bid)
            _geom_type.append(7)  # MESH
            _geom_dataid.append(-1)  # no mesh name
    _geom_bodyid = np.array(_geom_bodyid, dtype=np.int32)
    _geom_type = np.array(_geom_type, dtype=np.int32)
    _geom_dataid = np.array(_geom_dataid, dtype=np.int32)
    # Kinematic parent chain:
    #   world -> base -> link0 -> link1 -> ... -> link5 -> link6
    #         -> link7 -> right_hand -> gripper0_eef
    _body_parentid = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)

    class _MockBody:
        def __init__(self, name):
            self.name = name.encode("utf-8") if isinstance(name, str) else name

    class _MockModel:
        nbody = len(body_names)
        ngeom = int(n_geoms)
        body_geomnum = _body_geomnum
        body_geomadr = np.zeros(len(body_names), dtype=np.int32)
        body_parentid = _body_parentid
        geom_bodyid = _geom_bodyid
        geom_type = _geom_type
        geom_dataid = _geom_dataid

        def body(self, i):
            return _MockBody(body_names[i])

        def body_name2id(self, name):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            for bid, bname in body_names.items():
                if bname == name:
                    return bid
            raise ValueError(name)

    class _MockSim:
        model = _MockModel()

    class _MockEnv:
        sim = _MockSim()

    original_list = sg.list_robot_body_names
    sg.list_robot_body_names = lambda env: [
        body_names[i] for i in range(env.sim.model.nbody)
        if body_names[i].startswith("robot0_")
    ]
    try:
        result = sg.get_gripper_body_names(_MockEnv())
    finally:
        sg.list_robot_body_names = original_list

    # On the LIBERO / Panda layout, only link6 + link7 should be
    # picked up by the EEF-proximity fallback (the last 2 robot
    # bodies with visible geoms). The arm (link0..link5) must NOT
    # be included.
    assert "robot0_link6" in result, f"link6 (wrist) must be included; got {result}"
    assert "robot0_link7" in result, f"link7 (hand) must be included; got {result}"
    for arm_link in ("robot0_link0", "robot0_link1", "robot0_link2",
                     "robot0_link3", "robot0_link4", "robot0_link5"):
        assert arm_link not in result, (
            f"{arm_link} (arm) must NOT be a gripper body; got {result}"
        )
    assert "robot0_right_hand" not in result, (
        f"empty ref frame must not be a gripper body; got {result}"
    )


# ----------------------------------------------------------------------------
# Tests 5-7: surface sampling correctness for box / cylinder / sphere / mesh.
# ----------------------------------------------------------------------------

def test_box_surface_sampling_correctness() -> None:
    """Box: every sample lies on a face, normal is the face normal,
    axis distribution follows the face-area weighting."""
    from tools.libero.scene_geometry import _sample_box_surface
    rng = np.random.RandomState(0)
    hs = np.array([0.3, 0.2, 0.1])
    pts, normals = _sample_box_surface(rng, hs, n=2000)
    assert pts.shape == (2000, 3) and normals.shape == (2000, 3)
    # All points are on the box surface: at least one coordinate is at
    # its half-extent, and the others are within the half-extent.
    coord_at_extent_any = np.zeros(2000, dtype=bool)
    for i in range(3):
        coord_at_extent_any |= (np.abs(pts[:, i]) >= hs[i] - 1e-7)
    assert coord_at_extent_any.all(), "some points are not on any face"
    # Each normal is one of ±x, ±y, ±z.
    for n in normals[:100]:
        n = np.round(n, 6)
        assert any(np.allclose(n, e) for e in [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]), f"normal {n} is not a face axis"
    # The axis distribution should follow the face areas.
    face_areas = np.array([
        2.0 * hs[1] * hs[2],
        2.0 * hs[0] * hs[2],
        2.0 * hs[0] * hs[1],
    ])
    expected_axis_probs = face_areas / face_areas.sum()
    axis_pick = np.argmax(np.abs(normals), axis=-1)  # 0=x, 1=y, 2=z
    observed = np.array([(axis_pick == a).mean() for a in range(3)])
    assert np.allclose(observed, expected_axis_probs, atol=0.04), (
        f"axis distribution {observed} does not match expected "
        f"{expected_axis_probs} (face areas {face_areas})"
    )


def test_cylinder_surface_sampling_correctness() -> None:
    """Cylinder: lateral samples have radial normals, cap samples have
    ±z normals, all points are on the surface."""
    from tools.libero.scene_geometry import _sample_cylinder_surface
    rng = np.random.RandomState(0)
    r, h = 0.5, 0.2
    pts, normals = _sample_cylinder_surface(rng, r, h, n=3000)
    assert pts.shape == (3000, 3) and normals.shape == (3000, 3)
    # Classify each sample: lateral if |z| < h, cap otherwise.
    on_lateral = np.abs(pts[:, 2]) < h - 1e-9
    on_caps = ~on_lateral
    # Lateral: radial distance == r, normal == (cos theta, sin theta, 0).
    radial = np.linalg.norm(pts[on_lateral, :2], axis=-1)
    assert np.allclose(radial, r, atol=1e-6), (
        f"lateral samples not on the radius: {radial[:5]} (expected {r})"
    )
    lateral_normals = normals[on_lateral]
    lateral_points = pts[on_lateral]
    expected_normal = lateral_points[:, :2] / r
    assert np.allclose(lateral_normals[:, :2], expected_normal, atol=1e-6)
    assert np.allclose(lateral_normals[:, 2], 0.0, atol=1e-6)
    # Cap: z = ±h, normal = (0, 0, ±1).
    cap_z = pts[on_caps, 2]
    assert np.all(np.isclose(cap_z, h, atol=1e-6) | np.isclose(cap_z, -h, atol=1e-6))
    cap_norms = normals[on_caps]
    assert np.allclose(cap_norms[:, :2], 0.0, atol=1e-6)
    assert np.allclose(np.abs(cap_norms[:, 2]), 1.0, atol=1e-6)
    # Area-weighted distribution: lateral / (lateral + 2 * cap_area)
    # is the expected fraction of lateral samples.
    expected_lateral_frac = (2.0 * np.pi * r * (2 * h)) / (
        2.0 * np.pi * r * (2 * h) + 2.0 * np.pi * r * r
    )
    observed_lateral_frac = on_lateral.mean()
    assert abs(observed_lateral_frac - expected_lateral_frac) < 0.04, (
        f"lateral fraction {observed_lateral_frac:.3f} differs from "
        f"expected {expected_lateral_frac:.3f}"
    )


def test_sphere_surface_sampling_correctness() -> None:
    """Sphere: all points on the radius, normal == radial direction."""
    from tools.libero.scene_geometry import _sample_sphere_surface
    rng = np.random.RandomState(0)
    r = 0.7
    pts, normals = _sample_sphere_surface(rng, r, n=1000)
    radial = np.linalg.norm(pts, axis=-1)
    assert np.allclose(radial, r, atol=1e-6)
    expected_n = pts / r
    assert np.allclose(normals, expected_n, atol=1e-6)


def test_mesh_surface_sampling_correctness() -> None:
    """Mesh: barycentric samples lie inside their picked triangle, normal
    is the face normal."""
    from tools.libero.scene_geometry import _sample_mesh_surface
    # Two-triangle square in the xy-plane:
    #   t0: (0,0,0), (1,0,0), (0,1,0)
    #   t1: (0,0,0), (1,1,0), (0,1,0)
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    rng = np.random.RandomState(0)
    pts, normals = _sample_mesh_surface(rng, verts, faces, n=2000)
    assert pts.shape == (2000, 3)
    # All z values are zero.
    assert np.allclose(pts[:, 2], 0.0, atol=1e-9)
    # All points lie in the unit square [0, 1]^2.
    assert (pts[:, 0] >= -1e-9).all() and (pts[:, 0] <= 1 + 1e-9).all()
    assert (pts[:, 1] >= -1e-9).all() and (pts[:, 1] <= 1 + 1e-9).all()
    # Both faces have the same area (= 0.5) so each is picked ~50%.
    lower_tri = (pts[:, 0] + pts[:, 1] <= 1 + 1e-6)
    upper_tri = ~lower_tri
    assert 0.4 < lower_tri.mean() < 0.6, (
        f"face distribution {lower_tri.mean():.3f} should be ~0.5"
    )
    # Normal is +z everywhere (CCW winding).
    assert np.allclose(normals, np.array([0, 0, 1]), atol=1e-6)


# ----------------------------------------------------------------------------
# Test 8: BDDL parsing from env_args attribute (P1-1).
# ----------------------------------------------------------------------------

def test_parse_bddl_from_env_args() -> None:
    """``_parse_bddl_from_env_args`` must handle JSON strings, dicts,
    bytes, and the missing-key case.

    The official LIBERO dataset writes the BDDL hint inside an
    ``env_args`` JSON blob at the ``/data`` level. Different writer
    versions store the same blob as a JSON string, a Python dict,
    or (rarely) bytes; the resolver must accept all three.
    """
    # ``_parse_bddl_from_env_args`` lives in export_clip, which imports
    # LIBERO at module load. Inject a tiny stub for the bits we touch
    # so the test runs in pointworld-env (no LIBERO installed).
    import sys
    import types
    # The export_clip module does ``from libero.libero.envs import
    # OffScreenRenderEnv`` etc. We only need the symbols it references
    # *at import time* for this test (just the function), so stub the
    # rest away.
    for mod_name in (
        "libero", "libero.libero", "libero.libero.envs", "libero.libero.utils",
    ):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            sys.modules[mod_name] = m
    sys.modules["libero.libero.envs"].OffScreenRenderEnv = object
    utils_stub = types.ModuleType("libero.libero.utils.utils")
    utils_stub.postprocess_model_xml = lambda x, y: x
    sys.modules["libero.libero.utils.utils"] = utils_stub
    sys.modules["libero.libero.utils"].utils = utils_stub
    xml_post_stub = types.ModuleType("libero.libero.envs.utils")
    xml_post_stub.postprocess_model_xml = lambda x, y: x
    sys.modules["libero.libero.envs.utils"] = xml_post_stub
    # Stub get_libero_path (used by _rewrite_libero_asset_paths if it
    # ever runs during the test). We don't call it, so a no-op is fine.
    lib_pkg = types.ModuleType("libero.libero")
    lib_pkg.get_libero_path = lambda key: "/tmp/libero_does_not_exist"
    sys.modules["libero.libero"] = lib_pkg
    sys.modules["libero"].libero = lib_pkg

    # The import below also triggers ``from .sample_schema import ...``,
    # which has no third-party deps (just numpy). That part should work
    # in pointworld-env.
    from tools.libero import export_clip  # type: ignore

    # JSON-string form.
    raw_json = '{"bddl_file_name": "/tmp/foo.bddl", "robots": ["Panda"]}'
    assert export_clip._parse_bddl_from_env_args(raw_json) == "/tmp/foo.bddl"
    # Dict form.
    raw_dict = {"bddl_file_name": "/tmp/bar.bddl"}
    assert export_clip._parse_bddl_from_env_args(raw_dict) == "/tmp/bar.bddl"
    # Bytes form (some HDF5 writers store raw bytes).
    raw_bytes = b'{"bddl_file_name": "/tmp/baz.bddl"}'
    assert export_clip._parse_bddl_from_env_args(raw_bytes) == "/tmp/baz.bddl"
    # bddl_file_name field itself stored as bytes.
    raw_nested = {"bddl_file_name": b"/tmp/qux.bddl"}
    assert export_clip._parse_bddl_from_env_args(raw_nested) == "/tmp/qux.bddl"
    # Missing key.
    assert export_clip._parse_bddl_from_env_args({"robots": ["Panda"]}) is None
    # Malformed JSON.
    assert export_clip._parse_bddl_from_env_args("not-json") is None
    # Empty / None.
    assert export_clip._parse_bddl_from_env_args(None) is None
    assert export_clip._parse_bddl_from_env_args("") is None
    # Wrong type.
    assert export_clip._parse_bddl_from_env_args(42) is None


# ----------------------------------------------------------------------------
# Runner.
# ----------------------------------------------------------------------------

def main() -> int:
    failures = []
    for name, fn in [
        ("backproject_depth_non_identity_camera",
         test_backproject_depth_non_identity_camera),
        ("get_per_pixel_bodies_raw_geom_id",
         test_get_per_pixel_bodies_raw_geom_id),
        ("get_gripper_body_names_includes_hand_excludes_wrist",
         test_get_gripper_body_names_includes_hand_excludes_wrist),
        ("get_gripper_body_names_eef_proximity_fallback",
         test_get_gripper_body_names_eef_proximity_fallback),
        ("box_surface_sampling_correctness",
         test_box_surface_sampling_correctness),
        ("cylinder_surface_sampling_correctness",
         test_cylinder_surface_sampling_correctness),
        ("sphere_surface_sampling_correctness",
         test_sphere_surface_sampling_correctness),
        ("mesh_surface_sampling_correctness",
         test_mesh_surface_sampling_correctness),
        ("parse_bddl_from_env_args",
         test_parse_bddl_from_env_args),
    ]:
        try:
            fn()
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}", file=sys.stderr)
        except Exception as e:
            failures.append((name, repr(e)))
            print(f"[ERROR] {name}: {e!r}", file=sys.stderr)
        else:
            print(f"[OK]   {name}")
    if failures:
        print(f"\n{len(failures)} test(s) failed.", file=sys.stderr)
        return 1
    print("\nall tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
