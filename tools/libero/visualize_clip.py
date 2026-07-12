"""Interactive 3D visualization of a LIBERO clip exported by
``tools.libero.export_clip`` using the viser web viewer.

Loads the .npz clip and exposes:
  - Scene point cloud (colored) for the current frame, plus trajectory
    polylines showing where each point moves over the T=11 clip.
  - Robot (gripper) point cloud with its own motion lines.
  - Gripper end-effector pose as a coordinate frame over time.
  - Sliders for ``t`` (frame index) and ``max_points`` (downsample for
    FPS), plus a play/pause button that animates the clip.

Usage:
    conda activate pointworld-env
    python tools/libero/visualize_clip.py --clip /tmp/libero_stove_clip.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import viser


# --- Camera / gripper pose helpers ----------------------------------------- #

def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert (qx, qy, qz, qw) to viser's (w, x, y, z) convention."""
    qx, qy, qz, qw = q
    return np.array([qw, qx, qy, qz], dtype=np.float64)


# --- Main viser app -------------------------------------------------------- #

def build_app(clip: dict[str, np.ndarray], server: viser.ViserServer) -> None:
    # --- Load arrays --------------------------------------------------- #
    def _cam(cam_idx: int, name: str) -> np.ndarray:
        """Fetch camera_{cam_idx}_{name} or fall back to camera_0."""
        key = f"camera_{cam_idx}_{name}"
        if key in clip:
            return clip[key]
        if f"camera_0_{name}" in clip:
            return clip[f"camera_0_{name}"]
        raise KeyError(name)

    primary_cam = 0
    scene_flows = _cam(primary_cam, "scene_flows").astype(np.float32)  # (T, N, 3)
    scene_colors = _cam(primary_cam, "scene_colors")  # (T, N, 3) uint8
    depth_mask = _cam(primary_cam, "scene_depth_valid_mask").astype(bool)  # (T, N)
    vis_mask = _cam(primary_cam, "scene_visibility").astype(bool) if (
        f"camera_{primary_cam}_scene_visibility" in clip
    ) else np.ones_like(depth_mask)
    valid_mask = depth_mask & vis_mask  # only render points that are both
                                        # depth-valid and not occluded

    # Combine the two cameras' scene point clouds at each frame so the
    # viewer gets a richer point cloud.
    flows_c0 = _cam(0, "scene_flows").astype(np.float32)
    flows_c1 = _cam(1, "scene_flows").astype(np.float32)
    cols_c0 = _cam(0, "scene_colors")
    cols_c1 = _cam(1, "scene_colors")
    mask_c0 = _cam(0, "scene_depth_valid_mask").astype(bool)
    vis_c0 = _cam(0, "scene_visibility").astype(bool) if (
        f"camera_0_scene_visibility" in clip
    ) else np.ones_like(mask_c0)
    mask_c1 = _cam(1, "scene_depth_valid_mask").astype(bool)
    vis_c1 = _cam(1, "scene_visibility").astype(bool) if (
        f"camera_1_scene_visibility" in clip
    ) else np.ones_like(mask_c1)
    valid_c0 = mask_c0 & vis_c0
    valid_c1 = mask_c1 & vis_c1

    flows = np.concatenate([flows_c0, flows_c1], axis=1)  # (T, N_cam0+N_cam1, 3)
    colors = np.concatenate([cols_c0, cols_c1], axis=1)  # (T, N, 3) uint8
    valid = np.concatenate([valid_c0, valid_c1], axis=1)  # (T, N)
    T, N, _ = flows.shape

    robot_flows = clip.get("robot_flows")
    robot_colors = clip.get("robot_colors")
    has_robot = robot_flows is not None and robot_flows.shape[1] > 0

    # Gripper pose: 7-vector (x, y, z, qx, qy, qz, qw)
    gripper_pose_7d = clip["right_gripper_pose"].astype(np.float32)
    gripper_open = clip["right_gripper_open"].astype(np.float32).squeeze(-1)

    print(
        f"clip loaded: T={T}, scene points/frame={N}, robot points={0 if not has_robot else robot_flows.shape[1]}"
    )

    # --- UI ------------------------------------------------------------ #
    with server.gui.add_folder("Clip controls"):
        t_slider = server.gui.add_slider(
            "frame t", min=0, max=T - 1, step=1, initial_value=0
        )
        max_points = server.gui.add_slider(
            "max scene points (downsample)",
            min=500, max=min(50000, N), step=500,
            initial_value=min(8000, N),
        )
        show_trails = server.gui.add_checkbox(
            "show motion trails", initial_value=True
        )
        show_robot = server.gui.add_checkbox(
            "show gripper point cloud", initial_value=True
        )
        show_gripper_axis = server.gui.add_checkbox(
            "show gripper pose axis", initial_value=True
        )
        play_button = server.gui.add_button("Play / Pause")
        info_text = server.gui.add_text("clip info", initial_value="", disabled=True)

    # --- Point cloud handles ------------------------------------------- #
    scene_handle = server.scene.add_point_cloud(
        name="/scene",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.005,
    )
    robot_handle = server.scene.add_point_cloud(
        name="/robot",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.012,
    )

    # --- Trail handles ------------------------------------------------- #
    scene_trail_handle = None
    robot_trail_handle = None

    if show_trails.value:
        # Scene trails: pick points whose max-frame displacement > 5mm.
        disp = np.linalg.norm(flows[1:] - flows[:1], axis=-1)  # (T-1, N)
        per_pt_max = disp.max(axis=0)
        moving = valid[0] & (per_pt_max > 0.005)
        if moving.any():
            max_trails = 200
            idxs = np.where(moving)[0]
            if idxs.shape[0] > max_trails:
                sel = np.random.default_rng(0).choice(
                    idxs.shape[0], size=max_trails, replace=False
                )
                idxs = idxs[sel]
            trails = flows[:, idxs, :]  # (T, n, 3)
            n_pts = trails.shape[1]
            segs = np.empty((n_pts * (T - 1), 2, 3), dtype=np.float32)
            for i in range(T - 1):
                segs[i * n_pts:(i + 1) * n_pts, 0, :] = trails[i]
                segs[i * n_pts:(i + 1) * n_pts, 1, :] = trails[i + 1]
            scene_trail_handle = server.scene.add_line_segments(
                name="/scene_trails",
                points=segs,
                colors=(80, 180, 255),
                line_width=1.5,
            )

    if has_robot and show_trails.value:
        disp_r = np.linalg.norm(robot_flows[1:] - robot_flows[:1], axis=-1)
        moving_r = (disp_r.max(axis=0) > 0.005)
        if moving_r.any():
            trails_r = robot_flows[:, moving_r, :]
            n_pts = trails_r.shape[1]
            segs = np.empty((n_pts * (T - 1), 2, 3), dtype=np.float32)
            for i in range(T - 1):
                segs[i * n_pts:(i + 1) * n_pts, 0, :] = trails_r[i]
                segs[i * n_pts:(i + 1) * n_pts, 1, :] = trails_r[i + 1]
            robot_trail_handle = server.scene.add_line_segments(
                name="/robot_trails",
                points=segs,
                colors=(255, 120, 80),
                line_width=2.0,
            )

    # --- Gripper axis frame -------------------------------------------- #
    gripper_frame_handle = None

    # --- State --------------------------------------------------------- #
    state = {"playing": False, "last_t": time.time()}

    def _update_frame(t: int) -> None:
        # --- Scene ----------------------------------------------------- #
        valid_t = valid[t]
        if not valid_t.any():
            scene_handle.points = np.zeros((0, 3), dtype=np.float32)
            scene_handle.colors = np.zeros((0, 3), dtype=np.uint8)
        else:
            idxs = np.where(valid_t)[0]
            n_cap = int(max_points.value)
            if idxs.shape[0] > n_cap:
                sel = np.random.default_rng(t).choice(
                    idxs.shape[0], size=n_cap, replace=False
                )
                idxs = idxs[sel]
            scene_handle.points = flows[t, idxs, :]
            scene_handle.colors = colors[t, idxs, :]

        # --- Robot ----------------------------------------------------- #
        if has_robot and show_robot.value:
            pts_r = robot_flows[t]
            if pts_r.shape[0] == 0:
                robot_handle.points = np.zeros((0, 3), dtype=np.float32)
                robot_handle.colors = np.zeros((0, 3), dtype=np.uint8)
            else:
                robot_handle.points = pts_r
                if robot_colors is not None and robot_colors.shape[1] > 0:
                    robot_handle.colors = robot_colors[t]
                else:
                    # Fallback color: gripper-orange so the user can see
                    # the cloud even if per-vertex colors are missing.
                    robot_handle.colors = np.full(
                        (pts_r.shape[0], 3), [255, 120, 80], dtype=np.uint8
                    )
        else:
            robot_handle.points = np.zeros((0, 3), dtype=np.float32)
            robot_handle.colors = np.zeros((0, 3), dtype=np.uint8)

        # --- Gripper axis ---------------------------------------------- #
        nonlocal gripper_frame_handle
        gp = gripper_pose_7d[t]
        pos = gp[:3]
        quat = _quat_xyzw_to_wxyz(gp[3:7])
        if show_gripper_axis.value:
            if gripper_frame_handle is None:
                gripper_frame_handle = server.scene.add_frame(
                    name="/gripper",
                    position=pos,
                    wxyz=quat,
                    axes_length=0.08,
                    axes_radius=0.005,
                )
            else:
                gripper_frame_handle.position = pos
                gripper_frame_handle.wxyz = quat
                gripper_frame_handle.visible = True
        elif gripper_frame_handle is not None:
            gripper_frame_handle.visible = False

        # --- Trail visibility ------------------------------------------ #
        if scene_trail_handle is not None:
            scene_trail_handle.visible = bool(show_trails.value)
        if robot_trail_handle is not None:
            robot_trail_handle.visible = bool(show_trails.value)

        # --- Info text ------------------------------------------------- #
        info_text.value = (
            f"frame t={t}/{T - 1}  gripper=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
            f"open={'open' if gripper_open[t] > 0.5 else 'closed'}"
        )

    @t_slider.on_update
    def _on_slider(_evt) -> None:
        _update_frame(int(t_slider.value))

    @max_points.on_update
    def _on_max(_evt) -> None:
        _update_frame(int(t_slider.value))

    @show_trails.on_update
    def _on_trails(_evt) -> None:
        _update_frame(int(t_slider.value))

    @show_robot.on_update
    def _on_robot(_evt) -> None:
        _update_frame(int(t_slider.value))

    @show_gripper_axis.on_update
    def _on_axis(_evt) -> None:
        _update_frame(int(t_slider.value))

    def _toggle_play(_evt) -> None:
        state["playing"] = not state["playing"]
        state["last_t"] = time.time()

    play_button.on_click(_toggle_play)

    # First frame.
    _update_frame(0)

    # Animation loop: while playing, advance the slider every ~0.15s.
    while True:
        if state["playing"]:
            now = time.time()
            if now - state["last_t"] >= 0.15:
                state["last_t"] = now
                next_t = (int(t_slider.value) + 1) % T
                t_slider.value = next_t
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="Path to .npz clip")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    data = np.load(clip_path, allow_pickle=True)
    print(f"loaded {clip_path}, keys={len(data.files)}")

    server = viser.ViserServer(port=args.port, label="libero-clip")
    print(f"viser running on http://localhost:{args.port}")
    build_app(dict(data), server)


if __name__ == "__main__":
    main()
