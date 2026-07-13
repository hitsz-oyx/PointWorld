"""Interactive 3D visualization of a LIBERO clip exported by
``tools.libero.export_clip`` using the viser web viewer.

Two modes:

1. ``--clip clip.npz`` only: show scene / robot point clouds and
   motion trails (the GT trajectory is the per-point flow stored in
   the clip).
2. ``--clip clip.npz --pred_npz pred.npz``: also overlay the model's
   **predicted** flow as a per-point line segment from the t=0 GT
   position to the t=10 predicted position, with the per-point
   endpoint colored by End-Point Error (EPE) relative to the GT
   endpoint. The model's predicted point at each frame is rendered
   side-by-side with the GT point so you can see where the model
   put each scene point vs where it actually went.

Loads the .npz clip and exposes:
  - Scene point cloud (colored) for the current frame, plus trajectory
    polylines showing where each point moves over the T=11 clip.
  - Robot (gripper) point cloud with its own motion lines.
  - Gripper end-effector pose as a coordinate frame over time.
  - Sliders for ``t`` (frame index) and ``max_points`` (downsample for
    FPS), plus a play/pause button that animates the clip.
  - When ``--pred_npz`` is given, an additional slider controls the
    predicted-flow overlay (per-point EPE colormap).

Usage:
    conda activate pointworld-env
    # GT-only view:
    python tools/libero/visualize_clip.py --clip /tmp/libero_stove_clip.npz
    # GT vs predicted view:
    python tools/libero/visualize_clip.py \
        --clip /tmp/libero_stove_clip.npz \
        --pred_npz /tmp/libero_stove_clip_pred.npz
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

def build_app(
    clip: dict[str, np.ndarray],
    server: viser.ViserServer,
    pred: dict[str, np.ndarray] | None = None,
) -> None:
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

    # --- Predicted overlay (optional) ----------------------------------- #
    pred_overlay = None
    if pred is not None and "pred_scene_flows" in pred and "gt_scene_flows" in pred:
        # The saved predicted-flow npz carries (T, N, 3) for both pred
        # and GT plus per_point_epe (T, N). We re-use the *input clip's*
        # per-point validity mask to decide which points to display, so
        # the overlay lines up with the GT cloud.
        pred_overlay = _build_predicted_overlay(
            server,
            pred["pred_scene_flows"].astype(np.float32),
            pred["gt_scene_flows"].astype(np.float32),
            pred["per_point_epe"].astype(np.float32),
        )
        print(
            f"predicted overlay loaded: pred shape "
            f"{pred['pred_scene_flows'].shape}, "
            f"per_point_epe shape {pred['per_point_epe'].shape}",
        )

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

        # --- Predicted overlay (optional) ------------------------------ #
        if pred_overlay is not None:
            (
                show_pred, epe_colormap, epe_max_slider, pred_line_width,
                pred_summary, pred_handle, pred_seg_handle, pred_marker_handle,
                final_epe,
            ) = pred_overlay
            # Per-point EPE for this frame (use the model's per-frame EPE,
            # or fall back to the constant final-frame EPE if the saved
            # npz only has one row).
            T_p = pred["pred_scene_flows"].shape[0]
            if "per_point_epe" in pred and pred["per_point_epe"].ndim == 2:
                if pred["per_point_epe"].shape[0] > t:
                    frame_epe = pred["per_point_epe"][t]
                else:
                    frame_epe = pred["per_point_epe"][-1]
            else:
                frame_epe = final_epe
            # The predicted cloud lives in its own point ordering
            # (the pipeline subsamples from the clip's full
            # ``scene_flows`` to ``max_scene_points`` before the
            # forward pass, so pred's N is smaller than the clip's).
            # Render the pred cloud on its own with its own
            # downsample, instead of trying to index into pred with
            # the clip's per-camera valid mask.
            pred_t = pred["pred_scene_flows"][t]  # (N_p, 3)
            N_p = pred_t.shape[0]
            n_cap = int(max_points.value)
            if N_p > n_cap:
                sel_p = np.random.default_rng(t + 1234).choice(
                    N_p, size=n_cap, replace=False
                )
                pred_t = pred_t[sel_p]
                frame_epe_disp = frame_epe[sel_p]
            else:
                frame_epe_disp = frame_epe
            if N_p > 0:
                if epe_colormap.value:
                    pred_cols = _epe_to_rgb(
                        frame_epe_disp, float(epe_max_slider.value)
                    )
                else:
                    pred_cols = np.tile(
                        np.array([255, 220, 80], dtype=np.uint8),
                        (pred_t.shape[0], 1),
                    )
                if show_pred.value:
                    pred_handle.points = pred_t
                    pred_handle.colors = pred_cols
                    if pred_seg_handle is not None:
                        pred_seg_handle.visible = True
                else:
                    pred_handle.points = np.zeros((0, 3), dtype=np.float32)
                    pred_handle.colors = np.zeros((0, 3), dtype=np.uint8)
                    if pred_seg_handle is not None:
                        pred_seg_handle.visible = False
                # Per-frame EPE summary text.
                e_mean = float(frame_epe_disp.mean())
                e_max = float(frame_epe_disp.max())
                pred_summary.value = (
                    f"frame t={t} | mean EPE = {e_mean*1000:.1f} mm | "
                    f"max EPE = {e_max*1000:.1f} mm | "
                    f"colormap max = {float(epe_max_slider.value)*1000:.0f} mm"
                )
            else:
                pred_handle.points = np.zeros((0, 3), dtype=np.float32)
                pred_handle.colors = np.zeros((0, 3), dtype=np.uint8)
                if pred_seg_handle is not None:
                    pred_seg_handle.visible = False
                pred_summary.value = f"frame t={t} | no predicted points"

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

    # Predicted overlay UI hooks (only added when pred is loaded).
    if pred_overlay is not None:
        (show_pred, epe_colormap, epe_max_slider, pred_line_width,
         _, _, _, _, _) = pred_overlay

        @show_pred.on_update
        def _on_show_pred(_evt) -> None:
            _update_frame(int(t_slider.value))

        @epe_colormap.on_update
        def _on_epe_colormap(_evt) -> None:
            _update_frame(int(t_slider.value))

        @epe_max_slider.on_update
        def _on_epe_max(_evt) -> None:
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


def _build_predicted_overlay(
    server: viser.ViserServer,
    pred_flows: np.ndarray,
    gt_flows: np.ndarray,
    per_point_epe: np.ndarray,
) -> tuple:
    """Build a (predicted point cloud, predicted trajectory segments,
    EPE slider, show_pred checkbox) tuple.

    ``pred_flows`` and ``gt_flows`` are (T, N, 3) float32. The returned
    handles let the caller hide / show the predicted overlay and tune
    the EPE colormap upper bound interactively.
    """
    T_p, N_p, _ = pred_flows.shape
    # Per-point EPE colormap. EPE is the per-point magnitude of
    # ``pred - gt`` at the final frame; we use that as the colormap
    # driver for the predicted endpoint cloud.
    final_epe = per_point_epe[-1] if per_point_epe.ndim == 2 else per_point_epe
    epe_max_init = float(np.percentile(final_epe, 99)) if N_p > 0 else 0.1
    epe_max_init = max(epe_max_init, 0.01)  # never below 1 cm so slider is monotonic
    slider_min = 0.005
    slider_max = max(0.5, epe_max_init * 2.0)
    # Clamp initial value to the slider range so viser's
    # ``max >= value >= min`` assertion holds even when the per-point
    # 99th-percentile EPE is unusually small (e.g. mostly-static clips).
    epe_max_init = float(np.clip(epe_max_init, slider_min, slider_max))

    with server.gui.add_folder("Predicted flow (vs GT)"):
        show_pred = server.gui.add_checkbox("show predicted overlay",
                                            initial_value=True)
        epe_colormap = server.gui.add_checkbox(
            "color predicted by EPE (vs GT endpoint)",
            initial_value=True,
        )
        epe_max_slider = server.gui.add_slider(
            "EPE colormap max (m)",
            min=slider_min, max=slider_max,
            step=0.001, initial_value=epe_max_init,
        )
        # Track thickness for the predicted trajectory lines.
        pred_line_width = server.gui.add_slider(
            "predicted line width",
            min=0.5, max=5.0, step=0.5, initial_value=2.0,
        )
        pred_summary = server.gui.add_text(
            "pred summary", initial_value="", disabled=True,
        )

    # Predicted point cloud (initial color = neutral white; will be
    # recolored by EPE when the checkbox is on).
    pred_handle = server.scene.add_point_cloud(
        name="/pred",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.006,
    )
    # Predicted trajectory: for each point, draw a polyline through
    # all 11 frames. We pack it as T-1 line segments per point.
    n_segs = N_p * (T_p - 1)
    if n_segs > 0:
        seg_pts = np.empty((n_segs, 2, 3), dtype=np.float32)
        for tt in range(T_p - 1):
            s = tt * N_p
            seg_pts[s:s + N_p, 0, :] = pred_flows[tt]
            seg_pts[s:s + N_p, 1, :] = pred_flows[tt + 1]
        pred_seg_handle = server.scene.add_line_segments(
            name="/pred_traj",
            points=seg_pts,
            colors=(255, 220, 80),  # yellow
            line_width=2.0,
        )
    else:
        pred_seg_handle = None

    # Predicted endpoint frame: a small marker (sphere) at the
    # predicted position of the chosen (single) point at each frame.
    # We use a per-frame point cloud with one "marker" point per
    # tracked point, scaled by EPE.
    pred_marker_handle = server.scene.add_point_cloud(
        name="/pred_markers",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.012,
    )

    return (
        show_pred,
        epe_colormap,
        epe_max_slider,
        pred_line_width,
        pred_summary,
        pred_handle,
        pred_seg_handle,
        pred_marker_handle,
        final_epe,
    )


def _epe_to_rgb(epe: np.ndarray, epe_max: float) -> np.ndarray:
    """Map per-point EPE (m) to RGB. Blue = small, red = large."""
    norm = np.clip(epe / max(epe_max, 1e-6), 0.0, 1.0)
    # 5-stop colormap: blue -> cyan -> green -> yellow -> red.
    stops = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    pos = np.linspace(0, 1, stops.shape[0])
    # Linear interp along stops.
    out = np.zeros((norm.shape[0], 3), dtype=np.float32)
    for i, p in enumerate(pos):
        if i + 1 < len(pos):
            mask = (norm >= p) & (norm < pos[i + 1])
            if mask.any():
                a = (norm[mask] - p) / (pos[i + 1] - p)
                out[mask] = (1 - a[:, None]) * stops[i] + a[:, None] * stops[i + 1]
        else:
            mask = norm >= p
            out[mask] = stops[i]
    return (out * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="Path to .npz clip")
    parser.add_argument(
        "--pred_npz",
        default=None,
        help=(
            "Optional path to the predicted-flow .npz produced by "
            "scripts/eval_libero_clip.py --out_pred_npz. When given, "
            "the viewer renders the predicted trajectory alongside the "
            "GT, with a per-point EPE colormap."
        ),
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    data = np.load(clip_path, allow_pickle=True)
    print(f"loaded {clip_path}, keys={len(data.files)}")

    pred_data = None
    if args.pred_npz is not None:
        pred_path = Path(args.pred_npz)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        pred_data = np.load(pred_path, allow_pickle=True)
        print(f"loaded {pred_path}, keys={list(pred_data.files)}")

    server = viser.ViserServer(port=args.port, label="libero-clip")
    print(f"viser running on http://localhost:{args.port}")
    build_app(dict(data), server, pred=dict(pred_data) if pred_data is not None else None)


if __name__ == "__main__":
    main()
