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
    pred: "np.lib.npyio.NpzFile | dict[str, np.ndarray] | None" = None,
) -> None:
    # --- Load arrays --------------------------------------------------- #
    def _cam(cam_idx: int, name: str) -> np.ndarray:
        """Fetch camera_{cam_idx}_{name} (no silent fallback).

        The previous implementation silently fell back to ``camera_0_{name}``
        when the requested camera was missing. That fallback masquerades a
        real bug as a valid visualization: concatenating ``[cam0, cam0]``
        shows the same table twice with a slight numerical offset (and
        therefore looks like "two misregistered tables"). The doc
        ``docs/指导.md`` calls this out as a debug-hostile trap, so we
        now raise instead.
        """
        key = f"camera_{cam_idx}_{name}"
        if key not in clip:
            raise KeyError(
                f"clip is missing required camera payload {key!r}; "
                "the two-camera fusion will not silently fall back to "
                "camera_0 (see docs/指导.md)."
            )
        return clip[key]

    primary_cam = 0
    scene_flows = _cam(primary_cam, "scene_flows").astype(np.float32)  # (T, N, 3)
    scene_colors = _cam(primary_cam, "scene_colors")  # (T, N, 3) uint8
    depth_mask = _cam(primary_cam, "scene_depth_valid_mask").astype(bool)  # (T, N)
    vis_mask = _cam(primary_cam, "scene_visibility").astype(bool) if (
        f"camera_{primary_cam}_scene_visibility" in clip
    ) else np.ones_like(depth_mask)
    valid_mask = depth_mask & vis_mask  # only render points that are both
                                        # depth-valid and not occluded

    # Combine the available cameras' scene point clouds at each frame so
    # the viewer gets a richer point cloud. We scan for any camera
    # ``camera_N_scene_flows`` key (rather than hardcoding camera_0 +
    # camera_1) so 5-camera exports (e.g. agentview + birdview +
    # sideview + frontview + robot0_eye_in_hand) also get fused.
    cam_ids = []
    for _ci in range(8):
        if f"camera_{_ci}_scene_flows" in clip:
            cam_ids.append(_ci)
    print(f"discovered {len(cam_ids)} camera(s) in clip: {cam_ids}")
    flows_c_list = []
    cols_c_list = []
    valid_c_list = []
    for cid in cam_ids:
        fc = _cam(cid, "scene_flows").astype(np.float32)
        cc = _cam(cid, "scene_colors")
        mc = _cam(cid, "scene_depth_valid_mask").astype(bool)
        vc = _cam(cid, "scene_visibility").astype(bool) if (
            f"camera_{cid}_scene_visibility" in clip
        ) else np.ones_like(mc)
        flows_c_list.append(fc)
        cols_c_list.append(cc)
        valid_c_list.append(mc & vc)

    flows = np.concatenate(flows_c_list, axis=1)  # (T, sum N_c, 3)
    colors = np.concatenate(cols_c_list, axis=1)  # (T, N, 3) uint8
    valid = np.concatenate(valid_c_list, axis=1)  # (T, N)
    T, N, _ = flows.shape

    # The t_slider is bounded by the larger of (clip length, prediction
    # length). When --rollout was used the pred is much longer than the
    # input clip (e.g. 121 frames vs 11), so the user can scrub past the
    # last clip frame to see only the model prediction continue. The
    # scene point cloud and gripper trail are clipped to the clip
    # length; the predicted overlay continues to render. We compute
    # these up front because the slider creation needs T_max.
    T_pred = T  # will be overwritten below if pred is loaded
    T_max = T

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
    # If a prediction .npz was loaded, prepend a bright MODEL PREDICTION
    # banner to the GUI so the user can never confuse this view for the
    # ground-truth physics replay. This addresses the user feedback:
    # "你要确定可视化出来是预测的结果，而不是在libero里面使用物理引擎回放"
    # -- they want it to be obvious at a glance that the rendered point
    # cloud + trajectories are the model's forward prediction, not the
    # LIBERO sim's deterministic playback.
    if pred is not None and "pred_scene_flows" in pred:
        server.gui.add_markdown(
            "## ⚠ MODEL PREDICTION (not LIBERO physics replay)\n"
            "**Yellow = model predicted trajectory. Blue = GT (input clip). "
            "Magenta = per-point flow arrows (t→t+1). Green = subsampled GT "
            "(model input subset).**"
        )
    # The data mode itself (GT-only / GT+pred) is summarized at the bottom
    # in a separate, non-decorative text box.
    mode_text = "MODE: GT + MODEL PREDICTION" if (
        pred is not None and "pred_scene_flows" in pred
    ) else "MODE: GT (input clip only, no model output)"

    with server.gui.add_folder("Clip controls"):
        t_slider = server.gui.add_slider(
            "frame t", min=0, max=T_max - 1, step=1, initial_value=0
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
        # Per-point flow arrows: a small line segment from each
        # tracked point at the current frame ``t`` to its position at
        # ``t+1`` (clipped at the last frame). This is the most direct
        # "current per-point flow" indicator -- you see exactly which
        # way each scene point is heading at the current moment.
        # Default ON because the user explicitly asked for
        # "每个点对应的点流" (per-point flow) to be visible.
        show_flow_arrows = server.gui.add_checkbox(
            "show per-point flow arrows (t -> t+1)",
            initial_value=True,
        )
        # Multiplier on the per-point flow arrow length. Some clips have
        # very small per-frame motion (1-2 mm) which is hard to see even
        # with thick lines. A length multiplier amplifies the visual
        # displacement so the arrow's *direction* is obvious, even if
        # its *magnitude* is no longer to scale.
        flow_arrow_length = server.gui.add_slider(
            "flow arrow length multiplier",
            min=1.0, max=50.0, step=0.5, initial_value=10.0,
        )
        # Thickness of the GT scene trails (and the predicted-flow
        # trails if --pred_npz is loaded). Default is generous because
        # the old 1.5-px lines were too thin to read in the dense union.
        trail_width = server.gui.add_slider(
            "trail line width",
            min=1.0, max=6.0, step=0.5, initial_value=3.0,
        )
        # Subsampled-GT toggle lives in the "Predicted flow (vs GT)"
        # folder (so it's grouped with the other model-output controls)
        # and is created in _build_predicted_overlay; we initialize the
        # local reference to None here and overwrite it below.
        show_subsampled_gt = None
        play_button = server.gui.add_button("Play / Pause")
        info_text = server.gui.add_text("clip info", initial_value=mode_text, disabled=True)

    # --- Predicted overlay (optional) ----------------------------------- #
    pred_overlay = None
    if pred is not None and "pred_scene_flows" in pred and "gt_scene_flows" in pred:
        # The model's ``pred_scene_flows`` and ``gt_scene_flows`` are
        # emitted in the **centered (robot-base) frame**, because the
        # release pipeline runs ``center_shift`` on the input sample
        # before the model sees it. We have to undo it for the overlay
        # to line up with the world-frame cam0/cam1 point cloud.
        #
        # Two sources for the shift:
        #   1. Newer pred npz files stamp ``__shift_amount__`` /
        #      ``shift_amount`` from the eval script, computed using
        #      the *post-grid-sample* scene+robot centroid (i.e. the
        #      exact transform the model saw).
        #   2. As a fallback, recompute from the raw clip by
        #      replicating ``center_shift``'s math on the same set of
        #      points the model saw (t=0 valid scene + t=0 robot).
        #      This is *not* exact, because the release pipeline runs
        #      ``grid_sample_transform`` + ``enforce_max_num_points``
        #      before ``center_shift``, but it gets within a few cm.
        shift_amount = None
        if "shift_amount" in pred.files:
            shift_amount = np.asarray(pred["shift_amount"], dtype=np.float32)
        elif "__shift_amount__" in pred.files:
            shift_amount = np.asarray(pred["__shift_amount__"], dtype=np.float32)
        if shift_amount is not None:
            # ``__shift_amount__`` semantics (see center_shift in
            # dataset_components/transforms.py): centered = world + s,
            # so pred_world = pred - s.
            pred_world = pred["pred_scene_flows"].astype(np.float32) - shift_amount
            gt_world = pred["gt_scene_flows"].astype(np.float32) - shift_amount
            print(
                f"predicted overlay: using stamped shift_amount = "
                f"({shift_amount[0]:+.3f}, {shift_amount[1]:+.3f}, "
                f"{shift_amount[2]:+.3f}) m (pred stored in world frame)"
            )
        else:
            valid_t0 = valid[0] if valid.shape[0] > 0 else None
            scene_t0 = flows[0]
            if valid_t0 is not None and valid_t0.shape[0] == scene_t0.shape[0]:
                scene_t0_valid = scene_t0[valid_t0]
            else:
                scene_t0_valid = scene_t0
            robot_t0 = robot_flows[0] if has_robot else np.zeros(
                (0, 3), dtype=np.float32
            )
            if scene_t0_valid.shape[0] + robot_t0.shape[0] > 0:
                combined_t0 = np.concatenate(
                    [scene_t0_valid, robot_t0], axis=0
                )
                shift = combined_t0.mean(axis=0).astype(np.float32)
            else:
                shift = np.zeros(3, dtype=np.float32)
            pred_world = pred["pred_scene_flows"].astype(np.float32) + shift
            gt_world = pred["gt_scene_flows"].astype(np.float32) + shift
            print(
                f"predicted overlay: fallback shift = "
                f"({shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}) m "
                f"(may be off by a few cm; re-run eval with --out_pred_npz "
                f"to stamp shift_amount)"
            )
        # The saved predicted-flow npz carries (T, N, 3) for both pred
        # and GT plus per_point_epe (T, N). We re-use the *input clip's*
        # per-point validity mask to decide which points to display, so
        # the overlay lines up with the GT cloud.
        pred_overlay = _build_predicted_overlay(
            server,
            pred_world,
            gt_world,
            pred["per_point_epe"].astype(np.float32),
        )
        print(
            f"predicted overlay loaded: pred shape {pred_world.shape}, "
            f"per_point_epe shape {pred['per_point_epe'].shape}",
        )
        T_pred = pred_world.shape[0]
        T_max = max(T, T_pred)
        # Extend the t_slider so the user can scrub past the last clip
        # frame into the model-rollout region. Viser allows updating
        # slider bounds after creation.
        t_slider.max = T_max - 1
        info_text.value = (
            f"MODE: GT + MODEL PREDICTION | clip T={T}, pred T={T_pred}"
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
    # 3D watermark: a labeled "MODEL PREDICTION" anchor placed in the
    # scene above the table. Viser renders this as a 2D label always
    # facing the camera, so it's visible no matter what angle the
    # user has rotated to. We position it at a fixed offset above the
    # scene centroid + the gripper's t=0 location, so it floats above
    # the action area regardless of the specific clip.
    if pred is not None and "pred_scene_flows" in pred:
        # Estimate a stable anchor point: median of valid scene flow
        # at t=0, lifted ~30 cm in +z so it sits above the table
        # and is unlikely to be occluded.
        valid_t0 = valid[0]
        scene_t0 = flows[0]
        if valid_t0 is not None and valid_t0.shape[0] == scene_t0.shape[0]:
            scene_t0_valid = scene_t0[valid_t0]
        else:
            scene_t0_valid = scene_t0
        if scene_t0_valid.shape[0] > 0:
            anchor = np.median(scene_t0_valid, axis=0).astype(np.float32)
        else:
            anchor = np.zeros(3, dtype=np.float32)
        anchor[2] += 0.30
        try:
            server.scene.add_label(
                name="/pred_watermark",
                text="⚠ MODEL PREDICTION\n(not LIBERO physics replay)",
                position=anchor,
            )
        except Exception:
            # Older viser may not have add_label; fall back silently.
            pass
    # Subsampled-GT point cloud (only present when --pred_npz is given).
    # This is the *exact* world-frame point set the model was asked to
    # predict: the 12000-point grid_sample + enforce_max_num_points
    # subset of the union. Rendering it side-by-side with the predicted
    # overlay makes the alignment self-evident: for static scene points
    # the two clouds should be on top of each other (EPE ~ 0), and the
    # sparse GT shows the user that the prediction lives at the same
    # *physical* locations as the GT, not at some shifted "average"
    # position derived from the dense union's centroid.
    gt_world_handle = None
    gt_trail_handle = None
    if pred is not None and "gt_scene_flows" in pred and "shift_amount" in pred.files:
        gt_world = pred["gt_scene_flows"].astype(np.float32) - np.asarray(
            pred["shift_amount"], dtype=np.float32
        )
        gt_world_handle = server.scene.add_point_cloud(
            name="/gt_subsampled",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.full(
                (1, 3), [120, 220, 120], dtype=np.uint8
            ),  # green
            point_size=0.005,
        )
        # Draw short GT trails for moving points (only those whose
        # per-frame displacement > 5mm) so the user can see how the
        # model-tracked subset moves over time.
        disp_g = np.linalg.norm(gt_world[1:] - gt_world[:1], axis=-1)
        per_pt_max_g = disp_g.max(axis=0)
        moving_g = per_pt_max_g > 0.005
        if moving_g.any():
            max_trails_g = 200
            idxs_g = np.where(moving_g)[0]
            if idxs_g.shape[0] > max_trails_g:
                sel_g = np.random.default_rng(0).choice(
                    idxs_g.shape[0], size=max_trails_g, replace=False
                )
                idxs_g = idxs_g[sel_g]
            trails_g = gt_world[:, idxs_g, :]
            n_pts_g = trails_g.shape[1]
            segs_g = np.empty((n_pts_g * (gt_world.shape[0] - 1), 2, 3), dtype=np.float32)
            for i in range(gt_world.shape[0] - 1):
                segs_g[i * n_pts_g:(i + 1) * n_pts_g, 0, :] = trails_g[i]
                segs_g[i * n_pts_g:(i + 1) * n_pts_g, 1, :] = trails_g[i + 1]
            gt_trail_handle = server.scene.add_line_segments(
                name="/gt_subsampled_trails",
                points=segs_g,
                colors=(120, 220, 120),
                line_width=1.0,
            )

    # --- Trail handles ------------------------------------------------- #
    scene_trail_handle = None
    robot_trail_handle = None
    # Per-point flow arrows: one short segment per tracked point,
    # going from position at frame t to position at frame t+1 (clipped
    # at the last frame). Updated each frame via ``_update_frame``.
    # ``flow_arrow_segments`` holds the (n_pts, 2, 3) geometry; the
    # actual frame-dependent segment endpoints are re-uploaded by
    # ``_update_frame`` (we resize the handle to the current moving
    # count and assign the segments for the current t).
    flow_arrow_handle = None
    flow_arrow_segments_per_t: list[np.ndarray] = []

    if show_trails.value:
        # Scene trails: pick points whose max-frame displacement > 2mm.
        # Lowered from the original 5mm so static-but-slightly-shifting
        # points (e.g. the grasped bowl as it sits in the gripper) also
        # show up; bumped the per-frame cap to 500 so the trails feel
        # less sparse.
        disp = np.linalg.norm(flows[1:] - flows[:1], axis=-1)  # (T-1, N)
        per_pt_max = disp.max(axis=0)
        moving = valid[0] & (per_pt_max > 0.002)
        if moving.any():
            max_trails = 500
            idxs = np.where(moving)[0]
            if idxs.shape[0] > max_trails:
                # Bias the downsample toward the *most* moving points so
                # the static-but-moving bowl is over-represented in the
                # trail set (otherwise the dense union overwhelms it).
                sorted_idxs = idxs[np.argsort(-per_pt_max[idxs])]
                idxs = sorted_idxs[:max_trails]
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
                line_width=trail_width.value,
            )
            # Pre-compute per-frame flow-arrow *direction* (unit
            # displacement) for the same moving points. We store the
            # *direction* rather than the actual endpoint so we can
            # re-scale the arrow at render time with the
            # ``flow_arrow_length`` slider -- some clips have per-frame
            # motion of <2 mm which is invisible regardless of line
            # width. ``flow_arrow_segments_per_t[t]`` is a (n_pts, 2, 3)
            # array of segment endpoints; ``_update_frame`` uses the
            # current multiplier + line width when pushing the data.
            for tt in range(T - 1):
                seg = np.stack([trails[tt], trails[tt + 1]], axis=1)
                # Direction = endpoint - startpoint. Keep the raw delta
                # so the user can see *which way* each point is going;
                # the renderer multiplies it by the length slider.
                flow_arrow_segments_per_t.append(seg)

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
                line_width=trail_width.value,
            )

    # Initialize the flow-arrow line-segment handle lazily on the
    # first frame, once we know how many moving points we have.
    # (Re-uploaded each frame by ``_update_frame``.)
    if flow_arrow_segments_per_t:
        # Pre-allocate a handle of the right size with zeroed points;
        # the real endpoints land there on the first ``_update_frame``
        # call. We default the line width to a generous 4 px so the
        # arrow stands out from the blue GT trails (which default to
        # 3 px); the user can still tune it via the trail_width
        # slider.
        n_init = flow_arrow_segments_per_t[0].shape[0]
        flow_arrow_handle = server.scene.add_line_segments(
            name="/flow_arrows",
            points=np.zeros((n_init, 2, 3), dtype=np.float32),
            colors=(255, 80, 200),  # magenta-pink so they stand out
            line_width=4.0,
        )

    # --- Gripper axis frame -------------------------------------------- #
    gripper_frame_handle = None

    # --- State --------------------------------------------------------- #
    state = {"playing": False, "last_t": time.time()}

    def _update_frame(t: int) -> None:
        # If t is past the clip's last frame, only the predicted overlay
        # should render. The scene point cloud / gripper / trails are
        # frozen at their last available frame; the predicted cloud
        # (yellow) keeps updating to follow the model's rollout.
        t_in_clip = min(t, T - 1)

        # --- Scene ----------------------------------------------------- #
        valid_t = valid[t_in_clip]
        if not valid_t.any():
            scene_handle.points = np.zeros((0, 3), dtype=np.float32)
            scene_handle.colors = np.zeros((0, 3), dtype=np.uint8)
        else:
            idxs = np.where(valid_t)[0]
            n_cap = int(max_points.value)
            if idxs.shape[0] > n_cap:
                sel = np.random.default_rng(t_in_clip).choice(
                    idxs.shape[0], size=n_cap, replace=False
                )
                idxs = idxs[sel]
            scene_handle.points = flows[t_in_clip, idxs, :]
            scene_handle.colors = colors[t_in_clip, idxs, :]

        # --- Subsampled GT (when --pred_npz is given) ----------------- #
        # The GT is the 12000-point subset of the union that the model
        # actually saw. For static points the predicted cloud overlays
        # this exactly; rendering both makes the alignment obvious.
        if gt_world_handle is not None and show_subsampled_gt.value:
            gt_t_in_clip = min(t, gt_world.shape[0] - 1)
            gt_world_handle.points = gt_world[gt_t_in_clip]
            gt_world_handle.visible = True
        elif gt_world_handle is not None:
            gt_world_handle.visible = False

        # --- Per-point flow arrows (t -> t+1) ------------------------- #
        # The most direct "what is each point doing right now"
        # indicator: a short magenta line from position at frame t to
        # position at frame min(t+1, T-1). Re-uploaded every frame so
        # the arrow *direction* changes as the slider moves. The
        # endpoint is the raw ``t+1`` position plus a user-controlled
        # length multiplier so very small per-frame motion (1-2 mm)
        # is still visible.
        if flow_arrow_handle is not None and show_flow_arrows.value and t < T - 1:
            t_arrow = t
            seg = flow_arrow_segments_per_t[t_arrow]  # (n_pts, 2, 3)
            mult = float(flow_arrow_length.value)
            # Apply length multiplier: shift endpoint away from start
            # by ``(mult - 1) * (end - start)``. mult=1 means no
            # amplification (true-to-scale). mult=10 means the arrow
            # is 10x its real length, which is great for inspecting
            # direction even at very slow motion.
            delta = (seg[:, 1] - seg[:, 0]) * (mult - 1.0)
            flow_arrow_handle.points = np.stack(
                [seg[:, 0], seg[:, 1] + delta], axis=1
            ).astype(np.float32)
            flow_arrow_handle.visible = True
        elif flow_arrow_handle is not None:
            flow_arrow_handle.visible = False

        # --- Robot ----------------------------------------------------- #
        if has_robot and show_robot.value:
            pts_r = robot_flows[t_in_clip]
            if pts_r.shape[0] == 0:
                robot_handle.points = np.zeros((0, 3), dtype=np.float32)
                robot_handle.colors = np.zeros((0, 3), dtype=np.uint8)
            else:
                robot_handle.points = pts_r
                if robot_colors is not None and robot_colors.shape[1] > 0:
                    robot_handle.colors = robot_colors[t_in_clip]
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
        gp = gripper_pose_7d[t_in_clip]
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
                final_epe, _show_subsampled_gt,
            ) = pred_overlay
            # Per-point EPE for this frame (use the model's per-frame EPE,
            # or fall back to the constant final-frame EPE if the saved
            # npz only has one row). EPE is computed against the
            # world-frame GT (``gt_world``/``pred_world``) so it is
            # invariant to the choice of centering.
            T_p = pred_world.shape[0]
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
            #
            # CRITICAL: read from ``pred_world`` (not
            # ``pred["pred_scene_flows"]``) so the predicted cloud
            # lines up with the world-frame cam0/cam1 point cloud.
            # ``pred_world`` was already corrected for the model's
            # ``center_shift`` at the top of ``build_app``.
            pred_t = pred_world[t]  # (N_p, 3) in world frame
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
            f"frame t={t}/{T_max - 1} (clip T={T}, pred T={T_pred})  "
            f"gripper=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
            f"open={'open' if gripper_open[t_in_clip] > 0.5 else 'closed'}"
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

    @show_flow_arrows.on_update
    def _on_arrows(_evt) -> None:
        _update_frame(int(t_slider.value))

    @flow_arrow_length.on_update
    def _on_arrow_length(_evt) -> None:
        _update_frame(int(t_slider.value))

    @trail_width.on_update
    def _on_trail_width(_evt) -> None:
        # Viser doesn't allow changing line_width on an existing
        # line_segments handle, so just re-upload the segments to
        # re-trigger the update. The width itself is bound at handle
        # creation, so a width change here is best-effort: the
        # segments still get re-uploaded, which forces a redraw.
        if scene_trail_handle is not None:
            scene_trail_handle.line_width = trail_width.value
        if robot_trail_handle is not None:
            robot_trail_handle.line_width = trail_width.value
        if flow_arrow_handle is not None:
            flow_arrow_handle.line_width = trail_width.value
        if pred_overlay is not None:
            # pred_seg_handle lives in the overlay tuple
            try:
                pred_seg_handle = pred_overlay[6]
                if pred_seg_handle is not None:
                    pred_seg_handle.line_width = trail_width.value
            except (IndexError, AttributeError):
                pass

    # Predicted overlay UI hooks (only added when pred is loaded).
    if pred_overlay is not None:
        (show_pred, epe_colormap, epe_max_slider, pred_line_width,
         _, _, _, _, _, show_subsampled_gt) = pred_overlay
        # ``show_subsampled_gt`` is created inside _build_predicted_overlay
        # (it lives in the "Predicted flow (vs GT)" folder). The local
        # binding here is the one ``_update_frame`` reads via the closure
        # of ``build_app``.

        @show_pred.on_update
        def _on_show_pred(_evt) -> None:
            _update_frame(int(t_slider.value))

        @epe_colormap.on_update
        def _on_epe_colormap(_evt) -> None:
            _update_frame(int(t_slider.value))

        @epe_max_slider.on_update
        def _on_epe_max(_evt) -> None:
            _update_frame(int(t_slider.value))

        @show_subsampled_gt.on_update
        def _on_show_gt(_evt) -> None:
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
                next_t = (int(t_slider.value) + 1) % T_max
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
        # The subsampled-GT (the 12000-point subset of the union that
        # the model actually saw) is on by default so the user can see
        # the alignment between pred and the model-input GT. Toggle
        # off if the dense union is the only thing of interest.
        show_subsampled_gt = server.gui.add_checkbox(
            "show subsampled GT (model input)", initial_value=True,
        )
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
        show_subsampled_gt,
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
    # The clip payload uses ``dict.get`` style access (which NpzFile
    # does not implement), so we convert it to a plain dict. The pred
    # payload, on the other hand, needs ``pred.files`` to probe for
    # optional keys like ``shift_amount`` -- that attribute is only
    # on the lazy NpzFile, so we pass it through unchanged.
    build_app(dict(data), server, pred=pred_data)


if __name__ == "__main__":
    main()
