"""Build a self-contained HTML viewer for exported LIBERO ``.npz`` clips.

The viewer separates three concepts that are easy to confuse in the raw files:

* valid scene points back-projected by each exported camera;
* the point subset produced by the PointWorld training/evaluation sampler;
* robot mesh points, which are not camera points.

By default the training preview uses the current repository defaults (three
cameras, 15 mm voxels, at most 12,000 scene points).  ``--pipeline_mode train``
also runs the stochastic sphere crop and random cap with a reproducible seed.

Example::

    python -m tools.libero.viz_html \
        --clips /path/to/clip_a.npz /path/to/clip_b.npz \
        --output /tmp/libero_viz.html
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset_components.constants import (
    RELEASE_SPHERECROP_BUFFER,
    RELEASE_SPHERECROP_MAX_R,
    RELEASE_SPHERECROP_MIN_R,
    RELEASE_SPHERECROP_PROB,
    RELEASE_NUM_CANDIDATES,
    RELEASE_MAX_NUM_SPHERES,
)
from dataset_components.transforms import (
    center_shift,
    enforce_max_num_points,
    filter_within_bounds,
    grid_sample_transform,
    sphere_crop_transform,
)


def _camera_prefixes(raw: Any) -> list[str]:
    prefixes = {
        key[: -len("_scene_flows")]
        for key in raw.files
        if key.endswith("_scene_flows") and key != "scene_flows"
    }

    def sort_key(prefix: str) -> tuple[int, str]:
        suffix = prefix.rsplit("_", 1)[-1]
        return (int(suffix), prefix) if suffix.isdigit() else (10**9, prefix)

    return sorted(prefixes, key=sort_key)


def _scene_valid_mask(raw: Any, prefix: str, flows: np.ndarray) -> np.ndarray:
    visibility_key = f"{prefix}_scene_visibility"
    if visibility_key in raw.files:
        visibility = np.asarray(raw[visibility_key], dtype=bool)
        if visibility.shape[:2] == flows.shape[:2]:
            return visibility[0]
    xyz0 = flows[0]
    return np.isfinite(xyz0).all(axis=1) & np.any(xyz0 != 0, axis=1)


def _encode_ids(ids: np.ndarray, timesteps: int) -> np.ndarray:
    """Store integer point ids losslessly in the RGB bytes transforms preserve."""
    ids = np.asarray(ids, dtype=np.uint32)
    packed = np.stack((ids & 255, (ids >> 8) & 255, (ids >> 16) & 255), axis=1)
    return np.repeat(packed[None, :, :], timesteps, axis=0).astype(np.uint8)


def _decode_ids(colors: np.ndarray) -> np.ndarray:
    rgb = colors[0].astype(np.uint32)
    return rgb[:, 0] | (rgb[:, 1] << 8) | (rgb[:, 2] << 16)


def _training_subset(
    raw: Any,
    prefixes: list[str],
    *,
    num_cameras: int,
    grid_size: float,
    max_scene_points: int,
    pipeline_mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run the point-selection part of the real pipeline and retain source ids.

    Coordinates and RGB returned here are read back from the unaugmented camera
    union.  This lets the selected points overlay the raw cameras exactly even
    when the train-mode selection used centered coordinates and sphere crop.
    """
    selected = prefixes[:num_cameras]
    all_flows = [np.asarray(raw[f"{p}_scene_flows"], dtype=np.float32) for p in selected]
    all_colors = [np.asarray(raw[f"{p}_scene_colors"], dtype=np.uint8) for p in selected]
    flows = np.concatenate(all_flows, axis=1)
    colors = np.concatenate(all_colors, axis=1)
    timesteps, point_count = flows.shape[:2]
    if point_count >= 2**24:
        raise ValueError("Training preview supports fewer than 2**24 source points")

    camera_ids = np.concatenate([
        np.full(cam_flows.shape[1], cam_id, dtype=np.uint8)
        for cam_id, cam_flows in enumerate(all_flows)
    ])
    sample = {
        "__key__": str(raw["__key__"]) if "__key__" in raw.files else "",
        "scene_flows": flows.copy(),
        "scene_colors": _encode_ids(np.arange(point_count), timesteps),
        "robot_flows": np.asarray(raw["robot_flows"], dtype=np.float32).copy(),
    }
    if "workspace_bounds" in raw.files:
        # Match the real training pipeline: exported workspace crops opt out
        # of the otherwise redundant far-background sphere crop.
        sample["workspace_bounds"] = np.asarray(raw["workspace_bounds"], dtype=np.float32)

    # Training sampling uses numpy's process RNG.  Scope and restore it so HTML
    # generation is reproducible without changing the caller's random state.
    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        sample = center_shift(sample)
        sample = filter_within_bounds(sample)
        sample = grid_sample_transform(sample, grid_size=grid_size, mode=pipeline_mode)
        if pipeline_mode == "train":
            sample = sphere_crop_transform(
                sample,
                prob=RELEASE_SPHERECROP_PROB,
                min_radius=RELEASE_SPHERECROP_MIN_R,
                max_radius=RELEASE_SPHERECROP_MAX_R,
                buffer=RELEASE_SPHERECROP_BUFFER,
                num_candidates=RELEASE_NUM_CANDIDATES,
                max_num_spheres=RELEASE_MAX_NUM_SPHERES,
                max_scene_points=max_scene_points,
            )
            sample = enforce_max_num_points(sample, max_scene_points=max_scene_points)
        else:
            sample = enforce_max_num_points(
                sample,
                max_scene_points=max_scene_points,
                deterministic=True,
                seed=seed,
            )
    finally:
        np.random.set_state(random_state)

    source_indices = _decode_ids(sample["scene_colors"])
    return {
        "flows": flows[:, source_indices, :],
        "colors": colors[:, source_indices, :],
        "camera_ids": camera_ids[source_indices],
        "source_indices": source_indices,
    }


def _clip_points(
    npz_path: Path,
    *,
    max_cam_points: int,
    num_cameras: int,
    grid_size: float,
    max_scene_points: int,
    pipeline_mode: str,
    seed: int,
) -> dict[str, Any]:
    raw = np.load(npz_path, allow_pickle=True)
    prefixes = _camera_prefixes(raw)
    if not prefixes:
        raise ValueError(f"No camera-prefixed scene_flows in {npz_path}")
    if not 1 <= num_cameras <= len(prefixes):
        raise ValueError(
            f"--num_cameras={num_cameras}, but {npz_path} contains {len(prefixes)} cameras"
        )

    cameras = []
    for prefix in prefixes:
        flows = np.asarray(raw[f"{prefix}_scene_flows"], dtype=np.float32)
        colors = np.asarray(raw[f"{prefix}_scene_colors"], dtype=np.uint8)
        if flows.ndim != 3 or flows.shape[-1] != 3 or colors.shape != flows.shape:
            raise ValueError(
                f"{prefix}: expected flows/colors with matching (T, N, 3), "
                f"got {flows.shape} and {colors.shape}"
            )
        valid = _scene_valid_mask(raw, prefix, flows)
        valid_indices = np.flatnonzero(valid)
        if max_cam_points > 0 and valid_indices.size > max_cam_points:
            sample_idx = np.linspace(
                0, valid_indices.size - 1, max_cam_points, dtype=np.int64
            )
            valid_indices = valid_indices[sample_idx]
        cameras.append({
            "prefix": prefix,
            "flows": flows[:, valid_indices, :],
            "colors": colors[:, valid_indices, :],
            "valid_count": int(valid.sum()),
        })

    training = _training_subset(
        raw,
        prefixes,
        num_cameras=num_cameras,
        grid_size=grid_size,
        max_scene_points=max_scene_points,
        pipeline_mode=pipeline_mode,
        seed=seed,
    )
    camera_names = [str(value) for value in raw.get("camera_names", [])]
    return {
        "name": npz_path.name,
        "key": str(raw["__key__"]) if "__key__" in raw.files else npz_path.stem,
        "T": int(cameras[0]["flows"].shape[0]),
        "cameras": cameras,
        "training": training,
        "robot_flows": np.asarray(raw["robot_flows"], dtype=np.float32),
        "robot_colors": np.asarray(raw["robot_colors"], dtype=np.uint8),
        "gripper_pose": np.asarray(raw["right_gripper_pose"], dtype=np.float32),
        "gripper_open": np.asarray(raw["right_gripper_open"], dtype=np.float32),
        "camera_names": camera_names,
        "pipeline": {
            "mode": pipeline_mode,
            "num_cameras": num_cameras,
            "grid_size": grid_size,
            "max_scene_points": max_scene_points,
            "seed": seed,
            "sphere_crop": (
                "skipped_workspace"
                if pipeline_mode == "train" and "workspace_bounds" in raw.files
                else ("enabled" if pipeline_mode == "train" else "not_used")
            ),
        },
        "workspace_bounds": (
            np.asarray(raw["workspace_bounds"], dtype=np.float32)
            if "workspace_bounds" in raw.files else None
        ),
        "workspace_crop_reference": (
            str(raw["workspace_crop_reference"])
            if "workspace_crop_reference" in raw.files else ""
        ),
        "workspace_points_before": (
            np.asarray(raw["workspace_points_before_per_cam"], dtype=np.int64).tolist()
            if "workspace_points_before_per_cam" in raw.files else []
        ),
        "workspace_points_after": (
            np.asarray(raw["workspace_points_after_per_cam"], dtype=np.int64).tolist()
            if "workspace_points_after_per_cam" in raw.files else []
        ),
    }


def _b64(array: np.ndarray, dtype: np.dtype) -> str:
    data = np.ascontiguousarray(array, dtype=dtype).tobytes()
    return base64.b64encode(data).decode("ascii")


def _serialize_clip(clip: dict[str, Any]) -> dict[str, Any]:
    def cloud(flows: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
        return {
            "flows_b64": _b64(flows, np.float32),
            "colors_b64": _b64(colors, np.uint8),
            "N": int(flows.shape[1]),
        }

    cameras = []
    for camera in clip["cameras"]:
        payload = cloud(camera["flows"], camera["colors"])
        payload.update(prefix=camera["prefix"], valid_count=camera["valid_count"])
        cameras.append(payload)
    training = cloud(clip["training"]["flows"], clip["training"]["colors"])
    training["camera_ids_b64"] = _b64(clip["training"]["camera_ids"], np.uint8)
    robot = cloud(clip["robot_flows"], clip["robot_colors"])
    return {
        "name": clip["name"],
        "key": clip["key"],
        "T": clip["T"],
        "cameras": cameras,
        "training": training,
        "robot": robot,
        "gripper_pose_b64": _b64(clip["gripper_pose"], np.float32),
        "gripper_open_b64": _b64(clip["gripper_open"], np.float32),
        "camera_names": clip["camera_names"],
        "pipeline": clip["pipeline"],
        "workspace_bounds": (
            clip["workspace_bounds"].tolist()
            if clip["workspace_bounds"] is not None else None
        ),
        "workspace_crop_reference": clip["workspace_crop_reference"],
        "workspace_points_before": clip["workspace_points_before"],
        "workspace_points_after": clip["workspace_points_after"],
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PointWorld 训练点检查器</title>
__THREE_SCRIPT__
__ORBIT_SCRIPT__
<style>
  :root { color-scheme: dark; --panel: rgba(20,22,24,.94); --line: #454a50; --text: #f1f3f5; --muted: #aeb4bb; --accent: #42a5f5; }
  * { box-sizing: border-box; letter-spacing: 0; }
  body { margin: 0; overflow: hidden; font: 13px/1.4 system-ui, "Segoe UI", sans-serif; background: #17191b; color: var(--text); }
  canvas { display: block; }
  #toolbar { position: fixed; inset: 0 0 auto 0; z-index: 10; min-height: 52px; padding: 8px 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; background: var(--panel); border-bottom: 1px solid var(--line); }
  .group { display: flex; align-items: center; gap: 6px; padding-right: 10px; border-right: 1px solid var(--line); }
  .group:last-child { border-right: 0; }
  button, select { height: 32px; border: 1px solid #565c63; border-radius: 4px; background: #2b2f33; color: var(--text); cursor: pointer; }
  button { width: 32px; padding: 0; font-size: 17px; }
  button:hover, select:hover { border-color: #8c949d; }
  button.active { background: #185f8d; border-color: #54b7f3; }
  select { padding: 0 28px 0 9px; }
  input[type="range"] { width: 180px; accent-color: var(--accent); }
  label { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; color: #dce0e4; }
  input[type="checkbox"] { accent-color: var(--accent); }
  .counter { min-width: 58px; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }
  #status { position: fixed; top: 62px; left: 12px; z-index: 5; max-width: min(520px, calc(100vw - 24px)); padding: 9px 11px; border: 1px solid var(--line); border-radius: 5px; background: rgba(20,22,24,.84); color: var(--muted); pointer-events: none; }
  #status strong { color: var(--text); }
  #legend { position: fixed; right: 12px; bottom: 12px; z-index: 5; padding: 9px 11px; border: 1px solid var(--line); border-radius: 5px; background: rgba(20,22,24,.84); line-height: 1.8; pointer-events: none; }
  .dot { display: inline-block; width: 10px; height: 10px; margin-right: 7px; border-radius: 50%; vertical-align: -1px; }
  .cam0 { background: #20c997; } .cam1 { background: #ffd43b; } .cam2 { background: #ff6b9a; } .train { background: #fff; border: 1px solid #777; }
  @media (max-width: 900px) { #toolbar { gap: 5px; padding: 6px; } .group { padding-right: 5px; } input[type="range"] { width: 115px; } #status { top: 165px; } }
</style>
</head>
<body>
<div id="toolbar">
  <div class="group"><button id="prev-clip" title="上一个 clip">‹</button><span id="clip-count" class="counter">0 / 0</span><button id="next-clip" title="下一个 clip">›</button></div>
  <div class="group"><button id="prev-frame" title="上一帧">↑</button><input id="frame" type="range" min="0" max="10" value="0" title="时间帧"><button id="next-frame" title="下一帧">↓</button><span id="frame-count" class="counter">0 / 10</span><button id="play" title="播放或暂停">▶</button></div>
  <div class="group"><label>着色<select id="color-mode" title="按相机来源或原始 RGB 着色"><option value="source">相机来源</option><option value="rgb">原始 RGB</option></select></label></div>
  <div class="group"><label><input id="raw" type="checkbox" checked>相机点</label><label><input id="training" type="checkbox" checked>训练点</label><label><input id="robot" type="checkbox">机器人点</label><label><input id="workspace" type="checkbox" checked>工作区</label></div>
  <div class="group"><label><input id="cam0" type="checkbox" checked>相机 0</label><label><input id="cam1" type="checkbox" checked>相机 1</label><label><input id="cam2" type="checkbox" checked>相机 2</label></div>
  <div class="group"><label>点大小<input id="point-size" type="range" min="0.5" max="4" step="0.25" value="1"></label><button id="reset" title="重置视角">⌂</button></div>
</div>
<div id="status">正在解码...</div>
<div id="legend"><div><span class="dot cam0"></span>相机 0</div><div><span class="dot cam1"></span>相机 1</div><div><span class="dot cam2"></span>相机 2</div><div><span class="dot train"></span>训练点使用相同来源色，尺寸更大</div><div style="color:#9ccc65">□ 导出工作区边界</div></div>
<script>
"use strict";
const DATA = __DATA_JSON__;
const SOURCE_COLORS = [[0.125,0.788,0.592],[1.0,0.831,0.231],[1.0,0.420,0.604]];
let clipIndex = 0, frameIndex = 0, playing = false, timer = null, pointScale = 1;

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight); renderer.setClearColor(0x17191b); document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, .001, 100);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AxesHelper(.2));

function resetCamera() {
  const c=DATA[clipIndex];
  if (!c || !c.train || !c.train.N) { camera.position.set(1.25,.95,1.15); controls.target.set(0,.45,.25); controls.update(); return; }
  const values=c.train.flows, count=c.train.N;
  let min=[Infinity,Infinity,Infinity], max=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<count;i++) for(let d=0;d<3;d++){const value=values[i*3+d];min[d]=Math.min(min[d],value);max[d]=Math.max(max[d],value);}
  const center=new THREE.Vector3((min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2);
  const diagonal=new THREE.Vector3(max[0]-min[0],max[1]-min[1],max[2]-min[2]).length();
  const distance=Math.max(.45,diagonal/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*1.25);
  const direction=new THREE.Vector3(1,.7,1).normalize();
  camera.position.copy(center).addScaledVector(direction,distance); controls.target.copy(center);
  camera.near=Math.max(.001,distance/1000); camera.far=Math.max(100,distance*20); camera.updateProjectionMatrix(); controls.update();
}

function decode(b64, Type) {
  const binary = atob(b64), bytes = new Uint8Array(binary.length);
  for (let i=0; i<binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Type(bytes.buffer);
}
function decodeCloud(payload, T) {
  const flows = decode(payload.flows_b64, Float32Array), colors = decode(payload.colors_b64, Uint8Array);
  const expected = T * payload.N * 3;
  if (flows.length !== expected || colors.length !== expected) throw new Error(`点云长度不匹配: N=${payload.N}`);
  return {N:payload.N, flows, colors};
}
function makePoints(size, opacity, depthTest=true) {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({size, vertexColors:true, sizeAttenuation:true, transparent:opacity<1, opacity, depthTest, depthWrite:depthTest});
  const points = new THREE.Points(geometry, material); scene.add(points); return points;
}
function unpackClip(c) {
  c.raw = c.cameras.map((payload, i) => ({...decodeCloud(payload,c.T), prefix:payload.prefix, validCount:payload.valid_count, source:i, points:makePoints(.004,.72,true)}));
  c.train = {...decodeCloud(c.training,c.T), cameraIds:decode(c.training.camera_ids_b64,Uint8Array), points:makePoints(.009,1,false)};
  if (c.train.cameraIds.length !== c.train.N) throw new Error("训练点来源长度不匹配");
  c.robotCloud = {...decodeCloud(c.robot,c.T), points:makePoints(.011,1,true)};
  c.pose = decode(c.gripper_pose_b64,Float32Array); c.open = decode(c.gripper_open_b64,Float32Array);
  c.workspaceHelper=null;
  if(c.workspace_bounds){
    const lo=c.workspace_bounds[0], hi=c.workspace_bounds[1];
    const box=new THREE.Box3(new THREE.Vector3(...lo),new THREE.Vector3(...hi));
    c.workspaceHelper=new THREE.Box3Helper(box,0x9ccc65); c.workspaceHelper.material.transparent=true; c.workspaceHelper.material.opacity=.8; scene.add(c.workspaceHelper);
  }
}

function setFrame(cloud, frame, sourceIds, uniformSource) {
  const base = frame * cloud.N * 3, positions = cloud.flows.subarray(base, base + cloud.N*3);
  const rgb = document.getElementById("color-mode").value === "rgb";
  let colors = new Float32Array(cloud.N*3);
  for (let i=0; i<cloud.N; i++) {
    const sid = sourceIds ? sourceIds[i] : uniformSource;
    for (let d=0; d<3; d++) colors[i*3+d] = rgb ? cloud.colors[base+i*3+d]/255 : SOURCE_COLORS[sid % SOURCE_COLORS.length][d];
  }
  cloud.points.geometry.setAttribute("position",new THREE.BufferAttribute(positions,3));
  cloud.points.geometry.setAttribute("color",new THREE.BufferAttribute(colors,3));
  cloud.points.geometry.computeBoundingSphere();
}
function cameraEnabled(i) { const input=document.getElementById("cam"+i); return input ? input.checked : true; }
function esc(value) { return String(value).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[ch]); }

function update() {
  if (!DATA.length) return;
  const c=DATA[clipIndex], frame=Math.max(0,Math.min(c.T-1,frameIndex)); frameIndex=frame;
  const showRaw=document.getElementById("raw").checked, showTrain=document.getElementById("training").checked;
  c.raw.forEach((cloud,i)=>{ setFrame(cloud,frame,null,i); cloud.points.visible=showRaw&&cameraEnabled(i); });
  setFrame(c.train,frame,c.train.cameraIds,0); c.train.points.visible=showTrain;
  setFrame(c.robotCloud,frame,null,0); c.robotCloud.points.visible=document.getElementById("robot").checked;
  if(c.workspaceHelper)c.workspaceHelper.visible=document.getElementById("workspace").checked;
  applyPointSize(c);
  document.getElementById("frame").max=c.T-1; document.getElementById("frame").value=frame;
  document.getElementById("frame-count").textContent=`${frame} / ${c.T-1}`; document.getElementById("clip-count").textContent=`${clipIndex+1} / ${DATA.length}`;
  const perSource=SOURCE_COLORS.map((_,i)=>Array.from(c.train.cameraIds).filter(x=>x===i).length);
  const rawCounts=c.raw.map(x=>x.N).join(" / ");
  const crop=c.workspace_bounds?`<br>工作区: [${c.workspace_bounds[0].join(", ")}] → [${c.workspace_bounds[1].join(", ")}]<br>导出裁剪: ${c.workspace_points_before.join(" / ")} → <strong>${c.workspace_points_after.join(" / ")}</strong>`:"";
  const sphere=c.pipeline.sphere_crop==="skipped_workspace"?" · sphere crop 已跳过":"";
  document.getElementById("status").innerHTML=`<strong>${esc(c.name)}</strong> · ${esc(c.key)}<br>相机: ${c.camera_names.map((name,i)=>`${i}:${esc(name)}`).join(" · ")}<br>相机有效显示点: ${rawCounts}<br>实际采样训练点: <strong>${c.train.N}</strong>（来源 ${perSource.join(" / ")}）${crop}<br>${c.pipeline.mode} · ${c.pipeline.num_cameras} 相机 · voxel ${c.pipeline.grid_size} m · cap ${c.pipeline.max_scene_points} · seed ${c.pipeline.seed}${sphere}`;
}
function applyPointSize(c=DATA[clipIndex]) {
  c.raw.forEach(cloud=>cloud.points.material.size=.004*pointScale);
  c.train.points.material.size=.009*pointScale;
  c.robotCloud.points.material.size=.011*pointScale;
}
function changeClip(delta) {
  DATA[clipIndex].raw.forEach(x=>x.points.visible=false); DATA[clipIndex].train.points.visible=false; DATA[clipIndex].robotCloud.points.visible=false;if(DATA[clipIndex].workspaceHelper)DATA[clipIndex].workspaceHelper.visible=false;
  clipIndex=(clipIndex+delta+DATA.length)%DATA.length; frameIndex=0; resetCamera(); update();
}
function changeFrame(delta) { const T=DATA[clipIndex].T; frameIndex=(frameIndex+delta+T)%T; update(); }

document.getElementById("prev-clip").onclick=()=>changeClip(-1); document.getElementById("next-clip").onclick=()=>changeClip(1);
document.getElementById("prev-frame").onclick=()=>changeFrame(-1); document.getElementById("next-frame").onclick=()=>changeFrame(1);
document.getElementById("frame").oninput=e=>{frameIndex=Number(e.target.value);update();};
document.getElementById("color-mode").onchange=update;
["raw","training","robot","workspace","cam0","cam1","cam2"].forEach(id=>document.getElementById(id).onchange=update);
// Point size only changes material size.  In particular, it does not rebuild
// color attributes, so dragging this slider cannot alter point colors.
document.getElementById("point-size").oninput=e=>{pointScale=Number(e.target.value);applyPointSize();}; document.getElementById("reset").onclick=resetCamera;
document.getElementById("play").onclick=e=>{playing=!playing;e.currentTarget.textContent=playing?"Ⅱ":"▶";e.currentTarget.classList.toggle("active",playing);if(playing)timer=setInterval(()=>changeFrame(1),250);else{clearInterval(timer);timer=null;}};
addEventListener("keydown",e=>{if(e.key==="ArrowLeft")changeClip(-1);if(e.key==="ArrowRight")changeClip(1);if(e.key==="ArrowUp")changeFrame(-1);if(e.key==="ArrowDown")changeFrame(1);if(e.key===" "){e.preventDefault();document.getElementById("play").click();}});
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});

try { DATA.forEach(unpackClip); for(let i=0;i<3;i++){const input=document.getElementById("cam"+i);input.disabled=!DATA.some(c=>c.raw.length>i);} resetCamera(); update(); }
catch(error) { document.getElementById("status").textContent="加载失败: "+error.message; throw error; }
(function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);})();
</script>
</body>
</html>
"""


def _script_tag(local_path: Path, cdn_url: str) -> str:
    if local_path.is_file():
        return f"<script>{local_path.read_text()}</script>"
    return f'<script src="{cdn_url}"></script>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", required=True, help="One or more exported LIBERO .npz files.")
    parser.add_argument("--output", required=True, help="Output HTML path (overwritten).")
    parser.add_argument("--num_cameras", type=int, default=3, help="First N cameras used by the training preview (default: 3).")
    parser.add_argument("--grid_size", type=float, default=0.015, help="Training voxel size in metres (default: 0.015).")
    parser.add_argument("--max_scene_points", type=int, default=12000, help="Training scene point cap (default: 12000).")
    parser.add_argument("--pipeline_mode", choices=("train", "test"), default="train", help="Point-selection pipeline to preview (default: train).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the reproducible training preview.")
    parser.add_argument("--max_cam_points", type=int, default=0, help="Display cap per raw camera; 0 keeps all valid camera points.")
    args = parser.parse_args()
    if args.num_cameras < 1 or args.grid_size <= 0 or args.max_scene_points < 1 or args.max_cam_points < 0:
        parser.error("camera count, grid size and point caps must be positive")

    clips = [
        _serialize_clip(_clip_points(
            Path(path),
            max_cam_points=args.max_cam_points,
            num_cameras=args.num_cameras,
            grid_size=args.grid_size,
            max_scene_points=args.max_scene_points,
            pipeline_mode=args.pipeline_mode,
            seed=args.seed,
        ))
        for path in args.clips
    ]
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(clips, ensure_ascii=True))
    html = html.replace("__THREE_SCRIPT__", _script_tag(Path("/tmp/three.min.js"), "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"))
    html = html.replace("__ORBIT_SCRIPT__", _script_tag(Path("/tmp/OrbitControls.js"), "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"[viz_html] wrote {output} ({output.stat().st_size // 1024} KiB, {len(clips)} clip(s))")


if __name__ == "__main__":
    main()
