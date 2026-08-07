"""Build an offline GT/baseline/finetuned LIBERO comparison viewer."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def _b64(value: np.ndarray, dtype: np.dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(value, dtype=dtype).tobytes()).decode("ascii")


def _payload(record: dict, max_points: int, seed: int) -> dict:
    baseline = np.load(record["baseline"]["prediction"], allow_pickle=True)
    finetuned = np.load(record["finetuned"]["prediction"], allow_pickle=True)
    baseline_gt = baseline["gt_scene_flows"].astype(np.float32)
    finetuned_gt = finetuned["gt_scene_flows"].astype(np.float32)
    if baseline_gt.shape != finetuned_gt.shape or not np.allclose(
        baseline_gt, finetuned_gt, atol=1e-5
    ):
        raise ValueError(f"GT point ordering differs between checkpoints: {record['clip']}")
    baseline_shift = baseline["shift_amount"].astype(np.float32).reshape(1, 1, 3)
    finetuned_shift = finetuned["shift_amount"].astype(np.float32).reshape(1, 1, 3)
    gt = finetuned_gt - finetuned_shift
    baseline_pred = baseline["pred_scene_flows"].astype(np.float32) - baseline_shift
    finetuned_pred = finetuned["pred_scene_flows"].astype(np.float32) - finetuned_shift
    colors = finetuned["scene_colors"].astype(np.uint8)
    moved = finetuned["moved_mask"].astype(bool)

    moved_idx = np.flatnonzero(moved)
    static_idx = np.flatnonzero(~moved)
    rng = np.random.default_rng(seed)
    keep_moved = moved_idx
    if keep_moved.size > max_points:
        keep_moved = rng.choice(keep_moved, max_points, replace=False)
    remaining = max_points - keep_moved.size
    keep_static = (
        rng.choice(static_idx, min(remaining, static_idx.size), replace=False)
        if remaining > 0 else np.empty((0,), dtype=np.int64)
    )
    indices = np.sort(np.concatenate([keep_moved, keep_static]))
    gt = gt[:, indices]
    baseline_pred = baseline_pred[:, indices]
    finetuned_pred = finetuned_pred[:, indices]
    colors = colors[:, indices]
    moved = moved[indices]
    baseline_epe = np.linalg.norm(baseline_pred - gt, axis=-1).astype(np.float32)
    finetuned_epe = np.linalg.norm(finetuned_pred - gt, axis=-1).astype(np.float32)
    movement = np.linalg.norm(gt - gt[:1], axis=-1).astype(np.float32)

    with np.load(record["clip"], allow_pickle=True) as raw:
        robot = np.asarray(raw["robot_flows"], dtype=np.float32)
        bounds = (
            np.asarray(raw["workspace_bounds"], dtype=np.float32).tolist()
            if "workspace_bounds" in raw else None
        )
    b_metrics = record["baseline"]["metrics"]
    f_metrics = record["finetuned"]["metrics"]
    moved_improvement = None
    if b_metrics["epe_moved_m"] and f_metrics["epe_moved_m"] is not None:
        moved_improvement = 1.0 - f_metrics["epe_moved_m"] / b_metrics["epe_moved_m"]
    return {
        "name": Path(record["clip"]).stem,
        "task": record["task"],
        "clip": record["clip"],
        "category": record["category"],
        "T": int(gt.shape[0]),
        "N": int(gt.shape[1]),
        "Nr": int(robot.shape[1]),
        "gt": _b64(gt, np.float32),
        "baseline": _b64(baseline_pred, np.float32),
        "finetuned": _b64(finetuned_pred, np.float32),
        "colors": _b64(colors, np.uint8),
        "moved": _b64(moved, np.uint8),
        "baseline_epe": _b64(baseline_epe, np.float32),
        "finetuned_epe": _b64(finetuned_epe, np.float32),
        "movement": _b64(movement, np.float32),
        "robot": _b64(robot, np.float32),
        "workspace_bounds": bounds,
        "baseline_metrics": b_metrics,
        "finetuned_metrics": f_metrics,
        "moved_improvement": moved_improvement,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO Spatial · PointWorld Evaluation</title>
<style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#111315;color:#e8ecef;font:14px system-ui,sans-serif;letter-spacing:0}
#canvas{position:fixed;inset:0}.top{position:fixed;left:16px;right:16px;top:14px;height:48px;display:flex;align-items:center;gap:10px;background:#191c1fdd;border:1px solid #343a3f;padding:7px 10px;z-index:2;border-radius:6px;backdrop-filter:blur(8px)}
button,select,input{font:inherit;color:inherit}button,select{height:32px;border:1px solid #42494f;background:#252a2e;border-radius:4px;padding:0 10px}button{cursor:pointer}button:hover{background:#30363b}.icon{width:34px;padding:0;font-size:18px}.title{min-width:0;flex:1}.title strong{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.title small{color:#9da7ae}.segments{display:flex}.segments button{border-radius:0;border-right:0}.segments button:first-child{border-radius:4px 0 0 4px}.segments button:last-child{border-radius:0 4px 4px 0;border-right:1px solid #42494f}.segments button.active{background:#dbe6eb;color:#111315}
.panel{position:fixed;right:16px;top:76px;width:310px;background:#191c1fee;border:1px solid #343a3f;border-radius:6px;padding:14px;z-index:2;backdrop-filter:blur(8px)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}.row label{color:#9da7ae;font-size:12px}.row select{width:100%;margin-top:4px}.toggles{display:flex;gap:14px;margin:10px 0 14px;color:#c6cdd2}.metric-title{font-size:12px;color:#9da7ae;margin:10px 0 5px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}.metric{background:#24292d;padding:8px;border-radius:4px}.metric b{display:block;font-size:17px}.metric span{font-size:11px;color:#9da7ae}.improve{margin-top:8px;padding:9px;background:#17362c;color:#8ce0bd;border-radius:4px}.status{margin-top:10px;color:#aeb7bd;font-size:12px;line-height:1.45}.framebar{position:fixed;left:18px;right:350px;bottom:18px;height:44px;display:flex;align-items:center;gap:10px;background:#191c1fee;border:1px solid #343a3f;border-radius:6px;padding:7px 10px;z-index:2}.framebar input{flex:1}.legend{display:flex;gap:12px;font-size:12px;color:#b7c0c6}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
.slider-row{display:grid;grid-template-columns:82px 1fr 38px;align-items:center;gap:8px;margin:4px 0 12px;color:#9da7ae;font-size:12px}.slider-row input{width:100%}.slider-row output{text-align:right;color:#dbe2e6}
@media(max-width:850px){.top{right:10px;left:10px;height:auto;flex-wrap:wrap}.title{order:-1;flex-basis:100%}.panel{right:10px;top:112px;width:280px}.framebar{right:10px;left:10px;bottom:10px}.segments button{padding:0 6px}}
</style></head><body><canvas id="canvas"></canvas>
<div class="top"><button class="icon" id="prev" title="Previous clip">‹</button><button class="icon" id="next" title="Next clip">›</button><div class="title"><strong id="clip-title"></strong><small id="clip-subtitle"></small></div><div class="segments" id="modes"><button data-mode="gt">GT</button><button data-mode="baseline">Baseline</button><button data-mode="finetuned" class="active">Finetuned</button><button data-mode="overlay">Overlay</button></div></div>
<div class="panel"><div class="row"><label>Point subset<select id="subset"><option value="all">All points</option><option value="moved">Moved points</option><option value="static">Static points</option></select></label><label>Color mode<select id="color"><option value="rgb">RGB</option><option value="epe">EPE</option><option value="motion">GT motion</option></select></label></div><div class="slider-row"><label for="point-size">Point size</label><input id="point-size" type="range" min="0.5" max="3" value="1" step="0.1"><output id="point-size-value">1.0×</output></div><div class="toggles"><label><input id="robot" type="checkbox" checked> Robot</label><label><input id="workspace" type="checkbox"> Box</label><label><input id="trails" type="checkbox" checked> Trails</label></div><div class="metric-title">Mean EPE · all predicted frames</div><div class="metrics"><div class="metric"><b id="b-all"></b><span>Baseline · all</span></div><div class="metric"><b id="f-all"></b><span>Finetuned · all</span></div><div class="metric"><b id="b-moved"></b><span>Baseline · moved</span></div><div class="metric"><b id="f-moved"></b><span>Finetuned · moved</span></div></div><div class="improve" id="improve"></div><div class="status" id="status"></div><div class="legend"><span><i class="dot" style="background:#45b7e8"></i>GT</span><span><i class="dot" style="background:#f3a44a"></i>Baseline</span><span><i class="dot" style="background:#55d59a"></i>Finetuned</span></div></div>
<div class="framebar"><button class="icon" id="play">▶</button><span id="frame-label"></span><input id="frame" type="range" min="0" max="10" value="10" step="1"></div>
__THREE____ORBIT__<script>
const DATA=__DATA__;let clipIndex=0,frame=10,mode="finetuned",playing=false,timer=null,boxHelper=null,pointScale=1;
const canvas=document.getElementById("canvas"),renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);renderer.setClearColor(0x111315);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(48,innerWidth/innerHeight,.001,100),controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;scene.add(new THREE.AmbientLight(0xffffff,.8));
function decode(s,Type){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Type(u.buffer)}
for(const c of DATA){for(const k of ["gt","baseline","finetuned","baseline_epe","finetuned_epe","movement","robot"])c[k]=decode(c[k],Float32Array);c.colors=decode(c.colors,Uint8Array);c.moved=decode(c.moved,Uint8Array)}
function points(size,opacity=1){const g=new THREE.BufferGeometry(),m=new THREE.PointsMaterial({size,vertexColors:true,transparent:opacity<1,opacity,depthWrite:opacity===1});const p=new THREE.Points(g,m);scene.add(p);return p}const clouds={gt:points(.008,.65),baseline:points(.009,.85),finetuned:points(.009,1)},robotCloud=points(.012,1);
function lines(color){const g=new THREE.BufferGeometry(),m=new THREE.LineBasicMaterial({color,transparent:true,opacity:.85});const x=new THREE.LineSegments(g,m);scene.add(x);return x}const trail={gt:lines(0x45b7e8),baseline:lines(0xf3a44a),finetuned:lines(0x55d59a)};
function applyPointSize(){clouds.gt.material.size=.008*pointScale;clouds.baseline.material.size=.009*pointScale;clouds.finetuned.material.size=.009*pointScale;robotCloud.material.size=.012*pointScale;document.getElementById("point-size-value").textContent=pointScale.toFixed(1)+"×"}
function turbo(x){x=Math.max(0,Math.min(1,x));const r=Math.max(0,Math.min(1,1.5-Math.abs(4*x-3))),g=Math.max(0,Math.min(1,1.5-Math.abs(4*x-2))),b=Math.max(0,Math.min(1,1.5-Math.abs(4*x-1)));return [r,g,b]}
function chosen(c){const subset=document.getElementById("subset").value,out=[];for(let i=0;i<c.N;i++)if(subset==="all"||(subset==="moved"&&c.moved[i])||(subset==="static"&&!c.moved[i]))out.push(i);return out}
function setCloud(handle,key,c,idx,fixed){const pos=new Float32Array(idx.length*3),col=new Float32Array(idx.length*3),base=frame*c.N*3,colorMode=document.getElementById("color").value;for(let j=0;j<idx.length;j++){const i=idx[j],s=base+i*3;pos[j*3]=c[key][s];pos[j*3+1]=c[key][s+1];pos[j*3+2]=c[key][s+2];let rgb;if(fixed)rgb=fixed;else if(colorMode==="rgb")rgb=[c.colors[s]/255,c.colors[s+1]/255,c.colors[s+2]/255];else{const values=colorMode==="epe"?(key==="baseline"?c.baseline_epe:c.finetuned_epe):c.movement;const max=colorMode==="epe"?.03:.08;rgb=turbo(values[frame*c.N+i]/max)}col.set(rgb,j*3)}handle.geometry.setAttribute("position",new THREE.BufferAttribute(pos,3));handle.geometry.setAttribute("color",new THREE.BufferAttribute(col,3));handle.visible=true}
function setTrail(handle,key,c,idx){const focus=idx.filter(i=>c.moved[i]).slice(0,220),segments=new Float32Array(focus.length*Math.max(frame,0)*6);let o=0;for(const i of focus)for(let t=0;t<frame;t++){for(let d=0;d<3;d++)segments[o++]=c[key][(t*c.N+i)*3+d];for(let d=0;d<3;d++)segments[o++]=c[key][((t+1)*c.N+i)*3+d]}handle.geometry.setAttribute("position",new THREE.BufferAttribute(segments,3));handle.visible=document.getElementById("trails").checked&&segments.length>0}
function mm(v){return v==null?"n/a":(v*1000).toFixed(2)+" mm"}function resetCamera(c){const p=c.gt,box=new THREE.Box3();for(let i=0;i<c.N;i++)box.expandByPoint(new THREE.Vector3(p[i*3],p[i*3+1],p[i*3+2]));const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()).length();camera.position.set(center.x+size*.8,center.y-size*.9,center.z+size*.65);controls.target.copy(center);camera.near=Math.max(.001,size/1000);camera.far=Math.max(20,size*20);camera.updateProjectionMatrix();controls.update()}
function update(){const c=DATA[clipIndex],idx=chosen(c);frame=Math.min(frame,c.T-1);for(const x of Object.values(clouds))x.visible=false;for(const x of Object.values(trail))x.visible=false;if(mode==="overlay"){setCloud(clouds.gt,"gt",c,idx,[.27,.72,.91]);setCloud(clouds.baseline,"baseline",c,idx,[.95,.64,.29]);setCloud(clouds.finetuned,"finetuned",c,idx,[.33,.84,.60]);setTrail(trail.gt,"gt",c,idx);setTrail(trail.baseline,"baseline",c,idx);setTrail(trail.finetuned,"finetuned",c,idx)}else{setCloud(clouds[mode],mode,c,idx,null);setTrail(trail[mode],mode,c,idx);if(mode!=="gt")setTrail(trail.gt,"gt",c,idx)}const rp=new Float32Array(c.Nr*3),rc=new Float32Array(c.Nr*3);for(let i=0;i<c.Nr*3;i++){rp[i]=c.robot[frame*c.Nr*3+i];rc[i]=i%3===0?.92:i%3===1?.25:.28}robotCloud.geometry.setAttribute("position",new THREE.BufferAttribute(rp,3));robotCloud.geometry.setAttribute("color",new THREE.BufferAttribute(rc,3));robotCloud.visible=document.getElementById("robot").checked;if(boxHelper)scene.remove(boxHelper);boxHelper=null;if(c.workspace_bounds){const lo=c.workspace_bounds[0],hi=c.workspace_bounds[1];boxHelper=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...lo),new THREE.Vector3(...hi)),0x9ccc65);scene.add(boxHelper);boxHelper.visible=document.getElementById("workspace").checked}document.getElementById("clip-title").textContent=c.task;document.getElementById("clip-subtitle").textContent=`${clipIndex+1}/${DATA.length} · ${c.category} · ${c.name}`;document.getElementById("frame").max=c.T-1;document.getElementById("frame").value=frame;document.getElementById("frame-label").textContent=`t=${frame}/${c.T-1}`;document.getElementById("b-all").textContent=mm(c.baseline_metrics.epe_all_m);document.getElementById("f-all").textContent=mm(c.finetuned_metrics.epe_all_m);document.getElementById("b-moved").textContent=mm(c.baseline_metrics.epe_moved_m);document.getElementById("f-moved").textContent=mm(c.finetuned_metrics.epe_moved_m);document.getElementById("improve").textContent=c.moved_improvement==null?"Moved-point improvement: n/a":`Moved-point improvement: ${(c.moved_improvement*100).toFixed(1)}%`;document.getElementById("status").textContent=`display ${idx.length}/${c.N} points · moved ${c.finetuned_metrics.n_moved_points}/${c.finetuned_metrics.n_points}`}
document.getElementById("modes").onclick=e=>{if(!e.target.dataset.mode)return;mode=e.target.dataset.mode;document.querySelectorAll("#modes button").forEach(x=>x.classList.toggle("active",x.dataset.mode===mode));update()};for(const id of ["subset","color","robot","workspace","trails"])document.getElementById(id).onchange=update;document.getElementById("point-size").oninput=e=>{pointScale=+e.target.value;applyPointSize()};document.getElementById("frame").oninput=e=>{frame=+e.target.value;update()};function change(d){clipIndex=(clipIndex+d+DATA.length)%DATA.length;frame=DATA[clipIndex].T-1;resetCamera(DATA[clipIndex]);update()}document.getElementById("prev").onclick=()=>change(-1);document.getElementById("next").onclick=()=>change(1);document.getElementById("play").onclick=e=>{playing=!playing;e.target.textContent=playing?"Ⅱ":"▶";if(playing)timer=setInterval(()=>{frame=(frame+1)%DATA[clipIndex].T;update()},250);else{clearInterval(timer);timer=null}};addEventListener("keydown",e=>{if(e.key==="ArrowLeft")change(-1);if(e.key==="ArrowRight")change(1)});addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});applyPointSize();resetCamera(DATA[0]);update();(function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)})();
</script></body></html>"""


def _script(local: Path, url: str) -> str:
    return f"<script>{local.read_text()}</script>" if local.is_file() else f'<script src="{url}"></script>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads(Path(args.comparison_manifest).read_text())
    data = [_payload(record, args.max_points, args.seed + i) for i, record in enumerate(manifest["selected"])]
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=True))
    html = html.replace("__THREE__", _script(Path("/tmp/three.min.js"), "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"))
    html = html.replace("__ORBIT__", _script(Path("/tmp/OrbitControls.js"), "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"[eval_viz_html] wrote {output} ({output.stat().st_size // 1024} KiB, {len(data)} clips)")


if __name__ == "__main__":
    main()
