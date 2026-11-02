"""Offline, multi-video field setup. No HTTP server or background process."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import webbrowser
from pathlib import Path
import cv2
import numpy as np
from .media import grab_frame

_PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Scout - pick the field</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#15171a;color:#e8eaed}
 header{padding:12px 18px;background:#1e2126;border-bottom:1px solid #2c3036;
        position:sticky;top:0;z-index:5}
 h1{margin:0 0 4px;font-size:16px;font-weight:600}
 p{margin:0;color:#9aa4b2;font-size:13px}
 #wrap{padding:16px 18px;position:relative}
 canvas{border-radius:6px;cursor:crosshair;display:block;max-width:100%}
 .bar{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
 button{font:inherit;padding:7px 14px;border-radius:6px;border:1px solid #3a4048;
        background:#262b31;color:#e8eaed;cursor:pointer}
 button:hover{background:#2f353c}
 button.primary{background:#2f7d4f;border-color:#2f7d4f}
 button.primary:hover{background:#38945d}
 button:disabled{opacity:.45;cursor:not-allowed}
 code{background:#0e1013;padding:2px 6px;border-radius:4px;color:#8fd3a4}
 #out{margin-top:12px;white-space:pre-wrap;font-family:ui-monospace,monospace;
      font-size:12px;color:#8fd3a4;background:#0e1013;padding:10px;border-radius:6px}
 #msg{margin-left:auto;color:#9aa4b2}
</style>
<header>
  <h1>Click the corners of the field you want to cut</h1>
  <p>Click to add points &middot; drag a point to move it &middot; right-click a point to delete
     &middot; <code>u</code> undo &middot; at least 3 points, select the next video, then <b>Download match setup</b>.
     Include the whole area players run in, plus a little margin.</p>
</header>
<div id="wrap">
  <canvas id="c"></canvas>
  <div class="bar">
    <button class="primary" id="save" disabled>Download match setup</button>
    <button id="undo">Undo</button>
    <button id="clear">Clear</button>
    <button id="rect">Suggest rectangle</button>
    <span id="msg"></span>
  </div>
  <div id="out">no points yet</div>
</div>
<script>
const videos = __VIDEOS__;
let current = 0;
const selection = document.createElement('select');
document.querySelector('header').append(selection);
videos.forEach((v, i) => { const o = new Option(v.name, i); selection.add(o); });
selection.onchange = () => { remember(); current = +selection.value; img.src = videos[current].image; };
function remember() { videos[current].polygon = pts.map(p => [p[0]/W, p[1]/H]); }
const IMG_SRC = videos[0].image;
const c = document.getElementById('c'), ctx = c.getContext('2d');
const img = new Image();
let pts = [], drag = -1, W = 0, H = 0;

img.onload = () => {
  const maxw = Math.min(1680, window.innerWidth - 40);
  W = maxw; H = Math.round(maxw * img.height / img.width);
  c.width = W; c.height = H;
  pts = (videos[current].polygon || []).map(p => [p[0]*W,p[1]*H]);
  draw();
};
img.src = IMG_SRC;

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.drawImage(img,0,0,W,H);
  if(pts.length){
    ctx.beginPath();
    pts.forEach((p,i)=> i? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]));
    if(pts.length>2) ctx.closePath();
    ctx.fillStyle='rgba(60,200,120,.22)';
    ctx.strokeStyle='#3ddc84'; ctx.lineWidth=2;
    if(pts.length>2) ctx.fill();
    ctx.stroke();
    pts.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],6,0,7);
      ctx.fillStyle = i===drag ? '#ffd166' : '#3ddc84'; ctx.fill();
      ctx.strokeStyle='#0b0d10'; ctx.lineWidth=2; ctx.stroke();
      ctx.fillStyle='#0b0d10'; ctx.font='bold 11px system-ui';
      ctx.fillText(i+1, p[0]-3, p[1]+4);
    });
  }
  const norm = pts.map(p=>[+(p[0]/W).toFixed(4), +(p[1]/H).toFixed(4)]);
  document.getElementById('out').textContent =
    pts.length ? '"field_polygon": ' + JSON.stringify(norm) : 'no points yet';
  document.getElementById('save').disabled = pts.length < 3;
}
function at(e){
  const r = c.getBoundingClientRect();
  const sx = c.width / r.width, sy = c.height / r.height;
  return [Math.max(0,Math.min(W,(e.clientX-r.left)*sx)), Math.max(0,Math.min(H,(e.clientY-r.top)*sy))];
}
function near(p){
  for(let i=0;i<pts.length;i++){
    const dx=pts[i][0]-p[0], dy=pts[i][1]-p[1];
    if(dx*dx+dy*dy < 144) return i;
  }
  return -1;
}
c.addEventListener('mousedown', e=>{
  if(e.button!==0) return;
  const p = at(e), i = near(p);
  if(i>=0){ drag=i; } else { pts.push(p); }
  draw();
});
c.addEventListener('mousemove', e=>{ if(drag>=0){ pts[drag]=at(e); draw(); }});
window.addEventListener('mouseup', ()=>{ drag=-1; draw(); });
c.addEventListener('contextmenu', e=>{
  e.preventDefault();
  const i = near(at(e));
  if(i>=0){ pts.splice(i,1); draw(); }
});
document.addEventListener('keydown', e=>{
  if(e.key==='u'){ pts.pop(); draw(); }
});
document.getElementById('undo').onclick = ()=>{ pts.pop(); draw(); };
document.getElementById('clear').onclick = ()=>{ pts=[]; draw(); };
document.getElementById('rect').onclick = ()=>{
  pts = [[W*0.02,H*0.34],[W*0.98,H*0.30],[W*0.98,H*0.62],[W*0.02,H*0.68]];
  draw();
};
document.getElementById('save').onclick = async ()=>{
  remember();
  const missing = videos.filter(v => !v.polygon || v.polygon.length < 3);
  if (missing.length) { document.getElementById('msg').textContent = 'Still need fields: ' + missing.map(v => v.name).join(', '); return; }
  const data = {version: 1, videos: videos.map(({image, ...v}) => v)};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'setup.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  document.getElementById('msg').textContent = 'Downloaded setup.json. Import it using scout pick --import-setup.';
};
</script>
"""


def _frame_data_uri(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("could not encode reference frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def field_path(cfg, clip):
    identity = hashlib.sha256(str(clip.path.resolve()).encode()).hexdigest()[:16]
    return cfg.source_path(cfg.fields_dir, clip.path) / f"{clip.path.stem}-{identity}.json"


def validate_polygon(value):
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("a field needs at least three points")
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("field points must be [x, y]")
        x, y = map(float, point)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError("field coordinates must be finite and between 0 and 1")
        points.append([x, y])
    area = sum(a[0]*b[1] - b[0]*a[1] for a,b in zip(points, points[1:]+points[:1]))
    if abs(area) < 1e-6:
        raise ValueError("field polygon has no area")
    return points


def export_setup(cfg, clips, at=None, open_browser=True):
    from .cli import _region_for
    videos = []
    for clip in clips:
        t = at if at is not None else min(60., clip.info.duration / 2)
        if not 0 <= t < clip.info.duration:
            raise ValueError(f"reference time outside {clip.path.name}")
        region = _region_for(cfg, clips, clip)
        videos.append(dict(name=str(clip.path.relative_to(Path(cfg.media_dir))),
                           source=str(clip.path.resolve()), time=t,
                           polygon=region["polygon"] if region else [],
                           image=_frame_data_uri(grab_frame(clip.path, t))))
    dst = cfg.path(cfg.out_dir) / "field-setup.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_PAGE.replace("__VIDEOS__", json.dumps(videos).replace("<", "\\u003c")), encoding="utf-8")
    if open_browser:
        webbrowser.open(dst.resolve().as_uri())
    return dst


def import_setup(cfg, clips, path):
    data = json.loads(path.read_text())
    if data.get("version") != 1:
        raise ValueError("unsupported setup version")
    expected = {str(c.path.resolve()): c for c in clips}
    rows = data.get("videos", [])
    if len(rows) != len(expected) or {r["source"] for r in rows} != set(expected):
        raise ValueError("setup videos do not match the selected match")
    pending = []
    for row in rows:
        c = expected[row["source"]]
        poly = validate_polygon(row["polygon"])
        t = float(row["time"])
        if not math.isfinite(t) or not 0 <= t < c.info.duration:
            raise ValueError(f"invalid reference time for {c.path}")
        pending.append((field_path(cfg, c), dict(field_polygon=poly,
                        field_ref_file=str(c.path.resolve()), field_ref_time=t)))
    for dst, content in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_suffix(".tmp")
        temp.write_text(json.dumps(content, indent=2))
        temp.replace(dst)
    print(f"  Saved fields for {len(pending)} videos")
