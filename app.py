"""
app.py — API server cho mô phỏng gió 3D đơn giản hoá.

3 endpoint:
  POST /simulate   — nhận file STL + tham số gió, khởi chạy job nền, trả job_id
  GET  /status/<id> — kiểm tra tiến trình job
  GET  /result/<id> — lấy kết quả (JSON đường dòng) khi job xong

Chạy job trong 1 luồng nền (threading) — đủ cho quy mô nhỏ 1 instance,
CHƯA phù hợp nếu nhiều người chạy cùng lúc (cần Celery/Redis nếu mở rộng).
Kết quả lưu tạm trong bộ nhớ (mất khi server restart) — đủ dùng cho MVP.
"""

import io
import uuid
import threading
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from solver import parse_binary_stl, build_solid_grid, run_stable_fluids_3d, trace_streamlines

app = Flask(__name__)
CORS(app)  # cho phép gọi từ GitHub Pages / bất kỳ origin nào (siết lại sau nếu cần)

jobs = {}  # job_id -> { status, progress, result, error }


def run_simulation_job(job_id, stl_bytes, wind_dir, speed, resolution, iterations, seed_grid):
    try:
        jobs[job_id]["status"] = "reading_stl"
        triangles = parse_binary_stl(stl_bytes)
        if len(triangles) == 0:
            raise ValueError("File STL rỗng hoặc không đọc được tam giác nào.")

        mins = triangles.reshape(-1, 3).min(axis=0)
        maxs = triangles.reshape(-1, 3).max(axis=0)
        size = maxs - mins
        print(f"[wind] bbox hình học: size={size.tolist()}, tam giác={len(triangles)}")

        # Vùng khảo sát: rộng hơn vật thể vừa đủ để gió có chỗ tăng tốc/tách
        # dòng — không nới quá tay kẻo vật thể chỉ chiếm vài ô trên lưới thô,
        # khiến voxel hoá bỏ sót hoàn toàn (chính là lỗi "gió xuyên qua vật").
        pad_up = size[0] * 1.0 + 1.0     # phía đón gió
        pad_down = size[0] * 2.0 + 1.0   # phía sau, chừa chỗ cho vệt gió khuất
        pad_side = size[1:].max() * 0.5 + 1.0 if size[1:].max() > 0 else 2.0

        if wind_dir[0] >= 0:
            bx0, bx1 = mins[0] - pad_up, maxs[0] + pad_down
        else:
            bx0, bx1 = mins[0] - pad_down, maxs[0] + pad_up
        bounds = (
            bx0, bx1,
            mins[1] - pad_side, maxs[1] + pad_side,
            mins[2] - pad_side, maxs[2] + pad_side,
        )

        jobs[job_id]["status"] = "voxelizing"
        solid = build_solid_grid(triangles, bounds, resolution)
        solid_count = int(solid.sum())
        print(f"[wind] voxel hoá xong: {solid_count}/{solid.size} ô là vật cản ({100*solid_count/solid.size:.1f}%)")
        if solid_count == 0:
            print("[wind] ⚠️ CẢNH BÁO: không ô nào được nhận là vật cản — gió sẽ đi thẳng, "
                  "khả năng do độ phân giải quá thô hoặc hình học quá mỏng/rời rạc so với lưới.")

        jobs[job_id]["status"] = "solving"

        def progress_cb(step, total):
            jobs[job_id]["progress"] = f"{step}/{total}"

        u, v, w = run_stable_fluids_3d(solid, wind_dir, speed, iterations=iterations, progress_cb=progress_cb)

        jobs[job_id]["status"] = "tracing_streamlines"
        minx, maxx, miny, maxy, minz, maxz = bounds
        seeds = []
        ny_s, nz_s = seed_grid
        inflow_x = minx if wind_dir[0] >= 0 else maxx
        for j in range(ny_s):
            for k in range(nz_s):
                y = miny + (maxy - miny) * (j + 0.5) / ny_s
                z = minz + (maxz - minz) * (k + 0.5) / nz_s
                seeds.append([inflow_x + (maxx - minx) * 0.02 * (1 if wind_dir[0] >= 0 else -1), y, z])

        lines = trace_streamlines(u, v, w, bounds, resolution, seeds)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = lines
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        traceback.print_exc()


@app.route("/simulate", methods=["POST"])
def simulate():
    if "stl" not in request.files:
        return jsonify({"error": "Thiếu file STL (field 'stl')"}), 400

    stl_bytes = request.files["stl"].read()

    dir_deg = float(request.form.get("dirDeg", 0))
    speed = float(request.form.get("speed", 5))
    res = int(request.form.get("resolution", 32))       # mỗi chiều, mặc định vừa phải cho free tier
    iterations = int(request.form.get("iterations", 60))
    seed_count = int(request.form.get("seedCount", 5))   # lưới hạt gieo NxN

    rad = np.radians(dir_deg)
    wind_dir = (np.cos(rad), 0.0, np.sin(rad))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": None, "result": None, "error": None}

    thread = threading.Thread(
        target=run_simulation_job,
        args=(job_id, stl_bytes, wind_dir, speed, (res, res, res), iterations, (seed_count, seed_count)),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Không tìm thấy job"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"], "error": job["error"]})


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Không tìm thấy job"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job chưa xong, status hiện tại: " + job["status"]}), 400
    return jsonify(job["result"])


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "message": "Wind simulation backend đang chạy."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
