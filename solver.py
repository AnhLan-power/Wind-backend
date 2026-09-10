"""
solver.py — Bộ giải gió 3D đơn giản hoá (KHÔNG PHẢI CFD chuẩn công nghiệp)

Thuật toán: mở rộng 3D của "Stable Fluids" (Jos Stam) — advection bán-Lagrange
+ chiếu áp suất (pressure projection) qua lặp Jacobi để đảm bảo dòng chảy
không nén được (incompressible). KHÔNG có mô hình nhiễu loạn (turbulence
closure) chuẩn như k-epsilon/k-omega mà phần mềm CFD thật dùng — nên xoáy/
tách dòng thể hiện được về mặt ĐỊNH TÍNH (đúng xu hướng: chậm sau vật cản,
tăng tốc qua khe hẹp) nhưng KHÔNG chính xác về mặt ĐỊNH LƯỢNG như OpenFOAM/
Ansys thật.

Quy trình:
1. Đọc file STL (binary) -> danh sách tam giác
2. Voxel hoá: xác định ô lưới nào nằm trong vật thể (ray casting theo trục X)
3. Giải trường vận tốc 3D bằng Stable Fluids
4. Dò các đường dòng (streamline) bằng tích phân Runge-Kutta 4 từ các điểm
   gieo (seed) ở mặt đón gió
5. Trả về danh sách đường dòng dạng JSON (points + velocities mỗi điểm)
"""

import struct
import numpy as np


# ============================================================
# 1. ĐỌC FILE STL (BINARY)
# ============================================================
def parse_binary_stl(data: bytes):
    """Trả về mảng numpy (N, 3, 3): N tam giác, mỗi tam giác 3 đỉnh x,y,z."""
    tri_count = struct.unpack_from("<I", data, 80)[0]
    triangles = np.zeros((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    for i in range(tri_count):
        # bỏ qua normal (12 byte đầu), đọc 3 đỉnh (36 byte), bỏ qua attribute (2 byte)
        v = struct.unpack_from("<9f", data, offset + 12)
        triangles[i, 0] = v[0:3]
        triangles[i, 1] = v[3:6]
        triangles[i, 2] = v[6:9]
        offset += 50
    return triangles


# ============================================================
# 2. VOXEL HOÁ — xác định ô lưới nào là vật cản (ray casting +X)
# ============================================================
def _ray_triangle_intersect_x(origin_yz, triangles):
    """
    Kiểm tra tia bắn theo +X (từ origin_yz=(y,z), x=-inf) cắt bao nhiêu tam
    giác — dùng thuật toán Möller–Trumbore, vector hoá qua toàn bộ tam giác
    cho 1 điểm gốc. Trả về số giao điểm có x lớn hơn 1 mốc tham chiếu.
    """
    oy, oz = origin_yz
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    # Chiếu bài toán sang mặt phẳng YZ (bỏ trục X vì tia song song trục X)
    # Dùng công thức giao điểm tia-tam giác rút gọn cho tia trục X.
    dir_x = np.array([1.0, 0.0, 0.0])
    h = np.cross(np.full_like(e1, dir_x), e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > 1e-9

    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]

    origin = np.array([-1e6, oy, oz])
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)

    q = np.cross(s, e1)
    v = f * np.einsum("ij,ij->i", np.tile(dir_x, (len(e2), 1)), q)

    t = f * np.einsum("ij,ij->i", e2, q)

    hit = valid & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 0)
    x_hits = origin[0] + t[hit]
    return x_hits


def mark_surface_cells(triangles, bounds, resolution):
    """
    Đánh dấu trực tiếp các ô lưới có ĐỈNH tam giác nằm gần đó — không cần
    hình học kín (watertight). Phù hợp với kết cấu dàn/khung/dầm mảnh
    (nhiều khoảng hở thật sự giữa các thanh) mà cách "điểm nằm trong khối
    kín" (build_solid_grid) dễ bỏ sót vì tưởng nhầm là rỗng.
    """
    from scipy.ndimage import binary_dilation
    minx, maxx, miny, maxy, minz, maxz = bounds
    nx, ny, nz = resolution
    verts = triangles.reshape(-1, 3)

    ix = np.clip(((verts[:, 0] - minx) / (maxx - minx) * (nx - 1)).astype(int), 0, nx - 1)
    iy = np.clip(((verts[:, 1] - miny) / (maxy - miny) * (ny - 1)).astype(int), 0, ny - 1)
    iz = np.clip(((verts[:, 2] - minz) / (maxz - minz) * (nz - 1)).astype(int), 0, nz - 1)

    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[ix, iy, iz] = True
    # Nới rộng thêm 1 ô mỗi hướng để nối liền các khoảng hở nhỏ giữa các
    # đỉnh tam giác liền kề (tránh "gió lọt qua khe" do lấy mẫu rời rạc).
    solid = binary_dilation(solid, iterations=1)
    return solid


def build_solid_grid(triangles, bounds, resolution):
    """
    bounds: (minx, maxx, miny, maxy, minz, maxz)
    resolution: (nx, ny, nz)
    Trả về mảng bool 3D — True = ô đó nằm trong vật cản.
    """
    minx, maxx, miny, maxy, minz, maxz = bounds
    nx, ny, nz = resolution
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    zs = np.linspace(minz, maxz, nz)

    solid = np.zeros((nx, ny, nz), dtype=bool)
    for j, y in enumerate(ys):
        for k, z in enumerate(zs):
            x_hits = _ray_triangle_intersect_x(np.array([-1e6, y, z])[1:], triangles)
            if len(x_hits) == 0:
                continue
            x_hits_sorted = np.sort(x_hits)
            # đếm số lần cắt trước mỗi điểm lưới theo x -> lẻ = bên trong
            counts = np.searchsorted(x_hits_sorted, xs)
            inside = (counts % 2) == 1
            solid[:, j, k] = inside
    return solid


# ============================================================
# 3. STABLE FLUIDS 3D
# ============================================================
def run_stable_fluids_3d(solid, wind_dir, speed, iterations=80, progress_cb=None):
    """
    solid: mảng bool (nx,ny,nz)
    wind_dir: vector đơn vị (dx,dy,dz) hướng gió thổi TỚI (thường dy=0)
    speed: tốc độ gió tự do (đơn vị quy ước)
    Trả về (u, v, w): 3 mảng vận tốc cùng shape với solid.
    """
    nx, ny, nz = solid.shape
    u = np.where(solid, 0.0, wind_dir[0] * speed)
    v = np.where(solid, 0.0, wind_dir[1] * speed)
    w = np.where(solid, 0.0, wind_dir[2] * speed)

    fluid = ~solid
    inflow_x = wind_dir[0] > 0.3
    outflow_x = wind_dir[0] < -0.3

    def enforce_boundary(u, v, w):
        u[solid] = 0; v[solid] = 0; w[solid] = 0
        if inflow_x:
            u[0, :, :] = wind_dir[0] * speed
            v[0, :, :] = wind_dir[1] * speed
            w[0, :, :] = wind_dir[2] * speed
        elif outflow_x:
            u[-1, :, :] = wind_dir[0] * speed
            v[-1, :, :] = wind_dir[1] * speed
            w[-1, :, :] = wind_dir[2] * speed
        return u, v, w

    def project(u, v, w):
        div = np.zeros_like(u)
        div[1:-1, 1:-1, 1:-1] = -0.5 * (
            (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) +
            (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) +
            (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2])
        ) / nx
        div[solid] = 0

        p = np.zeros_like(u)
        for _ in range(35):
            p_new = p.copy()
            p_new[1:-1, 1:-1, 1:-1] = (
                div[1:-1, 1:-1, 1:-1] +
                p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
            ) / 6.0
            p_new[solid] = 0
            p = p_new

        u[1:-1, 1:-1, 1:-1] -= 0.5 * nx * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1])
        v[1:-1, 1:-1, 1:-1] -= 0.5 * nx * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1])
        w[1:-1, 1:-1, 1:-1] -= 0.5 * nx * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2])
        return u, v, w

    def advect_semilagrangian(field, u, v, w, dt):
        gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        src_x = np.clip(gx - dt * nx * u, 0.5, nx - 1.5)
        src_y = np.clip(gy - dt * ny * v, 0.5, ny - 1.5)
        src_z = np.clip(gz - dt * nz * w, 0.5, nz - 1.5)
        from scipy.ndimage import map_coordinates
        return map_coordinates(field, [src_x, src_y, src_z], order=1, mode="nearest")

    def advect(field, u, v, w, dt):
        """
        MacCormack advection — chính xác hơn bán-Lagrange bậc 1 thường nhờ
        bước dự đoán + hiệu chỉnh sai số 2 chiều, đỡ "mờ tan" xoáy hơn
        nhiều. Có clamp để tránh dao động/mất ổn định số.
        """
        phi1 = advect_semilagrangian(field, u, v, w, dt)
        phi2 = advect_semilagrangian(phi1, u, v, w, -dt)
        result = phi1 + 0.5 * (field - phi2)

        lo = field.copy()
        hi = field.copy()
        for ax in (0, 1, 2):
            for shift in (1, -1):
                shifted = np.roll(field, shift, axis=ax)
                lo = np.minimum(lo, shifted)
                hi = np.maximum(hi, shifted)
        return np.clip(result, lo, hi)

    def apply_vorticity_confinement(u, v, w, dt, epsilon=3.0):
        """
        Vorticity confinement (Fedkiw và cộng sự) — tính độ xoáy (curl) của
        trường vận tốc rồi "bơm" thêm lực theo đúng hướng xoáy để bù lại
        phần năng lượng xoáy bị khuếch tán số làm mất — nếu bỏ bước này,
        xoáy sinh ra sẽ tự mờ dần rồi biến mất sau vài chục bước, đúng
        hiện tượng đang gặp.
        """
        wx = np.gradient(w, axis=1) - np.gradient(v, axis=2)
        wy = np.gradient(u, axis=2) - np.gradient(w, axis=0)
        wz = np.gradient(v, axis=0) - np.gradient(u, axis=1)
        wmag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2) + 1e-6

        gx = np.gradient(wmag, axis=0)
        gy = np.gradient(wmag, axis=1)
        gz = np.gradient(wmag, axis=2)
        glen = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-6
        nx_, ny_, nz_ = gx / glen, gy / glen, gz / glen

        fx = epsilon * (ny_ * wz - nz_ * wy)
        fy = epsilon * (nz_ * wx - nx_ * wz)
        fz = epsilon * (nx_ * wy - ny_ * wx)

        u2 = u + fx * dt
        v2 = v + fy * dt
        w2 = w + fz * dt
        u2[solid] = 0; v2[solid] = 0; w2[solid] = 0
        return u2, v2, w2

    dt = 0.1
    for step in range(iterations):
        u, v, w = enforce_boundary(u, v, w)
        u, v, w = project(u, v, w)
        u, v, w = apply_vorticity_confinement(u, v, w, dt)
        u2 = advect(u, u, v, w, dt)
        v2 = advect(v, u, v, w, dt)
        w2 = advect(w, u, v, w, dt)
        u, v, w = u2, v2, w2
        u, v, w = enforce_boundary(u, v, w)
        if progress_cb:
            progress_cb(step + 1, iterations)

    return u, v, w


# ============================================================
# 4. DÒ ĐƯỜNG DÒNG (STREAMLINE) BẰNG RUNGE-KUTTA 4
# ============================================================
def _sample_velocity(u, v, w, pos, bounds, resolution):
    minx, maxx, miny, maxy, minz, maxz = bounds
    nx, ny, nz = resolution
    fx = (pos[0] - minx) / (maxx - minx) * (nx - 1)
    fy = (pos[1] - miny) / (maxy - miny) * (ny - 1)
    fz = (pos[2] - minz) / (maxz - minz) * (nz - 1)
    if fx < 0 or fx > nx - 1 or fy < 0 or fy > ny - 1 or fz < 0 or fz > nz - 1:
        return None
    from scipy.ndimage import map_coordinates
    coord = [[fx], [fy], [fz]]
    vx = map_coordinates(u, coord, order=1, mode="nearest")[0]
    vy = map_coordinates(v, coord, order=1, mode="nearest")[0]
    vz = map_coordinates(w, coord, order=1, mode="nearest")[0]
    return np.array([vx, vy, vz])


def trace_streamlines(u, v, w, bounds, resolution, seed_points, dt=0.3, max_steps=150):
    lines = []
    for idx, seed in enumerate(seed_points):
        pos = np.array(seed, dtype=np.float64)
        points, velocities = [pos.tolist()], []

        for _ in range(max_steps):
            k1 = _sample_velocity(u, v, w, pos, bounds, resolution)
            if k1 is None:
                break
            k2 = _sample_velocity(u, v, w, pos + 0.5 * dt * k1, bounds, resolution)
            if k2 is None:
                break
            k3 = _sample_velocity(u, v, w, pos + 0.5 * dt * k2, bounds, resolution)
            if k3 is None:
                break
            k4 = _sample_velocity(u, v, w, pos + dt * k3, bounds, resolution)
            if k4 is None:
                break

            vel = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            pos = pos + dt * vel
            points.append(pos.tolist())
            velocities.append(float(np.linalg.norm(vel)))

        if len(points) > 2:
            velocities.append(velocities[-1] if velocities else 0.0)
            lines.append({"id": idx + 1, "points": points, "velocities": velocities})

    return lines
