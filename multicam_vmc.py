#!/usr/bin/env python3
"""
multicam_vmc.py - multi-webcam face tracking fused into one VMC stream.

Each cam runs MediaPipe Face Landmarker (52 ARKit blendshapes + head pose).
Per-cam weight = how frontal face is to that cam. Weighted average across cams
-> smooth, no popping, fewer occlusion/side-angle dropouts.
Output: VMC protocol over UDP (VNyan, Warudo, VSeeFace, etc. VMC receiver).

Install:  pip install -r requirements.txt   (or use the installer build, see README)
Run:      python multicam_vmc.py --cams 0 1
Phone cam: --auto-phone (finds stream on LAN) | --discover | --cams 0 http://IP:8080/video
No args:   opens launcher window (pick cams, find phone, start)
Helpers:   --list-cams  show local cam indexes

Cam placement auto-calibrated: first run (or 'c' key) sit in view of ALL cams,
slowly turn head left/right/up/down ~6 s. Relative cam rotations solved from
head pose seen by each cam at same time, saved to calibration.json.

Preview: cam feeds + face skeleton overlay + weight bars, 3D rig view (head skeleton +
solved cam positions), expression bars.
Hotkeys:  q quit | c recalibrate | y/p/r flip head yaw/pitch/roll sign | m mirror L/R
          f dense landmark dots | a/d/w/s orbit rig view
"""
import argparse
import math
import os
import sys
import threading
import time
import urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_NAME = "face_landmarker.task"


# ---------------------------------------------------------------- math ----
def rot_to_euler(R):
    """3x3 rotation (unit columns) -> yaw, pitch, roll degrees (Unity ZXY order)."""
    yaw = math.atan2(R[0][2], R[2][2])
    pitch = math.asin(max(-1.0, min(1.0, -R[1][2])))
    roll = math.atan2(R[1][0], R[1][1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def euler_to_quat(yaw, pitch, roll):
    """degrees -> quaternion (x,y,z,w), applied roll(Z), pitch(X), yaw(Y)."""
    y, p, r = (math.radians(v) / 2 for v in (yaw, pitch, roll))
    qy = (0.0, math.sin(y), 0.0, math.cos(y))
    qx = (math.sin(p), 0.0, 0.0, math.cos(p))
    qz = (0.0, 0.0, math.sin(r), math.cos(r))
    return qmul(qmul(qy, qx), qz)


def swap_lr(name):
    if name.endswith("Left"):
        return name[:-4] + "Right"
    if name.endswith("Right"):
        return name[:-5] + "Left"
    return name


# ------------------------------------------------------------- capture ----
class Cam:
    def __init__(self, idx, w, h, fps):
        import cv2
        self.idx = idx
        if isinstance(idx, int):  # local/virtual webcam
            api = cv2.CAP_DSHOW if os.name == "nt" else 0
            self.cap = cv2.VideoCapture(idx, api)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        else:  # stream URL (http MJPEG / rtsp) e.g. phone IP cam app
            self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cam {idx} failed to open")
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import cv2
        fails = 0
        while not self.stop:
            ok, f = self.cap.read()
            if ok:
                fails = 0
                with self.lock:
                    self.frame = f
                    self.seq += 1
            else:
                fails += 1
                time.sleep(0.01)
                if fails > 200 and not isinstance(self.idx, int):  # phone stream dropped
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(self.idx)
                    fails = 0

    def latest(self):
        with self.lock:
            return self.frame, self.seq

    def release(self):
        self.stop = True
        time.sleep(0.05)
        self.cap.release()


# ------------------------------------------------------------- tracker ----
class Tracker:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.last_ts = 0

    def process(self, bgr):
        """-> (blendshape dict, 3x3 head rotation in cam frame, Nx2 landmarks 0-1) or None."""
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_blendshapes or not res.facial_transformation_matrixes:
            return None
        blends = {c.category_name: float(c.score) for c in res.face_blendshapes[0]
                  if c.category_name != "_neutral"}
        m = np.array(res.facial_transformation_matrixes[0])
        R = m[:3, :3]
        R = R / np.linalg.norm(R, axis=0)
        pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float32)
        return blends, R, pts


# -------------------------------------------------------- calibration ----
def proj_so3(M):
    """nearest rotation matrix (chordal mean)."""
    import numpy as np
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class Calibrator:
    """
    Head pose seen by cam i:  R_i = C_i^-1 * H   (H = true head rot, C_i = cam rot in world)
    so  R_i * R_j^T = C_i^-1 * C_j  = rotation taking cam j frame -> cam i frame,
    independent of head pose. Average over many frames -> cam layout.
    T[i] = rotation taking cam i frame -> cam 0 frame.
    """

    def __init__(self, n):
        self.n = n
        self.S = {}
        self.t0 = time.monotonic()

    def add(self, Rs):
        import numpy as np
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if Rs[i] is None or Rs[j] is None:
                    continue
                s = self.S.setdefault((i, j), [np.zeros((3, 3)), 0])
                s[0] += Rs[i] @ Rs[j].T
                s[1] += 1

    def counts(self):
        return {k: v[1] for k, v in self.S.items()}

    def solve(self, min_samples):
        import numpy as np
        T = {0: np.eye(3)}
        changed = True
        while changed:
            changed = False
            for (i, j), (M, c) in self.S.items():
                if c < min_samples:
                    continue
                Tij = proj_so3(M)  # j -> i
                if i in T and j not in T:
                    T[j] = T[i] @ Tij
                    changed = True
                elif j in T and i not in T:
                    T[i] = T[j] @ Tij.T
                    changed = True
        return [T[i] for i in range(self.n)] if len(T) == self.n else None


def save_calib(path, cam_ids, Ts):
    import json
    with open(path, "w") as f:
        json.dump({"cams": cam_ids, "T": [t.tolist() for t in Ts]}, f)


def load_calib(path, cam_ids):
    import json
    import numpy as np
    try:
        with open(path) as f:
            d = json.load(f)
        if d["cams"] == cam_ids:
            return [np.array(t) for t in d["T"]]
    except Exception:
        pass
    return None


def describe_layout(cam_ids, Ts):
    for i, T in enumerate(Ts):
        if i == 0:
            print(f"cam{cam_ids[0]}: reference")
            continue
        y, p, r = rot_to_euler(T)
        print(f"cam{cam_ids[i]} vs cam{cam_ids[0]}: yaw {y:+.0f} pitch {p:+.0f} roll {r:+.0f} deg")


# -------------------------------------------------------------- fusion ----
class Fuser:
    def __init__(self, alpha):
        self.alpha = alpha
        self.blends = {}
        self.euler = [0.0, 0.0, 0.0]

    @staticmethod
    def weight(R):
        yaw, pitch, _ = rot_to_euler(R)
        if abs(yaw) > 80 or abs(pitch) > 70:
            return 0.0
        w = max(0.0, math.cos(math.radians(yaw))) * max(0.0, math.cos(math.radians(pitch)))
        return w * w

    def update(self, results, Ts):
        """results: per cam (blends, R_cam) or None. Ts: cam->ref rotations.
        returns per-cam weights."""
        import numpy as np
        weights = []
        acc_b, acc_R, tot = {}, np.zeros((3, 3)), 0.0
        for r, T in zip(results, Ts):
            if r is None:
                weights.append(0.0)
                continue
            blends, R, _ = r
            w = self.weight(R)
            weights.append(w)
            if w <= 0:
                continue
            tot += w
            for k, v in blends.items():
                acc_b[k] = acc_b.get(k, 0.0) + v * w
            acc_R += (T @ R) * w
        if tot > 1e-6:
            a = self.alpha
            for k, v in acc_b.items():
                old = self.blends.get(k, v)
                self.blends[k] = old + a * (v / tot - old)
            target = rot_to_euler(proj_so3(acc_R))
            for i in range(3):
                self.euler[i] += a * (target[i] - self.euler[i])
        return weights


# ----------------------------------------------------------------- VMC ----
def vrm_extra(b):
    g = b.get
    bl, br = g("eyeBlinkLeft", 0.0), g("eyeBlinkRight", 0.0)
    jaw = g("jawOpen", 0.0)
    smile = (g("mouthSmileLeft", 0.0) + g("mouthSmileRight", 0.0)) / 2
    stretch = (g("mouthStretchLeft", 0.0) + g("mouthStretchRight", 0.0)) / 2
    return {
        "Blink": (bl + br) / 2, "Blink_L": bl, "Blink_R": br,
        "A": jaw, "O": g("mouthFunnel", 0.0), "U": g("mouthPucker", 0.0),
        "I": min(1.0, smile * (1 - jaw)), "E": min(1.0, stretch),
    }


def send_vmc(client_sock, addr, blends, euler, flips, mirror, vrm):
    from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
    from pythonosc.osc_message_builder import OscMessageBuilder

    bundle = OscBundleBuilder(IMMEDIATELY)

    def add(address, *args):
        m = OscMessageBuilder(address=address)
        for a in args:
            m.add_arg(a)
        bundle.add_content(m.build())

    out = dict(blends)
    if mirror:
        out = {swap_lr(k): v for k, v in out.items()}
    if vrm:
        out.update(vrm_extra(out))
    for k, v in out.items():
        add("/VMC/Ext/Blend/Val", k, float(v))
    add("/VMC/Ext/Blend/Apply")

    yaw, pitch, roll = euler
    q = euler_to_quat(yaw * flips[0], pitch * flips[1], roll * flips[2])
    add("/VMC/Ext/Bone/Pos", "Head", 0.0, 0.0, 0.0, *(float(x) for x in q))
    add("/VMC/Ext/OK", 1)
    client_sock.sendto(bundle.build().dgram, addr)


# ----------------------------------------------------------------- main ----
# ------------------------------------------------------------- preview ----
FACE_LINES = [
    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
     377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
     67, 109, 10],
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
     40, 185, 61],
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81,
     80, 191, 78],
    [276, 283, 282, 295, 285],
    [46, 53, 52, 65, 55],
    [168, 6, 197, 195, 5, 4],
]
TILE = (320, 240)


def euler_to_mat(yaw, pitch, roll):
    """inverse of rot_to_euler: R = Ry(yaw) Rx(pitch) Rz(roll), degrees."""
    import numpy as np
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def draw_cam_tile(frame, res, weight, label, dense):
    import cv2
    import numpy as np
    tw, th = TILE
    t = cv2.resize(frame, TILE) if frame is not None else np.zeros((th, tw, 3), np.uint8)
    if res is not None:
        pts = res[2] * np.array([tw, th], np.float32)
        g = int(60 + 195 * min(1.0, weight))
        col = (0, g, 255 - g)  # BGR: red (weak) -> green (strong)
        if dense:
            for p in pts:
                cv2.circle(t, (int(p[0]), int(p[1])), 1, col, -1)
        for line in FACE_LINES:
            poly = pts[line].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(t, [poly], False, col, 1, cv2.LINE_AA)
    else:
        cv2.putText(t, "no face", (tw // 2 - 40, th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    cv2.rectangle(t, (0, th - 6), (int(tw * min(1.0, weight)), th), (0, 220, 0), -1)
    cv2.putText(t, f"{label} w={weight:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    return t


def draw_rig(size, euler, Ts, cam_ids, weights, view):
    """3D view in cam0 frame: head skeleton + where solved cams sit around it."""
    import cv2
    import numpy as np
    w, h = size
    img = np.full((h, w, 3), 28, np.uint8)
    s = min(w, h) * 0.40
    cx, cy = w // 2, int(h * 0.54)
    V = euler_to_mat(view[0], view[1], 0.0)

    def P(p):
        q = V @ np.asarray(p, dtype=float)
        return int(cx + s * q[0]), int(cy - s * q[1])

    def L(a, b, col, th=1):
        cv2.line(img, P(a), P(b), col, th, cv2.LINE_AA)

    def poly(pts, col, th=1, closed=True):
        arr = np.array([P(p) for p in pts], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], closed, col, th, cv2.LINE_AA)

    ts = [i * 2 * math.pi / 48 for i in range(48)]
    poly([(math.cos(t), 0, math.sin(t)) for t in ts], (60, 60, 60))  # cam ring

    Hm = euler_to_mat(*euler)

    def hp(p):
        return Hm @ np.asarray(p, dtype=float)

    poly([hp((0.28 * math.cos(t), 0.36 * math.sin(t), 0.0)) for t in ts], (220, 220, 220), 2)
    poly([hp((0.0, 0.36 * math.sin(t), 0.3 * math.cos(t))) for t in ts], (140, 140, 140))
    poly([hp((0.28 * math.cos(t), 0.0, 0.3 * math.sin(t))) for t in ts], (140, 140, 140))
    L(hp((0, -0.36, 0)), hp((0, -0.8, -0.1)), (200, 200, 200), 2)  # neck
    L(hp((-0.4, -0.8, -0.1)), hp((0.4, -0.8, -0.1)), (200, 200, 200), 2)  # shoulders
    L(hp((0, 0.0, 0.3)), hp((0, -0.1, 0.42)), (220, 220, 220), 2)  # nose
    L(hp((-0.08, -0.17, 0.27)), hp((0.08, -0.17, 0.27)), (220, 220, 220), 2)  # mouth
    for sx in (-1, 1):
        cv2.circle(img, P(hp((0.1 * sx, 0.1, 0.29))), 3, (255, 255, 255), -1)
    for axis, col in (((0.55, 0, 0), (0, 0, 255)), ((0, 0.55, 0), (0, 255, 0)),
                      ((0, 0, 0.55), (255, 0, 0))):
        L((0, 0, 0), hp(axis), col, 2)

    for i, (T, cid) in enumerate(zip(Ts, cam_ids)):
        wt = min(1.0, weights[i]) if i < len(weights) else 0.0
        col = (110 - int(110 * wt), 110 + int(110 * wt), 110 - int(110 * wt))
        pos = T @ np.array([0.0, 0.0, 1.0])  # cam sits along +z of own frame, looks at head
        ex, ey = T[:, 0], T[:, 1]
        base = pos * 0.72
        corners = [base + 0.16 * (sx * ex + sy * ey)
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for c in corners:
            L(pos, c, col)
        poly(corners, col, 2)
        L(base, (0, 0, 0), (70, 70, 70))
        lx, ly = P(pos * 1.3)
        cv2.putText(img, f"cam{cid}" + (" (ref)" if i == 0 else ""), (lx - 20, ly - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, "rig (cam0 frame)  a/d/w/s orbit", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def draw_bars(size, blends):
    import cv2
    import numpy as np
    w, h = size
    img = np.full((h, w, 3), 28, np.uint8)
    cv2.putText(img, "expression (fused)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    items = sorted(((v, k) for k, v in blends.items() if v > 0.02), reverse=True)
    rows = max(1, (h - 30) // 18)
    for n, (v, k) in enumerate(items[:rows]):
        y = 34 + n * 18
        cv2.putText(img, k, (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1,
                    cv2.LINE_AA)
        x0 = 160
        cv2.rectangle(img, (x0, y - 6), (x0 + int((w - x0 - 40) * min(1.0, v)), y + 6),
                      (0, 200, 255), -1)
        cv2.putText(img, f"{v:.2f}", (w - 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (160, 160, 160), 1, cv2.LINE_AA)
    return img


def compose_preview(frames, results, weights, cam_ids, Ts, fuser, view, dense, status):
    import cv2
    import numpy as np
    tiles = [draw_cam_tile(f, r, w, f"cam{cid}", dense)
             for f, r, w, cid in zip(frames, results, weights, cam_ids)]
    W = max(TILE[0] * len(tiles), 640)
    top = np.zeros((TILE[1], W, 3), np.uint8)
    for i, t in enumerate(tiles):
        top[:, i * TILE[0]:(i + 1) * TILE[0]] = t
    half = W // 2
    rig = draw_rig((half, 340), fuser.euler, Ts, cam_ids, weights, view)
    bars = draw_bars((W - half, 340), fuser.blends)
    strip = np.full((24, W, 3), 12, np.uint8)
    cv2.putText(strip, status, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                cv2.LINE_AA)
    return np.vstack([top, np.hstack([rig, bars]), strip])


# ----------------------------------------------------------- discovery ----
STREAM_CANDIDATES = [(8080, "/video"), (4747, "/video"), (4747, "/mjpegfeed?640x480"),
                     (8081, "/video"), (8080, "/videofeed")]


def list_cams(max_idx=8):
    import cv2
    print("local/virtual cams (use these numbers in --cams):")
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if cap.isOpened():
            ok, f = cap.read()
            print(f"  index {i}: {f.shape[1]}x{f.shape[0]}" if ok else f"  index {i}: opened, no frame")
        cap.release()


def local_subnet():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # UDP connect, sends nothing
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip, ip.rsplit(".", 1)[0]


def probe_stream(ip, port, path, timeout=0.4):
    """-> stream URL if ip:port serves an MJPEG/video stream at path, else None."""
    import http.client
    import socket
    try:
        socket.create_connection((ip, port), timeout=timeout).close()
    except OSError:
        return None
    try:
        c = http.client.HTTPConnection(ip, port, timeout=timeout * 3)
        c.request("GET", path)
        r = c.getresponse()
        ct = (r.getheader("Content-Type") or "").lower()
        ok = r.status == 200 and ("multipart" in ct or "jpeg" in ct or "video" in ct)
        c.close()
        return f"http://{ip}:{port}{path}" if ok else None
    except Exception:
        return None


def discover_phone_cams():
    from concurrent.futures import ThreadPoolExecutor
    ip, base = local_subnet()
    print(f"scanning {base}.0/24 for phone cam streams (you are {ip}) ...")
    hosts = [f"{base}.{i}" for i in range(1, 255) if f"{base}.{i}" != ip]
    jobs = [(h, p, path) for h in hosts for p, path in STREAM_CANDIDATES]
    found, seen = [], set()
    with ThreadPoolExecutor(128) as ex:
        for url in ex.map(lambda j: probe_stream(*j), jobs):
            if url:
                host = url.split("/")[2].split(":")[0]
                if host not in seen:  # one URL per phone
                    seen.add(host)
                    found.append(url)
    return found


def ensure_model(path):
    if os.path.exists(path):
        return
    print("downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, path)


def main():
    if len(sys.argv) == 1:  # double-click / no args -> launcher window
        gui_args = launcher()
        if gui_args is None:
            return
        sys.argv = [sys.argv[0]] + gui_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["0", "1"],
                    help="cam indexes and/or stream URLs, e.g. 0 http://192.168.1.50:8080/video")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39539, help="VMC receiver port")
    ap.add_argument("--smooth", type=float, default=0.5, help="0-1, higher = snappier")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--calib-file", default=os.path.join(data_dir(), "calibration.json"))
    ap.add_argument("--calib-secs", type=float, default=6.0)
    ap.add_argument("--recalibrate", action="store_true", help="ignore saved calibration")
    ap.add_argument("--vrm", action="store_true",
                    help="also send VRM names (Blink/A/I/U/E/O) for non-perfect-sync models")
    ap.add_argument("--mirror", action="store_true", help="swap left/right blendshapes")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--nogui", action="store_true", help="skip launcher window, use CLI defaults")
    ap.add_argument("--list-cams", action="store_true", help="show local cam indexes, exit")
    ap.add_argument("--discover", action="store_true",
                    help="scan LAN for phone cam streams, print URLs, exit")
    ap.add_argument("--auto-phone", action="store_true",
                    help="scan LAN, add first phone stream found to --cams")
    args = ap.parse_args()
    args.cams = [int(c) if c.isdigit() else c for c in args.cams]

    if args.list_cams:
        list_cams()
        return
    if args.discover or args.auto_phone:
        found = discover_phone_cams()
        if not found:
            print("no phone stream found. phone app running? same WiFi? firewall?")
            if args.discover:
                return
        for u in found:
            print("found:", u)
        if args.discover:
            return
        if found:
            args.cams.append(found[0])

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    mp_file = resolve_model_path()
    ensure_model(mp_file)
    cams, opened = [], []
    for cid in args.cams:
        try:
            cams.append(Cam(cid, args.width, args.height, args.fps))
            opened.append(cid)
        except RuntimeError as e:
            print(f"skipping: {e}")
    if not cams:
        sys.exit("no cams opened. try --list-cams or --discover")
    args.cams = opened
    trackers = [Tracker(mp_file) for _ in cams]
    n = len(cams)
    fuser = Fuser(args.smooth)
    flips = [1, 1, 1]
    mirror = args.mirror
    last_seq = [-1] * n
    last_res = [None] * n
    last_t = [time.monotonic()] * n
    view = [35.0, 25.0]
    dense = False

    import numpy as np
    Ts = None if args.recalibrate else load_calib(args.calib_file, args.cams)
    calib = None
    if n == 1:
        Ts = [np.eye(3)]
    if Ts is None:
        calib = Calibrator(n)
        print("CALIBRATING: stay in view of ALL cams, slowly turn head left/right/up/down.")
    else:
        describe_layout(args.cams, Ts)
    print(f"sending VMC to {args.host}:{args.port} | cams {args.cams} | q to quit")

    try:
        while True:
            frames = []
            new = False
            for i, cam in enumerate(cams):
                frame, seq = cam.latest()
                frames.append(frame)
                if seq == last_seq[i] and time.monotonic() - last_t[i] > 0.5:
                    last_res[i] = None  # cam stalled/dropped -> stop trusting it
                if frame is None or seq == last_seq[i]:
                    continue
                last_seq[i] = seq
                last_t[i] = time.monotonic()
                new = True
                last_res[i] = trackers[i].process(frame)
            if not new:
                time.sleep(0.003)
                continue

            weights = [0.0] * n
            status = ""
            if calib is not None:
                calib.add([None if r is None else r[1] for r in last_res])
                counts = calib.counts()
                need = 90
                status = "CALIBRATING " + " ".join(
                    f"{i}-{j}:{c}/{need}" for (i, j), c in sorted(counts.items()))
                if time.monotonic() - calib.t0 >= args.calib_secs:
                    solved = calib.solve(need)
                    if solved is not None:
                        Ts = solved
                        save_calib(args.calib_file, args.cams, Ts)
                        print("calibration done:")
                        describe_layout(args.cams, Ts)
                        calib = None
                        fuser = Fuser(args.smooth)
                    elif not counts:
                        status = "CALIBRATING: no cam pair sees face together"
            else:
                weights = fuser.update(last_res, Ts)
                if fuser.blends:
                    send_vmc(sock, addr, fuser.blends, fuser.euler, flips, mirror, args.vrm)

            if not args.no_preview:
                import cv2
                y, p, r = fuser.euler
                line = status or (f"yaw {y:6.1f} pitch {p:6.1f} roll {r:6.1f} | "
                                  f"flip {flips} mirror {mirror} | "
                                  "q quit c calib y/p/r flip m mirror f dots")
                Td = Ts if Ts is not None else [np.eye(3)] * n
                disp = [(f if f is not None else np.zeros((480, 640, 3), np.uint8))
                        for f in frames]
                cv2.imshow("multicam_vmc", compose_preview(
                    disp, last_res, weights, args.cams, Td, fuser, view, dense, line))
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or cv2.getWindowProperty("multicam_vmc", cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif k == ord("c") and n > 1:
                    calib = Calibrator(n)
                    print("CALIBRATING: turn head slowly in view of all cams.")
                elif k == ord("y"):
                    flips[0] *= -1
                elif k == ord("p"):
                    flips[1] *= -1
                elif k == ord("r"):
                    flips[2] *= -1
                elif k == ord("m"):
                    mirror = not mirror
                elif k == ord("f"):
                    dense = not dense
                elif k == ord("a"):
                    view[0] -= 8
                elif k == ord("d"):
                    view[0] += 8
                elif k == ord("w"):
                    view[1] += 8
                elif k == ord("s"):
                    view[1] -= 8
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


# ------------------------------------------------------ app plumbing ----
def data_dir():
    """per-user writable dir (no admin): calibration, settings, log, downloaded model."""
    appdata = os.environ.get("APPDATA") if os.name == "nt" else None
    d = os.path.join(appdata, "multicam_vmc") if appdata else os.path.join(
        os.path.expanduser("~"), ".multicam_vmc")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_model_path():
    bundled = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
                           MODEL_NAME)
    return bundled if os.path.exists(bundled) else os.path.join(data_dir(), MODEL_NAME)


def load_settings():
    import json
    try:
        with open(os.path.join(data_dir(), "settings.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    import json
    try:
        with open(os.path.join(data_dir(), "settings.json"), "w") as f:
            json.dump(d, f)
    except Exception:
        pass


def show_error(msg):
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("multicam_vmc", msg)
        r.destroy()
    except Exception:
        pass


def launcher():
    """small settings window. returns CLI arg list, or None if closed."""
    import queue
    import tkinter as tk
    from tkinter import messagebox, ttk

    st = load_settings()
    root = tk.Tk()
    root.title("multicam_vmc")
    root.resizable(False, False)
    f = ttk.Frame(root, padding=12)
    f.grid()
    q = queue.Queue()
    result = []
    cam_vars = {}

    ttk.Label(f, text="Local cameras", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, sticky="w")
    scan_btn = ttk.Button(f, text="Rescan")
    scan_btn.grid(row=0, column=1, sticky="e")
    cams_box = ttk.Frame(f)
    cams_box.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))

    def scan_worker():
        import cv2
        found = []
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
            if cap.isOpened():
                ok, fr = cap.read()
                found.append((i, f"{fr.shape[1]}x{fr.shape[0]}" if ok else "no frame"))
            cap.release()
        q.put(("cams", found))

    def start_scan():
        scan_btn.state(["disabled"])
        for w in cams_box.winfo_children():
            w.destroy()
        ttk.Label(cams_box, text="scanning cameras ...").grid()
        threading.Thread(target=scan_worker, daemon=True).start()

    def show_cams(found):
        for w in cams_box.winfo_children():
            w.destroy()
        cam_vars.clear()
        saved = [c for c in st.get("cams", []) if isinstance(c, int)]
        if not found:
            ttk.Label(cams_box, text="no local cameras found").grid()
        for n, (idx, res) in enumerate(found):
            v = tk.BooleanVar(value=(idx in saved) if saved else n < 2)
            cam_vars[idx] = v
            ttk.Checkbutton(cams_box, text=f"Camera {idx}  ({res})", variable=v).grid(
                row=n, column=0, sticky="w")
        scan_btn.state(["!disabled"])

    scan_btn.config(command=start_scan)

    ttk.Label(f, text="Phone cam stream URL (optional)", font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, columnspan=2, sticky="w")
    phone_var = tk.StringVar(value=st.get("phone", ""))
    ttk.Entry(f, textvariable=phone_var, width=44).grid(row=3, column=0, sticky="w")
    find_btn = ttk.Button(f, text="Find on WiFi")
    find_btn.grid(row=3, column=1, sticky="e", padx=(6, 0))
    phone_status = ttk.Label(f, text="phone app running + same WiFi, then press Find",
                             foreground="#666")
    phone_status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 10))

    def find_worker():
        q.put(("phone", discover_phone_cams()))

    def start_find():
        find_btn.state(["disabled"])
        phone_status.config(text="scanning WiFi (few sec) ...")
        threading.Thread(target=find_worker, daemon=True).start()

    def phone_done(found):
        find_btn.state(["!disabled"])
        if found:
            phone_var.set(found[0])
            phone_status.config(text=f"found {found[0]}")
        else:
            phone_status.config(text="none found. app running? same WiFi? needs no password?")

    find_btn.config(command=start_find)

    opt = ttk.Frame(f)
    opt.grid(row=5, column=0, columnspan=2, sticky="w")
    ttk.Label(opt, text="VMC port").grid(row=0, column=0)
    port_var = tk.StringVar(value=str(st.get("port", 39539)))
    ttk.Entry(opt, textvariable=port_var, width=7).grid(row=0, column=1, padx=(4, 12))
    vrm_var = tk.BooleanVar(value=st.get("vrm", False))
    mir_var = tk.BooleanVar(value=st.get("mirror", False))
    rec_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt, text="VRM names", variable=vrm_var).grid(row=0, column=2)
    ttk.Checkbutton(opt, text="Mirror L/R", variable=mir_var).grid(row=0, column=3, padx=6)
    ttk.Checkbutton(opt, text="Recalibrate", variable=rec_var).grid(row=0, column=4)

    ttk.Label(f, justify="left", foreground="#666", text=(
        "First start calibrates cam positions: sit in view of ALL cams,\n"
        "turn head slowly left/right/up/down.\n"
        "Preview keys: q quit, c recalibrate, y/p/r flip head axis, m mirror."
    )).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 10))

    def go():
        cams = [str(i) for i, v in cam_vars.items() if v.get()]
        url = phone_var.get().strip()
        if url:
            cams.append(url)
        if not cams:
            messagebox.showwarning("multicam_vmc", "Pick at least one camera.")
            return
        port = port_var.get().strip()
        if not port.isdigit():
            messagebox.showwarning("multicam_vmc", "Port must be a number.")
            return
        argv = ["--cams", *cams, "--port", port]
        if vrm_var.get():
            argv.append("--vrm")
        if mir_var.get():
            argv.append("--mirror")
        if rec_var.get():
            argv.append("--recalibrate")
        save_settings({"cams": [int(c) for c in cams if c.isdigit()], "phone": url,
                       "port": int(port), "vrm": vrm_var.get(), "mirror": mir_var.get()})
        result.append(argv)
        root.destroy()

    ttk.Button(f, text="Start", command=go).grid(row=7, column=0, columnspan=2, sticky="ew")

    def poll():
        try:
            while True:
                kind, data = q.get_nowait()
                if kind == "cams":
                    show_cams(data)
                elif kind == "phone":
                    phone_done(data)
        except queue.Empty:
            pass
        root.after(150, poll)

    root.after(100, start_scan)
    poll()
    root.mainloop()
    return result[0] if result else None


def setup_logging():
    if getattr(sys, "frozen", False):  # windowed exe has no console
        log = open(os.path.join(data_dir(), "log.txt"), "w", buffering=1, encoding="utf-8")
        sys.stdout = log
        sys.stderr = log


def run():
    setup_logging()
    frozen = getattr(sys, "frozen", False)
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
            if frozen:
                show_error(e.code)
            sys.exit(1)
        raise
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        if frozen:
            show_error("Crashed. Log: " + os.path.join(data_dir(), "log.txt") + "\n\n" + tb[-800:])
        sys.exit(1)


if __name__ == "__main__":
    run()
