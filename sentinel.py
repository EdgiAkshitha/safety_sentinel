"""Workplace Safety Sentinel - laptop-only demo (no GPU, no TensorRT).
Detects people, checks a hazard zone, logs JSON incidents, serves a live web dashboard.
Sources: --source synthetic | 0 (webcam) | path/to/video.mp4
"""
import argparse, json, os, time, threading, math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

W, H = 640, 480
ZONE_N = [(0.58, 0.25), (0.95, 0.25), (0.95, 0.95), (0.58, 0.95)]
ZONE = np.array([(int(x * W), int(y * H)) for x, y in ZONE_N], np.int32)
LOCK = threading.Lock()
STATE = {"jpg": b"", "incidents": [], "stats": {"frames": 0, "people": 0, "in_zone": 0, "fps": 0.0, "mode": ""}}


class Synth:
    """Synthetic workers walking around; returns frame + boxes (ground truth stands in for a detector)."""
    def __init__(self):
        rng = np.random.default_rng(3)
        self.p = [[rng.uniform(50, 550), rng.uniform(150, 400), rng.uniform(-2.5, 2.5), rng.uniform(-1.5, 1.5)] for _ in range(3)]

    def read(self):
        f = np.full((H, W, 3), 60, np.uint8)
        boxes = []
        for p in self.p:
            p[0] += p[2]; p[1] += p[3]
            if not 30 < p[0] < W - 30: p[2] *= -1
            if not 120 < p[1] < H - 40: p[3] *= -1
            x, y = int(p[0]), int(p[1])
            cv2.rectangle(f, (x - 18, y - 70), (x + 18, y), (200, 160, 120), -1)
            cv2.circle(f, (x, y - 82), 12, (220, 190, 160), -1)
            boxes.append((x - 18, y - 94, 36, 94))
        return True, f, boxes


class Detector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def __call__(self, frame):
        rects, _ = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [tuple(map(int, r)) for r in rects]


class Tracker:
    def __init__(self): self.t, self.next = {}, 1

    def update(self, boxes):
        out, used = [], set()
        for b in boxes:
            c = (b[0] + b[2] / 2, b[1] + b[3])
            best, bd = None, 90
            for i, (pc, _) in self.t.items():
                d = math.dist(c, pc)
                if i not in used and d < bd: best, bd = i, d
            if best is None: best = self.next; self.next += 1
            used.add(best); out.append((best, b, c))
        self.t = {i: (c, time.time()) for i, b, c in out}
        return out


def serve(port):
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"), "rb").read()

    class H_(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                with LOCK: body, ct = STATE["jpg"], "image/jpeg"
            elif self.path.startswith("/api/state"):
                with LOCK: body, ct = json.dumps({"incidents": STATE["incidents"][-50:], "stats": STATE["stats"]}).encode(), "application/json"
            else: body, ct = page, "text/html"
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)
    ThreadingHTTPServer(("0.0.0.0", port), H_).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds inside zone before incident")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mqtt", default="", help="MQTT broker host (optional, needs paho-mqtt)")
    ap.add_argument("--topic", default="sentinel/incidents")
    a = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    log = open("out/incidents.jsonl", "a", buffering=1)
    mq = None
    if a.mqtt:
        import paho.mqtt.client as mqtt
        mq = mqtt.Client(); mq.connect(a.mqtt, 1883); mq.loop_start()

    synth = a.source == "synthetic"
    src = Synth() if synth else cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    det = None if synth else Detector()
    trk, entered, fired = Tracker(), {}, set()
    STATE["stats"]["mode"] = "SYNTHETIC (ground-truth boxes)" if synth else "OpenCV HOG on CPU"
    threading.Thread(target=serve, args=(a.port,), daemon=True).start()
    print(f"Dashboard: http://localhost:{a.port}   (Ctrl+C to stop)")
    n, t0 = 0, time.time()
    try:
        while True:
            if synth: ok, frame, boxes = src.read()
            else:
                ok, frame = src.read()
                if not ok:
                    if isinstance(a.source, str) and not a.source.isdigit(): src.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                    break
                frame = cv2.resize(frame, (W, H)); boxes = det(frame)
            cv2.polylines(frame, [ZONE], True, (0, 0, 255), 2)
            ov = frame.copy(); cv2.fillPoly(ov, [ZONE], (0, 0, 255)); frame = cv2.addWeighted(ov, 0.2, frame, 0.8, 0)
            cv2.putText(frame, "HAZARD ZONE", (ZONE[0][0] + 6, ZONE[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            now, inzone = time.time(), 0
            tracks = trk.update(boxes)
            for i, b, c in tracks:
                inside = cv2.pointPolygonTest(ZONE, (float(c[0]), float(c[1])), False) >= 0
                col = (0, 0, 255) if inside else (0, 200, 0)
                cv2.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), col, 2)
                cv2.putText(frame, f"ID{i}", (b[0], b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                if inside:
                    inzone += 1; entered.setdefault(i, now)
                    if now - entered[i] >= a.dwell and i not in fired:
                        fired.add(i)
                        ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": "HAZARD_ZONE_INTRUSION", "track_id": i,
                              "dwell_s": round(now - entered[i], 2), "bbox": list(b)}
                        log.write(json.dumps(ev) + "\n")
                        if mq: mq.publish(a.topic, json.dumps(ev))
                        with LOCK: STATE["incidents"].append(ev)
                        print("ALERT", ev)
                else:
                    entered.pop(i, None); fired.discard(i)
            n += 1
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with LOCK:
                STATE["jpg"] = jpg.tobytes()
                STATE["stats"].update(frames=n, people=len(tracks), in_zone=inzone, fps=round(n / (time.time() - t0), 1))
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
