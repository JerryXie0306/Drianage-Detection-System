"""Check that a camera is reachable BEFORE starting the app.

    python check_camera.py                                   # uses the first line of camera.txt
    python check_camera.py snapshot:http://192.168.1.50/capture
    python check_camera.py http://192.168.1.50:81/stream

Saves camera_test.jpg so you can see what the camera sees.
"""

import io
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
HINTS = """What to check / 请逐项检查:
  1. Laptop and camera on the SAME Wi-Fi. Campus Wi-Fi (eduroam) often blocks device-to-device traffic: use a phone hotspot.
     电脑和摄像头要在同一个 Wi-Fi。校园网常常屏蔽设备互访，请改用手机热点。
  2. ESP32 only works on 2.4 GHz Wi-Fi (iPhone hotspot: turn ON "Maximise Compatibility").
     ESP32 只支持 2.4GHz（iPhone 热点请打开"最大化兼容性"）。
  3. Is the IP right? Read it in the Arduino Serial Monitor (115200 baud), or press RST and read it again.
     IP 对吗？在 Arduino 串口监视器（115200）里看，或按 RST 重新看。
  4. Open the address in a browser. If the page loads there, close that tab (the camera serves one viewer at a time).
     用浏览器能打开吗？能的话把那个页面关掉（摄像头同一时间只服务一个观看者）。
  5. Stream needs port 81 (http://IP:81/stream); one picture is http://IP/capture. 视频流是 81 端口。
  6. Power: use a good cable / 5 V supply; a weak USB port makes the ESP32-CAM reboot. 供电不足会反复重启。"""


def source_from_file():
    p = ROOT / "camera.txt"
    for line in p.read_text().splitlines() if p.exists() else []:
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return "0"


def check_snapshot(url):
    sizes = []
    for i in range(3):
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=8) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data))
        img.load()
        print(f"  picture {i + 1}: {img.size[0]}x{img.size[1]}  {len(data) // 1024} KB  {time.time() - t0:.2f} s")
        sizes.append(img)
        time.sleep(0.5)
    return sizes[-1]


def check_cv(src):
    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV is not installed, so streams/webcams cannot be read.\n"
                         "Run install_ai (or use snapshot:http://IP/capture, which needs no OpenCV).")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError("cannot open the stream")
    frame = None
    for i in range(3):
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("opened, but no frames arrive")
        print(f"  frame {i + 1}: {frame.shape[1]}x{frame.shape[0]}  {time.time() - t0:.2f} s")
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else source_from_file()
    print(f"Checking: {src}")
    try:
        if src.startswith("snapshot:"):
            img = check_snapshot(src[len("snapshot:"):])
        elif src.startswith("folder:"):
            files = [p for p in Path(src[7:]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if not files:
                raise RuntimeError("the folder has no images yet")
            img = Image.open(max(files, key=lambda p: p.stat().st_mtime))
        else:
            img = check_cv(src)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFAIL: {e}\n\n{HINTS}")
        raise SystemExit(1)
    out = ROOT / "camera_test.jpg"
    img.convert("RGB").save(out)
    print(f"\nPASS: camera works. A test picture was saved to {out.name}. 连接成功，已保存测试图。")


if __name__ == "__main__":
    main()
