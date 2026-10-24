"""Pretend to be the ESP32 motor controller, to dry-run DrainGuard without moving any hardware.

    python fake_motor.py            # listens on 127.0.0.1:8082 and prints every command it receives

Then put in motor.txt:   http://127.0.0.1:8082

It also imitates the board's water-level AUTO mode ("/auto" only from CLOSED). Fake the water:
    http://127.0.0.1:8082/_level?x=1   water rises -> in AUTO the cover opens by itself
    http://127.0.0.1:8082/_level?x=0   water falls  -> in AUTO the cover closes by itself
(The /_level URL exists only in this fake, not on the real board. The AUTO status text is a guess at the real format.)
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MOVE_S = 1.5  # how long the fake board takes to open or close by itself


class Board:
    def __init__(self):
        self.motion, self.mode, self.pos = "STOPPED", "MANUAL", "CLOSED"
        self.moving_to = None
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            if self.mode == "AUTO":
                pos = {"OPEN": "OPENING", "CLOSED": "CLOSING"}.get(self.moving_to, self.pos) if self.moving_to else self.pos
                return f"{self.motion} | AUTO | {pos}"
            return f"{self.motion} | MANUAL"

    def _finish(self, target):
        with self.lock:
            if self.moving_to == target:
                self.motion, self.pos, self.moving_to = "STOPPED", target, None

    def move_by_itself(self, target):
        with self.lock:
            self.motion, self.moving_to = ("FORWARD" if target == "OPEN" else "REVERSE"), target
        threading.Timer(MOVE_S, self._finish, args=(target,)).start()


def make_handler(board):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, text):
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            stamp = time.strftime("%H:%M:%S")
            if u.path == "/status":
                return self.reply(200, board.status())
            if u.path == "/auto":
                with board.lock:
                    ok = board.pos == "CLOSED"
                    if ok:
                        board.mode = "AUTO"
                print(f"[{stamp}] /auto -> {'water-level AUTO ON' if ok else 'REFUSED (cover is not closed)'}")
                return self.reply(200 if ok else 400, "OK" if ok else "Only from closed")
            if u.path == "/_level":
                rising = q.get("x", ["0"])[0] == "1"
                with board.lock:
                    act = board.mode == "AUTO" and board.moving_to is None and board.pos == ("CLOSED" if rising else "OPEN")
                print(f"[{stamp}] water {'RISES' if rising else 'FALLS'} -> " + ("cover moves by itself" if act else "nothing (not in AUTO / already there)"))
                if act:
                    board.move_by_itself("OPEN" if rising else "CLOSED")
                return self.reply(200, "OK")
            if u.path in ("/timed", "/forward", "/reverse", "/stop", "/"):
                with board.lock:
                    board.mode, board.moving_to = "MANUAL", None  # manual controls cancel AUTO
                    if u.path == "/stop":
                        board.motion = "STOPPED"
                if u.path == "/timed":
                    d, ms = q.get("dir", ["?"])[0], q.get("ms", ["?"])[0]
                    print(f"[{stamp}] RUN {d} for {ms} ms (stops by itself); AUTO cancelled")
                    try:
                        end = "OPEN" if d == "forward" else "CLOSED"
                        threading.Timer(int(ms) / 1000.0, lambda: setattr(board, "pos", end)).start()
                    except ValueError:
                        pass
                elif u.path in ("/forward", "/reverse"):
                    print(f"[{stamp}] RUN {u.path[1:]} CONTINUOUSLY (nothing stops it except /stop)")
                else:
                    print(f"[{stamp}] {u.path}")
                return self.reply(200, "OK")
            self.reply(404, "not found")

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8082)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Board()))
    srv.daemon_threads = True
    print(f"Fake motor on http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
