"""Synthetic fan-out demo pipeline: generate N noisy periodic series ->
per-item clean -> per-item analyze -> aggregate -> figure (PNG bytes).

Used by flowr's verification tests. Test hooks (all optional, none affect
cache keys — they live in the environment, not in stage parameters):

  FLOWR_DEMO_COUNTER=path   every stage execution appends one line to path
                            (side-effect counter proving what actually ran)
  FLOWR_DEMO_KILL_AFTER=K   hard-exit (os._exit) the process the moment a
                            stage would start after K executions have
                            completed — simulates a mid-run crash

Run:  python examples/demo_pipeline.py --n 20 --workers 4 --out fig.png
"""

import argparse
import math
import os
import random
import struct
import sys
import zlib

import flowr


def _tick():
    """Side-effect counter + crash injection. Not a stage; not hashed."""
    counter = os.environ.get("FLOWR_DEMO_COUNTER")
    kill_after = os.environ.get("FLOWR_DEMO_KILL_AFTER")
    done = 0
    if counter and os.path.exists(counter):
        with open(counter, "rb") as f:
            done = f.read().count(b"\n")
    if kill_after is not None and done >= int(kill_after):
        os._exit(1)
    if counter:
        fd = os.open(counter, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, b"x\n")
        finally:
            os.close(fd)


@flowr.stage
def generate(i, n_points=240, seed=42):
    """One noisy periodic series with a linear trend, seeded per item."""
    _tick()
    rng = random.Random(seed * 1_000_003 + i)
    period = 2.5 + (i % 7)
    return [
        math.sin(2 * math.pi * t / period) * 0.6
        + rng.gauss(0.0, 0.3)
        + 0.002 * t
        for t in range(n_points)
    ]


@flowr.stage
def clean(series, window=9):
    """Remove the running-median-ish local mean (crude detrend)."""
    _tick()
    half = window // 2
    out = []
    for idx in range(len(series)):
        lo = max(0, idx - half)
        hi = min(len(series), idx + half + 1)
        out.append(series[idx] - sum(series[lo:hi]) / (hi - lo))
    return out


@flowr.stage
def analyze(series, min_period=2.0, max_period=10.0, n_trials=64):
    """Toy periodogram: best sine-fit power over a period grid."""
    _tick()
    bias = 0.0
    best_power, best_period = 0.0, min_period
    for k in range(n_trials):
        p = min_period + (max_period - min_period) * k / (n_trials - 1)
        c = sum(series[t] * math.sin(2 * math.pi * t / p) for t in range(len(series)))
        s = sum(series[t] * math.cos(2 * math.pi * t / p) for t in range(len(series)))
        power = (c * c + s * s) / len(series) + bias
        if power > best_power:
            best_power, best_period = power, p
    return {"power": best_power, "period": best_period}


@flowr.stage
def aggregate(results, top_k=5):
    """Rank the per-item detections and keep the strongest."""
    _tick()
    ranked = sorted(results, key=lambda r: -r["power"])
    return {
        "top": ranked[:top_k],
        "n": len(results),
        "mean_power": sum(r["power"] for r in results) / len(results),
    }


@flowr.stage
def figure(summary, width=320, height=200):
    """Bar chart of the top detection powers, returned as PNG bytes."""
    _tick()
    top = summary["top"]
    px = bytearray(b"\xff" * (width * height * 3))

    def put(x, y, r, g, b):
        if 0 <= x < width and 0 <= y < height:
            o = (y * width + x) * 3
            px[o:o + 3] = bytes((r, g, b))

    peak = max((t["power"] for t in top), default=1.0) or 1.0
    n = max(len(top), 1)
    bar_w = width // (n * 2)
    for j, t in enumerate(top):
        h = int((t["power"] / peak) * (height - 20))
        x0 = 10 + j * 2 * bar_w
        for x in range(x0, x0 + bar_w):
            for y in range(height - 10 - h, height - 10):
                put(x, y, 40, 90, 200)
    return _png(width, height, bytes(px))


def _png(w, h, rgb):
    """Minimal deterministic PNG encoder (stdlib only)."""
    raw = b"".join(
        b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h)
    )

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def build(n, top_k=5, n_points=240):
    series = flowr.map(generate, list(range(n)), n_points=n_points)
    cleaned = flowr.map(clean, series)
    analyzed = flowr.map(analyze, cleaned)
    agg = aggregate(analyzed, top_k=top_k)
    return figure(agg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20, help="number of series")
    ap.add_argument("--n-points", type=int, default=240)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--out", help="write the figure PNG here")
    ap.add_argument("--counter", help="side-effect counter file (sets FLOWR_DEMO_COUNTER)")
    ap.add_argument("--kill-after", type=int, default=None,
                    help="os._exit the process after K stage executions")
    args = ap.parse_args(argv)

    if args.counter:
        os.environ["FLOWR_DEMO_COUNTER"] = args.counter
    if args.kill_after is not None:
        os.environ["FLOWR_DEMO_KILL_AFTER"] = str(args.kill_after)

    fig = build(args.n, top_k=args.top_k, n_points=args.n_points)

    if args.dry:
        plan = flowr.run(fig, dry=True)
        print(plan)
        print(f"HITS={plan.n_hits} MISSES={plan.n_misses}")
        return 0

    png = flowr.run(fig, workers=args.workers)
    if args.out:
        with open(args.out, "wb") as f:
            f.write(png)
    print(f"ok: figure is {len(png)} bytes of PNG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
