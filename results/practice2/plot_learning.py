"""Extract TensorBoard curves and verify them against notebook output."""
from pathlib import Path
import csv
import json
import re
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = {
    "baseline": "baseline.ipynb",
    "max_iters_500": "max_iters_500.ipynb",
    "box_target_strong": "box_target_strong.ipynb",
    "qpos_strong": "qpos_strong.ipynb",
    "learning_rate_3e4": "learning_rate_3e4.ipynb",
}
curves = {}

def varint(data, pos):
    value, shift = 0, 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7

def fields(data):
    pos = 0
    while pos < len(data):
        key, pos = varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = varint(data, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value, pos = data[pos:pos+size], pos+size
        elif wire == 2:
            size, pos = varint(data, pos)
            value, pos = data[pos:pos+size], pos+size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value

def event_points(run_dir):
    values = {}
    for path in sorted(run_dir.glob("events*")):
        with path.open("rb") as stream:
            while header := stream.read(12):
                if len(header) != 12:
                    print(f"{run_dir.name}: ignored incomplete trailing event header")
                    break
                size = struct.unpack("<Q", header[:8])[0]
                payload = stream.read(size)
                if len(payload) != size or len(stream.read(4)) != 4:
                    print(f"{run_dir.name}: ignored incomplete trailing event record")
                    break
                event = list(fields(payload))
                step = next((v for n, w, v in event if n == 2 and w == 0), 0)
                for n, w, summary in event:
                    if n != 5 or w != 2:
                        continue
                    for sn, sw, item in fields(summary):
                        if sn != 1 or sw != 2:
                            continue
                        entry = list(fields(item))
                        tag = next((v.decode() for k, t, v in entry if k == 1 and t == 2), "")
                        scalar = next((struct.unpack("<f", v)[0] for k, t, v in entry if k == 2 and t == 5), None)
                        if tag == "Train/mean_reward" and scalar is not None:
                            values[step] = scalar
    return sorted(values.items())

with (ROOT / "learning-curves.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["experiment", "iteration", "mean_reward"])
    for name, filename in RUNS.items():
        points = event_points(ROOT / "events" / name)
        if not points:
            raise ValueError(f"No Train/mean_reward scalars in {name}")
        notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        content = "\n".join(
            "".join(output.get("text", []))
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        )
        printed = {}
        for part in re.split(r"Learning iteration\s+", content)[1:]:
            iteration = re.match(r"(\d+)/\d+", part)
            reward = re.search(r"Mean reward:\s+([-+\d.eE]+)", part)
            if iteration and reward:
                printed[int(iteration[1])] = float(reward[1])
        matched = [(step, value) for step, value in points if step in printed]
        assert matched, f"No printed overlap in {name}"
        assert all(abs(value-printed[step]) < 0.011 for step, value in matched), name
        for point in points:
            writer.writerow([name, *point])
        curves[name] = points

fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
for ax, (name, points) in zip(axes.flat, curves.items()):
    xs, ys = zip(*points)
    ax.plot(xs, ys, alpha=0.35, linewidth=0.7, label="Raw reward")
    window = 25
    smooth = [sum(ys[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(ys))]
    ax.plot(xs, smooth, linewidth=1.5, label="25-point trailing mean")
    ax.set(title=name, xlabel="Iteration", ylabel="Training mean_reward")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
for unused in list(axes.flat)[len(curves):]:
    unused.set_visible(False)
fig.suptitle("Training curves — reward weights differ between experiments")
fig.savefig(ROOT / "learning-curves.png", dpi=150)
plt.close(fig)
for name, points in curves.items():
    tail = [v for _, v in points[-50:]]
    print(f"{name}: {len(points)} event points; last_event={points[-1]}; last50 mean={sum(tail)/len(tail):.2f}; range={min(tail):.2f}..{max(tail):.2f}")
