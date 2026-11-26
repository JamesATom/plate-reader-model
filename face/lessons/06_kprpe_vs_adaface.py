"""Face lesson 06: KP-RPE (MSU, CVPR 2024) against AdaFace. Does knowing the keypoints help?

KP-RPE (Kim, Su, Liu, Jain, Liu, "KeyPoint Relative Position Encoding for Face Recognition",
CVPR 2024) is a ViT that is told where the 5 face keypoints are inside the aligned crop, so
a badly aligned face hurts it less. How it works: see vit_kprpe.py.

The models (all from the same CVLFace authors, all trained with the AdaFace loss):
  AdaFace    IR-101 (CNN)          WebFace12M   the recognizer our app uses (lessons 04, 05)
  KP-RPE     ViT-Base + KP-RPE     WebFace12M   same data, same loss -> the fair head-to-head
  Ada_4M     IR-101                WebFace4M  \
  ViT_4M     ViT-Base, no KP-RPE   WebFace4M   > controls: is a gain the keypoints, or just the ViT?
  KPRPE_4M   ViT-Base + KP-RPE     WebFace4M  /
  KP-RPE*    the KP-RPE model, but always told "keypoints are on the template" (we hide the truth)

Keypoints: YuNet's 5 points (lesson 01), moved into the 112x112 crop. The paper used its own
landmark net (DFA); we keep our detector, like a real deployment would.

Tests:
  bench    LFW (easy), CPLFW (big pose changes), XQLFW (one photo of each pair is low quality).
           6000 pairs each, the standard 10-fold accuracy. Then the same pairs again with one face
           of each pair misaligned on purpose (shifted / rotated / zoomed), keypoints moved along.
  P1E_S1   ChokePoint ceiling cameras, indoor: same test as lesson 05.
  P2E_S1   ChokePoint outdoor, other day; also with the thresholds chosen on P1E_S1.

Data:   face/data/bench/{lfw,cplfw,xqlfw}_bin/shard-*.tar
          from https://huggingface.co/datasets/LSIbabnikz/{lfw,cplfw,xqlfw}_bin (aligned 112x112 PNG pairs)
        face/data/chokepoint/   as in lesson 05
Models: uv run python face/download_models.py

Run:  uv run python face/lessons/06_kprpe_vs_adaface.py bench
      uv run python face/lessons/06_kprpe_vs_adaface.py speed        (close other GPU programs first)
      uv run python face/lessons/06_kprpe_vs_adaface.py P1E_S1
      uv run python face/lessons/06_kprpe_vs_adaface.py P2E_S1       (after P1E_S1)
      add --rebuild to recompute the cached embeddings
"""

import importlib
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "face/lessons")
lesson02 = importlib.import_module("02_align_faces")
from ir_resnet import load_adaface  # noqa: E402
from vit_kprpe import load_cvlface_vit  # noqa: E402

torch.set_grad_enabled(False)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path("face/output")
TEMPLATE_KP = (lesson02.TEMPLATE / 112).astype(np.float32)

# name -> (weights file, network type)
MODELS = {
    "AdaFace": ("face/models/adaface_ir101/model.safetensors", "ir101"),
    "KP-RPE": ("face/models/kprpe_vit_base_webface12m/model.safetensors", "kprpe"),
    "Ada_4M": ("face/models/adaface_ir101_webface4m/model.safetensors", "ir101"),
    "ViT_4M": ("face/models/vit_base_webface4m/model.safetensors", "vit"),
    "KPRPE_4M": ("face/models/kprpe_vit_base_webface4m/model.safetensors", "kprpe"),
    "KP-RPE*": ("face/models/kprpe_vit_base_webface12m/model.safetensors", "kprpe_hidden"),
}
_loaded = {}


def network(name):
    path, kind = MODELS[name]
    if path not in _loaded:
        _loaded[path] = load_adaface(path) if kind == "ir101" else load_cvlface_vit(path, kprpe=kind != "vit")
    return _loaded[path]


def _to_tensor(faces_bgr):
    """(N, 112, 112, 3) BGR uint8 -> RGB float tensor in -1..1 (what all these models were trained on)."""
    rgb = np.ascontiguousarray(np.asarray(faces_bgr)[..., ::-1], dtype=np.float32) / 255.0
    return torch.from_numpy((rgb - 0.5) / 0.5).permute(0, 3, 1, 2)


def embed(name, faces, keypoints, batch=128):
    """Aligned faces + their keypoints (N, 5, 2 in 0..1) -> (N, 512) unit vectors."""
    kind = MODELS[name][1]
    net = network(name).to(DEVICE)
    if kind == "kprpe_hidden":
        keypoints = np.broadcast_to(TEMPLATE_KP, (len(faces), 5, 2))
    out = []
    for i in range(0, len(faces), batch):
        x = _to_tensor(faces[i:i + batch]).to(DEVICE)
        if kind in ("kprpe", "kprpe_hidden"):
            kp = torch.from_numpy(np.ascontiguousarray(keypoints[i:i + batch], dtype=np.float32)).to(DEVICE)
            out.append(net(x, kp).cpu().numpy())
        else:
            out.append(net(x).cpu().numpy())
    net.to("cpu")
    v = np.concatenate(out).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# =====================================================================================
# Part 1: benchmarks (pairs of already aligned 112x112 faces)
# =====================================================================================
BENCHMARKS = ["lfw", "cplfw", "xqlfw"]


def load_pairs(bench):
    """-> img1 (N,112,112,3) BGR, img2, same (N,) bool, fold (N,) 0..9.

    Pairs are stored as genuine/ (same person) and imposter/, both in the original order of
    the official pair list. That list has 10 folds of 300 same + 300 different pairs.
    """
    parts = defaultdict(dict)
    for shard in sorted(Path(f"face/data/bench/{bench}_bin").glob("shard-*.tar")):
        with tarfile.open(shard) as tar:  # read in memory, nothing is unpacked to disk
            for m in tar:
                if m.isfile():
                    key, ext = m.name.rsplit(".", 1)
                    parts[key][ext] = tar.extractfile(m).read()
    img1, img2, same, fold = [], [], [], []
    position = defaultdict(int)
    for key in sorted(parts):
        p = parts[key]
        decode = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
        img1.append(decode(p["img1"]))
        img2.append(decode(p["img2"]))
        kind = key.split("/")[0]
        same.append(kind == "genuine")
        fold.append(position[kind] // 300)
        position[kind] += 1
    return np.array(img1), np.array(img2), np.array(same), np.array(fold)


def keypoints_in_crop(faces):
    """5 YuNet points of already aligned 112x112 faces, as 0..1. Template if YuNet finds nothing."""
    pad = 56  # YuNet expects some space around a face
    kps, missed = [], 0
    for f in faces:
        found = lesson02.find_faces(cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT))
        if found:
            kps.append((found[0][1] - pad) / 112)
        else:
            kps.append(TEMPLATE_KP)
            missed += 1
    return np.array(kps, dtype=np.float32), missed


def tenfold_accuracy(scores, same, fold):
    """Standard protocol: choose the threshold on 9 folds, measure on the 10th, average."""
    thresholds = np.arange(-1, 1, 0.002)
    accs = []
    for k in range(10):
        train, test = fold != k, fold == k
        acc_train = ((scores[train][None, :] >= thresholds[:, None]) == same[train][None, :]).mean(axis=1)
        thr = thresholds[acc_train.argmax()]
        accs.append(((scores[test] >= thr) == same[test]).mean())
    return float(np.mean(accs))


# how much we misalign one face of every pair: (shift in pixels, rotation in degrees, zoom)
MISALIGN = {"none": (0, 0, 0), "strong": (12, 15, 0.15), "extreme": (16, 20, 0.20)}


def misalign(faces, keypoints, shift, degrees, zoom, seed=0):
    """Random shift/rotate/zoom of every face (same randomness for every model), keypoints moved along."""
    rng = np.random.default_rng(seed)
    out_faces, out_kp = [], []
    for f, kp in zip(faces, keypoints):
        m = cv2.getRotationMatrix2D((56, 56), rng.uniform(-degrees, degrees), 1 + rng.uniform(-zoom, zoom))
        m[:, 2] += rng.uniform(-shift, shift, 2)
        out_faces.append(cv2.warpAffine(f, m, (112, 112), borderMode=cv2.BORDER_REPLICATE))
        out_kp.append((kp * 112 @ m[:, :2].T + m[:, 2]) / 112)
    return np.array(out_faces), np.array(out_kp, dtype=np.float32)


CACHE = OUT / "face_06_bench"  # one file per (faces, model): a stopped run continues where it was


def cached(file, compute, rebuild):
    path = CACHE / file
    if path.exists() and not rebuild:
        return np.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    value = compute()
    np.save(path, value)
    return value


def speed_ms(name, faces, kps, n=256):
    """Milliseconds per face, batches of 128 (second run, so loading is not counted)."""
    embed(name, faces[:n], kps[:n])
    t0 = time.perf_counter()
    embed(name, faces[:n], kps[:n])
    return (time.perf_counter() - t0) / n * 1000


def run_benchmarks(rebuild):
    res = {"acc": {}, "misalign": {}, "kp_missed": {}, "ms": {}}
    for bench in BENCHMARKS:
        t0 = time.perf_counter()
        img1, img2, same, fold = load_pairs(bench)
        kp1 = cached(f"{bench}_kp1.npy", lambda: keypoints_in_crop(img1)[0], rebuild)
        kp2 = cached(f"{bench}_kp2.npy", lambda: keypoints_in_crop(img2)[0], rebuild)
        # a face where YuNet found nothing got exactly the template
        missed = int((np.abs(np.concatenate([kp1, kp2]) - TEMPLATE_KP).sum(axis=(1, 2)) == 0).sum())
        res["kp_missed"][bench] = (missed, 2 * len(same))
        print(f"{bench}: {len(same)} pairs, YuNet keypoints missing on {missed} of {2 * len(same)} faces "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        for name in MODELS:
            t1 = time.perf_counter()
            e1 = cached(f"{bench}_img1_{name}.npy", lambda: embed(name, img1, kp1), rebuild)
            e2 = cached(f"{bench}_img2_{name}.npy", lambda: embed(name, img2, kp2), rebuild)
            res["acc"][(bench, name)] = tenfold_accuracy((e1 * e2).sum(1), same, fold)
            print(f"  {name:<9} {res['acc'][(bench, name)]:.2%}   ({time.perf_counter() - t1:.0f}s)", flush=True)
            if bench == "lfw":
                res["ms"][name] = speed_ms(name, img1, kp1)
            for level, (s, d, z) in MISALIGN.items():
                if level == "none":
                    res["misalign"][(bench, level, name)] = res["acc"][(bench, name)]
                    continue
                bad = lambda: embed(name, *misalign(img2, kp2, s, d, z))  # noqa: E731
                e_bad = cached(f"{bench}_img2_{level}_{name}.npy", bad, rebuild)
                res["misalign"][(bench, level, name)] = tenfold_accuracy((e1 * e_bad).sum(1), same, fold)
            print("            misaligned: " + "  ".join(f"{lv} {res['misalign'][(bench, lv, name)]:.2%}" for lv in MISALIGN),
                  flush=True)
        if bench == "xqlfw":
            save_misalign_picture(img2, kp2)

    print("\n10-fold verification accuracy (6000 pairs each; in brackets: wrong pairs out of 6000)")
    print(f"{'model':<9} " + "".join(f"{b.upper():>17}" for b in BENCHMARKS) + f"{'ms/face':>10}")
    for name in MODELS:
        cells = "".join(f"{res['acc'][(b, name)]:>11.2%} ({(1 - res['acc'][(b, name)]) * 6000:>3.0f})" for b in BENCHMARKS)
        print(f"{name:<9} {cells}{res['ms'][name]:>10.1f}")
    print("\nSame pairs, one face of every pair misaligned on purpose. Levels (shift px, rotate deg, zoom): "
          + ", ".join(f"{lv} {MISALIGN[lv]}" for lv in MISALIGN if lv != "none"))
    print(f"{'model':<9} " + "".join(f"{b.upper() + ' ' + lv:>17}" for b in BENCHMARKS for lv in MISALIGN))
    for name in MODELS:
        print(f"{name:<9} " + "".join(f"{res['misalign'][(b, lv, name)]:>17.2%}" for b in BENCHMARKS for lv in MISALIGN))
    missed = ", ".join(f"{b} {m}/{n}" for b, (m, n) in res["kp_missed"].items())
    print(f"\nYuNet found no keypoints (template used instead) on: {missed}")
    print("KP-RPE* = the KP-RPE network with the keypoints hidden (always the template)")


def save_misalign_picture(faces, kps, n=6):
    rows = []
    for level, (s, d, z) in MISALIGN.items():
        f, k = misalign(faces[:n], kps[:n], s, d, z)
        tiles = []
        for face, kp in zip(f, k):
            t = cv2.resize(face, (168, 168))
            for x, y in kp * 168:
                cv2.circle(t, (int(x), int(y)), 3, (0, 255, 0), -1)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(OUT / "face_06_misaligned.png"), np.vstack(rows))
    print(f"  saved {OUT / 'face_06_misaligned.png'} (XQLFW faces; rows: {', '.join(MISALIGN)}; green = keypoints given to KP-RPE)")


# =====================================================================================
# Part 1b: speed (only meaningful when nothing else is using the GPU)
# =====================================================================================
def _timed(fn, repeats):
    fn()  # warm-up: first call compiles kernels / allocates memory
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_speed():
    rng = np.random.default_rng(0)
    faces = rng.integers(0, 255, (128, 112, 112, 3), dtype=np.uint8)
    kps = np.broadcast_to(TEMPLATE_KP, (128, 5, 2)).copy()
    print(f"{'model':<9} {'params':>8} {'file':>8} {'GPU batch 128':>15} {'GPU 1 face':>11} {'CPU 1 face':>11}")
    for name in ["AdaFace", "KP-RPE"]:
        net = network(name)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        size = Path(MODELS[name][0]).stat().st_size / 1e6
        kp_args = MODELS[name][1] != "ir101"

        def run(device, n):
            x = _to_tensor(faces[:n]).to(device)
            args = (x, torch.from_numpy(kps[:n]).to(device)) if kp_args else (x,)
            return lambda: net(*args).cpu()

        net.to(DEVICE)
        batch_s = _timed(run(DEVICE, 128), 10)
        one_gpu = _timed(run(DEVICE, 1), 30)
        net.to("cpu")
        one_cpu = _timed(run("cpu", 1), 10)
        print(f"{name:<9} {params:>7.0f}M {size:>6.0f}MB {128 / batch_s:>9.0f} face/s {one_gpu * 1000:>8.1f} ms {one_cpu * 1000:>8.1f} ms")
    print(f"(GPU = Apple '{DEVICE}', median of repeated runs, after a warm-up run. KP-RPE also needs the 5 keypoints, "
          "but YuNet already gives them with every detection, so they cost nothing extra here.)")


# =====================================================================================
# Part 2: ChokePoint surveillance video (same test as lesson 05, other models)
# =====================================================================================
def run_chokepoint(sequence, rebuild):
    lesson05 = importlib.import_module("05_chokepoint_test")
    cache = OUT / f"face_06_embeddings_{sequence}.npz"
    if rebuild or not cache.exists():
        build_chokepoint(lesson05, sequence, cache)
    reference = OUT / "face_06_embeddings_P1E_S1.npz"
    fixed = lesson05.evaluate(reference, quiet=True) if sequence != "P1E_S1" and reference.exists() else None
    lesson05.evaluate(cache, fixed)


def build_chokepoint(lesson05, sequence, cache):
    root = lesson05.ROOT
    gallery_ids, g_faces, g_kp = [], [], []
    for path in sorted((root / "Neutral").glob("ID*.JPG")):
        pid = re.search(r"ID(\d+)", path.name).group(1)
        if pid in lesson05.UNKNOWN_IDS:
            continue
        img = cv2.imread(str(path))
        img = cv2.resize(img, None, fx=1200 / img.shape[0], fy=1200 / img.shape[0], interpolation=cv2.INTER_AREA)
        faces = lesson02.find_faces(img)
        if not faces:
            print(f"  enrol {path.name}: no face found")
            continue
        face, kp = lesson02.align_with_keypoints(img, faces[0][1])
        gallery_ids.append(pid)
        g_faces.append(face)
        g_kp.append(kp)
    print(f"enrolled {len(gallery_ids)} people; {len(lesson05.UNKNOWN_IDS)} kept out as 'unknown'")

    cameras = sorted(p.stem.removeprefix(sequence + "_") for p in (root / "groundtruth").glob(f"{sequence}_C*.xml"))
    meta, p_faces, p_kp, eye_dists, missed = [], [], [], [], defaultdict(int)
    t0 = time.perf_counter()
    for cam in cameras:
        labels = lesson05.read_groundtruth(root / "groundtruth" / f"{sequence}_{cam}.xml")
        by_frame = defaultdict(list)
        for row in labels:
            by_frame[row[0]].append(row)
        for frame_no, rows in by_frame.items():
            img = cv2.imread(str(root / f"{sequence}_{cam}" / f"{frame_no}.jpg"))
            if img is None:
                missed[cam] += len(rows)
                continue
            faces = lesson02.find_faces(img)
            for _, pid, cx, cy, eye_dist in rows:
                points = lesson05.matching_face(faces, cx, cy, eye_dist)
                if points is None:
                    missed[cam] += 1
                    continue
                face, kp = lesson02.align_with_keypoints(img, points)
                p_faces.append(face)
                p_kp.append(kp)
                meta.append((cam, pid, frame_no))
                eye_dists.append(eye_dist)
        print(f"  {cam}: {len(labels)} labelled faces, detector missed {missed[cam]}   ({time.perf_counter() - t0:.0f}s)")

    g_faces, g_kp, p_faces, p_kp = np.array(g_faces), np.array(g_kp), np.array(p_faces), np.array(p_kp)
    emb = {}
    for name in MODELS:
        emb[f"gallery_{name}"] = embed(name, g_faces, g_kp)
        emb[f"probe_{name}"] = embed(name, p_faces, p_kp)
    print(f"  embedded {len(p_faces)} faces with {len(MODELS)} models   ({time.perf_counter() - t0:.0f}s)")
    OUT.mkdir(exist_ok=True)
    np.savez(cache, gallery_ids=np.array(gallery_ids), probe_meta=np.array(meta), eye_dists=np.array(eye_dists),
             missed=np.array([missed[c] for c in cameras]), **emb)


if __name__ == "__main__":
    what = next((a for a in sys.argv[1:] if not a.startswith("--")), "bench")
    if what == "bench":
        run_benchmarks("--rebuild" in sys.argv)
    elif what == "speed":
        run_speed()
    else:
        run_chokepoint(what, "--rebuild" in sys.argv)
