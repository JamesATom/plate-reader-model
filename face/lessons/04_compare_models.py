"""Face lesson 04: three recognizers, same faces, who separates people best?

Models (all take the same 112x112 aligned face from lesson 02):
  SFace     128 numbers  OpenCV, Apache-2.0           (what lesson 03 used)
  AdaFace   512 numbers  IR-101 trained on WebFace12M  (good on low-quality faces)
  EdgeFace  512 numbers  EdgeNeXt-Base, WebFace260M    (built for small devices; weights are NON-commercial)

Two tests:
  clean  every photo as it is
  cctv   one side of each pair is shrunk until the face is only CCTV_FACE_PX wide,
         like a person far from a ceiling camera. The other side stays clean (= enrolment photo).

What we look at, per model:
  same     lowest score between two photos of the SAME person   (want high)
  diff     highest score between two DIFFERENT people            (want low)
  gap      same - diff. Positive = a threshold exists that makes zero mistakes. Bigger = safer.
  errors   mistakes at the best possible threshold for that model

Run:  uv run python face/lessons/04_compare_models.py face/data/samples/*.jpg
"""

import importlib
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

sys.path.insert(0, "face/lessons")
lesson02 = importlib.import_module("02_align_faces")
from ir_resnet import load_adaface  # noqa: E402

CCTV_FACE_PX = 28
torch.set_grad_enabled(False)


# ---------- the three recognizers: aligned BGR 112x112 face -> unit-length vector ----------
_sface = cv2.FaceRecognizerSF.create("face/models/sface.onnx", "")
_adaface = load_adaface("face/models/adaface_ir101/model.safetensors")
_edgeface = timm.create_model("edgenext_base")
_edgeface.reset_classifier(512)
_sd = torch.load("face/models/edgeface_base/edgeface_base.pt", map_location="cpu", weights_only=True)
_edgeface.load_state_dict({k.removeprefix("model."): v for k, v in _sd.items()})
_edgeface.eval()


def _to_tensor(face_bgr):
    """BGR uint8 -> RGB float tensor in -1..1, the input AdaFace and EdgeFace were trained on."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy((rgb - 0.5) / 0.5).permute(2, 0, 1)[None]


def _unit(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / np.linalg.norm(v)


MODELS = {
    "SFace": lambda f: _unit(_sface.feature(f)),
    "AdaFace": lambda f: _unit(_adaface(_to_tensor(f)).numpy()),
    "EdgeFace": lambda f: _unit(_edgeface(_to_tensor(f)).numpy()),
}


def person(path):
    """'ronaldo_3.jpg' -> 'ronaldo' (the file name tells us who it is)."""
    return Path(path).stem.split("_")[0]


def shrink(img, box, face_px):
    """Make the whole photo smaller so the face is only face_px wide (simulated far-away CCTV)."""
    scale = face_px / box[2]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def score_table(vecs_a, vecs_b, names):
    """Every pair (i, j), i != j: (same_person?, similarity)."""
    out = []
    for i, j in itertools.permutations(range(len(names)), 2):
        out.append((person(names[i]) == person(names[j]), float(np.dot(vecs_a[i], vecs_b[j])), names[i], names[j]))
    return out


def summary(pairs):
    same = [s for is_same, s, *_ in pairs if is_same]
    diff = [s for is_same, s, *_ in pairs if not is_same]
    # best threshold = the one with the fewest mistakes
    best_err, best_t = len(pairs), 0.0
    for t in sorted(s for _, s, *_ in pairs):
        err = sum(s < t for s in same) + sum(s >= t for s in diff)
        if err < best_err:
            best_err, best_t = err, t
    return min(same), max(diff), min(same) - max(diff), best_err, best_t


if __name__ == "__main__":
    names, clean_faces, cctv_faces = [], [], []
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        faces = lesson02.find_faces(img)
        if not faces:
            print(f"{Path(path).name}: no face, skipped")
            continue
        box, points, _ = faces[0]
        names.append(Path(path).name)
        clean_faces.append(lesson02.align(img, points))
        small, scale = shrink(img, box, CCTV_FACE_PX)
        cctv_faces.append(lesson02.align(small, points * scale))  # same points, scaled down

    people = sorted({person(n) for n in names})
    print(f"{len(names)} photos, {len(people)} people: " + ", ".join(f"{p} x{sum(person(n) == p for n in names)}" for p in people))

    rows, hardest = [], []
    for model_name, embed in MODELS.items():
        t0 = time.perf_counter()
        clean = [embed(f) for f in clean_faces]
        ms = (time.perf_counter() - t0) / len(names) * 1000
        cctv = [embed(f) for f in cctv_faces]
        for test, pairs in [("clean", score_table(clean, clean, names)), (f"cctv {CCTV_FACE_PX}px", score_table(clean, cctv, names))]:
            rows.append((model_name, test, ms, *summary(pairs), len(pairs)))
            worst_same = min((p for p in pairs if p[0]), key=lambda p: p[1])
            worst_diff = max((p for p in pairs if not p[0]), key=lambda p: p[1])
            hardest.append((model_name, test, worst_same, worst_diff))

    print(f"\n{'model':<9} {'test':<10} {'ms/face':>7} {'same':>6} {'diff':>6} {'gap':>6} {'errors':>8} {'best thr':>9}")
    for model_name, test, ms, same, diff, gap, err, thr, n in rows:
        print(f"{model_name:<9} {test:<10} {ms:7.1f} {same:6.2f} {diff:6.2f} {gap:+6.2f} {err:>4}/{n:<3} {thr:9.2f}")

    print("\nhardest pairs (clean side first):")
    for model_name, test, ws, wd in hardest:
        print(f"  {model_name:<9} {test:<10} same person scored lowest: {ws[2]} vs {ws[3]} {ws[1]:.2f}   "
              f"different people scored highest: {wd[2]} vs {wd[3]} {wd[1]:.2f}")

    # picture: the clean and the shrunk face for every photo, so we can see what the model sees
    strip = np.vstack([np.hstack([cv2.resize(c, (112, 112)), cv2.resize(s, (112, 112))])
                       for c, s in zip(clean_faces, cctv_faces)])
    strip = np.hstack([strip[: len(strip) // 2 // 112 * 112], strip[len(strip) // 2 // 112 * 112:][: len(strip) // 2 // 112 * 112]]) \
        if len(names) % 2 == 0 else strip
    out = "face/output/face_04_clean_vs_cctv.png"
    cv2.imwrite(out, strip)
    print(f"\nsaved {out}   (each row: clean face | same face at {CCTV_FACE_PX}px)")
