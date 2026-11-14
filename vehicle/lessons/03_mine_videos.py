"""Vehicle lesson 03: turn Uzbek videos into re-ID training data, labelled by OUR plate reader.

Idea: re-ID needs "these crops are the SAME car". Nobody has labelled Uzbek videos, but our plate
reader can: two crops with the same plate text = the same car. No hand labelling.

For every video, one frame every STEP seconds:
  1. find vehicles     RF-DETR Medium, COCO-pretrained (car / bus / truck)
  2. find the plate    OUR plate detector (plate/models), run on the vehicle crop
  3. read the plate    OUR OCR ensemble + Uzbek format check (same as plate lesson 32)
  4. save the vehicle crop + plate text into a CSV
Then (summary): keep only surely-read plates, merge 1-letter misreads, count cars and crops per car,
and write the usable crops as vehicle/data/mined/reid/{plate}_{video}_{frame}.jpg (same naming as
RoundaboutHD, so lessons 01/02 can read them).

Data: vehicle/data/videos/*.mp4 (YouTube videos used with the owners' permission; internal use only,
plates are personal data: do not publish the crops).

Run:  uv run python vehicle/lessons/03_mine_videos.py                 (all videos, then summary)
      uv run python vehicle/lessons/03_mine_videos.py --summary       (summary only)
Env:  STEP=0.5  VIDEOS=cctv_mediapark,salon_nbn
"""

import csv
import importlib.util
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from PIL import Image

VIDEO_DIR = Path("vehicle/data/videos")
OUT = Path("vehicle/data/mined")
CSV = OUT / "detections.csv"
STEP = float(os.environ.get("STEP", 0.5))
VEHICLE_IDS = {3: "car", 6: "bus", 8: "truck"}   # RF-DETR returns COCO category ids
MIN_W, MIN_H = 120, 90       # smaller vehicles: plate too small to read surely
CAR_CONF, PLATE_CONF = 0.5, 0.5
OCR_SURE = 0.85              # mean per-slot OCR probability needed to trust a plate text
MIN_CROPS = 4                # a car needs >= K crops for PK training (lesson 02)
MAX_PER_CAR = 30             # cap per car, so one long-filmed car does not dominate
MAX_OVERLAP = 0.3            # crop overlaps another vehicle by more than this share -> dropped
MARGINS = [0.12, 0.16, 0.20]
TMP = OUT / "_plate.png"


def load_lesson32():
    """plate lesson 32: decode_format() + the OCR config (no code runs except loading the config)"""
    spec = importlib.util.spec_from_file_location("read_better", "plate/lessons/32_read_better.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def mine():
    from fast_plate_ocr.core.process import read_and_resize_plate_image
    from fast_plate_ocr.train.utilities.utils import load_keras_model
    from rfdetr import RFDETRMedium, RFDETRNano

    l32 = load_lesson32()
    cfg = l32.cfg
    ocr = [load_keras_model(f"plate/models/plate_ocr_{n}.keras", cfg) for n in ("cct_s", "cct_small")]
    cars = RFDETRMedium()
    plates = RFDETRNano(pretrain_weights="plate/models/plate_detector_rfdetr_nano.pth")

    wanted = os.environ.get("VIDEOS")
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if wanted:
        videos = [v for v in videos if v.stem in wanted.split(",")]
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    last = {}                                   # video -> last frame already mined (resume after a stop)
    if CSV.exists():
        for r in csv.DictReader(open(CSV)):
            last[r["video"]] = max(last.get(r["video"], -1), int(r["frame"]))
    new = not CSV.exists()
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["video", "frame", "x1", "y1", "x2", "y2", "kind", "car_conf", "plate_conf",
                    "px1", "py1", "px2", "py2", "text", "ocr_p", "crop"])

    for video in videos:
        if (OUT / f"done_{video.stem}").exists():
            print(f"{video.stem}: already mined, skipped")
            continue
        cap = cv2.VideoCapture(str(video))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps * STEP))
        (OUT / "crops" / video.stem).mkdir(exist_ok=True)
        t0, rows_v, read_v = time.perf_counter(), 0, 0
        start = last.get(video.stem, -step) + step
        if start:
            print(f"{video.stem}: resuming at {start / fps / 60:.1f} min", flush=True)
        for fno in range(start, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, bgr = cap.read()
            if not ok:
                break
            frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            det = cars.predict(frame, threshold=CAR_CONF)
            found = []                                    # (vehicle box, crop, plate box in crop or None)
            for box, cid, conf in zip(det.xyxy, det.class_id, det.confidence):
                x1, y1, x2, y2 = (int(v) for v in box)
                if int(cid) not in VEHICLE_IDS or x2 - x1 < MIN_W or y2 - y1 < MIN_H:
                    continue
                crop = frame.crop((x1, y1, x2, y2))
                p = plates.predict(crop, threshold=PLATE_CONF)
                best = None
                for pb, pc in zip(p.xyxy, p.confidence):
                    cx, cy = (pb[0] + pb[2]) / 2 / crop.width, (pb[1] + pb[3]) / 2 / crop.height
                    # the plate must be this car's: horizontally inside, lower part of the box
                    # (a neighbour's plate peeking in at the edge is skipped)
                    if 0.1 < cx < 0.9 and cy > 0.3 and (best is None or pc > best[1]):
                        best = (pb, float(pc))
                found.append(((x1, y1, x2, y2), int(cid), float(conf), crop, best))

            # read all plates of this frame in one go: 2 models x 3 margins, averaged
            to_read = [(i, b) for i, (*_, b) in enumerate(found) if b is not None]
            texts = {}
            if to_read:
                imgs = []
                for i, (pb, _) in to_read:
                    crop = found[i][3]
                    for mg in MARGINS:
                        m = mg * (pb[2] - pb[0])
                        crop.crop((pb[0] - m, pb[1] - m, pb[2] + m, pb[3] + m)).save(TMP)
                        imgs.append(read_and_resize_plate_image(TMP, cfg.img_height, cfg.img_width, cfg.image_color_mode))
                shape = (len(to_read), len(MARGINS), cfg.max_plate_slots, cfg.vocabulary_size)
                prob = np.mean([m.predict(np.stack(imgs), verbose=0)["plate"].reshape(shape).mean(1) for m in ocr], axis=0)
                for j, (i, _) in enumerate(to_read):
                    text, score = l32.decode_format(prob[j])
                    texts[i] = (text, float(np.exp(score)))

            for i, (box, cid, conf, crop, best) in enumerate(found):
                name = OUT / "crops" / video.stem / f"{fno:06d}_{i}.jpg"
                crop.save(name, quality=92)
                text, p = texts.get(i, ("", 0.0))
                pbox = [int(v) for v in best[0]] if best else ["", "", "", ""]   # plate box inside the crop
                w.writerow([video.stem, fno, *box, VEHICLE_IDS[cid], f"{conf:.3f}",
                            f"{best[1]:.3f}" if best else "", *pbox, text, f"{p:.3f}", name.name])
                rows_v += 1
                read_v += p >= OCR_SURE
            if fno // step % 100 == 0:
                print(f"  {video.stem}: {fno / fps / 60:.1f}/{n / fps / 60:.1f} min of video, "
                      f"{rows_v} vehicle crops, {read_v} surely read  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            f.flush()
        (OUT / f"done_{video.stem}").touch()
        print(f"{video.stem}: {rows_v} vehicle crops, {read_v} with a surely-read plate "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    f.close()


def one_edit(a, b):
    """a and b differ in exactly one letter (same length): a typical OCR misread"""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def reid_name(plate, r):
    """{car}_{camera}_{frame}.jpg like RoundaboutHD; "_" separates fields, so the video name uses "-",
    and the vehicle's number in the frame is glued to the frame number (frame 120, vehicle 3 -> 12003)"""
    k = int(r["crop"].split("_")[1].split(".")[0])
    return f"{plate}_{r['video'].replace('_', '-')}_{int(r['frame']) * 100 + k}.jpg"


def summary():
    rows = [r for r in csv.DictReader(open(CSV)) if r["text"] and float(r["ocr_p"]) >= OCR_SURE]
    all_rows = list(csv.DictReader(open(CSV)))
    print(f"\n{len(all_rows)} vehicle crops, {len(rows)} with a surely-read plate ({len(rows) / max(1, len(all_rows)):.0%})")

    # a misread variant (1 letter off, rarer) is merged into the common reading of the same video
    by_video = defaultdict(Counter)
    for r in rows:
        by_video[r["video"]][r["text"]] += 1
    fix = {}
    for v, cnt in by_video.items():
        for t, c in cnt.items():
            better = [u for u, cu in cnt.items() if cu > c and one_edit(t, u)]
            if better:
                fix[(v, t)] = max(better, key=lambda u: cnt[u])
    for r in rows:
        r["text"] = fix.get((r["video"], r["text"]), r["text"])

    # drop crops that overlap another vehicle a lot: the crop may show the neighbour car too
    boxes = defaultdict(list)
    for r in all_rows:
        boxes[(r["video"], r["frame"])].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])

    def crowded(r):
        a = [int(r[k]) for k in ("x1", "y1", "x2", "y2")]
        for b in boxes[(r["video"], r["frame"])]:
            if b == a:
                continue
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if inter > MAX_OVERLAP * (a[2] - a[0]) * (a[3] - a[1]):
                return True
        return False

    before = len(rows)
    rows = [r for r in rows if not crowded(r)]
    print(f"dropped {before - len(rows)} crops that overlap another vehicle by > {MAX_OVERLAP:.0%}")

    per_car = defaultdict(list)
    for r in rows:
        per_car[r["text"]].append(r)
    # one car must not take over (a blogger talking 5 minutes about one car): keep <= MAX_PER_CAR crops,
    # spread evenly over time so different angles survive
    for t, rs in per_car.items():
        rs.sort(key=lambda r: (r["video"], int(r["frame"])))
        if len(rs) > MAX_PER_CAR:
            per_car[t] = [rs[int(i)] for i in np.linspace(0, len(rs) - 1, MAX_PER_CAR)]
    rows = [r for rs in per_car.values() for r in rs]

    print(f"\n{'video':<16} {'crops':>6} {'read':>6} {'cars':>6} {'cars >=' + str(MIN_CROPS):>9}")
    for v in sorted({r["video"] for r in all_rows}):
        cars_v = Counter(r["text"] for r in rows if r["video"] == v)
        print(f"{v:<16} {sum(r['video'] == v for r in all_rows):>6} {sum(cars_v.values()):>6} "
              f"{len(cars_v):>6} {sum(c >= MIN_CROPS for c in cars_v.values()):>9}")
    usable = {t: rs for t, rs in per_car.items() if len(rs) >= MIN_CROPS}
    multi = [t for t, rs in per_car.items() if len({r["video"] for r in rs}) > 1]
    print(f"{'TOTAL':<16} {len(all_rows):>6} {len(rows):>6} {len(per_car):>6} {len(usable):>9}")
    print(f"\nusable for training (>= {MIN_CROPS} crops): {len(usable)} cars, {sum(map(len, usable.values()))} crops, "
          f"median {int(np.median([len(r) for r in usable.values()])) if usable else 0} crops per car")
    print(f"cars seen in more than one video: {len(multi)} {multi[:10]}")
    print(f"merged 1-letter misreads: {len(fix)}")

    reid = OUT / "reid"
    reid.mkdir(exist_ok=True)
    for old in reid.glob("*.jpg"):
        old.unlink()
    for t, rs in usable.items():
        for r in rs:
            img = cv2.imread(str(OUT / "crops" / r["video"] / r["crop"]))
            # cover the plate: the model must learn the CAR (shape, colour, dents, stickers),
            # not read the plate, because re-ID is needed exactly when the plate is NOT readable
            px1, py1, px2, py2 = (int(r[k]) for k in ("px1", "py1", "px2", "py2"))
            m = int(0.15 * (px2 - px1))
            img[max(0, py1 - m):py2 + m, max(0, px1 - m):px2 + m] = img.reshape(-1, 3).mean(0)
            cv2.imwrite(str(reid / reid_name(t, r)), img)
    print(f"wrote {reid}/ ({sum(map(len, usable.values()))} crops, plates covered)")

    # contact sheet: 12 cars x 6 crops each, to SEE that one plate = one car
    sample = sorted(usable, key=lambda t: -len(usable[t]))[:12]
    rows_img = []
    for t in sample:
        rs = usable[t]
        pick = [rs[int(k)] for k in np.linspace(0, len(rs) - 1, 6)]
        ims = [cv2.resize(cv2.imread(str(reid / reid_name(t, r))), (160, 120))
               for r in pick]
        label = np.full((120, 120, 3), 255, np.uint8)
        cv2.putText(label, t, (4, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(label, f"{len(rs)} crops", (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        rows_img.append(np.hstack([label] + ims))
    if rows_img:
        Path("vehicle/output").mkdir(exist_ok=True)
        cv2.imwrite("vehicle/output/v03_mined_cars.jpg", np.vstack(rows_img))
        print("contact sheet: vehicle/output/v03_mined_cars.jpg")


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        mine()
    summary()
