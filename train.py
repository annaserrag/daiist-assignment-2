"""
Assignment 2 — training pipeline.

Business question (see REPORT.md): the same M&A boutique as Assignment 1, now on the
buy side. During due diligence an associate has to confirm that every execution page in
the target's data room carries the company seal, because in the jurisdictions these
documents come from an unsealed contract may not bind the company. A YOLO detector
scores each page, and two thresholds on that score send it to pass, associate review
or fail.

Run with `uv run python main.py train`. The script is meant to be read top to bottom:

    Part 1  The dataset           download it and describe it before training on it
    Part 2  Finetuning            yolo11s from its COCO checkpoint
    Part 3  Evaluation            per class, against COCO zero-shot and predict-nothing
    Part 4  From boxes to pages   one score per page, and a stress test with seals removed
    Part 5  Cost and review       the threshold sweep, the review band, the alternatives
    Part 6  Save                  weights/ and artifacts/, everything app.py loads

Training is the only slow part (3.6 hours for 30 epochs on an Apple M5 Pro GPU). Setting
REUSE_RUN=1 skips it and evaluates the last run in runs/seal instead, which is only a
convenience for re-running Parts 3 to 6.
"""

# %% Setup
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

pd.set_option("display.width", 120)

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
WEIGHTS = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"   # everything app.py loads; nothing there is retrained
RUNS = ROOT / "runs"

# The exact Roboflow Universe version this pipeline is built on (REPORT.md, "Dataset").
WORKSPACE, PROJECT, VERSION = "hrsdyolo", "stamp-signature-qr", 1
DATASET_DIR = DATASETS / f"{PROJECT}-{VERSION}"
CLASS_NAMES = ["Signature", "Text", "qr-code", "stamp"]  # order of the label files
STAMP_ID = CLASS_NAMES.index(costing.STAMP_CLASS)

BASE_MODEL = "yolo11s.pt"
EPOCHS, IMGSZ, BATCH, PATIENCE, SEED = 30, 640, 16, 10, 0
RUN_NAME = "seal"
FINAL_WEIGHTS = WEIGHTS / "seal_yolo11s.pt"
LOW_CONF = 0.001     # keep almost every box, so thresholds are chosen here, not by NMS
IOU_MATCH = 0.5      # a prediction counts as a hit at IoU >= 0.5 (AP50)
N_BOOT = 1000
LINE = "=" * 78

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon
else:
    DEVICE = "cpu"


def section(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


# =================================================================================
# PART 1 — THE DATASET
# =================================================================================

# %% 1.1 Download the exact version. Roboflow's own data.yaml uses relative paths that
# resolve differently depending on where Ultralytics is run from, so we write our own.
section("1.1 Download")
if not (DATASET_DIR / "test" / "labels").is_dir():
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATASET_DIR)
    )
DATA_YAML = DATASETS / "seal_data.yaml"
DATA_YAML.write_text(
    f"path: {DATASET_DIR}\n"
    "train: train/images\nval: valid/images\ntest: test/images\n"
    "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
)
print(f"{WORKSPACE}/{PROJECT}/{VERSION} in {DATASET_DIR.relative_to(ROOT)}")

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def read_labels(split, stem):
    """One YOLO label file as a list of (class_id, cx, cy, w, h), all fractions."""
    path = DATASET_DIR / SPLITS[split] / "labels" / f"{stem}.txt"
    if not path.exists():
        return []
    return [
        (int(parts[0]), *map(float, parts[1:5]))
        for parts in (line.split() for line in path.read_text().splitlines())
        if parts
    ]


def image_paths(split):
    return sorted((DATASET_DIR / SPLITS[split] / "images").glob("*.jpg"))


# %% 1.2 Instances per class, images per class, and object size relative to the page.
section("1.2 Composition")
composition, sizes, combos = [], {}, {}
for split in SPLITS:
    instances, images_with = Counter(), Counter()
    areas = {name: [] for name in CLASS_NAMES}
    combos[split] = Counter()
    paths = image_paths(split)
    for path in paths:
        labels = read_labels(split, path.stem)
        present = {CLASS_NAMES[c] for c, *_ in labels}
        combos[split][" + ".join(sorted(present)) or "(no labels)"] += 1
        for c, _, _, w, h in labels:
            instances[CLASS_NAMES[c]] += 1
            areas[CLASS_NAMES[c]].append(w * h)
        for name in present:
            images_with[name] += 1
    for name in CLASS_NAMES:
        a = np.array(areas[name])
        composition.append({
            "split": split, "class": name, "images": len(paths),
            "instances": instances[name], "images_with_class": images_with[name],
            "median_area": float(np.median(a)) if a.size else float("nan"),
            "p10_area": float(np.percentile(a, 10)) if a.size else float("nan"),
            "p90_area": float(np.percentile(a, 90)) if a.size else float("nan"),
            "degenerate_boxes": int(np.sum(a < 1e-4)),
        })
composition = pd.DataFrame(composition)
print(composition.round(4).to_string(index=False))
print("\nArea is box area as a fraction of the page (all images are 640x640, stretched).")

# %% 1.3 Which classes appear together on a page. This is the finding that shapes the
# whole framing: stamps, QR codes and signatures never share a page, because the
# dataset is several source collections merged into one (stamps on deeds and notarised pages, signatures
# on inverted 1970s-90s letters, QR codes in photographs, a little text).
section("1.3 Classes per page")
for split, counter in combos.items():
    print(f"{split:<6}", dict(counter.most_common()))
print(
    "\nStamps, QR codes and signatures never share a page (signatures only come with"
    "\ntext, from the same letter archive), so a stamp page can carry a signature nobody"
    "\nlabelled, and no page is a document with nothing on it. Stamp is the only class with both"
    "\nenough instances and documents that look like a data room, so it carries the"
    "\ndecision; the other three are trained and reported, not relied on."
)


# =================================================================================
# PART 2 — FINETUNING
# =================================================================================

# %% 2.1 Finetune yolo11s from its COCO weights. yolo11s rather than yolo11n: stamps
# are ~3% of the page and the extra capacity is cheap at inference (one page at a time,
# no video), so the trade is worth it. deterministic=False for the reason given in the
# Session 9 notebook: MPS has no deterministic backward for some ops, and seed=0 still
# fixes the shuffling and augmentation.
section("2.1 Finetuning")
run_dir = RUNS / RUN_NAME
if os.environ.get("REUSE_RUN") == "1" and (run_dir / "weights" / "best.pt").exists():
    print(f"REUSE_RUN=1: evaluating the existing run in {run_dir.relative_to(ROOT)}")
else:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        project=str(RUNS),  # otherwise this lands wherever Ultralytics' settings point
        name=RUN_NAME,
        exist_ok=True,
        seed=SEED,
        deterministic=False,
    )
    run_dir = Path(model.trainer.save_dir)

WEIGHTS.mkdir(exist_ok=True)
shutil.copy(run_dir / "weights" / "best.pt", FINAL_WEIGHTS)
history = pd.read_csv(run_dir / "results.csv")
history.columns = history.columns.str.strip()
print(f"best weights -> {FINAL_WEIGHTS.relative_to(ROOT)} "
      f"({FINAL_WEIGHTS.stat().st_size / 1e6:.1f} MB), {len(history)} epochs run")

finetuned = YOLO(str(FINAL_WEIGHTS))
coco = YOLO(BASE_MODEL)


# =================================================================================
# PART 3 — EVALUATION ON THE TEST SPLIT
# =================================================================================

# %% 3.1 The finetuned model through Ultralytics' own validator: per-class precision,
# recall, AP50 and AP50-95, and the confusion matrix (IoU 0.45, conf 0.25 by default).
section("3.1 Finetuned model, Ultralytics metrics on test")
test_val = finetuned.val(
    data=str(DATA_YAML), split="test", imgsz=IMGSZ, device=DEVICE,
    project=str(RUNS), name="seal_test", exist_ok=True, verbose=False,
)
box = test_val.box
per_class = []
for row, class_id in enumerate(box.ap_class_index):
    name = CLASS_NAMES[int(class_id)]
    test_instances = composition.query("split == 'test' and `class` == @name")
    per_class.append({
        "class": name,
        "test_instances": int(test_instances["instances"].iloc[0]),
        "precision": float(box.p[row]),
        "recall": float(box.r[row]),
        "AP50": float(box.ap50[row]),
        "AP50-95": float(box.ap[row]),
    })
per_class = pd.DataFrame(per_class)
overall = {"mAP50": float(box.map50), "mAP50-95": float(box.map)}
print(per_class.round(3).to_string(index=False))
print(f"\noverall mAP50 {overall['mAP50']:.3f}  mAP50-95 {overall['mAP50-95']:.3f}")

# Ultralytics' P and R are read at the confidence that maximises mean F1 over classes;
# AP needs every box, so that pass keeps boxes down to conf 0.001. The confusion matrix
# would then count thousands of near-zero boxes as background FPs, so it gets its own
# pass at conf 0.25, the cut the model would plausibly be run at.
# Rows are predicted class, columns are true class; the last row/column is background.
cm_val = finetuned.val(
    data=str(DATA_YAML), split="test", imgsz=IMGSZ, device=DEVICE, conf=0.25,
    project=str(RUNS), name="seal_test_cm", exist_ok=True, verbose=False,
)
confusion = cm_val.confusion_matrix.matrix.astype(int)
confusion_labels = CLASS_NAMES + ["background"]
print("\nconfusion matrix (rows predicted, columns true):")
print(pd.DataFrame(confusion, index=confusion_labels, columns=confusion_labels))


# %% 3.2 Running a model once over a split and keeping every box. Everything in Parts
# 3 and 4 reuses these lists, so each model sees each page exactly once.
def predict_split(model, paths):
    """{stem: [(class_id, conf, (x0, y0, x1, y1) as fractions)]} for every page."""
    out = {}
    for start in range(0, len(paths), 32):
        batch = paths[start:start + 32]
        results = model.predict(
            [str(p) for p in batch], conf=LOW_CONF, imgsz=IMGSZ, device=DEVICE,
            verbose=False,
        )
        for path, result in zip(batch, results):
            out[path.stem] = [
                (int(c), float(s), tuple(xyxy))
                for c, s, xyxy in zip(
                    result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                    result.boxes.xyxyn.tolist(),
                )
            ]
    return out


def to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_ap50(predictions, split, true_class, predicted_class=None):
    """AP50, and precision/recall at conf 0.25, for one class, the Session 9 way.

    Predictions are matched greedily in confidence order and each labelled box can only
    be claimed once. predicted_class lets the COCO model's nearest class stand in for
    one it does not have. AP is the all-point area under the precision-recall curve, so
    it can differ from Ultralytics' 101-point figure in the third decimal.
    """
    predicted_class = true_class if predicted_class is None else predicted_class
    truth = {
        stem: [to_xyxy(*box) for c, *box in read_labels(split, stem) if c == true_class]
        for stem in predictions
    }
    n_true = sum(len(boxes) for boxes in truth.values())
    scored = sorted(
        ((conf, stem, xyxy) for stem, boxes in predictions.items()
         for c, conf, xyxy in boxes if c == predicted_class),
        key=lambda row: -row[0],
    )
    claimed = {stem: [False] * len(boxes) for stem, boxes in truth.items()}
    hits = []
    for conf, stem, xyxy in scored:
        overlaps = [iou(xyxy, t) for t in truth[stem]]
        best = int(np.argmax(overlaps)) if overlaps else -1
        hit = best >= 0 and overlaps[best] >= IOU_MATCH and not claimed[stem][best]
        if hit:
            claimed[stem][best] = True
        hits.append((conf, hit))
    if not hits or n_true == 0:
        return {"AP50": 0.0, "precision@0.25": float("nan"), "recall@0.25": 0.0}
    confs = np.array([c for c, _ in hits])
    tp = np.cumsum([h for _, h in hits])
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / n_true
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * envelope))
    keep = confs >= 0.25
    return {
        "AP50": ap,
        "precision@0.25": float(precision[keep][-1]) if keep.any() else float("nan"),
        "recall@0.25": float(recall[keep][-1]) if keep.any() else 0.0,
    }


section("3.2 Predictions over validation and test")
val_paths, test_paths = image_paths("val"), image_paths("test")
pred = {
    ("finetuned", "val"): predict_split(finetuned, val_paths),
    ("finetuned", "test"): predict_split(finetuned, test_paths),
    ("coco", "val"): predict_split(coco, val_paths),
    ("coco", "test"): predict_split(coco, test_paths),
}
print("done:", ", ".join(f"{m}/{s} {len(p)} pages" for (m, s), p in pred.items()))

# %% 3.3 Baseline 1: the pretrained COCO model, zero-shot. It has no stamp class, so
# like the playing card turning into `book` in Session 9 it can only reach for its
# nearest word. We give it the best chance it has: the COCO class whose boxes overlap
# the most labelled stamps on VALIDATION stands in for "stamp", then it is scored on
# test exactly like the finetuned model.
section("3.3 Baseline: COCO zero-shot")
stand_in_hits = Counter()
for stem, boxes in pred[("coco", "val")].items():
    stamps = [to_xyxy(*b) for c, *b in read_labels("val", stem) if c == STAMP_ID]
    for c, conf, xyxy in boxes:
        if conf >= 0.25 and any(iou(xyxy, s) >= IOU_MATCH for s in stamps):
            stand_in_hits[coco.names[c]] += 1
print("COCO classes landing on validation stamps (conf >= 0.25):",
      dict(stand_in_hits.most_common(5)) or "none")
COCO_STAND_IN = stand_in_hits.most_common(1)[0][0] if stand_in_hits else "clock"
coco_stand_in_id = {v: k for k, v in coco.names.items()}[COCO_STAND_IN]
print(f"stand-in for stamp: '{COCO_STAND_IN}'")

# %% 3.4 Baseline 2, predict nothing, and the comparison on the class that carries the
# decision. Predicting nothing has no false positives and finds no stamps: AP 0,
# recall 0, the detector version of "always pass".
section("3.4 Stamp class on test: finetuned vs baselines")
baselines = pd.DataFrame([
    {"model": "finetuned yolo11s",
     **class_ap50(pred[("finetuned", "test")], "test", STAMP_ID)},
    {"model": f"COCO yolo11s zero-shot ('{COCO_STAND_IN}' as stamp)",
     **class_ap50(pred[("coco", "test")], "test", STAMP_ID, coco_stand_in_id)},
    {"model": "predict nothing", "AP50": 0.0, "precision@0.25": float("nan"),
     "recall@0.25": 0.0},
])
print(baselines.round(3).to_string(index=False))
print(
    "\nThe other three classes do not exist in COCO's vocabulary either, so both"
    "\nbaselines score 0 AP on them by construction; the per-class table in 3.1 is the"
    "\ncomparison that matters there."
)


# =================================================================================
# PART 4 — FROM BOXES TO PAGES
# =================================================================================

# %% 4.1 One score per page: the confidence of its most confident stamp box (0 if it
# has none). Truth comes from the labels: a page is sealed if it has a labelled stamp.
# Only document pages count. The QR-code images are photographs of codes on desks and
# phones, which never reach a data room, so they are left out of every page-level
# number and reported separately.
section("4.1 Page scores")


def page_table(predictions, split, model_names):
    rows = []
    for stem, boxes in predictions.items():
        labels = {CLASS_NAMES[c] for c, *_ in read_labels(split, stem)}
        rows.append({
            "stem": stem,
            "source": "qr photo" if labels == {"qr-code"} else "document",
            "sealed": "stamp" in labels,
            "score": costing.page_score([(c, s) for c, s, _ in boxes], model_names),
        })
    return pd.DataFrame(rows)


pages = {
    (m, s): page_table(pred[(m, s)], s, finetuned.names if m == "finetuned" else
                       {i: ("stamp" if i == coco_stand_in_id else n)
                        for i, n in coco.names.items()})
    for (m, s) in pred
}
for (m, s), table in pages.items():
    docs = table[table.source == "document"]
    print(f"{m:<10}{s:<5} {len(docs):>4} document pages: {int(docs.sealed.sum())} sealed, "
          f"{int((~docs.sealed).sum())} unsealed; {int((table.source == 'qr photo').sum())}"
          f" QR photos left out")

# %% 4.2 Stress test: the same sealed pages with the seal removed. The only unsealed
# documents in the split are 40 inverted letters (25 in validation) from a different
# archive, which look nothing like an unsealed deed, so measuring recall on them alone
# would flatter the model. Painting every labelled stamp box with the paper colour
# gives an unsealed version of each real sealed page: same layout, same signatures, no
# seal. No labels are created; the truth is "no labelled seal left".
# Its weakness, visible on the pages themselves: the labels often box only some of the
# seals on a page, so an unlabelled seal stays behind and is counted as a miss. That
# makes this test harsher than reality, which is why 5.6 also costs the decision
# without it.
section("4.2 Stress test: seals painted out")
MASKED_DIR = DATASETS / "masked"


def masked_scores(split):
    out_dir = MASKED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = [p for p in image_paths(split)
              if any(c == STAMP_ID for c, *_ in read_labels(split, p.stem))]
    masked_paths = []
    for path in sealed:
        target = out_dir / path.name
        if not target.exists():
            image = Image.open(path).convert("RGB")
            pixels = np.array(image)
            # The paper is the bright end of the page, not its median (text and scan
            # shadows pull the median towards grey)
            fill = np.percentile(pixels.reshape(-1, 3), 90, axis=0).astype(np.uint8)
            W, H = image.size
            for c, cx, cy, w, h in read_labels(split, path.stem):
                if c == STAMP_ID:
                    x0, y0, x1, y1 = to_xyxy(cx, cy, w, h)
                    # 5% margin so the stamp's faint outer ring goes too
                    x0, x1 = max(0, x0 - 0.05 * w), min(1, x1 + 0.05 * w)
                    y0, y1 = max(0, y0 - 0.05 * h), min(1, y1 + 0.05 * h)
                    pixels[int(y0 * H):int(np.ceil(y1 * H)),
                           int(x0 * W):int(np.ceil(x1 * W))] = fill
            Image.fromarray(pixels).save(target, quality=95)
        masked_paths.append(target)
    boxes = predict_split(finetuned, masked_paths)
    return pd.DataFrame({
        "stem": list(boxes),
        "source": "seal painted out",
        "sealed": False,
        "score": [costing.page_score([(c, s) for c, s, _ in b], finetuned.names)
                  for b in boxes.values()],
    })


masked = {split: masked_scores(split) for split in ("val", "test")}
for split, table in masked.items():
    print(f"{split:<5} {len(table)} pages with the seal painted out, median score "
          f"{table.score.median():.3f}, {int((table.score >= 0.5).sum())} still score >= 0.5")


def decision_sets(model, split, stress=True):
    """(unsealed scores, sealed scores) for the page-level decision."""
    table = pages[(model, split)]
    docs = table[table.source == "document"]
    unsealed = docs.loc[~docs.sealed, "score"].to_numpy()
    if model == "finetuned" and stress:
        unsealed = np.concatenate([unsealed, masked[split]["score"].to_numpy()])
    return unsealed, docs.loc[docs.sealed, "score"].to_numpy()


# %% 4.3 How separable the two kinds of page are, by source.
section("4.3 Score distribution by source (test, finetuned)")
test_scores = pd.concat([pages[("finetuned", "test")], masked["test"]])
summary = test_scores.groupby(["source", "sealed"])["score"].describe(
    percentiles=[0.05, 0.5, 0.95])
print(summary.round(3).to_string())

# %% 4.4 Calibration (Session 12). If the page score were a calibrated P(sealed), the
# cost table alone would give the cut: flag when P(unsealed) * COST_FN > COST_FLAG, so
# flag below t* = 1 - COST_FLAG / COST_FN. YOLO confidence is not trained to be a
# probability and the test mix is not the data room mix, so we check it before
# trusting that formula, with a reliability table on the test pages.
section("4.4 Is the page score a probability?")
T_STAR = 1 - costing.COST_FLAG / costing.COST_FN
unsealed_test, sealed_test = decision_sets("finetuned", "test")
all_scores = np.concatenate([unsealed_test, sealed_test])
all_sealed = np.concatenate([np.zeros(len(unsealed_test)), np.ones(len(sealed_test))])
bins = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0001])
reliability = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (all_scores >= lo) & (all_scores < hi)
    if mask.any():
        reliability.append({"bin": f"[{lo:.2f}, {min(hi, 1):.2f})", "pages": int(mask.sum()),
                            "mean_score": float(all_scores[mask].mean()),
                            "share_sealed": float(all_sealed[mask].mean())})
reliability = pd.DataFrame(reliability)
print(reliability.round(3).to_string(index=False))
print(f"\ncost-derived cut if the score were calibrated: t* = {T_STAR:.3f}")


# =================================================================================
# PART 5 — COST AND HUMAN REVIEW
# =================================================================================

# %% 5.1 Single cut, swept. Chosen on VALIDATION, reported on test, so the number we
# quote was not used to pick itself (the same discipline as C in Assignment 1).
section("5.1 Single-cut sweep")
sweeps = {
    split: pd.DataFrame(costing.single_cut_sweep(*decision_sets("finetuned", split)))
    for split in ("val", "test")
}
best_single = sweeps["val"].loc[sweeps["val"]["cost"].idxmin()]
CHOSEN_CUT = float(best_single["low"])
test_at_cut = sweeps["test"].loc[np.isclose(sweeps["test"]["low"], CHOSEN_CUT)].iloc[0]
print(f"cheapest single cut on validation: {CHOSEN_CUT:.4f}")
print(f"on test at that cut: EUR {test_at_cut['cost']:,.0f} per 1,000 pages, "
      f"{test_at_cut['fn']:.2f} FN, {test_at_cut['fp']:.1f} FP per 1,000")
print(f"cheapest single cut on test itself: "
      f"{sweeps['test'].loc[sweeps['test']['cost'].idxmin(), 'low']:.4f}")

# %% 5.2 Three outputs: two cuts. The band between them goes to an associate.
section("5.2 Pass / review / fail band")
ranked_val = costing.rank_bands(*decision_sets("finetuned", "val"))
best_band = ranked_val[0]
LOW, HIGH = best_band["low"], best_band["high"]
test_band = costing.three_way_cost(*decision_sets("finetuned", "test"), LOW, HIGH)
print(f"cheapest band on validation: [{LOW:.4f}, {HIGH:.4f})")
for name, row in (("validation", best_band), ("test", test_band)):
    print(f"  {name:<11} EUR {row['cost']:>9,.0f}/1,000   reviewed {row['review_share']:.1%}"
          f"   FN {row['fn']:.2f}   FP {row['fp']:.1f}")

# %% 5.2b The cost-against-threshold plot for REPORT.md (the dashboard draws its own,
# live version). Single cut on both splits, the chosen cut, the band, and t* from 4.4.
import matplotlib

matplotlib.use("Agg")  # no window: this runs as a script
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sweeps["test"]["low"], sweeps["test"]["cost"], lw=2.5, label="single cut, test pages")
ax.plot(sweeps["val"]["low"], sweeps["val"]["cost"], ls=":", lw=2,
        label="single cut, validation pages")
ax.scatter([CHOSEN_CUT], [test_at_cut["cost"]], s=80, color="#dc2626", zorder=5,
           label=f"chosen single cut {CHOSEN_CUT:.2f} (from validation)")
ax.axvspan(LOW, HIGH, color="#f59e0b", alpha=0.15,
           label=f"chosen band [{LOW:.2f}, {HIGH:.2f}): EUR {test_band['cost']:,.0f} on test")
ax.axvline(T_STAR, color="grey", ls="--", label=f"t* = {T_STAR:.2f} if the score were calibrated")
ax.set(yscale="log", xlabel="threshold on the stamp score (fail below, pass at or above)",
       ylabel="EUR per 1,000 pages (log scale)",
       title=f"Expected cost against threshold, base rate {costing.BASE_RATE:.0%}")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
ARTIFACTS.mkdir(exist_ok=True)
fig.savefig(ARTIFACTS / "cost_curve.png", dpi=150)
plt.close(fig)
print("saved artifacts/cost_curve.png")

# %% 5.3 Priced against the three alternatives, on test.
section("5.3 Against the alternatives (test, per 1,000 pages)")
designs = costing.reference_designs(*decision_sets("finetuned", "test"))
designs[f"model alone at the chosen cut {CHOSEN_CUT:.3f}"] = test_at_cut.to_dict()
designs[f"three-way band [{LOW:.3f}, {HIGH:.3f})"] = test_band
coco_unsealed, coco_sealed = decision_sets("coco", "test")
coco_sweep = pd.DataFrame(costing.single_cut_sweep(coco_unsealed, coco_sealed))
designs["COCO zero-shot, its best cut"] = coco_sweep.loc[coco_sweep["cost"].idxmin()].to_dict()
alternatives = pd.DataFrame([
    {"design": name, "cost_per_1000": row["cost"], "fn": row["fn"], "fp": row["fp"],
     "review_share": row["review_share"]}
    for name, row in designs.items()
])
print(alternatives.round(2).to_string(index=False))

# %% 5.4 Staffing. Whether the review queue fits the people the boutique has.
section("5.4 Staffing")
staff = costing.staffing(test_band["review_share"], flagged_per_1000=test_band["flagged"])
print(f"{costing.PAGES_PER_DAY} pages a day -> {staff['reviewed_per_day']:.0f} reviewed, "
      f"{staff['flagged_per_day']:.0f} flagged, {staff['hours_per_day']:.1f} associate hours,"
      f" {staff['associates']:.2f} associates at {costing.REVIEW_HOURS_PER_DAY}h a day")

# %% 5.5 How sure. Bootstrap the test pages within each truth class (Session 3), and
# re-cost the chosen band each time: the spread is what the test set can tell us.
section("5.5 Bootstrap of the chosen band on test")
rng = np.random.default_rng(SEED)
unsealed_test, sealed_test = decision_sets("finetuned", "test")
boot = np.array([
    costing.three_way_cost(rng.choice(unsealed_test, len(unsealed_test)),
                           rng.choice(sealed_test, len(sealed_test)), LOW, HIGH)["cost"]
    for _ in range(N_BOOT)
])
boot_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
print(f"EUR {test_band['cost']:,.0f} per 1,000, 95% interval EUR {boot_ci[0]:,.0f} to "
      f"{boot_ci[1]:,.0f}")

# %% 5.6 Sensitivity: which assumption moves the answer. Each row re-chooses the band
# on validation with one figure changed and prices it on test.
section("5.6 Sensitivity")
sensitivity = []
for label, kwargs in [
    ("as assumed", {}),
    ("base rate 1%", {"base_rate": 0.01}),
    ("base rate 20%", {"base_rate": 0.20}),
    ("missed seal costs EUR 500", {"cost_fn": 500.0}),
    ("missed seal costs EUR 10,000", {"cost_fn": 10_000.0}),
    ("review takes 2 minutes", {"cost_review": costing.ASSOCIATE_RATE * 120 / 3600}),
    ("associate recall 80%", {"associate_recall": 0.80}),
    ("unsealed = real letters only, no stress test", {"stress": False}),
]:
    stress = kwargs.pop("stress", True)
    choice = costing.rank_bands(*decision_sets("finetuned", "val", stress), **kwargs)[0]
    priced = costing.three_way_cost(*decision_sets("finetuned", "test", stress),
                                    choice["low"], choice["high"], **kwargs)
    sensitivity.append({"assumption": label, "low": choice["low"], "high": choice["high"],
                        "review_share": priced["review_share"], "cost_per_1000": priced["cost"]})
sensitivity = pd.DataFrame(sensitivity)
print(sensitivity.round(4).to_string(index=False))


# =================================================================================
# PART 6 — SAVE WHAT THE APP NEEDS
# =================================================================================

# %% 6.1 Everything app.py reads. Small files, committed with the weights, so the
# dashboard runs on a fresh clone without the dataset and without training.
section("6.1 Artifacts")
ARTIFACTS.mkdir(exist_ok=True)
composition.to_csv(ARTIFACTS / "composition.csv", index=False)
per_class.to_csv(ARTIFACTS / "per_class.csv", index=False)
baselines.to_csv(ARTIFACTS / "baselines.csv", index=False)
alternatives.to_csv(ARTIFACTS / "alternatives.csv", index=False)
sensitivity.to_csv(ARTIFACTS / "sensitivity.csv", index=False)
reliability.to_csv(ARTIFACTS / "reliability.csv", index=False)
sweeps["test"].to_csv(ARTIFACTS / "sweep_test.csv", index=False)
sweeps["val"].to_csv(ARTIFACTS / "sweep_val.csv", index=False)
history.to_csv(ARTIFACTS / "training_history.csv", index=False)
pd.concat([
    pages[("finetuned", "test")].assign(split="test"),
    masked["test"].assign(split="test"),
    pages[("finetuned", "val")].assign(split="val"),
    masked["val"].assign(split="val"),
]).to_csv(ARTIFACTS / "page_scores.csv", index=False)
summary_json = {
    "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"},
    "training": {"base_model": BASE_MODEL, "epochs_run": int(len(history)),
                 "epochs_max": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                 "patience": PATIENCE, "seed": SEED, "device": DEVICE},
    "weights": str(FINAL_WEIGHTS.relative_to(ROOT)),
    "overall": overall,
    "confusion_matrix": {"labels": confusion_labels, "matrix": confusion.tolist(),
                         "orientation": "rows predicted, columns true"},
    "coco_stand_in": COCO_STAND_IN,
    "positive_class": "unsealed page",
    "chosen_cut": CHOSEN_CUT,
    "chosen_band": {"low": LOW, "high": HIGH},
    "test_band": test_band,
    "bootstrap_ci_cost": boot_ci,
    "t_star_if_calibrated": T_STAR,
    "staffing": staff,
    "costs": {"fn": costing.COST_FN, "flag": costing.COST_FLAG,
              "review": costing.COST_REVIEW, "base_rate": costing.BASE_RATE,
              "associate_recall": costing.ASSOCIATE_RECALL,
              "associate_specificity": costing.ASSOCIATE_SPECIFICITY,
              "pages_per_day": costing.PAGES_PER_DAY},
}
(ARTIFACTS / "summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
shutil.copy(RUNS / "seal_test_cm" / "confusion_matrix.png", ARTIFACTS / "confusion_matrix.png")
print("saved:", ", ".join(sorted(p.name for p in ARTIFACTS.iterdir())))
