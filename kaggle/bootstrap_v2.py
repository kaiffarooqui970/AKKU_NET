# %% [markdown]
# # AkkuNet bootstrap_v2 on Kaggle (free GPU)
#
# Trains bootstrap_v2 exactly like bootstrap_v1 (YOLO11n from COCO-pretrained weights,
# same data, same settings) PLUS thousands of public **"no battery"** images (COCO,
# empty labels, TRAIN only). The only change is the negatives, so the model learns what is *not* a battery
# (Tic Tacs, phones, furniture, faces, ...). **BOOTSTRAP ONLY: internet photos,
# not tipping-floor performance.**
#
# Before running, in the notebook settings:
# 1. **Accelerator: GPU** (P100 or T4).
# 2. **Internet: on** (needs a phone-verified Kaggle account).
# 3. **Add input:** your private dataset with `akkunet_bootstrap_v2.zip`.
#
# Then **Run all**. At the end, download `/kaggle/working/bootstrap_v2_results.zip`
# (Output tab) and unzip it into `~/akkunet/runs/bootstrap/`.
#
# COCO images are used only if their Flickr licence allows commercial use
# without share-alike (CC BY, no known copyright, US government work); every
# image used is listed with its licence and URL for attribution.

# %%
SEED = 0
N_TRAIN_NEGATIVES = 3000      # COCO images with empty labels, added to TRAIN only
N_FP_TEST = 500               # other COCO images, never trained on: false-alarm test for v1 vs v2
EXCLUDE_CATEGORIES = []       # COCO categories to keep OUT of the negatives, e.g. ["cell phone"]
                              # if whole phones should alert later (open decision)
EPOCHS, PATIENCE, BATCH, IMGSZ = 60, 15, 16, 640   # = bootstrap_v1
INIT = "yolo11n.pt"           # = bootstrap_v1. (Fine-tuning v1 with default LR wrecked it in a local test.)
ALERT_CONF = 0.5              # same as the demo's alert threshold
ALLOWED_LICENCES = {"Attribution License", "No known copyright restrictions", "United States Government Work"}
CLASSES = ["vape", "power_bank", "battery_pack", "tool_battery", "loose_cell"]

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.179", "lap"], check=True)

# %%
import csv, json, random, zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

import yaml

src = next(Path("/kaggle/input").rglob("bootstrap/images/train")).parents[2]
print("Uploaded data:", src)
work = Path("/kaggle/working")
v1_weights = src / "weights" / "bootstrap_v1_best.pt"
assert v1_weights.exists(), "bootstrap_v1 weights missing from the upload"

# %% [markdown]
# ## Public negatives: COCO 2017 (licence-filtered, no images with excluded categories)

# %%
ann_zip = work / "annotations_trainval2017.zip"
if not ann_zip.exists():
    urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)
with zipfile.ZipFile(ann_zip) as z:
    coco = json.loads(z.read("annotations/instances_train2017.json"))
licences = {l["id"]: l for l in coco["licenses"]}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}
cats_in = {}
for a in coco["annotations"]:
    cats_in.setdefault(a["image_id"], set()).add(cat_names[a["category_id"]])
pool = [im for im in coco["images"]
        if licences[im["license"]]["name"] in ALLOWED_LICENCES
        and not (cats_in.get(im["id"], set()) & set(EXCLUDE_CATEGORIES))]
print(f"{len(coco['images'])} COCO train2017 images, {len(pool)} with an allowed licence and no excluded category")
random.Random(SEED).shuffle(pool)
picked = pool[:N_TRAIN_NEGATIVES + N_FP_TEST]
del coco

# %%
neg_dir = work / "coco_negatives"
for sub in ("train", "fp_test"):
    (neg_dir / "images" / sub).mkdir(parents=True, exist_ok=True)
    (neg_dir / "labels" / sub).mkdir(parents=True, exist_ok=True)

def fetch(item):
    i, im = item
    sub = "train" if i < N_TRAIN_NEGATIVES else "fp_test"
    out = neg_dir / "images" / sub / im["file_name"]
    if not out.exists():
        urllib.request.urlretrieve(im["coco_url"], out)
    (neg_dir / "labels" / sub / (out.stem + ".txt")).write_text("")   # empty label = "no battery here"
    return sub, out

with ThreadPoolExecutor(16) as ex:
    fetched = list(ex.map(fetch, enumerate(picked)))
neg_train = [str(p) for s, p in fetched if s == "train"]
fp_test = [str(p) for s, p in fetched if s == "fp_test"]
print(len(neg_train), "train negatives,", len(fp_test), "false-alarm test images")

with open(work / "coco_negatives_attribution.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["use", "file", "licence", "licence_url", "flickr_url", "coco_url"])
    for i, im in enumerate(picked):
        lic = licences[im["license"]]
        w.writerow(["train" if i < N_TRAIN_NEGATIVES else "fp_test", im["file_name"], lic["name"], lic["url"],
                    im.get("flickr_url", ""), im["coco_url"]])
print("Categories in the train negatives (top 15):",
      Counter(c for im in picked[:N_TRAIN_NEGATIVES] for c in cats_in.get(im["id"], {"(none)"})).most_common(15))

# %% [markdown]
# ## Dataset: bootstrap train + negatives (train only); val = the same bootstrap_val as v1

# %%
train = sorted(str(p) for p in (src / "bootstrap" / "images" / "train").iterdir())
val = sorted(str(p) for p in (src / "bootstrap" / "images" / "val").iterdir())
own = sorted(str(p) for p in (src / "own_negatives" / "images" / "train").glob("*.jpg")) \
    if (src / "own_negatives").exists() else []
(work / "lists").mkdir(exist_ok=True)
(work / "lists" / "train.txt").write_text("\n".join(train + neg_train + own) + "\n")
(work / "lists" / "val.txt").write_text("\n".join(val) + "\n")
data_yaml = work / "bootstrap_v2.yaml"
data_yaml.write_text(yaml.safe_dump({"train": str(work / "lists" / "train.txt"),
                                     "val": str(work / "lists" / "val.txt"),
                                     "names": dict(enumerate(CLASSES))}, sort_keys=False))
print(f"train: {len(train)} bootstrap + {len(neg_train)} COCO negatives + {len(own)} own negatives; val: {len(val)}")

# %% [markdown]
# ## False-alarm test, before: bootstrap_v1 on the held-out COCO images

# %%
from ultralytics import YOLO

def false_alarms(weights):
    """Share of held-out no-battery images with >= 1 box at the demo's alert confidence."""
    hits, per_class = 0, Counter()
    for r in YOLO(weights).predict(fp_test, conf=ALERT_CONF, imgsz=IMGSZ, stream=True, verbose=False):
        cls = [CLASSES[int(c)] for c in r.boxes.cls]
        hits += bool(cls)
        per_class.update(set(cls))
    return {"images": len(fp_test), "with_false_alarm": hits, "share": round(hits / len(fp_test), 3),
            "by_class": dict(per_class)}

fa_v1 = false_alarms(str(v1_weights))
print("bootstrap_v1:", fa_v1)

# %% [markdown]
# ## Train bootstrap_v2 (same recipe as v1, plus negatives)

# %%
model = YOLO(INIT)
model.train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, batch=BATCH, imgsz=IMGSZ, seed=SEED,
            deterministic=False, device=0, workers=4, project=str(work / "runs"), name="bootstrap_v2", exist_ok=True)

# %% [markdown]
# ## Results: per-class metrics on bootstrap_val, false alarms before / after

# %%
best = work / "runs" / "bootstrap_v2" / "weights" / "best.pt"
results = {}
for name, w in (("bootstrap_v1", v1_weights), ("bootstrap_v2", best)):
    m = YOLO(str(w)).val(data=str(data_yaml), imgsz=IMGSZ, batch=BATCH, device=0, plots=False, verbose=False)
    per = {CLASSES[int(c)]: {"precision": round(float(m.box.p[i]), 3), "recall": round(float(m.box.r[i]), 3),
                             "mAP50": round(float(m.box.ap50[i]), 3), "mAP50-95": round(float(m.box.ap[i]), 3)}
           for i, c in enumerate(m.box.ap_class_index)}
    results[name] = {"all": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                             "mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)},
                     "per_class": per}
results["bootstrap_v1"]["false_alarms"] = fa_v1
results["bootstrap_v2"]["false_alarms"] = false_alarms(str(best))
results["setup"] = {"train_bootstrap": len(train), "train_coco_negatives": len(neg_train), "train_own_negatives": len(own),
                    "val": len(val), "fp_test": len(fp_test), "epochs": EPOCHS, "alert_conf": ALERT_CONF,
                    "exclude_categories": EXCLUDE_CATEGORIES, "ultralytics": "8.4.179"}
print(json.dumps(results, indent=2))
(work / "comparison.json").write_text(json.dumps(results, indent=2))

# %%
out = work / "bootstrap_v2_results.zip"
run = work / "runs" / "bootstrap_v2"
with zipfile.ZipFile(out, "w") as z:
    for p in run.rglob("*"):
        if p.is_file() and (p.suffix in {".pt", ".csv", ".png", ".yaml"}):
            z.write(p, Path("bootstrap_v2") / p.relative_to(run))
    z.write(work / "comparison.json", "bootstrap_v2/comparison.json")
    z.write(work / "coco_negatives_attribution.csv", "bootstrap_v2/coco_negatives_attribution.csv")
print("Download from the Output tab:", out, f"{out.stat().st_size / 1e6:.0f} MB")
