
"""
field_param_tuner.py

A lightweight parameter-tuning harness for your baseball field recognition pipeline.
- Uses random search with reproducible seeding.
- Scores each parameter set via IoU/F1 on masks or via proxy features.

Quick start:
    python field_param_tuner.py \
        --images_dir data/dev_images \
        --masks_dir  data/dev_masks \
        --trials 300 \
        --seed 13 \
        --save_dir ./tuning_runs/run1
"""

import os, json, argparse, random, time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None

# =========================
# 1) DEFINE YOUR PARAMETER SPACE
# =========================
PARAM_SPACE: Dict[str, Any] = {
    # HSV thresholds for "green" (grass)
    "h_low_g":   (25, 45),
    "h_high_g":  (65, 90),
    "s_low_g":   (40, 90),
    "s_high_g":  (200, 255),
    "v_low_g":   (30, 80),
    "v_high_g":  (200, 255),

    # HSV thresholds for "brown" (dirt)
    "h_low_d":   (5, 20),
    "h_high_d":  (25, 40),
    "s_low_d":   (40, 100),
    "s_high_d":  (200, 255),
    "v_low_d":   (20, 60),
    "v_high_d":  (200, 255),

    # Morphology
    "morph_kernel": [3, 5, 7, 9, 11],
    "morph_iters":  [0, 1, 2, 3, 4],

    # Contour filtering
    "min_area_frac": (0.02, 0.3),
    "max_eccentricity": (1.2, 2.2),

    # Hough Lines (if you use cv2.HoughLinesP)
    "hl_threshold": (40, 180),
    "hl_minLineLength": (20, 300),
    "hl_maxLineGap": (5, 40),

    # Hough Circles (if used; currently only used if you add it)
    "hc_dp": (0.8, 2.0),
    "hc_minDist": (20, 200),
    "hc_param1": (50, 180),
    "hc_param2": (20, 120),
    "hc_minRadius": (5, 80),
    "hc_maxRadius": (60, 250),
}

# =========================
# 2) SAMPLING
# =========================
def sample_params(rng: random.Random) -> Dict[str, Any]:
    params = {}
    for k, v in PARAM_SPACE.items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if float(lo).is_integer() and float(hi).is_integer():
                params[k] = rng.randint(int(lo), int(hi))
            else:
                params[k] = rng.uniform(float(lo), float(hi))
        elif isinstance(v, list):
            params[k] = rng.choice(v)
        else:
            raise ValueError(f"Unsupported spec for {k}: {v}")
    return params

# =========================
# 3) YOUR PIPELINE
# =========================
def run_pipeline(img_bgr: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal field pipeline driven by PARAM_SPACE knobs.
    Returns:
      {"mask": (H,W) uint8 0/255, "features": {...}}.
    """
    if img_bgr is None:
        return {"mask": None, "features": {}}

    H, W = img_bgr.shape[:2]
    imgrgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    imghsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- Color masks ---
    def in_range_mask(hsv, h_low, h_high, s_low, s_high, v_low, v_high):
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    grass_mask = in_range_mask(
        imghsv,
        params["h_low_g"], params["h_high_g"],
        params["s_low_g"], params["s_high_g"],
        params["v_low_g"], params["v_high_g"],
    )
    dirt_mask = in_range_mask(
        imghsv,
        params["h_low_d"], params["h_high_d"],
        params["s_low_d"], params["s_high_d"],
        params["v_low_d"], params["v_high_d"],
    )
    mask = cv2.bitwise_or(grass_mask, dirt_mask)

    # --- Morphology ---
    k = int(params["morph_kernel"])
    k = max(1, k | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    iters = int(params["morph_iters"])
    if iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=max(0, iters - 1))

    # --- Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field_mask = np.zeros((H, W), dtype=np.uint8)
    features: Dict[str, float] = {}

    if not contours:
        return {"mask": field_mask, "features": {"found": 0.0}}

    maxcont = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(maxcont))
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(maxcont)
    rect_area = float(rect_w * rect_h) if rect_w * rect_h > 0 else 1.0
    hull_area = float(cv2.contourArea(cv2.convexHull(maxcont))) if len(maxcont) >= 3 else 1.0
    perimeter = float(cv2.arcLength(maxcont, True)) if len(maxcont) >= 2 else 1.0

    extent = area / rect_area
    solidity = area / hull_area
    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
    aspect = rect_w / rect_h if rect_h > 0 else 1.0

    min_area_frac = float(params["min_area_frac"])
    if area < min_area_frac * (H * W):
        return {"mask": field_mask, "features": {"found": 0.0, "area_frac": area / (H * W)}}

    # aspect tolerance
    max_ecc = float(params["max_eccentricity"])
    if aspect > max_ecc and (1.0 / aspect) > max_ecc:
        return {"mask": field_mask, "features": {"found": 0.0, "aspect": aspect}}

    cv2.drawContours(field_mask, [maxcont], -1, 255, -1)

    # --- Hough lines (optional features) ---
    masked_rgb = cv2.bitwise_and(imgrgb, imgrgb, mask=field_mask)
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=int(params["hl_threshold"]),
        minLineLength=int(params["hl_minLineLength"]),
        maxLineGap=int(params["hl_maxLineGap"]),
    )

    line_count = 0
    mean_line_len = 0.0
    if lines is not None and len(lines) > 0:
        lens = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            lens.append(float(np.hypot(x2 - x1, y2 - y1)))
        if lens:
            line_count = len(lens)
            mean_line_len = float(np.mean(lens))

    features.update({
        "found": 1.0,
        "area_frac": area / (H * W),
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "aspect": aspect,
        "line_count": float(line_count),
        "mean_line_len": float(mean_line_len),
    })

    return {"mask": field_mask, "features": features}

# =========================
# 4) SCORING
# =========================
def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    true = (true_mask > 0).astype(np.uint8)
    inter = (pred & true).sum()
    union = (pred | true).sum()
    return float(inter) / float(union + 1e-9)

def precision_recall_f1(pred_mask: np.ndarray, true_mask: np.ndarray) -> Tuple[float,float,float]:
    pred = (pred_mask > 0).astype(np.uint8)
    true = (true_mask > 0).astype(np.uint8)
    tp = (pred & true).sum()
    fp = (pred & (1 - true)).sum()
    fn = ((1 - pred) & true).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1

def score_from_features(features: Dict[str, float]) -> float:
    """
    Proxy score when GT masks aren't available.
    Rewards plausible shape + enough coverage + some line structure.
    Higher = better (≈0..1.5).
    """
    if not features or features.get("found", 0.0) <= 0.0:
        return 0.0

    af = features.get("area_frac", 0.0)
    extent = features.get("extent", 0.0)
    solidity = features.get("solidity", 0.0)
    circ = features.get("circularity", 0.0)
    aspect = features.get("aspect", 1.0)
    line_count = features.get("line_count", 0.0)
    mean_len = features.get("mean_line_len", 0.0)

    def clamp01(x): 
        return max(0.0, min(1.0, x))

    # Area sweet spot ~[0.05, 0.40]
    area_term = clamp01((af - 0.02) / 0.10) * clamp01((0.45 - af) / 0.10)
    extent_term = clamp01((extent - 0.5) / 0.3)
    solidity_term = clamp01((solidity - 0.8) / 0.2)
    circ_term = clamp01((circ - 0.2) / 0.3)
    aspect_term = 1.0 - clamp01(abs(aspect - 1.0) / 0.7)
    line_term = clamp01(line_count / 12.0) * 0.3 + clamp01(mean_len / 150.0) * 0.2

    score = (
        0.25 * area_term +
        0.20 * extent_term +
        0.20 * solidity_term +
        0.15 * aspect_term +
        0.10 * circ_term +
        0.10 * line_term
    )
    return float(score)

def evaluate_on_dataset(images_dir: Path, masks_dir: Path, params: Dict[str, Any]) -> Dict[str, float]:
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    have_gt = masks_dir is not None and masks_dir.exists()

    ious, f1s, precisions, recalls = [], [], [], []
    feature_scores = []

    if cv2 is None:
        raise RuntimeError("OpenCV not found. Please install opencv-python.")

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        out = run_pipeline(img, params)

        if have_gt:
            gt_mask_path = masks_dir / (img_path.stem + ".png")
            if gt_mask_path.exists():
                gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    continue
                pmask = out.get("mask", None)
                if pmask is None:
                    continue
                iou = iou_score(pmask, gt)
                p, r, f1 = precision_recall_f1(pmask, gt)
                ious.append(iou); f1s.append(f1); precisions.append(p); recalls.append(r)
            else:
                feature_scores.append(score_from_features(out.get("features", {})))
        else:
            feature_scores.append(score_from_features(out.get("features", {})))

    metrics = {}
    if ious:
        metrics["iou_mean"] = float(np.mean(ious))
        metrics["f1_mean"]  = float(np.mean(f1s))
        metrics["prec_mean"]= float(np.mean(precisions))
        metrics["recall_mean"]= float(np.mean(recalls))
        metrics["score"] = metrics["iou_mean"]
    else:
        metrics["feature_score_mean"] = float(np.mean(feature_scores)) if feature_scores else 0.0
        metrics["score"] = metrics["feature_score_mean"]

    return metrics

# =========================
# 5) MAIN TUNING LOOP
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Folder of images for evaluation")
    ap.add_argument("--masks_dir", type=str, default=None, help="Folder of ground-truth masks (optional)")
    ap.add_argument("--trials", type=int, default=200, help="Number of random samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./tuning_runs/run_" + str(int(time.time())))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best = {"score": -1e18, "params": None, "metrics": None, "trial": -1}
    history = []

    for t in range(1, args.trials + 1):
        params = sample_params(rng)
        metrics = evaluate_on_dataset(images_dir, masks_dir, params)
        score = metrics["score"]
        trial_rec = {"trial": t, "score": float(score), "params": params, "metrics": metrics}
        history.append(trial_rec)

        if score > best["score"]:
            best = {"score": float(score), "params": params, "metrics": metrics, "trial": t}
            with open(save_dir / "best_config.json", "w") as f:
                json.dump(best, f, indent=2)
            print(f"[{t}/{args.trials}] NEW BEST score={score:.5f} -> saved best_config.json")

        if t % 10 == 0:
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"[{t}/{args.trials}] checkpoint saved.")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "best_config.json", "w") as f:
        json.dump(best, f, indent=2)

    print("DONE. Best trial:", best["trial"])
    print("Best score:", best["score"])
    print("Best params:", json.dumps(best["params"], indent=2))

if __name__ == "__main__":
    main()
