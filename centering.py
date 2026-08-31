#!/usr/bin/env python3
"""
Cross-frame character centering.

Once every frame of an animation state has been cropped into one uniform
canvas, we still need each frame's character to sit at the SAME horizontal
position, or the sprite jitters side to side during playback. The hard
part: transient stuff -- a billowing cape, swinging hands, flying hair, a
held item -- is not part of the "character position" but naive measures
get dragged around by it.

Methods, best -> simplest:

  feature   ORB keypoint matches between each frame and a reference frame;
            the horizontal shift is the MAD-robust median of matched-point
            dx. Keypoints on the rigid core (face, eyes, collar, belt)
            match consistently and agree on one shift; keypoints on the
            cape / hair / hands are inconsistent frame to frame and fall
            outside the robust band, so they're ignored. Handles front
            views, side walks and jumps with no anatomy assumptions.
            Needs OpenCV; individual frames with too few matches fall back
            to `centroid`, and if OpenCV is missing the whole run does.

  centroid  Center of mass of the silhouette, restricted to a head+torso
            vertical band (skip the top hair spikes and the leg/foot zone
            where a stride dominates). Pure NumPy. A large one-sided cape
            still pulls it a little, but far less than bbox.

  bbox      Midpoint of the full silhouette bounding box -- the original
            behavior. Any one-sided appendage moves it by half the
            appendage's reach. Kept for comparison and as a last resort.

Only a HORIZONTAL offset is estimated and applied. Vertical position is
left exactly as authored so jump arcs and crouches survive (same contract
as extract_frames).
"""
import numpy as np

BG_EXACT = (0, 255, 0)

# head+torso band as fractions of the per-frame silhouette height:
# skip the top (hair spikes vary a lot) and stop above the legs.
_CORE_TOP = 0.10
_CORE_BOT = 0.58


def _silhouette_rows(mask):
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1]


def _bbox_mid(mask):
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return mask.shape[1] / 2.0
    return (cols[0] + cols[-1]) / 2.0


def _core_band(mask):
    yr = _silhouette_rows(mask)
    if yr is None:
        return 0, mask.shape[0]
    y0, y1 = yr
    h = y1 - y0
    a = int(round(y0 + _CORE_TOP * h))
    b = int(round(y0 + _CORE_BOT * h))
    return a, max(b, a + 1)


def _core_centroid(mask):
    """Mean x of foreground pixels inside the head+torso band. Falls back
    to the whole silhouette, then the canvas center, if that band is
    empty."""
    a, b = _core_band(mask)
    band = mask[a:b]
    xs = np.nonzero(band)[1]
    if len(xs) == 0:
        xs = np.nonzero(mask)[1]
    if len(xs) == 0:
        return mask.shape[1] / 2.0
    return float(xs.mean())


# --------------------------------------------------------------------------
# per-method offset estimators. Each returns a float ndarray of length
# n_frames: the x offset (pixels, +right) to add to that frame so the
# character lands centered.
# --------------------------------------------------------------------------

def _offsets_bbox(masks, W):
    return np.array([W / 2.0 - _bbox_mid(m) for m in masks])


def _offsets_centroid(masks, W):
    return np.array([W / 2.0 - _core_centroid(m) for m in masks])


def _pick_reference(masks):
    """Frame with the median foreground-pixel count -- a middle-of-the-road
    pose, not a full stride or the top of a jump. Ties break toward the
    middle frame."""
    counts = np.array([int(m.sum()) for m in masks])
    order = np.argsort(counts, kind="stable")
    return int(order[len(order) // 2])


def _offsets_feature(frames_rgb, masks, W, min_matches=8):
    """ORB match every frame to a reference; shift = MAD-robust median of
    matched keypoint dx. Returns None if OpenCV is unavailable. Frames
    that don't get enough matches fall back to their core centroid."""
    try:
        import cv2
    except ImportError:
        return None

    grays = []
    for rgb, m in zip(frames_rgb, masks):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g[~m] = 0  # blank the background so no keypoints land on it
        grays.append(g)

    ref = _pick_reference(masks)
    orb = cv2.ORB_create(nfeatures=1500)
    kp_ref, des_ref = orb.detectAndCompute(grays[ref], None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # anchor: put the reference frame's core centroid on the canvas center
    ref_off = W / 2.0 - _core_centroid(masks[ref])
    centroid_fallback = _offsets_centroid(masks, W)
    offs = np.array(centroid_fallback, dtype=float)
    offs[ref] = ref_off
    matched = 0

    for i, g in enumerate(grays):
        if i == ref:
            continue
        kp, des = orb.detectAndCompute(g, None)
        if des is None or des_ref is None or len(kp) < 3:
            continue
        matches = bf.match(des, des_ref)
        if len(matches) < min_matches:
            continue
        dxs = np.array([kp_ref[mt.trainIdx].pt[0] - kp[mt.queryIdx].pt[0]
                        for mt in matches])
        # two MAD passes to shed cape / hair / hand outliers
        keep = dxs
        for _ in range(2):
            med = np.median(keep)
            mad = np.median(np.abs(keep - med)) + 1e-6
            nxt = keep[np.abs(keep - med) < 3.0 * mad]
            if len(nxt) < 5:
                break
            keep = nxt
        shift = float(np.median(keep))
        # a degenerate frame (nearly empty, few keypoints) can yield a wild
        # match; a real inter-frame drift is never half the canvas, so
        # reject that and keep this frame's centroid fallback.
        if abs(shift) <= 0.5 * W:
            offs[i] = ref_off + shift
            matched += 1

    # if hardly any frame produced a usable match (low-texture / tiny
    # sprite) the ORB estimate is untrustworthy -- let the caller fall back
    others = len(masks) - 1
    if others > 0 and matched < max(1, others // 2):
        return None
    return offs


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

# if the raw silhouettes barely move frame to frame (spread below this, in
# source px) the state is already aligned -- skip the feature step, whose
# keypoint noise could otherwise inject a jitter that wasn't there.
_LOW_DRIFT_PX = 3.0


def compute_offsets(frames_rgb, masks, method="feature"):
    """Estimate an integer horizontal offset per frame.

    Returns (offsets:list[int], method_used:str). `method_used` may differ
    from `method`: `feature` downgrades to `centroid` if OpenCV is
    missing, or if the state has no real drift to correct.
    """
    W = masks[0].shape[1]
    method_used = method

    if method == "feature":
        bbox_off = _offsets_bbox(masks, W)
        drift = float(bbox_off.max() - bbox_off.min())
        if drift <= _LOW_DRIFT_PX:
            method_used = f"centroid (feature skipped: drift {drift:.1f}px)"
            offs = _offsets_centroid(masks, W)
        else:
            offs = _offsets_feature(frames_rgb, masks, W)
            if offs is None:
                method_used = "centroid (feature: OpenCV not installed)"
                offs = _offsets_centroid(masks, W)
    elif method == "centroid":
        offs = _offsets_centroid(masks, W)
    elif method == "bbox":
        offs = _offsets_bbox(masks, W)
    else:
        raise ValueError(f"unknown centering method: {method!r}")

    # global recenter: after shifting, the mean core-centroid should sit on
    # the canvas center (keeps the group from drifting off to one side).
    shifted_centroids = np.array([_core_centroid(m) + o
                                  for m, o in zip(masks, offs)])
    offs = offs - (shifted_centroids.mean() - W / 2.0)

    return [int(round(o)) for o in offs], method_used


def apply_offsets(frames_rgb, masks, offsets, bg_rgb=BG_EXACT):
    """Roll each frame (and its mask) horizontally by its offset, filling
    the exposed edge with the background color. Returns new lists."""
    out_rgb, out_mask = [], []
    bg = np.array(bg_rgb, dtype=np.uint8)
    for rgb, m, dx in zip(frames_rgb, masks, offsets):
        r = np.roll(rgb, dx, axis=1)
        mm = np.roll(m, dx, axis=1)
        if dx > 0:
            r[:, :dx] = bg
            mm[:, :dx] = False
        elif dx < 0:
            r[:, dx:] = bg
            mm[:, dx:] = False
        out_rgb.append(r)
        out_mask.append(mm)
    return out_rgb, out_mask


def residual_core_jitter(masks):
    """Spread (max-min) of the HEAD centroid x across frames, in source
    pixels. A quick, method-neutral 'how aligned is it' number for the
    report -- the head is used (not the torso) because a cape covering the
    chest keeps the torso centroid noisy even when the character is
    perfectly aligned. Lower is better; a couple of px vanishes in the
    downscale."""
    cc = []
    for m in masks:
        yr = _silhouette_rows(m)
        if yr is None:
            continue
        y0, y1 = yr
        h = y1 - y0
        a = int(round(y0 + 0.02 * h))
        b = int(round(y0 + 0.30 * h))
        xs = np.nonzero(m[a:max(b, a + 1)])[1]
        if len(xs):
            cc.append(float(xs.mean()))
    return float(max(cc) - min(cc)) if cc else 0.0
