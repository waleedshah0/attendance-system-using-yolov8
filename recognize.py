# recognize.py
import os
import faiss
import numpy as np
import pickle
import traceback

SAVE_DIR = "embeddings"  # same as your project folder for faiss index and labels
INDEX_PATH = os.path.join(SAVE_DIR, "face_index.faiss")
LABELS_PATH = os.path.join(SAVE_DIR, "labels.pkl")
# optional: store raw_embeddings.npy (N x D) if you have them; used as a fallback for L2 checks
RAW_EMB_PATH = os.path.join(SAVE_DIR, "raw_embeddings.npy")

# Tune these thresholds for your setup:
COSINE_SIM_THRESHOLD = 0.45   # for normalized vectors + inner-product index (higher = more similar)
L2_DIST_THRESHOLD = 1.2       # for L2-distance indices (lower = more similar)

# Internal globals
_index = None
_labels = None
_has_raw_embeddings = False
_raw_embeddings = None
_index_metric = None  # "ip" or "l2" or None


def _safe_load():
    global _index, _labels, _has_raw_embeddings, _raw_embeddings, _index_metric

    if _index is not None and _labels is not None:
        return

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

    # Load index
    _index = faiss.read_index(INDEX_PATH)
    # Try to detect metric: many index types have .metric_type attribute (older/newer faiss)
    metric = None
    try:
        # Some faiss Python builds support index.metric_type
        metric = getattr(_index, "metric_type", None)
    except Exception:
        metric = None

    if metric is not None:
        # FAISS uses enum MetricType.METRIC_INNER_PRODUCT or METRIC_L2
        # inner product value is 1, l2 is 0 in some builds — we won't rely on exact numbers cross-build
        # We'll try to infer textually as fallback below.
        pass

    # Heuristic: class name often contains "IndexFlatIP" or "IndexIVFPQ" etc.
    clsname = _index.__class__.__name__.lower()
    if "ip" in clsname or "inner" in clsname:
        _index_metric = "ip"
    elif "l2" in clsname or "flatl2" in clsname or "distance" in clsname:
        _index_metric = "l2"
    else:
        # fallback: assume ip if index.reconstruct returns vectors in normalized range? we'll try search probe below
        _index_metric = None

    # Load labels
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
    with open(LABELS_PATH, "rb") as f:
        _labels = pickle.load(f)

    # Optional raw embeddings
    if os.path.exists(RAW_EMB_PATH):
        try:
            _raw_embeddings = np.load(RAW_EMB_PATH)
            _has_raw_embeddings = True
        except Exception:
            _has_raw_embeddings = False

    # If we still don't know metric, do a small probe:
    if _index_metric is None:
        try:
            # create a small random vector and search k=1: see value scale
            d = _index.d
            probe = np.random.randn(1, d).astype("float32")
            # If index is inner-product expecting normalized vectors, we should normalize probe
            faiss.normalize_L2(probe)
            D, I = _index.search(probe, 1)
            val = float(D[0][0])
            # Heuristic: inner-product values are often between -1..1 if vectors normalized, L2 distances are >=0 and often >0.1
            if val <= 1.5 and val >= -1.0:
                _index_metric = "ip"
            else:
                _index_metric = "l2"
        except Exception:
            _index_metric = "l2"

    print(f"[recognize.py] Loaded index ({INDEX_PATH}) metric={_index_metric}, labels={len(_labels)}")
    if _has_raw_embeddings:
        print(f"[recognize.py] Raw embeddings available ({RAW_EMB_PATH}), shape={_raw_embeddings.shape}")


def _normalize_vec(vec: np.ndarray):
    """L2 normalize single vector in-place, return a new array"""
    v = vec.astype("float32").copy()
    faiss.normalize_L2(v.reshape(1, -1))
    return v


def recognize_face(embedding: np.ndarray, threshold=None, topk=1):
    """
    Recognize a single normalized or non-normalized embedding.
    Returns: (name, score)
      - If using inner-product style index, score is cosine similarity (higher better).
      - If using L2 index, score is L2 distance (lower better).
    """
    try:
        _safe_load()
    except Exception as e:
        print(f"[recognize.py][ERROR] _safe_load failed: {e}")
        traceback.print_exc()
        return "Unknown", None

    if embedding is None:
        return "Unknown", None

    q = np.array(embedding).astype("float32").reshape(1, -1)
    d = q.shape[1]
    if hasattr(_index, "d") and _index.d != d:
        print(f"[recognize.py][WARN] Query dim {d} != index dim {_index.d}. Attempting to continue.")

    # Metric-specific handling
    if _index_metric == "ip":
        # Normalize query and use inner product as cosine similarity
        qn = _normalize_vec(q)
        D, I = _index.search(qn, topk)  # D contains similarity scores (larger better)
        score = float(D[0][0])
        idx = int(I[0][0])
        if threshold is None:
            threshold = COSINE_SIM_THRESHOLD
        if score >= threshold:
            name = _labels[idx] if idx < len(_labels) else "Unknown"
            return name, float(score)
        else:
            return "Unknown", float(score)

    else:
        # treat as L2-style index (smaller is better)
        D, I = _index.search(q, topk)
        dist = float(D[0][0])
        idx = int(I[0][0])

        # If we have raw embeddings, we can compute exact L2 or cosine for additional verification
        if _has_raw_embeddings:
            try:
                candidate = _raw_embeddings[idx].astype("float32").reshape(1, -1)
                # compute both
                # L2:
                l2 = np.linalg.norm(q - candidate)
                # cosine:
                qq = _normalize_vec(q)
                cc = _normalize_vec(candidate)
                cosine = float((qq @ cc.T).squeeze())
                # Prefer using l2 for final decision if user supplied L2 threshold
                if threshold is None:
                    threshold = L2_DIST_THRESHOLD
                if l2 <= threshold:
                    return _labels[idx], float(l2)
                else:
                    # fail but return similarity as info (negative)
                    return "Unknown", float(l2)
            except Exception:
                # fallback to faiss-provided distance
                pass

        # No raw embeddings, rely on faiss result
        if threshold is None:
            threshold = L2_DIST_THRESHOLD
        if dist <= threshold:
            name = _labels[idx] if idx < len(_labels) else "Unknown"
            return name, float(dist)
        else:
            return "Unknown", float(dist)


# Convenience wrapper to expose previous signature
def recognize_face_simple(embedding, threshold=None):
    return recognize_face(embedding, threshold=threshold)
