import os
import cv2
import numpy as np
import faiss
import pickle
import argparse
from tqdm import tqdm
from insightface import app

def get_arcface_embedding(img, face_app, min_score=0.5):
    """Extract ArcFace embeddings from all faces in an image using InsightFace."""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)

        embeddings = []
        for face in faces:
            if face.det_score < min_score:
                print("[WARN] Low quality face detected — skipping")
                continue
            emb = face.embedding.astype("float32")
            emb = emb / np.linalg.norm(emb)  # normalize for cosine similarity
            embeddings.append(emb)

        if not embeddings:
            print("[WARN] No valid faces detected")
            return None

        return embeddings

    except Exception as e:
        print(f"[ERROR] Failed to extract ArcFace embedding: {e}")
        return None


def build_embeddings(employee_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Initializing InsightFace model (CPU mode)...")
    face_app = app.FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []
    labels = []

    print(f"[INFO] Building embeddings from directory: {employee_dir}\n")

    for person_name in tqdm(os.listdir(employee_dir), desc="Processing Employees"):
        person_path = os.path.join(employee_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            embs = get_arcface_embedding(img, face_app)
            if embs is not None:
                for emb in embs:
                    embeddings.append(emb)
                    labels.append(person_name)
                print(f"[INFO] Added {len(embs)} embeddings for {person_name} - {img_name}")
            else:
                print(f"[WARN] No embedding extracted for {person_name} - {img_name}")

    if len(embeddings) == 0:
        print("[ERROR] No embeddings found. Please check your images.")
        return

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print(f"\n[INFO] Creating FAISS index with dimension {dim} (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)  # inner product (for cosine)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(save_dir, "face_index.faiss"))
    with open(os.path.join(save_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    print(f"\n✅ [SUCCESS] Saved FAISS DB with {len(embeddings)} ArcFace embeddings in '{save_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ArcFace embeddings database using InsightFace + FAISS")
    parser.add_argument("--dir", type=str, default="employee_faces", help="Directory containing employee face folders")
    parser.add_argument("--save", type=str, default="embeddings", help="Directory to save FAISS index and labels")

    args = parser.parse_args()
    build_embeddings(args.dir, args.save)
