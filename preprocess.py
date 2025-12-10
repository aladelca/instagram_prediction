import argparse
import json
import lzma
import re
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# NLTK imports with safe downloader
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover
    raise RuntimeError("NLTK is required. Please install it before running preprocess.py")


def ensure_nltk():
    """Ensure required NLTK resources are present."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for pkg, locator in resources.items():
        try:
            nltk.data.find(locator)
        except LookupError:  # pragma: no cover
            nltk.download(pkg)


def clean_caption(text: str) -> str:
    text = text or ""
    text = text.lower()
    # keep alphanumeric
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    try:
        tokens = word_tokenize(text)
    except LookupError:  # pragma: no cover
        ensure_nltk()
        tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and tok.isalnum()]
    return " ".join(tokens)


def extract_sample(folder: Path) -> Dict:
    json_path = next(folder.glob("*.json.xz"))
    # TODO: use caption from .json.xz if .txt not present
    txt_path = next(folder.glob("*.txt"))
    image_paths = sorted([p for p in folder.glob("*.jpg")])

    with lzma.open(json_path) as f:
        meta = json.load(f)
    node = meta.get("node", {})

    likes_raw = float(node.get("edge_media_preview_like", {}).get("count", 0.0))
    likes = max(likes_raw, 0.0)
    owner = node.get("owner", {})
    followers = float(owner.get("edge_followed_by", {}).get("count", 0.0))
    following = float(owner.get("edge_follow", {}).get("count", 0.0))
    is_professional = 1.0 if owner.get("is_professional_account", False) else 0.0
    is_verified = 1.0 if owner.get("is_verified", False) else 0.0

    sidecar = node.get("edge_sidecar_to_children")
    num_photos = len(sidecar.get("edges", [])) if sidecar else 1
    tagged = node.get("edge_media_to_tagged_user", {}).get("edges", []) or []
    num_tagged = float(len(tagged))
    is_video = 1.0 if node.get("is_video", False) else 0.0

    # caption
    try:
        caption = node.get("edge_media_to_caption", {}).get("edges", [])[0]["node"].get("text", "")
    except Exception:
        caption = ""
    if txt_path.exists():
        try:
            caption = txt_path.read_text(encoding="utf-8") or caption
        except Exception:
            pass

    cleaned = clean_caption(caption)

    return {
        "id": folder.name,
        "image_paths": [str(p) for p in image_paths],
        "caption_clean": cleaned,
        "meta_raw": np.array([followers, following, is_professional, is_verified, float(num_photos), num_tagged, is_video], dtype=np.float32),
        # almacenamos en log1p para estabilizar entrenamiento
        "target": np.log1p(likes),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Instagram dataset")
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--out_dir", default="processed", type=str)
    parser.add_argument("--max_features", default=5000, type=int)
    args = parser.parse_args()

    ensure_nltk()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    samples: List[Dict] = []
    for folder in folders:
        try:
            sample = extract_sample(folder)
            samples.append(sample)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] skipping {folder}: {exc}")
            continue

    captions = [s["caption_clean"] for s in samples]

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2)
    text_matrix = vectorizer.fit_transform(captions).astype(np.float32)
    text_dense = text_matrix.toarray()

    text_scaler = MinMaxScaler()
    text_scaled = text_scaler.fit_transform(text_dense).astype(np.float32)

    meta_matrix = np.stack([s["meta_raw"] for s in samples])
    meta_scaler = MinMaxScaler()
    meta_scaled = meta_scaler.fit_transform(meta_matrix).astype(np.float32)

    targets = np.array([s["target"] for s in samples], dtype=np.float32)

    processed = []
    for idx, sample in enumerate(samples):
        processed.append({
            "id": sample["id"],
            "image_paths": sample["image_paths"],
            "text_vector": text_scaled[idx],
            "meta_vector": meta_scaled[idx],
            "target": targets[idx],
        })

    torch.save(processed, out_dir / "processed_data.pt")
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(text_scaler, out_dir / "text_scaler.joblib")
    joblib.dump(meta_scaler, out_dir / "meta_scaler.joblib")

    print(f"Saved {len(processed)} samples")
    print(f"Text dim: {text_scaled.shape[1]} | Meta dim: {meta_scaled.shape[1]}")


if __name__ == "__main__":
    main()
