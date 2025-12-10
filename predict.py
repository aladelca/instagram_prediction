import argparse
import json
import lzma
import re
from pathlib import Path
from typing import Dict, List

import joblib
import torch
from PIL import Image
from torchvision import transforms

from models import MultiModalRegressor, get_device

# NLTK helpers
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover
    raise RuntimeError("NLTK is required. Please install it before running predict.py")


def ensure_nltk():
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:  # pragma: no cover
            nltk.download(pkg)


def clean_caption(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and tok.isalnum()]
    return " ".join(tokens)


def load_sample(folder: Path) -> Dict:
    json_path = next(folder.glob("*.json.xz"))
    txt_path = next(folder.glob("*.txt"))
    image_paths = sorted([p for p in folder.glob("*.jpg")])
    with lzma.open(json_path) as f:
        meta = json.load(f)
    node = meta.get("node", {})
    try:
        caption = node.get("edge_media_to_caption", {}).get("edges", [])[0]["node"].get("text", "")
    except Exception:
        caption = ""
    if txt_path.exists():
        try:
            caption = txt_path.read_text(encoding="utf-8") or caption
        except Exception:
            pass
    clean = clean_caption(caption)

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

    meta_raw = torch.tensor([followers, following, is_professional, is_verified, float(num_photos), num_tagged, is_video], dtype=torch.float32)

    return {
        "id": folder.name,
        "caption_clean": clean,
        "image_paths": image_paths,
        "meta_raw": meta_raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict likes for posts")
    parser.add_argument("--data_dir", default="data", type=str, help="Folder with post subfolders or single post folder")
    parser.add_argument("--model", default="processed/model.pt")
    parser.add_argument("--vectorizer", default="processed/vectorizer.joblib")
    parser.add_argument("--text_scaler", default="processed/text_scaler.joblib")
    parser.add_argument("--meta_scaler", default="processed/meta_scaler.joblib")
    args = parser.parse_args()

    ensure_nltk()

    device = get_device()
    ckpt = torch.load(args.model, map_location=device)
    vectorizer = joblib.load(args.vectorizer)
    text_scaler = joblib.load(args.text_scaler)
    meta_scaler = joblib.load(args.meta_scaler)

    model = MultiModalRegressor(text_dim=ckpt["text_dim"], meta_dim=7)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    data_path = Path(args.data_dir)
    if data_path.is_dir() and any(str(p).endswith(".json.xz") for p in data_path.iterdir()):
        folders = [data_path]
    else:
        folders = sorted([p for p in data_path.iterdir() if p.is_dir()])

    tfm = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    for folder in folders:
        sample = load_sample(folder)
        text_vec = vectorizer.transform([sample["caption_clean"]]).toarray()
        text_scaled = torch.tensor(text_scaler.transform(text_vec), dtype=torch.float32, device=device).squeeze(0)

        meta_scaled_np = meta_scaler.transform(sample["meta_raw"].unsqueeze(0).numpy())
        meta_scaled = torch.tensor(meta_scaled_np.squeeze(0), dtype=torch.float32, device=device)

        images: List[torch.Tensor] = []
        for img_path in sample["image_paths"]:
            with Image.open(img_path).convert("RGB") as img:
                images.append(tfm(img).to(device))
    with torch.no_grad():
        pred_log = model(images, text_scaled, meta_scaled).item()
        pred = torch.expm1(torch.tensor(pred_log)).item()
        print(f"{sample['id']}: predicted likes={pred:.2f}")


if __name__ == "__main__":
    main()
