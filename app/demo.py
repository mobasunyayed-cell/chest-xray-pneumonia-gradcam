"""Step 7: Gradio demo - upload a chest X-ray, get a probability + Grad-CAM heatmap.

    pip install gradio
    python app/demo.py --ckpt outputs/models/best.pt [--share]

Research demo only. NOT a medical device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import CFG  # noqa: E402
from src.gradcam import explain_pil  # noqa: E402
from src.model import build_model  # noqa: E402

DISCLAIMER = (
    "**Research demo - not a medical device.** Trained on pediatric chest X-rays "
    "(Kermany et al., Guangzhou) only. On the held-out external set specificity was "
    "~72% at the chosen threshold, so false alarms are common. "
    "**Adult or non-chest images are outside the training distribution, so the output is not meaningful.** "
    "The heatmap shows what raised the pneumonia score (it is weak/meaningless when the score is low); "
    "it is not proof of the cause."
)


def load_model(ckpt, device):
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model.eval()


def default_threshold() -> float:
    p = CFG.paths.figures / "metrics.json"
    return json.loads(p.read_text())["val"]["threshold"] if p.exists() else 0.5


def build_app(model, device, thr):
    def analyze(img, threshold):
        if img is None:
            return None, "Please upload an image."
        prob, cam = explain_pil(model, img, device)
        label = "Pneumonia pattern suspected" if prob >= threshold else "No pneumonia pattern found"
        text = f"{label}\nP(pneumonia) = {prob:.3f}   (threshold {threshold:.2f})"
        return (cam * 255).astype(np.uint8), text

    with gr.Blocks(title="Explainable chest X-ray pneumonia detection") as demo:
        gr.Markdown("# Explainable Chest X-ray Pneumonia Detection")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Chest X-ray")
            out = gr.Image(label="Grad-CAM overlay")
        slider = gr.Slider(0.05, 0.95, value=thr, step=0.01, label="Decision threshold")
        result = gr.Textbox(label="Result", lines=2)
        gr.Button("Analyze").click(analyze, [inp, slider], [out, result])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.paths.models / "best.pt"))
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--share", action="store_true", help="public temporary link (e.g. on Kaggle)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    thr = args.thr if args.thr is not None else default_threshold()
    build_app(model, device, thr).launch(share=args.share)


if __name__ == "__main__":
    main()
