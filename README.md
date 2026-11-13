# Explainable Chest X-ray Pneumonia Detection

A PyTorch pipeline that classifies pediatric chest X-rays as **NORMAL vs PNEUMONIA** with a DenseNet121 and explains each prediction with **Grad-CAM** heatmaps. The focus is on an honest evaluation: patient-level splits, a threshold chosen on validation data only, an external test set, and a check for shortcut learning.

> **Disclaimer:** research / educational project. Not a medical device and not for clinical use.

## Results

Trained on Kaggle (GPU), seed 42, early stopping on validation AUROC (best at epoch 7 of 11 run). Decision threshold **0.2901**, chosen on the validation set (Youden's J) and then frozen.

| Split | Images | Accuracy | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|---|
| val | 751 | 0.988 | 0.988 | 0.989 | 0.9987 |
| test (same distribution as train) | 764 | 0.987 | 0.983 | 1.000 | 0.9996 |
| **external** (original `test/` folder) | 624 | 0.894 | 0.997 | 0.722 | 0.979 |

Confusion counts (TN / FP / FN / TP): val 187 / 2 / 7 / 555, test 192 / 0 / 10 / 562, external 169 / 65 / 1 / 389.

**How to read this**
- The *test* split comes from the same pool as training (split by patient), so it is an optimistic upper bound.
- The *external* split is the original Kaggle test folder, which is distributed differently. It is the more honest estimate of generalisation: ranking quality stays high (AUROC 0.979) but the operating point shifts: 65 of 234 normal images are flagged as pneumonia at the validation-chosen threshold.
- All data are pediatric (ages 1 to 5). Results do not transfer to adults.

Figures (add after running): `docs/figures/roc.png`, `confusion.png`, `history.png`, `gradcam_test.png`, `gradcam_external.png`.

## Method

| Step | What | Where |
|---|---|---|
| Split | Pool original train+val (5,232 images), split **by patient** 70/15/15 with `StratifiedGroupKFold`; patient ids are parsed from file names. Original `test/` is kept untouched as *external*. | `src/data.py` |
| Imbalance | About 74% of images are pneumonia. Loss uses `pos_weight = n_neg / n_pos` (0.352). | `src/data.py`, `src/train.py` |
| Augmentation | Small rotation/translation/scale, brightness/contrast. No horizontal flip (anatomy). | `src/data.py` |
| Model | ImageNet-pretrained DenseNet121, dropout 0.3, one output logit. | `src/model.py` |
| Training | AdamW, lr 1e-4, cosine schedule, mixed precision, early stopping on val AUROC. | `src/train.py` |
| Evaluation | Sensitivity, specificity, precision, F1, AUROC, AUPRC, ROC and confusion plots. | `src/evaluate.py` |
| Explainability | Self-contained Grad-CAM on the last feature maps for the pneumonia logit. | `src/gradcam.py` |
| Demo | Gradio app: image in, probability and heatmap out, adjustable threshold. | `app/demo.py` |

## Is the model using shortcuts?

Grad-CAM on correct pneumonia cases mostly highlights lung fields. However, some normal and false-positive images show hot spots at corners or edges (for example the "R" laterality marker). To test this, `src/shortcut_check.py` hides regions and re-evaluates at the same threshold:

| Split | Variant | Accuracy | Specificity | AUROC |
|---|---|---|---|---|
| test | original | 0.987 | 1.000 | 0.9996 |
| test | corners masked | 0.987 | 0.974 | 0.9989 |
| test | border masked (12%) | 0.834 | 0.339 | 0.9963 |
| external | original | 0.894 | 0.722 | 0.9790 |
| external | corners masked | 0.851 | 0.607 | 0.9791 |
| external | border masked (12%) | 0.647 | 0.060 | 0.9355 |

Masking the corners leaves AUROC essentially unchanged, so corner markers are not required for ranking. Masking the border hurts the external AUROC, but that also removes real peripheral lung and the masked image is itself out of distribution, so this test is **inconclusive** about the border. Fixed-threshold specificity drops in every masked variant, which mainly reflects score shift caused by the masking.

## Limitations

- Single dataset, single seed, one train/val/test partition; no confidence intervals.
- Patient ids come from file names; if the dataset's naming does not capture every patient, some leakage between splits could remain.
- Pediatric-only; the threshold chosen in-distribution does not transfer to the external set (specificity 72%).
- Grad-CAM is coarse (7x7 feature maps) and shows evidence for the pneumonia logit; it is not proof of the cause. For low-probability images the heatmap is not meaningful.
- The shortcut check is a coarse ablation, not a full causal analysis.

## Reproduce (Kaggle notebook with GPU and Internet on)

```bash
pip install -r requirements.txt
python -m src.data                 # writes data/processed/splits.csv and prints the split table
python -m src.train --smoke        # quick pipeline check
python -m src.train                # full training, saves outputs/models/best.pt + metrics
python -m src.gradcam              # heatmap galleries
python -m src.shortcut_check       # masking experiment
python app/demo.py                 # Gradio demo (add --share for a public link)
```
Dataset: add Kaggle's *Chest X-Ray Images (Pneumonia)* by Paul Mooney as notebook input. `notebooks/02_run_on_kaggle.ipynb` writes all source files and runs the steps in order.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
Tests cover patient-id parsing, patient-level split (no leakage, proportions, reproducibility, leak detection), class weights and the metric functions.

## Repository layout

```
src/        config, data, model, train, evaluate, gradcam, shortcut_check
app/        demo.py (Gradio)
notebooks/  01_data_eda.ipynb, 02_run_on_kaggle.ipynb
tests/      pytest suite
docs/       figures
```

## Data and credit

Dataset: Kermany, D. S. et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell 172(5), 2018; Kaggle mirror by Paul Mooney (see the dataset page for license terms).

## Development notes

Developed with the help of an AI coding assistant (Claude). Every number in this README was produced by running the code in this repository on Kaggle.

## License

MIT
