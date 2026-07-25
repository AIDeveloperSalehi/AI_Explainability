# MNIST Explainability Study

A demonstration of how a CNN can reach near-perfect accuracy on standard MNIST metrics while relying on a spurious, unintended feature — exposed through a distribution-shift test and Grad-CAM.

## Problem

A model that scores well on a held-out test set can still be right for the wrong reasons: it may key on a spurious correlation in the training data rather than the actual digit shape. Accuracy alone won't reveal this — you need to test the model on inputs that break the spurious correlation, and inspect *where* the model is looking.

## Approach

1. **Baseline (`Train_CNN_Normal_MNIST.ipynb`)** — a CNN trained on standard MNIST.
2. **Spurious-feature injection (`Train_CNN_Framed_MNIST.ipynb`)** — a modified training set where every digit "9" is drawn with an added frame/border, then trained on a *mixed* set (framed 9s + normal other digits). The model is evaluated every epoch on three test sets simultaneously: the mixed test set (same distribution as training), the original unmodified MNIST test set, and a fully framed test set (every digit, not just 9, gets the frame).
3. **Explainability (`Explainability.ipynb`)** — Grad-CAM is used to visualize which pixels the trained model actually attends to, to check whether it's using the digit shape or the frame.

## Result

At epoch 10 of training on the framed/mixed dataset (`Train_CNN_Framed_MNIST.ipynb`), the same model scores wildly differently depending on which test distribution it sees:

| Test set | Accuracy |
|---|---|
| Mixed test set (matches training distribution) | **99.22%** |
| Original, unmodified MNIST test set | **89.13%** |
| Fully framed test set (frame added to *every* digit, not just 9) | **10.24%** |

A model that looks like a 99%-accurate MNIST classifier collapses to near-random (10.24% ≈ chance level for 10 classes) the moment the frame appears on digits it wasn't trained to associate with a frame — strong evidence it partly learned "frame → predict 9" rather than the digit's shape. Grad-CAM visualizations in `Explainability.ipynb` show the model's attention shifting toward the frame region on framed inputs, consistent with this. This is the core point of the project: standard accuracy metrics alone would have hidden this failure mode entirely.

## How to run

```bash
git clone https://github.com/Hojat-Salehi/AI_Explainability.git
cd AI_Explainability
pip install -r requirements.txt
python download_MNIST.py
```

Then, in order:
1. `Train_CNN_Normal_MNIST.ipynb` — train the baseline CNN on standard MNIST.
2. `Train_CNN_Framed_MNIST.ipynb` — train on the framed/mixed dataset and reproduce the three-way accuracy table above.
3. `Explainability.ipynb` — run Grad-CAM on the trained models (`mnist_cnn.pth`, `mixed_mnist_cnn.pth`) to visualize the attention shift.

## Project Structure

- `Train_CNN_Normal_MNIST.ipynb` — trains a CNN on the standard MNIST dataset.
- `Train_CNN_Framed_MNIST.ipynb` — trains a CNN on the framed/mixed MNIST dataset and logs the three-way accuracy comparison above.
- `Explainability.ipynb` — Grad-CAM analysis of the trained models.
- `download_MNIST.py` — downloads the MNIST dataset.
- `Utils.py` — model definitions and shared utility functions.
- `mnist_cnn.pth` / `mixed_mnist_cnn.pth` — trained model weights.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
