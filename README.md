# Multi-class Softmax Regression with Mini-Batch Stochastic Gradient Descent

Course project for **McGill COMP 551: Applied Machine Learning** — Mini-Project 2, Group 49, Fall 2020.  
Authors: Doreen Duoduaah, Ameer Ibrahim Osman, Ella Malone.

This repository contains a from-scratch NumPy implementation of multi-class logistic regression (softmax regression) trained with mini-batch stochastic gradient descent (SGD) and momentum. The full write-up is available in [Report/writeup.pdf](Report/writeup.pdf).

---

## Overview

The project implements two core components from scratch:

- **Multi-class softmax regression** (`code/softmax_Regression.py`) — one-hot encoded targets, softmax probabilities, and cross-entropy gradient.
- **Mini-batch SGD with momentum** (`code/GradientDescent.py`) — random mini-batch creation, velocity-style weight update, and a custom termination condition based on 5-fold cross-validation accuracy.

A custom grid-search hyper-parameter optimizer (`code/hyper_optimization.py`) searches over batch size, learning rate, and momentum using 5-fold cross-validation. Performance is compared against off-the-shelf scikit-learn K-Nearest Neighbours and Decision Tree classifiers (`code/comparison_other_classifier.py`).

---

## Datasets

| Dataset | Source | Instances | Features | Classes | Notes |
|---|---|---|---|---|---|
| Scikit-learn Digits | `sklearn.datasets.load_digits` | 1,797 | 64 (8×8 pixel images) | 10 (digits 0–9) | Already pre-processed by scikit-learn; no missing values. |
| OpenML Cardiotocography | `fetch_openml(name='cardiotocography', version=2)` | 2,126 | 35 | 3 (fetal states N, S, P) | Non-numeric metadata removed; min-max normalized to prevent softmax overflow. |

Dataset loading and normalization are handled in `code/loader.py`.

---

## Implementation Notes

### Softmax Regression

- A bias column is appended to the feature matrix during `fit` and `predict`.
- Targets are one-hot encoded with `sklearn.preprocessing.OneHotEncoder`.
- Predictions are made by selecting the class with the highest softmax probability.

### Mini-Batch SGD with Momentum

The weight update rule in `code/GradientDescent.py` is:

```
d_w = beta * w + (1 - beta) * grad
w   = w - learning_rate * d_w
```

where `grad` is the average cross-entropy gradient over the current mini-batch and `beta` is the momentum coefficient.

### Termination Condition

Training stops when any of the following is true:

1. The L2 norm of the gradient falls below `epsilon` (default `1e-5`).
2. The number of iterations reaches `max_iter` (default `1e4`).
3. The mean 5-fold cross-validation accuracy is no longer strictly increasing over the last 20 iterations.

The third condition returns the model at the best cross-validation accuracy rather than the final gradient-descent iterate, and it reduces runtime when convergence is slow.

### Hyper-Parameter Optimization

`code/hyper_optimization.py` performs a grid search with 5-fold cross-validation over:

| Hyper-parameter | Search range | Step size |
|---|---|---|
| Batch size | 50 to dataset size | 100 |
| Learning rate | 0.01 to 0.1 | 0.018 |
| Momentum beta | 0.9 to 0.99 | 0.018 |

The configuration with the highest average cross-validation accuracy is selected.

---

## Results

### Softmax Regression

| Dataset | Batch Size | Learning Rate | Momentum | Runtime | Accuracy |
|---|---|---|---|---|---|
| Digits | 950 | 0.027 | 0.9 | 3.7 s | 92.7% |
| Cardiotocography | 50 | 0.010 | 0.9 | 4.9 s | 77.8% |

L1 and L2 regularization (lambda = 0.1) were implemented but did not significantly improve accuracy; the report suggests this is likely because the softmax cross-entropy objective already has a unique global minimum in weight space.

### Comparison with Off-the-Shelf Classifiers

| Dataset | K-NN (optimal k) | Decision Tree (optimal depth) | Our Softmax Regression |
|---|---|---|---|
| Digits | 98.9% (k = 1) | 91.7% (depth = 20) | **92.7%** |
| Cardiotocography | 99.1% (k = 3) | 98.6% (depth = 5) | 77.8% |

On the Digits dataset, the softmax regressor outperformed the decision tree. On Cardiotocography, both off-the-shelf classifiers scored higher.

---

## Repository Structure

```
.
├── code/
│   ├── softmax_Regression.py          # Softmax regression model
│   ├── GradientDescent.py             # Mini-batch SGD with momentum and 5-fold CV termination
│   ├── hyper_optimization.py          # Grid-search hyper-parameter optimizer
│   ├── loader.py                      # Dataset loading and min-max normalization
│   ├── comparison_other_classifier.py # K-NN and Decision Tree baselines
│   └── main.py                        # Example end-to-end run
├── Report/
│   └── writeup.pdf                    # Full project report
└── README.md
```

## Usage

Run the complete pipeline from the `code/` directory:

```bash
cd code
python main.py
```

`main.py` performs the following steps:

1. Loads both datasets via `loader.Loader`.
2. Min-max normalizes the Cardiotocography features.
3. Trains the softmax regressor on Digits with default gradient descent.
4. Runs the hyper-parameter optimizer on Digits.
5. Compares K-NN and Decision Tree baselines on Cardiotocography.

To train on a different dataset or use the optimized hyper-parameters, instantiate the classes directly:

```python
from GradientDescent import GradientDescent
from softmax_Regression import softmax_Regression
from loader import Loader

loader = Loader()
X, y = loader.numpy_dataset01()   # or numpy_dataset02()

gd = GradientDescent(batch_size=950, learning_rate=0.027, momentum_beta=0.9)
model = softmax_Regression()
w_opt, encoder = model.fit(X, y, gd)

y_pred = model.predict(X, w_opt)
y_pred = encoder.inverse_transform(y_pred)
```

## Dependencies

- Python 3
- NumPy
- scikit-learn
- pandas
- Matplotlib

Install with pip:

```bash
pip install numpy scikit-learn pandas matplotlib
```

## References

- Cardiotocography dataset: https://www.openml.org/d/1466
- Full project report: [Report/writeup.pdf](Report/writeup.pdf)
