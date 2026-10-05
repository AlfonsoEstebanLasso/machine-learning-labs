# Machine Learning Labs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python 3](https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?style=flat-square)](https://gymnasium.farama.org/)

**Four hands-on notebooks spanning the core of applied ML — from SVMs on clinical data to CNN transfer learning with ResNet50 and tabular Q-learning 🧪**

Coursework project — BSc in Applied Data Science, Universitat Oberta de Catalunya (UOC), Machine Learning course.

Four Jupyter notebooks covering the main blocks of the course: supervised classification, regression with ensembles, deep learning for computer vision, and reinforcement learning. All notebooks were developed and executed on Google Colab and keep their original outputs. Notebook narrative translated to English from the original Spanish; printed outputs and figure labels are shown as originally executed (in Spanish).

## Objective

Practice the complete machine learning workflow on public datasets: loading and preprocessing data, training and comparing models, evaluating them with appropriate metrics, and discussing the results — from classical scikit-learn models to convolutional networks, transfer learning and tabular Q-learning.

## Notebooks

| Notebook | Topic | What it demonstrates |
| --- | --- | --- |
| [`01_supervised_classification.ipynb`](01_supervised_classification.ipynb) | Classification | SVM vs. decision tree on the Cleveland Heart Disease dataset: exploratory analysis, target binarization, confusion matrices, classification reports, grid-search hyperparameter tuning, and a discussion of robust evaluation in a clinical context. |
| [`02_regression_ensembles.ipynb`](02_regression_ensembles.ipynb) | Regression / ensembles | Linear regression, decision tree and random forest on the Appliances Energy Prediction dataset, compared with 5-fold cross-validation (MAE/MSE/RMSE), plus a gradient boosting model. |
| [`03_cnn_transfer_learning.ipynb`](03_cnn_transfer_learning.ipynb) | Deep learning / computer vision | A baseline CNN and a deeper CNN on CIFAR-10; transfer learning with an ImageNet-pretrained ResNet50 (feature extraction vs. full fine-tuning, with data augmentation) on horses-vs-humans; object localization on Fashion-MNIST by bounding-box regression, evaluated with IoU. |
| [`04_q_learning.ipynb`](04_q_learning.ipynb) | Reinforcement learning | Tabular Q-learning implemented from scratch for Gymnasium's CliffWalking-v0: Q-table, epsilon-greedy policy, Bellman update, 10,000-episode training loop with epsilon decay, episode videos and Q-table analysis. |

## Data & methods

No datasets are stored in this repository; each notebook downloads its data at runtime.

- [Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) — UCI Machine Learning Repository, fetched via `ucimlrepo`.
- [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — UCI Machine Learning Repository, downloaded as CSV.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — via `tensorflow.keras.datasets`.
- [horses_or_humans](https://www.tensorflow.org/datasets/catalog/horses_or_humans) — via `tensorflow_datasets`.

Models are built with scikit-learn (SVC, DecisionTree, RandomForest, GradientBoosting, GridSearchCV) and TensorFlow/Keras (sequential CNNs, ResNet50 backbone, a regression head for bounding boxes). The reinforcement learning notebook uses Gymnasium and records episodes with the `RecordVideo` wrapper.

## Key results

- **Classification (01):** the SVM reaches 0.90 test accuracy on the binarized heart-disease target, clearly ahead of the decision tree (0.69); grid search selects `C=0.1`, `kernel=rbf` (0.89 test accuracy).
- **CIFAR-10 (03):** the deeper CNN (three Conv2D + MaxPooling blocks with dropout) reaches about 76% validation accuracy, versus roughly 60% for the single-layer baseline, which overfits.
- **Transfer learning (03):** ResNet50 feature extraction reaches 99.5% validation accuracy on horses-vs-humans; unfreezing all 23.5M weights converges faster and touches 100% validation accuracy.
- **Object localization (03):** the bounding-box regression CNN scores 67% accuracy at an IoU threshold of 0.7 on the validation generator.
- **Q-learning (04):** the agent converges after roughly 4,000 episodes to the optimal 13-step path (episode reward −13).

Known limitation: in notebook 02, the target column (`Appliances`) is not removed from the feature matrix before scaling and splitting, so the reported regression errors are optimistic (target leakage). The notebook is kept as originally submitted and its error figures are therefore not quoted here.

## Attribution

Code cells marked `# Provided by course materials` contain scaffolding supplied with the assignments (dataset loaders, the Fashion-MNIST localization dataset generator, bounding-box visualization and IoU helpers). All remaining code and all written analysis are my own. Assignment statements were removed and replaced with brief English section headers.

## Tech stack

Python, scikit-learn, TensorFlow/Keras, tensorflow-datasets, Gymnasium, NumPy, pandas, Matplotlib, seaborn, moviepy, ucimlrepo.

## How to run

The notebooks were developed on Google Colab (notebook 03 benefits from a GPU runtime), and can be opened there directly. To run locally:

```bash
pip install -r requirements.txt
jupyter notebook
```

Notes:

- `moviepy` must be a 1.x version (`moviepy.editor` was removed in 2.0).
- Rendering CliffWalking episodes as video requires `pygame` (listed in the requirements).

## Repository structure

```
machine-learning-labs/
├── 01_supervised_classification.ipynb
├── 02_regression_ensembles.ipynb
├── 03_cnn_transfer_learning.ipynb
├── 04_q_learning.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```
