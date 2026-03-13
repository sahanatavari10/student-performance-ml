"""Evaluation and visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", ax=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not at risk", "At risk"],
        yticklabels=["Not at risk", "At risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curve(model, X_test, y_test, title: str = "ROC Curve", ax=None):
    """Plot the ROC curve for a fitted model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    return ax


def print_report(y_true, y_pred):
    """Print a classification report."""
    print(classification_report(y_true, y_pred, target_names=["Not at risk", "At risk"]))
