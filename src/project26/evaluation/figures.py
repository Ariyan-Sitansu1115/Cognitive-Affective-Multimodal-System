import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_required_figures(hybrid_history, hybrid_targets, hybrid_preds, class_names, figures_dir):
    plt.figure(figsize=(8, 5)); plt.plot(hybrid_history['val_loss'], label='Hybrid val loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Proposed Model Validation Loss'); plt.tight_layout(); plt.savefig(f'{figures_dir}/fig01_loss_curves.png', dpi=300); plt.show()
    plt.figure(figsize=(6, 5)); cm = confusion_matrix(hybrid_targets, hybrid_preds); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Hybrid confusion matrix"); plt.tight_layout(); plt.savefig(f'{figures_dir}/fig02_confusion_or_scatter.png', dpi=300); plt.show()
    report = classification_report(hybrid_targets, hybrid_preds, target_names=class_names, output_dict=True, zero_division=0)
    per_class_f1 = __import__('pandas').Series({name: report[name]["f1-score"] for name in class_names})
    plt.figure(figsize=(7, 4)); per_class_f1.plot(kind="barh"); plt.xlabel("F1 score"); plt.title("Hybrid per-class F1"); plt.tight_layout(); plt.savefig(f'{figures_dir}/fig03_roc_pr_curve.png', dpi=300); plt.show()
