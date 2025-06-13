import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def load_scores(path):
    with open(path) as f:
        return [int(tok) for tok in f.read().split() if tok.isdigit()]

# load both classes
scores_human = load_scores("scores_human.txt")   # label 0
scores_llm   = load_scores("scores_llm.txt")     # label 1

y_true  = np.array([0] * len(scores_human) + [1] * len(scores_llm))
scores  = np.array(scores_human + scores_llm)    # larger means more human like

# sweep thresholds
thr = np.linspace(scores.min(), scores.max(), 200)
acc_list, tpr_list, fpr_list = [], [], []

for t in thr:
    y_pred = np.where(scores >= t, 0, 1)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tn / (tn + fp)           # human detection
    fpr = fp / (tn + fp)           # llm seen as human
    acc_list.append(acc)
    tpr_list.append(tpr)
    fpr_list.append(fpr)

best = int(np.argmax(acc_list))
print("best threshold", thr[best])
print("best accuracy", acc_list[best])
print("human detection", tpr_list[best])
print("false positive", fpr_list[best])

# draw threshold curves
plt.figure(figsize=(11,6))
plt.plot(thr, acc_list, label="accuracy")
plt.plot(thr, tpr_list, label="human detection")
plt.plot(thr, fpr_list, label="false positive", linestyle="--")
plt.xlabel("reactivity threshold")
plt.ylabel("metric")
plt.title("AIS performance versus threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("reactivity_threshold_curve4.png")

# draw ROC
fpr, tpr, _ = roc_curve(y_true, -scores)   # minus because low score marks llm
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, linewidth=2, label="ROC")
plt.plot([0,1],[0,1], linestyle="--", label="random")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title(f"ROC curve   AUC = {roc_auc:.3f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
