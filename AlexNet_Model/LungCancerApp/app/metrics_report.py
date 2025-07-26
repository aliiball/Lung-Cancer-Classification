import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    cohen_kappa_score, accuracy_score, roc_auc_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================
# Ayarlar
# ======================
model_path = "C:\\Projects\\AlexNet_Model\\LungCancerApp\\app\\alexnet_best_model.h5"
val_dir = "C:\\Projects\\Datasets\\lung_split_dataset\\validation"
image_size = (224, 224)
batch_size = 32
save_dir = "grafikler"
os.makedirs(save_dir, exist_ok=True)

# ======================
# Modeli Yükle
# ======================
model = load_model(model_path)

# ======================
# Doğrulama Verisini Yükle
# ======================
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ======================
# Tahminler
# ======================
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# ======================
# Metrikler
# ======================
accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels)
cm = confusion_matrix(y_true, y_pred)

# ROC için: Her sınıf için binary karşılaştırma
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro ve Weighted ROC-AUC skorları
macro_roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
weighted_roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')

# ======================
# TXT ve Grafik Kaydı
# ======================
# Classification Report TXT
with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Cohen Kappa Score: {kappa:.4f}\n")
    f.write(f"Macro ROC-AUC Score: {macro_roc_auc:.4f}\n")
    f.write(f"Weighted ROC-AUC Score: {weighted_roc_auc:.4f}\n\n")
    f.write(report)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

# ROC Curve Plot
plt.figure()
for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_dir, "roc_curve.png"))
plt.close()

print("Tüm metrikler başarıyla hesaplandı ve 'grafikler' klasörüne kaydedildi.")