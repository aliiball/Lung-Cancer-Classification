import os
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight  # <--- Bunu ekle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, regularizers
from collections import Counter
import pickle

# === GPU Ayarı ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU bellek büyümesi aktif.")
else:
    print("CUDA desteklenmiyor. Model eğitimi yapılamayacak.")
    exit()

# === Mixed Precision Ayarı ===
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision (float16) aktif.")

# === Ayarlar ===
image_size = (224, 224)
batch_size = 32
epochs = 30
fine_tune_at_epoch = 5
k = 5
data_dir = r"C:\\Projects\\Datasets\\lung_dataset"
model_output_path = "C:\\Projects\\KFold_Model\\LungCancerApp\\app"
os.makedirs(model_output_path, exist_ok=True)

# === Dosya yollarını ve etiketleri al ===
image_paths = []
labels = []
class_indices = {}

for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    class_indices[class_name] = idx
    for file_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, file_name))
        labels.append(idx)

image_paths = np.array(image_paths)
labels = np.array(labels)
num_classes = len(np.unique(labels))

print("Toplam görüntü:", len(image_paths))
print("Sınıf sayısı:", num_classes)
print("Sınıf eşleşmeleri:", class_indices)
print("Sınıf dağılımı:", Counter(labels))

# === Yardımcı preprocess fonksiyonu ===
def preprocess(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.02)
        img = tf.image.rot90(img, k=np.random.randint(4))
    return img, tf.one_hot(label, num_classes)

# === Model oluşturucu fonksiyon ===
def build_model(input_shape=(224, 224, 3), num_classes=4):
    l2 = regularizers.l2(0.001)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Model summary'i dosyaya yaz ===
os.makedirs("grafikler", exist_ok=True)
model_summary_path = os.path.join("grafikler", "model_summary.txt")
model_for_summary = build_model(input_shape=(224, 224, 3), num_classes=num_classes)
with open(model_summary_path, "w", encoding="utf-8") as f:
    model_for_summary.summary(print_fn=lambda x: f.write(x + "\n"))
del model_for_summary
tf.keras.backend.clear_session()
gc.collect()

# === K-Fold ===
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
fold_no = 1
acc_per_fold, loss_per_fold, training_times = [], [], []

for train_idx, val_idx in skf.split(image_paths, labels):
    print(f"\n==== Fold {fold_no} ====")

    train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(input_shape=(224, 224, 3), num_classes=num_classes)

    class_weights = dict(enumerate(class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(train_labels), y=train_labels)))

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1)
    model_save_path = os.path.join(model_output_path, f"fold{fold_no}_best_model.h5")
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

    print(">>> Eğitim başlıyor...")
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    duration = time.time() - start
    training_times.append(duration)

    scores = model.evaluate(val_ds, verbose=0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # === history objesini kaydet ===
    fold_dir = os.path.join("grafikler", f"Fold{fold_no}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    # === Grafik çizimi ===
    plt.figure()
    plt.plot(history.history['accuracy'], label='Eğitim')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title(f'Fold {fold_no} - Doğruluk')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fold_dir, "accuracy_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Eğitim')
    plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title(f'Fold {fold_no} - Kayıp')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fold_dir, "loss_plot.png"))
    plt.close()

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    fold_no += 1

# === Özet ===
summary_path = os.path.join("grafikler", "kfold_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    for i in range(k):
        f.write(f"Fold {i+1}: Doğruluk = {acc_per_fold[i]:.2f}%, Kayıp = {loss_per_fold[i]:.4f}, Süre = {training_times[i]:.2f} sn\n")
    f.write(f"\nOrtalama Doğruluk: {np.mean(acc_per_fold):.2f}%\n")
    f.write(f"Ortalama Eğitim Süresi: {np.mean(training_times):.2f} sn\n")
    f.write(f"Toplam Eğitim Süresi: {sum(training_times):.2f} sn\n")

print("✅ K-Fold eğitim tamamlandı. Sonuçlar 'grafikler/' klasöründe.")