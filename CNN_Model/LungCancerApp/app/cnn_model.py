import os
import time
import sys
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# GPU kontrolü
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("CUDA destekli GPU bulunamadı. Eğitim sadece GPU ile yapılabilir. Program sonlandırılıyor.")
    sys.exit(1)
else:
    print(f"GPU bulundu: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) # Bellek büyümesini aç
    tf.config.set_visible_devices(gpus[0], 'GPU') # Birden fazla GPU varsa ilkini kullan

# =====================
# Klasörleri oluştur
# =====================
os.makedirs("grafikler", exist_ok=True)

# =====================
# Ayarlar
# =====================
image_size = (224, 224)
batch_size = 32
train_dir = "C:\\Projects\\Datasets\\lung_split_dataset\\train"
val_dir = "C:\\Projects\\Datasets\\lung_split_dataset\\validation"

# =====================
# Veri artırma
# =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# =====================
# Model oluştur
# =====================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Sınıf sayısı otomatik ayarlanır
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# =====================
# Callback'ler
# =====================
model_checkpoint = ModelCheckpoint(
    "C:\\Projects\\CNN_Model\\LungCancerApp\\app\\cnn_best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Grafik klasörü oluştur
os.makedirs("grafikler", exist_ok=True)

# Model özetini yazdır ve dosyaya kaydet
print("\n--- MODEL MİMARİSİ ---")
with open("grafikler/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1)

# =====================
# Eğitim
# =====================
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr, model_checkpoint],
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

# =====================
# Grafik ve kayıtlar
# =====================

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("grafikler/accuracy_plot.png")
plt.close()

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("grafikler/loss_plot.png")
plt.close()

# Eğitim süresi dosyası
with open("grafikler/egitim_suresi.txt", "w") as f:
    f.write(f"Egitim suresi: {training_time:.2f} saniye\n")

# History kaydet
with open("grafikler/history.pkl", "wb") as f:
    pickle.dump(history.history, f)