import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # ModelCheckpoint eklendi
import sys

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

# Klasör yolu
dataset_path = "C:\\Projects\\Datasets\\lung_split_dataset"

# Parametreler
batch_size = 32
img_size = (224, 224)
epochs = 30
num_classes = 4

# Veri artırma
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verisi
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Doğrulama verisi
val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# VGG16 temel model (başlık kısmı hariç)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: sadece son 4 blok eğitilebilir
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Model derleme
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Grafik klasörü oluştur
os.makedirs("grafikler", exist_ok=True)

# Model özetini yazdır ve dosyaya kaydet
print("\n--- MODEL MİMARİSİ ---")
with open("grafikler/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# ModelCheckpoint (en iyi modeli kaydet)
model_checkpoint = ModelCheckpoint(
    "C:\\Projects\\VGG16_Model\\LungCancerApp\\app\\vgg16_best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Eğitim başlangıç zamanı
start_time = time.time()

# Modeli eğitme (ModelCheckpoint eklendi)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Eğitim süresi
end_time = time.time()
training_time = end_time - start_time
print(f"Egitim suresi: {training_time:.2f} saniye")

# Grafik klasörü oluştur
os.makedirs("grafikler", exist_ok=True)

# History verisini kaydet
with open("grafikler/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Accuracy grafiği
plt.figure()
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("grafikler/accuracy_plot.png")
plt.show()

# Loss grafiği
plt.figure()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("grafikler/loss_plot.png")
plt.show()

# Eğitim süresini yaz
with open("grafikler/egitim_suresi.txt", "w", encoding="utf-8") as f:
    f.write(f"Eğitim süresi: {training_time:.2f} saniye\n")

print("Model eğitimi tamamlandı, tüm çıktılar 'grafikler' klasörüne kaydedildi.")