import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys

# GPU Kontrolü
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("CUDA destekli GPU bulunamadı. Eğitim sadece GPU ile yapılabilir. Program sonlandırılıyor.")
    sys.exit(1)
else:
    print(f"GPU bulundu: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

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

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ----- AlexNet Modeli (Sıfırdan Tanımlı ve BatchNormalization ile) -----
input_layer = Input(shape=(224, 224, 3))

x = Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(4096, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Derleme
model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Grafik klasörü oluştur
os.makedirs("grafikler", exist_ok=True)

# Model özeti
print("\n--- MODEL MİMARİSİ ---")
with open("grafikler/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    "C:\\Projects\\AlexNet_Model\\LungCancerApp\\app\\alexnet_best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-6)

# Eğitim
start_time = time.time()
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
end_time = time.time()
training_time = end_time - start_time
print(f"Eğitim süresi: {training_time:.2f} saniye")

# History kaydetme
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

# Eğitim süresi yaz
with open("grafikler/egitim_suresi.txt", "w", encoding="utf-8") as f:
    f.write(f"Egitim suresi: {training_time:.2f} saniye\n")

print("Model eğitimi tamamlandı, tüm çıktılar 'grafikler' klasörüne kaydedildi.")