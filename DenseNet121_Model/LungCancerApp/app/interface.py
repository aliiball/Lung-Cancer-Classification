import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt

print("Arayüz başlatılıyor...")

# Modeli yükle
model_path = r"C:\\Projects\\DenseNet121_Model\\LungCancerApp\\app\\densenet121_best_model.h5"
model = load_model(model_path)

# Sınıf isimleri
class_names = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma', 'Unknown']

def run_app():
    root = tk.Tk()
    root.title("DenseNet121 - Akciğer Kanseri Tahmin Arayüzü")
    root.state('zoomed')  # Tam ekran başlat

    # Son tahmin verilerini saklayacak değişken
    last_img_array = None
    last_file_path = None

    def load_image():
        nonlocal last_img_array, last_file_path
        file_path = filedialog.askopenfilename(title="Resim Seç", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = Image.open(file_path).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            last_img_array = img_array
            last_file_path = file_path
            predict_image()

    def predict_image():
        if last_img_array is None:
            return

        prediction = model.predict(last_img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[predicted_index]

        threshold = threshold_slider.get() / 100  # Slider 0–100, normalde 0.00–1.00

        if confidence >= threshold:
            result_text = f'Tahmin Sonucu: {predicted_class}\nDoğruluk Oranı: %{confidence * 100:.2f}'
        else:
            result_text = f'Tahmin Güvenliği Düşük!\nBu görüntü sınıflandırılamadı.\n(Max Doğruluk: %{confidence * 100:.2f})'

        result_label.config(text=result_text)

        # Görseli göster
        img_display = Image.open(last_file_path).resize((300, 300))
        img_display = ImageTk.PhotoImage(img_display)
        panel.config(image=img_display)
        panel.image = img_display

    # Arayüz bileşenleri
    load_button = tk.Button(root, text="Resim Yükle", font=("Arial", 14), command=load_image)
    load_button.pack(pady=10)

    threshold_slider = tk.Scale(root, from_=50, to=100, orient="horizontal", label="Eşik Değeri (% Güven)", font=("Arial", 12))
    threshold_slider.set(80)  # Varsayılan %80
    threshold_slider.pack(pady=5)

    update_button = tk.Button(root, text="Tahmini Güncelle", font=("Arial", 12), command=predict_image)
    update_button.pack(pady=5)

    panel = tk.Label(root)
    panel.pack(pady=5)

    result_label = tk.Label(root, text="Tahmin Sonucu: ", font=("Arial", 14))
    result_label.pack(pady=10)

    bar_panel = tk.Label(root)
    bar_panel.pack(pady=10)

    root.mainloop()