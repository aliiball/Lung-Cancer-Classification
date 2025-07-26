# 🫁 Lung Cancer Classification with Deep Learning (CNN) :
Bu proje, derin öğrenme teknikleri kullanarak akciğer kanseri hücre görüntülerini sınıflandırmayı amaçlar. Üç farklı hücre tipi arasında ayrım yapılır:

- **Benign (İyi Huylu)**
- **Adenocarcinoma**
- **Squamous Cell Carcinoma**
- **Unknown**

Model, eğitim süreci sonrası `.h5` formatında kaydedilir ve kullanıcı arayüzü aracılığıyla tahmin yapılabilir. Ayrıca eğitim ve test sürecinde oluşan metrikler, grafik olarak kaydedilir.

# 📁 Proje Yapısı :
LungCancerApp/
│
├── cnn_env/                   # Virtual environment klasörü
│
├── lung_split_dataset/        # Veri setinizin bulunduğu klasör (Adenocarcinoma, Benign, Squamous_Cell_Carcinoma, Unknown)
│
├── app/                       # Uygulama kodları
│   ├── CUDA_kontrol.py		   # Cuda Kontrolünün yapıldığı yer	   
│   ├── interface.py           # Arayüz kodu
│   ├── cnn_model.py           # CNN modelini burada oluşturacağız
│   ├── main.py                # Ana uygulama başlatma dosyası
│   └── metriccs_report.py     # Metrikler Hesaplanır
│
├── LungCancerApp.ico          # Exe haline getirilicekse app'in simgesi
├── README.md                  # Projeye dair açıklamalar
└── requirements.txt           # Projede kullanılacak kütüphanelerin listesi

# ⚙️ Ortam Kurulumu :
Ortam Kurulum ve Uygulama Adımları :
- Sanal Ortamım : python -m venv cnn_env
- Sanal Ortam Aktif Etme : .\cnn_env\Scripts\Activate
- App'i Çalıştırma : python app/main.py
- Exe Build : python setup.py build

# 📦 Gerekli Kütüphaneleri Yükleme :
- pip install -r requirements.txt

# 📄 requirements.txt İçeriği :
- tensorflow==2.15.0
- scikit-learn==1.3.0
- matplotlib==3.7.1
- pillow==10.0.0
- numpy==1.24.3
- pandas==2.0.3
- seaborn==0.12.2
- tk

# 🧠 Model Eğitimi :
- python train.py

Eğitim tamamlandıktan sonra aşağıdaki çıktılar oluşur :
- app/cnn_best_model.h5
- grafikler/accuracy_plot.png
- grafikler/loss_plot.png

# 📊 Performans Metrikleri :
- python metrics_report.py

Çıktılar :
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix → confusion_matrix.png
- ROC Curve → roc_curve.png
- AUC Skoru
- Cohen's Kappa
- Hepsi grafikler/metrics_report.txt dosyasında özetlenir

# 🖥️ Arayüzü Kullanma :
Arayüzü başlatmak için :
- python app/main.py

# 📦 .exe Dosyasına Dönüştürme (Opsiyonel) :
Uygulamanın tek bir çalıştırılabilir dosyaya çevrilmesi için :
- pip install pyinstaller
- pyinstaller --onefile --windowed app/main.py

# ℹ️ Ek Bilgiler :
- .h5 modeli yalnızca ağırlık ve mimariyi içerir.
- Metrikler .py dosyası aracılığıyla hesaplanır.
- grafikler/ klasörü eğitim sonrası oluşur, tekrar oluşturulmaz.
- ROC/AUC yalnızca binary sınıflandırmalarda anlamlıdır.

# 📜 Lisans :
Bu proje Muhammet Ali BAL tarafından sadece akademik ve eğitimsel amaçlar için yapılmıştır. Ticari kullanım için izin gereklidir.