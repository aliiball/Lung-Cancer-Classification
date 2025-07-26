# 🫁 Lung Cancer Classification with Deep Learning (VGG16) :
Bu proje, derin öğrenme teknikleri kullanarak akciğer kanseri hücre görüntülerini sınıflandırmayı amaçlar. Üç farklı hücre tipi arasında ayrım yapılır:

- **Benign (İyi Huylu)**
- **Adenocarcinoma**
- **Squamous Cell Carcinoma**
- **Unknown**

Model, eğitim süreci sonrası `.h5` formatında kaydedilir ve kullanıcı arayüzü aracılığıyla tahmin yapılabilir. Ayrıca eğitim ve test sürecinde oluşan metrikler, grafik olarak kaydedilir.

# 📁 Proje Yapısı :
LungCancerApp/
│
├── vgg16_env/                 # Virtual environment klasörü
│
├── lung_split_dataset/        # Veri setinizin bulunduğu klasör (Adenocarcinoma, Benign, Squamous_Cell_Carcinoma, Unknown)
│
├── app/                       # Uygulama kodları
│   ├── CUDA_kontrol.py		   # Cuda Kontrolünün yapıldığı yer	   
│   ├── interface.py           # Arayüz kodu
│   ├── vgg16_model.py         # VGG16 modelini burada oluşturacağız
│   ├── main.py                # Ana uygulama başlatma dosyası
│   └── metriccs_report.py     # Metrikler Hesaplanır
│
├── LungCancerApp.ico          # Exe haline getirilicekse app'in simgesi
├── README.md                  # Projeye dair açıklamalar
└── requirements.txt           # Projede kullanılacak kütüphanelerin listesi

# ⚙️ Ortam Kurulumu :
Ortam Kurulum ve Uygulama Adımları :
- Sanal Ortam Oluşturma : python -m venv vgg16_env
- Sanal Ortam Aktif Etme : .\vgg16_env\Scripts\Activate
- Uygulama Başlatma : python app/main.py
- Exe Build : python setup.py build

# 📦 Gerekli Kütüphaneleri Yükleme :
- pip install -r requirements.txt

# 📄 requirements.txt İçeriği :
- tensorflow==2.10.0
- scikit-learn==1.1.3
- matplotlib==3.6.2
- opencv-python==4.8.0.76
- pillow==9.5.0
- numpy==1.23.5
- pandas==2.0.3
- seaborn==0.12.2
- tk==0.1.0

# 🧠 Model Eğitimi :
python train.py

Eğitim tamamlandıktan sonra aşağıdaki çıktılar oluşur :
- app/vgg16_best_model.h5
- grafikler/accuracy_plot.png
- grafikler/loss_plot.png

# 📊 Performans Metrikleri :
python metrics_report.py

Çıktılar :
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix → grafikler/confusion_matrix.png
- ROC Curve → grafikler/roc_curve.png
- AUC Skoru
- Cohen's Kappa
- Tüm metriklerin özeti: grafikler/metrics_report.txt

# 🖥️ Arayüzü Kullanma :
Arayüzü başlatmak için :
- python app/main.py

# 📦 .exe Dosyasına Dönüştürme (Opsiyonel) :
Uygulamanın tek bir çalıştırılabilir dosyaya çevrilmesi için :
- pip install pyinstaller
- pyinstaller --onefile --windowed app/main.py

# ℹ️ Ek Bilgiler :
- .h5 modeli yalnızca eğitilmiş ağırlıkları ve mimariyi içerir.
- Metrikler metrics_report.py dosyası aracılığıyla hesaplanır.
- grafikler/ klasörü eğitim sonrası oluşur, mevcutsa üzerine yazılır.
- ROC/AUC skorları sadece binary sınıflandırma durumlarında anlamlıdır.

# 📜 Lisans :
Bu proje Muhammet Ali BAL tarafından sadece akademik ve eğitimsel amaçlar için yapılmıştır. Ticari kullanım için izin gereklidir.