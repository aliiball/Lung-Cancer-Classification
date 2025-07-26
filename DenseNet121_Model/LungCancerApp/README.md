# 🫁 Lung Cancer Classification with Deep Learning (DenseNet121)
Bu proje, **DenseNet121** derin öğrenme mimarisi kullanılarak akciğer kanseri hücre görüntülerini sınıflandırmayı amaçlar. Üç farklı hücre tipi arasında ayrım yapılır:

- **Benign (İyi Huylu)**
- **Adenocarcinoma**
- **Squamous Cell Carcinoma**
- **Unknown**

Model eğitimden sonra `.pth` formatında kaydedilir. Kullanıcı arayüzü sayesinde son kullanıcı, yüklediği görüntü üzerinde sınıflandırma yapabilir. Eğitim sürecine ait metrikler ve grafikler de projeye dahil edilmiştir.

# 📁 Proje Yapısı :
LungCancerApp/
│
├── densenet121_env/           # Virtual environment klasörü
│
├── lung_split_dataset/        # Veri setinizin bulunduğu klasör (Adenocarcinoma, Benign, Squamous_Cell_Carcinoma, Unknown)
│
├── app/                       # Uygulama kodları
│   ├── CUDA_kontrol.py		   # Cuda Kontrolünün yapıldığı yer	   
│   ├── interface.py           # Arayüz kodu
│   ├── densenet121_model.py   # DenseNet121 modelini burada oluşturacağız
│   ├── main.py                # Ana uygulama başlatma dosyası
│   └── metriccs_report.py     # Metrikler Hesaplanır
│
├── LungCancerApp.ico          # Exe haline getirilicekse app'in simgesi
├── README.md                  # Projeye dair açıklamalar
└── requirements.txt           # Projede kullanılacak kütüphanelerin listesi

# ⚙️ Ortam Kurulumu :
Ortam Kurulum ve Uygulama Adımları :
- Sanal Ortam Oluşturma : python -m venv densenet121_env
- Sanal Ortam Aktif Etme : .\densenet121_env\Scripts\Activate
- Uygulama Başlatma : python app/main.py
- Exe Build : python setup.py build

# 📦 Gerekli Kütüphaneleri Yükleme :
- pip install -r requirements.txt

# 📄 requirements.txt İçeriği :
- tensorflow==2.10.0
- tensorboard>=2.10
- torch==2.1.0
- torchvision==0.16.0
- scikit-learn>=1.0
- matplotlib>=3.4
- opencv-python-headless>=4.5
- Pillow>=9.0
- numpy<2
- pandas>=1.3
- seaborn>=0.11
- joblib>=1.1
- tk

# 🧠 Model Eğitimi :
python train.py

Eğitim tamamlandığında aşağıdaki dosyalar oluşur :
- app/densenet121_best_model.pth
- grafikler/accuracy.png
- grafikler/loss.png

# 📊 Performans Metrikleri :
python metrics_report.py

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix → grafikler/confusion_matrix.png
- ROC Curve → grafikler/roc_curve.png
- AUC Skoru
- Cohen's Kappa
- Özet dosyası → grafikler/metrics_report.txt

# 🖥️ Arayüzü Kullanma :
Arayüzü başlatmak için :
- python app/main.py

# 📦 .exe Dosyasına Dönüştürme (Opsiyonel) :
Uygulamanın tek bir çalıştırılabilir dosyaya çevrilmesi için :
- pip install pyinstaller
- pyinstaller --onefile --windowed app/main.py

# ℹ️ Ek Bilgiler :
- .pth uzantılı model dosyası yalnızca eğitilmiş ağırlıkları içerir.
- metrics_report.py, tüm metrikleri hesaplar ve dosyalara kaydeder.
- grafikler/ klasörü eğitim sonrası otomatik oluşur.
- ROC/AUC skorları mikro ve makro olarak hesaplanır.

# 📜 Lisans :
Bu proje Muhammet Ali BAL tarafından sadece akademik ve eğitimsel amaçlar için yapılmıştır. Ticari kullanım için izin gereklidir.