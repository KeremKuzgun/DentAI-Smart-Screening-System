# 🦷 DentAI  — Ağız Sağlığı & Ortodonti Ön Değerlendirme Sistemi

Derin öğrenme tabanlı **9 farklı** ağız sağlığı ve ortodonti problemini tespit eden dual-model web uygulaması.

## Tespit Edilen Problemler

| Model | Sınıflar | mAP@0.5 |
|-------|----------|---------|
| **Ağız Sağlığı** (6 sınıf) | Çürük, Diş Taşı, Diş Eti İltihabı, Eksik Diş, Renk Değişikliği, Ağız Ülseri | %80.3 |
| **Ortodonti** (3 sınıf) | Diş Torsiyonu (TT), Derin Overjet (DO), Alt Çene Geriliği (MR) | %66.7 |

## Kurulum

### Gereksinimler
- Python 3.11+
- GPU önerilir (NVIDIA CUDA destekli) ama CPU'da da çalışır

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/KeremKuzgun/DentAI-Smart-Screening-System.git
cd DentAI-Smart-Screening-System

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. Çalıştır
python app.py
veya
py -3.11 app.py
```

Tarayıcıda **http://localhost:8000** adresini aç.

## Kullanım

1. Ana sayfada **Dosya Seç** veya sürükle-bırak ile intraoral fotoğraf yükle.
2. Sistem görüntüyü her iki modelden geçirip sonuçları gösterir.
3. Sağ panelde tespit edilen problemler kategori bazında (Ağız Sağlığı / Ortodonti) listelenir.

### Doğru Fotoğraf Çekimi
Uygulama içinde **"Çekim Kılavuzu"** bölümünden doğru açı ve koşulları görebilirsiniz.
- **Ortodonti:** Dişler kapalı, ön + yan görünüm (zorunlu)
- **Ağız Sağlığı:** Flaş açık, yakın çekim, dudaklar aralık

## Proje Yapısı

```
dentai-v2/
├── app.py                  # FastAPI backend (dual-model inference)
├── index.html              # Web arayüzü
├── oral_health_best.pt     # Ağız sağlığı modeli (YOLOv8s, 6 sınıf)
├── ortho_best.pt           # Ortodonti modeli (YOLOv8s, 3 sınıf)
├── requirements.txt        # Python bağımlılıkları
└── README.md
```

## Teknoloji

- **Model:** YOLOv8s (Ultralytics)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS
- **Eğitim Verileri:**
  - Ağız Sağlığı: Oral Disease Detection Dataset
  - Ortodonti: OMNI Dataset (COCO format, 4166 görüntü)(Yolo formatına dönüştürüldü)

## Uyarı

Bu sistem yalnızca **ön değerlendirme** amaçlıdır. Kesin tanı için mutlaka bir diş hekimine başvurunuz.

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.
