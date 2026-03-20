# 🦾 JARVIS | Gesture Control System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange?style=for-the-badge&logo=opencv&logoColor=white)

Gerçek zamanlı el takibi ve jest algılama yeteneklerine sahip, fütüristik bir **HUD (Head-Up Display)** arayüzüyle donatılmış interaktif kontrol sistemi. MediaPipe'ın yüksek performanslı `hand_landmarker` modelini kullanarak el hareketlerini dijital komutlara dönüştürür.

---

## 🔥 Öne Çıkan Özellikler

*   **⚡ Gerçek Zamanlı Analiz:** Düşük gecikme süresi ile anlık el takibi.
*   **🕶️ Fütüristik HUD:** Scanline efektleri, dinamik grid yapısı ve parlayan el iskeleti ile JARVIS estetiği.
*   **🧠 Akıllı Jest Algılama:** Parmakların durumuna göre önceden tanımlanmış 5 farklı komut.
*   **📡 Kesintisiz Geri Bildirim:** Hareketlerinize göre renk değiştiren dinamik UI panelleri.
*   **🛠️ Kolay Entegrasyon:** Algılanan jestlere özel fonksiyonlar (lambda) atayabilme.

---

## 🎮 Desteklenen Jestler & Komutlar

Sistem aşağıdaki el hareketlerini tanır ve karşılık gelen komutları tetikler:

| Jest | Etiket | Fonksiyon | HUD Rengi |
| :--- | :--- | :--- | :--- |
| **Yumruk** | `STOP` | Durdur (⏹) | Turuncu/Kırmızı |
| **Açık El** | `PAUSE` | Bekle (✋) | Turkuaz |
| **Bir Parmak** | `SEL` | Seç (☝) | Yeşil |
| **İki Parmak** | `SCRLL` | Kaydırma (✌) | Mavi |
| **Üç Parmak** | `MENU` | Menü (3) | Mor |

---

## 🚀 Kurulum

### 1. Gereksinimler
Sistemi çalıştırmak için bilgisayarınızda Python yüklü olmalıdır. Gerekli kütüphaneleri aşağıdaki komutla kurabilirsiniz:

```bash
pip install opencv-python mediapipe numpy
```

### 2. Model Dosyası
`hand_landmarker.task` dosyasının projenin ana dizininde olduğundan emin olun. (Dosya eksikse MediaPipe resmi sitesinden indirilebilir).

### 3. Çalıştırma
Sistemi başlatmak için:

```bash
python temel_el_takip.py
```

*Çıkış yapmak için klavyeden **'Q'** tuşuna basmanız yeterlidir.*

---

## 🛠️ Teknik Detaylar

Bu proje MediaPipe **Vision Tasks** API'sini kullanmaktadır. 
- **Model:** Hand Landmarker
- **Çalışma Modu:** `LIVE_STREAM` (Canlı Yayın Modu)
- **Görselleştirme:** OpenCV ile özel çizim fonksiyonları (Konveks hull, glow efektleri ve HUD katmanları).

---

## 👨‍💻 Geliştirici
Bu proje, bilgisayarlı görü (Computer Vision) ve insan-bilgisayar etkileşimi (HCI) prensipleri üzerine inşa edilmiştir.

> "Genius, billionaire, playboy, philanthropist..." - **JARVIS System is Online.**
