# 🚢 Titanic - Machine Learning from Disaster

Bu repo, [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) yarışmasına katılım amacıyla oluşturulmuştur. Projede, veri ön işleme, farklı makine öğrenmesi modelleriyle tahmin ve model karşılaştırmaları yapılmıştır.

## 🎯 Projenin Amacı

Titanic yolcularının demografik ve seyahat bilgilerinden yola çıkarak hayatta kalıp kalmadıklarını tahmin etmek. Bu klasik sınıflandırma problemi üzerinden temel veri bilimi ve makine öğrenmesi tekniklerini uygulamak amaçlanmıştır.

---

## 🗂️ Klasör ve Dosya Yapısı

```
Titanic-Machine-Learning-from-Disaster/
│
├── catboost_info/                    # CatBoost çalışması sırasında oluşan dosyalar
├── dataset/                          # Veri dosyaları
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── .gitignore
├── best_model_score.txt             # En iyi modelin skor çıktısı
│
├── config.py                        # Sabit ayarlar ve yapılandırmalar
├── data_preprocessing.py           # Veri temizleme ve ön işleme fonksiyonları
├── dataset.py                       # Veri yükleme ve ayırma işlemleri
├── helpers.py                       # Yardımcı fonksiyonlar
├── main.py                          # Ana pipeline dosyası (veri → model → tahmin)
├── models.py                        # Farklı model tanımlamaları ve eğitimi
│
├── RandomForestModel.py            # Random Forest modeli örneği
├── RandomForestWithGreadSearch.py  # Hiperparametre optimizasyonlu RF modeli
│
├── Notes.md                         # Model notları, sonuçlar
├── Research.ipynb                   # Jupyter not defteri üzerinde yapılan analizler
│
├── Readme.md                        # Bu dosya
│
├── submission.csv                   # Final tahmin sonuçları
├── submission_CART.csv              # CART modeli tahminleri
├── submission_CatBoost.csv          # CatBoost tahminleri
├── submission_RF.csv                # Random Forest tahminleri
└── submission_XGBoost.csv           # XGBoost tahminleri
```

---

## 🔍 Kullanılan Yöntemler

- Eksik veri analizi ve doldurma (Age, Embarked vb.)
- Label encoding, One-Hot Encoding
- Özellik mühendisliği (örn: Title çıkarımı, aile boyutu hesaplama)
- Modelleme:
  - Decision Tree (CART)
  - Random Forest
  - XGBoost
  - CatBoost
- Hiperparametre optimizasyonu (`GridSearchCV`)
- Model karşılaştırma ve değerlendirme (`accuracy_score`, `confusion_matrix`)

---

## 📊 Çıktılar

Aşağıdaki modeller kullanılarak oluşturulan `submission_*.csv` dosyaları Kaggle sistemine yüklenmiştir:

| Model              | Public Score (Kaggle) |
|-------------------|-----------------------|
| CART              | 0.75598               |
| Random Forest     | 0.78468               |
| XGBoost           | 0.78229               |
| CatBoost          | 0.77511               |

**Not:** Gerçek skorlar, `best_model_score.txt` dosyasında detaylı şekilde yer almaktadır.

---

## 🛠️ Kullanılan Teknolojiler

- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- XGBoost, CatBoost
- Jupyter Notebook

---

## 📌 Notlar

Bu proje eğitim amaçlıdır ve klasikleşmiş bir Kaggle problemi olan Titanic veri seti üzerinden temel makine öğrenmesi süreçlerini uygulama fırsatı sunar.

yagramı veya örnek çıktı da ekleyebilirim. Geri bildirim verirsen ona göre özelleştiririm.

---
