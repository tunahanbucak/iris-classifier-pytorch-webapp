# 🌸 Iris Classifier Web App & Derin Öğrenme Çalışma Rehberi

Bu proje, PyTorch ile eğitilmiş 3 katmanlı çok sınıflı (multiclass) yapay sinir ağı modelini (`models/iris_classification_model.pth`) **FastAPI** ve cam efektli (Glassmorphism) modern bir web arayüzü ile canlıya alan uçtan uca bir yapay zeka uygulamasıdır.

Aynı zamanda **Çok Sınıflı Sınıflandırma** ve **PyTorch Model Kaydetme / Yükleme** konuları için kapsamlı bir ders tekrarı ve referans kaynağıdır.

---

## 🚀 Hızlı Başlangıç (Nasıl Çalıştırılır?)

1. **Sanal Ortamı Aktif Edin (Terminalde):**
   ```bash
   source .venv/bin/activate
   ```

2. **Web Sunucusunu Başlatın:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Tarayıcınızda Açın:**
   * **Web Arayüzü:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
   * **Swagger API Dokümantasyonu:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧠 Derin Öğrenme Ders Tekrarı (Cheat Sheet & Teori)

### 1. Çok Sınıflı (Multiclass) Sınıflandırma Nedir?
İkili (Binary) sınıflandırmada tek çıkış nöronu varken (örneğin kedi mi değil mi?), Iris probleminde **3 ayrı sınıf** vardır:
* `0`: Setosa
* `1`: Versicolor
* `2`: Virginica

Bu yüzden çıkış katmanında **3 adet nöron** (`out_features=3`) bulunur.

### 2. Model Mimarisi (`model.py`)
```
Girdi [4 Özellik] ➔ Linear(4, 16) ➔ ReLU ➔ Linear(16, 16) ➔ ReLU ➔ Linear(16, 3) ➔ Ham Lojitler [3 Skor]
```
* **Neden `ReLU`?** Doğrusal olmayan (non-linear) aktivasyon fonksiyonu koymazsak, arka arkaya eklenen tüm katmanlar matematiksel olarak tek bir matris çarpımına indirgenir. `ReLU` ağa kıvrımlı karar sınırları çizme yeteneği kazandırır.
* **Neden Model Sonunda Softmax Yok?**
  PyTorch'ta eğitim sırasında kullanılan `nn.CrossEntropyLoss()`, arkada `LogSoftmax` ve `NLLLoss` işlemlerini sayısal açıdan en kararlı (Log-Sum-Exp) yöntemle birleşik uygular. Bu yüzden PyTorch modelleri çıkışta ham skorlar (**Lojit / Logits**) üretir.

### 3. Model Kaydetme ve Yükleme (`state_dict`)
* **Tüm Nesneyi Kaydetmek:** `torch.save(model, path)` ➔ ❌ Dosya yapısı değişince bozulabilir, tavsiye edilmez.
* **Ağırlıkları Kaydetmek (Standart):** `torch.save(model.state_dict(), path)` ➔ ✅ Sadece eğitilmiş ağırlık ve bias matrislerini saklar. En güvenli yöntemdir.

**Modeli Geri Yükleme:**
```python
model = IrisClassifier() # 1. Boş mimariyi RAM'de kur
model.load_state_dict(torch.load("path.pth", map_location="cpu")) # 2. Ağırlıkları doldur
model.eval() # 3. Test/Değerlendirme moduna al
```
* `map_location="cpu"`: Model GPU'da eğitilmiş olsa bile CPU'da hata almadan çalıştırılmasını sağlar.
* `model.eval()`: Modeli eğitim modundan çıkararak tahminlerin deterministik ve tutarlı olmasını sağlar.

### 4. Tahmin Boru Hattı (Inference Pipeline)
```
[SL, SW, PL, PW] (Girdiler)
       ↓
torch.tensor(...) [Boyut: 1x4, float32]
       ↓
with torch.inference_mode():  (Autograd/türev takibini kapatır, bellek ve hız kazandırır)
       ↓
logits = model(features)       (Ham Lojit Skorları, örn: [+14.78, +5.78, -18.87])
       ↓
probs = torch.softmax(logits)  (Olasılık Dağılımı %0-%100, toplamı tam 1.0)
       ↓
pred_idx = torch.argmax(probs) (En yüksek olasılığa sahip sınıf indeksi: 0, 1 veya 2)
```

---

## 🌟 Proje Özellikleri

* **🧠 Model İç Mekanizması Paneli:** Ham Lojit skorları, Softmax formülü ve Argmax kararını canlı gösterir.
* **🌺 Botanik Fotoğraf Portresi:** 3 türe ait yüksek çözünürlüklü ve birbirinden botanik olarak net ayrılan makro fotoğraflar.
* **⚡ Canlı Asenkron Tahmin:** Kaydırıcılar (slider) hareket ettirildikçe 100ms debounce ile anlık tahmin.
* **🎯 Hazır Test Düğmeleri & Galeri:** Türlerin tipik yaprak ölçülerini tek tıkla yükleyip modeli test etme.
