import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from model import IrisClassifier

# ==============================================================================
# 🌸 FASTAPI UYGULAMA YAPILANDIRMASI
# ==============================================================================
app = FastAPI(
    title="Iris Çiçek Türü Sınıflandırma API",
    description="PyTorch ile eğitilmiş Çoklu Sınıflandırma (Multiclass) Yapay Sinir Ağı Web Servisi.",
    version="1.2.0"
)

# Statik varlıklar (CSS, JS, Görseller) ve HTML şablon dizinleri
BASE_DIR = Path(__file__).parent
static_path = BASE_DIR / "static"
templates_path = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# ==============================================================================
# 🧠 MODELİ YÜKLEME VE INFERENCE (TAHMİN) HAZIRLIĞI
# ==============================================================================
# 1. Kaydedilmiş model ağırlık dosyasının (.pth) yolu
MODEL_PATH = BASE_DIR / "models" / "iris_classification_model.pth"

# 2. Önce boş mimariyi (architecture) RAM'de oluşturuyoruz
model = IrisClassifier()

if MODEL_PATH.exists():
    # 3. Modelin eğitilmiş ağırlıklarını (state_dict) yüklüyoruz.
    #    - map_location=torch.device('cpu'): Model GPU'da (CUDA) eğitilmiş olsa dahi,
    #      sunucu veya yerel CPU ortamında çökmeden sorunsuz çalışmasını sağlar.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    
    # 4. Modeli Değerlendirme Moduna (Evaluation Mode) alıyoruz.
    #    - Neden model.eval()? Modelde Dropout veya BatchNorm katmanları varsa
    #      onların rastgele davranışlarını kapatır ve deterministik tahmin sağlar.
    model.eval()
    print("PyTorch Iris Modeli başarıyla yüklendi ve değerlendirme moduna alındı! ✅")
else:
    print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}")

# ==============================================================================
# 🌺 BOTANİK BİLGİ VERİTABANI & METADATA
# ==============================================================================
# Her bir sınıf indeksi (0, 1, 2) için detaylı botanik bilgiler, renk teması ve görsel
SPECIES_INFO = {
    0: {
        "name": "Iris-setosa",
        "tr_name": "Setosa Süseni",
        "tag": "Setosa",
        "color": "#10B981", # Zümrüt Yeşili
        "color_rgb": "16, 185, 129",
        "image": "/static/images/iris_setosa.jpg?v=botanical_v2",
        "badge": "🌸 Setosa",
        "sepal_typical": "4.8 - 5.5 cm",
        "petal_typical": "1.0 - 1.9 cm (Çok Küçük)",
        "description": "Minyatür dik taç yaprakları (petals) ve geniş, narin damarlı çanak yaprakları (sepals) ile tanınır. Veri setinde diğer türlerden doğrusal olarak en net ayrılan türdür."
    },
    1: {
        "name": "Iris-versicolor",
        "tr_name": "Versicolor Süseni",
        "tag": "Versicolor",
        "color": "#3B82F6", # Kobalt Mavi
        "color_rgb": "59, 130, 246",
        "image": "/static/images/iris_versicolor.jpg?v=botanical_v2",
        "badge": "🌺 Versicolor",
        "sepal_typical": "5.5 - 6.5 cm",
        "petal_typical": "3.5 - 5.0 cm (Orta Boy)",
        "description": "Dengeli orta boy taç yaprakları, parlak kobalt mavisi tonları ve dipteki sarı-beyaz kaplan çizgili deseniyle karakterizedir. Virginica ile sınır bölgesinde geçiş gösterir."
    },
    2: {
        "name": "Iris-virginica",
        "tr_name": "Virginica Süseni",
        "tag": "Virginica",
        "color": "#8B5CF6", # Asil Koyu Mor
        "color_rgb": "139, 92, 246",
        "image": "/static/images/iris_virginica.jpg?v=botanical_v2",
        "badge": "🪻 Virginica",
        "sepal_typical": "6.0 - 7.9 cm",
        "petal_typical": "5.0 - 6.9 cm (En İri)",
        "description": "Üç türün en iri, en gösterişli süsenidir. Uzun, kadifemsi koyu mor taç yaprakları ve çanak yaprak tabanındaki parlak altın sarısı sakal deseniyle belirgindir."
    }
}

# ==============================================================================
# 📋 PYDANTIC VERİ DOĞRULAMA (INPUT VALIDATION)
# ==============================================================================
class IrisInput(BaseModel):
    """
    Kullanıcıdan gelen fiziksel yaprak ölçümlerinin tipini ve geçerlilik sınırlarını denetler.
    ge: Greater or equal (en az 0.1 cm)
    le: Less or equal (en fazla 15.0 cm)
    """
    sepal_length: float = Field(..., ge=0.1, le=15.0, description="Çanak yaprak uzunluğu (cm)", example=5.1)
    sepal_width: float = Field(..., ge=0.1, le=15.0, description="Çanak yaprak genişliği (cm)", example=3.5)
    petal_length: float = Field(..., ge=0.1, le=15.0, description="Taç yaprak uzunluğu (cm)", example=1.4)
    petal_width: float = Field(..., ge=0.1, le=15.0, description="Taç yaprak genişliği (cm)", example=0.2)

# ==============================================================================
# 🌐 WEB ARAYÜZÜ ROTASI
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Ana sayfa HTML dosyasını sunar."""
    index_file = templates_path / "index.html"
    return FileResponse(index_file)

# ==============================================================================
# 🔮 TAHMİN (PREDICTION) API ENDPOINT
# ==============================================================================
@app.post("/api/predict")
async def predict_species(data: IrisInput):
    """
    ÇOKLU SINIFLANDIRMA TAHMİN ADIMLARI (INFERENCE PIPELINE):
    1. Girdileri PyTorch Tensörüne Dönüştürme:
       - Kullanıcının girdiği 4 float değer, [1, 4] boyutlu torch.float32 tensörüne çevrilir.
       
    2. torch.inference_mode():
       - Gradyan (türev) takibini kapatır.
       - PyTorch'un Autograd motorunu devre dışı bırakarak RAM tasarrufu ve 2x hız sağlar.
       
    3. İleri Yayılım (Forward Pass):
       - logits = model(features)
       - Model ham skorlar (Lojitler) üretir. Örn: [14.78, 5.77, -18.86]
       
    4. Softmax Aktivasyonu:
       - probs = torch.softmax(logits, dim=1)
       - Ham puanları, toplamı tam 1.0 (%100) olan olasılık dağılımına sıkıştırır.
       
    5. Argmax ile En Yüksek Sınıfı Seçme:
       - pred_idx = torch.argmax(probs).item()
       - En yüksek olasılığa sahip sınıfın indeksini (0, 1 veya 2) çeker.
    """
    try:
        # Adım 1: PyTorch Tensörüne dönüştür
        features = torch.tensor(
            [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
            dtype=torch.float32
        )

        # Adım 2 & 3 & 4: Gradyansız tahmin ve olasılık hesabı
        with torch.inference_mode():
            logits = model(features)                      # Ham lojitler
            probs = torch.softmax(logits, dim=1)[0]       # Olasılıklar (0.0 - 1.0)
            pred_idx = torch.argmax(probs).item()         # Kazanan sınıf indeksi (0, 1 veya 2)

        species_data = SPECIES_INFO[pred_idx]
        confidence = float(probs[pred_idx].item() * 100)

        # Ham lojitleri de eğitim ve analiz için dışarı açıyoruz
        raw_logits = {
            SPECIES_INFO[i]["name"]: round(float(logits[0][i].item()), 3)
            for i in range(3)
        }

        # Tüm sınıfların yüzde olasılıkları
        all_probs = {
            SPECIES_INFO[i]["name"]: round(float(probs[i].item() * 100), 2)
            for i in range(3)
        }

        return {
            "status": "success",
            "prediction": {
                "class_index": pred_idx,
                "species": species_data["name"],
                "tr_name": species_data["tr_name"],
                "tag": species_data["tag"],
                "color": species_data["color"],
                "color_rgb": species_data["color_rgb"],
                "image": species_data["image"],
                "badge": species_data["badge"],
                "description": species_data["description"],
                "confidence": round(confidence, 2)
            },
            "probabilities": all_probs,
            "raw_logits": raw_logits,
            "inputs": {
                "sepal_length": data.sepal_length,
                "sepal_width": data.sepal_width,
                "petal_length": data.petal_length,
                "petal_width": data.petal_width
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin işlemi sırasında hata: {str(e)}")

# ==============================================================================
# 🩺 SAĞLIK KONTROLÜ (HEALTH CHECK) ENDPOINT
# ==============================================================================
@app.get("/api/health")
async def health_check():
    """Modelin diskte var olup olmadığını ve servisin ayakta olduğunu doğrular."""
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
        "device": "cpu"
    }
