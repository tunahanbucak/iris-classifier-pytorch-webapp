import torch
from torch import nn

class IrisClassifier(nn.Module):
    """
    🌸 Iris Çiçek Veri Seti için Çok Sınıflı (Multiclass) Yapay Sinir Ağı Mimarisi
    -------------------------------------------------------------------------
    PyTorch'ta bir model tanımlarken `nn.Module` sınıfından kalıtım (inheritance) alırız.
    
    KATMAN DETAYLARI:
    1. Giriş Katmanı (in_features=4):
       - Çiçeğin 4 temel fiziksel ölçümü:
         [Sepal Length, Sepal Width, Petal Length, Petal Width]
       - Ağırlık (Weight) Matrisi: 16x4, Bias Vektörü: 16 elemanlı
       
    2. Aktivasyon Fonksiyonu (ReLU):
       - f(x) = max(0, x)
       - Neden Gerekli? Eğer katmanlar arasına doğrusal olmayan (non-linear) bir aktivasyon
         koymazsak, arka arkaya kaç katman dizerseniz dizin matematiksel olarak tek bir
         matris çarpımına indirgenir. ReLU sayesinde ağ, kıvrımlı ve karmaşık karar sınırları çizer.
         
    3. Gizli Katman (Hidden Layer):
       - 16 nöron -> 16 nöron doğrusal katman + ReLU aktivasyonu.
       - Modelin veri setindeki özellikleri (features) soyutlayarak daha derin temsil etmesini sağlar.
       
    4. Çıkış Katmanı (out_features=3):
       - 3 farklı Iris sınıfı için ham puanlar üretir:
         [0: Setosa, 1: Versicolor, 2: Virginica]
         
    🔥 KRİTİK NOT (LOJİT vs SOFTMAX):
       - Neden modelin sonuna `nn.Softmax()` koymadık?
       - Çünkü PyTorch'ta eğitim sırasında kullanılan `nn.CrossEntropyLoss()` fonksiyonu,
         arkada `LogSoftmax` ve `NLLLoss` (Negative Log Likelihood) adımlarını matematiksel
         olarak birleşik ve sayısal açıdan en kararlı şekilde (Log-Sum-Exp hilesiyle) uygular.
       - Bu yüzden PyTorch modellerinin çıkışı daima normalleştirilmemiş ham puanlar,
         yani LOJİT (Logits) olmalıdır!
    """
    def __init__(self):
        super().__init__()
        # nn.Sequential: Katmanları sırasıyla birbirine bağlayan boru hattı (pipeline)
        self.linear_layer_stack = nn.Sequential(
            # Katman 1: 4 giriş -> 16 gizli nöron
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU(),
            
            # Katman 2: 16 nöron -> 16 nöron
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            
            # Çıkış Katmanı: 16 nöron -> 3 sınıf lojiti
            nn.Linear(in_features=16, out_features=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri Yayılım (Forward Pass):
        Tensör modele girdiğinde bu fonksiyon tetiklenir ve katmanlar sırasıyla çalışır.
        
        Girdi:  [batch_size, 4] boyutunda float32 tensör
        Çıktı:  [batch_size, 3] boyutunda lojit (ham skor) tensörü
        """
        return self.linear_layer_stack(x)
