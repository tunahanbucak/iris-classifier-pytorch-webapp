/**
 * ==============================================================================
 * 🌸 IRIS CLASSIFIER - İSTEMCİ TARAFI ETKİLEŞİM & TAHMİN MOTORU (JS)
 * ==============================================================================
 * Bu dosya, kullanıcının girdiği slider değerlerini dinler, bunları asenkron
 * olarak FastAPI (/api/predict) servisine gönderir ve PyTorch modelinden dönen
 * ham lojitler, Softmax olasılıkları ve görsel verilerini arayüze basar.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const inputs = ["sepal_length", "sepal_width", "petal_length", "petal_width"];

    // 1. Slider girdilerini canlı değer rozetlerine (badges) bağla
    inputs.forEach(id => {
        const slider = document.getElementById(id);
        const badge = document.getElementById(`val_${id}`);
        slider.addEventListener("input", () => {
            badge.textContent = `${slider.value} cm`;
            // Debounce: Kullanıcı slider'ı kaydırırken arka arkaya yüzlerce istek
            // atıp sunucuyu boğmamak için 100ms gecikmeli tetikleme yapıyoruz.
            debouncePredict();
        });
    });

    // Yardımcı: Belirli bir slider'ın değerini programatik olarak güncelle
    function setSlider(id, val) {
        const el = document.getElementById(id);
        el.value = val;
        document.getElementById(`val_${id}`).textContent = `${val} cm`;
    }

    // 2. Hazır Test Butonları (Presets)
    document.querySelectorAll(".btn-preset").forEach(btn => {
        btn.addEventListener("click", () => {
            applyValues(
                btn.getAttribute("data-sl"),
                btn.getAttribute("data-sw"),
                btn.getAttribute("data-pl"),
                btn.getAttribute("data-pw")
            );
        });
    });

    // 3. Karşılaştırma Galerisindeki Kartlara Tıklanınca Değerleri Yükle
    document.querySelectorAll(".species-card").forEach(card => {
        card.addEventListener("click", () => {
            applyValues(
                card.getAttribute("data-sl"),
                card.getAttribute("data-sw"),
                card.getAttribute("data-pl"),
                card.getAttribute("data-pw")
            );
            if (window.innerWidth <= 860) {
                document.getElementById("photo-frame").scrollIntoView({ behavior: "smooth" });
            }
        });
    });

    function applyValues(sl, sw, pl, pw) {
        setSlider("sepal_length", sl);
        setSlider("sepal_width", sw);
        setSlider("petal_length", pl);
        setSlider("petal_width", pw);
        executePrediction();
    }

    // 4. Debounce Zamanlayıcısı (Performans Optimizasyonu)
    let debounceTimer;
    function debouncePredict() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(executePrediction, 100);
    }

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        executePrediction();
    });

    // 5. Asenkron Tahmin İsteği (Fetch API -> FastAPI)
    async function executePrediction() {
        const payload = {
            sepal_length: parseFloat(document.getElementById("sepal_length").value),
            sepal_width: parseFloat(document.getElementById("sepal_width").value),
            petal_length: parseFloat(document.getElementById("petal_length").value),
            petal_width: parseFloat(document.getElementById("petal_width").value)
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("API hatası");
            const data = await response.json();
            updateUI(data);
        } catch (err) {
            console.error("Tahmin alınamadı:", err);
        }
    }

    // 6. Arayüzü Model Çıktılarıyla Güncelleme (UI Synchronizer)
    function updateUI(data) {
        const pred = data.prediction;
        const probs = data.probabilities;
        const logits = data.raw_logits || {};

        // A. Botanik Görsel ve Renk Teması
        const flowerImg = document.getElementById("flower-img");
        flowerImg.src = pred.image;

        const photoFrame = document.getElementById("photo-frame");
        photoFrame.style.borderColor = `rgba(${pred.color_rgb}, 0.6)`;
        photoFrame.style.boxShadow = `0 14px 34px rgba(${pred.color_rgb}, 0.3)`;

        const speciesTagBadge = document.getElementById("species-tag-badge");
        speciesTagBadge.style.color = pred.color;
        speciesTagBadge.style.borderColor = `rgba(${pred.color_rgb}, 0.5)`;
        document.getElementById("badge-species-text").textContent = pred.species;

        document.getElementById("res-tr-name").textContent = pred.tr_name;
        document.getElementById("res-badge").textContent = pred.species;
        document.getElementById("res-desc").textContent = pred.description;
        document.getElementById("res-desc").style.borderLeftColor = pred.color;

        // B. Model Güven Skoru (Confidence)
        document.getElementById("res-conf").textContent = `%${pred.confidence}`;
        document.getElementById("res-conf").style.color = pred.color;
        const confBar = document.getElementById("res-conf-bar");
        confBar.style.width = `${pred.confidence}%`;
        confBar.style.background = pred.color;

        // C. Softmax Olasılık Çubukları Dağılımı
        const setosaP = probs["Iris-setosa"] || 0;
        const versicolorP = probs["Iris-versicolor"] || 0;
        const virginicaP = probs["Iris-virginica"] || 0;

        document.getElementById("p-setosa").textContent = `${setosaP}%`;
        document.getElementById("b-setosa").style.width = `${setosaP}%`;

        document.getElementById("p-versicolor").textContent = `${versicolorP}%`;
        document.getElementById("b-versicolor").style.width = `${versicolorP}%`;

        document.getElementById("p-virginica").textContent = `${virginicaP}%`;
        document.getElementById("b-virginica").style.width = `${virginicaP}%`;

        // D. 🧠 Derin Öğrenme İç Mekanizması (Ham Lojitler ve Argmax Kararı)
        if (logits["Iris-setosa"] !== undefined) {
            const formatLogit = (val) => (val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2));
            document.getElementById("logit-setosa").textContent = formatLogit(logits["Iris-setosa"]);
            document.getElementById("logit-versicolor").textContent = formatLogit(logits["Iris-versicolor"]);
            document.getElementById("logit-virginica").textContent = formatLogit(logits["Iris-virginica"]);
        }

        const argmaxBadge = document.getElementById("res-argmax-idx");
        if (argmaxBadge) {
            argmaxBadge.textContent = `Sınıf ${pred.class_index} (${pred.tag})`;
            argmaxBadge.style.color = pred.color;
            argmaxBadge.style.borderColor = `rgba(${pred.color_rgb}, 0.5)`;
            argmaxBadge.style.background = `rgba(${pred.color_rgb}, 0.15)`;
        }

        // E. Galerideki İlgili Kartı Vurgula
        document.querySelectorAll(".species-card").forEach(c => {
            c.classList.remove("active-card");
            c.style.removeProperty("--card-active-color");
        });

        const activeCardId = pred.class_index === 0 ? "card-setosa" :
                             pred.class_index === 1 ? "card-versicolor" : "card-virginica";
        const activeCard = document.getElementById(activeCardId);
        if (activeCard) {
            activeCard.classList.add("active-card");
            activeCard.style.setProperty("--card-active-color", pred.color);
        }
    }

    // Sayfa ilk yüklendiğinde varsayılan değerlerle ilk tahmini çalıştır
    executePrediction();
});
