import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { getLlama, LlamaChatSession } from "node-llama-cpp";

const app = express();
const PORT = process.env.PORT || 10000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());
app.use(express.static(__dirname));

// Llama Engine & Model değişkenleri
let llama;
let model;
let context;
let session;

// HuggingFace'ten otomatik indirme ve modeli yükleme fonksiyonu
async function initLlamaModel() {
    console.log("⚙️  Llama Engine başlatılıyor...");
    llama = await getLlama();

    console.log("🔍 Model kontrol ediliyor / HuggingFace'ten indiriliyor...");

    // getModelFile otomatik olarak:
    // 1. Modeli yerel önbellekte (cache) arar.
    // 2. Bulamazsa verilen HuggingFace reposundan Q4_K_M (4-bit) GGUF dosyasını indirir!
    const modelPath = await llama.getModelFile({
        // Buraya istediğin HuggingFace reposunu yazabilirsin (Örn: Qwen 1.5 1.8B, Llama 3B, vb.)
        repo: "Qwen/Qwen1.5-1.8B-Chat-GGUF",
        // 4-bit quantization dosyasını hedefliyoruz:
        fileName: "*q4_k_m.gguf" 
    });

    console.log(`✅ Model hazır: ${modelPath}`);

    // Modeli belleğe yükle
    model = await llama.loadModel({ modelPath });
    context = await model.createContext();
    
    // Sistem promptu ile oturum oluştur
    session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt: "Sen GooberAI'sın. Sevecen, yardımsever ve neşeli bir yapay zeka asistanısın."
    });

    console.log("🚀 Goober Model Oturumu Başarıyla Oluşturuldu!");
}

// Sunucu başlamadan önce modeli hazırla
initLlamaModel().catch((err) => {
    console.error("❌ Model yüklenirken hata oluştu:", err);
});

// Chat API Endpoint
app.post("/api/chat", async (req, res) => {
    try {
        const { messages } = req.body;

        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({ error: "Geçerli bir mesaj bulunamadı." });
        }

        // Son kullanıcı mesajını al
        const lastUserMessage = messages[messages.length - 1]?.content;

        if (!lastUserMessage) {
            return res.status(400).json({ error: "Boş mesaj gönderilemez." });
        }

        if (!session) {
            return res.status(503).json({ error: "Model henüz yükleniyor, lütfen birkaç saniye sonra tekrar deneyin." });
        }

        // Modelleri yormamak için basit Prompt hazırlığı
        const reply = await session.prompt(lastUserMessage, {
            maxTokens: 512,
            temperature: 0.7
        });

        res.json({ reply });

    } catch (err) {
        console.error("Inference Hatası:", err);
        res.status(500).json({ error: "Model yanıt üretirken bir hata oluştu." });
    }
});

app.listen(PORT, () => {
    console.log(`Goober Local Engine http://localhost:${PORT} adresinde aktif!`);
});