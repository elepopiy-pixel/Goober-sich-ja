import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { getLlama, resolveModelFile, LlamaChatSession } from "node-llama-cpp";

const app = express();
const PORT = process.env.PORT || 10000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());
app.use(express.static(__dirname));

let llama;
let model;
let context;
let session;

async function initLlamaModel() {
    console.log("⚙️  Llama Engine başlatılıyor...");
    llama = await getLlama();

    console.log("🔍 Model HuggingFace üzerinden kontrol ediliyor / indiriliyor...");

    // v3+ ile model indirme/çözümleme resolveModelFile ile yapılır:
    const modelPath = await resolveModelFile({
        repo: "Qwen/Qwen1.5-1.8B-Chat-GGUF", // İstediğin 1B/1.8B repo
        fileName: "*q4_k_m.gguf"
    });

    console.log(`✅ Model hazır: ${modelPath}`);

    model = await llama.loadModel({ modelPath });
    context = await model.createContext();

    session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt: "Sen GooberAI'sın. Sevecen, yardımsever ve neşeli bir yapay zeka asistanısın."
    });

    console.log("🚀 Goober Model Oturumu Başarıyla Oluşturuldu!");
}

initLlamaModel().catch((err) => {
    console.error("❌ Model yüklenirken hata oluştu:", err);
});

app.post("/api/chat", async (req, res) => {
    try {
        const { messages } = req.body;
        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({ error: "Geçerli bir mesaj bulunamadı." });
        }

        const lastUserMessage = messages[messages.length - 1]?.content;

        if (!session) {
            return res.status(503).json({ error: "Model henüz yükleniyor, lütfen bekleyin..." });
        }

        const reply = await session.prompt(lastUserMessage, {
            maxTokens: 512,
            temperature: 0.7
        });

        res.json({ reply });

    } catch (err) {
        console.error("Inference Hatası:", err);
        res.status(500).json({ error: "Model yanıt üretirken hata oluştu." });
    }
});

app.listen(PORT, () => {
    console.log(`Goober Local Engine http://localhost:${PORT} adresinde aktif!`);
});