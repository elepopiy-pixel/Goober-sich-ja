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

let llama;
let model;
let context;
let session;

async function initLlamaModel() {
    console.log("⚙️  Llama Engine başlatılıyor...");
    llama = await getLlama();

    console.log("🔍 Model HuggingFace'ten çekiliyor...");

    // HuggingFace URI formatı (v3 uyumlu)
    const modelPath = await llama.getModelFile(
        "hf:bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    );

    console.log(`✅ Model indirildi/hazırlandı: ${modelPath}`);

    model = await llama.loadModel({ modelPath });
    context = await model.createContext();

    session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt: "Sen Goober'sın! Neşeli ve yardımsever bir yapay zeka asistanısın."
    });

    console.log("🚀 Goober Model Oturumu Hazır!");
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