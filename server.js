import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { getLlama, createModelDownloader, LlamaChatSession } from "node-llama-cpp";

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

    const modelFileName = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    const modelPath = path.join(__dirname, modelFileName);

    // Modeli yerelde yoksa HuggingFace'ten indiriyoruz
    if (!fs.existsSync(modelPath)) {
        console.log("🔍 0.5B Model yerelde bulunamadı, HuggingFace'ten indiriliyor...");
        
        const downloader = await createModelDownloader({
            modelUrl: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            dirPath: __dirname,
            fileName: modelFileName
        });

        await downloader.download();
        console.log("✅ 0.5B Model başarıyla indirildi!");
    } else {
        console.log("📂 0.5B Model zaten diskte mevcut, yükleniyor...");
    }

    console.log("🧠 Model minimum RAM ayarlarıyla belleğe yükleniyor...");
    
    model = await llama.loadModel({ 
        modelPath,
        gpuLayers: 0
    });

    // RAM Tüketimini 512MB altında tutmak için kritik ayarlar:
    context = await model.createContext({
        contextSize: 512,  // Bellek tasarrufu için 512 token
        batchSize: 128
    });

    session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt: "Sen Goober'sın! Neşeli, sevecen ve yardımsever bir yapay zeka asistanısın."
    });

    console.log("🚀 Goober 0.5B Local Model Oturumu Başarıyla Oluşturuldu!");
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
            return res.status(503).json({ error: "Model yükleniyor, lütfen bekleyin..." });
        }

        const reply = await session.prompt(lastUserMessage, {
            maxTokens: 256,
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