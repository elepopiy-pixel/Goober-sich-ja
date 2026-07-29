import express from "express";
import path from "path";
import fs from "fs";
import cors from "cors";
import { fileURLToPath } from "url";
import { getLlama, createModelDownloader, LlamaChatSession } from "node-llama-cpp";

const app = express();
const PORT = process.env.PORT || 10000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

let llama;
let model;
let context;
let session;

async function initLlamaModel() {
    console.log("⚙️  Llama Engine başlatılıyor...");
    llama = await getLlama();

    // 🎯 Hugging Face'deki %100 TAM ORİJİNAL KÜÇÜK HARFLİ DOSYA ADI:
    const modelFileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const modelPath = path.join(__dirname, modelFileName);

    // Sunucu başlatıldığında dosya diskte yoksa indirir, varsa direkt yükler
    if (!fs.existsSync(modelPath)) {
        console.log(`🔍 Model (${modelFileName}) bulunamadı, Hugging Face'ten indiriliyor...`);
        
        const downloader = await createModelDownloader({
            modelUrl: `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${modelFileName}`,
            dirPath: __dirname,
            fileName: modelFileName
        });

        await downloader.download();
        console.log("✅ Model başarıyla indirildi!");
    } else {
        console.log(`📂 Model (${modelFileName}) zaten diskte mevcut, yükleniyor...`);
    }

    console.log("🧠 Model belleğe yükleniyor...");
    
    model = await llama.loadModel({ 
        modelPath,
        gpuLayers: 0
    });

    context = await model.createContext({
        contextSize: 384,
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
            return res.status(503).json({ error: "Model henüz yükleniyor, lütfen bekleyin..." });
        }

        const reply = await session.prompt(lastUserMessage, {
            maxTokens: 128,
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