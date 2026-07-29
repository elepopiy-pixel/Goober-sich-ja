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

    const modelPath = path.join(__dirname, "Llama-3.2-1B-Instruct-Q4_K_M.gguf");

    if (!fs.existsSync(modelPath)) {
        console.log("🔍 Model yerelde bulunamadı, HuggingFace'ten indiriliyor...");
        
        const downloader = await createModelDownloader({
            modelUrl: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            dirPath: __dirname,
            fileName: "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        });

        await downloader.download();
        console.log("✅ İndirme tamamlandı!");
    } else {
        console.log("📂 Model zaten diskte mevcut, direkt yükleniyor...");
    }

    console.log("🧠 Model belleğe düşük RAM ayarlarıyla yükleniyor...");
    
    // RAM ÇÖKMESİNİ ENGELLEYEN AYARLAR:
    model = await llama.loadModel({ 
        modelPath,
        gpuLayers: 0 // Sadece CPU kullan
    });

    context = await model.createContext({
        contextSize: 512, // Varsayılan 4096 yerine 512 token (RAM'i %70 tasarruf ettirir)
        batchSize: 256
    });

    session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt: "Sen Goober'sın! Neşeli ve yardımsever bir yapay zeka asistanısın."
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
            return res.status(503).json({ error: "Model henüz yükleniyor veya indiriliyor, lütfen bekleyin..." });
        }

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
    console.log(`Goober Local Server http://localhost:${PORT} adresinde aktif!`);
});