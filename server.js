import express from "express";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());
app.use(express.static(__dirname));

// Çevre değişkenlerinden API anahtarlarını al ve boş olanları filtrele
const apiKeys = [
    process.env.GROQ_API_KEY_1,
    process.env.GROQ_API_KEY_2,
    process.env.GROQ_API_KEY_3,
    process.env.GROQ_API_KEY_4
].filter(Boolean);

let keyIndex = 0;

function getApiKey() {
    if (apiKeys.length === 0) {
        throw new Error("Hiçbir GROQ API anahtarı tanımlanmamış!");
    }
    const key = apiKeys[keyIndex];
    keyIndex = (keyIndex + 1) % apiKeys.length;
    return key;
}

app.post("/api/chat", async (req, res) => {
    try {
        const { message } = req.body;

        // 1. Boş mesaj kontrolü
        if (!message || typeof message !== "string" || !message.trim()) {
            return res.status(400).json({
                error: "Mesaj boş olamaz."
            });
        }

        // 2. Karakter / Token sınırı kontrolü (8500 karakter sınırı)
        const MAX_CHAR_LIMIT = 8500;
        if (message.length > MAX_CHAR_LIMIT) {
            return res.status(400).json({
                error: `Mesajınız çok uzun! Maksimum ${MAX_CHAR_LIMIT} karakter (yaklaşık 2048 token) gönderebilirsiniz. Gönderilen: ${message.length} karakter.`
            });
        }

        const apiKey = getApiKey();
        console.log("Kullanılan key indeksi:", keyIndex);

        const response = await fetch(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: "llama-3.1-8b-instant",
                    messages: [
                        {
                            role: "system",
                            content: "Sen GooberAI'sın. Sevecen, yardımsever ve ciddi bir yapay zekasın."
                        },
                        {
                            role: "user",
                            content: message
                        }
                    ],
                    max_tokens: 2048 // Modelin vereceği cevabın da taşmaması için güvenlik sınırı
                })
            }
        );

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error("Groq API Hatası:", errorData);
            return res.status(response.status).json({
                error: errorData.error?.message || "Yapay zeka servisinden yanıt alınamadı."
            });
        }

        const data = await response.json();

        res.json({
            reply:
                data.choices?.[0]?.message?.content ??
                "Boş cevap geldi."
        });

    } catch (err) {
        console.error("Sunucu İçi Hata:", err.message);

        res.status(500).json({
            error: "Sunucu hatası oluştu."
        });
    }
});

app.listen(PORT, () => {
    console.log("Goober server running on port", PORT);
});