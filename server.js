import express from "express";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());

// Static dosyalar
app.use(express.static(__dirname));

// API Key havuzu
const apiKeys = [
    process.env.GROQ_API_KEY_1,
    process.env.GROQ_API_KEY_2,
    process.env.GROQ_API_KEY_3,
    process.env.GROQ_API_KEY_4
].filter(Boolean);
console.log("Bulunan API key:", apiKeys.length);

let keyIndex = 0;

function getApiKey() {
    const key = apiKeys[keyIndex];
    keyIndex = (keyIndex + 1) % apiKeys.length;
    return key;
}


// AI endpoint
app.post("/api/chat", async (req, res) => {
    try {
        const { message } = req.body;

        if (!message) {
            return res.status(400).json({
                error: "Mesaj yok"
            });
        }

        const apiKey = getApiKey();

        if (!apiKey) {
            return res.status(500).json({
                error: "API key bulunamadı"
            });
        }


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
                            content:
                            "Sen Goober AI'sın. Sevecen, yardımsever ve arkadaş canlısısın."
                        },
                        {
                            role: "user",
                            content: message
                        }
                    ]
                })
            }
        );

        console.log("Groq status:", response.status);
        const data = await response.json();

        res.json({
            reply:
            data.choices?.[0]?.message?.content ||
            "Cevap alınamadı."
        });


    } catch (err) {
        console.error(err);

        res.status(500).json({
            error: "Sunucu hatası"
        });
    }
});


app.listen(PORT, () => {
    console.log(`Goober server running on ${PORT}`);
});