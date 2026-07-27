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


const apiKeys = [
    process.env.GROQ_API_KEY_1,
    process.env.GROQ_API_KEY_2,
    process.env.GROQ_API_KEY_3,
    process.env.GROQ_API_KEY_4
].filter(Boolean);


let keyIndex = 0;

function getApiKey(){
    const key = apiKeys[keyIndex];
    keyIndex = (keyIndex + 1) % apiKeys.length;
    return key;
}


app.post("/api/chat", async (req,res)=>{

    try {

        const {message} = req.body;

        if(!message){
            return res.status(400).json({
                error:"Mesaj boş"
            });
        }


        const apiKey = getApiKey();

        console.log("Kullanılan key:", keyIndex);


        const response = await fetch(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                method:"POST",
                headers:{
                    "Content-Type":"application/json",
                    "Authorization":`Bearer ${apiKey}`
                },
                body:JSON.stringify({
                    model:"llama-3.1-8b-instant",
                    messages:[
                        {
                            role:"system",
                            content:"Sen GooberAI'sın. Sevecen, yardımsever ve ciddi bir yapay zekasın."
                        },
                        {
                            role:"user",
                            content:message
                        }
                    ]
                })
            }
        );


        const data = await response.json();


        res.json({
            reply:
            data.choices?.[0]?.message?.content ??
            "Boş cevap geldi."
        });


    }catch(err){

        console.error(err);

        res.status(500).json({
            error:"Sunucu hatası"
        });

    }

});


app.listen(PORT,()=>{
    console.log("Goober server running on",PORT);
});