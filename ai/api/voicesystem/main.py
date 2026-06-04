from fastapi import FastAPI, File, UploadFile, Query
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
import torch
import json
import httpx
from typing import Union

app = FastAPI()

modelPath = "/app/qwen_syrian_finetuned"

tokenizer = PreTrainedTokenizerFast.from_pretrained(modelPath, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(
    modelPath,
    dtype=torch.float32,
    device_map="auto"
)

systemMessage = {
    "role": "system",
    "content": """
أنت مساعد طبي سورري تقوم بتلخيص الدخل ضمن ملف json، كمثال عندما تستقبل جملة تقول "جايينا حالة من دوما، شوفير شاحنة، ستة وعشرين سنة. العين اليسار فيها احتمال يكون عنده قرنية مخروطية" فعندها تقوم بارجاع ملف json بالشكل التالي:


       "extracted_fields": {
           "جهة العين": "اليسار",
           "الجنس": "ذكر",
           "العمر": 26,
           "المدينة": "دوما",
           "الحالة": "مشتبه به",
           "المهنة": "شوفير شاحنة",
           "ملاحظات": null
       }

 مثال آخر: الجملة: " هاد المريض الله يسلمك من النبك، أنثى، بيشتغل خياط، وعمره خمسين. العين التنتين سوا احتمال يكون عنده قرنية مخروطية. ملاحظة: عامل عملية سابقة. "
تقوم بارجاعها على الشكل:
       "extracted_fields": {
           "جهة العين": "كلتا العينين",
           "الجنس": "أنثى",
           "العمر": 50,
           "المدينة": "النبك",
           "الحالة": "مشتبه به",
           "المهنة": "خياط",
           "ملاحظات": "عامل عملية سابقة"
       }

       مثال آخر: "العين اليمين قرنيته سليمة مية بالمية. هاد المريض شوفير شاحنة من حمص وعمره خمسة وتلاتين. ملاحظة: القرنية رقيقة"

       تصبخ:

               "extracted_fields": {
            "جهة العين": "اليمين",
            "الجنس": "أنثى",
            "العمر": 35,
            "المدينة": "حمص",
            "الحالة": "سليم",
            "المهنة": "شوفير شاحنة",
            "ملاحظات": "القرنية رقيقة"
        }
       
لا ترسل اي شيء اضافي!!
       
         """
}

ARABIC_KEY_MAP = {
    "جهة العين": "EyeSide",
    "الجنس": "Gender",
    "العمر": "Age",
    "المدينة": "City",
    "الحالة": "Status",
    "المهنة": "Profession",
    "ملاحظات": "Notes",
}

whisperUrl = "http://whisper:9000/api/Samples/asr"
# whisperUrl = "http://whisper:9000/asr"

def mapArabicKeys(arabicJson: dict) -> dict:
    fields = arabicJson.get("extracted_fields", arabicJson)
    return {ARABIC_KEY_MAP.get(k, k): v for k, v in fields.items()}

async def transcribeAudio(audioFile: UploadFile, encode: bool, task: str, language: Union[str, None], output: str):
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"audioFile": (audioFile.filename, await audioFile.read(), audioFile.content_type)}
        params = {"encode": encode, "task": task, "output": output}
        if language:
            params["language"] = language
        response = await client.post(whisperUrl, files=files, params=params)
        response.raise_for_status()
        return response.text

def runLlm(text: str):
    messages = [systemMessage, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"raw_response": response, "error": "Could not parse JSON"}

@app.post("/api/Samples/{sampleId}/analyze")
async def analyzeSample(
    sampleId: str,
    audioFile: UploadFile = File(...),
    encode: bool = Query(default=True),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    transcribedText = await transcribeAudio(audioFile, encode, task, language, output)
    arabicResult = runLlm(transcribedText)
    englishResult = mapArabicKeys(arabicResult)
    return englishResult

@app.get("/api/Samples/{sampleId}/analyze")
async def getSampleAnalysis(sampleId: str):
    return {"sample_id": sampleId, "message": "Use POST to submit an audio file for analysis."}

@app.get("/api/Samples/asr")
async def getAsrInfo():
    return {"message": "Use POST to transcribe an audio file via Whisper."}

@app.get("/health")
def health():
    return {"status": "ok"}
