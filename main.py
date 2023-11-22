"""メイン"""
import argparse
from typing import Final, List
from faster_whisper import WhisperModel
from fastapi import FastAPI
import uvicorn
import numpy as np
from pydantic import BaseModel



class TranscribeRequestBody(BaseModel):
    """
    transcribe()のリクエストボディ
    {
        "audio":[...]
    }
    """
    audio: List[float]


MODEL_SIZE: Final[str] = "large-v2"

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float32")

app: FastAPI = FastAPI()


@app.get("/")
def root():
    """Hello World"""
    return {"message": "Hello World"}


@app.post("/transcribe")
def transcribe(request_body: TranscribeRequestBody):
    """文字起こし"""
    audio = np.array(request_body.audio, dtype=np.float32)
    segments, info = model.transcribe(audio, word_timestamps=True)

    list_segments = []
    for seg in segments:
        list_segments.append({"text": seg.text})
    return {"language": info.language, "segments": list_segments}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=17860)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
