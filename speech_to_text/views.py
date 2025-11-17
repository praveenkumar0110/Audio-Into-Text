from django.shortcuts import render
from transformers import pipeline
import tempfile

model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def index(request):
    
    text = None

    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        result =  model(tmp_path, return_timestamps=True)
        text = result["text"]

    return render(request, "index.html", {"text": text})
