FROM python:3.11-slim

# تثبيت مكتبات النظام المطلوبة لـ OpenCV و MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rehab_api.py .

# بورت 7860 هو الافتراضي لـ Hugging Face (ويشتغل على أي سيرفر تاني)
EXPOSE 7860

CMD ["uvicorn", "rehab_api:app", "--host", "0.0.0.0", "--port", "7860"]
