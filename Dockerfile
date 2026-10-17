FROM python:3.10-slim

# System libraries for OpenCV and for building insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image much smaller; remove the index URL for GPU builds
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN mkdir -p data outputs

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
