FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Preload and store PLMs in cache, increases image size by ~ 10 GBs, saves models
# to /root/.cache/huggingface/hub/ inside the Docker image.
# Comment for skipping caching e.g., for mounting your own cache directories
# or downloading weights every new docker run.
RUN python -c "from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer; \
    AutoModelForMaskedLM.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True); \
    AutoTokenizer.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True); \
    AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    AutoModelForMaskedLM.from_pretrained('facebook/esm1v_t33_650M_UR90S_3'); \
    AutoTokenizer.from_pretrained('facebook/esm1v_t33_650M_UR90S_3')"

# Copy application code after pip install
COPY run.py /app/
COPY pypef/ /app/pypef/

RUN ["python", "-c", "import torch;print(torch.__version__)"]

EXPOSE 5000

# Not defining CMD["python", "/app/run.py"] as CMD/ENTRYPOINT herein 
# but "standard" /bin/bash terminal.
CMD ["/bin/bash"]
