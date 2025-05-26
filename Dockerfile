FROM python:3.12-slim

WORKDIR /app
RUN mkdir -p pypef

COPY requirements.txt run.py /app/
COPY pypef/ /app/pypef/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN ["python", "-c", "import torch;print(torch.__version__)"]

EXPOSE 5000

# Not defining entrypoint herein for eased chaining of multiple commands 
# with /bin/bash -c "command1 && command2..."
#ENTRYPOINT ["python", "/app/run.py"]
