FROM python:3.12-slim-bookworm

WORKDIR /code

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

RUN apt-get update && apt-get install -y \
    libmagic-dev \
    libreoffice \
    pandoc \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 7860
EXPOSE 5678

CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "rag_chatbot", "--host", "host.docker.internal"]