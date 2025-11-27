FROM python:3.12-slim-bookworm

WORKDIR /code

RUN apt-get update && apt-get install -y \
    libmagic-dev \
    libreoffice \
    pandoc \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh ./install.sh
RUN chmod +x ./install.sh && ./install.sh

ENV PATH=/root/.local/bin:${PATH}

COPY pyproject.toml .

RUN uv sync --no-cache

EXPOSE 7860
EXPOSE 5678

CMD ["uv", "run", "python3", "-Xfrozen_modules=off", \
        "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", \
        "-m", "rag_chatbot", "--host", "host.docker.internal"]