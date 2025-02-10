#FROM downloads.unstructured.io/unstructured-io/unstructured:latest
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV PYTHONPATH="/app"
WORKDIR /code
USER root
COPY ./ /code


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip install --break-system-packages .

EXPOSE 7860

CMD ["python", "-m", "rag_chatbot", "--host", "host.docker.internal"]