FROM downloads.unstructured.io/unstructured-io/unstructured:latest
ENV PYTHONPATH="/app"
WORKDIR /code
USER root
COPY ./ /code

RUN pip install --break-system-packages .

EXPOSE 7860

CMD ["python3", "-m", "rag_chatbot", "--host", "host.docker.internal"]