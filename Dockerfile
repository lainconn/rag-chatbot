FROM downloads.unstructured.io/unstructured-io/unstructured:latest
ENV PYTHONPATH="/app"
WORKDIR /code
USER root
COPY ./ /code

RUN pip install --break-system-packages .

EXPOSE 7860
EXPOSE 5678

CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "rag_chatbot", "--host", "host.docker.internal"]