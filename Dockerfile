FROM downloads.unstructured.io/unstructured-io/unstructured:latest
ENV PYTHONPATH="/app"
WORKDIR /code
USER root
COPY ./ /code
RUN pip install .

EXPOSE 7860

CMD ["python", "-m", "rag_chatbot", "--host", "host.docker.internal"]