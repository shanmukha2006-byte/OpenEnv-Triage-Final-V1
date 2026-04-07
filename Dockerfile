FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pydantic openai
EXPOSE 7860
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
CMD ["/bin/bash", "-c", "python3 -m http.server 7860 & python3 inference.py"]