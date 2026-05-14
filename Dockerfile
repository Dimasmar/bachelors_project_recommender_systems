FROM python:3.13-slim
LABEL authors="Марченко"

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m appuser
USER appuser
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]