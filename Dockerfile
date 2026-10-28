FROM python:3.12-slim

WORKDIR /app
# API dependencies only (the Gradio demo isn't needed in the serving image).
RUN pip install --no-cache-dir numpy pandas scikit-learn joblib matplotlib fastapi uvicorn

COPY src/ src/
COPY models/ models/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "fraudguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
