FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost mlxtend pyarrow
COPY . .
ENV PYTHONPATH=/app
CMD ["python", "-m", "ml.run_ml"]