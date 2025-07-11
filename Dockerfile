# Use an official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U langchain-community
# Copy your application code into the container
COPY ./app ./app
COPY ./faiss_index ./faiss_index
COPY ./data ./data
COPY .env .env

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

