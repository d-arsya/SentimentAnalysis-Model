# Use Python 3.10 slim image (lebih kecil dari full image)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu agar cache layer lebih optimal
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the application port
EXPOSE 8000

# Run FastAPI with Uvicorn (production ready, multiple workers)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]