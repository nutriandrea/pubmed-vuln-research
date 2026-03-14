FROM python:3.11

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application with uvicorn (ASGI server for FastAPI)
CMD ["uvicorn", "web.api:app", "--host", "0.0.0.0", "--port", "8000"]
