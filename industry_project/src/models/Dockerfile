# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (this makes cloud building faster)
COPY requirements.txt .

# 4. Install the libraries you defined
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your Python scripts and the data/ folder into the container
COPY . .

# 6. Expose BOTH ports: 8000 for FastAPI (Backend) and 8501 for Streamlit (Frontend)
EXPOSE 8000
EXPOSE 8501

# 7. Default start command (FastAPI)
# Note: Our Kubernetes deployment.yaml automatically overrides this for the Streamlit container!
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]