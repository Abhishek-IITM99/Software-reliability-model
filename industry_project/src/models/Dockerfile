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

# 6. Expose the port the API runs on
EXPOSE 8000

# 7. Start the FastAPI server when the container wakes up
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]