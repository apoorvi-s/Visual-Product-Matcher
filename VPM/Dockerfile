# Dockerfile
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Collect static files
RUN python VPM/manage.py collectstatic --noinput || true

# Expose port
EXPOSE 7860

# Start Django server
CMD ["python", "VPM/manage.py", "runserver", "0.0.0.0:7860"]
