FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages required for embeddings
RUN pip install --no-cache-dir torch torchvision torchaudio \
    transformers sentence-transformers

# Copy app files
COPY . .

# Fix Streamlit async issue
RUN python -c "import asyncio; asyncio.set_event_loop(asyncio.new_event_loop())"

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

