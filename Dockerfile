FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt ./

# Set environment variable for Hugging Face API key
ENV HF_API_KEY=${HF_API_KEY}

# Install dependencies
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install torch torchvision torchaudio

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "webpage_B.py", "--server.port=8501", "--server.address=0.0.0.0"]