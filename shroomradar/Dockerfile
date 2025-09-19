# Use official Python image as a base
FROM python:3.13

# Install system dependencies including curl, unzip, and fuse
RUN apt-get update && apt-get install -y \
    curl unzip fuse \
    && rm -rf /var/lib/apt/lists/*

# Install Rclone
RUN curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o rclone.zip && \
    unzip rclone.zip && \
    cd rclone-*-linux-amd64 && \
    cp rclone /usr/local/bin/ && \
    chmod +x /usr/local/bin/rclone && \
    cd .. && rm -rf rclone.zip rclone-*-linux-amd64


# Add Tippecanoe
RUN apt-get update && apt-get install -y build-essential libsqlite3-dev zlib1g-dev \
&& git clone https://github.com/mapbox/tippecanoe.git \
&& cd tippecanoe && make -j && make install \
&& cd .. && rm -rf tippecanoe


# Create Rclone config directory
RUN mkdir -p /root/.config/rclone/

# Copy pre-generated Rclone configuration file
COPY data/rclone.conf /root/.config/rclone/rclone.conf

# Set the working directory inside the container
WORKDIR /app

# Copy your Python script and any other required files into the container
COPY . /app/

# Install Python dependencies (if needed)
COPY requirements.txt /app/ 
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"

# Run the script automatically when the container starts
CMD ["python", "main.py"]
