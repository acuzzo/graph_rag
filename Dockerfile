FROM python:3.12-slim

# System deps for FAISS and others
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set work directory
WORKDIR /the_mapper

# Copy project files
COPY pyproject.toml uv.lock ./
# COPY . .

# Install uv
RUN pip install uv

# Install dependencies
RUN uv venv .venv
RUN uv sync

# Default command
CMD ["bash"]
