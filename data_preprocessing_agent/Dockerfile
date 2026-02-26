FROM python:3.11-slim

WORKDIR /workspace

# Install data science libraries
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.26.2 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0

# Create directories for data
RUN mkdir -p /workspace/data /workspace/output

# Run as non-root user for security
RUN useradd -m -u 1000 sandbox && \
    chown -R sandbox:sandbox /workspace

USER sandbox

# Default command
CMD ["python3"]
