FROM python:3.12-slim

LABEL maintainer="Prompt2ML"
LABEL description="Isolated ML execution sandbox for the Prompt2ML agent pipeline"

ENV MPLBACKEND=Agg
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages needed for scipy/lightgbm/catboost C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
        curl \
        ca-certificates \
    && apt-get clean && apt-get autoclean -y

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# PyTorch CPU-only (avoids the 2 GB CUDA wheel)
RUN pip install --no-cache-dir "torch>=2.2.0" \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -c "\
import nltk; \
[nltk.download(c, quiet=True) for c in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']]; \
print('NLTK corpora downloaded')"

# Non-root user (UID 1000 matches typical host UIDs for clean volume mounts)
RUN useradd -m -u 1000 -s /bin/bash sandbox && \
    mkdir -p /home/sandbox /workspace && \
    chown sandbox:sandbox /home/sandbox /workspace

WORKDIR /workspace
USER sandbox

# Smoke-test: fail the build if any critical import is broken
RUN python -c "import numpy, pandas, sklearn, scipy, matplotlib, seaborn, xgboost, lightgbm, catboost, statsmodels, imblearn, openpyxl, plotly, nltk, torch, joblib, optuna, shap, category_encoders; print('All OK')"

# Keep container alive when started detached by sandbox_executor.py
CMD ["tail", "-f", "/dev/null"]