FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py hotel_bookings_cleaned_enhanced.csv ./
RUN useradd --create-home appuser && mkdir -p artifacts && chown -R appuser:appuser /app
USER appuser
RUN python -m hotel_cancellation.train --output artifacts
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
