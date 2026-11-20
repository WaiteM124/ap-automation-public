FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY main.py .
COPY ap_automation/ ap_automation/
COPY data/invoices.csv data/vendor_map.csv data/
COPY onboarding/ onboarding/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
