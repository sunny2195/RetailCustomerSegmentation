FROM python:3.10-slim AS builder

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY data/Online_Retail.xlsx data/Online_Retail.xlsx
RUN python main.py
FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir streamlit pandas numpy "dill==0.3.8"
COPY app.py .
COPY src/pipeline/predict_pipeline.py src/pipeline/predict_pipeline.py
COPY src/utils/common.py src/utils/common.py
COPY src/entity/config_entity.py src/entity/config_entity.py
COPY src/__init__.py src/__init__.py
COPY src/pipeline/__init__.py src/pipeline/__init__.py
COPY src/utils/__init__.py src/utils/__init__.py
COPY src/entity/__init__.py src/entity/__init__.py


COPY --from=builder /app/artifacts/ ./artifacts/


EXPOSE 8501


CMD ["streamlit", "run", "app.py"]