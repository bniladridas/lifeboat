FROM jupyter/datascience-notebook:latest
WORKDIR /home/user
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY notebooks/ ./notebooks/
COPY src/ ./src/
USER user
CMD ["python", "notebooks/run.py"]
