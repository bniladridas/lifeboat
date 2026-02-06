FROM databricks/runtime-notebooks:14.1 LTS
WORKDIR /home/user
COPY notebooks/ ./notebooks/
COPY requirements.txt ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "notebooks/run.py"]
