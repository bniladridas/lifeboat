FROM databricks/runtime-notebooks:14.1 LTS
WORKDIR /home/user
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY notebooks/ ./notebooks/
COPY src/ ./src/
CMD ["python", "notebooks/run.py"]
