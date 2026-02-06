FROM jupyter/datascience-notebook:latest
WORKDIR /home/user
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER user
CMD ["python", "lifeboat.ipynb"]
