FROM astromis/tritonserver:24.05-onnx-python-cpu
RUN pip install transformers==4.51.3 numpy==1.26.4 --no-cache-dir
WORKDIR /models
COPY model_repository ./
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
CMD ["tritonserver", "--model-repository=./"]