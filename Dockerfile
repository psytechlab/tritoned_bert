FROM nvcr.io/nvidia/tritonserver:22.12-py3
RUN pip install transformers
COPY model_repository /models
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
CMD ["tritonserver", "--model-repository=/models"]