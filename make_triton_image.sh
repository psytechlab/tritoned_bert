MODEL_PATH=$1
TOKENIZER_PATH=$2
ID2LABEL_PATH=$3
NUM_CLASSES=$4
MODEL_NAME=$5
ONNX_MODEL="temp_model"

echo $MODEL_PATH
echo "Converting model to onnx"
python utils/convert_to_onnx.py -m $MODEL_PATH -t $TOKENIZER_PATH -s $ONNX_MODEL

echo "Processing templates"
export NUM_CLASSES=$NUM_CLASSES
export MODEL_NAME=$MODEL_NAME
bash process_template.sh
rm -rf model_repository/**/*.template

echo "Moving files"
# moving model
mv "$ONNX_MODEL/model.onnx" "model_repository/model_onnx/1"
# moving tokenizer files
mv "$ONNX_MODEL"/*.json "model_repository/text_preprocessing/1/tokenizer"
mv "$ONNX_MODEL"/vocab.txt "model_repository/text_preprocessing/1/tokenizer"
# moving id2label
cp $ID2LABEL_PATH "model_repository/post_processing/1"
mv model_repository/ensemble_model model_repository/"$MODEL_NAME"

echo "Building docker images"
docker build -t triton_test:v2 . --no-cache

# cleanup
bash cleanup.sh