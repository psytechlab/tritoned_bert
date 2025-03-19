set -e

MODEL_PATH=$1
TOKENIZER_PATH=$2
ID2LABEL_PATH=$3

if [ ! -z "$4" ]; 
then 
    MODEL_NAME=$4 
else 
    MODEL_NAME="ensemble_model"
fi

if [ ! -z "$5" ]; 
then 
    CONTAINER_TAG=$5
else 
    CONTAINER_TAG="latest"
fi

if [ ! -z "$6" ]; 
then 
    MAX_BATCH_SIZE=$6
else 
    MAX_BATCH_SIZE="4" 
fi

ONNX_MODEL="temp_model"

NUM_CLASSES=$(python utils/count_classes.py $ID2LABEL_PATH)

echo "Converting model to onnx"
python utils/convert_to_onnx.py -m $MODEL_PATH -t $TOKENIZER_PATH -s $ONNX_MODEL

echo "Processing templates"
export NUM_CLASSES=$NUM_CLASSES
export MODEL_NAME=$MODEL_NAME
export MAX_BATCH_SIZE=$MAX_BATCH_SIZE
bash process_template.sh
rm -rf model_repository/**/*.template

echo "Moving files"
# moving model
mv "$ONNX_MODEL/model.onnx" "model_repository/model_onnx/1"
# moving tokenizer files
mv "$ONNX_MODEL"/*.json "model_repository/text_preprocessing/1/tokenizer"
mv "$ONNX_MODEL"/vocab.txt "model_repository/text_preprocessing/1/tokenizer"
# moving id2label
cp $ID2LABEL_PATH "model_repository/post_processing/1/id2label.json"
if [ ! "$MODEL_NAME" = "ensemble_model" ];
then
    mv model_repository/ensemble_model model_repository/"$MODEL_NAME"
fi

echo "Building docker images"
docker build -t tritoned_$MODEL_NAME:$CONTAINER_TAG . --no-cache

# cleanup
echo "Cleaning up"
bash cleanup.sh "$MODEL_NAME"

echo "Docker image tritoned_$MODEL_NAME:$CONTAINER_TAG successfully created"