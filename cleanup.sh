unset NUM_CLASSES
unset MODEL_NAME
unset MAX_BATCH_SIZE

rm -rf model_repository/**/*.pbtxt
rm -rf model_repository/model_onnx/1/model.onnx
rm -rf model_repository/post_processing/1/id2label.json
rm -rf model_repository/text_preprocessing/1/tokenizer/*
rm -rf temp_model
if [ ! -z "$1" ];
then
    rm -rf model_repository/"$1"
fi
git checkout ./model_repository