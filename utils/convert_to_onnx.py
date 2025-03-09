import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from pathlib import Path
import clearml

def download_from_clearml(model_id: str):
    model_path = clearml.InputModel(model_id=model_id)
    path = model_path.get_local_copy(extract_archive=None)
    print(path)
    return path

def get_artifacts_from_storage(path, storage_type):
    if storage_type == "local" or storage_type == "hf":
        return path
    elif storage_type == "clearml":
        return download_from_clearml(path)
    else:
        raise ValueError("Unknown storage type")
    
def main(args):
    model_path = get_artifacts_from_storage(args.model_path, args.where_model)
    tokenizer_path = get_artifacts_from_storage(args.tokenizer_path, args.where_tokenizer)

    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    ort_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Done")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        "-m",
        type=str,
        required=True,
        help='The abstact path to the model'
    )
    parser.add_argument(
        '--tokenizer_path',
        "-t",
        type=str,
        required=True,
        help='The abstact path to the tokenizer'
    )
    parser.add_argument(
        '--save_path',
        "-s",
        type=str,
        required=True,
        help='The path to save the convert model'
    )
    parser.add_argument(
        '--where_model',
        type=str,
        default="local",
        help='The type of the model storage'
    )
    parser.add_argument(
        '--where_tokenizer',
        type=str,
        default="local",
        help='The type of the tokenizer storage'
    )
   
    args = parser.parse_args()
    main(args)