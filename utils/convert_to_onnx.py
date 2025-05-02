"""Convert trained AutoModelForSequenceClassification model from transformers into ONNX format"""
import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from pathlib import Path
from shutil import copytree, rmtree

def download_model_from_clearml(model_id: str, target_path: str | Path | None = None, override: bool = False):
    from clearml import InputModel
    if isinstance(target_path, str):
        target_path = Path(target_path)
    model_path = InputModel(model_id).get_local_copy(extract_archive=True)
    if target_path is None:
        return model_path
    if target_path.exists() and override:
        rmtree(target_path)
        copytree(model_path, target_path)
    elif not target_path.exists():
        copytree(model_path, target_path)
    else:
        raise ValueError(f"Path {target_path} exists.")
    return target_path

def get_artifacts_from_storage(path: str, storage_type: str):
    """Resolve path to get actual path to the artifact.
    
    This is a interface switcher to any kind of storages where
    the models and tokenizers could be stored. This func takes
    "abstact" path to the artifact and type of the storage as input
    and route it to the function that can conver "path" of the specific
    storage to the actual path.
    
    Currently implemented:
        * local - this is just a normal path to the artifact in the system
        * hf - model path in Hugging face Hub
        * clearml - id of the "Model" instance in the ClearML system
    
    Args:
        path (str): abstact path in the specific storage
        storage_type (str): type of the storage
    
    Returns:
        (str): actual path in the system
    """
    if storage_type == "local" or storage_type == "hf":
        return path
    elif storage_type == "clearml":
        return download_model_from_clearml(path, )
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