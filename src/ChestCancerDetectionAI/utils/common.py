import os 
import joblib
import yaml
import json 
from box.exceptions import BoxValueError
from typing import Any
from ChestCancerDetectionAI import logger 
from ensure import ensure_annotations 
import base64
from box import ConfigBox
from pathlib import Path 


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Args:
         Reads yaml file and returns
    
    Raises:
           Value Error if yaml file is empty 
           e: empty type 

    Returns:
            ConfigBox: returns ConfigBox type output

    """

    try:
        with open(path_to_yaml) as yaml_file :
            content = yaml.safe_load(yaml_file)
            logger.info(f"{path_to_yaml} : file loaded sucssefully")

    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e 
    



@ensure_annotations
def make_directory(path_to_directory : list ,  verbose = True ):
    """
    Args:
         path_to_directory (list) : list of path to directory 
        
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.

    """
    for path in path_to_directory:
        os.makedirs(path, exist_ok =True)  
        if verbose:
            logger.info(f"created directory at path: {path}")


@ensure_annotations
@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path:Path) -> str:
    """
    Args:
         path (Path): path of the file

    returns:
             str: size in kb 
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb}KB"

def decodeImage(imagestring , filename):
    imagedata = base64.b64decode(imagestring)
    with open(filename) as f:
        f.write(imagedata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())     

