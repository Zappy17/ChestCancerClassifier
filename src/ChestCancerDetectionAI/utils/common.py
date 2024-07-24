import os
from box.exceptions import BoxValueError
import yaml
from ChestCancerDetectionAI import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path ) -> ConfigBox :
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
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logger.info(f"yaml file: {path_to_yaml} loaded sucssefully")
        return ConfigBox(content)
    
 except BoxValueError:
    raise ValueError("yaml file is empty")
 
 
 except Exception as e:
    raise e 
 
