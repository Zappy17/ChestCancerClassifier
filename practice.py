import os 
from ChestCancerDetectionAI.constants import *
from ChestCancerDetectionAI.utils.common import read_yaml, create_directories
from ChestCancerDetectionAI.entity.config_entity import (DataIngestionConfig,
                                                         PrepareBaseModelConfig,
                                                         TrainingConfig,
                                                         EvaluationConfig )




class ConfigurationManager:
    def __init__(self, 
                config_filepath = CONFIG_FILE_PATH, 
                params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artificat_root])

