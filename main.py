import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ChestCancerDetectionAI import logger
from ChestCancerDetectionAI.pipeline.stage_01_data_ingestion import data_ingestion_traning_pipeline
from ChestCancerDetectionAI.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline  
from ChestCancerDetectionAI.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from ChestCancerDetectionAI.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Pipeline"

try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = data_ingestion_traning_pipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} Completed <<<<")

    
except Exception as e:
        logger.exception(e)
        raise e   


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Evaluation stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e