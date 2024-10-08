from ChestCancerDetectionAI import logger
from ChestCancerDetectionAI.pipeline.stage_01_data_ingestion import data_ingestion_traning_pipeline


STAGE_NAME = "Data Ingestion Pipeline"

try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = data_ingestion_traning_pipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} Completed <<<<")

    
except Exception as e:
        logger.exception(e)
        raise e   