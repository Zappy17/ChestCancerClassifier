from ChestCancerDetectionAI.config.configuration import ConfigurationManager
from ChestCancerDetectionAI.components.data_ingestion import *
from ChestCancerDetectionAI import logger


STAGE_NAME = "Data Ingestion Pipeline"

class data_ingestion_traning_pipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == '__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = data_ingestion_traning_pipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} Completed <<<<")

    except Exception as e:
        logger.exception(e)
        raise e   

