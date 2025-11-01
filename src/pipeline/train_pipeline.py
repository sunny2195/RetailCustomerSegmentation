from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.utils.common import logger

class TrainPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        logger.info("Training Pipeline initialized.")

    def run_data_ingestion(self):
        logger.info("--- Starting Data Ingestion stage ---")
        ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=ingestion_config)
        data_ingestion.ingest_data()
        logger.info("--- Completed Data Ingestion stage ---")

    def run(self):
        logger.info(">>> Starting entire training pipeline <<<")
        self.run_data_ingestion()
        logger.info(">>> Completed entire training pipeline <<<")
