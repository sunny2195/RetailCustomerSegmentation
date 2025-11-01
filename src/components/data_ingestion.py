import os
import pandas as pd
from src.utils.common import logger # Our custom logger
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self):
        logger.info("Data Ingestion component: Starting data ingestion...")
        try:
            
            logger.info(f"Reading Excel file from: {self.config.source_path}")
            df = pd.read_excel(self.config.source_path)
            logger.info(f"Successfully read data from: {self.config.source_path}")
            os.makedirs(os.path.dirname(self.config.ingested_data_path), exist_ok=True)
            df.to_csv(self.config.ingested_data_path, index=False)
            logger.info(f"Data ingested and saved to: {self.config.ingested_data_path}")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e
            
