from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
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

    def run_data_validation(self):
        logger.info("--- Starting Data Validation stage ---")
        validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(config=validation_config)
        data_validation.run_validation()
        logger.info("--- Completed Data Validation stage ---")

    def run_data_transformation(self):
        logger.info("--- Starting Data Transformation stage ---")
        try:
            transform_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=transform_config)
            data_transformation.run_transformation()
            logger.info("--- Completed Data Transformation stage ---")
        except Exception as e:
            logger.error(f"Data Transformation stage FAILED: {e}")
            raise e

    def run_model_trainer(self):
        logger.info("--- Starting Model Trainer stage ---")
        try:
            trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=trainer_config)
            model_trainer.train_model()
            logger.info("--- Completed Model Trainer stage ---")
        except Exception as e:
            logger.error(f"Model Trainer stage FAILED: {e}")
            raise e
        
    def run_model_evaluation(self):
        logger.info("--- Starting Model Evaluation stage ---")
        try:
            eval_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=eval_config)
            model_evaluation.evaluate_model()
            logger.info("--- Completed Model Evaluation stage ---")
        except Exception as e:
            logger.error(f"Model Evaluation stage FAILED: {e}")
            raise e


    def run(self):
        logger.info(">>> Starting entire training pipeline <<<")
        self.run_data_ingestion()
        self.run_data_validation()
        self.run_data_transformation()
        self.run_model_trainer()
        self.run_model_evaluation()
        logger.info(">>> Completed entire training pipeline <<<")
