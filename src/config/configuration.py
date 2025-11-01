from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig)

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_path=Path(config.source_path), 
            ingested_data_path=Path(config.ingested_data_path)
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([Path(config.root_dir)])
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path), 
            validation_status_file=Path(config.validation_status_file), #
            required_columns=config.required_columns, 
            column_schemas=config.column_schemas 
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path), 
            transformed_data_path=Path(config.transformed_data_path), 
            scaler_path=Path(config.scaler_path) 
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.config.model_trainer.params
        create_directories([Path(config.root_dir)])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path), 
            model_path=Path(config.model_path), 
            model_name=config.model_name, 
            params=params 
        )
        return model_trainer_config

       
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            scaler_path=Path(config.scaler_path), 
            data_path=Path(config.data_path), 
            metrics_file_path=Path(config.metrics_file_path), 
            silhouette_plot_path=Path(config.silhouette_plot_path) 
        )
        return model_evaluation_config









