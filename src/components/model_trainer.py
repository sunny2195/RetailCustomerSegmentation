import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans, Birch 
from src.utils.common import logger, save_dill
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        logger.info(f"Model Trainer component initialized.")

    def train_model(self):
        logger.info("--- Starting Model Training ---")
        try:
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded transformed data from: {self.config.data_path}")

            features_for_clustering = data.drop('CustomerID', axis=1)
            logger.info(f"Features for clustering prepared (dropped CustomerID). Shape: {features_for_clustering.shape}")

            logger.info(f"Initializing model: {self.config.model_name}")
            if self.config.model_name == "sc":
                model = SpectralClustering(
                    n_clusters=self.config.params['num_clusters'],
                    random_state=42 )
                
            elif self.config.model_name == "kmeans":
                model = KMeans(
                    n_clusters=self.config.params['num_clusters'],
                    init=self.config.params.get('init', 'k-means++'), 
                    random_state=42
                )
            
            elif self.config.model_name == "birch":
                model = Birch(
                    n_clusters=self.config.params['num_clusters']
                )

            else:
                raise ValueError(f"Unknown model name in config: {self.config.model_name}")
            
            logger.info(f"Training {self.config.model_name}...")
            model.fit(features_for_clustering)
            logger.info(f"Model training complete.")

            save_dill(data=model, path=self.config.model_path)
            logger.info(f"Trained model saved to: {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e


