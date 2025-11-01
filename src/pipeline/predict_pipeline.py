import os
import pandas as pd
import numpy as np
from src.utils.common import load_dill, logger
from pathlib import Path
from scipy.spatial.distance import cdist
class PredictionPipeline:
    """
    This class loads the trained model and scaler, and uses them
    to predict the cluster for new, incoming RFM data.
    """
    def __init__(self):
        # We hardcode the paths to the artifacts, which are relative to the root project directory.
        self.model_path = Path('artifacts/models/model.dill')
        self.scaler_path = Path('artifacts/models/scaler.dill')
        
        # Load the objects into memory *once* when the class is initialized.
        logger.info("Loading model and scaler objects for prediction...")
        try:
            model_artifacts = load_dill(self.model_path)
            self.model = model_artifacts['model']
            
            # Convert centroids dictionary back to a DataFrame for easy math
            self.centroids = pd.DataFrame(model_artifacts['centroids'])
            self.scaler = load_dill(self.scaler_path)
            logger.info("Model and scaler (with centroids) loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
            raise e

    def transform_input(self, rfm_data):
        """
        Applies the *exact* pre-processing (log + scale) to new data.
        """
        try:
            # 1. Log-transform (using np.log1p which is log(x+1))
            # We assume rfm_data contains the necessary numerical columns
            rfm_log = np.log1p(rfm_data)
            
            # 2. Scale using our *loaded* scaler
            rfm_scaled = self.scaler.transform(rfm_log)
            
            return rfm_scaled
            
        except Exception as e:
            logger.error(f"Error during data transformation for prediction: {e}")
            raise e

    def predict(self, rfm_data_df):
        """
        Predicts the cluster for new RFM data.
        """
        try:
            logger.info("Starting prediction...")
            
            # 1. Transform the new data
            scaled_data = self.transform_input(rfm_data_df)
            centroid_points = self.centroids.values
            # 2. Predict
            # The [0] gets the first (and only) prediction from the array
            distances = cdist(scaled_data, centroid_points)
            prediction = np.argmin(distances, axis=1)[0]
            
            logger.info(f"Prediction complete. Cluster: {prediction}")
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e