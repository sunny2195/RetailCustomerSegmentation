import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from src.utils.common import logger, save_dill
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        logger.info(f"Data Transformation component initialized with config.")

    def run_transformation(self):
        logger.info("--- Starting Data Transformation ---")
        
        try:
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded raw data from {self.config.data_path}. Shape: {df.shape}")

            logger.info("Starting 'The Great Cleanse'...")
            df_clean = df.dropna(subset=['CustomerID'])

            df_clean = df_clean[df_clean['Quantity'] > 0]
            junk_codes_list = ['POST', 'D', 'M', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'B', 'S']

            df_clean = df_clean[~df_clean['StockCode'].str.upper().isin(junk_codes_list)]
            df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
            df_clean['total_price'] = df_clean['Quantity'] * df_clean['UnitPrice']
            
            logger.info(f"Cleaning complete. New shape: {df_clean.shape}")

            logger.info("Starting RFM Feature Engineering...")
            snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

            recency_df = df_clean.groupby('CustomerID').agg(
                Last_Purchase_Date=('InvoiceDate', 'max')
            ).reset_index()
            recency_df['Recency'] = (snapshot_date - recency_df['Last_Purchase_Date']).dt.days
            
            frequency_df = df_clean.groupby('CustomerID').agg(
                Frequency=('InvoiceNo', 'nunique')
            ).reset_index()
            
            monetary_df = df_clean.groupby('CustomerID').agg(
                Monetary=('total_price', 'sum')
            ).reset_index()
            
            rfm_df = recency_df[['CustomerID', 'Recency']].merge(
                frequency_df[['CustomerID', 'Frequency']], on='CustomerID'
            )
            rfm_df = rfm_df.merge(
                monetary_df[['CustomerID', 'Monetary']], on='CustomerID'
            )
            
            logger.info(f"RFM table created successfully. Shape: {rfm_df.shape}")
            logger.info("Starting log-transform and scaling...")
            
            rfm_features = rfm_df.drop('CustomerID', axis=1)
            rfm_log = np.log1p(rfm_features)
            

            scaler = StandardScaler()
            scaler.fit(rfm_log)
            
            
            rfm_scaled_data = scaler.transform(rfm_log)
            rfm_scaled_df = pd.DataFrame(rfm_scaled_data, columns=rfm_features.columns)
            rfm_scaled_df['CustomerID'] = rfm_df['CustomerID'].values
            
            logger.info("Log-transform and scaling complete.")

            
            rfm_scaled_df.to_csv(self.config.transformed_data_path, index=False)
            logger.info(f"Transformed data saved to: {self.config.transformed_data_path}")
            
            
            save_dill(data=scaler, path=self.config.scaler_path)
            logger.info(f"Scaler object saved to: {self.config.scaler_path}")

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise e