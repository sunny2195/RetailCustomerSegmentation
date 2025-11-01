import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  
from src.utils.common import logger, load_dill, save_json
from src.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        logger.info(f"Model Evaluation component initialized.")

    def _generate_silhouette_plot(self, data, cluster_labels, n_clusters, silhouette_avg):
        logger.info(f"Generating Silhouette Plot for {n_clusters} clusters...")
        try:
            
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(10, 7)

            ax1.set_xlim([-0.2, 0.7])
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
                
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10  

            ax1.set_title("Silhouette Plot for the various clusters")
            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.7])

            
            plt.savefig(self.config.silhouette_plot_path)
            logger.info(f"Silhouette plot saved to: {self.config.silhouette_plot_path}")
            plt.close() 
        except Exception as e:
            logger.error(f"Error generating silhouette plot: {e}")
            raise e

    def evaluate_model(self):
        """
        Main method to run the evaluation.
        """
        logger.info("--- Starting Model Evaluation ---")
        try:
            
            data = pd.read_csv(self.config.data_path)
            model = load_dill(path=self.config.model_path)
            
            
            
            features = data.drop('CustomerID', axis=1)
            
            
            logger.info("Predicting cluster labels on the data...")
            cluster_labels = model.labels_
            
            
            logger.info("Calculating Silhouette Score...")
            score = silhouette_score(features, cluster_labels)
            logger.info(f"Calculated Silhouette Score: {score:.4f}")
            
            
            metrics = {"silhouette_score": score}
            save_json(path=self.config.metrics_file_path, data=metrics)
            logger.info(f"Metrics saved to: {self.config.metrics_file_path}")
            
            
            self._generate_silhouette_plot(
                data=features,
                cluster_labels=cluster_labels,
                n_clusters=model.n_clusters,
                silhouette_avg=score
            )
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise e