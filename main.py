from src.pipeline.train_pipeline import TrainPipeline
from src.utils.common import logger

try:
    logger.info(">>> Main: Starting training pipeline <<<")
    pipeline = TrainPipeline()
    pipeline.run()
    
    logger.info(">>> Main: Training pipeline completed successfully <<<")
except Exception as e:
    logger.error(f"Error encountered in main.py: {e}")
    raise e