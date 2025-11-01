import pandas as pd
from src.utils.common import logger
from src.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self) -> bool:
        validation_passed = True
        try:
            data = pd.read_csv(self.config.data_path)
            all_cols = list(data.columns)
            required_cols = self.config.required_columns

            logger.info("Validating column presence...")
            for col in required_cols:
                if col not in all_cols:
                    validation_passed = False
                    logger.warning(f"Validation FAILED: Missing column: {col}")
            
            if validation_passed:
                logger.info("Column presence validation PASSED.")
            
            return validation_passed

        except Exception as e:
            logger.error(f"Error during column validation: {e}")
            return False

    def validate_schemas(self) -> bool:
        validation_passed = True
        try:
            data = pd.read_csv(self.config.data_path)
            schemas = self.config.column_schemas
            
            logger.info("Validating column data types (schemas)...")
            for col, expected_dtype in schemas.items():
                actual_dtype = str(data[col].dtype)
                
                if actual_dtype != expected_dtype:
                    validation_passed = False
                    logger.warning(f"Validation FAILED for column '{col}': Expected type '{expected_dtype}', but got '{actual_dtype}'")
            
            if validation_passed:
                logger.info("Column schema validation PASSED.")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            return False

    def run_validation(self):
        logger.info("Data Validation component: Starting validation...")
        columns_ok = self.validate_columns()
        schemas_ok = self.validate_schemas()
        
        validation_status = (columns_ok and schemas_ok)
        try:
            with open(self.config.validation_status_file, 'w') as f:
                if validation_status:
                    f.write("Validation Status: PASS")
                    logger.info("Data validation successful. Status file written.")
                else:
                    f.write("Validation Status: FAIL")
                    logger.warning("Data validation FAILED. Status file written.")
                    
                    raise Exception("Data validation failed. Check logs for details.")
                
        except Exception as e:
            logger.error(f"Error writing validation status file or raising exception: {e}")
            
            if "Data validation failed" in str(e):
                raise e 
            else:
                raise Exception(f"Failed to write status file: {e}")


