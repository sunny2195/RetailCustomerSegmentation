import os
import yaml
from box import ConfigBox  
from ensure import ensure_annotations 
from pathlib import Path
import dill                 
import json
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
logger = logging.getLogger(__name__)

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file at: {path_to_yaml}\n{e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_dill(data: any, path: Path):
    try:
        with open(path, "wb") as file:
            dill.dump(data, file)
        logger.info(f"Dill file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Error saving dill file at: {path}\n{e}")
        raise e
    
@ensure_annotations
def load_dill(path: Path) -> any:
    try:
        with open(path, "rb") as file:
            data = dill.load(file)
        logger.info(f"Dill file loaded successfully from: {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading dill file at: {path}\n{e}")
        raise e

@ensure_annotations
def save_json(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON file at: {path}\n{e}")
        raise e