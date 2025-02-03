import yaml
import os
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    """
    Loads configuration settings from config.yaml and environment variables.
    """

    def __init__(self, config_path="examples/config.yaml"):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get_llm_settings(self):
        """Get LLM settings from config."""
        if not self.config or "llm" not in self.config:
            return {
                "model": "qwen-uraf",
                "api_url": "http://localhost:11434/api/generate",
                "max_tokens": 4000,
                "temperature": 0.5,
                "top_p": 0.85,
                "top_k": 50,
                "min_p": 0.2,
                "presence_penalty": 1.0
            }

        llm_settings = self.config["llm"]
        logger.info(f"Loaded LLM settings: {llm_settings}")
        return llm_settings

    def get_evaluation_thresholds(self):
        """Returns evaluation readiness thresholds."""
        if "evaluation" in self.config and "readiness_thresholds" in self.config["evaluation"]:
            return self.config["evaluation"]["readiness_thresholds"]
        else:
            logger.error("Missing 'evaluation: readiness_thresholds' in config file.")
            raise KeyError("Missing 'evaluation: readiness_thresholds' in config file.")
