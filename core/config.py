import yaml
import argparse
from typing import List, Dict, Any
from pydantic import BaseModel, Field


  # Network architecture settings

class ModelConfig(BaseModel):
    input_dim: int = 784
    hidden_dims: List[int] = [512, 512]
    output_dim: int = 10

 
  # Dataset and training sequence settings

class TaskConfig(BaseModel):
    class_pairs: List[List[int]] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]




class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    method_kwargs: Dict[str, Any] = Field(default_factory=dict)  # Algorithm-specific hyperparameters


    @classmethod
    def load_from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)



def get_config(default_method_kwargs: Dict[str, Any] = None) -> ExperimentConfig:
    """
    Parses CLI for a --config argument.
    If provided, loads the YAML file.
    If not, falls back to the default_method_kwargs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # parse_known_args allows other scripts to have their own CLI args if needed
    args, _ = parser.parse_known_args()
    
    if args.config:
        return ExperimentConfig.load_from_yaml(args.config)
    
    return ExperimentConfig(method_kwargs=default_method_kwargs or {})

