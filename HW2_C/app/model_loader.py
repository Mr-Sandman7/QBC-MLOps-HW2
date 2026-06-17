from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """Load the HW02 model from MLflow.

    Required behavior:
    - Use MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD.
    - If MLFLOW_RUN_ID is set, load runs:/<run_id>/model.
    - Otherwise auto-select a clean/selected run from MLFLOW_EXPERIMENT_NAME.
    - Do not crash the API on startup. Store the error in self.state.error.
    """

    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        try:
            # Set MLflow credentials
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            
            run_id = config.MLFLOW_RUN_ID
            
            # If no run_id is set, auto-select a clean/selected run
            if not run_id:
                client = MlflowClient()
                experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
                
                if experiment is None:
                    raise ValueError(f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found")
                
                # Try to get run marked as selected_for_serving
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.selected_for_serving = 'true'",
                    max_results=1
                )
                
                # If no selected run, get best clean run by f1 score
                if not runs:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="tags.leakage_status = 'clean'",
                        order_by=["metrics.`f1 score` DESC"],
                        max_results=1
                    )
                
                if not runs:
                    raise ValueError(f"No clean runs found in experiment '{config.MLFLOW_EXPERIMENT_NAME}'")
                
                run_id = runs[0].info.run_id
            
            # Load model from MLflow
            model_uri = f"runs:/{run_id}/model"
            self.state.model = mlflow.sklearn.load_model(model_uri)
            
            # Load run metadata
            client = MlflowClient()
            run = client.get_run(run_id)
            
            self.state.model_uri = model_uri
            self.state.run_id = run_id
            self.state.run_name = run.data.tags.get("mlflow.runName", "")
            self.state.metrics = dict(run.data.metrics)
            self.state.params = dict(run.data.params)
            self.state.tags = dict(run.data.tags)
            self.state.loaded = True
            self.state.error = None
            
            print(f"Model loaded from MLflow!")
            print(f"   Run ID: {run_id}")
            print(f"   Run name: {self.state.run_name}")
            
        except Exception as e:
            self.state.loaded = False
            self.state.error = f"Failed to load model: {str(e)}"
            print(f"Error loading model: {e}")

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
