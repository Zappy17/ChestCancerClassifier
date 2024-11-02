import os
import mlflow
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import mlflow.keras
from ChestCancerDetectionAI.entity.config_entity import EvaluationConfig
from ChestCancerDetectionAI.utils.common import read_yaml, create_directories, save_json
import numpy as np
import random

# Set random seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Call set_seeds at the start of the script
set_seeds()

import dagshub
dagshub.init(repo_owner='Zappy17', repo_name='ChestCancerClassifier', mlflow=True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        This function establishes connection to MLflow (DagsHub) and logs the model & metrics.
        """
        # Set the MLflow tracking URI (using DagsHub)
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Zappy17/ChestCancerClassifier.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Zappy17"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "4a1d531e8bf77ea028d421de91152fa71273fe31"  # Replace with secure method

        # Set the tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Parse the tracking URL type (file store or remote)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Start MLflow run and log parameters, metrics, and model
        try:
            with mlflow.start_run() as run:
                print(f"Started MLflow run: {run.info.run_id}")

                # Log parameters
                mlflow.log_params(self.config.all_params)

                # Log evaluation metrics
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
                print("Logged metrics to MLflow")

                # Register or log the model in MLflow
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                    print("Model registered in MLflow")
                else:
                    mlflow.keras.log_model(self.model, "model")
                    print("Model logged to file system")

                print(f"Successfully completed MLflow run {run.info.run_id}")

        except Exception as e:
            print(f"Error during MLflow logging: {e}")
