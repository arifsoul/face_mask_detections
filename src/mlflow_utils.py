import mlflow
import os
from ultralytics.utils import LOGGER


class MLflowYOLOCallback:
    """
    Custom Callback for integrating MLflow with YOLO training.
    """

    def __init__(self, run_name=None, tracking_uri=None, experiment_name=None):
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def on_pretrain_routine_start(self, trainer):
        """Called before training routine starts."""
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)

            if not mlflow.active_run():
                mlflow.start_run(run_name=self.run_name)

            # Log Parameters
            if hasattr(trainer, "args"):
                params = vars(trainer.args)
                # Filter params to avoid logging overly large objects if any
                loggable_params = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                mlflow.log_params(loggable_params)

        except Exception as e:
            LOGGER.warning(f"MLflowCallback: Error accessing MLflow: {e}")

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        try:
            if mlflow.active_run():
                # Log Metrics
                # trainer.metrics is a dict, keys might need cleaning
                if hasattr(trainer, "metrics"):
                    mlflow.log_metrics(trainer.metrics, step=trainer.epoch)

                # Log Loss values if available separately
                if hasattr(trainer, "loss_items"):
                    # Mapping depends on YOLO version, usually box, cls, dfl
                    losses = {
                        f"loss/{k}": v
                        for k, v in zip(trainer.loss_names, trainer.loss_items)
                    }
                    mlflow.log_metrics(losses, step=trainer.epoch)

        except Exception as e:
            LOGGER.warning(f"MLflowCallback: Error logging metrics: {e}")

    def on_train_end(self, trainer):
        """Called at the end of training."""
        try:
            if mlflow.active_run():
                # Log Best Model
                if hasattr(trainer, "best") and os.path.exists(trainer.best):
                    mlflow.log_artifact(str(trainer.best), artifact_path="weights")

                # Log Last Model
                if hasattr(trainer, "last") and os.path.exists(trainer.last):
                    mlflow.log_artifact(str(trainer.last), artifact_path="weights")

                # Log all other artifacts in the save_dir (plots, csv, etc.)
                if hasattr(trainer, "save_dir") and os.path.exists(trainer.save_dir):
                    for root, dirs, files in os.walk(trainer.save_dir):
                        for file in files:
                            # Skip weights as they are already logged or large
                            if file.endswith(".pt"):
                                continue

                            file_path = os.path.join(root, file)
                            # Calculate relative path for artifact structure
                            rel_path = os.path.relpath(file_path, trainer.save_dir)
                            # Log artifact preserving structure if needed, or just flat
                            # Putting them in 'training_results' folder in MLflow to keep it clean
                            mlflow.log_artifact(
                                file_path,
                                artifact_path=os.path.join(
                                    "training_results", os.path.dirname(rel_path)
                                ),
                            )

                # End Run?
                # If we are in a script, yes.
                # If in a notebook where user might want to log more, maybe.
                # But typically training is a discrete run.
                mlflow.end_run()

        except Exception as e:
            LOGGER.warning(f"MLflowCallback: Error ending run: {e}")
