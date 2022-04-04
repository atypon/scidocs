import argparse, yaml, mlflow

from utils import model_evaluation
from mlflow.tracking import MlflowClient

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/conf_evaluation.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    mlflow.set_tracking_uri(uri=conf['mlflow']['tracking_uri'])
    client = MlflowClient()
    # Get or create experiment
    experiment = client.get_experiment_by_name(name=conf['mlflow']['experiment_name'])
    if experiment is None:
        experiment_id = client.create_experiment(name=conf['mlflow']['experiment_name'])
    else:
        if dict(experiment)['lifecycle_stage'] == 'deleted':
            client.restore_experiment(dict(experiment)['experiment_id'])
        experiment_id = dict(experiment)['experiment_id']

    with mlflow.start_run(experiment_id=experiment_id, run_name=conf['mlflow']['run_name']):
        
        metrics = model_evaluation(embeddings_path=conf['embeddings_path'],
                        results_file=conf['results_file'])

        mlflow.log_metrics(metrics)