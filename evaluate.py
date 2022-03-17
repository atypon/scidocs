import argparse, yaml
from utils import model_evaluation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/conf.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    model_evaluation(embeddings_path=conf['embeddings_path'],
                     results_file=conf['results_file'])
