import yaml, argparse
from utils import ONNXModel, embed_scidocs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/conf_inference.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    model = ONNXModel(conf['model']['path'], 
                      conf['tokenizer']['pretrained_model'],
                      conf['tokenizer']['max_length'])
    
    embed_scidocs(model,
                 conf['model']['name'],
                 conf['data_dir'])