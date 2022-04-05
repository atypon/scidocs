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

    model = ONNXModel(path_to_onnx=conf['model']['path'], 
                      tokenizer_pretrained_model=conf['tokenizer']['pretrained_model'],
                      tokenizer_max_length=conf['tokenizer']['max_length'],
                      inputs=conf['model']['inputs'])
    
    embed_scidocs(model=model,
                 model_name=conf['model']['name'],
                 data_dir=conf['data_dir'])