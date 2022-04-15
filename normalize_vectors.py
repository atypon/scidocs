import argparse
import yaml
import os
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/conf_inference.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    emb_folder_path = conf['input_folder']
    output_folder = conf['output_folder']
    emb_folder = os.listdir(emb_folder_path)
    for emb_file in emb_folder:
        embeddings = []
        with open(os.path.join(emb_folder_path, emb_file), 'r') as open_emb_file:
            for line in open_emb_file:
                line = json.loads(line)
                emb = line['embedding'] / np.linalg.norm(line['embedding'])
                embeddings.append({
                    'paper_id': line['paper_id'],
                    'embedding': emb.tolist()
                })
        with open(os.path.join(output_folder, emb_file), 'w') as open_emb_file:
            for record in embeddings:
                open_emb_file.write(json.dumps(record) + '\n')