import os
import json
import numpy as np

if __name__ == '__main__':
    emb_folder_path = os.path.join('data', 'specter-2hops-fp32')
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
        with open(os.path.join(emb_folder_path, emb_file), 'w') as open_emb_file:
            for record in embeddings:
                open_emb_file.write(json.dumps(record) + '\n')