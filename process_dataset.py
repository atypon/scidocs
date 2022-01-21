import os
import json 


MAG_MESH_DATA = 'data/paper_metadata_mag_mesh.json'
RECOMM_DATA = 'data/paper_metadata_recomm.json'
VIEW_CITE_READ_DATA = 'data/paper_metadata_view_cite_read.json'

def load_data(path):
    with open(path) as f:
        data =  json.load(f)
    return data

def json_to_jsonl(data, path):
    entries = len(data.keys())
    with open(path, 'w') as f:
        for i, id in enumerate(data.keys()):
            entry = {'paper_id' : id}
            entry.update(data[id])
            entry = json.dumps(entry)
            if i < entries-1:
                f.write(entry + '\n')
            else:
                f.write(entry)
            


if __name__ == "__main__":

    if not os.path.isdir('processed_datasets'):
        os.mkdir('processed_datasets')

    
    data = load_data(MAG_MESH_DATA)
    json_to_jsonl(data, 'processed_datasets/paper_metadata_mag_mesh.jsonl')

    data = load_data(RECOMM_DATA)
    json_to_jsonl(data, 'processed_datasets/paper_metadata_recomm.jsonl')
    
    data = load_data(VIEW_CITE_READ_DATA)
    json_to_jsonl(data, 'processed_datasets/paper_metadata_view_cite_read.jsonl')
