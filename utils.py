import json
import os

from time import perf_counter
from torch.utils.data import Dataset
from process_dataset import load_data

class Timer():

    # Timer class to calculate time 
    # intervals between function calls

    def __init__(self):
        self.duration = 0
        self.laps = 0
        self.t0 = 0

    def start(self):
        self.t0 = perf_counter()
        self.laps += 1
    
    def stop(self):
        t1 = perf_counter()
        self.duration += (t1 - self.t0)
        return t1 - self.t0
    
    def reset(self):
        self.duration = 0
        self.laps = 0

    def averageLapTime(self):
        return self.duration / self.laps

    def totalTime(self):
        return self.duration

class DocumentDataset(Dataset):

    def __init__(self, path, for_model = 'oag-bert-v1'):
        papers = load_data(path)
        self.ids = list(papers.keys())
        self.titles = [papers[id]['title'] for id in self.ids]

        if for_model == 'oag-bert-v1':
            self.sequence = [str(papers[id]['title']) + '.' + self.parse_abstract(papers[id]['abstract']) for id in self.ids]
        elif for_model == 'specter':
            self.sequence = [str(papers[id]['title']) + '[SEP]' + (self.parse_abstract(papers[id].get('abstract')) or '') for id in self.ids]
        else:
            raise('No such model')

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.titles[idx], self.sequence[idx]
    
    def parse_abstract(self, abstract):
        if abstract is None:
            return ''
        return ' ' + str(abstract)

class EnrichedDocumentDataset(Dataset):

    # Expects other fields like FOS affiliation etc for 
    # use with BERT-V2 models

    def __init__(self, path, for_model = 'oag-bert-v1'):
        papers = load_data(path)
        self.ids = list(papers.keys())
        self.titles = [papers[id]['title'] for id in self.ids]
        self.sequence = [self.parse_abstract(papers[id]['abstract']) for id in self.ids]

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.titles[idx], self.sequence[idx]
    
    def parse_abstract(self, abstract):
        if abstract is None:
            return ''
        return str(abstract)

def log_results(fp, ids, titles, embeddings):

    # Responsible for logging the incoming batch 
    # of ids, titles and embeddings to file
    # pointed by fp

    for id, title, embedding in zip(ids, titles, embeddings):
        entry = {"paper_id" : id, "title" : title, "embedding" : embedding.squeeze(0).tolist()}
        entry = json.dumps(entry)
        fp.write(entry + '\n')

def model_evaluation(embeddings_path, results_file):

    from scidocs import get_scidocs_metrics
    from scidocs.paths import DataPaths

    # Make dir to save results
    if not os.path.isdir('Results'):
        os.mkdir('Results')

    # point to the data, which should be in scidocs/data by default
    data_paths = DataPaths()

    # point to the included embeddings jsonl
    
    classification_embeddings_path = os.path.join(embeddings_path, 'cls.jsonl')
    user_activity_and_citations_embeddings_path = os.path.join(embeddings_path, 'user-citation.jsonl')
    recomm_embeddings_path = os.path.join(embeddings_path, 'recomm.jsonl')

    # now run the evaluation
    scidocs_metrics = get_scidocs_metrics(
        data_paths,
        classification_embeddings_path,
        user_activity_and_citations_embeddings_path,
        recomm_embeddings_path,
        val_or_test='test',  # set to 'val' if tuning hyperparams
        n_jobs=15,  # the classification tasks can be parallelized
        cuda_device=-1  # the recomm task can use a GPU if this is set to 0, 1, etc
    )

    print(scidocs_metrics)

    # Log scores to file
    with open(os.path.join('Results', results_file), 'w') as f:
        f.write(str(scidocs_metrics))

    return scidocs_metrics

