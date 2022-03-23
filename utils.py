import json, logging, onnxruntime, os

from time import perf_counter
from torch.utils.data import Dataset
from os.path import join
from tqdm import tqdm
from process_dataset import load_data
from transformers import AutoTokenizer

logger = logging.getLogger("scidocs")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

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

    def average_lap_time(self):
        return self.duration / self.laps

    def total_time(self):
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

class ONNXModel():
    """Class that implements model using ONNX runtime"""

    def __init__(self, path_to_onnx, tokenizer_pretrained_model, tokenizer_max_length):
        """Initiliaze model"""
        self.model = onnxruntime.InferenceSession(path_to_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_model)
    
    def forward(self, text):
        """Implement models forward method"""
        tokens = self.tokenizer(text,
                    add_special_tokens=True,
                    max_length=self.tokenizer_max_length,
                    return_token_type_ids=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True)

        inputs_ids = tokens['input_ids']
        token_type_ids = tokens['token_type_ids']
        attention_mask = tokens['attention_mask']
        # Calculating embeddings
        ort_inputs = {
            'input_ids': inputs_ids.detach().cpu().numpy(),
            'attention_mask': attention_mask.detach().cpu().numpy(),
            'token_type_ids': token_type_ids.detach().cpu().numpy()
            }
        return self.model.run(None, ort_inputs)[0][:,0,:].squeeze(0).tolist()

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
    with open(results_file, 'w') as f:
        f.write(str(scidocs_metrics))

    return scidocs_metrics

def embed_scidocs(model, model_name, data_dir):
    """
    Embed dataset regarding scidocs framework
    
    :param model: Model that implements forward method and infers vectors
    :param model_name: Name of the model
    :param data_dir: Directory of scidocs dataset
    """
    embeddings_dir = join(data_dir, model_name)
    dataset_dir = data_dir

    if not os.path.isdir(embeddings_dir):
        os.mkdir(embeddings_dir)

    datasets = ['paper_metadata_mag_mesh.json', 'paper_metadata_recomm.json', 'paper_metadata_view_cite_read.json']
    emb_files = ['cls.jsonl', 'recomm.jsonl', 'user-citation.jsonl']

    for dataset, emb_file in zip(datasets, emb_files):
        logger.info(f'Embedding scidocs {dataset}...')
        dataset_path = join(dataset_dir, dataset)
        embeddings_path = join(embeddings_dir, emb_file)
        with open(dataset_path) as f:
            data = json.load(f)
        with open(embeddings_path, 'w') as f:
            for doc_id in tqdm(data.keys(), total=len(data)):
                title = data[doc_id]['title']
                abstract = data[doc_id]['abstract']
                model_input = str(title) + '[SEP]' + str(abstract)
                model_output = model.forward(model_input)
                entry = json.dumps({'paper_id' : doc_id, 'embedding' : model_output})
                f.write(entry + '\n')