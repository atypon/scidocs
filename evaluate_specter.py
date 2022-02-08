from process_dataset import load_data
from utils import Timer, DocumentDataset, log_results, model_evaluation

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import os
import json

# path to model embeddings folder
PATH = 'data/specter-mpnet-embeddings'
BATCH_SIZE = 4

if not os.path.isdir(PATH):
	os.mkdir(PATH)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('cuda')

ft_model = torch.load('mpnet-specter.ckpt', map_location='cuda:0')
renamed_state_dict = {}
for key in ft_model['state_dict']:
	rename = key.replace('model.', '')
	renamed_state_dict[rename] = ft_model['state_dict'][key]
model.load_state_dict(renamed_state_dict)


def embed_dataset(dataset_path, results_file):
	# Function responsible for creating the embeddings for file
	# containing documents descibed by dataset parameter

	timer1 = Timer()
	timer2 = Timer()

	print(f'Loading sequencies from {dataset_path}...')

	dataset = DocumentDataset(dataset_path, for_model='specter')
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

	# Due to out of memory errors, tokenization and embedding
	# calculation will be performed not in a batched way but
	# one by one. This method saves RAM and the overall process
	# is finished faster because for each document only the necessary
	# calculations are performed

	with open(os.path.join(PATH, results_file), 'w') as f:
		for ids, titles, sequences in tqdm(loader):
			sequences = list(sequences)
			# Tokenizing
			timer1.start()
			tokens = tokenizer(sequences, return_tensors="pt", padding=True,
			                   truncation=True, max_length=386).to('cuda')
			timer1.stop()
			# Calculating embeddings
			timer2.start()
			result = model(**tokens)
			pooled_output = result.last_hidden_state[:, 0, :]
			timer2.stop()

			log_results(f, ids, titles, pooled_output)

	print(f'Total inference time : {timer2.totalTime()} s')
	print(f'Mean tokenization time : {timer1.averageLapTime()} s')
	print(f'Mean inference time : {timer2.averageLapTime()} s')


with torch.inference_mode():
	embed_dataset('data/paper_metadata_mag_mesh.json', 'cls.jsonl')
	embed_dataset('data/paper_metadata_recomm.json', 'recomm.jsonl')
	embed_dataset('data/paper_metadata_view_cite_read.json', 'user-citation.jsonl')

model_evaluation(embeddings_path=PATH,
                 results_file='specter-mpnet-reproduced_scores.txt')