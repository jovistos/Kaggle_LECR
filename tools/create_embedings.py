import pandas as pd
import numpy as np

import re

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

import pickle


SEED = 42
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/home/jovis/Documents/WORK/Kaggle/LECR/data/learning-equality-curriculum-recommendations/"
model_checkpoint = "google/flan-t5-large"
SAVE_PATH = "content_id_vector_v3_FlanEncoder_large_128.csv"

def clean_text(text):
        
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\|", " ", text)

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text

def prepare_input(tokenizer, text):
    inputs = tokenizer(text,
                           add_special_tokens=True,
                           max_length=MAX_LEN,
                           padding="max_length",
                           truncation=True,
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, df):
        self.inputs = df['source'].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        #label = self.label[item]
        outputs = prepare_input(tokenizer, inputs)
        #outputs['label'] = torch.tensor(label, dtype=torch.float32)
        return outputs

def get_embedding(model, dataloader):
    model.eval().to(DEVICE)
    outputs = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = {}
            for k, v in data.items():
                inputs[k] = v.to(DEVICE)     
            output = model.encoder(**inputs)
            outputs.append(output['last_hidden_state'][:,0,:].to('cpu').numpy())
    outputs = np.concatenate(outputs, axis=0)
    return outputs

def pickle_dump(obj_, filename):
    with open(f'{filename}', 'wb') as f:
        pickle.dump(obj_, f)
        
def pickle_load(path_):
    with open(f'{path_}', 'rb') as f:
        obj_ = pickle.load(f)
    return obj_


if __name__ == "__main__":

    content = pd.read_csv(DATA_PATH + "content.csv")
    content = content.replace(np.nan, 'NaN')
    df_2_ready = pd.DataFrame()
    df_2_ready["source"]= "%B" + " %7 " + content.kind + " %8 "+content.language + " %9 "+ content.title+ " %10 "+content.description+ " %11 "+ content.text 
    df_2_ready["source"] = df_2_ready["source"].apply(lambda x: clean_text(x))
    df_2_ready["target"] = content["id"] 

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    content_dataset = TrainDataset(df_2_ready)

    content_loader = DataLoader(content_dataset,
                            batch_size=256,
                            shuffle=False,
                            drop_last=False,
                            num_workers=32)
    
    contents_emb = get_embedding(model, content_loader)
    contents_emb_list = contents_emb.tolist()
    df_2_ready["vector"]=contents_emb_list
    df_2_ready["vector"] = df_2_ready["vector"].apply(lambda row: "|".join([str(x) for x in row]))
    df_2_ready = df_2_ready.drop_duplicates(subset='vector', keep="first").reset_index(drop=True)

    df_2_ready.to_csv(SAVE_PATH,index=False)