import pandas as pd
import glob
import json
import os
from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

files = glob.glob(os.path.join("comments", "*"))
df = pd.concat((pd.read_json(f) for f in files), ignore_index=True)
# model cannot handle > 2000 char
df['text'] = df['text'].str.slice(0,2000)
dataset = Dataset.from_pandas(df)

pipe = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-italian")
res = pipe(dataset["text"])
    
# not sure how to get the results back with the original input
# so here we're adding the results into the original df..
df['created_at'] = df['created_at'].astype(str)
join = [x | y for x, y in zip(df.to_dict('records'), res)]
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(join, f, ensure_ascii=False, indent=4)
