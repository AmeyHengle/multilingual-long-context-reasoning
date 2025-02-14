#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pickle
import os
from tqdm import tqdm

device = 'cuda:3'

MODEL_DIR = '/home/models'
MODEL_NAME = 'Meta-Llama-3-8B-Instruct'

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR + '/' + MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR + '/' + MODEL_NAME)


# In[19]:


ROOT_DIR = '/home/prasoon/multilingual-long-reasoning'
for folder in ['prompts_0k', 'prompts_2k']:
    for file in os.listdir(ROOT_DIR + '/' + folder):
        print(file)
        df = pd.read_csv(ROOT_DIR + '/' + folder + '/' + file)
        target_columns = []
        new_columns = {}
        for column in list(df.columns):
            if(column.startswith('rag')):
                target_columns.append(column)
                new_columns[column + '_inference'] = []
        for target_column in tqdm(target_columns):
            for index, row in tqdm(df.iterrows()):
                prompt = row[target_column]
                input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
                output = model.generate(input_ids, do_sample=False, return_dict_in_generate=True, max_new_tokens = 64, pad_token_id = 128001)
                new_columns[target_column + '_inference'].append(tokenizer.decode(output.sequences[0]))
            
        for new_column in list(new_columns.keys()):
            df[new_column] = new_columns[new_column]
        df.to_csv(ROOT_DIR + '/' + 'results' + '/' + folder + '/' + file, index=False)

