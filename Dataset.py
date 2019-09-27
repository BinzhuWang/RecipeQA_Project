import json
import numpy as np
import tqdm
from allennlp.modules.elmo import Elmo,batch_to_ids
from torch.utils.data import Dataset, DataLoader

import torch
from gensim.models import KeyedVectors
import re
import numpy as np
import json
from Elmo import elmo_embedding


train_dataset = open('/Users/wangbinzhu/Desktop/Final/Recipe_Qa_dataset/recipeQA/textCloze_train.json', 'r',
                  encoding='utf8').read()
train_dict = json.loads(train_dataset)  # train_dict is dict
train_data = train_dict['data']

vali_dataset = open('/Users/wangbinzhu/Desktop/Final/Recipe_Qa_dataset/recipeQA/textCloze_vali.json', 'r',
                  encoding='utf8').read()
vali_dict = json.loads(vali_dataset)  # train_dict is dict
vali_data = vali_dict['data']


class RecipeQA(Dataset):
    def __init__(self, train_data, vali_data,test_data=None):
        self.train_data = train_data
        self.SEP = '[SEP]'
        self.is_val = test_data
        self.vali_data = vali_data
        # self.test_data = test_data

    def __getitem__(self, index):
        if not self.is_val:
            rec_dict = self.train_data[index]
            context = rec_dict['context']
            body = [step['body'] for step in context]
            return self.SEP.join(body)
        else:
            rec_dict = self.vali_data[index]
            context = rec_dict['context']
            body = [step['body'] for step in context]
            return self.SEP.join(body)

    def __len__(self):
        if not self.is_val:
            return len(train_data)
        else:
            return len(vali_data)


recipe_test = RecipeQA(train_data,vali_data,None)
tmp_loader = DataLoader(recipe_test,batch_size=3,num_workers=1,shuffle=False)

for i , sample in enumerate(tmp_loader):
    a = sample
    for j in range(len(a)):
        a[j] = a[j].replace('[SEP]', " ").split()
        # print((len(a[0])))
    break
print(a[0])
    # tmp,mask = elmo_embedding(a)
    # print(tmp.size())
    # print(mask)