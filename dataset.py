import enum
from torch.utils.data import  Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Dict, List, Tuple, Union
import json
from transformers import PreTrainedTokenizer
import numpy as np
import collections
from tqdm import tqdm
import re
import copy


FCInst = collections.namedtuple('FCInst', 'question answer fact entity')
FCFeat = collections.namedtuple('FCFeat', 'input_ids question_ids flag is_training')

def convert_instances_to_feature_tensors(instances: List[FCInst],
                                         tokenizer: PreTrainedTokenizer,
                                         max_seq_length: int,
                                         is_training: bool=False,
                                         sep_token_extra: bool = False,
                                         max_answer_length: int=20, model_num=None, task=None, tokenizer1=None) -> List[FCFeat]:
    features = []
    for idx, inst in enumerate(tqdm(instances)):
        target = inst.question
        fact = inst.fact
        fact = fact + '[SEP]' + inst.answer
        entity = inst.entity
        token_fact = tokenizer.tokenize(fact)
        token_entity = [tokenizer.tokenize(ner) for ner in entity]
        token_q = tokenizer.tokenize(target)
        mapmap = {}
        flag = [[0] * (len(token_fact) + 2)]
        for e in token_entity:
            idx = -1
            for i in range(len(token_fact)):
                idx = i
                flag1 = True
                for j in range(len(e)):
                    try:
                        if e[j] != token_fact[i+j]:
                            flag1=False
                            break
                    except:
                        flag1=False
                        break
                if flag1 == True:
                    for i in range(idx+1, idx+1+len(e)):
                        flag[0][i] = 1
                    break

        if is_training:    
            for i in range(len(token_q)):
                if i == 0:
                    continue
                tmp = copy.deepcopy(flag[i-1])
                for j in range(len(token_fact)):
                    if token_fact[j] == token_q[i-1] and tmp[j+1] == 1:
                        tmp[j+1] = 2
                flag.append(tmp)
        batch_input = tokenizer.encode(fact, add_special_tokens=True, truncation=True)
        target_ids = tokenizer.encode(target, add_special_tokens=True, truncation=True)
        features.append(FCFeat(input_ids=batch_input,
                        question_ids=target_ids,
                        flag = flag,
                        is_training=is_training))
    return features 

class FCDataset(Dataset):

    def __init__(self, file: Union[str, None], tokenizer: PreTrainedTokenizer,
                 pretrain_model_name: str,
                 number: int = -1,
                 max_question_len: int = 100,
                 max_answer_length: int = 30,
                 model_num=None,
                 tokenizer1=None,
                 mode='generation',
                 is_training=False) -> None:
        insts = []
        self.skip_num = 0
        self.tokenizer = tokenizer
        self.pretrain_model_name = pretrain_model_name
        print(f"[Data Info] Reading file: {file}")
        #print(file)
        with open(file, 'r', encoding='utf-8') as read_file:
            data = json.load(read_file)
        if number >= 0:
            data = data[:number]

        for sample in tqdm(data):
            insts.append(FCInst(question=str(sample['question']),
                                answer=str(sample['answer']),
                                fact = str(sample['fact']),
                                entity = sample['entity']))
            
        self._features = convert_instances_to_feature_tensors(instances=insts,
                                                              tokenizer=tokenizer,
                                                              max_seq_length=max_question_len,
                                                              sep_token_extra= "roberta" in pretrain_model_name or "bart" in pretrain_model_name or "checkpoint" in pretrain_model_name,
                                                              max_answer_length=max_answer_length, model_num=model_num, is_training=is_training)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> FCFeat:
        return self._features[idx]

    def collate_fn(self, batch: List[FCFeat]):
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        max_question_length = max([len(feature.question_ids) for feature in batch])
        #print(batch[0].entity, batch[0].label)
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            pad_label_length = max_question_length - len(feature.question_ids)
            flag = feature.flag
            flag = [item + [0] * (max_wordpiece_length - len(item)) for item in flag]
            if feature.is_training:
                flag = flag + [[0] * max_wordpiece_length] * (max_question_length - len(flag))

            #pad_label_length = max_label_length - len(feature.label)
            batch[i] = FCFeat(input_ids=np.asarray(feature.input_ids + [0] * padding_length),
                               question_ids=np.asarray(feature.question_ids + [0] * pad_label_length),
                               flag=np.asarray(flag),
                               is_training=np.asarray(feature.is_training))
        #print(batch)
        results = FCFeat(*(default_collate(samples) for samples in zip(*batch)))
        return results

'''
def convert_instances_to_feature_tensors_entity(instances: List[FCInst],
                                         tokenizer: PreTrainedTokenizer,
                                         max_seq_length: int,
                                         sep_token_extra: bool = False,
                                         max_answer_length: int=20, model_num=None, task=None, tokenizer1=None) -> List[FCFeat]:
    features = []
    for idx, inst in enumerate(tqdm(instances)):
        target = inst.question
        fact = inst.supporting_facts
        fact_sents = []
        q_ners = inst.q_ners
        f_ners = inst.f_ners
        context = inst.context
        id = inst.id
        doc_dict = {}
        q_entity = []
        for item in context:
            doc_dict[item[0]] = item[1]
        for item in fact:
            try:
                fact_sents.append(doc_dict[item[0]][item[1]])
            except:
                continue

        #batch_input = tokenizer.encode('[unused3]'.join(fact_sents) + '[SEP]' + inst.answer, add_special_tokens=True, truncation=True)
        target_ids = tokenizer.encode(target, add_special_tokens=True, truncation=True)

        dd = {}
        for entity in f_ners:
            entity_ids = tokenizer.encode(entity[0], add_special_tokens=True, truncation=True)
            flag = False
            for tmp in q_ners:
                if entity[0] == tmp[0]:
                    flag = True
                    break
            if entity[0] in dd:
                    continue
            dd[entity[0]] = 1
            if flag == True:
                #print(entity, q_ners)
                batch_input = tokenizer.encode('[unused3]'.join(fact_sents) + '[SEP]' + inst.answer + '[SEP]' + entity[0], add_special_tokens=True, truncation=True)
                features.append(FCFeat(input_ids=batch_input,
                            entity=entity_ids,
                            question_ids=target_ids,
                            label=[1],
                            id=id))
            else:
                batch_input = tokenizer.encode('[unused3]'.join(fact_sents) + '[SEP]' + inst.answer + '[SEP]' + entity[0], add_special_tokens=True, truncation=True)
                features.append(FCFeat(input_ids=batch_input,
                            entity=entity_ids,
                            question_ids=target_ids,
                            label=[0],
                            id=id))
    return features 
'''                  