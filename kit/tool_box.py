#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Reza

import torch
import torch.nn as nn
import torch.nn.functional as F
# import tensorflow as tf
import os
import re
import pickle
import bcolz
import numpy as np
import pynvml
import random
import csv
import logging as log

from torch.nn.init import xavier_normal_,xavier_uniform_,uniform_,constant_,calculate_gain
from torch.utils.data import Dataset, DataLoader

def seed_torch(seed=42,deter=False):
    '''
    `deter` means use deterministic algorithms for GPU training reproducibility, 
    if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    '''
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if deter:
        torch.set_deterministic(True)  # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
        # torch.use_deterministic_algorithms(deter)


def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def search_free_cuda():
    pynvml.nvmlInit()
    id = 2
    for i in range(4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.used == 0:
            id = i
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)


def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda(), True
    else:
        return module.cpu(), False


def cuda_avaliable():
    if torch.cuda.is_available():
        return True, torch.device("cuda")
    else:
        return False, torch.device("cpu")


def show_parameters(model: nn.Module, if_show_parameters=False):
    ''' show all named parameters
    i.e. "name:xxx ; size: 1000 "
    :param model: target model
    :param if_show_parameters: whether print param (tensor)
    :return: NULL
    '''
    for name, parameters in model.named_parameters():
        if parameters.requires_grad == False:
            continue
        print("name:{} ; size:{} ".format(name, parameters.shape))
        if if_show_parameters:
            print("parameters:", parameters)


def count_parameters(model: nn.Module, verbose:bool=False):
    ''' count all parameters '''
    param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    if not verbose:
        print("total model param num:",param_num)
        
    return param_num


def shuffle2list(a: list, b: list):
    # shuffle two list with same rule, you can also use sklearn.utils.shuffle package
    c = list(zip(a, b))
    random.shuffle(c)
    a[:], b[:] = zip(*c)
    return a, b


def gather(param, ids):
    # Take the line corresponding to IDS subscript from param and form a new tensor
    if param.is_cuda:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float().cuda()
    else:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float()
    ans = torch.mm(mask, param)
    return ans


def pairwise_distance(embeddings, squared=False):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    if pairwise_distances.is_cuda:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data]))[0])
    else:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))

    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def generate_logger(log_dir=".", log_name="LOG.log", level=1):
    ''' generate a python logger, write training information into terminal and a file
    :param: log_dir -> path to log file
    :param: log_name -> name of log file
    :param: level -> showing level, range from 1 to 5 with the importance rising
    :return: a logger class which can both write info to terminal and log_file
    '''
    log_levels = {1: log.DEBUG,
                  2: log.INFO,
                  3: log.WARNING,
                  4: log.ERROR,
                  5: log.CRITICAL}
    log_level = log_levels[level]

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    log_file = os.path.join(log_dir, log_name)
    file_handler = log.FileHandler(log_file)
    file_handler.setLevel(log_level)
    log.getLogger().addHandler(file_handler)

    return log


def FindAllSuffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path
    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result


def py_strip(p_str, d_str=' '):
    ''' use re to filter multiple str
    example:
        st1 = 'abc123abc'
        st2 = 'abc'
        res = '123'
    '''
    temp = re.search(r'[^(' + d_str + ')].*', p_str).group()
    # print(temp)
    res = re.search(r'.*[^(' + d_str + ')]', temp).group()
    # print(res)
    return res


def clean_tokenize(data, lower=False):
    ''' used to clean token, split all token with space and lower all tokens
    this function usually use in some language models which don't require strict pre-tokenization
    such as LSTM(with glove vector) or ELMO(already has tokenizer)
    :param data: string
    :return: list, contain all cleaned tokens from original input
    '''
    # split all tokens with a space
    data = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", data)
    data = re.sub(r"\'s", " \'s", data)
    data = re.sub(r"\'ve", " \'ve", data)
    data = re.sub(r"n\'t", " n\'t", data)
    data = re.sub(r"\'re", " \'re", data)
    data = re.sub(r"\'d", " \'d", data)
    data = re.sub(r"\'ll", " \'ll", data)
    data = re.sub(r",", " , ", data)
    data = re.sub(r"!", " ! ", data)
    data = re.sub(r"\(", " ( ", data)
    data = re.sub(r"\)", " ) ", data)
    data = re.sub(r"\?", " ? ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # del all redundant space, split all tokens, form a list
    return [x.strip() for x in re.split('(\W+)', data) if x is not None and x.strip()]


def find_sublist(lis: list, sublis: list, i=0, j=0):
    '''
    recursion, find the **end** index of the sub-list in the list
    if multiple sub-list, calculate the first one
    if not the sub-list, there will be a IndexError
    '''
    if lis[i] == sublis[j]:
        if j == len(sublis) - 1:
            return i
        else:
            return find_sublist(lis, sublis, i + 1, j + 1)
    else:
        return find_sublist(lis, sublis, i + 1, 0)

def word_embedding(embed_path:str,over_writte:bool,special_tk:bool=True,freeze:bool=False):
    ''' return a torch.nn.Embedding layer, utilizing the pre-trained word vector (e.g., Glove), add 'unk' and 'pad'.
    :param embed_path: the path where pre-trained matrix cached (e.g., './glove.6B.300d.txt').
    :param over_writte: force to rewritte the existing matrix.
    :param special_tk: whether adding special token -- 'unk' and 'pad', at position 1 and 0 by default.
    :param freeze: whether trainable.
    :return: embed -> nn.Embedding, weights_matrix -> np.array, word2idx -> dict, embed_dim -> int
    '''
    root_dir = embed_path.rsplit(".",1)[0]+".dat"
    out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
    out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
    if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
        ## process and cache glove ===========================================
        words = []
        idx = 0
        word2idx = {}    
        vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
        with open(os.path.join(embed_path),"rb") as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
        vectors = bcolz.carray(vectors[1:].reshape((idx, vect.shape[0])), rootdir=root_dir, mode='w')
        vectors.flush()
        pickle.dump(words, open(out_dir_word, 'wb'))
        pickle.dump(word2idx, open(out_dir_idx, 'wb'))
        print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
        ## =======================================================
    ## load glove
    vectors = bcolz.open(root_dir)[:]
    words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
    word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))
    print("Successfully load Golve from {}, the shape of cached matrix: {}".format(embed_path.rsplit("/",1)[0],vectors.shape))

    word_num, embed_dim = vectors.shape
    word_num += 2  if special_tk else 0  ## e.g., 400002
    weights_matrix = np.zeros((word_num, embed_dim)) 
    if special_tk:
        weights_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim, ))
        weights_matrix[2:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(weights_matrix)
        pad_idx,unk_idx = 0,1
        embed = torch.nn.Embedding(word_num, embed_dim,padding_idx=pad_idx)  
        embed.from_pretrained(weights_matrix_tensor,freeze=freeze,padding_idx=pad_idx)
        word2idx = dict([(k,v+2) for k,v in word2idx.items()])
        assert len(word2idx) + 2 == weights_matrix.shape[0]
    else:
        weights_matrix[:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(weights_matrix)
        embed = torch.nn.Embedding(word_num, embed_dim)  
        embed.from_pretrained(weights_matrix_tensor,freeze=freeze)
        assert len(word2idx) == weights_matrix.shape[0]

    return embed, weights_matrix, word2idx, embed_dim

def bert_find_max_len(attention:list):
    ''' for bert, encoding complete sentences in a mini-batch may cost a lot. So find the real max seq len in a batch. 
    :param: attention -> List[List[int]], attention mask matrix of all sentences in a mini-batch
    :return: real_max_len
    '''
    mask_padding_with_zero = True
    mask_pad_ids = 0 if mask_padding_with_zero else 1
    length = []
    for att in attention:
        length.append(att.index(mask_pad_ids))
    return max(length)

def xavier_norm(model:nn.Module):
    ''' use xavier normalization to initialize model parameters.
    :param: model -> pointer to an nn.Module
    '''
    for n,p in model.named_parameters():
        if p.requires_grad:
            if 'weight' in n:
                xavier_normal_(p)
            elif "bias" in n:
                constant_(p,0)

# the flowing two tsv method are deprecated maybe
# ================================================================================
def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')


def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas

class SimpleDataset(Dataset):
    ''' simply use a list of data '''
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
def SimpleDataloader(dataset:Dataset,batch_size:int,pin_memory=True,shuffle=False):
    ''' simply load the items in the dataset (a list) '''
    naive_collate = lambda data:[data]
    loader = DataLoader(dataset,batch_size=batch_size,pin_memory=pin_memory,collate_fn=naive_collate,shuffle=shuffle)
    return loader
    
# =======================================================================================


if __name__ == "__main__":
    # p = torch.rand(3, 3)
    # ids = torch.from_numpy(np.arange(3))
    # ans = gather(p, ids)
    # print("p:", p)
    # print("ans", ans)
    # centroid_ids = torch.tensor([0, 2, 3])
    # pd = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [2, 3, 7, 6, 4], [2, 9, 7, 0, 4], [1, 8, 2, 3, 0]])
    # for i in range(5):
    #     pd[i][i] = 0
    # ans = gather(pd, centroid_ids)
    # print("ans:", ans)
    # w = torch.where(ans == 0)
    # print(w)

    # logger = generate_logger()
    # logger.info("hello")

    # ori = ["i love you,but it seems. like you love me too!ok? ",".yes,be quickly!"]
    # print(clean_tokenize("i love you,but it seems. like you love me too!ok? "))

    # write_to_tsv("./test.csv", ["index", "pred"], [{"index": 1, "pred": 2}, {"index": 1, "pred": 2}])
    # s = "I catch you so much as looking at another woman, I will kill you."
    # r = clean_tokenize(s)
    # print(r)
    
    t = [{1:1},{2:2},{3:3},
         {1:1},{2:2},{3:3}]
    data = SimpleDataset(t)
    loader = SimpleDataloader(data,batch_size=4,pin_memory=True,shuffle=True)
    for i, d in enumerate(loader):
        print(f"{i}: {d}")
        break
    # loader = SimpleDataloader(data,batch_size=4,pin_memory=True,shuffle=True)
    for i, d in enumerate(loader):
        print(f"{i}: {d}")
        break
    # loader = SimpleDataloader(data,batch_size=4,pin_memory=True,shuffle=True)
    for i, d in enumerate(loader):
        print(f"{i}: {d}")
        break