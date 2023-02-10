'''
implement neg instruction with some slight changes:

1. del stop words randomly
2. del words randomly
3. del one sentence randomly
4. shuffle the order
5. repeat one sentence randomly
'''
from math import ceil
import random
import copy
from unittest import result
import numpy as np
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from string import punctuation

# split_sen = lambda sen:sen.split(".")    
sen_end_indi = [".","!","?"]

def split_sen(sen:str):
    '''
    tokenize the ori instruction. i.e., split it into tokens and punctuations
    '''
    return word_tokenize(sen)

def del_stop_words(sen:str,ratio:float=0.2,threshold:int=5):
    # get stop words
    stop_words = set(stopwords.words('english')) 
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    all_stop_idx = [idx for idx,tok in enumerate(all_toks) if tok in stop_words]
    # num = max(ceil(len(all_stop_idx) * ratio),threshold) ## avoid the ratio is too small
    num = ceil(len(all_stop_idx) * ratio)
    # randomly select stop words
    del_index = random.sample(all_stop_idx,num)
    # del stop words
    results = [tok for i,tok in enumerate(all_toks) if i not in del_index]
    
    return " ".join(results)

def del_words(sen:str,ratio:float=0.2,threshold:int=5):
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    # get words (except punctuations)
    words_idx = [idx for idx,tok in enumerate(all_toks) if tok not in punctuation]
    num = ceil(len(words_idx) * ratio)
    # num = max(ceil(len(words_idx) * ratio),threshold) ## avoid the ratio is too small
    assert num >= 1, "del at least one word!"
    # del_index = np.random.choice(len(all_toks),num)
    del_index = random.sample(words_idx,num)
    # del words
    results = [tok for i,tok in enumerate(all_toks) if i not in del_index]
    
    assert len(results) + num == len(all_toks) 
    
    return " ".join(results)
    
def randomly_del_sen(sen:str):
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    # count dot num (sen num)
    dot_dic = dict(Counter(all_toks))
    dot_num = np.sum([dot_dic.get(t,0) for t in sen_end_indi])
    assert dot_num > 0, "at leat one dot"
    # random select a sentence
    del_target = random.choice(range(dot_num))
    # get the index of dots
    all_dot_idx = [idx for idx,tok in enumerate(all_toks) if tok in sen_end_indi]
    # get start and end
    end = all_dot_idx[del_target] + 1
    start = all_dot_idx[del_target-1] + 1 if del_target > 0 else 0
    results = all_toks[:start] + all_toks[end:]
    
    return " ".join(results)

def randomly_rep_sen(sen:str):
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    # count dot num (sen num)
    dot_dic = dict(Counter(all_toks))
    dot_num = np.sum([dot_dic.get(t,0) for t in sen_end_indi])
    assert dot_num > 0, "at leat one dot"
    # random select a sentence
    rep_target = random.choice(range(dot_num))
    # get the index of dots
    all_dot_idx = [idx for idx,tok in enumerate(all_toks) if tok in sen_end_indi]
    # get start and end
    start = all_dot_idx[rep_target-1] + 1 if rep_target > 0 else 0
    end = all_dot_idx[rep_target] + 1
    rep = copy.deepcopy(all_toks[start:end])
    results = all_toks[:start] + rep + all_toks[start:]
    
    return " ".join(results)


def randomly_shuffle(sen:str):
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    # get the index of dots
    all_dot_idx = [idx for idx,tok in enumerate(all_toks) if tok in sen_end_indi]
    all_sens = []
    for i,idx in enumerate(all_dot_idx):
        if i == 0:
            cu_sen = all_toks[0:idx+1]
        else:
            cu_sen = all_toks[all_dot_idx[i-1]+1:idx+1]
        all_sens.append(cu_sen)
    
    if len(all_sens) <= 3:
        all_sens.reverse()  ## for those short sentences, just reverse the order
    else:
        random.shuffle(all_sens)
    
    results = []
    for t in all_sens:
        results += t
    
    return " ".join(results)


if __name__ == "__main__":
    sen = "This is renze, you can also call me reza. I am a PhD@TU. Pls feel free to reach out! Have any questions? Here is my email: XXX@temple.edu."
    print("==>ori:",sen,sep="\n")
    print("\n1. del stop words randomly:",del_stop_words(sen),sep="\n")
    print("\n2. del words randomly:",del_words(sen,ratio=0.2),sep="\n")
    print("\n3. del one sentence randomly:",randomly_del_sen(sen),sep="\n")
    print("\n4. shuffle the order:",randomly_shuffle(sen),sep="\n")
    print("\n5. repeat one sentence randomly:",randomly_rep_sen(sen),sep="\n")


