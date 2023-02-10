'''
implement neg instruction with some huge changes:

1. remain one sentence randomly
2. remain the first sentence
3. shuffle word order
4. replace the instruction with the one in other tasks randomly
5. insert one sentence from other tasks in this instruction
'''
from math import ceil
import random
from re import T
from unittest import result
import numpy as np
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from string import punctuation
import spacy

# split_sen = lambda sen:sen.split(".")    
sen_end_indi = [".","!","?"]

def split_sen(sen:str):
    '''
    tokenize the ori instruction. i.e., split it into tokens and punctuations
    '''
    return word_tokenize(sen)

def randomly_remain_sen(sen:str):
    all_toks = split_sen(sen)
    if not any([t in all_toks for t in sen_end_indi]):
        all_toks += ["."]
    # count dot num (sen num)
    dot_dic = dict(Counter(all_toks))
    dot_num = np.sum([dot_dic.get(t,0) for t in sen_end_indi])
    assert dot_num > 0, "at leat one dot"
    # randomly select a sentence
    remain_target = random.choice(range(dot_num))
    # get the index of dots
    all_dot_idx = [idx for idx,tok in enumerate(all_toks) if tok in sen_end_indi]
    # get start and end
    end = all_dot_idx[remain_target] + 1
    start = all_dot_idx[remain_target-1] + 1 if remain_target > 0 else 0
    results = all_toks[start:end]
    
    return " ".join(results)

def remain_first_sen(sen:str):
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
    # only remain the first sentence
    results = all_sens[0]
    
    return " ".join(results)

def remain_last_sen(sen:str):
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
    # only remain the last sentence
    results = all_sens[-1]
    
    return " ".join(results)

def remain_one_sen(sen:str)->list:
    ''' return a list, each element is one sentence from ori def '''
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
    # only remain the first sentence
    results = [" ".join(s) for s in all_sens]
    
    return results

def remain_one_sen_v2(sen:str)->list:
    ''' return a list, each element is one sentence from ori def '''
    sentinizer = spacy.load('en_core_web_sm')
    doc = sentinizer(sen)
    results = list(doc.sents)
    results = [str(s) for s in results]
    
    return results

def del_one_sen(sen:str)->list:
    ''' return a list, each element is a neg def which delete one sentence from ori def '''
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
    # del one sentence
    results = []
    for i in range(len(all_sens)):
        r = all_sens[0:i] + all_sens[i+1:]
        r = [" ".join(s) for s in r]
        r = " ".join(r)
        results.append(r)
    
    return results

def randomly_shuffle_words(sen:str):
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
    results = []
    # shuffle the words in each sentence
    for t in all_sens:
        random.shuffle(t)
        results += t
    
    return " ".join(results)

def randomly_replace(sen:str,others:list,num:int=1):
    results = [sen]
    while sen in results:
        results = random.sample(others,num)
    
    if num == 1:
        return results[0]
    else:
        return results

def randomly_insert(sen:str,others:list):
    '''the 'others' can be the instructions of other tasks
    or even the instances
    '''
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
    
    # randomly select one instance
    src = random.sample(others,1)[0]
    # randomly select one sentence from this instance
    all_toks_src = split_sen(src)
    all_dot_idx_src = [idx for idx,tok in enumerate(all_toks_src) if tok in sen_end_indi]
    if len(all_dot_idx_src) == 0:
        all_dot_idx_src = [len(all_toks_src)]
        all_toks_src += ["."]
    all_sens_src = []
    for i,idx in enumerate(all_dot_idx_src):
        if i == 0:
            cu_sen = all_toks_src[0:idx+1]
        else:
            cu_sen = all_toks_src[all_dot_idx_src[i-1]+1:idx+1]
        all_sens_src.append(cu_sen)
    src_sen = random.sample(all_sens_src,1)[0]
    
    # randomly insert to the original instruction
    inser_point = random.choice(range(len(all_sens)+1))   
    all_sens.insert(inser_point,src_sen)
    results = []
    for t in all_sens:
        results += t
    
    return " ".join(results)       


if __name__ == "__main__":
    sen = "This is renze, you can also call me reza. I am a PhD@TU. Pls feel free to reach out! Have any questions? Here is my email: XXX@temple.edu."
    others = ["Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will \"give him lodgings for nothing, and never ask for change\". Grateful for the unexpected assistance, Oliver follows the Dodger to the \"old gentleman's\" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.",
            "What is the alias of the person whose sidekick had a humorous nature?.",
            "This question is based on the following sentence in the passage \"Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the \"Artful Dodger\", and his sidekick, a boy of a humorous nature named Charley Bates\". The pronoun \"his\" refers to a person with multiple names. But since the question explicitly asks for the alias, the answer is unambiguous."]
    
    print("==>ori:",sen,sep="\n")
    # print("\n1. remain one sentence randomly:",randomly_remain_sen(sen),sep="\n")
    # print("\n2. remain the first sentence:",remain_first_sen(sen),sep="\n")
    # print("\n3. shuffle word order:",randomly_shuffle_words(sen),sep="\n")
    # print("\n4. replace the instruction:",randomly_replace(sen,others=others),sep="\n")
    # print("\n5. insert one sentence:",randomly_insert(sen,others=others),sep="\n")
    print("\n6.1. remain one sen:",remain_one_sen(sen),sep="\n")
    print("\n6.2. remain one sen:",remain_one_sen_v2(sen),sep="\n")
    # print("\n7. del_one_sen:",del_one_sen(sen),sep="\n")
    