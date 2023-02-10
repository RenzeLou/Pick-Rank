import os
import torch
import json
import sys
import argparse
import random
import copy
import numpy as np
from tqdm import tqdm
from shutil import copyfile

sys.path.append(os.path.abspath("./kit/"))
sys.path.append(os.path.abspath("./"))

import edition_slight as ed_sl
import edition_huge as ed_hg

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    result = []
    for p in all_path:
        if not os.path.isdir(p) and sufix in p:
            result.append(os.path.join(task_path,p))
            
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    task_path = "./data/tasks"
    PREFIX = len(task_path) + 1
    split_path = "./data/splits/add_dev"

    del_ent = lambda x:x[:-1]
    add_ent = lambda x:x+"\n"
    
    train_num,test_num,dev_num = 0,0,0
    train_ins_num,test_ins_num,dev_ins_num = 0,0,0
    train_tk_name, test_tk_name = [], []
    
    samples = []
    train_w_num,test_w_num,dev_w_num = [],[],[]
    train_s_num,test_s_num,dev_s_num = [],[],[]
    all_w_num = []
    all_s_num = []
    train_tk_type, test_tk_type, dev_tk_type = [],[],[]
    train_dm_type, test_dm_type, dev_dm_type = [],[],[]
    train_sc_type, test_sc_type, dev_sc_type = [],[],[]
    en_tk_num,all_tk_num = 0,0
    tr_en_tk_num = 0
    te_en_tk_num = 0
    de_en_tk_num = 0
    
    import spacy
    sentinizer = spacy.load('en_core_web_sm')
    
    with open(split_path+"/test_tasks.txt","r") as sp:
        all_te_tasks = sp.readlines()
        all_te_tasks = list(map(del_ent,all_te_tasks))
        all_te_tasks_key = dict([(t,1) for t in all_te_tasks])

    with open(split_path+"/dev_tasks.txt","r") as sp:
        all_de_tasks = sp.readlines()
        all_de_tasks = list(map(del_ent,all_de_tasks))
        all_de_tasks_key = dict([(t,1) for t in all_de_tasks])    
    
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        

    all_tasks_pt = FindAllSuffix(task_path,"json")
    other_instructions = dict()
    other_instructions_test = dict()
    for _,tk_pt in tqdm(enumerate(all_tasks_pt)):
        all_tk_num += 1
        # tk_name = tk_pt.rsplit("/",1)[-1].rsplit(".",1)[0]
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_tr_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                cate = tk_info["Categories"][0]
                train_tk_type.append(cate)
                definition = tk_info["Definition"][0]
                all_w_num.append(len(definition.strip().split(" ")))
                train_w_num.append(len(definition.strip().split(" ")))
                train_dm_type.append(tk_info["Domains"][0])
                train_sc_type.append(tk_info["Source"][0])
                train_ins_num += len(tk_info["Instances"][:100])
                # train_s_num.append(len(ed_hg.remain_one_sen_v2(definition)))
                doc = sentinizer(definition)
                results = list(doc.sents)
                results = [str(s) for s in results]
                train_s_num.append(len(results))
                all_s_num.append(len(results))
                instruction = copy.deepcopy(tk_info["Definition"][0])
                if other_instructions.get(cate,None) is None:
                    other_instructions[cate] = [instruction]
                else:
                    other_instructions[cate].append(instruction)
                all_lan = set([tk_info["Input_language"][0], tk_info["Output_language"][0], tk_info["Instruction_language"][0]])
                if len(all_lan) == 1 and "English" in all_lan:    
                    en_tk_num += 1
                    tr_en_tk_num += 1  
        elif all_te_tasks_key.get(tk_name,0) == 1:
            test_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                cate = tk_info["Categories"][0]
                test_tk_type.append(cate)
                definition = tk_info["Definition"][0]
                all_w_num.append(len(definition.strip().split(" ")))
                test_w_num.append(len(definition.strip().split(" ")))
                test_dm_type.append(tk_info["Domains"][0])
                test_sc_type.append(tk_info["Source"][0])
                test_ins_num += len(tk_info["Instances"][:100])
                # test_s_num.append(len(ed_hg.remain_one_sen_v2(definition)))
                doc = sentinizer(definition)
                results = list(doc.sents)
                results = [str(s) for s in results]
                test_s_num.append(len(results))
                all_s_num.append(len(results))
                instruction = copy.deepcopy(tk_info["Definition"][0])
                if other_instructions_test.get(cate,None) is None:
                    other_instructions_test[cate] = [instruction]
                else:
                    other_instructions_test[cate].append(instruction)
                all_lan = set([tk_info["Input_language"][0], tk_info["Output_language"][0], tk_info["Instruction_language"][0]])
                if len(all_lan) == 1 and "English" in all_lan:    
                    en_tk_num += 1
                    te_en_tk_num += 1
        elif all_de_tasks_key.get(tk_name,0) == 1:
            dev_num += 1
            with open(tk_pt,"r",encoding="utf-8") as tk:
                tk_info = json.load(tk) 
                cate = tk_info["Categories"][0]
                dev_tk_type.append(cate)
                dev_dm_type.append(tk_info["Domains"][0])
                dev_sc_type.append(tk_info["Source"][0])
                dev_ins_num += len(tk_info["Instances"][:100])
                definition = tk_info["Definition"][0]
                all_w_num.append(len(definition.strip().split(" ")))
                dev_w_num.append(len(definition.strip().split(" ")))
                # dev_s_num.append(len(ed_hg.remain_one_sen_v2(definition)))
                doc = sentinizer(definition)
                results = list(doc.sents)
                results = [str(s) for s in results]
                dev_s_num.append(len(results))
                all_s_num.append(len(results))
                all_lan = set([tk_info["Input_language"][0], tk_info["Output_language"][0], tk_info["Instruction_language"][0]])
                if len(all_lan) == 1 and "English" in all_lan:    
                    en_tk_num += 1
                    de_en_tk_num += 1
        # print all the statistics
        print("\n" + "="*40 + "\n")
        print("train_num: ",train_num,"test_num: ",test_num,"dev_num: ",dev_num)
        # print(train_ins_num,test_ins_num,dev_ins_num)
        print("train_w_num: ",np.mean(train_w_num),"test_w_num: ",np.mean(test_w_num),"dev_w_num: ",np.mean(dev_w_num))
        print("train_s_num: ",np.mean(train_s_num),"test_s_num: ",np.mean(test_s_num),"dev_s_num: ",np.mean(dev_s_num))
        print("all_w_num: ",np.mean(all_w_num),"all_s_num: ",np.mean(all_s_num),sep="\n")
        print("train_tk_type: ",len(set(train_tk_type)),"test_tk_type: ",len(set(test_tk_type)),"dev_tk_type: ",len(set(dev_tk_type)))
        print("train_dm_type: ",len(set(train_dm_type)),"test_dm_type: ",len(set(test_dm_type)),"dev_dm_type: ",len(set(dev_dm_type)))
        print("train_sc_type: ",len(set(train_sc_type)),"test_sc_type: ",len(set(test_sc_type)),"dev_sc_type: ",len(set(dev_sc_type)))
        train_overlap = set(train_sc_type) & set(test_sc_type)
        dev_overlap = set(dev_sc_type) & set(test_sc_type)
        print("train_overlap: ",len(train_overlap),len(train_overlap)/len(set(train_sc_type)),"dev_overlap: ",len(dev_overlap),len(dev_overlap)/len(set(dev_sc_type)))
        print("all_tk_num",all_tk_num,"en_tk_num: ",en_tk_num,"tr_en_tk_num: ",tr_en_tk_num,"te_en_tk_num: ",te_en_tk_num,"de_en_tk_num: ",de_en_tk_num)
        
        
if __name__ == "__main__":
    main()