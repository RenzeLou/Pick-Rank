''' 
process the data, split the sentences of the original definition.
'''
import os
import torch
import json
import sys
import argparse
import random
import copy
import numpy as np
import multiprocessing
from tqdm import tqdm

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
    parser.add_argument("--tgt_path",type=str,default="def_segmentation")
    parser.add_argument("--seed",type=int,default=42)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    # source path
    task_path = "./data/tasks"
    PREFIX = len(task_path) + 1
    split_path = "./data/splits/add_dev"

    # target path
    task_path_tgt = os.path.join(task_path,args.tgt_path) 
    os.makedirs(task_path_tgt,exist_ok=True)

    del_ent = lambda x:x[:-1]
    add_ent = lambda x:x+"\n"
    
    train_num,test_num,dev_num = 0,0,0
    train_tk_name, test_tk_name, dev_tk_name = [], [], []
    
    with open(split_path+"/test_tasks.txt","r") as sp:
        all_te_tasks = sp.readlines()
        all_te_tasks = list(map(del_ent,all_te_tasks))
        all_te_tasks_key = dict([(t,1) for t in all_te_tasks])
    
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
    
    with open(split_path+"/dev_tasks.txt","r") as sp:
        all_de_tasks = sp.readlines()
        all_de_tasks = list(map(del_ent,all_de_tasks))
        all_de_tasks_key = dict([(t,1) for t in all_de_tasks])
        

    all_tasks_pt = FindAllSuffix(task_path,"json")
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        # if all_tr_tasks_key.get(tk_name,0) == 1 or all_te_tasks_key.get(tk_name,0) == 1:
        if all_tr_tasks_key.get(tk_name,0) == 1:
            train_num += 1
            train_tk_name.append(tk_name)
            # current_other_instructions = copy.deepcopy(other_instructions)
        elif all_te_tasks_key.get(tk_name,0) == 1:
            test_num += 1
            test_tk_name.append(tk_name)
            # current_other_instructions = copy.deepcopy(other_instructions_test)
        elif all_de_tasks_key.get(tk_name,0) == 1:
            dev_num += 1
            dev_tk_name.append(tk_name)
        else:
            # skip those excluded tasks
            continue
        with open(tk_pt,"r",encoding="utf-8") as tk:
            tk_info = json.load(tk) 
            cate = tk_info["Categories"][0]
            instruction = copy.deepcopy(tk_info["Definition"][0])
            # sentence segmentation
            tk_info["Definition_POS"] = ed_hg.remain_one_sen_v2(instruction)
            tk_info["Definition_NEG"] = []  # deprecated, keep it for consistency
            new_instances = []
            for item in tk_info["Instances"]:
                item["neg_output"] = []  # deprecated, keep it for consistency
                new_instances.append(item)
            tk_info["Instances"] = new_instances
            ## save data
            with open(task_path_tgt+"/{}.json".format(tk_name),"w",encoding="utf-8") as tgt_file:
                json.dump(tk_info,tgt_file,indent=2,sort_keys=False)
    
    assert train_num == len(all_tr_tasks)
    assert test_num == len(all_te_tasks)
    assert dev_num == len(all_de_tasks)

    print("\n" + "="*40 + "\n")
    print("dump mixed info at {} successfully!\ntotally {} training tasks; {} test tasks; {} dev tasks.".format(task_path_tgt,train_num,test_num,dev_num))


if __name__ == "__main__":
    # main()
    # pool = multiprocessing.Pool(4)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.apply_async(func=main)
    pool.close()
    pool.join()
