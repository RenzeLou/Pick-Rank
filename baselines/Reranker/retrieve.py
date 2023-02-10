import os
import copy
import numpy as np
import argparse
import random
import json
import faiss                  
from tqdm import tqdm

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data.generate_nli_data import FindAllSuffix, construct_nli_instance


def convert_def_to_vector(tk_pt:str,tokenizer:AutoTokenizer,dense_encoder:AutoModelForSeq2SeqLM,
                          gpu:int,dim:int,max_len:int=1024):
    # TODO: batchfy
    with torch.no_grad():
        with open(tk_pt,"r",encoding="utf-8") as tk:
            tk_info = json.load(tk) 
            definition = copy.deepcopy(tk_info["Definition"][0])
            all_ins = tk_info["Instances"]
            input_ids = tokenizer(definition,max_length=max_len, return_tensors="pt").input_ids  # Batch size 1
            input_ids = input_ids.to(f"cuda:{gpu}")
            outputs = dense_encoder(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state # [1, seq_len, 768]
            assert dim == last_hidden_states.shape[-1], f"dimension error, should set 'DIM' to {last_hidden_states.shape[-1]}"
            rep = torch.mean(last_hidden_states,dim=1)
            rep.squeeze_()
            rep_final = rep.detach().cpu().numpy().astype('float32')
    
    return rep_final, all_ins, definition


def construct_matrix(dense_vec:dict):
    id2name, vec_lis = dict(),[]
    for id, (tk_name, vec) in enumerate(dense_vec.items()):
        id2name[id] = tk_name
        vec_lis.append(vec)
    vec_matrix = np.array(vec_lis)
    
    return id2name, vec_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_encoder_path",type=str,
                        default="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/teacher_model/POS_augmentation-num_0",
                        help="path to the pre-trained T5 model.")
    parser.add_argument("--reranker_path",type=str, default="./out/bert-base-cased", help="path to the reranker model.")
    parser.add_argument("--gpu",type=int,default=5)
    parser.add_argument("--src_path",type=str,default="/home/tuq59834/code/project/Tk-ins/Tk-Instruct/data",
                        help="path to the original NI dataset.")
    parser.add_argument("--tgt_path",type=str,default="./ranking_result",
                        help="results saving path.")
    parser.add_argument("--tgt_name",type=str,default="main_loss-bert_base",
                        help="results saving path.")
    parser.add_argument("--k",type=int,default=5,
                        help="for each query task, search top k candidate tasks.")
    # parser.add_argument("--exp_num",type=int,default=10,
    #                     help="example num for each test task (for the further in-context learning).")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max_seq_len",type=int,default=1024)
    
    

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    set_seed(args.seed)
    
    '''
    ********************************************** 
    1. Load the dense encoder to convert all def into vectors 
    **********************************************
    '''
    
    # pre-trained encoder 
    # assume after training on the (def, x) -> y, model is good at interpretating the definition
    # thus it can well encode the semantic rep for the task objective
    config = AutoConfig.from_pretrained(args.dense_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(args.dense_encoder_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.dense_encoder_path,config=config)
    model.resize_token_embeddings(len(tokenizer))
    dense_encoder = model.encoder
    dense_encoder.eval()
    dense_encoder.to(f"cuda:{args.gpu}")
    
    # load tasks (training & test)
    # for each task, use encoder to encode the task definition
    print("** Convert the definitions from all tasks to dense vectors **")
    task_path = os.path.join(args.src_path,"tasks")
    split_path = os.path.join(args.src_path,"splits/default")
    PREFIX = len(task_path) + 1
    all_tasks_pt = FindAllSuffix(task_path,"json")
    del_ent = lambda x:x[:-1]
    DIM = 768
    
    with open(split_path+"/test_tasks.txt","r") as sp:
        all_te_tasks = sp.readlines()
        all_te_tasks = list(map(del_ent,all_te_tasks))
        all_te_tasks_key = dict([(t,1) for t in all_te_tasks])
    
    with open(split_path+"/train_tasks.txt","r") as sp:
        all_tr_tasks = sp.readlines()
        all_tr_tasks = list(map(del_ent,all_tr_tasks))
        all_tr_tasks_key = dict([(t,1) for t in all_tr_tasks])
        
    tr_dense_vec, te_dense_vec = dict(), dict()
    tr_examples, te_defs = dict(), dict()  # used for further example-level ranking
    for _,tk_pt in enumerate(tqdm(all_tasks_pt)):
        tk_name = tk_pt[PREFIX:len(tk_pt)-5]
        if all_tr_tasks_key.get(tk_name,0) == 1:
            # training tasks
            def_rep, all_examples, _ = convert_def_to_vector(tk_pt,tokenizer,dense_encoder,args.gpu,DIM,args.max_seq_len) # single vector (numpy array) with 768 dim
            tr_dense_vec[tk_name] = def_rep
            tr_examples[tk_name] = all_examples
        elif all_te_tasks_key.get(tk_name,0) == 1:
            # test tasks
            def_rep, _ , definition = convert_def_to_vector(tk_pt,tokenizer,dense_encoder,args.gpu,DIM,args.max_seq_len) # single vector (numpy array) with 768 dim
            te_dense_vec[tk_name] = def_rep
            te_defs[tk_name] = definition
    
    assert len(tr_dense_vec) == len(all_tr_tasks) and len(te_dense_vec) == len(all_te_tasks)
    
    # get matrix
    tr_id2name, tr_dense_matrix = construct_matrix(tr_dense_vec)
    te_id2name, te_dense_matrix = construct_matrix(te_dense_vec)
    
    
    ''' 
    **********************************************
    2. Use MIPS to calculate the task-level scores 
    **********************************************
    '''
    
    print("** Use MIPS to retrieve training tasks **")
    
    search_res = dict()
    index = faiss.IndexFlatL2(DIM)   # build the index
    assert index.is_trained
    index.add(tr_dense_matrix)   # add vectors to the index
    D, I = index.search(te_dense_matrix, args.k)     # retrieve
    for id in tqdm(range(I.shape[0])):
        # top k candidate tasks
        candidates = I[id,:].tolist()
        distances = D[id,:]
        scores = ((distances.max()-distances)/(distances.max()-distances.min())).tolist()
        test_name = te_id2name[id]
        item = dict()
        for can_id, s in zip(candidates,scores):
            train_name = tr_id2name[can_id]
            item[train_name] = s  # save the score for reference
        search_res[test_name] = item
    
    
    ''' 
    **********************************************
    3. Use Re-ranker to calculate the example-level scores 
    **********************************************
    '''
    
    print("** Use pre-trained LM to retrieve training examples **")
    
    # BERT pre-trained on NLI tasks (i.e., [def, example] -> 1 / 0)
    # assume after ptr-training, the model can successfully classify
    # whether a example has the similar objective pattern with the test task definition
    config = AutoConfig.from_pretrained(args.reranker_path)
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_path,config=config)
    model.resize_token_embeddings(len(tokenizer))
    del dense_encoder  # avoid occuping GPU resources
    model.eval()
    model.to(f"cuda:{args.gpu}")
    
    # for each test task with the candidate training tasks
    # concatenate the definition (from test task) with each example (from candidate training task)
    # use PLM to predict the relevant score
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for test_name, definition in tqdm(te_defs.items()):
            tr_can_name = list(search_res[test_name].keys())
            # considering the time consumption, only sample 100 ins from each candidate
            tr_can_exp, tr_can_exp_len = [],[]
            for train_name in tr_can_name:
                sample_num = min(len(tr_examples[train_name]),100)
                exps = random.sample(tr_examples[train_name],sample_num)
                tr_can_exp.extend(exps)
                tr_can_exp_len.append(sample_num)
            def_exp_ins_lis = construct_nli_instance(definition,tr_can_exp,label=-1)
            # predict the score
            id2score = dict()
            for id, def_exp_ins in enumerate(def_exp_ins_lis):
                sen1, sen2 = def_exp_ins[0], def_exp_ins[1]
                inputs = tokenizer(*(sen1,sen2), truncation=True)
                # print(inputs)
                inputs = dict([(k,torch.tensor(v).unsqueeze(0).to(f"cuda:{args.gpu}")) for k,v in inputs.items()])
                outputs = model(**inputs)
                prob = softmax(outputs.logits)  # [1,2]
                prob = prob.detach().cpu().numpy().tolist()[0]  
                score = prob[1]  # the probability for model to predict '1'
                id2score[id] = score
            # add the example-level scores
            exp_add_score = []
            for id, exp in enumerate(tr_can_exp):
                exp["score"] = id2score[id]
                exp_add_score.append(exp)
            # save the search and scoring results for this test task
            temp = 0
            tr_for_this_te = dict()
            for i, name in enumerate(tr_can_name):
                new_item = dict()
                task_level_score = search_res[test_name][name]
                new_item["score"] = task_level_score
                new_item["examples"] = exp_add_score[temp:temp+tr_can_exp_len[i]]
                temp += tr_can_exp_len[i]
                tr_for_this_te[name] = new_item
            search_res[test_name] = tr_for_this_te
            
                   
    '''
    ********************************************** 
    4. save the final selection results with all task- and example-level scores 
    **********************************************
    '''
    
    print("** Save the final results **")
    
    save_dir = os.path.join(args.tgt_path,args.tgt_name)
    os.makedirs(save_dir,exist_ok=True)
    
    file_num = 0
    for test_name, content in tqdm(search_res.items()):
        file_num += 1
        with open(os.path.join(save_dir,f"{test_name}.json"),"w",encoding="utf-8") as f:
            json.dump(content,f,indent=2)

    print(f" dump {file_num} test files at {save_dir}.")
    
if __name__ == "__main__":
    main()