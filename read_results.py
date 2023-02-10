from copy import copy, deepcopy
import os
import json
from turtle import color
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from regex import R

'''
help load all the test results (i.e., predict_results.json) under a specific folder
directly print the results (a dict) on the console (in a performance dcreasing order)
'''

# variables (hyper-parameters)
model_name = "t5-base"
round_all = lambda x:round(x,1)
loss_mix_ratio_list = [0.001,0.005,0.01,0.05,0.1,0.5,1]
margin_list = [0.0001,0.0003,0.001,0.003,0.01,0.03,]
sample_num_list = [9]  
epoch = [2]
lr = [5e-05,] 
neg_loss_type_list = ["contrastive_loss_all"]

## input and output path
exp_name_template = "all-margin_{}-num_{}-ratio_{}-func_{}-lr_{}-epoch_{}"
base_name_template = "{}-pos"
exp_path = "../output"
out_path = "../plot/margin-num-ratio-func-lr-epoch" 

## metric list
EM_list = ["TE","CEC","CR","DAR","AC","WA"]
Rouge_list = ["OE","KT","QR","TG","DT","GEC"]
## abv dic
name_abv = {"answerability_classification":"AC"		
,"cause_effect_classification": "CEC"		
,"coreference_resolution": "CR"		
,"data_to_text": "DT"		
,"dialogue_act_recognition": "DAR"		
,"grammar_error_correction": "GEC"		
,"keyword_tagging": "KT"		
,"overlap_extraction": "OE"		
,"question_rewriting": "QR"		
,"textual_entailment": "TE"		
,"title_generation": "TG"		
,"word_analogy": "WA"}
abv_name = dict([(v,k) for k,v in name_abv.items()])


def plot_neg_exp_EM(all_results:dict,out_path:str,pic_name:str,plot_tasks:bool=True):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    x = [k for k in all_results.keys() if k != "base"]
    plt.figure(figsize=(8,6))
    plt.tick_params(labelsize=13)
    # plot each task 
    EM_tk_score = dict()
    for tk in EM_list:
        EM_tk_score[tk] = []
    EM_tk_score["AVG"] = []
    for ratio,results in all_results.items():
        if ratio != "base":
            for tk,score in results["EM"].items():
                EM_tk_score[tk].append(score)
        else:
            ## only consider avg of base
            base_AVG = results["EM"]["AVG"]
    EM_AVG = EM_tk_score.pop("AVG")
    x_index = list(range(1,len(x)+1))
    if plot_tasks:
        for tk, scores in EM_tk_score.items():
            plt.plot(x_index,scores,linewidth=1.5, linestyle='--', alpha = 0.5,label=tk)
    # plot the avg of all tasks and the baseline
    plt.plot(x_index, EM_AVG, color="red",linewidth=3, linestyle='-', label="AVG")
    plt.plot(x_index, [base_AVG]*len(x),color="blue", linewidth=3, linestyle='-', label="Base")
    
    ax = plt.gca()
    ax.set_aspect(1./ax.get_data_ratio())
    
    font = {
    'weight' : 'normal',
    'size'   : 16,
    }
    
    plt.xlabel("Sample Num",font)
    plt.ylabel("EM",font)
    plt.xticks(x_index,x)
    
    legend = plt.legend(loc=(1,0.5), frameon=False,shadow=True, fontsize='x-large')
    plt.tight_layout() 
    
    pic_name = pic_name + ".png" if not pic_name.endswith(".png") else pic_name
    plt.savefig(os.path.join(out_path, pic_name))
    plt.show()
    
def plot_neg_exp_Rouge(all_results:dict,out_path:str,pic_name:str,plot_tasks:bool=True):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    x = [k for k in all_results.keys() if k != "base"]
    plt.figure(figsize=(8,6))
    plt.tick_params(labelsize=13)
    # plot each task 
    Rouge_tk_score = dict()
    for tk in Rouge_list:
        Rouge_tk_score[tk] = []
    Rouge_tk_score["AVG"] = []
    for ratio,results in all_results.items():
        if ratio != "base":
            for tk,score in results["RougeL"].items():
                Rouge_tk_score[tk].append(score)
        else:
            ## only consider avg of base
            base_AVG = results["RougeL"]["AVG"]
    Rouge_AVG = Rouge_tk_score.pop("AVG")
    x_index = list(range(1,len(x)+1))  # force the x to be equally distributed
    if plot_tasks:
        for tk, scores in Rouge_tk_score.items():
            plt.plot(x_index,scores,linewidth=1.5, linestyle='--', alpha = 0.5,label=tk)
    # plot the avg of all tasks and the baseline
    plt.plot(x_index, Rouge_AVG, color="red",linewidth=3, linestyle='-', label="AVG")
    plt.plot(x_index, [base_AVG]*len(x),color="blue", linewidth=3, linestyle='-', label="Base")
    
    ax = plt.gca()
    ax.set_aspect(1./ax.get_data_ratio())
    
    font = {
    'weight' : 'normal',
    'size'   : 16,
    }
    
    plt.xlabel("Sample Num",font)
    plt.ylabel("RougeL",font)
    plt.xticks(x_index,x)
    
    legend = plt.legend(loc=(1,0.5), frameon=False,shadow=True, fontsize='x-large')
    plt.tight_layout() 
    
    pic_name = pic_name + ".png" if not pic_name.endswith(".png") else pic_name
    plt.savefig(os.path.join(out_path, pic_name))
    plt.show()

def read_exp_results(file_name:str):
    # return a dict {"EM":{t1:...tn...}, "RougeL":{t1:...tn...}}
    results = dict()
    results["EM"] = dict()
    results["RougeL"] = dict()
    with open(file_name,"r",encoding="utf-8") as f:
        data = json.load(f)
    for k,v in data.items():
        if "_for_" in k:
            tk_name = k.split("_for_")[-1]
            tk_name_abv = name_abv[tk_name]
        else:
            tk_name = k
            tk_name_abv = k ## no abv
        if "predict_exact_match_for_" in k and tk_name_abv in EM_list:
            # task using EM
            results["EM"][tk_name_abv] = v 
        elif "predict_rougeL_for_" in k and tk_name_abv in Rouge_list:
            # task using RougeL
            results["RougeL"][tk_name_abv] = v
            
    ## add avg
    results["EM"]['AVG'] = round(np.mean([v for v in results["EM"].values()]),4)
    results["RougeL"]['AVG'] = round(np.mean([v for v in results["RougeL"].values()]),4)
    
    return results

def read_exp_results_v2(file_name:str):
    with open(file_name,"r",encoding="utf-8") as f:
        data = json.load(f)
    em = data['predict_exact_match_avg']
    rg = data['predict_rougeL_avg']
    rg_all = data['predict_rougeL']
    
    return em,rg,rg_all

def get_avg(all_results:dict):
    all_results.pop("base")
    rougel,em = [],[]
    for key,results in all_results.items():
        r = results["RougeL"]["AVG"]
        e = results["EM"]["AVG"]
        rougel.append(r)
        em.append(e)
    rougel_avg,em_avg = np.mean(em),np.mean(rougel)
    
    return round((rougel_avg+em_avg)/2.,4)

def get_max(all_results:dict):
    all_results.pop("base")
    rougel,em = [],[]
    for key,results in all_results.items():
        r = results["RougeL"]["AVG"]
        e = results["EM"]["AVG"]
        rougel.append(r)
        em.append(e)
    rougel_max,em_max = np.max(em),np.max(rougel)
    
    return round((rougel_max+em_max)/2.,4)

def print_selection(results:dict,key:str):
    print("\nselect sample_num")
    print("===> sorted by {}:".format(key))
    for k,v in results.items():
        print("sample_num: {}\t{}: {}".format(k,key,v))
def find_global(results:dict,ratio:float,margin:float,sample_num:int,em_best_result,rouge_best_result,em_best_par,rouge_best_par):
    r = results["RougeL"]["AVG"]
    e = results["EM"]["AVG"]
    if r >= rouge_best_result:
        rouge_best_result = r
        rouge_best_par["ratio"] = ratio
        rouge_best_par["margin"] = margin
        rouge_best_par["sample"] = sample_num
        rouge_best_par["em"] = e
    if e >= em_best_result:
        em_best_result = e
        em_best_par["ratio"] = ratio
        em_best_par["margin"] = margin
        em_best_par["sample"] = sample_num
        em_best_par["rouge"] = r
    
    return em_best_result,rouge_best_result,em_best_par,rouge_best_par
    
sort_max = lambda x:x[1]["max"]
sort_avg = lambda x:x[1]["avg"]

def FindAllSuffix(task_path,sufix="json"):
    all_path = os.listdir(task_path)
    all_path = [os.path.join(task_path,p) for p in all_path]
    result = []
    for p in all_path:
        file_name = os.path.join(p,sufix)
        if os.path.isdir(p) and os.path.isfile(file_name):
            result.append(file_name)
            
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="./output/",help="the output path.")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    results = dict()
    all_results_file = FindAllSuffix(args.path,"predict_results.json")

    for file_name in all_results_file:
        try:
            em,rg,rg_all = read_exp_results_v2(file_name) 
            em_rg_avg = np.mean([em,rg])
            exp_name = file_name.rsplit("/")[-2]
            results[exp_name] = [em,rg,rg_all]
        except FileNotFoundError:
            print("warn: the exp '{}' is not ready!".format(exp_name))
            continue
    key = lambda x:x[1][-1]
    results = dict(sorted(results.items(),key=key,reverse=True))
    
    print("\nTotally {} EXP results under './{}':".format(len(results),args.path))
    print("EXP Name\t[EM, RougeL, RougeL(overall)]")
    for k,v in results.items():
        print("{}:\t{}".format(k,v))
    
if __name__ == "__main__":
    main()