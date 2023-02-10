from copy import copy, deepcopy
import os
import json
from turtle import color
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from regex import R
'''
load the exp results, plot it
for classification: EM only
for generalization: RougeL only

can be used to find a margin value
can also used to find global best found
'''

## TODO: 2022-10-3 15:19, the best margin value is 0.003
## TODO: here is the selection results:
'''
select margin based on ratio:[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
===> sorted by max:
margin: 0.003   max: 44.5888
margin: 0.03    max: 44.5467
margin: 0.01    max: 44.4197
margin: 0.1     max: 44.3794
margin: 0.0003  max: 44.3052
margin: 0.3     max: 44.2656
margin: 0.001   max: 44.2192
margin: 0.0001  max: 44.1382
'''

# variables (hyper-parameters)
model_name = "t5-base"
round_all = lambda x:round(x,1)
# loss_mix_ratio_list = [0.0001,0.0003,0.001,0.003,0.01,0.03]
loss_mix_ratio_list = [0.001]
margin_list = [0.0003] 
# loss_mix_ratio_list = [0.00001,0.00003,0.00006,0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1] 
sample_num_list = [1,2,3,4,5]  ## there are 5 kinds of negative instructions  ,2,3,4,5
# margin_list = [0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3]



## input and output path
exp_name_template = "more-neg_sample-{}-all-neg-cal"
base_name_template = "{}-pos"
exp_path = "../output"
out_path = "../plot/all_cal"

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

def main():
    em_best_result,rouge_best_result = -1,-1
    em_best_par,rouge_best_par = {"ratio":None,"margin":None,"sample":None,"rouge":None},{"ratio":None,"margin":None,"sample":None,"em":None}
    # set ratio as axis
    for loss_mix_ratio in loss_mix_ratio_list:
        # observe_par = dict()
        observe_par_avg = dict()
        observe_par_max = dict()
        for margin in margin_list:
            all_results = dict()  
            ## one exp result corresponding to one ratio
            for sample_num in sample_num_list:
                exp_name = exp_name_template.format(sample_num)
                file_name = os.path.join(exp_path,exp_name,"predict_results.json")
                try:
                    results = read_exp_results(file_name)
                except FileNotFoundError:
                    print("warn: file not found, need more time to complete all running (sample_num:{})!".format(sample_num))
                    continue
                all_results[sample_num] = results
                ## save the global best results
                em_best_result,rouge_best_result,em_best_par,rouge_best_par = find_global(results,loss_mix_ratio,margin,sample_num,em_best_result,rouge_best_result,em_best_par,rouge_best_par)
            if len(all_results) == 0:
                continue
            ## baseline, i.e., only pos
            base_name = base_name_template.format(model_name)
            base_file_name = os.path.join(exp_path,base_name,"predict_results.json")
            base_results = read_exp_results(base_file_name)
            all_results["base"] = base_results
            ## plot the results (2 metrics, 9 ratios, 12 tasks, 2 avg scoren and base score)
            plot_neg_exp_EM(all_results,out_path,pic_name="{}_sample_{}_EM".format(model_name,sample_num),plot_tasks=False)
            plot_neg_exp_Rouge(all_results,out_path,pic_name="{}_sample_{}_RougeL".format(model_name,sample_num),plot_tasks=False)
            # print(all_results)
            ## get the overall performance of this value (margin or ratio)
            avg_value = get_avg(deepcopy(all_results))
            max_value = get_max(deepcopy(all_results))
            observe_par_avg[margin] = avg_value
            observe_par_max[margin] = max_value
        ## print the exp results to find a good value of hyper-parameter
        observe_par_avg = dict(sorted(observe_par_avg.items(),key=lambda x:x[1],reverse=True))
        # print_selection(observe_par_avg,key="avg")
        observe_par_max = dict(sorted(observe_par_max.items(),key=lambda x:x[1],reverse=True))
        # print_selection(observe_par_max,key="max")
    
    # print the global best found
    print("\nHere is the best found (global):")
    print("best EM: {}\tparameters: {}".format(em_best_result,em_best_par))
    print("best RougeL: {}\tparameters: {}".format(rouge_best_result,rouge_best_par))
    print("\nHere is the pos results (for reference):")
    print("EM: {}\tRougeL: {}".format(base_results["EM"]["AVG"],base_results["RougeL"]["AVG"]))

if __name__ == "__main__":
    # file_name = "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/output/t5-base-sample-1-ratio-0.1/predict_results.json"
    # results = read_exp_results(file_name=file_name)
    # print(results)
    main()