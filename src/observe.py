from multiprocessing.connection import wait
import os
import json
from turtle import color
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
'''
load the exp results, plot it
for classification: EM only
for generalization: RougeL only
'''

# variables (hyper-parameters)
model_name = "t5-base"
round_all = lambda x:round(x,1)
loss_mix_ratio_list = list(map(round_all,list(np.arange(0.1,0.6,0.1))))
sample_num_list = [5]  ## there are 5 kinds of negative instructions  ,2,3,4,5


## input and output path
exp_name_template = "{}-sample-{}-ratio-{}"
base_name_template = "{}-pos"
exp_path = "../output"
out_path = "../plot"

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


def plot_neg_exp_EM(all_results:dict,out_path:str,pic_name:str):
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
    for tk, scores in EM_tk_score.items():
        plt.plot(x,scores,linewidth=1.5, linestyle='--', alpha = 0.5,label=tk)
    # plot the avg of all tasks and the baseline
    plt.plot(x, EM_AVG, color="red",linewidth=3, linestyle='-', label="AVG")
    plt.plot(x, [base_AVG]*len(x),color="blue", linewidth=3, linestyle='-', label="Base")
    
    ax = plt.gca()
    ax.set_aspect(1./ax.get_data_ratio())
    
    font = {
    'weight' : 'normal',
    'size'   : 16,
    }
    
    plt.xlabel("neg loss ratio",font)
    plt.ylabel("EM",font)
    
    legend = plt.legend(loc=(1,0.5), frameon=False,shadow=True, fontsize='x-large')
    plt.tight_layout() 
    
    pic_name = pic_name + ".png" if not pic_name.endswith(".png") else pic_name
    plt.savefig(os.path.join(out_path, pic_name))
    plt.show()
    
def plot_neg_exp_Rouge(all_results:dict,out_path:str,pic_name:str):
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
    for tk, scores in Rouge_tk_score.items():
        plt.plot(x,scores,linewidth=1.5, linestyle='--', alpha = 0.5,label=tk)
    # plot the avg of all tasks and the baseline
    plt.plot(x, Rouge_AVG, color="red",linewidth=3, linestyle='-', label="AVG")
    plt.plot(x, [base_AVG]*len(x),color="blue", linewidth=3, linestyle='-', label="Base")
    
    ax = plt.gca()
    ax.set_aspect(1./ax.get_data_ratio())
    
    font = {
    'weight' : 'normal',
    'size'   : 16,
    }
    
    plt.xlabel("neg loss ratio",font)
    plt.ylabel("RougeL",font)
    
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

def main():
    # set ratio as axis
    for sample_num in sample_num_list:
        all_results = dict()  
        ## one exp result corresponding to one ratio
        for loss_mix_ratio in loss_mix_ratio_list:
            exp_name = exp_name_template.format(model_name,sample_num,loss_mix_ratio)
            file_name = os.path.join(exp_path,exp_name,"predict_results.json")
            results = read_exp_results(file_name)
            all_results[loss_mix_ratio] = results
        ## baseline, i.e., only pos
        base_name = base_name_template.format(model_name)
        base_file_name = os.path.join(exp_path,base_name,"predict_results.json")
        base_results = read_exp_results(base_file_name)
        all_results["base"] = base_results
        ## plot the results (2 metrics, 9 ratios, 12 tasks, 2 avg scoren and base score)
        plot_neg_exp_EM(all_results,out_path,pic_name="{}_{}_EM".format(model_name,sample_num))
        plot_neg_exp_Rouge(all_results,out_path,pic_name="{}_{}_RougeL".format(model_name,sample_num))
        # print(all_results)


if __name__ == "__main__":
    # file_name = "/home/tuq59834/code/project/Tk-ins/Tk-Instruct/output/t5-base-sample-1-ratio-0.1/predict_results.json"
    # results = read_exp_results(file_name=file_name)
    # print(results)
    main()