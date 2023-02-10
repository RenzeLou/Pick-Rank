## 1. env

use the same env as the Tk-ins, try `conda create -n point_it_out python==3.8.0` and `pip install -r requirements_pointer.txt`.

## 2. main scripts

- run_s2s_pointer_train_first.py

## 3. data dir

at `./data/tasks/pos_neg_def_delete`, with split file at `./data/splits/default`, there is no actual dev tasks specified in the split files (only one row for consistence), try to use `dev_tasks_beta.txt` instead of `dev_tasks.txt` when you doing dev.

## 4. run scripts

- pointer_train_v2_null_pos.sh
- pointer_train_v2_null.sh

using `sh scripts/pointer_train_v2_null_pos.sh 6 1 2.5 2 1 0.01`

6 is gpu, 1 is batch_size (can be 2 if you use only one ranking, otherwise should be 1, testing on 23GB memory);
2.5 is the total epochs, 2 is the epochs to pre-tuning the pointer (without ranking loss added);
1 and 0.01 is the loss_ratio and margin.

to quickly observe the scores and del the big model files, run the following cmds:

- `python read_results.py`
- `python del.py --file_names spiece.model,pointer.pth.tar` (usually used)
- `python observe_results.py --path EXP_dir --em_shold 1.0 --rg_shold 0.8` (used for error analysis)
- `python del.py --file_names spiece.model,predicted_examples.jsonl,pointer_choice_test.txt,pointer.pth.tar` (del all files!! including those used for observation, make sure you check to use it)

## breakpoint

currently finish "null" and "null+pos", the results can be found at `output_pointer_first_train-v2`.

the following dirs are constantly used:

`output_pointer-null_ori`;
`output_pointer-cross_rep-pred_rep`;
`output_pointer_first_train-rank_null`;
`output_null_v2` (without pointer, can used as case comparison and hyper-references)
