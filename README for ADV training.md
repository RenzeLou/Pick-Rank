## README for ADV training

1. 预训练一个classifier和一个generator（用def+x去预测y），这一步已经做好了。预训练的模型分别在：``teacher_model/pretrain_classifier_bert-base-cased_add-def``,和``POS_augmentation-num_0``下面。

2. 运行``run_s2s_adv_train.py``，可以用launch.json 运行，也可以用``adv_train``。参数都在里面。注意加上``--generate_examples``参数选项，这样的话adv training结束之后，会调用test set上的examples进行预测（不然的话还得额外再跑一次），预测的结果保存在对应的output-dir下的``predicted_examples_for_inference.jsonl``文件。

3. 进入data下，运行：

   ```bash
   python generate_silver_output_for_examples.py --tgt_data_path adv_generator_pred_examples --pred_path /home/tuq59834/code/project/Tk-ins/Tk-Instruct/output_adv/test --only_test
   ```

   这样就会把第2步中，generator预测的silver y，加到数据集里面。最终的数据集在``data/tasks/adv_generator_pred_examples``下面，所有positive examples会多一个``"output_silver"``字段。

4. 运行``run_s2s_examples_silver_out.py``,加载预训练的T5（在训练集上，用def+examples+x预测y，预训练过），利用第3步中的数据集中的``"output_silver"``，将原先的example的output替换掉，最终跑出一个性能，即，用adv generator生成的伪examples（silver y）作为instruction的性能。可以用``adv_train_pred_on_silver_examples.sh``跑，最终的结果在对应路径下的``use_silver_y_pred_performance``这个文件夹下。


### TODO

做个case study?
因为性能太差了