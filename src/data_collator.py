import torch
import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    neg_out_sample_num: int = None
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        ori_def_list = []  ## a list containing tokenized tensors of definitions from different instances, their length can be different
        ori_att_list = []  
        x_tk_list = []  ## a list containing tokenized tensors of inputs from different instances, their length can be different
        x_att_list = []
        sources_def = []
        source_def_len = []
        indicators = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            # definition = ""
            def construct_mul_def(definition:str):
                definition = "Definition: " + definition.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
                
                return definition
            
            definitions = [""]
            if add_task_definition:
                assert isinstance(instance["Definition"], list), "In this experiment, we want use multiple neg samples to calculate loss!"
                # definitions = list(map(construct_mul_def,instance["Definition"]))
                definitions = instance["Definition"]
            
            # for each definition, try to add pos and neg examples
            source_def_list = []
            for i,definition in enumerate(definitions):
                # try to add positive examples.
                pos_examples = []
                for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                    pos_example_str = f" Positive Example {idx+1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                    pos_example_str += f" Output: {pos_example['output'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n" 
                    if add_explanation and "explanation" in pos_example:
                        pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                        pos_example_str += "\n"
                    pos_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        pos_examples.append(pos_example_str)
                    else:
                        break
                
                # try to add negative examples.
                neg_examples = []
                for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                    neg_example_str = f" Negative Example {idx+1} -\n"
                    neg_example_str += f"Input: {neg_example['input'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    neg_example_str += f" Output: {neg_example['output'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    if add_explanation and "explanation" in neg_example:
                        neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                        if not neg_example_str[-1] in string.punctuation:
                            neg_example_str += "."
                        neg_example_str += "\n"
                    neg_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        neg_examples.append(neg_example_str)
                    else:
                        break 
                
                # for ori def and x
                # note that we do not need padding 
                if i == 0:
                    # process ori def 
                    definition = construct_mul_def(definition) 
                    # tokenize ori def and x seperately
                    tokenized_source = self.tokenizer(definition,
                                                    max_length=self.max_source_length,
                                                    return_tensors=self.return_tensors, 
                                                    truncation=True,
                                                    pad_to_multiple_of=self.pad_to_multiple_of)
                    ori_def_tk, ori_def_att = tokenized_source["input_ids"], tokenized_source["attention_mask"]
                    assert len(ori_def_tk) <= self.max_source_length
                    ori_def_list.append(ori_def_tk)
                    ori_att_list.append(ori_def_att)
                    # process x
                    tokenized_source_x = self.tokenizer(task_input,
                                                    max_length=self.max_source_length,
                                                    return_tensors=self.return_tensors, 
                                                    truncation=True,
                                                    pad_to_multiple_of=self.pad_to_multiple_of)
                    x_tk, x_att = tokenized_source_x["input_ids"], tokenized_source_x["attention_mask"]
                    assert len(x_tk) <= self.max_source_length
                    x_tk_list.append(x_tk)
                    x_att_list.append(x_att)
                else:
                    # for each individual sentences from ori def
                    source_def = definition
                    tokenized_source_def = self.tokenizer(source_def)["input_ids"]
                    # ensure the input length is not too along after encoding
                    # each element in the sources is a string 
                    if len(tokenized_source_def) <= self.max_source_length:
                        source_def_list.append(source_def)
                    else:
                        source_def_list.append(self.tokenizer.decode(tokenized_source_def[:self.max_source_length], skip_special_tokens=True))
                    
            sources_def.append(source_def_list)
            source_def_len.append(len(source_def_list))
        
        max_def_len = max(source_def_len)
        source_def_pad = []
        # deal with the length difference of neg def of each instance
        for s_list in sources_def:
            if len(s_list) < max_def_len:
                s_list += [""]*(max_def_len - len(s_list))
            source_def_pad.append(s_list)
        sources_def = source_def_pad
        assert len(x_tk_list) == len(ori_def_list) == len(sources_def) , "should be the same length (batch_size)"
        
        batch_size = len(sources_def)
        s_num = len(sources_def[0])
        source_def_concat = [] # [batch_szie * (sample_num+2),seq_len]
        for t in sources_def:
            source_def_concat += t
        # to force all the input has the same length
        batch_def_tokenzied = self.tokenizer(
                source_def_concat, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        # this is used for the attention in the pointer network, so we need to pad and batchfy them, which is different from x_tk_list and ori_def_list
        batch_def_input_ids,batch_def_attention_mask = batch_def_tokenzied["input_ids"],batch_def_tokenzied["attention_mask"]
        
        model_inputs = dict()
        # split to a list
        def_ids_list = []
        def_attention_mask_list = []
        for index in range(s_num):
            index_list = [index+t*s_num for t in range(batch_size)]
            tmp = batch_def_input_ids[index_list,:]
            def_ids_list.append(tmp)
            tmp_att = batch_def_attention_mask[index_list,:]
            def_attention_mask_list.append(tmp_att)
        
        model_inputs["ori_def_list"] = ori_def_list 
        model_inputs["x_tk_list"] = x_tk_list
        model_inputs["def_input_ids_list"] = def_ids_list # [[batch_size,seq_len] * s_num]
        model_inputs["def_attention_mask_list"] = def_attention_mask_list
        model_inputs["def_len"] = source_def_len 
        model_inputs["rep_tk"], model_inputs["del_tk"] = self.tokenizer.additional_special_tokens_ids[0], None # special token ids
        model_inputs["rep_tk"], model_inputs["del_tk"] = torch.LongTensor([model_inputs["rep_tk"]]), None
        model_inputs["padding_token_id"] = self.tokenizer.pad_token_id
        model_inputs["tokenizer"] = self.tokenizer
        model_inputs["max_source_length"] = self.max_source_length
        
        assert not self.text_only, "this experiment doesn't use ``text_only''."

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            labels_neg,labels_neg_one = [],[]
            labels_neg_len = []
            for ex in batch:
                neg_out_ori = ex["Instance"]["neg_output"]
                if self.neg_out_sample_num is None or self.neg_out_sample_num > len(neg_out_ori):
                    sample_num = len(neg_out_ori)
                else:
                    sample_num = self.neg_out_sample_num
                neg_out = random.sample(neg_out_ori,sample_num)
                labels_neg_len.append(len(neg_out))
                labels_neg_one.extend(neg_out)
                labels_neg.append(neg_out)
            batch_size,neg_out_num = len(labels),len(labels_neg_one)
            if self.text_only:
                model_inputs["labels"] = labels
                model_inputs["labels_neg"] = labels_neg
            else:
                all_labels = labels + labels_neg_one
                with self.tokenizer.as_target_tokenizer():
                    all_labels = self.tokenizer(
                        all_labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = all_labels["attention_mask"].bool()
                all_label_ids = all_labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)  ## [batch_size + neg_out_num, max_seq_len]
                length_list = [batch_size,neg_out_num]
                labels_list = torch.split(all_label_ids,length_list,dim=0)
                labels, labels_neg_one = labels_list[0],labels_list[1]
                assert labels.shape[0] == batch_size and labels_neg_one.shape[0] == neg_out_num
                
                model_inputs["labels"] = labels
                model_inputs["labels_neg"] = labels_neg_one
                
                model_inputs["labels_neg_len"] = labels_neg_len
        else:
            model_inputs["labels"] = None
            model_inputs["labels_neg"] = None
        
        # prepare decoder_input_ids (teacher forcing)
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            if neg_out_num > 0:
                decoder_input_ids_neg = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels_neg"])
            else:
                decoder_input_ids_neg = None
            model_inputs["decoder_input_ids_neg"] = decoder_input_ids_neg
            

        return model_inputs