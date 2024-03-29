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

        sources = []  ## one batch input
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
                definitions = list(map(construct_mul_def,instance["Definition"]))
            
            # for each definition, try to add pos and neg examples
            source_list = []  ## a list stands for one instance containing both pos and neg instruction input
            for definition in definitions:
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
                
                source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
                tokenized_source = self.tokenizer(source)["input_ids"]
                # ensure the input length is not too along after encoding
                # each element in the sources is a string 
                if len(tokenized_source) <= self.max_source_length:
                    source_list.append(source)
                else:
                    source_list.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            
            sources.append(source_list)    
        
        # sources: [batch_szie,sample_num+2,seq_len]
        batch_size = len(sources)
        s_num = len(sources[0])
        # assert s_num == 1, "tis is a naive data augmentation strategy, where there is only one sample per instance."
        source_concat = [] # [batch_szie * (sample_num+2),seq_len]
        for t in sources:
            source_concat += t
            
        # to force all the input has the same length
        batch_tokenzied = self.tokenizer(
                source_concat, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        batch_input_ids,batch_attention_mask = batch_tokenzied["input_ids"],batch_tokenzied["attention_mask"]
        
        # split to a list
        model_inputs = dict()
        input_ids_list = []
        attention_mask_list = []
        for index in range(s_num):  ## TODO: should be more than one def (5 neg def), so the input_list length should be 6
            index_list = [index+t*s_num for t in range(batch_size)]
            tmp = batch_input_ids[index_list,:]
            input_ids_list.append(tmp)
            tmp_att = batch_attention_mask[index_list,:]
            attention_mask_list.append(tmp_att)
        # for index in range(0,batch_input_ids.size(0),batch_size):
        #     sample_input_ids = batch_input_ids[index:index+batch_size,:]
        #     sample_attention_mask = batch_attention_mask[index:index+batch_size,:]
        #     input_ids_list.append(sample_input_ids)
        #     attention_mask_list.append(sample_attention_mask)
        
        model_inputs["input_ids_list"] = input_ids_list
        model_inputs["attention_mask_list"] = attention_mask_list
        
        # sample_n = len(sources[0])
        # all_sources = []
        # model_inputs = dict()
        # for n in range(sample_n):
        #     batch_source_for_sample = [t[n] for t in sources]
        #     batch_tokenzied = self.tokenizer(
        #         batch_source_for_sample, 
        #         max_length=self.max_source_length, 
        #         padding=self.padding,
        #         return_tensors=self.return_tensors, 
        #         truncation=True,
        #         pad_to_multiple_of=self.pad_to_multiple_of)
        #     batch_input_ids,batch_attention_mask = batch_tokenzied["input_ids"],batch_tokenzied["attention_mask"]
        #     ## append batch_inputs / batch_attention for this sample_num
        #     if model_inputs.get("input_ids_list",None) is None:
        #         model_inputs["input_ids_list"] = [batch_input_ids]
        #     else:
        #         model_inputs["input_ids_list"].append(batch_input_ids)
        #     if model_inputs.get("attention_mask_list",None) is None:
        #         model_inputs["attention_mask_list"] = [batch_attention_mask]
        #     else:
        #         model_inputs["attention_mask_list"].append(batch_attention_mask) 
        
        assert not self.text_only, "this experiment only supports text_only"

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            labels_neg,labels_neg_one = [],[]
            labels_neg_len = []
            for ex in batch:
                neg_out_ori = ex["Instance"]["neg_output"]
                # assert len(neg_out) != 0, "the neg_output of this instance should not be an empty list, check your random seeds!"
                # if len(neg_out) == 0:
                #     wait = True
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
                    # labels = self.tokenizer(
                    #     labels,
                    #     max_length=self.max_target_length,
                    #     padding=self.padding,
                    #     return_tensors=self.return_tensors,
                    #     truncation=True,
                    #     pad_to_multiple_of=self.pad_to_multiple_of
                    # )
                    # labels_neg_one = self.tokenizer(
                    #     labels_neg_one,
                    #     max_length=self.max_target_length,
                    #     padding=self.padding,
                    #     return_tensors=self.return_tensors,
                    #     truncation=True,
                    #     pad_to_multiple_of=self.pad_to_multiple_of
                    # )
                label_mask = all_labels["attention_mask"].bool()
                all_label_ids = all_labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)  ## [batch_size + neg_out_num, max_seq_len]
                length_list = [batch_size,neg_out_num]
                labels_list = torch.split(all_label_ids,length_list,dim=0)
                labels, labels_neg_one = labels_list[0],labels_list[1]
                assert labels.shape[0] == batch_size and labels_neg_one.shape[0] == neg_out_num
                
                model_inputs["labels"] = labels
                # model_inputs["labels_neg"] = torch.split(labels_neg_one,labels_neg_len)  ## len(model_inputs["labels_neg"]) == batch_size
                # assert len(model_inputs["labels_neg"]) == batch_size
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