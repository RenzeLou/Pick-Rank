import string
import re
from tkinter.messagebox import NO
from typing import overload
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control


class NITrainer(Seq2SeqTrainer):
    def init_hyper(self,margin_pos:float,margin_neg:float,margin_null:float,
                   neg_loss_type:str,null_loss_type:str,pos_loss_type:str,
                   loss_mix_ratio_neg:float,loss_mix_ratio_pos:float,loss_mix_ratio_null:float,
                   sample_num_neg:int,sample_num_pos:int,main_loss_warm:int,pooling:str,
                   reverse:bool,lr_proj:float,lr_encoder:float=1e-5,save_pointer_choice:bool=False, 
                   predict_on_rep:bool=False, main_loss_on_rep:bool=False,save_choice_file:str=None,
                   base:bool=False,sample_times:int=1, pointer_train_epoch:int=2,
                   ranking_forbiden_on_pointer:bool=False,prob_save_file:str=None,
                   prob_save_on_rep:bool=False,prob_save_path:str="./"):
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.margin_null = margin_null
        self.margin_pos = margin_pos
        self.neg_loss_type = neg_loss_type
        self.null_loss_type = null_loss_type
        self.pos_loss_type = pos_loss_type
        self.loss_mix_ratio_neg = loss_mix_ratio_neg
        self.loss_mix_ratio_pos = loss_mix_ratio_pos
        self.loss_mix_ratio_null = loss_mix_ratio_null
        self.sample_num_neg = sample_num_neg  # deprecated
        self.sample_num_pos = sample_num_pos
        self.main_loss_warm = main_loss_warm if main_loss_warm is not None else 0
        self.pooling = pooling
        self.reverse = reverse  # deprecated, set False by default
        self.lr_proj = lr_proj
        self.lr_encoder = lr_encoder
        self.main_loss_on_rep = main_loss_on_rep 
        self.predict_on_rep = predict_on_rep
        self.save_pointer_choice = save_pointer_choice
        self.save_choice_file = save_choice_file
        self.base = base  # deprecated, set False by default
        self.sample_times = sample_times
        self.pointer_train_epoch = pointer_train_epoch
        self.ranking_forbiden_on_pointer = ranking_forbiden_on_pointer
        self.prob_save_file = prob_save_file
        self.prob_save_on_rep = prob_save_on_rep
        self.prob_save_path = prob_save_path
        
    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if self.lr_proj is not None:
            assert not self.args.deepspeed, "if you want to use self-defined lr_proj, you should not use deepspeed; otherwise it will not take effect"

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # give the whole inputs to the model
        # when doing test, all these hyperparameters are needed
        gen_kwargs = inputs.copy()
        gen_kwargs["max_length"] = self._max_length if self._max_length is not None else self.model.config.max_length
        gen_kwargs["num_beams"] = self._num_beams if self._num_beams is not None else self.model.config.num_beams
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False
       
    
        # add necessary hyperparameters for generation
        gen_kwargs["margin_neg"] = self.margin_neg
        gen_kwargs["margin_null"] = self.margin_null
        gen_kwargs["margin_pos"] = self.margin_pos
        
        gen_kwargs["loss_mix_ratio_neg"] = self.loss_mix_ratio_neg
        gen_kwargs["loss_mix_ratio_null"] = self.loss_mix_ratio_null
        gen_kwargs["loss_mix_ratio_pos"] = self.loss_mix_ratio_pos
        
        gen_kwargs["neg_loss_type"] = self.neg_loss_type
        gen_kwargs["null_loss_type"] = self.null_loss_type
        gen_kwargs["pos_loss_type"] = self.pos_loss_type
        
        # we need to assume that all the ``sample num'' are equal to 0, since we do not calculate ranking loss when doing test
        gen_kwargs["sample_num_pos"] = self.sample_num_pos
        gen_kwargs["sample_num_neg"] = self.sample_num_neg
        gen_kwargs["main_loss_warm"] = self.main_loss_warm
        gen_kwargs["current_epoch"] = self.state.epoch
        gen_kwargs["pooling"] = self.pooling
        gen_kwargs["reverse"] = self.reverse
        gen_kwargs["main_loss_on_rep"] = self.main_loss_on_rep 
        gen_kwargs["predict_on_rep"] = self.predict_on_rep,
        gen_kwargs["save_pointer_choice"] = self.save_pointer_choice
        gen_kwargs["save_choice_file"] = self.save_choice_file
        gen_kwargs["base"] = self.base
        gen_kwargs["sample_times"] = self.sample_times
        
        # directly pass the hyperparameters to the model
        model.add_gen_kwargs(gen_kwargs)
        
        # just making the API consistent with the other models
        gen_kwargs_2 = dict()
        generation_inputs = inputs["def_input_ids_list"][0]  # useless, just keep the API consistent
        gen_kwargs_2["attention_mask"] = inputs["def_attention_mask_list"][0] # useless
        gen_kwargs_2["max_length"] = self._max_length if self._max_length is not None else self.model.config.max_length
        gen_kwargs_2["num_beams"] = self._num_beams if self._num_beams is not None else self.model.config.num_beams
        gen_kwargs_2["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False
        
        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.model.generate( 
                generation_inputs,
                **gen_kwargs_2,
            )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    # add hyper-parameters
                    if self.prob_save_file is not None:
                        inputs["margin_neg"] = self.margin_neg
                        inputs["margin_null"] = self.margin_null
                        inputs["margin_pos"] = self.margin_pos
                        
                        inputs["loss_mix_ratio_neg"] = self.loss_mix_ratio_neg
                        inputs["loss_mix_ratio_null"] = self.loss_mix_ratio_null
                        inputs["loss_mix_ratio_pos"] = self.loss_mix_ratio_pos
                        
                        inputs["neg_loss_type"] = self.neg_loss_type
                        inputs["null_loss_type"] = self.null_loss_type
                        inputs["pos_loss_type"] = self.pos_loss_type
                        
                        # we need to assume the sample num all equal to 0, since we do not calculate addtional loss now
                        inputs["sample_num_pos"] = self.sample_num_pos
                        inputs["sample_num_neg"] = self.sample_num_neg
                        inputs["main_loss_warm"] = self.main_loss_warm
                        inputs["current_epoch"] = 9999999 # force doing loss calculation
                        inputs["pooling"] = self.pooling
                        inputs["reverse"] = self.reverse
                        if self.prob_save_on_rep:
                            inputs["main_loss_on_rep"] = True
                            inputs["predict_on_rep"] = True
                        else:
                            inputs["main_loss_on_rep"] = False
                            inputs["predict_on_rep"] = False
                        inputs["save_pointer_choice"] = self.save_pointer_choice
                        inputs["save_choice_file"] = self.save_choice_file
                        inputs["base"] = self.base
                        inputs["sample_times"] = self.sample_times
                        outputs = model.forward(**inputs)
                        # save the token-level probability for the further analysis
                        batched_prob,masks = model.get_logits(outputs.logits,inputs["labels"],return_mask=True)
                        batched_avg_prob = model.pooling_hidden(batched_prob,masks.squeeze(-1),pooling="mean")
                        batched_prob_list = batched_avg_prob.detach().cpu().squeeze(-1).tolist()
                        tgt_file = "{}/{}.txt".format(self.prob_save_path,self.prob_save_file)
                        with open(tgt_file,"a") as f:
                            f.writelines([str(i)+"\n" for i in batched_prob_list])
                # in order to make the testing procedure faster, we do not calculate the loss when doing test
                # if you really need the loss on the test set, please modify the ``model.forward()'' function
                loss = None
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.lr_proj is not None:
            assert not self.args.deepspeed, "if you want to use self-defined lr_proj, you should not use deepspeed; otherwise it will not take effect"
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        ## add hyper-parameters
        inputs["margin_neg"] = self.margin_neg
        inputs["margin_null"] = self.margin_null
        inputs["margin_pos"] = self.margin_pos
        
        inputs["loss_mix_ratio_neg"] = self.loss_mix_ratio_neg
        inputs["loss_mix_ratio_null"] = self.loss_mix_ratio_null
        inputs["loss_mix_ratio_pos"] = self.loss_mix_ratio_pos
        
        inputs["neg_loss_type"] = self.neg_loss_type
        inputs["null_loss_type"] = self.null_loss_type
        inputs["pos_loss_type"] = self.pos_loss_type
        
        inputs["sample_num_pos"] = self.sample_num_pos
        inputs["sample_num_neg"] = self.sample_num_neg
        inputs["main_loss_warm"] = self.main_loss_warm
        inputs["current_epoch"] = self.state.epoch
        inputs["pooling"] = self.pooling
        inputs["reverse"] = self.reverse
        inputs["main_loss_on_rep"] = self.main_loss_on_rep 
        inputs["predict_on_rep"] = self.predict_on_rep,
        inputs["save_pointer_choice"] = self.save_pointer_choice
        inputs["save_choice_file"] = self.save_choice_file
        inputs["base"] = self.base
        inputs["sample_times"] = self.sample_times
        
        def _fix_pointer(signal:bool=False):
            if model.pointer_encoder is not None:
                for param in model.pointer_encoder.parameters():
                    param.requires_grad = signal
            if model.pointer_projector is not None:
                for param in model.pointer_projector.parameters():
                    param.requires_grad = signal
        
        if self.state.epoch > self.pointer_train_epoch:
            # use the ranking loss
            if self.ranking_forbiden_on_pointer:
                # fix the parameters of pointer network when doing ranking loss
                _fix_pointer(False)
            else:
                _fix_pointer(True)
        else:
            _fix_pointer(True)  # train the pointer network
            # forbiden the ranking loss
            inputs["neg_loss_type"] = None
            inputs["null_loss_type"] = None
            inputs["pos_loss_type"] = None
            
            
        outputs = model.forward(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    # rewrite the create_optimizer function with a cutomized optimizer and learning rate for the pointer network
    # note that we can not use deepspeed when using customized optimizer
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "projector" not in n and "pointer_encoder" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "projector" not in n and "pointer_encoder" not in n],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "projector" in n],
                    "lr": self.lr_proj,
                    "weight_decay": 0.0
                }
            ]
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "pointer_encoder" in n],
                    "lr": self.lr_encoder,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "pointer_encoder" in n],
                    "lr": self.lr_encoder,
                    "weight_decay": 0.0
                }
            ]

            print([n for n, p in self.model.named_parameters() if "projector" in n])  # for debug
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer