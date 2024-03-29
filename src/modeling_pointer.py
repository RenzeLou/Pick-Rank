# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""


from cmath import isinf, isnan
import copy
import math
from optparse import Option
import os
from unittest import result
import warnings
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from transformers.activations import ACT2FN
# from activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers import PreTrainedTokenizer
# from modeling_t5_concat import T5Stack_k_v_sep, T5Block_k_v_sep, T5Attention_k_v_sep, T5LayerCrossAttention_k_v_sep


logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)

## ignore wanring in constractive loss
# from transformers.utils import logging as hug_logging
# hug_logging.set_verbosity_info()

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    T5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass


class T5DenseReluDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration_neg, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5Model

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration_neg(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.q_linear = None
        self.k_linear = None
         

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        self.softmax = nn.Softmax(dim=1)
        
        # count the steps of obvious neg loss
        self.all_step = 0.
        self.obvious = 0.
        self.loss_record = []
        self.loss_current = []
        self.record_step = 3000
        self.current_step = 0.

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
        
    def sum_neg_loss(self,logits_ori,labels,ignore_index=-100):
        '''simply sum up the probability corresponding to the ground truth tokens'''
        gt_logits_masked = self.get_logits(logits_ori,labels,ignore_index)
        loss_neg = torch.sum(gt_logits_masked)
        
        return loss_neg
    
    def add_gen_kwargs(self, gen_kwargs:dict):
        ''' 
        Since it is hard to pass hyperparameter in the ``model.generate()'' API, use this func to ensure the correct args passing.
        note that, this function can only be called when doing testing 
        '''
        self.gen_kwargs_used = False
        self.gen_kwargs = gen_kwargs
    
    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        x_tk_list: Optional[List[torch.LongTensor]] = None,
        ori_def_list : Optional[List[torch.FloatTensor]] = None,
        def_input_ids_list: Optional[List[torch.LongTensor]] = None,
        def_attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        def_len: Optional[List[int]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids_neg: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs_ori: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs_rep: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs_del: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs_null: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        ori_past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        rep_past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        del_past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        null_past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_neg: Optional[torch.LongTensor] = None,
        labels_neg_len: Optional[List[int]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_neg_ratio: Optional[float] = None,
        margin_null: Optional[float] = None,
        margin_neg: Optional[float] = None,
        margin_pos: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        null_loss_type: Optional[str] = None,
        pos_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio_null: Optional[float] = None,
        loss_mix_ratio_neg: Optional[float] = None,
        loss_mix_ratio_pos: Optional[float] = None,
        sample_num_neg: Optional[int] = None,
        sample_num_pos: Optional[int] = None,
        main_loss_warm: Optional[int] = 0,
        current_epoch: Optional[int] = 0,
        pooling: Optional[str] = "mean",
        pooling_att: Optional[str] = "max",
        pooling_memory: Optional[str] = "mean",
        reverse: Optional[bool] = False,
        test_phase: Optional[bool] = False,
        tau: Optional[int] = 1,
        rep_tk: Optional[torch.LongTensor] = None,
        del_tk: Optional[torch.LongTensor] = None,
        padding_token_id: Optional[int] = 0,
        main_loss_on_rep: Optional[bool] = False,
        predict_on_rep: Optional[bool] = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        save_pointer_choice: Optional[bool] = False,
        save_choice_file: Optional[str] = None,
        attention_mask: Optional[List[torch.FloatTensor]] = None,
        max_source_length: Optional[int] = 1024,
        base: Optional[bool] = False,
        sample_times: Optional[int] = 1,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        ''' 
        calculate the cross_entropy loss and ranking loss for the given batch
        this function can be used for both training and testing
        '''
        if ori_def_list is None and x_tk_list is None and def_input_ids_list is None and def_attention_mask_list is None:
            # when doing test, all the hyper-parameters are lost through the ``.generate()'' API
            # we use it as a signal to use the pre-defined hyper-parameters passed by ``.add_gen_kwargs()''
            # another optional way is to rewrite a customized ``.generate()'' and modify the ``self.prepare_inputs_for_generation()'', however, it is inconvenient
            assert test_phase, "this behavious should only happen in testing phase"
            ori_def_list = self.gen_kwargs["ori_def_list"]
            x_tk_list = self.gen_kwargs["x_tk_list"]
            def_input_ids_list = self.gen_kwargs["def_input_ids_list"]
            def_attention_mask_list = self.gen_kwargs["def_attention_mask_list"]
            def_len = self.gen_kwargs["def_len"]
            pooling = self.gen_kwargs["pooling"]
            pooling_att = self.gen_kwargs.get("pooling_att",None)
            pooling_memory = self.gen_kwargs.get("pooling_memory",None)
            neg_loss_type = self.gen_kwargs.get("neg_loss_type", None)
            null_loss_type = self.gen_kwargs.get("null_loss_type", None)
            pos_loss_type = self.gen_kwargs.get("pos_loss_type", None)
            neg_loss_only = self.gen_kwargs.get("neg_loss_only", False)
            sample_num_pos = self.gen_kwargs.get("sample_num_pos", None)
            sample_num_neg = self.gen_kwargs.get("sample_num_neg", None)
            current_epoch = self.gen_kwargs.get("current_epoch", 0)
            encoder_outputs_ori = self.gen_kwargs.get("encoder_outputs_ori", None)
            ori_def_x_att = self.gen_kwargs.get("ori_def_x_att", None)
            encoder_outputs_rep = self.gen_kwargs.get("encoder_outputs_rep", None)
            def_rep_att = self.gen_kwargs.get("def_rep_att", None)
            encoder_outputs_del = self.gen_kwargs.get("encoder_outputs_del", None)
            def_del_att = self.gen_kwargs.get("def_del_att", None)
            encoder_outputs_null = self.gen_kwargs.get("encoder_outputs_null", None)
            def_null_att = self.gen_kwargs.get("def_null_att", None)
            reverse = self.gen_kwargs.get("reverse", False)
            rep_tk = self.gen_kwargs.get("rep_tk", None)
            del_tk = self.gen_kwargs.get("del_tk", None)
            padding_token_id = self.gen_kwargs.get("padding_token_id", 0)
            main_loss_on_rep = self.gen_kwargs.get("main_loss_on_rep", False)
            predict_on_rep = self.gen_kwargs.get("predict_on_rep", False)
            tokenizer = self.gen_kwargs.get("tokenizer", None)
            save_pointer_choice = self.gen_kwargs.get("save_pointer_choice", False)
            save_choice_file = self.gen_kwargs.get("save_choice_file", None)
            max_source_length = self.gen_kwargs.get("max_source_length", 1024)
            base = self.gen_kwargs.get("base", False)
            sample_times = self.gen_kwargs.get("sample_times", 1)
            ori_past_key_values = self.gen_kwargs.get("ori_past_key_values",None)
            rep_past_key_values = self.gen_kwargs.get("rep_past_key_values",None)
            del_past_key_values = self.gen_kwargs.get("del_past_key_values",None)
            null_past_key_values = self.gen_kwargs.get("null_past_key_values",None)
            test_phase = True
            
            self.gen_kwargs_used = True  # for debugging
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # ensure the data type
        predict_on_rep = predict_on_rep[0] if isinstance(predict_on_rep,tuple) else predict_on_rep

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
                
        neg_num = len(def_input_ids_list) ## the number of individual sentences
        batch_size = len(x_tk_list)
        max_seq_list = [sample.size(1) for sample in def_input_ids_list]
        max_seq_len = max(max_seq_list)    
        assert len(set(max_seq_list)) == 1, "we need to batchfy the input, thus the max_seq_len of each instances mush be the same!"
        
        ''' 1. use pointer network to get the attention scores among the different sentences '''
        def_input_ids = torch.cat(def_input_ids_list,dim=0) if neg_num > 0 else None ## [batch_size * neg_num, max_seq_len]
        def_attention_mask = torch.cat(def_attention_mask_list,dim=0) if neg_num > 0 else None
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs_ori is None and encoder_outputs_rep is None and encoder_outputs_del is None and encoder_outputs_null is None:
            # assert neg_num > 0, "at leat one def, set sample_num_pos > 0 to overcome"
            # get the hidden states of def
            def_encoder_outputs = self.pointer_encoder(
                    input_ids=def_input_ids,
                    attention_mask=def_attention_mask,
                    inputs_embeds=None,
                    head_mask=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
            )
            def_hidden_states = def_encoder_outputs[0]  # [batch_size * neg_num, max_seq_len, hidden_size]
            
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states = hidden_states.to(self.decoder.first_device)
                def_hidden_states = def_hidden_states.to(self.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.decoder.first_device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            
            # pooling the hidden_states, apply pointer network to get scores
            def_hidden_pooled = self.pooling_hidden(def_hidden_states,def_attention_mask,pooling) # [batch_size * neg_num, memo_dim]
            def_for_each_ins = []
            def_input_ids_for_each_ins,def_attention_mask_for_each_ins = [],[] # used for pos and neg def generation
            for index in range(batch_size):
                index_list = [index+t*batch_size for t in range(neg_num)]
                def_for_each_ins.append(def_hidden_pooled[index_list,:])
                def_input_ids_for_each_ins.append(def_input_ids[index_list,:])
                def_attention_mask_for_each_ins.append(def_attention_mask[index_list,:])
            def_input_ids_for_each_ins = torch.cat(def_input_ids_for_each_ins,dim=0) # [batch_size * neg_num, max_seq_len]
            def_attention_mask_for_each_ins = torch.cat(def_attention_mask_for_each_ins,dim=0) # [batch_size * neg_num, max_seq_len]
            def_hidden_pooled = torch.cat(def_for_each_ins,dim=0)  # [batch_size * neg_num, memo_dim]
            def_logits = self.pointer_projector(def_hidden_pooled) # [batch_size * neg_num, 1]
            scores = def_logits.reshape(batch_size,neg_num) # [batch_size, neg_num]
            # note the def padding
            score_for_each_ins = torch.split(scores,1,dim=0)
            assert len(def_len) == batch_size == len(score_for_each_ins)
            weights = []
            for s,neg_l in zip(score_for_each_ins,def_len):
                # sample mutiple times, use 'OR' to combine the results
                weight_combined = torch.zeros_like(s)  # [1, neg_num]
                for i in range(sample_times):
                    weight = F.gumbel_softmax(s[:,:neg_l], tau=tau, hard=True,dim=-1) 
                    padding = torch.zeros_like(s[:,neg_l:])
                    weight = torch.cat([weight,padding],dim=1)  # [1, neg_num]
                    weight_combined = weight_combined + weight
                weights.append(weight_combined.clamp_max(1.0)) # [1, neg_num]
            weights = torch.cat(weights,dim=0)  # [batch_size, neg_num]
            weights = weights.reshape(-1,1) # [batch_size * neg_num, 1]
            def_atts = weights.repeat(1,def_input_ids.size(1)) # [batch_size * neg_num, max_seq_len]
            
            # save the attention scores (choices) on the test set for observation
            if save_pointer_choice and test_phase:
                assert save_choice_file is not None
                batch_choices = weights.detach().cpu().squeeze().tolist()
                batch_defs = def_input_ids_for_each_ins.detach().cpu().tolist()
                batch_defs = tokenizer.batch_decode(batch_defs, skip_special_tokens=True)
                batch_choices = [batch_choices] if not isinstance(batch_choices,list) else batch_choices
                batch_defs = [batch_defs] if not isinstance(batch_defs,list) else batch_defs
                with open(save_choice_file,"a") as f:
                    for b in range(batch_size):
                        choice = batch_choices[b*neg_num:(b+1)*neg_num]
                        defs = batch_defs[b*neg_num:(b+1)*neg_num]
                        f.write(f"{str(choice)}\t{str(defs)}\n")
                        
            # create attention mask according to the def_att scores
            rep_tk = rep_tk.reshape(1,1) if len(rep_tk.shape) < 2 else rep_tk
            # del_tk = del_tk.reshape(1,1) if len(del_tk.shape) < 2 else del_tk
            def_attention_masks_ = def_attention_mask_for_each_ins.mul(def_atts) # [batch_size * neg_num, max_seq_len]
            def_attention_masks_ = def_attention_masks_.reshape(batch_size,-1) # [batch_size, neg_num * max_seq_len]
            def_attention_masks_reverse = def_attention_mask_for_each_ins.mul(1-def_atts) # [batch_size * neg_num, max_seq_len]
            def_attention_masks_reverse = def_attention_masks_reverse.reshape(batch_size,-1) # [batch_size, neg_num * max_seq_len]
            def_attention_masks_reverse = def_attention_masks_reverse.detach()
            def_input_ids_ = def_input_ids_for_each_ins.reshape(batch_size,-1) # [batch_size, neg_num * max_seq_len]
            
            ''' 2. construct the pos and neg samples (i.e., repeated and deleted) '''
            # prepare pos def (repeated) + x
            if (not test_phase and (pos_loss_type is not None or main_loss_on_rep)) or (test_phase and predict_on_rep):
                def_rep_list = []
                att_rep_list = []  # containing a batch of samples 
                for i in range(batch_size):
                    temp_ori = ori_def_list[i]
                    temp_x = x_tk_list[i]
                    att_pre = torch.ones(temp_ori.shape[0],temp_ori.shape[1]+1, dtype=torch.float).to(temp_ori.device)
                    att_suf = torch.ones(temp_x.shape[0],temp_x.shape[1]+1, dtype=torch.float).to(temp_x.device)
                    att_rep = torch.cat([att_pre,def_attention_masks_[i,:].unsqueeze(0),att_suf],dim=-1)
                    def_rep = torch.cat([temp_ori,rep_tk,def_input_ids_[i,:].unsqueeze(0),rep_tk,temp_x],dim=-1)
                    assert def_rep.shape == att_rep.shape
                    att_rep_list.append(att_rep)
                    def_rep_list.append(def_rep)
                def_rep_ids, def_rep_att = self.padding_and_mask(def_rep_list,att_rep_list,padding_token_id=padding_token_id,max_source_length=max_source_length)
            else:
                def_rep_ids, def_rep_att = None, None
            # prepare neg def (deleted) + x
            if not test_phase and neg_loss_type is not None:
                def_del_list = []
                att_del_list = []  # containing a batch of samples 
                for i in range(batch_size):
                    temp_x = x_tk_list[i]
                    att_suf = torch.ones(temp_x.shape, dtype=torch.float).to(temp_x.device)
                    att_del = torch.cat([def_attention_masks_reverse[i,:].unsqueeze(0),att_suf],dim=-1)
                    def_del = torch.cat([def_input_ids_[i,:].unsqueeze(0),temp_x],dim=-1)
                    assert def_del.shape == att_del.shape
                    att_del_list.append(att_del)
                    def_del_list.append(def_del)
                def_del_ids, def_del_att = self.padding_and_mask(def_del_list,att_del_list,padding_token_id=padding_token_id,max_source_length=max_source_length)
            else:
                def_del_ids, def_del_att = None, None
            # prepare null def (only x)
            if not test_phase and null_loss_type is not None:
                def_null_list = []
                att_null_list = []  # containing a batch of samples 
                for i in range(batch_size):
                    temp_x = x_tk_list[i]
                    att_null = torch.ones(temp_x.shape[0],temp_x.shape[1], dtype=torch.float).to(temp_x.device)
                    def_null = torch.cat([temp_x],dim=-1)
                    assert def_null.shape == att_null.shape
                    att_null_list.append(att_null)
                    def_null_list.append(def_null)
                def_null_ids, def_null_att = self.padding_and_mask(def_null_list,att_null_list,padding_token_id=padding_token_id,max_source_length=max_source_length)
            else:
                def_null_ids, def_null_att = None, None
            
            # prepare ori def + x
            if (not test_phase and (pos_loss_type is not None or not main_loss_on_rep)) or (test_phase and not predict_on_rep):
                assert len(x_tk_list) == len(ori_def_list)
                ori_def_x_list = []
                for x,ori_def in zip(x_tk_list,ori_def_list):
                    temp = torch.cat([ori_def,x],dim=-1)
                    ori_def_x_list.append(temp)
                ori_def_x_ids, ori_def_x_att = self.padding_and_mask(ori_def_x_list,padding_token_id=padding_token_id,max_source_length=max_source_length)
            else:
                ori_def_x_ids, ori_def_x_att = None, None
            
                
            ''' 3. encoding and decodinig, get the logits '''
            # Convert encoder inputs in embeddings if needed
            # indicating it is a training procedure
            # or, it is the first prediction pass
            if ori_def_x_ids is not None:
                encoder_outputs_ori = self.encoder(
                    input_ids=ori_def_x_ids,
                    attention_mask=ori_def_x_att,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = encoder_outputs_ori[0]  # [batch_size, max_seq_len_1, hidden_size]
            else:
                encoder_outputs_ori = None
                hidden_states = None
            
            if def_rep_ids is not None:
                encoder_outputs_rep = self.encoder(
                    input_ids=def_rep_ids,
                    attention_mask=def_rep_att,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states_rep = encoder_outputs_rep[0]  # [batch_size, max_seq_len_2, hidden_size]
            else:
                encoder_outputs_rep = None
                hidden_states_rep = None
            
            if def_del_ids is not None:
                encoder_outputs_del = self.encoder(
                    input_ids=def_del_ids,
                    attention_mask=def_del_att,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states_del = encoder_outputs_del[0]  # [batch_size, max_seq_len_3, hidden_size]
            else:
                encoder_outputs_del = None
                hidden_states_del = None
            
            if def_null_ids is not None:
                encoder_outputs_null = self.encoder(
                    input_ids=def_null_ids,
                    attention_mask=def_null_att,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states_null = encoder_outputs_null[0]  # [batch_size, max_seq_len_4, hidden_size]
            else:
                encoder_outputs_null = None
                hidden_states_null = None

            # save the hidden states for the next prediction pass
            if test_phase:
                encoder_outputs_saved_ori = [hidden_states] if hidden_states is not None else None
                encoder_outputs_saved_rep = [hidden_states_rep] if hidden_states_rep is not None else None
                encoder_outputs_saved_del = [hidden_states_del] if hidden_states_del is not None else None
                encoder_outputs_saved_null = [hidden_states_null] if hidden_states_null is not None else None
                self.gen_kwargs["encoder_outputs_ori"] = encoder_outputs_saved_ori
                self.gen_kwargs["ori_def_x_att"] = ori_def_x_att
                self.gen_kwargs["encoder_outputs_rep"] = encoder_outputs_saved_rep
                self.gen_kwargs["def_rep_att"] = def_rep_att
                self.gen_kwargs["encoder_outputs_del"] = encoder_outputs_saved_del
                self.gen_kwargs["def_del_att"] = def_del_att
                self.gen_kwargs["encoder_outputs_null"] = encoder_outputs_saved_null
                self.gen_kwargs["def_null_att"] = def_null_att
        # re-use the hidden states for the next prediction pass
        else:
            encoder_outputs_ori = BaseModelOutput(
                last_hidden_state=encoder_outputs_ori[0],
                hidden_states=encoder_outputs_ori[1] if len(encoder_outputs_ori) > 1 else None,
                attentions=encoder_outputs_ori[2] if len(encoder_outputs_ori) > 2 else None,
            ) if encoder_outputs_ori is not None else None
            encoder_outputs_rep = BaseModelOutput(
                last_hidden_state=encoder_outputs_rep[0],
                hidden_states=encoder_outputs_rep[1] if len(encoder_outputs_rep) > 1 else None,
                attentions=encoder_outputs_rep[2] if len(encoder_outputs_rep) > 2 else None,
            ) if encoder_outputs_rep is not None else None
            encoder_outputs_del = BaseModelOutput(
                last_hidden_state=encoder_outputs_del[0],
                hidden_states=encoder_outputs_del[1] if len(encoder_outputs_del) > 1 else None,
                attentions=encoder_outputs_del[2] if len(encoder_outputs_del) > 2 else None,
            ) if encoder_outputs_del is not None else None
            encoder_outputs_null = BaseModelOutput(
                last_hidden_state=encoder_outputs_null[0],
                hidden_states=encoder_outputs_null[1] if len(encoder_outputs_null) > 1 else None,
                attentions=encoder_outputs_null[2] if len(encoder_outputs_null) > 2 else None,
            ) if encoder_outputs_null is not None else None
            
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states = hidden_states.to(self.decoder.first_device)
                hidden_states_rep = hidden_states_rep.to(self.decoder.first_device)
                hidden_states_del = hidden_states_del.to(self.decoder.first_device)
                hidden_states_null = hidden_states_null.to(self.decoder.first_device)
                def_hidden_states = def_hidden_states.to(self.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.decoder.first_device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
                    
            # get the hidden states directly
            hidden_states = encoder_outputs_ori[0] if encoder_outputs_ori is not None else None
            hidden_states_rep = encoder_outputs_rep[0] if encoder_outputs_rep is not None else None
            hidden_states_del = encoder_outputs_del[0] if encoder_outputs_del is not None else None
            hidden_states_null = encoder_outputs_null[0] if encoder_outputs_null is not None else None
       
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            hidden_states_rep = hidden_states_rep.to(self.decoder.first_device)
            hidden_states_del = hidden_states_del.to(self.decoder.first_device)
            hidden_states_null = hidden_states_null.to(self.decoder.first_device)
            def_hidden_states = def_hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        # Decode
        if hidden_states is not None:
            decoder_outputs_ori = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=ori_past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=ori_def_x_att,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_ori = decoder_outputs_ori[0]
        else:
            decoder_outputs_ori = None
            sequence_output_ori = None
        
        if hidden_states_rep is not None:
            decoder_outputs_rep = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=rep_past_key_values,
                encoder_hidden_states=hidden_states_rep,
                encoder_attention_mask=def_rep_att,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_rep = decoder_outputs_rep[0]
        else:
            decoder_outputs_rep = None
            sequence_output_rep = None
        
        if hidden_states_del is not None:
            decoder_outputs_del = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=del_past_key_values,
                encoder_hidden_states=hidden_states_del,
                encoder_attention_mask=def_del_att,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_del = decoder_outputs_del[0]
        else:
            decoder_outputs_del = None
            sequence_output_del = None
            
        if hidden_states_null is not None:
            decoder_outputs_null = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=null_past_key_values,
                encoder_hidden_states=hidden_states_null,
                encoder_attention_mask=def_null_att,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_null = decoder_outputs_null[0]
        else:
            decoder_outputs_null = None
            sequence_output_null = None

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output_ori = sequence_output_ori.to(self.lm_head.weight.device) if sequence_output_ori is not None else None
            sequence_output_rep = sequence_output_rep.to(self.lm_head.weight.device) if sequence_output_rep is not None else None
            sequence_output_del = sequence_output_del.to(self.lm_head.weight.device) if sequence_output_del is not None else None
            sequence_output_null = sequence_output_null.to(self.lm_head.weight.device) if sequence_output_null is not None else None
            
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output_ori = sequence_output_ori * (self.model_dim**-0.5) if sequence_output_ori is not None else None
            sequence_output_rep = sequence_output_rep * (self.model_dim**-0.5) if sequence_output_rep is not None else None
            sequence_output_del = sequence_output_del * (self.model_dim**-0.5) if sequence_output_del is not None else None
            sequence_output_null = sequence_output_null * (self.model_dim**-0.5) if sequence_output_null is not None else None

        lm_logits_ori = self.lm_head(sequence_output_ori) if sequence_output_ori is not None else None  ## [batch_size, max_seq_length, vocab_size]
        lm_logits_rep = self.lm_head(sequence_output_rep) if sequence_output_rep is not None else None
        lm_logits_del = self.lm_head(sequence_output_del) if sequence_output_del is not None else None
        lm_logits_null = self.lm_head(sequence_output_null) if sequence_output_null is not None else None
        
        ''' 4. loss '''
        loss = None
        if labels is not None:
            loss_fct_ori = CrossEntropyLoss(ignore_index=-100)
            if not main_loss_on_rep:
                assert lm_logits_ori is not None
                # calculate cross_entropy_loss on the original definition
                loss_ori = loss_fct_ori(lm_logits_ori.view(-1, lm_logits_ori.size(-1)), labels.view(-1))
            else:
                assert lm_logits_rep is not None
                # calculate cross_entropy_loss on the repeated definition
                loss_ori = loss_fct_ori(lm_logits_rep.view(-1, lm_logits_rep.size(-1)), labels.view(-1))
            
            # ranking with repeated definitions (pos) 
            loss_pos = None
            if pos_loss_type is not None:
                if pos_loss_type == "contrastive_loss_repeat":
                    loss_fct_pos = self.contrastive_loss_repeat
                elif pos_loss_type == "contrastive_loss_repeat_v2":
                    loss_fct_pos = self.contrastive_loss_repeat_v2
                else:
                    raise NotImplementedError("Error neg loss function type!")
                if current_epoch >= main_loss_warm and lm_logits_rep is not None and lm_logits_ori is not None:
                    def_len_for_loss = [1] * batch_size
                    loss_pos = loss_fct_pos(lm_logits_ori,lm_logits_rep,labels,labels,margin=margin_pos,ignore_index=-100,batch_size=batch_size,def_len=def_len_for_loss,max_seq_len=lm_logits_ori.size(1))     
                    if torch.isnan(loss_pos).any().item():
                        raise RuntimeError("the loss is nan!")
                    if torch.isinf(loss_pos).any().item():
                        raise RuntimeError("the loss is inf!")

            # ranking with deleted definitions (neg) 
            loss_neg = None
            if neg_loss_type is not None:
                if neg_loss_type == "contrastive_loss_max_v2":
                    loss_fct_neg = self.contrastive_loss_max_v2
                elif neg_loss_type == "contrastive_loss_all":
                    loss_fct_neg = self.contrastive_loss_all
                elif neg_loss_type == "contrastive_loss_softmax":
                    loss_fct_neg = self.contrastive_loss_softmax
                elif neg_loss_type == "contrastive_loss_max_v3":
                    loss_fct_neg = self.contrastive_loss_max_v3
                elif neg_loss_type == "contrastive_loss_max_v4":
                    loss_fct_neg = self.contrastive_loss_max_v4
                elif neg_loss_type == "contrastive_loss_max_v5":
                    loss_fct_neg = self.contrastive_loss_max_v5 
                elif neg_loss_type == "contrastive_loss_max_v6":
                    loss_fct_neg = self.contrastive_loss_max_v6
                else:
                    raise NotImplementedError("Error neg loss function type!")
                if current_epoch >= main_loss_warm and lm_logits_del is not None and lm_logits_rep is not None:
                    # loss_neg = loss_fct_neg(lm_logits_ori,lm_logits_del,labels,labels,margin=margin_neg,ignore_index=-100,batch_size=batch_size)     
                    loss_neg = loss_fct_neg(lm_logits_rep,lm_logits_del,labels,labels,margin=margin_neg,ignore_index=-100,batch_size=batch_size)     
                    if torch.isnan(loss_neg).any().item():
                        raise RuntimeError("the loss is nan!")
                    if torch.isinf(loss_neg).any().item():
                        raise RuntimeError("the loss is inf!")
                    
            # ranking with null (neg)
            loss_null = None
            if null_loss_type is not None:
                if null_loss_type == "contrastive_loss_max_v2":
                    loss_fct_null = self.contrastive_loss_max_v2
                elif null_loss_type == "contrastive_loss_all":
                    loss_fct_null = self.contrastive_loss_all
                elif null_loss_type == "contrastive_loss_softmax":
                    loss_fct_null = self.contrastive_loss_softmax
                elif null_loss_type == "contrastive_loss_max_v3":
                    loss_fct_null = self.contrastive_loss_max_v3
                elif null_loss_type == "contrastive_loss_max_v4":
                    loss_fct_null = self.contrastive_loss_max_v4
                elif null_loss_type == "contrastive_loss_max_v5":
                    loss_fct_null = self.contrastive_loss_max_v5
                elif null_loss_type == "contrastive_loss_max_v6":
                    loss_fct_null = self.contrastive_loss_max_v6
                else:
                    raise NotImplementedError("Error neg loss function type!")
                if current_epoch >= main_loss_warm and lm_logits_null is not None and lm_logits_rep is not None:
                    loss_null = loss_fct_null(lm_logits_rep,lm_logits_null,labels,labels,margin=margin_null,ignore_index=-100,batch_size=batch_size)      
                    if torch.isnan(loss_null).any().item():
                        raise RuntimeError("the loss is nan!")
                    if torch.isinf(loss_null).any().item():
                        raise RuntimeError("the loss is inf!")
                    
            # overall loss
            loss_addition = 0.
            loss_addition += loss_pos.clamp(min=0.0) * loss_mix_ratio_pos if loss_pos is not None else 0.
            loss_addition += loss_neg.clamp(min=0.0) * loss_mix_ratio_neg if loss_neg is not None else 0.
            loss_addition += loss_null.clamp(min=0.0) * loss_mix_ratio_null if loss_null is not None else 0.
            
            assert loss_mix_ratio_neg >= 0.0 and loss_mix_ratio_null >= 0.0 and loss_mix_ratio_pos >= 0.0
            
            loss = loss_ori + loss_addition  
            
            if current_epoch >= main_loss_warm:
                self.all_step += 1
                # for observation
                if (loss != loss_ori).item():
                    self.obvious += 1
            
            # save loss for observation
            self.current_step += batch_size
            self.loss_current.append(loss.item())
            if self.current_step >= self.record_step:
                avg_loss = np.mean(self.loss_current)
                self.loss_record.append(avg_loss)
                self.current_step = 0
        
        ''' 5. return hidden states for the sampling of prediction, note that we use hidden_rep instead of hidden_ori ''' 
        if predict_on_rep:
            assert lm_logits_rep is not None and decoder_outputs_rep is not None and encoder_outputs_rep is not None
            lm_logits_return = lm_logits_rep
            decoder_outputs_return = decoder_outputs_rep
            encoder_outputs_return = encoder_outputs_rep
        else:
            assert lm_logits_ori is not None and decoder_outputs_ori is not None and encoder_outputs_ori is not None
            lm_logits_return = lm_logits_ori
            decoder_outputs_return = decoder_outputs_ori
            encoder_outputs_return = encoder_outputs_ori
            
        ''' 6. update past_key_values for the next step (useful in testing) '''
        if test_phase:
            self.gen_kwargs["ori_past_key_values"] = decoder_outputs_ori.past_key_values if decoder_outputs_ori is not None else None
            self.gen_kwargs["rep_past_key_values"] = decoder_outputs_rep.past_key_values if decoder_outputs_rep is not None else None
            self.gen_kwargs["del_past_key_values"] = decoder_outputs_del.past_key_values if decoder_outputs_del is not None else None
            self.gen_kwargs["null_past_key_values"] = decoder_outputs_null.past_key_values if decoder_outputs_null is not None else None
        
        if not return_dict:
            # only return pos logits
            output = (lm_logits_return,) + decoder_outputs_return[1:] + encoder_outputs_return
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_return,
            past_key_values=decoder_outputs_return.past_key_values,
            decoder_hidden_states=decoder_outputs_return.hidden_states,
            decoder_attentions=decoder_outputs_return.attentions,
            cross_attentions=decoder_outputs_return.cross_attentions,
            encoder_last_hidden_state=encoder_outputs_return[0] if isinstance(encoder_outputs_return,list) else encoder_outputs_return.last_hidden_state,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
    
    def padding_and_mask(self,input_ids_list:list,attention_mask_list:list=None,padding_token_id:int=0,max_source_length:int=1024):
        ''' padding and batchfy the input_ids and attention_mask, which can be directly used in the model '''
        if attention_mask_list is None:
            attention_mask_list = [torch.ones_like(input_ids).float() for input_ids in input_ids_list]
        max_seq_length = max([input_ids.size(1) for input_ids in input_ids_list])
        assert len(input_ids_list) == len(attention_mask_list)
        
        new_input_ids_list = []
        new_attention_mask_list = []
        for input_ids,attention_mask in zip(input_ids_list,attention_mask_list):
            assert input_ids.size(1) == attention_mask.size(1)
            if input_ids.size(1) < max_seq_length:
                padding_ids = torch.ones(input_ids.size(0),max_seq_length-input_ids.size(1)).to(input_ids.device).type_as(input_ids) * padding_token_id
                padding_att = torch.zeros(attention_mask.size(0),max_seq_length-attention_mask.size(1)).to(attention_mask.device).type_as(attention_mask)
                input_ids_ = torch.cat([input_ids,padding_ids],dim=1)
                attention_mask_ = torch.cat([attention_mask,padding_att],dim=1)
            else:
                input_ids_ = input_ids
                attention_mask_ = attention_mask
            new_input_ids_list.append(input_ids_)
            new_attention_mask_list.append(attention_mask_)
        
        input_ids_batchfied = torch.cat(new_input_ids_list,dim=0)
        attention_mask_batchfied = torch.cat(new_attention_mask_list,dim=0)  ## [batch_size, max_seq_length]
        
        # ensure the max_seq_length is not too long
        if input_ids_batchfied.size(1) > max_source_length:
            input_ids_batchfied = input_ids_batchfied[:,:max_source_length]
            attention_mask_batchfied = attention_mask_batchfied[:,:max_source_length]
        
        return input_ids_batchfied, attention_mask_batchfied
                
                
    def pooling_hidden(self,hidden_states,attention_mask,pooling:str):
        ''' pooling the hidden states and ignore the padding tokens
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        return: [batch_size, hidden_size]
        '''
        assert pooling in ["max","mean"]
        if pooling == "mean":
            mask = attention_mask.type_as(hidden_states).unsqueeze(-1)  # [batch_size * (1+neg_num), max_seq_length, 1]
            mask = mask.repeat(1,1,hidden_states.size(-1))  # [batch_size * (1+neg_num), max_seq_length, hidden_size]
            hidden_states_masked = hidden_states * mask
            # sum and divide mannually to avoid the effect of the padding tokens
            hidden_summed = torch.sum(hidden_states_masked,dim=1)  # [batch_size * (1+neg_num), hidden_size]
            mask_summed = torch.sum(mask,dim=1)  # [batch_size * (1+neg_num), hidden_size]
            mask_summed = mask_summed.clamp(min=1e-9) # to avoid division by zero
            hidden_pooled = hidden_summed / mask_summed  # [batch_size * (1+neg_num), hidden_size]
        elif pooling == "max":
            INF = -999999
            mask = (1-attention_mask).type_as(hidden_states).unsqueeze(-1)  # [batch_size * (1+neg_num), max_seq_length, 1]
            mask = mask * INF
            mask = mask.repeat(1,1,hidden_states.size(-1))  # [batch_size * (1+neg_num), max_seq_length, hidden_size]
            hidden_states_masked = hidden_states + mask
            hidden_pooled = torch.max(hidden_states_masked,dim=1)[0]  # [batch_size * (1+neg_num), hidden_size]
        
        return hidden_pooled
    
    def memory_bank(self, def_hidden_states, def_attention_mask, batch_size, neg_num,pooling="max"):
        ''' get the the association between the ori definition and the purturbated definition
        def_hidden_states: [batch_size * (1+neg_num), max_seq_len, hidden_size]
        '''
        def_hidden_pooled = self.pooling_hidden(def_hidden_states,def_attention_mask,pooling) # [batch_size * (1+neg_num), hidden_size]
        def_hidden_pooled_lis = torch.split(def_hidden_pooled,[batch_size,neg_num*batch_size],dim=0)
        def_hidden_ori = def_hidden_pooled_lis[0]  # [batch_size, hidden_size]
        def_hidden_ori = def_hidden_ori.repeat(neg_num,1) # [neg_num * batch_size, hidden_size]
        def_hidden_neg = def_hidden_pooled_lis[1]  # [neg_num * batch_size, hidden_size]
        
        # compute the assosiation between the ori definition and the purturbated definition
        diff = def_hidden_ori - def_hidden_neg  # [neg_num * batch_size, hidden_size]
        dot = torch.mul(def_hidden_ori,def_hidden_neg)  # [neg_num * batch_size, hidden_size]
        association = torch.cat([def_hidden_neg,diff,dot],dim=1)  # [neg_num * batch_size, 3*hidden_size]
        
        memories = self.memory_projector(association) if self.memory_projector is not None else association  # [neg_num * batch_size, memory_dim]
        
        return memories
        
    def get_att_weights(self,hidden_states,attention_mask,batch_size,neg_num,pooling="max",reverse=False,neg_len=None):
        '''
        hidden_states: [batch_size * (1+neg_num), max_seq_length, hidden_size]
        attention_mask:  [batch_size * (1+neg_num), max_seq_length]
        '''
        hidden_pooled = self.pooling_hidden(hidden_states,attention_mask,pooling)
        # split the pooled hidden states
        # get the ori hidden (q) and neg hidden (v)
        hidden_pooled_lis = torch.split(hidden_pooled,[batch_size,neg_num*batch_size],dim=0)
        hidden_ori = hidden_pooled_lis[0]  # [batch_size, hidden_size]
        q_vecotrs = self.q_projector(hidden_ori) if self.q_projector is not None else hidden_ori # [batch_size, q_dim]
        
        # get each neg hidden (v)
        hidden_neg = hidden_pooled_lis[1]  # [neg_num*batch_size, hidden_size]
        hidden_neg_mapped = self.k_projector(hidden_neg) if self.k_projector is not None else hidden_neg # [neg_num*batch_size, k_dim]
        hidden_neg_lis = torch.split(hidden_neg_mapped,batch_size,dim=0)
        assert len(hidden_neg_lis) == neg_num
        k_vectors = torch.stack(hidden_neg_lis,dim=2)  # [batch_size, k_dim, neg_num]
        
        if len(q_vecotrs.shape) == 2:
            q_vecotrs = q_vecotrs.unsqueeze(1)  # [batch_size, 1, q_dim]
        
        score = torch.matmul(q_vecotrs,k_vectors)  # [batch_size, 1, neg_num]
        score = score.squeeze(1)  # [batch_size, neg_num] 
        
        softmax = torch.nn.Softmax(dim=1)
        if neg_len is None:
            weights = softmax(score)  # [batch_size, neg_num]
            if reverse:
                weights = 1.0 - weights + 1e-10
        else:
            # need to ignore some padding neg samples
            score_for_each_ins = torch.split(score,1,dim=0)
            assert len(neg_len) == batch_size == len(score_for_each_ins)
            weights = []
            for s,neg_l in zip(score_for_each_ins,neg_len):
                weight = softmax(s[:,:neg_l])
                if reverse:
                    weight = 1.0 - weight + 1e-10 
                padding = torch.zeros_like(s[:,neg_l:])
                weight = torch.cat([weight,padding],dim=1)  # [1, neg_num]
                weights.append(weight)
            weights = torch.cat(weights,dim=0)  # [batch_size, neg_num]
         
        print(weights)  # to observe 
        
        return weights
    
    def get_att_weights_linear(self,hidden_states,attention_mask,batch_size,neg_num,pooling="max",reverse=False,neg_len=None):
        '''
        hidden_states: [batch_size * neg_num, max_seq_length, hidden_size]
        attention_mask:  [batch_size * neg_num, max_seq_length]
        '''
        hidden_pooled = self.pooling_hidden(hidden_states,attention_mask,pooling)
        
        # get each neg hidden (v)
        hidden_neg = hidden_pooled  # [neg_num*batch_size, hidden_size]
        hidden_neg_mapped = self.k_projector(hidden_neg) if self.k_projector is not None else hidden_neg # [neg_num*batch_size, k_dim]
        # use nn.linear as the memory vecotor (Q) to mul the K, and get the score
        out_logits = self.projector_as_q(hidden_neg_mapped) # [neg_num*batch_size, 1]
        hidden_neg_lis = torch.split(out_logits,batch_size,dim=0)
        assert len(hidden_neg_lis) == neg_num
        score = torch.stack(hidden_neg_lis,dim=2)  # [batch_size, 1, neg_num]
        
        score = score.squeeze(1)  # [batch_size, neg_num] 
        
        softmax = torch.nn.Softmax(dim=1)
        if neg_len is None:
            weights = softmax(score)  # [batch_size, neg_num]
            if reverse:
                weights = 1.0 - weights + 1e-10
        else:
            # need to ignore some padding neg samples
            score_for_each_ins = torch.split(score,1,dim=0)
            assert len(neg_len) == batch_size == len(score_for_each_ins)
            weights = []
            for s,neg_l in zip(score_for_each_ins,neg_len):
                weight = softmax(s[:,:neg_l])
                if reverse:
                    weight = 1.0 - weight + 1e-10 
                padding = torch.zeros_like(s[:,neg_l:])
                weight = torch.cat([weight,padding],dim=1)  # [1, neg_num]
                weights.append(weight)
            weights = torch.cat(weights,dim=0)  # [batch_size, neg_num]
         
        print(weights)  # to observe 
        
        return weights
    
    def add_pointer(self,input_dim=768,hidden_dim=128,act="tanh",proj_dim=1,bias=False): 
        if hidden_dim is None:
            self.pointer_projector = nn.Linear(input_dim,proj_dim, bias=bias)
        else:
            # add hidden layer with activation function
            if act == "tanh":
                act_func = nn.Tanh()
            elif act == "relu":
                act_func = nn.ReLU()
            elif act == "prelu":
                act_func = nn.PReLU()
            else:
                raise ValueError("act should be tanh, relu or prelu")
            self.pointer_projector = nn.Sequential(
                nn.Linear(input_dim,hidden_dim, bias=bias),
                act_func,
                nn.Linear(hidden_dim,proj_dim, bias=bias)
            )
    
    def add_tk_sen_projector(self,hidden_dim=768,proj_dim=768):
        if hidden_dim is None or proj_dim is None:
            self.token_projector = None
            self.sentence_projector = None
        else:
            self.token_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
            self.sentence_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
    
    def add_k_projector(self,hidden_dim=768,proj_dim=768):
        if hidden_dim is None or proj_dim is None:
            self.k_projector = None
        else:
            self.k_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
            
    def add_v_projector(self,hidden_dim=768,proj_dim=768):
        if hidden_dim is None or proj_dim is None:
            self.v_projector = None
        else:
            self.v_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
            
    def add_q_projector(self,hidden_dim=768):
        if hidden_dim is None:
            self.projector_as_q = None 
        else:
            self.projector_as_q = nn.Linear(hidden_dim,1, bias=False)  # use linear as the Q vector (learnable)
            
    def add_hidden_projector(self,input_size,output_size,bias=False):
        if input_size is None or output_size is None:
            self.hidden_projector = None
        else:
            self.hidden_projector = nn.Linear(input_size,output_size,bias=bias)
            
    def add_memory_projector(self,input_size=None,output_size=None,bias=True):
        if input_size is None or output_size is None:
            self.memory_projector = None
        else:
            self.memory_projector = nn.Linear(input_size,output_size,bias=bias)
    
    def get_logits(self,logits_ori,labels,ignore_index=-100,return_mask=False):
        '''
        get the probability corresponding to the ground truth tokens (labels)
        the value corresponding to the ignore_idx is 0
        '''
        ## softmax
        softmax = torch.nn.Softmax(dim=2)
        logits = softmax(logits_ori)
        # make sure the sum of the probability == 1 
        # vd = torch.sum(logits,dim=2).detach().cpu().squeeze()
        # assert torch.sum((vd==1.0).int()).item() == vd.size(0) * vd.size(1)
        
        ## watch out the -100 indices!
        lb=labels.unsqueeze(-1)  
        mask = ~(lb == ignore_index)  ## [batch_size, seq_len, 1]
        mask_lb = torch.tensor(mask.clone().detach(),dtype=lb.dtype)
        mask_logits = torch.tensor(mask.clone().detach(),dtype=logits.dtype)
        ## mask the labels correspinding to -100 (i.e., convert them into 0)
        lb_masked = lb.mul(mask_lb)
        gt_logits = torch.gather(logits,2,lb_masked)  ## [batch_size, seq_len, 1]
        gt_logits_masked = gt_logits.mul(mask_logits)
        
        if not return_mask:
            return gt_logits_masked
        else:
            return (gt_logits_masked,mask_logits)
        
    def contrastive_loss_repeat(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None,strategy="max",max_seq_len:int=None,def_len:list=None):
        assert strategy in ["max","min"]
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        # mask the probability corresponding to the padding tokens 
        INF = 999
        temp = torch.zeros_like(gt_logits_neg)
        assert temp.size(0) == batch_size == len(def_len)
        assert all([l>0 for l in def_len])
        for i in range(batch_size):
            true_len = max_seq_len * def_len[i]
            temp[i,true_len:,:] = -INF if strategy == "max" else INF
        gt_logits_neg = temp + gt_logits_neg
        logits_neg_list = torch.split(gt_logits_neg,seq_len,dim=1)
        
        assert len(logits_neg_list) == sample_num
        gt_logits = torch.cat(logits_neg_list,dim=-1)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]  
        
        # choose the one with the max probability to calculate the loss
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1) if strategy == "max" else torch.min(gt_logits_seq_sum,dim=1) ## [batch_size]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice)  ## [batch_size,seq_len,1]
        
        # calculate the loss: P - P_repeat + margin
        delta = (gt_logits_pos - max_gt_logits) + (margin * logits_mask_pos)  
        
        # take the avg of the batch
        # do token-wise loss
        loss_neg = self.calculate_token_wise_neg_loss(delta=delta,logits_mask_pos=logits_mask_pos)
        
        return loss_neg
    
    def contrastive_loss_repeat_v2(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None,strategy="max",max_seq_len:int=None,def_len:list=None):
        '''
        calculate the token-wise loss, force the ground-truth probability of the repeated definition higher than the other definitions
        '''
        assert strategy in ["max","min"]
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        # mask the probability corresponding to the padding tokens 
        INF = 999
        temp = torch.zeros_like(gt_logits_neg)
        assert temp.size(0) == batch_size == len(def_len)
        assert all([l>0 for l in def_len])
        for i in range(batch_size):
            true_len = max_seq_len * def_len[i]
            temp[i,true_len:,:] = -INF if strategy == "max" else INF
        gt_logits_neg = temp + gt_logits_neg
        logits_neg_list = torch.split(gt_logits_neg,seq_len,dim=1)
        assert len(logits_neg_list) == sample_num
        
        gt_logits = torch.cat(logits_neg_list,dim=-1)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]  
        
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1) if strategy == "max" else torch.min(gt_logits_seq_sum,dim=1) ## [batch_size]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice)  ## [batch_size,seq_len,1]
        
        # calculate the loss: P - P_repeat + margin
        gt_logits_pos_avg = self.pooling_hidden(gt_logits_pos,logits_mask_pos.squeeze(-1),pooling="mean")
        max_gt_logits_avg = self.pooling_hidden(max_gt_logits,logits_mask_pos.squeeze(-1),pooling="mean") # [batch_size,1]
        
        delta = (gt_logits_pos_avg - max_gt_logits_avg) + margin  
        delta = delta.clamp(min=0)
        
        # take the avg of the batch
        # do token-wise loss
        loss_neg = torch.mean(delta)
        
        return loss_neg
    
    def contrastive_loss_max_v5(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        similar to v4, but apply -log(1-x) on the ori ranking probability (i.e., N - P + margin), to make the loss smoother 
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        # gt_logits = gt_logits_neg.view(batch_size,seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        logits_neg_list = torch.split(gt_logits_neg,seq_len,dim=1)
        assert len(logits_neg_list) == sample_num
        
        gt_logits = torch.cat(logits_neg_list,dim=-1)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num] 
        
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1)  ## [batch_size]
        # max_gt_logits = torch.index_select(gt_logits,2,max_choice)  ## [batch_size, seq_len]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice)  ## [batch_size,seq_len,1]
        
        # calculate the loss: N - P + margin
        # be aware that the margin shoud also be masked
        max_gt_logits = self.neg_log_v2(max_gt_logits,logits_mask_pos)
        gt_logits_pos = self.neg_log_v2(gt_logits_pos,logits_mask_pos)
        delta = (max_gt_logits - gt_logits_pos) + (margin * logits_mask_pos)  
        obs_mask = (delta > 0.0).bfloat16()
        delta_beta = torch.mul(delta,obs_mask)
        logits_mask_pos_beta = torch.mul(logits_mask_pos,obs_mask)
        sum_mask = torch.sum(logits_mask_pos_beta)
        if sum_mask.item() != 0.:
            loss_neg = torch.sum(delta_beta) / sum_mask
        else:
            loss_neg = torch.tensor(0.).type_as(delta_beta.detach())
            loss_neg.requires_grad_()
        
        return loss_neg
    
    def contrastive_loss_max_v6(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        calculate the loss: N - P + margin
        force the ground-truth probability of postive definitions (e.g., repeated) to be larger than the max of negative definitions (e.g., deleted, null)
        apply -log(1-x) on the probability to make the loss smoother 
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        logits_neg_list = torch.split(gt_logits_neg,seq_len,dim=1)
        assert len(logits_neg_list) == sample_num
        gt_logits = torch.cat(logits_neg_list,dim=-1)  ## [batch_size, seq_len, sample_num]
        
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]  
        
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1)  ## [batch_size]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice)  ## [batch_size,seq_len,1]
        
        # calculate the loss: N - P + margin
        # be aware that the margin shoud also be masked
        max_gt_logits = self.neg_log_v2(max_gt_logits,logits_mask_pos)
        gt_logits_pos = self.neg_log_v2(gt_logits_pos,logits_mask_pos)

        gt_logits_pos_avg = self.pooling_hidden(gt_logits_pos,logits_mask_pos.squeeze(-1),pooling="mean")
        max_gt_logits_avg = self.pooling_hidden(max_gt_logits,logits_mask_pos.squeeze(-1),pooling="mean") # [batch_size,1]
        
        delta = (max_gt_logits_avg-gt_logits_pos_avg) + margin  
        delta = delta.clamp(min=0)
        
        # take the avg of the batch
        # do token-wise loss
        loss_neg = torch.mean(delta)
        
        return loss_neg
    
    def contrastive_loss_max(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        get the maximun probability amomng all the neg outputs
        use this maximun value to calculate contrastive loss
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg = self.get_logits(logits_neg,labels_neg,ignore_index).squeeze()  ## [batch_size, seq_len * sample_num]
        gt_logits_neg = gt_logits_neg.unsqueeze(0) if len(gt_logits_neg.shape) < 2 else gt_logits_neg  ## make sure it has two dimension
        gt_logits_pos = self.get_logits(logits_pos,labels_pos,ignore_index).squeeze()  ## [batch_size, seq_len]
        gt_logits_pos = gt_logits_pos.unsqueeze(0) if len(gt_logits_pos.shape) < 2 else gt_logits_pos
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        # get the maximun value among all the neg outputs
        max_gt_logits, _ = torch.max(gt_logits,dim=2)  ## [batch_size, seq_len]
        # calculate the loss: N - P + margin
        delta = max_gt_logits - gt_logits_pos + margin
        # take the avg of the batch
        loss_neg = torch.mean(torch.sum(delta,dim=1))
        
        return loss_neg

    def calculate_token_wise_neg_loss(self,delta, logits_mask_pos):
        '''
        delta: the ranking loss, i.e., N-P+margin. [batch_size,seq_len,1] 
        logits_mask_pos: [batch_size,seq_len,1] 
        '''
        ## because there are some values less than 0, which will reduce the final loss
        obs_mask = (delta > 0.0).bfloat16()
        delta_beta = torch.mul(delta,obs_mask)
        logits_mask_pos_beta = torch.mul(logits_mask_pos,obs_mask)
        sum_mask = torch.sum(logits_mask_pos_beta)
        if sum_mask.item() != 0.:
            loss_neg = torch.sum(delta_beta) / sum_mask # token-wise loss
        else:
            loss_neg = torch.tensor(0.).type_as(delta_beta.detach())
            loss_neg.requires_grad_()
        
        return loss_neg
    
    def calculate_sentence_wise_neg_loss(self,delta, logits_mask_pos,batch_avg=True):
        '''
        delta: the ranking loss, i.e., N-P+margin. [batch_size,seq_len,1] 
        logits_mask_pos: [batch_size,seq_len,1] 
        
        return: sentence-wise loss, [batch_size,1] if batch_avg is False, else [1]
        '''
        ## because there are some values less than 0, which will reduce the final loss
        obs_mask = (delta > 0.0).bfloat16()
        delta_beta = torch.mul(delta,obs_mask)
        loss_neg = torch.sum(delta_beta,dim=1)  # [batch_size,1]
        if batch_avg:
            loss_neg = torch.mean(loss_neg)
        
        return loss_neg
    
    def neg_log(self,input:torch.Tensor, mask:torch.Tensor,beta:float=1e-8):
        '''
        calculate the -log(x) for input tensor x
        mask: ignore_idx 
        '''
        output = torch.log(input + beta) * -1
        output_masked = output.mul(mask) if mask is not None else output
        return output_masked
    
    def neg_log_v2(self,input:torch.Tensor, mask:torch.Tensor,beta:float=1e-8):
        '''
        calculate the -log(1-x) for input tensor x
        mask: ignore_idx 
        '''
        temp = (1 - input).clamp(beta,1-beta)
        output = torch.log(temp) * -1
        output_masked = output.mul(mask) if mask is not None else output
        return output_masked

    def unsqueeze_logits(self,logits):
        '''make sure the logits have two dimensions'''
        if len(logits.shape) == 0:
            ## batch_size == seq_len == 1
            return logits.unsqueeze(0).unsqueeze(0)
        elif len(logits.shape) == 1:
            ## batch_size or seq_len == 1
            return logits.unsqueeze(0)
        elif len(logits.shape) == 2:
            return logits
    
    def contrastive_loss_all(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        raise NotImplementedError("this function is deprecated")

    def contrastive_loss_attention(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,weights=None,batch_size=None):
        raise NotImplementedError("this function is deprecated")
    
    def contrastive_loss_softmax(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        raise NotImplementedError("this function is deprecated")
    
    def contrastive_loss_max_v2(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        raise NotImplementedError("this function is deprecated")
    
    def contrastive_loss_max_v4(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        raise NotImplementedError("this function is deprecated")
    
    def contrastive_loss_max_v3(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100):
        raise NotImplementedError("this function is deprecated")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            # if past is not None, indicating this is the intermediate step of generation
            # the encoder_outputs is the same as the first step, and can be used during decoding
            # then there is no need for model to do the encoding procedure again
            input_ids = input_ids[:, -1:]
        else:
            # if past is None, indicating this is the start point of generation
            # set encoder_outputs to None
            # then the model will do the encoding procedure as normal
            encoder_outputs,attention_mask = None,None

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "test_phase": True
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        ori_def_list: Optional[List[torch.LongTensor]] = None,
        x_tk_list: Optional[List[torch.LongTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        def_input_ids_list: Optional[List[torch.LongTensor]] = None,
        def_attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids_neg: Optional[torch.LongTensor] = None,
        labels_neg: Optional[torch.LongTensor] = None,
        labels_neg_len: Optional[List[int]] = None,
        pos_neg_ratio: Optional[float] = None,
        margin_null: Optional[float] = None,
        margin_neg: Optional[float] = None,
        margin: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        null_loss_type: Optional[str] = None,
        pos_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio_null: Optional[float] = None,
        loss_mix_ratio_neg: Optional[float] = None,
        loss_mix_ratio_pos: Optional[float] = None,
        sample_num_neg: Optional[int] = None,
        sample_num_pos: Optional[int] = None,
        main_loss_warm: Optional[int] = 0,
        current_epoch: Optional[int] = 0,
        pooling: Optional[str] = "mean",
        reverse: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5EncoderModel

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
