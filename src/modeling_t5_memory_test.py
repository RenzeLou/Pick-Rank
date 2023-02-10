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
            normed_hidden_states,  # decoder input hidden states (from self-attention)
            mask=attention_mask,
            key_value_states=key_value_states,  # encoder output hidden states
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

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.BoolTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     decoder_head_mask: Optional[torch.FloatTensor] = None,
    #     cross_attn_head_mask: Optional[torch.Tensor] = None,
    #     encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     indicators: Optional[list] = None,
    #     loss_mix_ratio: Optional[float] = None
    # ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
    #         config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
    #         labels in `[0, ..., config.vocab_size]`

    #     Returns:

    #     Examples:

    #     ```python
    #     >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

    #     >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
    #     >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

    #     >>> # training
    #     >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    #     >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    #     >>> outputs = model(input_ids=input_ids, labels=labels)
    #     >>> loss = outputs.loss
    #     >>> logits = outputs.logits

    #     >>> # inference
    #     >>> input_ids = tokenizer(
    #     ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
    #     >>> ).input_ids  # Batch size 1
    #     >>> outputs = model.generate(input_ids)
    #     >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #     >>> # studies have shown that owning a dog is good for you.
    #     ```"""
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    #     if head_mask is not None and decoder_head_mask is None:
    #         if self.config.num_layers == self.config.num_decoder_layers:
    #             warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
    #             decoder_head_mask = head_mask

    #     # Encode if needed (training, first prediction pass)
    #     if encoder_outputs is None:
    #         # Convert encoder inputs in embeddings if needed
    #         encoder_outputs = self.encoder(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             inputs_embeds=inputs_embeds,
    #             head_mask=head_mask,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #     elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
    #         encoder_outputs = BaseModelOutput(
    #             last_hidden_state=encoder_outputs[0],
    #             hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
    #             attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
    #         )

    #     hidden_states = encoder_outputs[0]

    #     if self.model_parallel:
    #         torch.cuda.set_device(self.decoder.first_device)

    #     if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
    #         # get decoder inputs from shifting lm labels to the right
    #         decoder_input_ids = self._shift_right(labels)

    #     # Set device for model parallelism
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.decoder.first_device)
    #         hidden_states = hidden_states.to(self.decoder.first_device)
    #         if decoder_input_ids is not None:
    #             decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
    #         if attention_mask is not None:
    #             attention_mask = attention_mask.to(self.decoder.first_device)
    #         if decoder_attention_mask is not None:
    #             decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    #     # Decode
    #     decoder_outputs = self.decoder(
    #         input_ids=decoder_input_ids,
    #         attention_mask=decoder_attention_mask,
    #         inputs_embeds=decoder_inputs_embeds,
    #         past_key_values=past_key_values,
    #         encoder_hidden_states=hidden_states,
    #         encoder_attention_mask=attention_mask,
    #         head_mask=decoder_head_mask,
    #         cross_attn_head_mask=cross_attn_head_mask,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     sequence_output = decoder_outputs[0]

    #     # Set device for model parallelism
    #     if self.model_parallel:
    #         torch.cuda.set_device(self.encoder.first_device)
    #         self.lm_head = self.lm_head.to(self.encoder.first_device)
    #         sequence_output = sequence_output.to(self.lm_head.weight.device)

    #     if self.config.tie_word_embeddings:
    #         # Rescale output before projecting on vocab
    #         # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    #         sequence_output = sequence_output * (self.model_dim**-0.5)

    #     lm_logits = self.lm_head(sequence_output)  ## [batch_size, batch_max_seq_length, vocab_size]
        
    #     if indicators is None:
    #         # logger.warning("indicators is None, automatically generate indicators\n"+
    #         #                "this is normal when you are doing prediction")
    #         indicators = [1] * lm_logits.size(0)
        
    #     indices_pos = np.argwhere(np.array(indicators)==1).squeeze()
    #     indices_pos = torch.tensor(indices_pos).to(lm_logits.device)
    #     indices_neg = np.argwhere(np.array(indicators)==0).squeeze()
    #     indices_neg = torch.tensor(indices_neg).to(lm_logits.device)
        
    #     # get both pos and neg input and labels
    #     lm_logits_pos = torch.index_select(lm_logits,0,indices_pos)
    #     lm_logits_neg = torch.index_select(lm_logits,0,indices_neg)
        
    #     loss = None
    #     if labels is not None:
    #         labels_pos = torch.index_select(labels,0,indices_pos)
    #         labels_neg = torch.index_select(labels,0,indices_neg)
            
    #         loss_fct_pos = CrossEntropyLoss(ignore_index=-100)
    #         loss_fct_neg = self.sum_neg_loss
            
    #         pos_num = lm_logits_pos.size(0)
    #         neg_num = lm_logits_neg.size(0)
            
    #         # assert loss_mix_ratio <= 1.0 and loss_mix_ratio >= 0.0, "loss mix ratio must be in [0,1]" 
            
    #         if pos_num !=0  and neg_num !=0: 
    #             # calculate pos loss
    #             loss_pos = loss_fct_pos(lm_logits_pos.view(-1, lm_logits_pos.size(-1)), labels_pos.view(-1))
    #             # calculate neg loss, sum up
    #             loss_neg = loss_fct_neg(lm_logits_neg,labels_neg,ignore_index=-100)
    #             # TODO: use BCE loss, namely loss = -log(1-x)
    #             loss = (1-loss_mix_ratio) * loss_pos + loss_mix_ratio * loss_neg
    #         elif neg_num != 0:
    #             loss_neg = loss_fct_neg(lm_logits_neg,labels_neg,ignore_index=-100)
    #             loss = loss_mix_ratio * loss_neg
    #         elif pos_num != 0:
    #             loss_pos = loss_fct_pos(lm_logits_pos.view(-1, lm_logits_pos.size(-1)), labels_pos.view(-1))
    #             loss = (1-loss_mix_ratio) * loss_pos
    #         else:
    #             raise RuntimeError("pos_num == neg_num == 0, check your indicators!")
    #     if not return_dict:
    #         output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
    #         return ((loss,) + output) if loss is not None else output

    #     return Seq2SeqLMOutput(
    #         loss=loss,
    #         logits=lm_logits,
    #         past_key_values=decoder_outputs.past_key_values,
    #         decoder_hidden_states=decoder_outputs.hidden_states,
    #         decoder_attentions=decoder_outputs.attentions,
    #         cross_attentions=decoder_outputs.cross_attentions,
    #         encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    #         encoder_hidden_states=encoder_outputs.hidden_states,
    #         encoder_attentions=encoder_outputs.attentions,
    #     )
        
    def sum_neg_loss(self,logits_ori,labels,ignore_index=-100):
        '''simply sum up the probability corresponding to the ground truth tokens'''
        gt_logits_masked = self.get_logits(logits_ori,labels,ignore_index)
        loss_neg = torch.sum(gt_logits_masked)
        
        return loss_neg
    
    def multiple_forward(
        self,
        input_ids_list: Optional[List[torch.LongTensor]] = None,
        attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_mix_ratio: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        '''
        foward multiple instances including pos and neg instruction.
        thus the input can be a list
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        hidden_states_list = []  ## TODO: use batch forward instead of for-loop
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_list = []
            for input_ids, attention_mask in zip(input_ids_list,attention_mask_list):
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoder_outputs_list.append(encoder_outputs)
                hidden_states_list.append(encoder_outputs[0])
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            # hidden_states = hidden_states.to(self.decoder.first_device)
            hidden_states_list = [hidden_states.to(self.decoder.first_device) for hidden_states in hidden_states_list]
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs_list = []
        sequence_output_list = []
        for hidden_states,attention_mask in zip(hidden_states_list,attention_mask_list):
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
            sequence_output = decoder_outputs[0]
            sequence_output_list.append(sequence_output)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            # sequence_output = sequence_output.to(self.lm_head.weight.device)
            sequence_output_list = [sequence_output.to(self.lm_head.weight.device) for sequence_output in sequence_output_list]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output_list = [sequence_output * (self.model_dim**-0.5) for sequence_output in sequence_output_list]

        lm_logits_list = [self.lm_head(sequence_output) for sequence_output in sequence_output_list]
        # lm_logits = self.lm_head(sequence_output)  ## [batch_size, batch_max_seq_length, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct_pos = CrossEntropyLoss(ignore_index=-100)
            loss_fct_neg = self.contrastive_loss_max_v2  ## v2 (no log) is a little better than v3
            # loss_fct_neg = self.contrastive_loss_max_v3
            
            # pos loss (ori loss, maximazing the ground truth probability)
            lm_logits_pos = lm_logits_list[0]  ## [batch_size,seq_len,vocab_size]
            loss_pos = loss_fct_pos(lm_logits_pos.view(-1, lm_logits_pos.size(-1)), labels.view(-1))
            # neg loss (contrastive loss) 
            lm_logits_neg = torch.cat(lm_logits_list[1:],dim=1) ## [batch_size,seq_len*3,vocab_size]
            labels_neg = labels.repeat(1,len(lm_logits_list[1:])) ## [batch_size,seq_len*3]
            loss_neg = loss_fct_neg(lm_logits_pos,lm_logits_neg,labels,labels_neg,margin=margin,ignore_index=-100)
            
            assert loss_mix_ratio >= 0.0
            
            loss = loss_pos + loss_mix_ratio * loss_neg.clamp(min=0.0)
        if not return_dict:
            # only return pos logits
            output = (lm_logits_pos,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_pos,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def multiple_forward_batchfy(
        self,
        input_ids_list: Optional[List[torch.LongTensor]] = None,
        attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_neg_ratio: Optional[float] = None,
        margin_pos: Optional[float] = None,
        margin_neg: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio: Optional[float] = None,
        sample_num_neg: Optional[int] = None,
        sample_num_pos: Optional[int] = None,
        main_loss_warm: Optional[int] = 0,
        current_epoch: Optional[int] = 0
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        '''
        foward multiple instances including pos and neg instruction.
        and batchfy all these input
        thus the actual batch size becomes batch_size * sample_num
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        sample_num = len(input_ids_list)
        pos_sample = input_ids_list[0]
        if len(pos_sample.shape) < 2:
            batch_szie = 1
        elif len(pos_sample.shape) == 2:
            batch_szie = pos_sample.size(0)
        else:
            raise ValueError

        max_seq_list = [sample.size(1) for sample in input_ids_list]
        max_seq_len = max(max_seq_list)    
        assert len(set(max_seq_list)) == 1, "we need to batchfy the input, thus the max_seq_len of each instances mush be the same!"
        
        # use batch forward instead of for-loop
        input_ids = torch.cat(input_ids_list,dim=0)  ## [batch_size * (sample_num + 2), max_seq_len]
        attention_mask = torch.cat(attention_mask_list,dim=0)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
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

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

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
        
        # decoder input also need to expand
        decoder_input_ids = decoder_input_ids.repeat(sample_num,1)

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

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)  ## [batch_size * (sample_num), max_seq_length, vocab_size]
        lm_logits_list = torch.split(lm_logits,batch_szie,dim=0)
        
        loss = None
        if labels is not None:
            loss_fct_ori = CrossEntropyLoss(ignore_index=-100)
            # choose neg loss type
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
            else:
                raise NotImplementedError("Error neg loss function type!")
            # loss_fct_neg = self.contrastive_loss_max_v2  ## v2 (no log) is a little better than v3
            # loss_fct_neg = self.contrastive_loss_max_v3
            # loss_fct_neg = self.contrastive_loss_all
            # loss_fct_neg = self.contrastive_loss_softmax
            loss_fct_pos = self.contrastive_loss_pos_v2
            
            assert len(lm_logits_list) == 1 + sample_num_pos + sample_num_neg, "len(lm_logits_list): {}; sample_num_pos + sample_num_neg: {}".format(len(lm_logits_list),sample_num_pos + sample_num_neg)
            # ori loss, maximazing the ground truth probability
            loss_ori = 0
            lm_logits_ori = lm_logits_list[0]  ## [batch_size,seq_len,vocab_size]
            if not neg_loss_only:
                loss_ori = loss_fct_ori(lm_logits_ori.view(-1, lm_logits_ori.size(-1)), labels.view(-1))
            # neg loss (contrastive loss) 
            loss_neg = None
            if sample_num_neg > 0 and current_epoch >= main_loss_warm:
                lm_logits_neg = torch.cat(lm_logits_list[sample_num_pos+1:],dim=1) ## [batch_size,seq_len*3,vocab_size]
                labels_neg = labels.repeat(1,sample_num_neg) ## [batch_size,seq_len*3]
                loss_neg = loss_fct_neg(lm_logits_ori,lm_logits_neg,labels,labels_neg,margin=margin_neg,ignore_index=-100,batch_size=batch_szie)      
                if torch.isnan(loss_neg).any().item():
                    raise RuntimeError("the loss is nan!")
                if torch.isinf(loss_neg).any().item():
                    raise RuntimeError("the loss is inf!")
            # pos loss (constractive loss)
            loss_pos = None
            if sample_num_pos > 0 and current_epoch >= main_loss_warm:
                lm_logits_pos = torch.cat(lm_logits_list[1:sample_num_pos+1],dim=1) ## [batch_size,seq_len*3,vocab_size]
                labels_pos = labels.repeat(1,sample_num_pos) ## [batch_size,seq_len*3]
                loss_pos = loss_fct_pos(lm_logits_ori,lm_logits_pos,labels,labels_pos,sample_num=sample_num_pos,margin=margin_pos,ignore_index=-100,batch_size=batch_szie)      
            # overall loss
            loss_addition = 0.
            loss_addition += pos_neg_ratio * loss_neg.clamp(min=0.0) if loss_neg is not None else 0.
            loss_addition += (1-pos_neg_ratio) * loss_pos.clamp(min=0.0) if loss_pos is not None else 0.
            
            assert loss_mix_ratio >= 0.0
            loss = loss_ori + loss_addition * loss_mix_ratio
            
            if current_epoch >= main_loss_warm:
                self.all_step += 1
                # to observe whether the neg loss has contributions
                if (loss != loss_ori).item():
                    self.obvious += 1
            # save loss for observation
            self.current_step += batch_szie
            self.loss_current.append(loss.item())
            if self.current_step >= self.record_step:
                avg_loss = np.mean(self.loss_current)
                self.loss_record.append(avg_loss)
                self.current_step = 0
            
        if not return_dict:
            # only return pos logits
            output = (lm_logits_ori,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_ori,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def multiple_forward_batchfy_no_additonal_loss(
        self,
        input_ids_list: Optional[List[torch.LongTensor]] = None,
        attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_neg_ratio: Optional[float] = None,
        margin_pos: Optional[float] = None,
        margin_neg: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio: Optional[float] = None,
        sample_num_neg: Optional[int] = None,
        sample_num_pos: Optional[int] = None,
        main_loss_warm: Optional[int] = 0,
        current_epoch: Optional[int] = 0
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        '''
        only calculate the crossentropy loss on the inputs
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        sample_num = len(input_ids_list)
        pos_sample = input_ids_list[0]
        if len(pos_sample.shape) < 2:
            batch_szie = 1
        elif len(pos_sample.shape) == 2:
            batch_szie = pos_sample.size(0)
        else:
            raise ValueError

        max_seq_list = [sample.size(1) for sample in input_ids_list]
        max_seq_len = max(max_seq_list)    
        assert len(set(max_seq_list)) == 1, "we need to batchfy the input, thus the max_seq_len of each instances mush be the same!"
        
        # use batch forward instead of for-loop
        input_ids = torch.cat(input_ids_list,dim=0)  ## [batch_size, max_seq_len]
        attention_mask = torch.cat(attention_mask_list,dim=0)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
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

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

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
        
        # decoder input also need to expand
        decoder_input_ids = decoder_input_ids.repeat(sample_num,1)

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

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)  ## [batch_size * (sample_num), max_seq_length, vocab_size]
        lm_logits_list = torch.split(lm_logits,batch_szie,dim=0)
        
        loss = None
        if labels is not None:
            loss_fct_ori = CrossEntropyLoss(ignore_index=-100)
            # choose neg loss type
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
            else:
                raise NotImplementedError("Error neg loss function type!")
            # loss_fct_neg = self.contrastive_loss_max_v2  ## v2 (no log) is a little better than v3
            # loss_fct_neg = self.contrastive_loss_max_v3
            # loss_fct_neg = self.contrastive_loss_all
            # loss_fct_neg = self.contrastive_loss_softmax
            loss_fct_pos = self.contrastive_loss_pos_v2
            
            assert len(lm_logits_list) == 1 + sample_num_pos + sample_num_neg, "len(lm_logits_list): {}; sample_num_pos + sample_num_neg: {}".format(len(lm_logits_list),sample_num_pos + sample_num_neg)
            # ori loss, maximazing the ground truth probability
            loss_ori = 0
            lm_logits_ori = lm_logits_list[0]  ## [batch_size,seq_len,vocab_size]
            if not neg_loss_only:
                loss_ori = loss_fct_ori(lm_logits_ori.view(-1, lm_logits_ori.size(-1)), labels.view(-1))
            # neg loss (contrastive loss) 
            loss_neg = None
            if sample_num_neg > 0 and current_epoch >= main_loss_warm:
                lm_logits_neg = torch.cat(lm_logits_list[sample_num_pos+1:],dim=1) ## [batch_size,seq_len*3,vocab_size]
                labels_neg = labels.repeat(1,sample_num_neg) ## [batch_size,seq_len*3]
                loss_neg = loss_fct_neg(lm_logits_ori,lm_logits_neg,labels,labels_neg,margin=margin_neg,ignore_index=-100,batch_size=batch_szie)      
                if torch.isnan(loss_neg).any().item():
                    raise RuntimeError("the loss is nan!")
                if torch.isinf(loss_neg).any().item():
                    raise RuntimeError("the loss is inf!")
            # pos loss (constractive loss)
            loss_pos = None
            if sample_num_pos > 0 and current_epoch >= main_loss_warm:
                lm_logits_pos = torch.cat(lm_logits_list[1:sample_num_pos+1],dim=1) ## [batch_size,seq_len*3,vocab_size]
                labels_pos = labels.repeat(1,sample_num_pos) ## [batch_size,seq_len*3]
                loss_pos = loss_fct_pos(lm_logits_ori,lm_logits_pos,labels,labels_pos,sample_num=sample_num_pos,margin=margin_pos,ignore_index=-100,batch_size=batch_szie)      
            # overall loss
            loss_addition = 0.
            loss_addition += pos_neg_ratio * loss_neg.clamp(min=0.0) if loss_neg is not None else 0.
            loss_addition += (1-pos_neg_ratio) * loss_pos.clamp(min=0.0) if loss_pos is not None else 0.
            
            assert loss_mix_ratio >= 0.0
            loss = loss_ori + loss_addition * loss_mix_ratio
            
            if current_epoch >= main_loss_warm:
                self.all_step += 1
                # to observe whether the neg loss has contributions
                if (loss != loss_ori).item():
                    self.obvious += 1
            # save loss for observation
            self.current_step += batch_szie
            self.loss_current.append(loss.item())
            if self.current_step >= self.record_step:
                avg_loss = np.mean(self.loss_current)
                self.loss_record.append(avg_loss)
                self.current_step = 0
            
        if not return_dict:
            # only return pos logits
            output = (lm_logits_ori,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_ori,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def add_gen_kwargs(self, gen_kwargs:dict):
        ''' 
        Since it is hard to pass hyperparameter in the generate() API provided by HuggingFace, use this func to ensure a correct args passing.
        note that, this function can only be called when doing testing 
        '''
        self.gen_kwargs_used = False
        self.gen_kwargs = gen_kwargs
    
    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids_list: Optional[List[torch.LongTensor]] = None,
        attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        def_input_ids_list: Optional[List[torch.LongTensor]] = None,
        def_attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids_neg: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
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
        margin_out: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        null_loss_type: Optional[str] = None,
        out_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio_null: Optional[float] = None,
        loss_mix_ratio_neg: Optional[float] = None,
        loss_mix_ratio_out: Optional[float] = None,
        sample_num_neg: Optional[int] = None,
        sample_num_pos: Optional[int] = None,
        main_loss_warm: Optional[int] = 0,
        current_epoch: Optional[int] = 0,
        pooling: Optional[str] = "mean",
        reverse: Optional[bool] = False,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        test_phase: Optional[bool] = False
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        '''
        only calculate the crossentropy loss on the inputs
        '''
        # useless ====================
        input_ids = None
        attention_mask = None
        # useless ====================
        
        if input_ids_list is None and attention_mask_list is None and def_input_ids_list is None and def_attention_mask_list is None:
            # logger.warning("all required inputs are None, assert this is a testing procedure, get self.gen_kwargs")
            # assert not self.gen_kwargs_used, "self.gen_kwargs_used == True, please call add_gen_kwargs() first"
            assert test_phase, "this behavious should only happen in testing phase"
            input_ids_list = self.gen_kwargs["input_ids_list"]
            attention_mask_list = self.gen_kwargs["attention_mask_list"]
            def_input_ids_list = self.gen_kwargs["def_input_ids_list"]
            def_attention_mask_list = self.gen_kwargs["def_attention_mask_list"]
            # decoder_input_ids = self.gen_kwargs["decoder_input_ids"]
            # decoder_attention_mask = self.gen_kwargs.get("decoder_attention_mask", None)  # here, we should use the prediction results of model
            # labels = self.gen_kwargs.get("labels", None)
            neg_loss_type = self.gen_kwargs.get("neg_loss_type", None)
            null_loss_type = self.gen_kwargs.get("null_loss_type", None)
            neg_loss_only = self.gen_kwargs.get("neg_loss_only", False)
            sample_num_pos = self.gen_kwargs.get("sample_num_pos", None)
            sample_num_neg = self.gen_kwargs.get("sample_num_neg", None)
            current_epoch = self.gen_kwargs.get("current_epoch", 0)
            encoder_outputs = self.gen_kwargs.get("encoder_outputs", None)
            pooling = self.gen_kwargs.get("pooling", "max")
            reverse = self.gen_kwargs.get("reverse", False)
            
            test_phase = True
            
            self.gen_kwargs_used = True
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
                
        neg_num = len(input_ids_list) - 1

        # Encode if needed (training, first prediction pass)
        sample_num = len(input_ids_list)  ## should be 1 (ori) + 1 (null) + 5 (neg)
        pos_sample = input_ids_list[0]
        if len(pos_sample.shape) < 2:
            batch_szie = 1
        elif len(pos_sample.shape) == 2:
            batch_szie = pos_sample.size(0)
        else:
            raise ValueError
        
        assert sample_num == 1 + neg_num, "should be 1 (ori) + 1 (null) + 5 (neg)"

        max_seq_list = [sample.size(1) for sample in input_ids_list]
        max_seq_len = max(max_seq_list)    
        assert len(set(max_seq_list)) == 1, "we need to batchfy the input, thus the max_seq_len of each instances mush be the same!"
        
        # use batch forward instead of for-loop
        input_ids = torch.cat(input_ids_list,dim=0)  ## [batch_size * (1 + null_num + neg_num), max_seq_len]
        attention_mask = torch.cat(attention_mask_list,dim=0)
        def_input_ids = torch.cat(def_input_ids_list,dim=0)  ## [batch_size * (1 + null_num + neg_num), max_seq_len]
        def_attention_mask = torch.cat(def_attention_mask_list,dim=0)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # indicating it is a training procedure
            # or, it is the first prediction pass
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            hidden_states = encoder_outputs[0]  # [batch_size * (1+neg_num), max_seq_len, hidden_size]

            if neg_num > 0:
                # get the hidden states of def
                def_encoder_outputs = self.encoder(
                        input_ids=def_input_ids,
                        attention_mask=def_attention_mask,
                        inputs_embeds=None,
                        head_mask=None,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                )
                def_hidden_states = def_encoder_outputs[0]  # [batch_size * (1+neg_num), max_seq_len, hidden_size]
                
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
                
                # get the def memory bank
                perturbation_memories = self.memory_bank(def_hidden_states, def_attention_mask,batch_szie,neg_num,pooling) # [batch_size * neg_num, memo_dim]
                memory_for_each_ins = []
                for index in range(batch_szie):
                    index_list = [index+t*batch_szie for t in range(neg_num)]
                    memory_for_each_ins.append(perturbation_memories[index_list,:])
                perturbation_memories = torch.cat(memory_for_each_ins,dim=0)  # [batch_size * neg_num, memo_dim]
                
                # get the attention weights
                weights = self.get_att_weights(hidden_states,attention_mask,batch_szie,neg_num,pooling,reverse=reverse)  # [batch_size,neg_num]
                weights = weights.reshape(batch_szie*neg_num,1).repeat(1,perturbation_memories.shape[-1])  # [batch_size * neg_num,memo_dim]
                
                # weighted sum memories
                perturbation_memories_weighted = torch.mul(perturbation_memories,weights)  # [batch_size * neg_num,memo_dim]
                memories_lis = torch.split(perturbation_memories_weighted,neg_num,dim=0)
                assert len(memories_lis) == batch_szie
                batched_memories_lis = []
                for m in memories_lis:  # [neg_num,memo_dim]
                    memory = torch.sum(m,dim=0,keepdim=True) # [1,memo_dim]
                    batched_memories_lis.append(memory)
                batched_memories = torch.cat(batched_memories_lis,dim=0)  # [batch_size,memo_dim]
                
                # expand the memory, concat to the hidden states
                hidden_states_ori = hidden_states[:batch_szie,:,:]  # [batch_size, max_seq_len, hidden_size]
                hidden_size = hidden_states_ori.shape[-1]
                attention_mask_ori = attention_mask[:batch_szie,:]
                batched_memories_expanded = batched_memories.unsqueeze(1).repeat(1,hidden_states_ori.shape[1],1)  # [batch_size, max_seq_len, memo_dim]
                hidden_states_mixed = torch.cat([batched_memories_expanded,hidden_states_ori],dim=-1)  # [batch_size, max_seq_len, hidden_size+memo_dim]
                hidden_states_mixed = self.hidden_projector(hidden_states_mixed) # [batch_size, max_seq_len, hidden_size]
                assert hidden_states_mixed.shape[-1] == hidden_size
            else:
                hidden_states_mixed = hidden_states if self.hidden_projector_test is None else self.hidden_projector_test(hidden_states)  ## TODO: test whether a projector can impact the performance
                attention_mask_ori = attention_mask
            
            return_encoder_outputs = False
            
            # save the hidden states for the next prediction pass
            if test_phase:
                # encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states_mixed,hidden_states=None,attentions=None)
                encoder_outputs_saved = [hidden_states_mixed]
                self.gen_kwargs["encoder_outputs"] = encoder_outputs_saved
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            
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
                    
            # get the hidden states directly
            hidden_states_mixed = encoder_outputs[0]
            attention_mask_ori = attention_mask[:batch_szie,:]
            return_encoder_outputs = True  # continue using the encoder outputs

       
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

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
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states_mixed,
            encoder_attention_mask=attention_mask_ori,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
            # if sequence_output_neg is not None:
            #     sequence_output_neg = sequence_output_neg.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
            # if sequence_output_neg is not None:
            #     sequence_output_neg = sequence_output_neg * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)  ## [batch_size, max_seq_length, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct_ori = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct_ori(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            # save loss for observation
            self.current_step += batch_szie
            self.loss_current.append(loss.item())
            if self.current_step >= self.record_step:
                avg_loss = np.mean(self.loss_current)
                self.loss_record.append(avg_loss)
                self.current_step = 0
            
        if not return_dict:
            # only return pos logits
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if return_encoder_outputs else hidden_states_mixed,  # this "encoder_last_hidden_state" can not be used 
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def pooling_hidden(self,hidden_states,attention_mask,pooling:str):
        assert pooling in ["max","mean"]
        if pooling == "mean":
            mask = attention_mask.type_as(hidden_states).unsqueeze(-1)  # [batch_size * (1+neg_num), max_seq_length, 1]
            mask = mask.repeat(1,1,hidden_states.size(-1))  # [batch_size * (1+neg_num), max_seq_length, hidden_size]
            hidden_states_masked = hidden_states * mask
            hidden_pooled = torch.mean(hidden_states_masked,dim=1)  # [batch_size * (1+neg_num), hidden_size]
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
        
    def get_att_weights(self,hidden_states,attention_mask,batch_size,neg_num,pooling="max",reverse=False):
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
        weights = softmax(score)  # [batch_size, neg_num]
        
        if reverse:
            weights = 1.0 - weights + 1e-10
            
        # print(weights)  # to observe 
        
        return weights
    
    def add_q_k_projector(self,hidden_dim=768,proj_dim=768):
        if hidden_dim is None or proj_dim is None:
            self.q_projector = None
            self.k_projector = None
            self.proj_dim = None
            self.att_dim = None
        else:
            self.q_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
            self.k_projector = nn.Linear(hidden_dim,proj_dim, bias=False)
            self.proj_dim = proj_dim
            self.att_dim = hidden_dim
            
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

    def add_hidden_projector_test(self,input_size,output_size,bias=False):
        if input_size is None or output_size is None:
            self.hidden_projector_test = None
        else:
            self.hidden_projector_test = nn.Linear(input_size,output_size,bias=bias)
        
    
    def get_logits(self,logits_ori,labels,ignore_index=-100,return_mask=False):
        '''
        get the logits (probability) corresponding to the ground truth tokens (labels)
        the value corresponding to the ignore_idx is 0
        '''
        ## softmax
        softmax = torch.nn.Softmax(dim=2)
        logits = softmax(logits_ori)
        ## make sure the sum of the probability == 1 
        # vd = torch.sum(logits,dim=2).detach().cpu().squeeze()
        # assert torch.sum((vd==1.0).int()).item() == vd.size(0) * vd.size(1)
        
        ## watch out the -100 indices!
        lb=labels.unsqueeze(-1)  
        mask = ~(lb == ignore_index)  ## [2,60,1]
        mask_lb = torch.tensor(mask.clone().detach(),dtype=lb.dtype)
        mask_logits = torch.tensor(mask.clone().detach(),dtype=logits.dtype)
        ## mask the labels correspinding to -100 (i.e., convert them into 0)
        lb_masked = lb.mul(mask_lb)
        gt_logits = torch.gather(logits,2,lb_masked)  ## [2,60,1]
        gt_logits_masked = gt_logits.mul(mask_logits)
        
        if not return_mask:
            return gt_logits_masked
        else:
            return (gt_logits_masked,mask_logits)
    
    def contrastive_loss_out_constrain_all(self,lm_logits_ori,labels,lm_logits_neg,labels_neg,labels_neg_len_tensor,
                                           labels_neg_len,margin,ignore_index=-100,batch_size=None):
        '''
        Take the constractive loss among all the negative outputs.
        '''
        gt_logits,logits_mask = self.get_logits(lm_logits_ori,labels,ignore_index,return_mask=True)  ## [batch_size,seq_len,1]
        # TODO: try to sum the probability of all tokens instead of average
        # take the avg of ground truth token probability
        sum_gt_logits = torch.sum(gt_logits,dim=1) ## [batch_size, 1]
        sum_logits_mask = torch.sum(logits_mask,dim=1)
        avg_gt_logits = torch.div(sum_gt_logits,sum_logits_mask)
        avg_gt_logits_rep = torch.repeat_interleave(avg_gt_logits,labels_neg_len_tensor,dim=0) ## [neg_out_num, 1]
        
        # take the avg of negative output token probability
        # logits_rep = lm_logits_neg ## [neg_out_num,seq_len,vocab_size]
        neg_logits, neg_logits_mask = self.get_logits(lm_logits_neg,labels_neg,ignore_index,return_mask=True) ## [neg_out_num,seq_len,1]
        sum_neg_logits = torch.sum(neg_logits,dim=1)
        sum_neg_logits_mask = torch.sum(neg_logits_mask,dim=1)
        # zero_indicator = (sum_neg_logits_mask == 0).type_as(sum_neg_logits_mask)
        sum_neg_logits_mask = sum_neg_logits_mask.clamp(min=1.0)  # in order to avoid 'nan'
        avg_neg_logits = torch.div(sum_neg_logits,sum_neg_logits_mask) ## [neg_out_num, 1]
        
        # calculate the constractive loss: N - P + margin
        delta = avg_neg_logits - avg_gt_logits_rep + margin ## [neg_out_num, 1]
        # sum the 'delta' of the neg_outputs of each instance
        all_ins_delta = torch.split(delta,labels_neg_len,dim=0)
        batch_delta_sum_list = [torch.sum(ins_delta.clamp(min=0.0)) for ins_delta in all_ins_delta]
        # then take batch-wise avg as the final loss
        loss_neg = 0.
        for batch_delta_sum in batch_delta_sum_list:
            loss_neg += batch_delta_sum
        assert len(batch_delta_sum_list) == batch_size
        loss_neg = loss_neg / len(batch_delta_sum_list)
        
        return loss_neg
        
    def contrastive_loss_pos(self, logits_ori,logits_pos,labels_ori,margin,ignore_index=-100,batch_size=None):
        '''
        if P > Ori:
           loss =  P - Ori
        else: 
           loss = Ori - P - margin
           
        Specifically, take the avg porbability of all tokens in a instance,
        use this sentence-level probability to calculate the final loss func
        '''
        ## assume there is only one pos instruction
        ## logits_ori.shape == logits_pos.shape == [batch_size,seq_len,vocab_size]
        gt_logits_ori,logits_mask_ori = self.get_logits(logits_ori,labels_ori,ignore_index,return_mask=True)  ## [batch_size, seq_len, 1]
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_ori,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        ## take avg token probability (notice the ignore_idx)
        ### first take the sum of all tokens
        sum_ori = torch.sum(gt_logits_ori,dim=1) ## [batch_size, 1]
        sum_ori_mask = torch.sum(logits_mask_ori,dim=1) 
        sum_pos = torch.sum(gt_logits_pos,dim=1)
        sum_pos_mask = torch.sum(logits_mask_pos,dim=1)
        ### then divide by the no_pad token num
        mean_ori = torch.div(sum_ori,sum_ori_mask) ## [batch_size, 1]
        mean_pos = torch.div(sum_pos,sum_pos_mask)
        assert not torch.isnan(mean_ori).any().item() and not torch.isnan(mean_pos).any().item(), "divide by 0, overflow!"  # divide by 0, overflow
        ## calculate loss
        greater = (mean_pos > mean_ori).float()
        smaller = (mean_pos <= mean_ori).float()
        loss_greater = torch.mul((mean_pos - mean_ori),greater)
        loss_smaller = torch.mul((mean_ori - mean_pos - margin),smaller)
        loss_batch = loss_greater + loss_smaller
        loss = torch.mean(loss_batch)
        
        return loss

    def contrastive_loss_pos_v2(self, logits_ori,logits_pos,labels,labels_pos,margin,sample_num,ignore_index,batch_size):
        '''
        if P > Ori:
           loss =  P - Ori - margin
        else: 
           loss = Ori - P - margin
           
        if (P > Ori): use the pos example with the max probability to calculate the loss
        else: use the pos example with the min probability instead
        '''
        ## assume there is only one pos instruction
        ## logits_ori.shape == [batch_size,seq_len,vocab_size]
        gt_logits_ori,logits_mask_ori = self.get_logits(logits_ori,labels,ignore_index,return_mask=True)  ## [batch_size, seq_len, 1]
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_ori.size(1)
        ## take avg token probability (notice the ignore_idx)
        ### first take the sum of all tokens
        sum_ori = torch.sum(gt_logits_ori,dim=1) ## [batch_size, 1]
        sum_ori_mask = torch.sum(logits_mask_ori,dim=1) 
        ### then divide by the no_pad token num
        mean_ori = torch.div(sum_ori,sum_ori_mask) ## [batch_size, 1]
        
        # split the multiple pos examples
        logits_pos_list = torch.split(gt_logits_pos,seq_len,dim=1)
        mask_pos_list = torch.split(logits_mask_pos,seq_len,dim=1)
        assert len(logits_pos_list) == len(mask_pos_list) == sample_num
        # take avg for every pos example
        mean_pos_list = []
        for logits_pos, mask_pos in zip(logits_pos_list,mask_pos_list):
            sum_pos = torch.sum(logits_pos,dim=1,keepdim=True)
            sum_pos_mask = torch.sum(mask_pos,dim=1,keepdim=True)
            mean_tmp = torch.div(sum_pos,sum_pos_mask)
            mean_pos_list.append(mean_tmp)
        mean_pos = torch.cat(mean_pos_list,dim=1) ## [batch_size, sample_num, 1]
        # find the max and the min
        max_mean_pos,_ = torch.max(mean_pos,dim=1) ## [batch_size, 1]
        min_mean_pos,_ = torch.min(mean_pos,dim=1) ## [batch_size, 1]
        # calculate the loss
        max_loss = max_mean_pos - mean_ori - margin
        min_loss = mean_ori - min_mean_pos - margin
        loss_batch = max_loss.clamp(min=0.0) + min_loss.clamp(min=0.0)  ## TODO: use one loss (use thw bigger one between max_loss and min loss, instead of the sum of them)
        
        loss = torch.mean(loss_batch)
        
        return loss
    
    def contrastive_loss_max(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        get the maximun probability amomng all the neg outputs
        use this maximun value to calculate contrastive loss
        TODO: warning! this function has some bugs need to be fixed, see v2 for reference
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
    
    def contrastive_loss_max_v2(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        get the maximun probability amomng all the neg outputs
        use this maximun value to calculate contrastive loss
        unlike v1, v2 chooses the neg output with the highest sum of probability
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_neg = gt_logits_neg.squeeze()  ## [batch_size, seq_len * sample_num]
        # gt_logits_neg = gt_logits_neg.unsqueeze(0) if len(gt_logits_neg.shape) < 2 else gt_logits_neg 
        gt_logits_neg = self.unsqueeze_logits(gt_logits_neg)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        gt_logits_pos = gt_logits_pos.squeeze()  ## [batch_size, seq_len]
        # gt_logits_pos = gt_logits_pos.unsqueeze(0) if len(gt_logits_pos.shape) < 2 else gt_logits_pos
        gt_logits_pos = self.unsqueeze_logits(gt_logits_pos)
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]  ## TODO: does sum make sense? 
        # gt_logits_seq_sum = torch.mean(gt_logits,dim=1)  ## [batch_size, sample_num]
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1)  ## [batch_size]
        # max_gt_logits = torch.index_select(gt_logits,2,max_choice)  ## [batch_size, seq_len]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice).squeeze()  ## [batch_size,seq_len]
        # max_gt_logits = max_gt_logits.unsqueeze(0) if len(max_gt_logits.shape) < 2 else max_gt_logits
        max_gt_logits = self.unsqueeze_logits(max_gt_logits)
        # calculate the loss: N - P + margin
        # be aware that the margin shoud also be masked
        delta = (max_gt_logits - gt_logits_pos) + (margin * logits_mask_pos.squeeze())  ## TODO: set the margin to 0.3 temporally according to the observation
        # take the avg of the batch
        assert len(delta.shape) == 2, 'no way?'
        if batch_size == 1:
            loss_neg = torch.sum(delta)
        else:
            if seq_len == 1:
                loss_neg = torch.mean(delta)
            else:
                loss_neg = torch.mean(torch.sum(delta,dim=1))
        # try:
            # loss_neg = torch.mean(torch.sum(delta,dim=1))
        # except IndexError:
        #     print("warning! index error")
        #     print("shape of delta: ",delta.shape)
        #     print("shape of max_gt_logit: ",max_gt_logits)
        #     print("shape of gt_logits_pos: ",gt_logits_pos)
        #     print("shape of logits_mask_pos.squeeze(): ",logits_mask_pos.squeeze().shape)
        #     exit()
        
        return loss_neg
    
    def contrastive_loss_max_v4(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        similar to v2, but take token-wise loss, just like ori_crossentropy_loss and pos_loss work
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        # gt_logits_neg = gt_logits_neg.squeeze()  ## [batch_size, seq_len * sample_num]
        # # gt_logits_neg = gt_logits_neg.unsqueeze(0) if len(gt_logits_neg.shape) < 2 else gt_logits_neg 
        # gt_logits_neg = self.unsqueeze_logits(gt_logits_neg)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        # gt_logits_pos = gt_logits_pos.squeeze()  ## [batch_size, seq_len]
        # # gt_logits_pos = gt_logits_pos.unsqueeze(0) if len(gt_logits_pos.shape) < 2 else gt_logits_pos
        # gt_logits_pos = self.unsqueeze_logits(gt_logits_pos)
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        # gt_logits = gt_logits_neg.view(batch_size,seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        logits_neg_list = torch.split(gt_logits_neg,seq_len,dim=1)
        assert len(logits_neg_list) == sample_num
        # logits_neg_list = [t.squeeze(-1) for t in logits_neg_list]
        gt_logits = torch.cat(logits_neg_list,dim=-1)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]  ## TODO: does sum make sense? 
        # gt_logits_seq_sum = torch.mean(gt_logits,dim=1)  ## [batch_size, sample_num]
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1)  ## [batch_size]
        # max_gt_logits = torch.index_select(gt_logits,2,max_choice)  ## [batch_size, seq_len]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice)  ## [batch_size,seq_len,1]
        # max_gt_logits = max_gt_logits.unsqueeze(0) if len(max_gt_logits.shape) < 2 else max_gt_logits
        # max_gt_logits = self.unsqueeze_logits(max_gt_logits)
        # calculate the loss: N - P + margin
        # be aware that the margin shoud also be masked
        delta = (max_gt_logits - gt_logits_pos) + (margin * logits_mask_pos)  ## TODO: i dont know whether token-wise loss make sense, it is better to use avg logits to calvulate loss (sentence-wise)
        # take the avg of the batch
        # do token-wise loss
        # loss_neg = torch.mean(delta)
        # note the mask
        # loss_neg = torch.sum(delta) / torch.sum(logits_mask_pos)
        loss_neg = self.calculate_token_wise_neg_loss(delta=delta,logits_mask_pos=logits_mask_pos)
        
        # if torch.isnan(loss_neg).any().item():
            # raise RuntimeError("the loss is nan!")
            # loss_neg = 0.
        
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
    
    def contrastive_loss_all(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        Same as contrastive_loss_max, but use all neg samples to calculate the neg losses
        instead of only the max one
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        # gt_logits_neg = gt_logits_neg.squeeze()  ## [batch_size, seq_len * sample_num]
        # gt_logits_neg = gt_logits_neg.unsqueeze(0) if len(gt_logits_neg.shape) < 2 else gt_logits_neg 
        # gt_logits_neg = self.unsqueeze_logits(gt_logits_neg)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        # gt_logits_pos = gt_logits_pos.squeeze()  ## [batch_size, seq_len]
        # gt_logits_pos = gt_logits_pos.unsqueeze(0) if len(gt_logits_pos.shape) < 2 else gt_logits_pos
        # gt_logits_pos = self.unsqueeze_logits(gt_logits_pos)
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        gt_logits = torch.split(gt_logits,1,dim=-1)
        ## TODO: we assume different neg samples are shared with the same hyper-parameters (i.e., ratio & margin) 
        loss_neg = 0.
        for logits in gt_logits:  ## [batch_size, seq_len, 1]
            delta = (logits - gt_logits_pos) + (margin * logits_mask_pos)  
            l = self.calculate_token_wise_neg_loss(delta=delta,logits_mask_pos=logits_mask_pos)
            # l = torch.mean(torch.sum(delta,dim=1))
            loss_neg += l.clamp(min=0.0)
        
        return loss_neg

    def contrastive_loss_attention(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,weights=None,batch_size=None):
        '''
        Same as contrastive_loss_all
        add attention to the neg loss of each neg samples
        
        weights: [batch_size, sample_num]
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        seq_len = gt_logits_pos.size(1)
        # reshape the size to get the max
        batch_size = logits_pos.size(0)
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        gt_logits = torch.split(gt_logits,1,dim=-1)
        weight_lis = torch.split(weights,1,dim=-1)
        assert len(weight_lis) == len(gt_logits) == sample_num
        ## TODO: we assume different neg samples are shared with the same hyper-parameters (i.e., ratio & margin) 
        loss_neg = torch.zeros(batch_size).type_as(logits_pos).cuda(logits_pos.device)
        for logits,weight in zip(gt_logits,weight_lis):  ## [batch_size, seq_len, 1]
            delta = (logits - gt_logits_pos) + (margin * logits_mask_pos)  
            l = self.calculate_sentence_wise_neg_loss(delta=delta,logits_mask_pos=logits_mask_pos,batch_avg=False)  # [batch_size,1]
            l = l.squeeze(-1).clamp(min=0.0) # [batch_size,]
            l = l * weight.squeeze(-1)  # [batch_size,]
            loss_neg += l
        # take batch average
        loss_neg = torch.mean(loss_neg)
        
        return loss_neg
    
    def contrastive_loss_softmax(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100,batch_size=None):
        '''
        Same as contrastive_loss_all, but use the softmax of POS among the NEG examples as the loss
        -log [ softmax (p, [p,n1,n2,n3] ) ]
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        # gt_logits_neg = gt_logits_neg.squeeze()  ## [batch_size, seq_len * sample_num]
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        # seq_len = gt_logits_pos.size(1)
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits_neg = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        gt_logits_pos = gt_logits_pos.view(gt_logits_pos.size(0),seq_len,-1)  ## [batch_size, seq_len, 1]
        gt_logits = torch.cat([gt_logits_pos,gt_logits_neg],dim=2)  ## [batch_size, seq_len, sample_num + 1]
        # sum the probability of each token in an instance
        # take all the samples (pos and neg) to calculate softmax
        # we want the probability of pos example is as high as possible
        gt_logits_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]
        results = self.softmax(gt_logits_sum)   ## check the value in torch.sum(results,dim=1)
        results_list = torch.split(results,1,dim=-1)
        results_pos = results_list[0]  ## [batch_size, 1]
        # put -log 
        output_pos = self.neg_log(results_pos,mask=None)
        loss_neg = torch.mean(output_pos)
        # loss_neg = loss_neg.clamp(min=0.0)
        
        return loss_neg
    
    def contrastive_loss_max_v3(self,logits_pos,logits_neg,labels_pos,labels_neg,margin=0.5,ignore_index=-100):
        '''
        Same as v2, but use [(-logP) - (-logN)] instead of (N-P)
        '''
        # get the corresponding probability of ground truth tokens
        gt_logits_neg,logits_mask_neg = self.get_logits(logits_neg,labels_neg,ignore_index,return_mask=True)
        gt_logits_neg = gt_logits_neg.squeeze()  ## [batch_size, seq_len * sample_num]
        gt_logits_neg = gt_logits_neg.unsqueeze(0) if len(gt_logits_neg.shape) < 2 else gt_logits_neg 
        gt_logits_pos,logits_mask_pos = self.get_logits(logits_pos,labels_pos,ignore_index,return_mask=True)
        gt_logits_pos = gt_logits_pos.squeeze()  ## [batch_size, seq_len]
        gt_logits_pos = gt_logits_pos.unsqueeze(0) if len(gt_logits_pos.shape) < 2 else gt_logits_pos
        # reshape the size to get the max
        seq_len = logits_pos.size(1)
        sample_num = int(logits_neg.size(1) / seq_len)  ## the num of neg samples
        gt_logits = gt_logits_neg.view(gt_logits_neg.size(0),seq_len,sample_num)  ## [batch_size, seq_len, sample_num]
        # sum the probability
        gt_logits_seq_sum = torch.sum(gt_logits,dim=1)  ## [batch_size, sample_num]
        # choose the one with max sum of probability
        _, max_choice = torch.max(gt_logits_seq_sum,dim=1)  ## [batch_size]
        # max_gt_logits = torch.index_select(gt_logits,2,max_choice)  ## [batch_size, seq_len]
        max_choice = max_choice.unsqueeze(-1)  ## [batch_size,1]
        max_choice = max_choice.repeat(1,seq_len).unsqueeze(-1)  ## [batch_size,seq_len,1]
        max_gt_logits = torch.gather(gt_logits,2,max_choice).squeeze()  ## [batch_size,seq_len]
        max_gt_logits = max_gt_logits.unsqueeze(0) if len(max_gt_logits.shape) < 2 else max_gt_logits
        # calculate loss
        max_gt_logits_neg_log = self.neg_log(max_gt_logits,mask=logits_mask_neg.squeeze(-1)[:,:seq_len])
        gt_logits_pos_neg_log = self.neg_log(gt_logits_pos,mask=logits_mask_pos.squeeze(-1))
        delta = (gt_logits_pos_neg_log - max_gt_logits_neg_log) + (margin * logits_mask_pos.squeeze())  ## TODO: set the margin to 10 temporally according to the observation
        # take the avg of the batch
        loss_neg = torch.mean(torch.sum(delta,dim=1))
        
        return loss_neg
    
    def neg_log(self,input:torch.Tensor, mask:torch.Tensor,beta:float=1e-8):
        '''
        calculate the -log(x) for input tensor x
        mask: ignore_idx 
        '''
        output = torch.log(input + beta) * -1
        output_masked = output.mul(mask) if mask is not None else output
        return output_masked

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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        input_ids_list: Optional[List[torch.LongTensor]] = None,
        attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        def_input_ids_list: Optional[List[torch.LongTensor]] = None,
        def_attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids_neg: Optional[torch.LongTensor] = None,
        labels_neg: Optional[torch.LongTensor] = None,
        labels_neg_len: Optional[List[int]] = None,
        pos_neg_ratio: Optional[float] = None,
        margin_null: Optional[float] = None,
        margin_neg: Optional[float] = None,
        margin_out: Optional[float] = None,
        neg_loss_type: Optional[str] = None,
        null_loss_type: Optional[str] = None,
        out_loss_type: Optional[str] = None,
        neg_loss_only: Optional[bool] = False,
        loss_mix_ratio_null: Optional[float] = None,
        loss_mix_ratio_neg: Optional[float] = None,
        loss_mix_ratio_out: Optional[float] = None,
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
