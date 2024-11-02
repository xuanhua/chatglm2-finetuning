import torch
import torch.nn as nn

from typing import (
    Union
)

from torch.nn import CrossEntropyLoss

import deepspeed

from deepspeed.pipe import (
    TiedLayerSpec,
    LayerSpec
)
from peft import (
  PeftModelForCausalLM,
  LoraModel
)

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

"""
TODO:
LLamaCasualLLM architecture is defined here
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L731
https://huggingface.co/docs/transformers/main/en/model_doc/llama (llama's documentation)

Some already exist implementation of pipeline mode
https://github.com/SparkJiao/llama-pipeline-parallel

Still unknown that if there are actually tied layers in llama model
"""

class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        input_ids, attention_mask, position_ids = ipt
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds, attention_mask, position_ids

class LlamaPipeLayer(torch.nn.Module):
    "One of the multiple layers"
    def __init__(self, 
                 model: Union[ LlamaForCausalLM, PeftModelForCausalLM ], 
                 layer_idx: int):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.gradient_checkpointing = model.model.gradient_checkpointing

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids = ipt

        if self.gradient_checkpointing and self.training:
            output_attentions = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            # layer_outputs = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.layer),
            #     hidden_states,
            #     attention_mask,
            #     position_ids,
            #     None,
            # )
            # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
            outputs = deepspeed.checkpointing.checkpoint(
                create_custom_forward(self.layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
            layer_outputs = [outputs]
        else:
            layer_outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # past_key_value=past_key_value,
                # output_attentions=output_attentions,
                # use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]
        return hidden_states, attention_mask, position_ids

class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.norm = model.model.norm

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids = ipt
        hidden_states = self.norm(hidden_states)

        return hidden_states


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.lm_head = model.lm_head
        self.weight = self.lm_head.weight
        self.config = model.config

    def forward(self, ipt):
        hidden_states = ipt
        logits = torch.nn.functional.linear(hidden_states, self.lm_head.weight)

        return logits

class LossLayer(torch.nn.Module):
    """
    TODO: It's expected input is different from output of its previous layer (LMPipeLayer)
    """
    def forward(self, args):
        logits, labels = args
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss

def get_pipeline_model(model: Union[LlamaForCausalLM, PeftModelForCausalLM]):
    if isinstance(model, PeftModelForCausalLM):
        transformer_layers = model.base_model.model.model.layers
        model = model.base_model.model
    elif isinstance(model, LlamaForCausalLM):
        transformer_layers = model.model.layers
    else:
        raise ValueError("Unsupported model type: ")
    # In below logic, model should be an instance of LlamaForCausalLM
    layers = [
        LayerSpec(EmbeddingPipeLayer, model=model),
        *[LayerSpec(LlamaPipeLayer, model=model, layer_idx=idx) for idx in range(len(transformer_layers))],
        LayerSpec(FLNPipeLayer, model=model),
        LayerSpec(LMPipeLayer, model=model),
        LayerSpec(LossLayer)
    ]
    return layers