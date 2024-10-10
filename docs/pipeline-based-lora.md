
We found that lora model cannot fit into a single GPU card (like lora based training for a 6B model on a single 24G gpu). But we have to use multiple GPU cards and split the original model into several pieces and put them on different gpus.

The training and parameters' saving are not easy as imagined. Here we provide an implementation FYI.

**Build model with pipeline style**

The original model need be rewritten as pipeline based. Deepspeed has provide such abstractions and it is relative easy.

**Model training**

This part takes several steps before reaching the real forwarding process. Here are the steps:
* Load the huggingface pretrained model M_0
* Turn it to a lora model M_1 by huggingface's peft library.
* Turn M_1 to a pipeline based model M_2 
* Use deepspeed to train M_2 and save checkpoints

**Transform saved pipeline based model to huggingface pretrained style based model**

* Load M_2 with deepspeed 
* Initialize another lora model from huggingface pretrained model, named it M_1'
* Use key `[query_key_value]` to find all matched model components and use them to replace the corresponding ones in M_1'
* Save lora adapter LA in M_1' to disk as lora model. 
* Load another M_0 and turn it into lora model with LA, let's call it M_3 
* Use `merge_and_upload()` function to merge lora adapter in M_3 into the original model M_4
* M_4 call `save_pretrained()` to make it become a after-fine-tuned model.

**Why doing so much model transformation**

* Try to use existing tools maximumly. But there is a single tool can do all of these things.
* The most important thing is to make fine-tuned model could be inferenced on third-party inference engine directly like vllm.