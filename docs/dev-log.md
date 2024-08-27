## 2024-08-27

1. First the input text will be wrapped like this:

```json
['[Round 1]\n\n问：\n你现在是一个行程预定助理，你从用户和机器人的对话中，提取出航班信息（出发城市、到达城市、出发时间）、酒店预定信息（入住时间、离店时间、入住城市）、火车票信息（出发城市、目的城市、出发时间）、用车信息（用车城市、开始日期、结束日期、用车次数）；下面是用户和机器人的对话：\n\nuser: 我当前所在的城市是北京，今天的日期是2024年3月25日\n\n根据这些对话，你将生成如下JSON结构的回复内容:"\n\n\n答：']
```

2. Call `SPTokenizer.tokenize()`  to do sentence piece based tokenization

3. `/home/ubuntu/anaconda3/lib/python3.9/site-packages/transformers/tokenization_utils.py`

   ```python
   # line 708
   return self.convert_tokens_to_ids(tokens)
   |-->return self.tokenizer.convert_token_to_id(token) # /home/ubuntu/.cache/huggingface/modules/transformers_modules/chatglm2_6b/tokenization_chatglm.py #119
   	|--># Now all special tokens are:  special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]

   ```

   `/home/ubuntu/.cache/huggingface/modules/transformers_modules/chatglm2_6b/tokenization_chatglm.py`

​        Convert all tokens to ids,  according to built sentence piece model

4. Add two special tokens to the end of current ids

   `[gMask]`(64790) and `[sop]`(64792) are added.

   {'input_ids': [64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211, 13, 50457, 32103, 35241, 44369, 34712, 31123, 54622, 54708, 32053, 54542, 33290, 31664, 34580, 54538, 31123, 36995, 54557, 35502, 31707, 31301, 33389, 31733, 31201, 34029, 31733, 31201, 33389, 31643, 32217, 32386, 44369, 31707, 31301, 38020, 31643, 31201, 55081, 55207, 31643, 31201, 38020, 31733, 32217, 35885, 55295, 31707, 31301, 33389, 31733, 31201, 32236, 31733, 31201, 33389, 31643, 32217, 39153, 31707, 31301, 39153, 31733, 31201, 31699, 33594, 31201, 32332, 33594, 31201, 39153, 36942, 34019, 33182, 54532, 32053, 54542, 33290, 31664, 34580, 31211, 13, 13, 4865, 30954, 34211, 32527, 32835, 36226, 54532, 31719, 31123, 35398, 33594, 54532, 30943, 30940, 30943, 30972, 54540, 30966, 54595, 30943, 30970, 54576, 13, 13, 31793, 31704, 34580, 31123, 54622, 54687, 36454, 33163, 20727, 45160, 34664, 31795, 12989, 13, 13, 13, 55437, 31211]}

    Usually the input comprised with 3 parts:

   ```python
   ['input_ids', 'attention_mask', 'position_ids']
   ```

5. Add `attention_mask` and `position_ids` respectly (suppose that current sequence length is 136)

   ```python
   attention_mask = [1]* 136
   position_ids = list(range(136))
   ```

6. The final output is an instance of type `<class 'transformers.tokenization_utils_base.BatchEncoding'>`

   [36474, 31714, 32103, 35241, 44369, 34712, 31123, 54622, 54708, 32053, 54542, 33290, 31664, 34580, 54538, 31123, 36995, 54557, 35502, 31707, 31301, 33389, 31733, 31201, 34029, 31733, 31201, 33389, 31643, 32217, 32386, 44369, 31707, 31301, 38020, 31643, 31201, 55081, 55207, 31643, 31201, 38020, 31733, 32217, 35885, 55295, 31707, 31301, 33389, 31733, 31201, 32236, 31733, 31201, 33389, 31643, 32217, 39153, 31707, ...]

## 2024-08-23
Core logic of decoding part:
```bash
/home/ubuntu/anaconda3/lib/python3.9/site-packages/transformers/generation/utils.py
```
