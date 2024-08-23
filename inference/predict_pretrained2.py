
from config import CHATGLM_6B_V2_BASE_MODEL_PATH


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_V2_BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_6B_V2_BASE_MODEL_PATH, trust_remote_code=True).half().cuda()
model = model.eval()


text = """
你现在是一个行程预定助理，你从用户和机器人的对话中，提取出航班信息（出发城市、到达城市、出发时间）、酒店预定信息（入住时间、离店时间、入住城市）、火车票信息（出发城市、目的城市、出发时间）、用车信息（用车城市、开始日期、结束日期、用车次数）；下面是用户和机器人的对话：

user: 我当前所在的城市是北京，今天的日期是2024年3月25日

根据这些对话，你将生成如下JSON结构的回复内容:"
"""

response, history = model.chat(tokenizer, text, history=[])
print(response)
#response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
#print(response)