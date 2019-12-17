#!/usr/bin/env python3

# pylint: disable=invalid-name

import torch

from transformers import BertForMaskedLM, BertTokenizer

text = "How many airports are there in Japan?"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_text = tokenizer.tokenize(text)
tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")

print(tokenized_text)
# ['[CLS]', 'how', 'many', 'airports', 'are', 'there', 'in', 'japan', '?', '[SEP]']

masked_index = 4
tokenized_text[masked_index] = "[MASK]"

print(tokenized_text)
# ['[CLS]', 'how', 'many', 'airports', '[MASK]', 'there', 'in', 'japan', '?', '[SEP]']

# テキストのままBERTに渡すのではなく、辞書で変換し、idになった状態にする。
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([tokens])

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# GPUがなければ、以下の2行をコメントアウト
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# masked_indexとなっている部分の単語の予測結果を取り出し、その予測結果top5を出す
_, predict_indexes = torch.topk(predictions[0, masked_index], k=5)
predict_tokens = tokenizer.convert_ids_to_tokens(predict_indexes.tolist())

print(predict_tokens)
# ['are', 'were', 'out', 'is', 'go']

# TODO: 以下の予測を試してみたい
# 1. BertForNextSentencePrediction
# 2. BertForSequenceClassification
# 3. BertForTokenClassification
# 4. BertForQuestionAnswering
