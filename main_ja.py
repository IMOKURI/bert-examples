#!/usr/bin/env python3

# pylint: disable=invalid-name

from pyknp import Juman

import torch

from transformers import BertModel, BertTokenizer

jumanpp = Juman()

text = "僕は友達とサッカーをすることが好きだ"
result = jumanpp.analysis(text)

tokenized_text = [mrph.midasi for mrph in result.mrph_list()]

print(tokenized_text)
# ['僕', 'は', '友達', 'と', 'サッカー', 'を', 'する', 'こと', 'が', '好きだ']

tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")

masked_index = 5
tokenized_text[masked_index] = "[MASK]"

print(tokenized_text)
# ['[CLS]', '僕', 'は', '友達', 'と', '[MASK]', 'を', 'する', 'こと', 'が', '好きだ', '[SEP]']

model = BertModel.from_pretrained("bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers")
bert_tokenizer = BertTokenizer(
    "bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers/vocab.txt",
    do_lower_case=False,
    do_basic_tokenize=False,
)

indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# masked_indexとなっている部分の単語の予測結果を取り出し、その予測結果top5を出す
_, predict_indexes = torch.topk(predictions[0, masked_index], k=5)
predict_tokens = bert_tokenizer.convert_ids_to_tokens(predict_indexes.tolist())

print(predict_tokens)
# ['##ン', '##イ', '製', '２００１', '戦争']
