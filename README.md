# BERT examples

[BERTを理解しながら自分のツイートを可視化してみるハンズオン](https://qiita.com/yuki_uchida/items/09fda4c5c608a9f53d2f)をやってみた。

## Prerequisites

- GPUセットアップ済み(GPUがなければ、ソースコード内の `cuda` に関する場所をコメントアウト)
- `pip install --user torch torchvision transformers scikit-learn pyknp`
- download `通常版: Japanese_L-12_H-768_A-12_E-30_BPE_transformers.zip (393M; 19/11/15公開)` from [BERT日本語Pretrainedモデル](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)
    - and extract to `bert/`
- install latest boost ([最新版をソースからインストール](https://www.mathkuro.com/?p=230)) or `apt install libboost-all-dev`
- install latest JUMAN++ ([JUMAN++のインストール](https://dev.classmethod.jp/server-side/python/pyknpjumann-tutorial/))

## Run

- `./main_en.py`

```
['[CLS]', 'how', 'many', 'airports', 'are', 'there', 'in', 'japan', '?', '[SEP]']
['[CLS]', 'how', 'many', 'airports', '[MASK]', 'there', 'in', 'japan', '?', '[SEP]']
['are', 'were', 'is', 'lie', 'go']
```

- `./main_ja.py`

```
['僕', 'は', '友達', 'と', 'サッカー', 'を', 'する', 'こと', 'が', '好きだ']
['[CLS]', '僕', 'は', '友達', 'と', '[MASK]', 'を', 'する', 'こと', 'が', '好きだ', '[SEP]']
['話', '仕事', 'キス', 'ゲーム', 'サッカー']
```
