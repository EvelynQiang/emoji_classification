# emoji_classification
This repo includes source code used for EECS 595 course project Emoji Prediction in English Tweets.

## Data Collection
Please go to [SemEval-2018 Task 2 Data](https://competitions.codalab.org/competitions/17344#learn_the_details-data) to download the datasets.

## SVM
For SVM, run the `svm/k-fold-linear.py` with `python3 svm/k-fold-linear.py -r 0.1 -L word -f 2 -C 6 -W 2 -i us_train -T us_test`. To reproduce the svm with bag-of-n-grams, annotate line 122 to line 146. To run svm with doc2vec, annotate line 112 to line 120.

## BiLSTM
For BiLSTM, run 
```
python BiLSTM/model/task2/train.py
```
Due to the limit, we didn't upload the pre-trained embedding files. You need to first put the pre-trained embeddings (e.g. `glove.6B.50d.txt`) into BiLSTM/embeddings.

To run character-level embedding, change the token type in `TASK2-A` in `BiLSTM/model/parameters.py` to "char".  You can also change the embedding dimensions by changing `embed_dim`.

## BERT/RoBERTa
You can train the BERT or RoBERTa model with
```
python BERT/run_emoji_classification.py [BERT | RoBERTa]
```
The resulting model will be saved into `saved_model` folder. Remember to modify the data paths inside the file `BERT/run_emoji_classification.py` in `Class Configs`.


You can then evaluate your model model with
```
python BERT/evaluate_emoji_classification.py [BERT | RoBERTa] <MODEL_NAME>
```
where `MODEL_NAME` is the name of the saved model. Remember to modify the test data paths inside the file `BERT/evaluate_emoji_classification.py`.
