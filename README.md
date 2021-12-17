# emoji_classification
This repo includes source code used for EECS 595 course project Emoji Prediction in English Tweets.

## SVM
For SVM, run the svm/k-fold-linear.py.

## BiLSTM
For BiLSTM, run 
```
python BiLSTM/model/task2/train.py
```
Due to the limit, we didn't upload the pre-trained embedding files. You need to first put the pre-trained embeddings (e.g. `glove.6B.50d.txt`) into BiLSTM/embeddings. 

To run character-level embedding, change the token type in `TASK2-A` in `BiLSTM/model/parameters.py` to "char". 

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
