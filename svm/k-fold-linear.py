#!/usr/bin/env python3

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np

from emoji_data import load
from features import doc_to_ngrams

from cmdline import add_args
from argparse import ArgumentParser
ap = ArgumentParser()
add_args(ap, ('general', 'preproc', 'linear', 'tune'))
opt = ap.parse_args()

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

seed=1234


class Sequencer():
    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                 ):
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.

        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        for word in temp_vocab:
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))

        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts) - 1):
                if counts[i] < counts[i + 1]:
                    counts[i + 1], counts[i] = counts[i], counts[i + 1]
                    indexes[i], indexes[i + 1] = indexes[i + 1], indexes[i]
                else:
                    cnt += 1

        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])

    def textToVector(self, text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens) - 1 if len(tokens) < self.seq_len else self.seq_len - 1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass

        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100, ))

        return np.asarray(vec).flatten()

if opt.class_weight:
    opt.class_weight = "balanced"
else:
    opt.class_weight = None

from logging import debug, info, basicConfig
basicConfig(level=opt.log_level,
                    format='%(asctime)s %(message)s')

info('----start----')
info(','.join([k + '=' + str(vars(opt)[k]) for k in sorted(vars(opt))]))


# ---main---

data = load(opt.input_prefix)
# print(len(data))
data1 = load(opt.input_test_prefix)
# print(len(data1))
train_len = len(data.docs)
test_len = len(data1.docs)
data.docs = data.docs + data1.docs
labels = np.array(data.labels)
labels1 = np.array(data1.labels)

docs, v, _ = doc_to_ngrams(data.docs, min_df=opt.min_df,
                          used_cache = False, cache = False,
                          dim_reduce = opt.dim_reduce,
                          c_ngmin = opt.c_ngmin,
                          c_ngmax = opt.c_ngmax,
                          w_ngmin = opt.w_ngmin,
                          w_ngmax = opt.w_ngmax,
                          lowercase = opt.lowercase,
                          input_name = opt.input_prefix)

data1 = []
for i in data.docs:
    data1.append(word_tokenize(i))

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data.docs)]
max_epochs = 10
vec_size = 20
alpha = 0.025
model = Doc2Vec(alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
docs = []
for i in data1:
    docs.append(model.infer_vector(i))

docs = np.array(docs)
docs1 = docs[train_len:]
docs = docs[:train_len]
print(docs.shape, docs1.shape)
print(labels.shape, labels1.shape)

# info("number of word/character features ({}/{}): {}".format(
#             opt.w_ngmax, opt.c_ngmax, len(v.vocabulary_)))

if opt.classifier == 'lr': 
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(dual=True, C=opt.C, verbose=0,
            class_weight=opt.class_weight)
elif opt.classifier == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(class_weight=opt.class_weight,n_estimators=300,random_state=seed)
else:
    from sklearn.svm import LinearSVC
    m = LinearSVC(dual=True, C=opt.C, verbose=0,
            class_weight=opt.class_weight)
   
if opt.mult_class == 'ovo':
    mc = OneVsOneClassifier
elif opt.mult_class == 'ovr':
    mc = OneVsRestClassifier
if opt.classifier != 'rf':
    m = mc(m, n_jobs=opt.n_jobs)

skf = StratifiedKFold(n_splits=opt.k)
#skf = StratifiedKFold(labels, opt.k)
acc = []
f1M = []
# for train, test in skf.split(docs, labels):
# #for train, test in skf:
#     m.fit(docs[train], labels[train])
#     pred = m.predict(docs[test])
#     acc.append(accuracy_score(labels[test], pred))
#     f1M.append(f1_score(labels[test], pred, average='macro'))

# info("Accuracy: {:0.4f}±{:0.4f}.".format( 100*np.mean(acc), 100*np.std(acc)))
# info("F1(macro): {:0.4f}±{:0.4f}.".format( 100*np.mean(f1M), 100*np.std(f1M)))

m.fit(docs, labels)
pred = m.predict(docs1)
print(docs.shape, docs1.shape)
print(pred.shape)
print("Accuracy:", accuracy_score(labels1, pred))
print("F1(macro):", f1_score(labels1, pred, average='macro'))


info('----end----')

# ll = sorted(set(data.labels), key=lambda x: int(x))
# fmt = "{:>3}" + "{:>7}" * (len(ll))
# print(fmt.format(" ", *data.labelchar))
# for i, row in enumerate(confusion_matrix(data.labels, pred, data.labels=ll)):
#     print(fmt.format(data.labelchar[i], *row))
