"""

Model Configurations

"""

TASK2_A = {
    "name": "TASK2_A",
    "token_type": "char",
    "batch_train": 32,
    "batch_eval": 32,
    "epochs": 50,
    "embeddings_file": "glove.6B.50d",
    "embed_dim": 300,
    "embed_finetune": False,
    "embed_noise": 0.2,
    "embed_dropout": 0.1,
    "encoder_dropout": 0.3,
    "encoder_size": 300,
    "encoder_layers": 1,
    "encoder_bidirectional": True,
    "attention": True,
    "attention_layers": 1,
    "attention_context": False,
    "attention_activation": "tanh",
    "attention_dropout": 0.,
    "base": 0.3,
    "patience": 5,
    "weight_decay": 0.0,
    "clip_norm": 1,
}

SEMEVAL_2017 = {
    "name": "SEMEVAL_2017",
    "token_type": "word",
    "batch_train": 32,
    "batch_eval": 32,
   "epochs": 50,
    "embeddings_file": "word2vec_300_6_concatened",
    "embed_dim": 310,
    "embed_finetune": False,
    "embed_noise": 0.2,
    "embed_dropout": 0.1,
    "encoder_dropout": 0.3,
    "encoder_size": 150,
    "encoder_layers": 2,
    "encoder_bidirectional": True,
    "attention": True,
    "attention_layers": 2,
    "attention_context": False,
    "attention_activation": "tanh",
    "attention_dropout": 0.3,
    "base": 0.68,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
