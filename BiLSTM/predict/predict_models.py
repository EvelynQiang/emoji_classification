from dataloaders.task2 import load_task2
from predict.predictions import dump_attentions
from utils.train import load_pretrained_model

model, conf = load_pretrained_model("TASK2_A_0.3711")
for task in ["train", "trial", "test"]:
    X, y = load_task2(task)
    dump_attentions(X, y, "TASK2_A_{}".format(task), model, conf, "clf")