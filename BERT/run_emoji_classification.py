import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import argparse


def load_data(text_path, label_path=None):
    with open(text_path) as f:
        lines = [line.rstrip() for line in f]
        texts = pd.DataFrame(lines, columns=['texts'])
    if label_path:
        with open(label_path) as f:
            lines = [line.rstrip() for line in f]
            labels = pd.DataFrame(lines, columns=['labels'])
            labels = pd.get_dummies(labels.labels)
        return texts.texts.values, labels.values.astype('float')
    else:
        return texts.texts.values, None


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return accuracy_score(labels_flat, preds_flat)


# def accuracy_per_class(preds, labels):
#     label_dict_inverse = {v: k for k, v in label_dict.items()}
#
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#
#     for label in np.unique(labels_flat):
#         y_preds = preds_flat[labels_flat == label]
#         y_true = labels_flat[labels_flat == label]
#         print(f'Class: {label_dict_inverse[label]}')
#         print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

# def calcuate_accu(big_idx, targets):
#     n_correct = (big_idx==targets).sum().item()
#     return n_correct


def main(arg):
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Configuration
    class Configs:
        EPOCHS = 5
        BATCH_SIZE = 16
        MAX_LEN = 100
        LEARNING_RATE = 1e-05
        EPS = 1e-08
        # TODO: change the model
        PRETRAINED_MODEL = 'bert-base-uncased' if arg == 'BERT' else 'roberta-base'
        TOKENIZER = BertTokenizer.from_pretrained(
            PRETRAINED_MODEL) if arg == 'BERT' else RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)
        # TRAIN_TEXT_PATH = '../train/us_train.text'
        # TRAIN_LABEL_PATH = '../train/us_train.labels'
        # VAL_TEXT_PATH = '../trial/us_trial.text'
        # VAL_LABEL_PATH = '../trial/us_trial.labels'
        TRAIN_TEXT_PATH = 'us_train.text'
        TRAIN_LABEL_PATH = 'us_train.labels'
        VAL_TEXT_PATH = 'us_trial.text'
        VAL_LABEL_PATH = 'us_trial.labels'
        # TEST_TEXT_PATH = '../test/us_test.text'
        # TEST_LABEL_PATH = '../test/us_test.labels'

    # load datasets
    train_texts, train_labels = load_data(Configs.TRAIN_TEXT_PATH, Configs.TRAIN_LABEL_PATH)
    val_texts, val_labels = load_data(Configs.VAL_TEXT_PATH, Configs.VAL_LABEL_PATH)
    val_texts, val_labels = val_texts[:5000], val_labels[:5000]
    # test_texts, test_labels = load_data(Configs.TEST_TEXT_PATH, Configs.TEST_LABEL_PATH)

    encoded_data_train = Configs.TOKENIZER.batch_encode_plus(
        train_texts,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=Configs.MAX_LEN,
        return_tensors='pt',
        truncation=True
    )

    encoded_data_val = Configs.TOKENIZER.batch_encode_plus(
        val_texts,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=Configs.MAX_LEN,
        return_tensors='pt',
        truncation=True
    )

    # encoded_data_test = Configs.TOKENIZER.batch_encode_plus(
    #     test_texts,
    #     add_special_tokens=True,
    #     return_attention_mask=True,
    #     padding='max_length',
    #     max_length=Configs.MAX_LEN,
    #     return_tensors='pt',
    #     truncation=True
    # )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(train_labels)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(val_labels)

    # input_ids_test = encoded_data_test['input_ids']
    # attention_masks_test = encoded_data_test['attention_mask']
    # labels_test = torch.tensor(test_labels)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    # dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_train = DataLoader(dataset_train, batch_size=Configs.BATCH_SIZE)
    dataloader_val = DataLoader(dataset_val, batch_size=Configs.BATCH_SIZE)
    # dataloader_test = DataLoader(dataset_test, batch_size=Configs.BATCH_SIZE)

    # create model
    if arg == 'BERT':
        model = BertForSequenceClassification.from_pretrained(Configs.PRETRAINED_MODEL,
                                                              num_labels=20,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
    else:
        model = RobertaForSequenceClassification.from_pretrained(Configs.PRETRAINED_MODEL,
                                                                 num_labels=20,
                                                                 output_attentions=False,
                                                                 output_hidden_states=False)
    model.to(device)
    # print(model)

    # TODO: change the loss function and optimizer
    optimizer = AdamW(model.parameters(), lr=Configs.LEARNING_RATE, eps=Configs.EPS)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * Configs.EPOCHS)

    # train
    def evaluate(dataloader_val):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    for epoch in tqdm(range(1, Configs.EPOCHS + 1)):
        accuracy_accumulate = []

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1].detach().cpu().numpy()
            accuracy_accumulate.append(accuracy_func(logits, inputs['labels'].cpu().numpy()))
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch)),
                                      'accuracy': '{:.2%}'.format(sum(accuracy_accumulate) / len(accuracy_accumulate))})

        torch.save(model.state_dict(), f'saved_model/finetuned_{arg.lower()}_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        tqdm.write(f'Average accuracy: {accuracy_func(predictions, true_vals)}')

    # evaluate on test
    # test_loss, predictions, true_vals = evaluate(dataloader_test)
    # test_f1 = f1_score_func(predictions, true_vals)
    # tqdm.write(f'Validation loss: {test_loss}')
    # tqdm.write(f'F1 Score (Weighted): {test_f1}')
    # tqdm.write(f'Average accuracy: {accuracy_score(true_vals, predictions)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT or RoBERTa for emoji classification.')
    parser.add_argument('model', choices=['BERT', 'RoBERTa'], type=str, help='The model used for '
                                                                             'training.')
    args = parser.parse_args()
    main(args.model)
