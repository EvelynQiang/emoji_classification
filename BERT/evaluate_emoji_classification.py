from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import cuda
import numpy as np
from run_emoji_classification import load_data, f1_score_func, accuracy_func
import argparse


def main(args):
    # initialize model
    device = 'cuda' if cuda.is_available() else 'cpu'
    if args.model == 'BERT':
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=20,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
    else:
        model = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                                 num_labels=20,
                                                                 output_attentions=False,
                                                                 output_hidden_states=False)
    model.to(device)
    model.load_state_dict(torch.load(f'saved_model/{args.name}', map_location=torch.device('cpu')))

    # load data
    TEST_TEXT_PATH = 'us_test.text'
    TEST_LABEL_PATH = 'us_test.labels'
    test_texts, test_labels = load_data(TEST_TEXT_PATH, TEST_LABEL_PATH)
    if args.model == 'BERT':
        encoded_data_test = BertTokenizer.from_pretrained('bert-base-uncased').batch_encode_plus(
            test_texts,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=100,
            return_tensors='pt',
            truncation=True
        )
    else:
        encoded_data_test = RobertaTokenizer.from_pretrained('roberta-base').batch_encode_plus(
            test_texts,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=100,
            return_tensors='pt',
            truncation=True
        )
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(test_labels)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=16)

    # evaluate model on test data
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_test:
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

    loss_val_avg = loss_val_total / len(dataloader_test)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    val_f1 = f1_score_func(predictions, true_vals)
    print(f'Validation loss: {loss_val_avg}')
    print(f'F1 Score (Weighted): {val_f1}')
    print(f'Average accuracy: {accuracy_func(predictions, true_vals)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BERT or RoBERTa for emoji classification.')
    parser.add_argument('model', choices=['BERT', 'RoBERTa'], type=str, help='The model used for '
                                                                             'evaluating.')
    parser.add_argument('name', type=str, help='The name of the saved model.')
    args = parser.parse_args()
    main(args)
