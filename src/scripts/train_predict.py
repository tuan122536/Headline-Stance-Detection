import argparse
from common.loadData import load_all_data
from model.roberta_model import OutRobertaForSequenceClassification, predict
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

feature_stance = ['polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']
feature_related = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs']
feature_all = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs',
               'polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']

def train_predict_model(df_train, df_test, is_predict, use_cuda, batch_size):  
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    # Thử mô hình Bert
    model_type = "bert"  
    model = OutClassificationModel(model_type, 'bert-base-uncased', num_labels=len(labels),
                                   use_cuda=use_cuda, args={
                                   'learning_rate': 5e-6,
                                   'num_train_epochs': 10,
                                   'reprocess_input_data': True,
                                   'overwrite_output_dir': True,
                                   'process_count': 10,
                                   'train_batch_size': batch_size,
                                   'eval_batch_size': batch_size,
                                   'max_seq_length': 512,
                                   'fp16': True,
                                   'fp16_opt_level': "O1",
                                   'early_stopping': True,
                                   'early_stopping_patience': 3,
                                   'early_stopping_threshold': 0.01})

    # Huấn luyện mô hình
    model.train_model(df_train)

    results = ''
    if is_predict:
        text_a = df_test['text_a']
        text_b = df_test['text_b']
        df_result = pd.concat([text_a, text_b], axis=1)
        value_in = df_result.values.tolist()

        # Loại bỏ externalFeature
        _, model_outputs_test = model.predict(value_in)  # Bỏ externalFeature

    else:
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']

    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))
    return results

def predict(df_test, use_cuda, model_dir):
    model = OutClassificationModel(model_type='bert', model_name=os.getcwd() + model_dir, use_cuda=use_cuda)
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()

    text_a = df_test['text_a']
    text_b = df_test['text_b']
    df_result = pd.concat([text_a, text_b], axis=1)
    value_in = df_result.values.tolist()

    # Loại bỏ externalFeature
    _, model_outputs_test = model.predict(value_in)  # Bỏ externalFeature

    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--type_class", choices=['related', 'stance', 'all'],
                        default='related', help="This parameter is used to choose the type of "
                                                "classifier (related, stance, all).")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be used if cuda is present.")

    parser.add_argument("--not_use_feature",
                        default=False,
                        action='store_true',
                        help="This parameter should be used if you don't want to train with the external features.")

    parser.add_argument("--training_set",
                        default="/data/FNC_summy_textRank_train_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="This parameter is the relative dir of the training set.")

    parser.add_argument("--test_set",
                        default="/data/FNC_summy_textRank_test_spacy_pipeline_polarity_v2.json",
                        type=str,
                        help="This parameter is the relative dir of the test set.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of the model for prediction.")
    
    parser.add_argument("--batch_size",
                        default=4,  # Giá trị mặc định
                        type=int,
                        help="This parameter is the batch size for training and evaluation.")

    main(parser)
