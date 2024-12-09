import os
from common.score import scorePredict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from model.OutClassificationModel import OutClassificationModel

def train_predict_model(df_train, df_test, is_predict, use_cuda, value_head, batch_size):
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    # mô hình với các tham số tối ưu hóa
    model = OutClassificationModel('roberta', 'roberta-large', num_labels=len(labels),
                                use_cuda=use_cuda, args={
                                'learning_rate': 1e-5,
                                'num_train_epochs': 3,
                                'reprocess_input_data': True,
                                'overwrite_output_dir': True,
                                'process_count': 10,
                                'train_batch_size': 4,
                                'eval_batch_size': 4,
                                'max_seq_length': 512,
                                'fp16': True,
                                'fp16_opt_level': "O1",
                                'value_head': value_head})
    # Huấn luyện mô hình
    model.train_model(df_train)

    results = ''
    if is_predict:
        # Chuẩn bị dữ liệu để dự đoán
        text_a = df_test['text_a']
        text_b = df_test['text_b']
        feature = df_test['feature']
        df_result = pd.concat([text_a, text_b, feature], axis=1)
        value_in = df_result.values.tolist()
        _, model_outputs_test = model.predict(value_in)
    else:
        # Đánh giá mô hình
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']
    
    # Dự đoán nhãn
    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))  # In ra điểm số
    return results


def predict(df_test, use_cuda, model_dir, value_head):
    # Khởi tạo mô hình từ thư mục đã lưu
    model = OutClassificationModel(model_type='roberta', model_name=os.getcwd() + model_dir, use_cuda=use_cuda, args={'value_head': value_head})
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    text_a = df_test['text_a']
    text_b = df_test['text_b']
    feature = df_test['feature']
    df_result = pd.concat([text_a, text_b, feature], axis=1)
    value_in = df_result.values.tolist()
    _, model_outputs_test = model.predict(value_in)
    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))  # In ra điểm số
