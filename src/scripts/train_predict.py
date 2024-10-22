import argparse
import torch  # Thêm thư viện PyTorch
from common.loadData import load_all_data
from model.roberta_model import train_predict_model, predict

feature_stance = ['polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']
feature_related = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs']
feature_all = ['cosine_similarity', 'max_score_in_position', 'overlap', 'soft_cosine_similarity', 'bert_cs',
               'polarityClaim_nltk_neg', 'polarityClaim_nltk_pos', 'polarityBody_nltk_neg', 'polarityBody_nltk_pos']

def main(parser):
    args = parser.parse_args()

    type_class = args.type_class
    use_cuda = args.use_cuda
    not_use_feature = args.not_use_feature
    training_set = args.training_set
    test_set = args.test_set
    model_dir = args.model_dir
    batch_size = args.batch_size  # Lấy giá trị batch_size từ đối số
    feature = []
    
    # Kiểm tra xem GPU có khả dụng không
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    if not not_use_feature:
        if type_class == 'stance':
            feature = feature_stance
        elif type_class == 'related':
            feature = feature_related
        elif type_class == 'all':
            feature = feature_all

    # Chỉnh sửa đường dẫn đến tập dữ liệu
    training_set = training_set.replace("/data/", "/content/apex/Headline-Stance-Detection/")
    test_set = test_set.replace("/data/", "/content/apex/Headline-Stance-Detection/")

    # Tải dữ liệu huấn luyện và kiểm tra
    df_train = load_all_data(training_set, type_class, feature)
    df_test = load_all_data(test_set, type_class, feature)

    if model_dir == "":
        # Bắt đầu huấn luyện mô hình
        print("Bắt đầu huấn luyện mô hình...")
        train_predict_model(df_train, df_test, True, use_cuda, len(feature), batch_size, device)  # Thêm device
    else:
        # Dự đoán với mô hình đã lưu
        predict(df_test, use_cuda, model_dir, len(feature), device)  # Thêm device

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
