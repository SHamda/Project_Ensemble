import pandas as pd
from dataset import CustomDataset
from model import TextModel
from os import listdir
from os.path import isfile, join
from transformers import BertTokenizer
import tez
import re


def clean_tweet(text):
    search = [
        "أ",
        "إ",
        "آ",
        "ة",
        "_",
        "-",
        "/",
        ".",
        "،",
        " و ",
        " يا ",
        '"',
        "ـ",
        "'",
        "ى",
        "\\",
        "\n",
        "\t",
        "&quot;",
        "?",
        "؟",
        "!",
    ]
    replace = [
        "ا",
        "ا",
        "ا",
        "ه",
        " ",
        " ",
        "",
        "",
        "",
        " و",
        " يا",
        "",
        "",
        "",
        "ي",
        "",
        " ",
        " ",
        " ",
        " ? ",
        " ؟ ",
        " ! ",
    ]

    tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(tashkeel, "", text)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"\r+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("وو", "و")
    text = text.replace("يي", "ي")
    text = text.replace("اا", "ا")

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()

    return text


def get_tokenizer(source):
    return BertTokenizer.from_pretrained(source)


def preprocess(data):
    data["label"] = data["label"].astype(int)
    data["tweetText"] = data["tweetText"].apply(clean_tweet)
    data.rename(columns={"tweetText": "text"}, inplace=True)
    return data


def add_fold_col(folds_path):
    folds = [f for f in listdir(folds_path) if isfile(join(folds_path, f))]
    i = 1
    dict_final = {"tweetID": [], "fold_number": []}
    for fold in folds:
        path_f = folds_path + "\\" + fold
        df_temp = pd.read_csv(path_f)
        id = "F_" + str(i)
        df_temp["fold_number"] = id
        dict_final["tweetID"].extend(df_temp["tweetID"].values)
        dict_final["fold_number"].extend(df_temp["fold_number"].values)
        i += 1
    df_final = pd.DataFrame(dict_final)
    return df_final


def train_model(fold, toknz_name):
    # read_data
    path = (
        r"C:\Users\slimi\Project_Ensemble\data\tweet_verification\Tweets_processed.csv"
    )
    df = preprocess(pd.read_csv(path))
    # apply 5-fold cross validation

    df_train = df[df.fold_number != fold].reset_index(drop=True)
    df_valid = df[df.fold_number == fold].reset_index(drop=True)
    tknz = get_tokenizer(toknz_name)
    train_dataset = CustomDataset(
        df_train.text, df_train.label, tokenizer=tknz, max_len=512
    )
    valid_dataset = CustomDataset(
        df_valid.text, df_valid.label, tokenizer=tknz, max_len=512
    )
    TRAIN_BS = 32
    EPOCHS = 10
    n_train_steps = int(len(df_train) / TRAIN_BS * EPOCHS)
    # you use num classes 1 because we are using BCwithLogitsLoss
    model = TextModel(num_classes=1, num_train_steps=n_train_steps)
    es = tez.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, model_path="model.bin"
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        device="cuda",
        epochs=EPOCHS,
        train_bs=TRAIN_BS,
        callbacks=[es],
    )
    return model


if __name__ == "__main__":
    folds = ["F_1", "F_2", "F_3", "F_4", "F_5"]
    for fold in folds:
        train_model(fold, "aubmindlab/bert-base-arabertv02")
