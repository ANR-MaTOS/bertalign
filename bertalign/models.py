from bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "LaBSE"
# model_name = "/gpfsdswork/dataset/HuggingFace_Models/sentence-transformers/LaBSE"
model = Encoder(model_name)


# from bertalign.aligner import Bertalign
# replace googletrans with fasttext, specify here the path to the fasttext model for language identification
import os
import fasttext
import urllib.request
from pathlib import Path

LID_MODEL_PATH = "cache/lid.176.ftz"
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def load_lid_model():
    if not os.path.exists(LID_MODEL_PATH):
        Path(os.path.dirname(LID_MODEL_PATH)).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LID_MODEL_URL, LID_MODEL_PATH)
    lid_model = fasttext.load_model(LID_MODEL_PATH)
    return lid_model

lid_model = load_lid_model()
