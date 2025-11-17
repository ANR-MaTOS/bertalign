"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

from bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "LaBSE"
# model_name = "/gpfsdswork/dataset/HuggingFace_Models/sentence-transformers/LaBSE"
model = Encoder(model_name)

# from bertalign.aligner import Bertalign


# replace googletrans with fasttext, specify here the path to the fasttext model for language identification
import fasttext
# lid_model = fasttext.load_model('/lustre/fswork/projects/rech/mrn/ujd84yr/FastText/lid.176.ftz')
lid_model = fasttext.load_model('/home/zpeng/scratch/MaTOS/resumeAllTHE/scripts/segment_align/lid_model/lid.176.ftz')


from bertalign.aligner import Bertalign
