import pickle

#import torch
#import transformers as tfm
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

production = pickle.load(open(r"eulascript/production.pth", 'rb'))

#requirements 

# Bag of word
words_counts = production["words_counts"]
WORDS_TO_INDEX = production["WORDS_TO_INDEX"]
DICT_SIZE = production["DICT_SIZE"]
classifier_mybag = production["classifier_mybag"]

# TF-IDF 
tfidf_vectorizer = production["tfidf_vectorizer"]
classifier_tfidf = production["classifier_tfidf"]

# BERT
#classifier_bert = production["classifier_bert"]
#max_input_length = production["max_input_length"]

# For DistilBERT:
# model_class, tokenizer_class, pretrained_weights = (tfm.DistilBertModel, tfm.DistilBertTokenizer, 'distilbert-base-uncased')

## For BERT :
#model_class, tokenizer_class, pretrained_weights = (tfm.BertModel, tfm.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
#tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#max_input_length = tokenizer.max_model_input_sizes['distilbert-base-uncased']
#max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

#model = model_class.from_pretrained(pretrained_weights)
#model = model.to(device)

def text_prepare(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) # delete stopwords from text
    return text

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    for item in text.split():
        if item in words_to_index.keys():
            result_vector[words_to_index[item]] += 1
    return result_vector



def get_meta_data(eula, vocab = None):
    if not vocab :
        vocab = tfidf_vectorizer.vocabulary_

    return [{text : vocab.get(text.lower(), 0) for text in clause.split()} for clause in eula]
    
def mybag_predict(eula):
    if type(eula) == str :
        eula = [eula]
    else :
        assert type(eula) == list

    output = []
    for clause in eula :
        clause = text_prepare(clause)
        vec = my_bag_of_words(clause , WORDS_TO_INDEX, DICT_SIZE)
        y = classifier_mybag.predict_proba([vec])[0]
        output.append(list(y))
    
    return output

def tfidf_predict(eula):
    if type(eula) == str :
        eula = [eula]
    else :
        assert type(eula) == list
    output = []
    for clause in eula :
        vec = tfidf_vectorizer.transform([text_prepare(clause)])
        y = classifier_tfidf.predict_proba(vec)[0]        
        output.append(list(y))

    return output


def bert_predict(eula):

    """
    if type(eula) == str :
        eula = [eula]
    else :
        assert type(eula) == list

    output = []
    model.eval()

    for clause in eula :
  
        tokens = tokenizer.tokenize(clause)
        tokens = tokens[:max_input_length-2]
        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            pooled_output, _ = model(tensor)
        vec = pooled_output[:,0,:].cpu().numpy()
        y = classifier_bert.predict(vec)[0]
        output.append(y)

    return output
    """
    return tfidf_predict(eula)


def get_predict_method(model_name):
    assert model_name in ["bag_of_word","tf_idf","bert"]
    if model_name == "bag_of_word" :
        return mybag_predict
    elif model_name == "tf_idf" :
        return tfidf_predict
    else :
        return bert_predict

    
   
