
__author__ = "Gürkan Şahin"
__date__ = "30.09.2020"
__version__ = "1.0"
__email__ = "gurkan.sahin@etiya.com"

"""
https://radimrehurek.com/gensim/models/ldamodel.html
"""

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import pandas as pd
import os
import pickle
import numpy as np

from preprocess import *

class Params(object):
    excel_dir = "data/data_6_class_balanced.xlsx"

    n_topic = 10 # same as train
    n_top_words = 50
    n_topic_range = 10 # 30  # upper topic value for grid search
    lda_max_iter = 50  # default
    topic_title_first_k_word = 5
    wordcloud_max_words = 100
    shuffle_count = 20

    preprocess_steps = {
        "lowercase": True,
        "remove_punctuations": True,
        "remove_numbers": True,
        "remove_stop_words": True,
        "zemberek_stemming": False,
        "first_5_char_stemming": False,
        "data_shuffle": True
    }

    vectorizer_name = "vectorizer.pickle"
    model_name = "model.pickle"

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

def read_data():
    """
    :return: dataframe
    """
    data = pd.read_excel(Params.excel_dir)
    return data

def run_preprocess(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    if Params.preprocess_steps["lowercase"]:
        data = lowercase(data)
    if Params.preprocess_steps["remove_punctuations"]:
        data = remove_punctuations(data)
    if Params.preprocess_steps["remove_numbers"]:
        data = remove_numbers(data)
    if Params.preprocess_steps["remove_stop_words"]:
        data = remove_stop_words(data)
    if Params.preprocess_steps["zemberek_stemming"]:
        data = zemberek_stemming(data) # gives connection error for long documents/dataframes
    if Params.preprocess_steps["first_5_char_stemming"]:
        data = first_5_char_stemming(data)
    if Params.preprocess_steps["data_shuffle"]:
        data = data_shuffle(data, Params.shuffle_count)
    return data

def run_lda(data):
    """
    :param data: dataframe
    :return: None, train LDA with best n_topic value and save model(s)
    """
    # Build LDA model
    texts = data["text"].map(lambda x: str(x).split()).tolist()  # list of list of word
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # save lda dictionary
    with open(os.path.join(Params.model_dir, Params.vectorizer_name), 'wb') as fp:
        pickle.dump(dictionary, fp)

    lda = LdaModel(corpus=corpus,
                   id2word=dictionary,
                   num_topics=Params.n_topic,
                   iterations=Params.lda_max_iter)
    # save lda model
    with open(os.path.join(Params.model_dir, Params.model_name), 'wb') as fp:
        pickle.dump(lda, fp)

def run_lda_grid_search(data):
    """
    :param data: dataframe
    :return: None, train LDA with grid search and find best n_topic value
    https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html
    """
    texts = data["text"].map(lambda x: str(x).split()).tolist()  # list of list of word
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_score = []
    for id, n in enumerate(range(1, Params.n_topic_range), start=1): # grid search
        lda = LdaModel(corpus=corpus,
                       id2word=dictionary,
                       num_topics=n,
                       iterations=Params.lda_max_iter)

        # The high value of topic coherence score model will be considered as a good topic model.
        coherence_model_lda = CoherenceModel(model=lda,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
        lda_coherence_score = coherence_model_lda.get_coherence()
        print("Coherence Score for n_topic:", n, " -> ", lda_coherence_score)
        coherence_score.append(lda_coherence_score)

    # find max score index
    max_index = np.argmax(coherence_score)
    print("best n_topic: ", max_index + 1)
    Params.n_topic = max_index + 1 # update n_topic value

def run_pipeline():
    data = read_data()
    data = run_preprocess(data)
    run_lda_grid_search(data) # find best n_topic value
    run_lda(data) # train lda model with best n_topic value

if __name__ == '__main__':

    run_pipeline()

