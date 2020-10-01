
__author__ = "Gürkan Şahin"
__date__ = "30.09.2020"
__version__ = "1.0"
__email__ = "gurkan.sahin@etiya.com"

from matplotlib import pyplot as plt
from wordcloud import WordCloud
import logging
import pandas as pd
import pickle
import json
import os

from train_topic_modeling import run_preprocess

class Params(object):
    n_topic = 7 # same as train
    n_top_words = 50
    topic_title_first_k_word = 5
    wordcloud_max_words = 100

    with open("model/model.pickle", 'rb') as fd:
        model = pickle.load(fd)
    with open("model/vectorizer.pickle", 'rb') as fd:
        vectorizer = pickle.load(fd)

def get_topic_names(model):
    """
    :param model: trained lda model
    :return: all topic idx and names in json format
    """
    result = {}
    for (topic_id, words) in model.print_topics(num_words=Params.n_top_words):
        topic_words = words.split(" + ")
        topic_title = "_".join([word.split("*")[1].replace('"', "") for word in
                                topic_words[:Params.topic_title_first_k_word]])
        #print("topic_id: ", topic_id)
        #print("topic_title: ", topic_title)
        result[topic_id] = topic_title
    result = dict(sorted(result.items()))
    return json.dumps(result, ensure_ascii=False)

def get_number_of_topics(model):
    """
    :param model: trained lda model
    :return: number of topics obtained from train
    """
    # n_topic = 30 ile train ettik ama 20 tane topic cluster üretti, bu durumlar oluyor demekki :)
    return len(model.print_topics(num_words=Params.n_top_words))

def get_topics_includes_target_word(model, target_word):
    """
    :param model: trained lda model
    :param target_word: target word (str)
    :return: topic idx and names in json format
    """
    result = {}
    for (topic_id, words) in model.print_topics(num_words=Params.n_top_words):
        topic_words = words.split(" + ")
        topic_title = "_".join([word.split("*")[1].replace('"', "") for word in
                                topic_words[:Params.topic_title_first_k_word]])
        topic_words = [word.split("*")[1].replace('"', "") for word in topic_words]
        # print(topic_words)
        if target_word in topic_words:
            result[topic_id] = topic_title
    result = dict(sorted(result.items()))
    return json.dumps(result, ensure_ascii=False)

def get_word_and_scores_given_topic_id(model, topic_id):
    """
    :param model: trained lda model
    :param topic_id: topic_id (starts from 0)
    :return: words and scores in json format
    """
    result = {}
    if topic_id < Params.n_topic:
        word_score = model.print_topic(topic_id, topn=Params.n_top_words)
        word_score = word_score.split(" + ")
        for x in word_score:
            score, word = x.split("*")
            word = str(word.replace('"', ""))
            score = float(score)
            result[word] = score
        #print(result)
        return json.dumps(result, ensure_ascii=False)
    else:
        logging.error("topic_id [" + str(0) + "-" + str(Params.n_topic - 1) + "] arasında olmalı !")

def predict_topic(model, data):
    """
    :param model: trained lda model
    :param data: dataframe
    :return: predicted topic_ids, topic_probs in list of dict format
    """
    data = run_preprocess(data)

    data = data.sort_index() # bu önemli, yoksa sıra değişiyor predict loop'u içinde !
    corpus = data["text"].map(lambda x: str(x).split()).tolist()  # list of list of word
    corpus_ = [Params.vectorizer.doc2bow(text) for text in corpus]
    predict = [model.get_document_topics(text) for text in corpus_]
    result_list = []
    for pred in predict:
        (topic_id, prob) = max(pred, key=lambda item: item[1])
        #print(topic_id, prob)
        result_list.append({"topic_id": topic_id, "topic_prob": prob})
    return result_list

def get_wordcloud_given_topic_id(model, topic_id):
    """
    :param model: trained lda model
    :param topic_id: topic id
    :return: None, save wordcloud img in "plot" dir for given topic_id
    """
    if topic_id < Params.n_topic:
        cloud = WordCloud(background_color='white',
                          width=2500,  # 1800
                          height=1800,  # 1000
                          max_words=Params.wordcloud_max_words,  # 10
                          colormap='tab10',
                          prefer_horizontal=1.0,
                          random_state=39) # random_state: her seferinde aynı img üretiliyor
        topics = model.show_topic(topicid=topic_id, topn=Params.n_top_words)
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        cloud.generate_from_frequencies(dict(topics), max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(topic_id), fontdict=dict(size=16))
        plt.gca().axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        # plt.show() # ok
        plt.savefig(os.path.join("plot", "topic_" + str(topic_id) + ".png"))
    else:
        logging.error("topic_id [" + str(0) + "-" + str(Params.n_topic - 1) + "] arasında olmalı !")

if __name__ == '__main__':

    Params.n_topic = get_number_of_topics(Params.model)
    print("n_topic: ", Params.n_topic)

    # get all topics (topic ids, and topic title consists of first 5 word)
    """
    result = get_topic_names(Params.model)
    print(result)
    """

    # get number of topics
    """
    result = get_number_of_topics(Params.model)
    print(result)
    """

    # get topics includes target word
    """
    result = get_topics_includes_target_word(Params.model, "başkan")
    print(result)
    """

    # get topic words and scores for given topic_id
    """
    result = get_word_and_scores_given_topic_id(Params.model, topic_id=6)
    print(result)
    """

    # predict topics of given sentences
    """
    texts = ["ddd GÜRKAN ŞAHİN ", "merhaba benim adım gürkan", "selam lar hacıcığım"]
    data = pd.DataFrame(texts, columns=["text"])
    result = predict_topic(Params.model, data)
    print(result)
    """

    # get/save wordcloud for given topic_id
    """
    get_wordcloud_given_topic_id(Params.model, topic_id=4)
    """
