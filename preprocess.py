
__author__ = "Gürkan Şahin"
__date__ = "30.09.2020"
__version__ = "1.0"
__email__ = "gurkan.sahin@etiya.com"

import re
import requests
import ast

stopword_file_dir = "turkish_stopwords.txt"

def lowercase(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    data["text"] = data["text"].map(lambda x: str(x).lower())
    return data

def remove_punctuations(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    data["text"] = data["text"].map(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    return data

def remove_numbers(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    data["text"] = data["text"].map(lambda x: ''.join(i for i in str(x) if not i.isdigit()))
    return data

def remove_stop_words(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    stop_words = []
    with open(stopword_file_dir, encoding="utf-8") as fp:
        lines = fp.readlines()
    for line in lines:
        stop_words.append(line.rstrip().lower())

    for id, row in data.iterrows():
        words = row["text"].split()
        new_words = []
        for word in words:
            if word not in stop_words:
                new_words.append(word)
        data.loc[id, "text"] = " ".join(new_words) # update using "loc"
        # print(data.loc[id, "text"]) # ok
    return data

def zemberek_stemming(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    API_ENDPOINT = "http://localhost:4567/stems"
    for id, row in data.iterrows():
        stemmed_word_list = []
        for word in row["text"].split():
            _data = {
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "word": word.rstrip()
            }
            result = requests.post(url=API_ENDPOINT, data=_data)
            result = result.content.decode("UTF-8")
            result = ast.literal_eval(result)
            # print(result)
            if len(result["results"]):
                stemmed_word_list.append(result["results"][0]["stems"][0])
        data.loc[id, "text"] = " ".join(stemmed_word_list)
    return data

def first_5_char_stemming(data):
    """
    :param data: dataframe
    :return: dataframe
    """
    for id, row in data.iterrows():
        new_words = []
        for word in row["text"].split():
            if len(word) >= 5:
                new_words.append(word[:5])
            else:
                new_words.append(word)
        data.loc[id, "text"] = " ".join(new_words)  # update
        # print(data.loc[id, "text"])
    return data

def data_shuffle(data, shuffle_count):
    """
    :param data: dataframe
    :param shuffle_count: number of shuffle dataframe
    :return: dataframe
    """
    for i in range(shuffle_count):
        # shuffle the DataFrame rows
        data = data.sample(frac=1)
    return data
