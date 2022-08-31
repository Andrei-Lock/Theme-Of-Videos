import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

import re


import time
import selenium.webdriver
from selenium import webdriver
from bs4 import BeautifulSoup as BS
import chromedriver_binary  # Adds chromedriver binary to path

import pymorphy2  # $ pip install pymorphy2
from sentence_transformers import SentenceTransformer

def make_arr(iv):
    arr = [str(row[1]) for row in df.itertuples()]
    arr.append(iv)
    return arr


def pos(word, morth=pymorphy2.MorphAnalyzer()):
    "Return a likely part of speech for the *word*."""
    return morth.parse(word)[0].tag.POS


def get_url():
    url = st.text_input(label="Введите URL видео для распознавания")
    return url


def get_theme():
    url = st.text_input(label="Введите темы видео для распознавания")
    return url


def print_predictions(preds):
    st.write(preds)

if __name__ == "__main__":
    st.write('Проверка рекомендаций тем видео на YouTube')

    URL = get_url()
    theme = get_theme()

    result = st.button('Распознать соответствие рекомендаций темам видео')

    if result:
        driver = selenium.webdriver.Chrome(executable_path='/app/theme-of-videos/chromedriver.exe')
        driver.get(URL)
        time.sleep(3)
        html = driver.page_source

        titles = []
        links = []

        soup = BS(html, "html.parser")
        videos = soup.find_all(
            "h3", {"class": "style-scope ytd-compact-video-renderer"})
        for video in videos:
            span = video.find("span", {"id": "video-title"})
            title = span['title']
            title = title.lower()
            title = re.sub(r'[^a-zA-Zа-яА-Я ]', '', title)
            titles.append(title)

        for i in range(len(titles)):
            titles[i] = titles[i].split()
            functors_pos = {'INTJ', 'PRCL', 'CONJ', 'PREP'}  # function words
            titles[i] = [title for title in titles[i]
                         if pos(title) not in functors_pos]
            titles[i] = " ".join(titles[i])

        df = pd.DataFrame({'title': titles})

        theme_ai = theme
        theme_word = 0
        while len(theme_ai.split()) < 4:
            theme_ai += " " + theme_ai.split()[theme_word]
            if theme_word == len(theme_ai.split()) - 1:
                theme_word = 0
            else:
                theme_word += 1

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/bert-base-nli-mean-tokens')

        sentence_embeddings = model.encode(make_arr(theme_ai))

        # calculate similarities (will store in array)
        scores = np.zeros(
            (sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
        for i in range(sentence_embeddings.shape[0]):
            scores[i, :] = cosine_similarity(
                [sentence_embeddings[i]],
                sentence_embeddings
            )[0]

        choose = scores[-1][:-1]
        
        ans = pd.DataFrame({'theme accuracy': choose})
        ans.index = df['title']

        print_predictions(ans)
