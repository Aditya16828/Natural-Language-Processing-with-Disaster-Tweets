import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

class CleanData:
    data = pd.DataFrame()
    wnl = WordNetLemmatizer()
    nlp = spacy.load('en_core_web_sm')

    def __init__(self, data) -> None:
        self.data = data

    def remove_link(self, sentence):
        cleaned_sentence = re.sub(r'http\S+|www\S+', '', sentence)
        return cleaned_sentence.strip()

    def extractHashtags(self, text):
        li = re.findall(r'\#([a-zA-Z0-9_]+)', text)
        return li

    def remove_words_starting_with_at(self, sentence):
        cleaned_sentence = re.sub(r'\@\w+\s*', '', sentence)
        return cleaned_sentence.strip()

    def removePunctuations(self, text):
        newText = "".join([i for i in text if i not in punctuation])
        return newText

    def removeStopwords(self, text):
        newtext = [i for i in text.split() if i not in stopwords.words("english")]
        return newtext

    def lemmatize(self, text):
        newText = [self.wnl.lemmatize(ele) for ele in text]
        return newText
    
    def findLocations(self, text):
        doc = self.nlp(text)
        locations = [entity.text for entity in doc.ents if entity.label_ == 'GPE' or entity.label_ == 'LOC']
        return locations
    
    def finalize(self, textList):
        text = " ".join(textList)
        return text
    
    def removeNA(self, li):
        ans = []
        for el in li:
            if el != 'NA':
                ans.append(el)
        return ans
    
    def extract_unique_elements(self, row):
        return [item for item in row['hashtags'] if item not in row['all_locations']]

    def clean(self):
        self.data.fillna("NA", inplace=True)
        self.data['text_nolink'] = self.data['text'].apply(lambda x:self.remove_link(x))
        self.data['hashtags'] = self.data['text_nolink'].apply(lambda x:self.extractHashtags(x))
        self.data['text_nomentions'] = self.data['text_nolink'].apply(lambda x:self.remove_words_starting_with_at(x))
        self.data['extracted_locations'] = self.data['text_nomentions'].apply(lambda x:self.findLocations(x))
        self.data['noPunctuations'] = self.data['text_nomentions'].apply(lambda x:self.removePunctuations(x))
        self.data['noStopwordsTokenized'] = self.data['noPunctuations'].apply(lambda x:self.removeStopwords(x))
        self.data['lemmatized'] = self.data['noStopwordsTokenized'].apply(lambda x:self.lemmatize(x))
        self.data['text_cleaned'] = self.data['lemmatized'].apply(lambda x:self.finalize(x).lower())

        self.data['location'] = self.data['location'].apply(lambda x:self.remove_link(x))
        self.data['location'] = self.data['location'].apply(lambda x:self.remove_words_starting_with_at(x))
        self.data['location'] = self.data['location'].apply(lambda x:self.removePunctuations(x))
        self.data['location'] = self.data['location'].apply(lambda x:self.removeStopwords(x))
        self.data['location'] = self.data['location'].apply(lambda x:self.lemmatize(x))

        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.remove_link(x))
        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.remove_words_starting_with_at(x))
        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.removePunctuations(x))
        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.removeStopwords(x))
        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.lemmatize(x))

        self.data['location'] = self.data['location'].apply(lambda x:self.finalize(x))
        self.data['keyword'] = self.data['keyword'].apply(lambda x:self.finalize(x))

        self.data['all_locations'] = self.data['location'] + ' ' + self.data['extracted_locations'].apply(lambda x: ' '.join(x))
        self.data['all_locations'] = self.data['all_locations'].apply(lambda x:x.split())

        self.data['all_locations'] = self.data['all_locations'].apply(lambda x:self.removeNA(x))
        self.data['all_locations'] = self.data['all_locations'].apply(lambda x:list(set(x)))

        self.data['extra_keywords'] = self.data.apply(self.extract_unique_elements, axis=1)
        self.data['all_keywords'] = self.data['keyword'] + ' ' + self.data['extra_keywords'].apply(lambda x:' '.join(x))
        self.data['all_keywords'] = self.data['all_keywords'].apply(lambda x:x.split())
        self.data['all_keywords'] = self.data['all_keywords'].apply(lambda x:self.removeNA(x))

        self.data['all_locations'] = self.data['all_locations'].apply(lambda x:self.finalize(x))
        self.data['all_keywords'] = self.data['all_keywords'].apply(lambda x:self.finalize(x))

        return self.data

        
