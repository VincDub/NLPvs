import pdfminer
from pdfminer import high_level
import os
import pandas as pd

import unicodedata
import re
import string
import time

#import tensorflow as tf
#from tensorflow import keras

import nltk
from nltk import tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def fichiers_cibles(chm):

        dossier = str(chm)
        liste_fichiers = os.listdir(dossier)
        noms_fichiers = list()
        for entry in liste_fichiers:
            chemin = os.path.join(dossier,entry)
            if os.path.isdir(chemin):
                noms_fichiers = noms_fichiers + fichiers_cibles(chemin)
            else:
                noms_fichiers.append(chemin)          
        return noms_fichiers


def nettoyage_1(txt):

    txt_n1 = list()

    for texte in txt:
        texte = "".join(texte.splitlines())
        texte = unicodedata.normalize('NFD', texte).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        txt_n1.append(texte)

    return txt_n1

def phrases(txt):

    phrases = list()

    for texte in txt:
        phrases.append(tokenize.sent_tokenize(texte))
    
    return phrases

def nettoyage_phrases_aux(txt):

    phrases_n = list()

    for phrase in txt:
            phrase = phrase.lower()
            rm_ponctuation = str.maketrans('','', string.punctuation)
            phrase = phrase.translate(rm_ponctuation)
            phrases_n.append(phrase)
    
    return phrases_n

def nettoyage_phrases(txt):

    texte_n = list()

    for texte in txt:
        texte = nettoyage_phrases_aux(texte)
        texte_n.append(texte)

    return texte_n


def nettoyage_2_aux(txt):

    mots_phrases = list()

    for phrase in txt:
        #phrase = re.sub('[^a-z_]', ' ', phrase)
        stop_words = set(stopwords.words("french"))

        #tk = word_tokenize(phrase)

        tk = [mot for mot in phrase.split() if (len(mot)>2) and (mot not in stop_words)]
        #phrase = [word for word in tk if word not in stop_words]

        stemmer=FrenchStemmer()
        #lm = FrenchLefffLemmatizer()
        tk = [stemmer.stem(token) for token in tk]
        #tk = [lm.lemmatize(token) for token in tk]
        blank = ""
        n_tk = [token for token in tk if (token not in blank)]

        mots_phrases.append(n_tk)
        

    return mots_phrases

def nettoyage_2(txt):

    texte_n2 = list()

    for texte in txt:
        texte = nettoyage_2_aux(texte)
        texte_n2.append(texte)

    return texte_n2

def trouver_hapaxes(txt):

    hapaxes = list()
    union_phrases = list()

    for texte in txt:
        for phrase in texte:
            union_phrases += phrase
        hapaxes.append(FreqDist.hapaxes(union_phrases))

    return hapaxes


def preprocess_pdf_masse(dir):

    noms_fichiers = fichiers_cibles(dir)
    textes_fichiers = list()
    
    for entry in noms_fichiers:
        textes_fichiers.append(pdfminer.high_level.extract_text(entry,"",None,0,True,'utf-8',None))
    
    textes_fichiers = nettoyage_1(textes_fichiers)
    phrases_fichiers = nettoyage_phrases(phrases(textes_fichiers))

    phrases_fichiers_tk = nettoyage_2(phrases_fichiers)
    
    return noms_fichiers, phrases_fichiers_tk

'''
    nbr_phrases = list()

    for ensemble_phrases in phrases_fichiers:
        nbr_phrases.append(len(ensemble_phrases))

    #hapaxes = trouver_hapaxes(phrases_fichiers_tk)

    pdf_dict = {'Titre' :noms_fichiers, 'Nombre de phrases' :nbr_phrases, 'Texte' :phrases_fichiers_tk}
    pdf_df = pd.DataFrame(pdf_dict)

    pdf_df.to_csv("CSV/output.csv", sep=',',index=False)

    t_1 = time.time()

    delta_t = "éxécuté en " + (str(t_1 - t_0))[:5] + " secondes"

    return pdf_df, delta_t
''' 
def sac_de_mots(txt):

    sac_de_mots = list()

    for texte in txt:

        sac_de_mots.append(vectoriser_entrainer_grp(texte))

    return sac_de_mots


def vectoriser_entrainer_grp(txt):

    txt_grp = ""

    for phrase in txt:
        for mot in phrase:
            txt_grp += (mot +" ") 

    return txt_grp



def vectoriser_entrainer(dir_src,n):

    sac = sac_de_mots(preprocess_pdf_masse(dir_src)[1])

    tfconv = TfidfVectorizer(input = 'content',max_features = 1000,ngram_range=(1,3),preprocessor=None,lowercase=False, tokenizer=None)
    tfidf = tfconv.fit_transform(sac)

    kmeans = KMeans(n_clusters=n).fit(tfidf)

    return kmeans, tfconv


def tester_texte(dir_src,dir_test,n):

    t_0 = time.time()

    modele, vect = vectoriser_entrainer(dir_src,n)[0],vectoriser_entrainer(dir_src,n)[1]

    txt_test = sac_de_mots(preprocess_pdf_masse(dir_test)[1])

    resultat_class = modele.predict(vect.transform(txt_test))

    t_1 = time.time()

    delta_t = "effectué en " + str((t_1 - t_0))[:5] + " secondes."

    return resultat_class , delta_t

print(tester_texte("PDF","TEST",4))