# import des bibliothèques de RNN
from tensorflow import keras
import tensorflow as tf

import torch
from transformers import CamembertModel, CamembertTokenizer

# racinisation des mots français
from nltk.stem.snowball import FrenchStemmer

# extracteur de texte pour les fichiers pdf
from pdfminer import high_level

# Vectoriseur TFIDF avant l'entrée dans le RNN
from sklearn.feature_extraction.text import TfidfVectorizer

# modules standards
import os
import string
import re
import unicodedata
# récupération des chemins des fichiers à parser

def fichiers_cibles(chemin_dossier):

        dossier = str(chemin_dossier)
        liste_fichiers = os.listdir(dossier)
        noms_fichiers = list()

        for entry in liste_fichiers:
            chemin = os.path.join(dossier,entry)
            if os.path.isdir(chemin):
                noms_fichiers = noms_fichiers + fichiers_cibles(chemin)
            else:
                noms_fichiers.append(chemin)   

        return noms_fichiers
    

# extraction du texte des fichiers à parser
def pdf_en_str(chemin_dossier):

    chemins_fichiers = fichiers_cibles(chemin_dossier)
    textes_fichiers,chemins_fichiers_sortie = [],[]

    for chemin in chemins_fichiers:
        try:
            textes_fichiers.append(high_level.extract_text(chemin,"",None,0,True,'utf-8',None))
            chemins_fichiers_sortie.append(chemin)

        except TypeError:
            
            print('la structure du fichier ' + entry + ' est invalide !')

    return chemins_fichiers_sortie, textes_fichiers

# nettoyage additionnel
def nettoyage_textes(textes_fichiers):

    textes_fichiers_clean = []

    for texte in textes_fichiers:

        texte = "".join(texte.splitlines())
        texte = unicodedata.normalize('NFKD', texte).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        texte = texte.lower()
        rm_ponctuation = str.maketrans('','',string.punctuation)
        texte = texte.translate(rm_ponctuation)
        texte = "".join([i for i in texte if not i.isdigit()])
        texte = re.sub(u'\u2019',u"\u0027", texte)
        texte = re.sub(u'\u200b',"", texte)
        textes_fichiers_clean.append(texte)

    return textes_fichiers_clean

# tokenisation des textes pour le vectoriseur
def tokeniseur(texte):

    stop_words = []

    f = open(('stopwordsfr.txt'),"r")

    for ligne in f.readlines():
        stop_words.append(ligne[:-1])
    

    tokens = [mot for mot in texte.split() if (len(mot)>3) and (mot not in stop_words)]

    blank = [""]
    n_tk = [token for token in tokens if (token not in blank)]

    stemmer=FrenchStemmer()
    n_tk = [stemmer.stem(token) for token in n_tk]

    return n_tk


# Calcul de la matrice TFIDF
def vectorisation(textes_fichiers):

    vectoriseur = TfidfVectorizer(input = 'content',max_df=0.8, min_df=0.2,max_features = 9000000,ngram_range=(2,2),preprocessor=None,lowercase=False,tokenizer=tokeniseur)
    matrice_tfidf= vectoriseur.fit_transform(textes_fichiers)

    vocabulaire = vectoriseur.get_feature_names()

    return matrice_tfidf, vocabulaire


# Fonction principale
def main(chemin_dossier):

    pdf_str = pdf_en_str(chemin_dossier)
    pdf_str = nettoyage_textes(pdf_str[1])

    print(vectorisation(pdf_str))


main('PDF')