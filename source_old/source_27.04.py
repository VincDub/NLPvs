import pdfminer
from pdfminer import high_level
import os
import pandas as pd

import unicodedata
import re
import string
import time
import matplotlib.pyplot as plt
import joblib
import sys
import os
from tkinter import *
from tkinter import filedialog
from datetime import date

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

        sac_de_mots.append(sac_de_mots_grp(texte))

    return sac_de_mots


def sac_de_mots_grp(txt):

    txt_grp = ""

    for phrase in txt:
        for mot in phrase:
            txt_grp += (mot +" ") 

    return txt_grp



def vectoriser_vrac(dir_src):

    preprocess = preprocess_pdf_masse(dir_src)

    sac = sac_de_mots(preprocess[1])

    tfconv = TfidfVectorizer(input = 'content',max_features = 1000,ngram_range=(1,3),preprocessor=None,lowercase=False, tokenizer=None)
    tfidf = tfconv.fit_transform(sac)

    return tfidf

def vectoriser_avec_noms(dir_src):

    preprocess = preprocess_pdf_masse(dir_src)

    tfconv = TfidfVectorizer(input = 'content',max_features = 1000,ngram_range=(1,3),preprocessor=None,lowercase=False, tokenizer=None)

    noms = preprocess[0]
    textes_vec = list()

    sac = sac_de_mots(preprocess[1])

    textes_vec.append(tfconv.fit_transform(sac))

    return noms, textes_vec



def entrainer_depuis_corpus(dir_src,n):

    corpus_vect = vectoriser_vrac(dir_src)

    modele_custom = KMeans(n_clusters=n).fit(corpus_vect)

    joblib.dump(modele_custom,('MODEL/custom/model_'+ str(date.today()) +'.sav'))

    return modele_custom


def tester_texte(dir_src,dir_test,n,entrainer,dir_model):

    t_0 = time.time()

    if entrainer == False :

        modele = joblib.load(dir_model)

        txt_= vectoriser_avec_noms(dir_test)
        txt_vec = txt_[1]
        noms = txt_[0]

        resultat_class = list()

        for i in range (0,len(noms)):
            
            for texte in txt_vec:   
            
                resultat = modele.predict(texte)

            resultat_class.append((noms[i],resultat[i]))

    else:

        modele = entrainer_depuis_corpus(dir_src,n)

        txt_= vectoriser_avec_noms(dir_test)
        txt_vec = txt_[1]
        noms = txt_[0]

        resultat_class = list()

        for i in range (0,len(noms)):

            for texte in txt_vec:
            
                resultat = modele.predict(texte)

            resultat_class.append((noms[i],resultat[i]))

    t_1 = time.time()

    delta_t = "effectué en " + str((t_1 - t_0))[:5] + " secondes."

    return resultat_class , delta_t


def start(frame,bench,src,test,n,train,mdl):

    dir_test = test.cget("text")
    dir_mdl = mdl.cget("text")
    n_cl = int()
    dir_src = str()

    if train == False:

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=False,dir_model=dir_mdl)
        d = 0
        result_bench = str(result[1])
        bench["text"] = result_bench

        for resultat in result[0]:

            nom_indiv = str(resultat[0]).split('/')

            result_txt = nom_indiv[-1] + ': cluster N° '+ str(resultat[1])

            frame.create_text(0,(10+d),text=result_txt, fill='red', anchor=NW)
            d += 20
        
    else:

        n_cl = n.get()
        dir_src = src.cget("text")

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=True,dir_model=dir_mdl)
        d = 0
        result_bench = str(result[1])
        bench["text"] = result_bench

        for resultat in result[0]:

            nom_indiv = str(resultat[0]).split('/')

            result_txt = nom_indiv[-1] + ': cluster N° '+ str(resultat[1])

            frame.create_text(0,(10+d),text=result_txt, fill='red', anchor=NW)
            d += 20

def chemin_dossier_src(frame,r):
    dossier_source = filedialog.askdirectory(title="Choisir un dossier de données d'entraînement")

    r["text"] = str(dossier_source)

def chemin_dossier_test(frame,r):
    dossier_source = filedialog.askdirectory(title="Choisir un dossier de données à catégoriser")

    r["text"] = str(dossier_source)


def chemin_fichier(frame,r):
    fichier_source = filedialog.askopenfilename(title="Choisir un modèle",filetypes=[('fichier sav','.sav')])

    r["text"] = str(fichier_source)

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

fen = Tk()
fen.title("NLPvs beta")

frame_input = Frame(fen, borderwidth = 3, relief = GROOVE)
frame_input.pack(pady = 20)

frame_output = Canvas(fen, borderwidth = 3, bg = 'white')
frame_output.pack(side = RIGHT, pady = 20)

bench = LabelFrame(fen,text = "Temps d'exécution",borderwidth = 3, relief = GROOVE)
bench.pack()

bench_txt = Label(bench, text="", bg='ivory')
bench_txt.pack()

#data à clusteriser

dir_test = LabelFrame(frame_input, text='Dossier contenant les fichiers à classer', padx = 30, pady = 30)
dir_test.pack(fill='both', expand = 'yes')

btn_test = Button(dir_test,text='Sélectionner un dossier', command=(lambda : chemin_dossier_test(dir_test, r_test)))
btn_test.pack()

r_test = Label(dir_test, text = "", bg = 'ivory')
r_test.pack(padx=20, pady=20)

#selection du modele

dir_mdl = LabelFrame(frame_input, text='Sélection du modèle de classification', padx = 30, pady = 30)
dir_mdl.pack(fill='both', expand = 'yes')

btn_mdl = Button(dir_mdl,text='Sélectionner un fichier', command=(lambda : chemin_fichier(dir_mdl, r_mdl)))
btn_mdl.pack()

r_mdl = Label(dir_mdl, text = "", bg = 'ivory')
r_mdl.pack(padx=20, pady=20)

#entrainement

train = LabelFrame(frame_input, text='''Classification selon mon propre modèle''', padx = 30, pady = 30)
train.pack(fill='both', expand = 'yes')
#choix
entrain = False

train_true = Radiobutton(train, text="oui", variable=entrain, value=True)
train_true.pack()
train_false = Radiobutton(train, text="non", variable=entrain, value=False)
train_false.pack()
#nombre de clusters

clusters = LabelFrame(train, text='Nombre de clusters', padx = 30, pady = 30)
clusters.pack(fill='both', expand = 'yes')

n_clusters = Spinbox(clusters, from_=2, to=50)
n_clusters.pack()

# data d'entraînement

dir_src = LabelFrame(train, text='''Dossier contenant les fichiers d'entraînement (facultatif)''', padx = 30, pady = 30)
dir_src.pack()

btn_src = Button(dir_src,text='Sélectionner un dossier', command=(lambda :chemin_dossier_src(dir_src, r_src)))
btn_src.pack()

r_src = Label(dir_src, text = "", bg = 'ivory')
r_src.pack(padx=20, pady=20)


btn_start = Button(fen, text="Lancer la classification", command=(lambda: start(frame_output, bench_txt, r_src, r_test, n_clusters, train=entrain, mdl = r_mdl)))
btn_start.pack(side = RIGHT, padx = 20, pady= 20)

btn_quitter = Button(fen, text="Quitter", command=fen.quit)
btn_quitter.pack(side = LEFT)

fen.mainloop()
