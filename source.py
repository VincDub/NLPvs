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
import datetime
import tkinter.font as tkFont 

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

def nettoyage_sans_phrases(txt):

    txt_ss = list()

    for texte in txt:
        texte = "".join(texte.splitlines())
        texte = unicodedata.normalize('NFD', texte).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        texte = texte.lower()
        rm_ponctuation = str.maketrans('','', string.punctuation)
        texte = texte.translate(rm_ponctuation)
        txt_ss.append(texte)

    return txt_ss

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


def token_aux(txt):

    mots_phrases = list()

    for phrase in txt:

        stop_words = set(stopwords.words("french"))


        tk = [mot for mot in phrase.split() if (len(mot)>2) and (mot not in stop_words)]

        blank = ""
        n_tk = [token for token in tk if (token not in blank)]

        mots_phrases.append(n_tk)
        

    return mots_phrases

def token_pour_tfidf(txt):

    stop_words = set(stopwords.words("french"))

    tk = [mot for mot in txt.split() if (len(mot)>2) and (mot not in stop_words)]

    blank = ""
    n_tk = [token for token in tk if (token not in blank)]

    stemmer=FrenchStemmer()
    n_tk = [stemmer.stem(token) for token in n_tk]

    return n_tk

def token_sans_racin_pour_tfidf(txt):

    stop_words = set(stopwords.words("french"))

    tk = [mot for mot in txt.split() if (len(mot)>2) and (mot not in stop_words)]

    blank = ""
    n_tk = [token for token in tk if (token not in blank)]

    return n_tk

def racin_pour_tfidf(lst):

    stemmer=FrenchStemmer()
    n_tk = [stemmer.stem(token) for token in lst]

    return n_tk

def token(txt):

    texte_n2 = list()

    for texte in txt:
        texte = token_aux(texte)
        texte_n2.append(texte)

    return texte_n2

def racin_aux(txt):

    mots_phrases = list()

    for phrase in txt:

        tk = [mot for mot in phrase]

        stemmer=FrenchStemmer()
        tk = [stemmer.stem(token) for token in tk]
        blank = ""
        n_tk = [token for token in tk if (token not in blank)]

        mots_phrases.append(n_tk)
        

    return mots_phrases

def racin(txt):

    texte_n2 = list()

    for texte in txt:
        texte = racin_aux(texte)
        texte_n2.append(texte)

    return texte_n2

def token_et_racin(txt):

    texte = racin(token(txt))

    return texte

#####

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

#####


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

    #phrases_fichiers_tk = nettoyage_2(phrases_fichiers)
    fichiers_clean = nettoyage_sans_phrases(textes_fichiers)

    fichiers_tk_nn_racin = token(fichiers_clean)

    fichiers_tk = racin(fichiers_tk_nn_racin)

    #phrases_fichiers_tk_nn_racin = token(phrases_fichiers)

    #phrases_fichiers_tk = racin(phrases_fichiers_tk_nn_racin)
 
    return noms_fichiers, fichiers_tk, fichiers_tk_nn_racin, fichiers_clean #,phrases_fichiers_tk, phrases_fichiers_tk_nn_racin, phrases_fichiers

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



'''def vectoriser_vrac(dir_src):

    preprocess = preprocess_pdf_masse(dir_src)

    sac = sac_de_mots(preprocess[1])

    tfconv = TfidfVectorizer(input = 'content',max_features = 9000000,ngram_range=(1,3),preprocessor=None,lowercase=False, tokenizer=None)
    tfidf = tfconv.fit_transform(sac)

    return tfconv'''

def vectoriser_avec_noms(dir_src,vectoriz):

    preprocess = preprocess_pdf_masse(dir_src)

    noms = preprocess[0]
    textes_vec = list()

    sac = preprocess[3]

    textes_vec.append(vectoriz.transform(sac))

    return noms, textes_vec
'''
def sepa(txt):

    brut = list()
    for texte in txt:
        for phrase in texte:
            for mot in phrase:
                brut.append(mot)
    
    return brut

def j_phrases(txt):

    liste_textes = list()

    for texte in txt:
        for phrase in texte:
            for mot in phrase:
                liste_mots= list()
                liste_mots.extend(mot)
        
        liste_textes.append(liste_mots)

    return liste_textes
'''
def entrainer_depuis_corpus(dir_src,n):

    preprocess = preprocess_pdf_masse(dir_src)

    sac2 = preprocess[3]

    ss_racin = []
    racin = []

    for txt in sac2:
        ss_racin.extend(token_sans_racin_pour_tfidf(txt))
        racin.extend(token_pour_tfidf(txt))

    vocab = pd.DataFrame({'words': ss_racin}, index = racin)

    tfconv = TfidfVectorizer(input = 'content',max_df=0.8, min_df=0.2,max_features = 9000000,ngram_range=(1,3), token_pattern=None,analyzer='word',preprocessor=None,lowercase=False, tokenizer=token_pour_tfidf)
    tfconv = tfconv.fit(sac2)

    corpus_vect = tfconv.transform(sac2)

    modele_custom = KMeans(n_clusters=n, init='k-means++').fit(corpus_vect)

    cd = 'MODEL'
    t = str(datetime.datetime.now()).split('.')
    path= os.path.join(cd,t[0])
    os.mkdir(path)

    termes = tfconv.get_feature_names()
    
    joblib.dump(modele_custom,(path + '/model.sav'))
    joblib.dump(tfconv,(path + '/vect.sav'))
    joblib.dump(n,(path + '/nb_clust.sav'))
    joblib.dump(vocab,(path + '/vocab.sav'))
    joblib.dump(termes,(path + '/termes.sav'))

    return modele_custom,tfconv,vocab


def tester_texte(dir_src,dir_test,n,entrainer,dir_model):

    t_0 = time.time()

    if entrainer == False :

        modele = joblib.load(dir_model+'/model.sav')
        vectoriseur = joblib.load(dir_model+'/vect.sav')
        termes = joblib.load(dir_model+'/termes.sav')
        vocab = joblib.load(dir_model+'/vocab.sav')

        txt_= vectoriser_avec_noms(dir_test,vectoriseur)
        txt_vec = txt_[1]
        noms = txt_[0]

        resultat_class = list()

        for i in range (0,len(noms)):
            
            for texte in txt_vec:   
            
                resultat = modele.predict(texte)

            resultat_class.append((noms[i],resultat[i]))

    else:

        entrainement = entrainer_depuis_corpus(dir_src,n)
        modele = entrainement[0]
        vectoriseur = entrainement[1]
        vocab = entrainement[2]

        termes = vectoriseur.get_feature_names()

        txt_= vectoriser_avec_noms(dir_test, vectoriseur)
        txt_vec = txt_[1]
        noms = txt_[0]

        resultat_class = list()

        for i in range (0,len(noms)):

            for texte in txt_vec:
            
                resultat = modele.predict(texte)

            resultat_class.append((noms[i],resultat[i]))

    t_1 = time.time()

    delta_t = "effectué en " + str((t_1 - t_0))[:5] + " secondes."

    return resultat_class , delta_t, modele, termes, vocab


def start(frame,bench,src,test,n,entrain,mdl,cl):

    dir_test = test.cget("text")
    dir_mdl = mdl.cget("text")
    n_cl = int()
    dir_src = str()

    train = entrain.get()

    if train == False:

        n_cl = joblib.load(dir_mdl+'/nb_clust.sav')

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=False,dir_model=dir_mdl)

        result_clusters = clusters_en_features(result[2],n_cl,result[3],result[4])

        result_bench = str(result[1])
        bench["text"] = result_bench

        for key in result_clusters:
            cl.insert(END,(str(key) + str(result_clusters[key])))

        for resultat in result[0]:

            nom_indiv = str(resultat[0]).split('/')

            result_txt = nom_indiv[-1] + ' affecté au cluster N° '+ str(resultat[1])

            frame.insert(END,result_txt)

        
        
    else:

        n_cl = int(n.get())
        dir_src = src.cget("text")

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=True,dir_model=dir_mdl)

        result_clusters = clusters_en_features(result[2],n_cl,result[3],result[4])

        result_bench = str(result[1])
        bench["text"] = result_bench

        for key in result_clusters:
            cl.insert(END,(str(key) + str(result_clusters[key])))

        for resultat in result[0]:

            nom_indiv = str(resultat[0]).split('/')

            result_txt = nom_indiv[-1] + ' affecté au cluster N° '+ str(resultat[1])

            frame.insert(END,result_txt)

def clusters_en_features(mdl,n,termes,vocab):

    order_centroids = mdl.cluster_centers_.argsort()[:, ::-1]

    blank =['',',']
    tm = []

    for term in termes:
        term.split(' ')
        if term not in blank:
            tm.append(term)

    clusters_dict = dict()
    for i in range(n):
        mots_par_clusters = []

        for ind in order_centroids[i, :4]:
            mot = (' %s' % vocab.loc[tm[ind].split(' ')].values.tolist()[0][0]).encode('ascii','ignore').decode('utf-8','ignore')
            
            '''
            accents = {
                "": "’"
            }

            for y in accents.values():
                mot = mot.replace(y,'')
            '''
            mots_par_clusters.append(mot)

        clusters_dict['Cluster N° ' + str(i) +':'] = mots_par_clusters
    
    return clusters_dict



def chemin_dossier_src(r):
    dossier_source = filedialog.askdirectory(title="Choisir un dossier de données d'entraînement")

    r["text"] = str(dossier_source)

def chemin_dossier_test(r):
    dossier_source = filedialog.askdirectory(title="Choisir un dossier de données à catégoriser")

    r["text"] = str(dossier_source)


def chemin_dossier_mdl(r):
    dossier_source = filedialog.askdirectory(title="Choisir un dossier de modèle")

    r["text"] = str(dossier_source)

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

### GUI ###


fen = Tk()
fen.title("NLPvs beta")
fen.configure(background = '#232323')
fen.geometry('1100x850')

fontdemarrer = tkFont.Font(family='Arial',size=50)
fontcredits = tkFont.Font(family='Arial',size=8)

frame_input = Frame(fen, borderwidth = 0, relief = FLAT, bg = '#232323')
frame_input.pack(side = LEFT, padx = 20)

ligne = Canvas(fen,bg= '#232323', height=700, borderwidth=0, width=50)
ligne.create_line(25,25,25,675,width=2,fill='#e6e6e6')
ligne.config(highlightthickness = 0, highlightbackground = "#232323", highlightcolor='#232323')
ligne.pack(side=LEFT, padx=70)


frame_droite = Frame(fen, borderwidth = 0, relief = FLAT, bg = '#232323')
frame_droite.pack(side=LEFT, padx=20)

titre = Label(frame_input,text='NLPvs beta',fg='#e6e6e6', bg ='#232323',font=fontdemarrer)
titre.pack()
credit = Label(frame_input,text='par Vincent DUBOIS',fg='#e6e6e6', bg ='#232323',font=fontcredits)
credit.pack()

frame_2 = Frame(frame_droite, borderwidth = 0, relief = FLAT, bg = '#232323')
frame_2.pack(pady=20)

frame_log = LabelFrame(frame_2, text = 'Documents clusterisés', borderwidth = 0, relief = FLAT, bg = '#232323')
frame_log.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
frame_log.pack(padx = 0)

frame_output = Listbox(frame_log, borderwidth = 0, bg = '#e6e6e6', fg = '#FF0000', width=300)
frame_output.pack(fill = X, padx=0)

frame_cl = LabelFrame(frame_2, text = 'Clusters', borderwidth = 0, relief = FLAT, bg = '#232323')
frame_cl.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
frame_cl.pack(fill=X, pady = 20)

cl_output = Listbox(frame_cl, borderwidth = 0, bg = '#e6e6e6', fg = '#FF0000',)
cl_output.pack(fill=X,padx=0)

bench_et_start = Frame(frame_2, relief = FLAT, bg = '#232323')
bench_et_start.pack(fill=X, padx=0)

bench = LabelFrame(bench_et_start,text = "Temps d'exécution",borderwidth = 0, relief = FLAT, bg = '#232323')
bench.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
bench.pack(pady=20, fill=X)

bench_txt = Label(bench, text="", bg='#e6e6e6',borderwidth=0)
bench_txt.pack(fill=X, padx=0, pady=0)

#data à clusteriser

dir_test = LabelFrame(frame_input, borderwidth=0, text='Dossier contenant les fichiers à classer', relief = FLAT, bg = '#232323')
dir_test.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_test.pack(fill=X, expand = 'yes', pady=20)

frame_btn_test = Frame(dir_test, borderwidth=0, relief = FLAT, bg= '#e6e6e6')
frame_btn_test.pack(fill=X)

btn_test = Button(frame_btn_test, relief = GROOVE,text='Sélectionner un dossier',bg='#e6e6e6', command=(lambda : chemin_dossier_test(r_test)))
btn_test.config(fg='#232323')
btn_test.pack(pady=10)

r_test = Label(frame_btn_test, text = "TEST", bg = '#e6e6e6')
r_test.pack(fill=X, pady=10)

#selection du modele

dir_mdl = LabelFrame(frame_input, borderwidth=0, text='Sélection du modèle de classification', relief = FLAT, bg = '#232323')
dir_mdl.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_mdl.pack(fill=X, expand = 'yes', pady=20)

frame_btn_mdl = Frame(dir_mdl, borderwidth=0, relief = FLAT, bg= '#e6e6e6')
frame_btn_mdl.pack(fill=X)

btn_mdl = Button(frame_btn_mdl, relief = GROOVE,text='Sélectionner un dossier',bg='#e6e6e6',command=(lambda : chemin_dossier_mdl(r_mdl)))
btn_mdl.config(fg='#232323')
btn_mdl.pack(pady=10)

r_mdl = Label(frame_btn_mdl, text = "MODEL/DEFAUT", bg = '#e6e6e6')
r_mdl.pack(fill=X, pady=10)

#entrainement

train = LabelFrame(frame_input, borderwidth=0, text='Entraîner un nouveau modèle', relief = FLAT, bg = '#232323')
train.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train.pack(fill=X, expand = 'yes', pady=20)

frame_train = Frame(train, borderwidth=0, relief = FLAT, bg= '#e6e6e6')
frame_train.pack(fill=X)

#choix
entrain = BooleanVar()

train_true = Radiobutton(frame_train, text="oui", variable=entrain, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_true.pack(pady=5)
train_false = Radiobutton(frame_train, text="non", variable=entrain, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_false.pack(pady=5)
#nombre de clusters

clusters = LabelFrame(frame_train, borderwidth=0, text='Nombre de clusters', relief = SUNKEN, bg = '#e6e6e6')
clusters.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
clusters.pack(fill=X, expand = 'yes', pady=20)

n_clusters = Spinbox(clusters, from_=3, to=50, bg= "#e6e6e6",fg='#232323')
n_clusters.pack(pady=10)

# data d'entraînement

dir_src= LabelFrame(frame_train, borderwidth=0, text='''Dossier contenant les fichiers d'entraînement''', relief = SUNKEN, bg = '#e6e6e6')
dir_src.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_src.pack(fill=X, expand = 'yes', pady=20)

btn_src = Button(dir_src,text='Sélectionner un dossier', relief = GROOVE,bg='#e6e6e6',command=(lambda :chemin_dossier_src(r_src)))
btn_src.config(fg='#232323')
btn_src.pack(pady=10)

r_src = Label(dir_src, text = "PDF", bg = '#e6e6e6')
r_src.pack(fill=X, pady=10)

# démarrer 

btn_start = Button(bench_et_start, text="Lancer la classification",height=3, bg = '#00C800',fg = '#E6E6E6', command=(lambda: start(frame_output, bench_txt, r_src, r_test, n_clusters, entrain=entrain, mdl = r_mdl, cl=cl_output)))
btn_start.config(highlightthickness = 1, highlightbackground='#009100', highlightcolor='#009100')
btn_start.pack(pady= 20, fill=X)

quitter =  Frame(frame_droite, bg ='#232323', relief = FLAT)
quitter.pack(side = BOTTOM, pady = 20)

btn_quitter = Button(quitter, text="Quitter", command=fen.quit, bg='#C80000', fg = '#E6E6E6')
btn_quitter.config(highlightthickness = 1, highlightbackground='#910000', highlightcolor='#910000')
btn_quitter.pack()

fen.mainloop()
