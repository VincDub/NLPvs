import pdfminer
from pdfminer import high_level

import networkx as nx

import pandas as pd

import unicodedata
import re
import string
import time
import joblib
import sys
import os
import random
import webbrowser
from tkinter import *
from tkinter import filedialog
import datetime
import tkinter.font as tkFont
from PIL import Image, ImageTk
import numpy as np
from bs4 import BeautifulSoup
import codecs
import collections
import shutil

import nltk
from nltk import tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.cluster as skcl
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
from mpld3.utils import get_id

#import cuml.cluster as cucl

try:
    import cuml.cluster as cucl
except ImportError:
    cucl = None

#### POUR JS

'''
def html_id_ok(objid, html5=False):
    """Check whether objid is valid as an HTML id attribute.
    If html5 == True, then use the more liberal html5 rules.
    """
    if html5:
        return not re.search('\s', objid)
    else:
        return bool(re.match("^[a-zA-Z][a-zA-Z0-9\-\.\:\_]*$", objid))


def get_id(obj, suffix="", prefix="el", warn_on_invalid=True):
    """Get a unique id for the object"""
    if not suffix:
        suffix = ""
    if not prefix:
        prefix = ""

    objid = prefix + str(os.getpid()) + str(id(obj)) + suffix

    if warn_on_invalid and not html_id_ok(objid):
        warnings.warn('"{0}" is not a valid html ID. This may cause problems')

    return objid
'''

### GENERATION COULEURS ALEATOIRES LES PLUS DISTINCTES POSSIBLE

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


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

def nettoyage_sans_phrases(txt):

    txt_ss = list()

    for texte in txt:
        texte = "".join(texte.splitlines())
        texte = unicodedata.normalize('NFD', texte).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        texte = texte.lower()
        rm_ponctuation = str.maketrans(string.punctuation, " "*len(string.punctuation))
        texte = texte.translate(rm_ponctuation)
        texte = "".join([i for i in texte if not i.isdigit()]) ## SUPPRESSION DES CHIFFRES
        texte = re.sub(u'\u2019',u"\u0027", texte)
        txt_ss.append(texte)

    return txt_ss


'''
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
'''

def tokenisation(txt):

    stop_words = []

    f = open(('stopwordsfr.txt'),"r")

    for ligne in f.readlines():

        stop_words.append(ligne[:-1])

    tk = [mot for mot in txt.split() if (len(mot)>3) and (mot not in stop_words)]

    blank = ""
    n_tk = [token for token in tk if (token not in blank)]

    stemmer=FrenchStemmer()
    n_tk_racin = [stemmer.stem(token) for token in n_tk]

    n_tk_racin_regroup = ""

    for n in n_tk_racin:

        a = ' ' + n
        n_tk_racin_regroup += a


    return n_tk,n_tk_racin,n_tk_racin_regroup

'''
def racin_pour_tfidf(lst):

    stemmer=FrenchStemmer()
    n_tk = [stemmer.stem(token) for token in lst]

    return n_tk

def token_aux(txt):

    mots_phrases = list()

    for phrase in txt:

        stop_words = set(stopwords.words("french"))


        tk = [mot for mot in phrase.split() if (len(mot)>2) and (mot not in stop_words)]

        blank = ""
        n_tk = [token for token in tk if (token not in blank)]

        mots_phrases.append(n_tk)
        

    return mots_phrases

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
'''

def preprocess_pdf_masse(dir):

    noms_fichiers = fichiers_cibles(dir)
    noms_fichiers_o = list()
    textes_fichiers = list()
    
    for entry in noms_fichiers:

        try:
            textes_fichiers.append(pdfminer.high_level.extract_text(entry,"",None,0,True,'utf-8',None))
            noms_fichiers_o.append(entry)

        except TypeError:
            
            print('la structure du fichier ' + entry + ' est invalide !')

    fichiers_clean = nettoyage_sans_phrases(textes_fichiers)

    return noms_fichiers_o, fichiers_clean


def preprocess_html_masse(dir):

    noms_fichiers = fichiers_cibles(dir)
    textes_fichiers = list()
    noms_fichiers_o= list()
    
    for nom in noms_fichiers:
 
        markup = codecs.open(nom, 'r', 'utf-8')
        doc = BeautifulSoup(markup,features="html.parser")

        '''
        ## Spécfique au classement de l'EVCAU

        resume = doc.find(id="comp-jrkv04r5")
        ''' 

        texte_p = ""

        paragraphes = doc.find_all('p')

        for pa in paragraphes:

            texte_p += " "
            texte_p += (pa.get_text())

        if len(texte_p) >= 100:

            noms_fichiers_o.append(nom)
            textes_fichiers.append(texte_p) 


    fichiers_clean = nettoyage_sans_phrases(textes_fichiers)

    return noms_fichiers_o, fichiers_clean

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

    tfconv = TfidfVectorizer(input = 'content',max_features = 9000000,ngram_range=(1,3),preprocessor=None,lowercase=False, tokenizer=None)
    tfidf = tfconv.fit_transform(sac)

    return tfconv

'''

def vectoriser_avec_noms(dir_src,vectoriz,pdf):

    if pdf==True:

        preprocess = preprocess_pdf_masse(dir_src)
    else:
        preprocess = preprocess_html_masse(dir_src)

    noms = preprocess[0]
    textes_vec = list()

    sac = preprocess[1]

    matrice = vectoriz.fit_transform(sac)

    textes_vec.append(matrice)

    return noms, textes_vec, matrice


def vectoriser_avec_noms_diff(dir_src,vectoriz,pdf):

    if pdf==True:
        preprocess = preprocess_pdf_masse(dir_src)
    else:
        preprocess = preprocess_html_masse(dir_src)

    noms = preprocess[0]
    textes_vec = list()

    sac = preprocess[1]

    matrice = vectoriz.transform(sac)

    textes_vec.append(matrice)

    return noms, textes_vec, matrice

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

def entrainer_depuis_corpus(dir_src,n,gpu,pdf):

    if pdf==True:
        preprocess = preprocess_pdf_masse(dir_src)
    else:
        preprocess = preprocess_html_masse(dir_src)

    sac2 = preprocess[1]
    noms = preprocess[0]

    ss_racin = []
    racin = []
    liste_tokens = []

    for txt in sac2:

        token = tokenisation(txt)
        liste_tokens.append(token[2])
        ss_racin.extend(token[0])
        racin.extend(token[1])


    vocab = pd.DataFrame({'words': ss_racin}, index = racin)

    vectoriseur = TfidfVectorizer(input = 'content',max_df=0.8, min_df=0.2,max_features = 9000000,token_pattern=r'\S+',preprocessor=None,lowercase=False, tokenizer=None)

    corpus_vect = vectoriseur.fit_transform(liste_tokens)

    if gpu == False:

        modele_custom = skcl.KMeans(n_clusters=n, init='k-means++').fit(corpus_vect)
        suffixe = "-CPU"
    
    elif cucl == None:

        modele_custom = skcl.KMeans(n_clusters=n, init='k-means++').fit(corpus_vect)
        suffixe = "-CPU"

    else:

        modele_custom = cucl.KMeans(n_clusters=n, init='k-means++').fit(corpus_vect)
        suffixe = "-CUDA_(GPU Nvidia Requis)"

    cd = 'MODEL'
    t = str(datetime.datetime.now()).replace(':','_')
    path= os.path.join(cd,t[:-7]+(suffixe))
    os.mkdir(path)

    termes = vectoriseur.get_feature_names()
    
    joblib.dump(modele_custom,(path + '/model.sav'))
    joblib.dump(vectoriseur,(path + '/vect.sav'))
    joblib.dump(n,(path + '/nb_clust.sav'))
    joblib.dump(vocab,(path + '/vocab.sav'))
    joblib.dump(termes,(path + '/termes.sav'))
    nm = open((path+"/noms.txt"),"w+")

    for x in noms:
        nm.write(x+"\n")
    nm.close()

    return modele_custom,vectoriseur,vocab,noms


def tester_texte(dir_src,dir_test,n,entrainer,dir_model,gpu,pdf):

    t_0 = time.time()

    if entrainer == False :

        modele = joblib.load(dir_model+'/model.sav')
        vectoriseur = joblib.load(dir_model+'/vect.sav')
        termes = joblib.load(dir_model+'/termes.sav')
        vocab = joblib.load(dir_model+'/vocab.sav')
        f = open((dir_model+'/noms.txt'),"r")

        noms_entrainement = list()

        for lignes in f.readlines():

            lignes.replace("\n","")
            noms_entrainement.append(lignes[4:-1])

        f.close()

        labels_cl = modele.labels_.tolist()

        noms = fichiers_cibles(dir_test)

        a = 0
        j = len(noms_entrainement)

        for x in noms:

            sp = os.path.split(x)

            if sp[-1] in noms_entrainement:
                a+=1
            else:
                break

        if a == j :

            txt_ = vectoriser_avec_noms(dir_test,vectoriseur,pdf)
            noms = txt_[0]
            txt_vec = txt_[1]
            resultat_matrix = txt_[2]

            resultat_class = list()

            for i in range (0,len(noms)):
                    
                for texte in txt_vec:   
                    
                    resultat = modele.predict(texte)

                resultat_class.append((noms[i],resultat[i]))

        else :

            txt_diff= vectoriser_avec_noms_diff(dir_test, vectoriseur,pdf)
            noms = txt_diff[0]
            txt_vec_diff= txt_diff[1]
            resultat_matrix = txt_diff[2]

            resultat_class = list()

            for i in range (0,len(noms)):
                    
                for texte in txt_vec_diff:   
                    
                    resultat = modele.predict(texte)

                resultat_class.append((noms[i],resultat[i]))

    else:

        entrainement = entrainer_depuis_corpus(dir_src,n,gpu,pdf)
        modele = entrainement[0]
        vectoriseur = entrainement[1]
        vocab = entrainement[2]
        noms_entrainement_originaux = entrainement[3]

        noms_entrainement = list()

        for y in noms_entrainement_originaux:

            noms_entrainement.append(y[4:])

        labels_cl = modele.labels_.tolist()

        termes = vectoriseur.get_feature_names()

        noms = fichiers_cibles(dir_test)

        a = 0
        j = len(noms_entrainement)

        for x in noms:

            sp = os.path.split(x)

            if sp[-1] in noms_entrainement:
                a+=1
            else:
                break

        if a == j :

            txt_ = vectoriser_avec_noms(dir_test,vectoriseur,pdf)
            noms = txt_[0]
            txt_vec = txt_[1]
            resultat_matrix = txt_[2]

            resultat_class = list()

            for i in range (0,len(noms)):

                for texte in txt_vec:
                
                    resultat = modele.predict(texte)

                resultat_class.append((noms[i],resultat[i]))
        else:

            txt_diff= vectoriser_avec_noms_diff(dir_test, vectoriseur,pdf)
            noms = txt_diff[0]
            txt_vec_diff = txt_diff[1]
            resultat_matrix = txt_diff[2]

            resultat_class = list()

            for i in range (0,len(noms)):

                for texte in txt_vec_diff:
                
                    resultat = modele.predict(texte)

                resultat_class.append((noms[i],resultat[i]))

            
    t_1 = time.time()

    delta_t = "effectué en " + str((t_1 - t_0))[:5] + " secondes."

    return resultat_class , delta_t, modele, termes, vocab, resultat_matrix, labels_cl,noms


def start(frame,bench,src,test,n,entrain,mdl,cl, gpu_bool,pdf_bool,nm):

    pdf_bool = pdf_bool.get()

    dir_test = test.cget("text")
    dir_mdl = mdl.cget("text")
    n_cl = int()
    dir_src = str()
    nom_perso = nm.get()
    train = entrain.get()

    if train == False:

        n_cl = joblib.load(dir_mdl+'/nb_clust.sav') #Chargement du nombre de clusters du modèle préentraîné

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=False,dir_model=dir_mdl, gpu = gpu_bool,pdf = pdf_bool) #Prédiction sur le corpus de textes à classifier

        result_clusters = clusters_en_features(result[2],n_cl,result[3],result[4]) #Extraction des 4 premières features par cluster

        bench["text"] = str(result[1]) #Impression du temps d'exécution

        #Impression des résultats des clusters selon le numéro du cluster

        for key in result_clusters:

            cl.insert(END,(str(key) + str(result_clusters[key])))

        liste_index = []

        for resultat in result[0]:

            nom_indiv = os.path.split(str(resultat[0])) #Récupération du nom des documents

            result_txt = nom_indiv[-1] + ' affecté au cluster N° '+ str(resultat[1])

            frame.insert(END,result_txt) #Insertion des résultats dans l'interface

            liste_index.append(resultat[1]) #Récupération du numéro du cluster pour chaque document

        # Génération du graph HTML

        graph = produire_graph_2d(result[5],result_clusters,liste_index,result[7],pdf_bool,nom_perso)

        
        
    else:

        n_cl = int(n.get())
        dir_src = test.cget("text")

        result = tester_texte(dir_src,dir_test,n_cl,entrainer=True,dir_model=dir_mdl,gpu = gpu_bool, pdf = pdf_bool)

        result_clusters = clusters_en_features(result[2],n_cl,result[3],result[4])

        bench["text"] = str(result[1])

        for key in result_clusters:

            cl.insert(END,(str(key) + str(result_clusters[key])))

        liste_index = []

        for resultat in result[0]:

            nom_indiv = os.path.split(str(resultat[0]))

            result_txt = nom_indiv[-1] + ' affecté au cluster N° '+ str(resultat[1])

            frame.insert(END,result_txt)

            liste_index.append(resultat[1])

        graph = produire_graph_2d(result[5],result_clusters,liste_index,result[7],pdf_bool,nom_perso)

    msg = 'Graph exporté sous :' + graph

    frame.insert(END,' ')
    frame.insert(END,msg)

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

            term_ind = tm[ind]

            mot = (vocab.at[term_ind,'words'])[0]
            
            mots_par_clusters.append(mot)

        clusters_dict['Cluster N° ' + str(i) +':'] = mots_par_clusters
    
    return clusters_dict


def produire_graph_2d(matrix,clusters_dict,labels_docs_class,titres_docs_class,pdf,nom):

    nom_perso = nom

    ## DISTANCE COSINUS

    distance = 1 - cosine_similarity(matrix)

    ## REDUCTION MULTIDIMENSIONELLE VERS 2 DIMENSIONS

    MDS()

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(distance)

    xs, ys = pos[:, 0], pos[:, 1]

    noms_clusters = dict()
    couleurs_clusters = dict()

    i = 0
    palette = list()

    for x in clusters_dict.values():

        nom = ', '.join(x)
        palette.append(generate_new_color(palette,pastel_factor = 0.5))
        couleurs_clusters[i] = str(mpl.colors.to_hex(palette[i]))
        noms_clusters[i] = nom
        
        i += 1

    noms_docs = list()

    for titre in titres_docs_class:

        if pdf == True:

            nom_solo = (os.path.split(titre))[-1]

            noms_docs.append(nom_solo[:-4])

        else:

            nom_solo = (os.path.split(titre))[-1]

            noms_docs.append(nom_solo)

    class InteractiveLegendPluginTxt(mpld3.plugins.PluginBase):


       #Modifié pour ajouter un paramètre d'affichage de texte


        JAVASCRIPT = """
        mpld3.register_plugin("interactive_legend", InteractiveLegend);
        InteractiveLegend.prototype = Object.create(mpld3.Plugin.prototype);
        InteractiveLegend.prototype.constructor = InteractiveLegend;
        InteractiveLegend.prototype.requiredProps = ["element_ids", "labels","text"];
        InteractiveLegend.prototype.defaultProps = {
                                                    "ax":null,
                                                    "alpha_unsel":0.2,
                                                    "alpha_over":1.0,
                                                    "start_visible":true}
        function InteractiveLegend(fig, props){
            mpld3.Plugin.call(this, fig, props);
        };

        InteractiveLegend.prototype.draw = function(){

            var alpha_unsel = this.props.alpha_unsel;
            var alpha_over = this.props.alpha_over;
            var text = this.props.text;


            var legendItems = new Array();
            for(var i=0; i<this.props.labels.length; i++){
                console.log(i);
                var obj = {};
                obj.label = this.props.labels[i];
                var texte_cluster_choisi = text[i];

                var element_id = this.props.element_ids[i];
                console.log(element_id);
                mpld3_elements = [];
                _texts = [];
                for(var j=0; j<element_id.length; j++){
                    var mpld3_element = mpld3.get_element(element_id[j], this.fig);
                    var objet = d3.select(mpld3_element);
                    console.log(objet);
 
                    // mpld3_element might be null in case of Line2D instances
                    // for we pass the id for both the line and the markers. Either
                    // one might not exist on the D3 side
                    if(mpld3_element){
                        mpld3_elements.push(mpld3_element);}
                }
                obj.mpld3_elements = mpld3_elements;
                obj.visible = this.props.start_visible[i];
                legendItems.push(obj);
                set_alphas(obj, false);

            }

            // determine the axes with which this legend is associated
            var ax = this.props.ax
            if(!ax){
                ax = this.fig.axes[0];
            } else{
                ax = mpld3.get_element(ax, this.fig);
            }

            // add a legend group to the canvas of the figure
            var legend = this.fig.canvas.append("svg:g")
                                .attr("class", "legend");

            // add the rectangles
            legend.selectAll("rect")
                    .data(legendItems)
                    .enter().append("rect")
                    .attr("height", 10)
                    .attr("width", 25)
                    .attr("x", ax.width + ax.position[0] + 25)
                    .attr("y",function(d,i) {
                            return ax.position[1] + i * 25 + 10;})
                    .attr("stroke", get_color)
                    .attr("class", "legend-box")
                    .style("fill", function(d, i) {
                            return d.visible ? get_color(d) : "white";})
                    .on("click", click).on('mouseover', over).on('mouseout', out);

            // add the labels
            legend.selectAll("text")
                .data(legendItems)
                .enter().append("text")
                .attr("x", function (d) {
                            return ax.width + ax.position[0] + 25 + 40;})
                .attr("y", function(d,i) {
                            return ax.position[1] + i * 25 + 10 + 10 - 1;})
                .text(function(d) { return d.label });


            // specify the action on click
            function click(d,i){
                d.visible = !d.visible;
                d3.select(this)
                .style("fill",function(d, i) {
                    return d.visible ? get_color(d) : "white";
                })
                set_alphas(d, false);
                var texte = text[i];
                console.log(d);
                var css = "mpld3-texte_" + d.label;
                console.log(css);

                for(var j=0; j<texte.length; j++){

                    console.log(d.mpld3_elements[0]);
                    console.log(texte[j]);
                    var elem = d.mpld3_elements[0].pathsobj._groups[0];
                    var coord = elem[j].getBoundingClientRect();

                    if (d.visible == true) {
                    var tooltip = (d3.select("body")).append("div")
                        .attr("class", css )
                        .style("position", "absolute")
                        .style("z-index", "10")
                        .style("visibility", "visible")
                        .style("top", coord.top + "px")
                        .style("left", coord.left + "px");

                    tooltip.html(texte[j])
                        .style("visibility", "visible");

                    } else {
                    
                    var bod = d3.select("body")._groups[0];
                    console.log(bod[0]);
                    for (var i=0; i < bod[0].childNodes.length; i++) {
                        if (bod[0].childNodes[i].className == css) {
                            (bod[0].childNodes[i]).parentNode.removeChild(bod[0].childNodes[i]);
                            }
                        }

                    }
                    
                }
            };

            // specify the action on legend overlay 
            function over(d,i){
                set_alphas(d, true);
            };

            // specify the action on legend overlay 
            function out(d,i){
                set_alphas(d, false);
            };

            // helper function for setting alphas
            function set_alphas(d, is_over){
                for(var i=0; i<d.mpld3_elements.length; i++){
                    var type = d.mpld3_elements[i].constructor.name;

                    if(type =="mpld3_Line"){
                        var current_alpha = d.mpld3_elements[i].props.alpha;
                        var current_alpha_unsel = current_alpha * alpha_unsel;
                        var current_alpha_over = current_alpha * alpha_over;
                        d3.select(d.mpld3_elements[i].path[0][0])
                            .style("stroke-opacity", is_over ? current_alpha_over :
                                                    (d.visible ? current_alpha : current_alpha_unsel))
                            .style("stroke-width", is_over ? 
                                    alpha_over * d.mpld3_elements[i].props.edgewidth : d.mpld3_elements[i].props.edgewidth);
                    } else if((type=="mpld3_PathCollection")||
                            (type=="mpld3_Markers")){
                        var current_alpha = d.mpld3_elements[i].props.alphas[0];
                        var current_alpha_unsel = current_alpha * alpha_unsel;
                        var current_alpha_over = current_alpha * alpha_over;
                        d3.selectAll(d.mpld3_elements[i].pathsobj[0])
                            .style("stroke-opacity", is_over ? current_alpha_over :
                                                    (d.visible ? current_alpha : current_alpha_unsel))
                            .style("fill-opacity", is_over ? current_alpha_over :
                                                    (d.visible ? current_alpha : current_alpha_unsel));
                    
                    } else{
                        console.log(type + " not yet supported");
                    }
                }
            };



            // helper function for determining the color of the rectangles
            function get_color(d){
                var type = d.mpld3_elements[0].constructor.name;
                var color = "black";
                if(type =="mpld3_Line"){
                    color = d.mpld3_elements[0].props.edgecolor;
                } else if((type=="mpld3_PathCollection")||
                        (type=="mpld3_Markers")){
                    color = d.mpld3_elements[0].props.facecolors[0];
                } else{
                    console.log(type + " not yet supported");
                }
                return color;
            };
        };
        """

        css_ = """
        .legend-box {
        cursor: pointer;
        }

        .legend > text {
        font-family:Arial, Helvetica, sans-serif;font-size:18px;
        }

        body > div {
        font-family:Arial, Helvetica, sans-serif;
        font-size:16px;
        color:rgba(0,0,0,0.7)
        background-color: rgba(255,255,255,0.8);
        }
        """

        def __init__(self, plot_elements,texte, labels, ax=None,
                    alpha_unsel=0.2, alpha_over=1., start_visible=True):

            self.ax = ax

            if ax:
                ax = get_id(ax)

            # start_visible could be a list
            if isinstance(start_visible, bool):
                start_visible = [start_visible] * len(labels)
            elif not len(start_visible) == len(labels):
                raise ValueError("{} out of {} visible params has been set"
                                .format(len(start_visible), len(labels)))     

            mpld3_element_ids = self._determine_mpld3ids(plot_elements)
            self.mpld3_element_ids = mpld3_element_ids
            self.dict_ = {"type": "interactive_legend",
                        "element_ids": mpld3_element_ids,
                        "text": texte,
                        "labels": labels,
                        "ax": ax,
                        "alpha_unsel": alpha_unsel,
                        "alpha_over": alpha_over,
                        "start_visible": start_visible}

        def _determine_mpld3ids(self, plot_elements):
            """
            Helper function to get the mpld3_id for each
                of the specified elements.
            """
            mpld3_element_ids = []

            # There are two things being done here. First,
            # we make sure that we have a list of lists, where
            # each inner list is associated with a single legend
            # item. Second, in case of Line2D object we pass
            # the id for both the marker and the line.
            # on the javascript side we filter out the nulls in
            # case either the line or the marker has no equivalent
            # D3 representation.
            for entry in plot_elements:
                ids = []
                if isinstance(entry, collections.Iterable):
                    for element in entry:
                        mpld3_id = get_id(element)
                        ids.append(mpld3_id)
                        if isinstance(element, mpl.lines.Line2D):
                            mpld3_id = get_id(element, 'pts')
                            ids.append(mpld3_id)
                else:
                    ids.append(get_id(entry))
                    if isinstance(entry, mpl.lines.Line2D):
                        mpld3_id = get_id(entry, 'pts')
                        ids.append(mpld3_id)
                mpld3_element_ids.append(ids)

            return mpld3_element_ids

    
    class PointHTMLTooltipC(mpld3.plugins.PluginBase):
    
        JAVASCRIPT = """
        mpld3.register_plugin("htmltooltip", HtmlTooltipPlugin);
        HtmlTooltipPlugin.prototype = Object.create(mpld3.Plugin.prototype);
        HtmlTooltipPlugin.prototype.constructor = HtmlTooltipPlugin;
        HtmlTooltipPlugin.prototype.requiredProps = ["id"];
        HtmlTooltipPlugin.prototype.defaultProps = {labels:null,
                                                    hoffset:0,
                                                    voffset:10};
        function HtmlTooltipPlugin(fig, props){
            mpld3.Plugin.call(this, fig, props);
        };

        HtmlTooltipPlugin.prototype.draw = function(){
        var obj = mpld3.get_element(this.props.id);
        var labels = this.props.labels;
        var tooltip = d3.select("body").append("div")
                        .attr("class", "mpld3-tooltip")
                        .style("position", "absolute")
                        .style("z-index", "10")
                        .style("visibility", "hidden");

        obj.elements()
            .on("mouseover", function(d, i){
                                tooltip.html(labels[i])
                                        .style("visibility", "visible");})
            .on("mousemove", function(d, i){
                    tooltip
                        .style("top", d3.event.pageY + this.props.voffset + "px")
                        .style("left",d3.event.pageX + this.props.hoffset + "px");
                    }.bind(this))
            .on("mouseout",  function(d, i){
                            tooltip.style("visibility", "hidden");});
        };
        """

        def __init__(self, points, labels=None,
                    hoffset=0, voffset=10, css=None):
            self.points = points
            self.labels = labels
            self.voffset = voffset
            self.hoffset = hoffset
            self.css_ = css or ""
            if isinstance(points, mpl.lines.Line2D):
                suffix = "pts"
            else:
                suffix = None
            self.dict_ = {"type": "htmltooltip",
                        "id": get_id(points, suffix),
                        "labels": labels,
                        "hoffset": hoffset,
                        "voffset": voffset}

    class TopToolbar(mpld3.plugins.PluginBase):

        #Modifié pour inclure la barre d'outils en haut à gauche plutôt qu'en bas à droite

        JAVASCRIPT = """
        mpld3.register_plugin("toptoolbar", TopToolbar);
        TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
        TopToolbar.prototype.constructor = TopToolbar;
        function TopToolbar(fig, props){mpld3.Plugin.call(this, fig, props);};

        TopToolbar.prototype.draw = function(){
        this.fig.toolbar.draw();

        this.fig.toolbar.toolbar.attr("x", 0);
        this.fig.toolbar.toolbar.attr("y", 0);
        this.fig.toolbar.toolbar.attr("width", 48);
        this.fig.toolbar.toolbar.attr("height", 16);

        this.fig.toolbar.draw = function() {}
        };
        """

        def __init__(self):
            self.dict_ = {"type": "toptoolbar"}
        

    css = """
    body > div > div {
        width: 100%;
        height: 100vh;
    }

    text.mpld3-text, div.mpld3-tooltip {
    font-family:Arial, Helvetica, sans-serif;font-size:18px;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure > g.mpld3-baseaxes {
        position: absolute !important;

     }

    svg.mpld3-figure > g.mpld3-baseaxes > g.mpld3-axes {
    width: auto;
    height: auto;
     }
    

    svg.mpld3-figure {
        position: absolute;
        width: auto;
        height: auto;}
    """

    df = pd.DataFrame(dict(x=xs, y=ys, label=labels_docs_class, titres=noms_docs))
    df.sort_values(by=['label'])

    groupes = df.groupby('label')

    fig, ax = plt.subplots(figsize=(17,9))
    ax.margins(0.01)


    elem_points = []
    titres_tot = []
    titres_clust = []

    for nom, groupe in groupes:

        titres = []
        pts = []

        for i in range(len(groupe)):

            titres.append(groupe.iat[i,3])
        
        points = ax.scatter(groupe.x, groupe.y, marker='o', s=150,label=noms_clusters[nom], color=couleurs_clusters[nom])

        titres_clust.append(noms_clusters[nom])

        titres_tot.append(titres)

        elem_points.append([points])


    ## POINTS TRANSPARENTS POUR L'AFFICHAGE INDIVIDUEL DES TITRES
    po = ax.plot(df.x, df.y, marker='o', linestyle='', ms=22, alpha = 0.0, mec='none')

    tot = []

    for i in range(len(df)):

        tot.append(df.iat[i,3])

    ##

    affichage_noms= PointHTMLTooltipC(po[0],tot,voffset=1, hoffset=1, css=css)

    ilegend = InteractiveLegendPluginTxt(elem_points,titres_tot,titres_clust,alpha_unsel=0.0, alpha_over=1.2,start_visible=False)

    mpld3.plugins.connect(fig, affichage_noms,ilegend, TopToolbar())

    plt.subplots_adjust(right=0.72)

    cd = 'GRAPH'
    t = str(datetime.datetime.now()).replace(':','_')
    nm = os.path.join(cd, (nom_perso + ' ' + t[:-7] + '.html'))

    ax.set_title((nom_perso +  ' - ' + t[:-16]), size=20, y=0, x = 1.2)

    mpld3.save_html(fig,nm)

    return nm

def afficher_graph(lst):

    path = (lst.get(END).split(' :'))[1]

    webbrowser.open(path, new=2)
    

def generer_gexf(df,couleurs_clusters, noms_clusters):

    df.sort_values('label')

    index = df['label'].tolist()

    for i in index:
        couleurs = couleurs_clusters[i]
        nom = noms_clusters[i]
    df.insert(4,"couleur_cluster",couleurs, True)
    df.insert(5,"nom_cluster",nom, True)

    print(df)

    G = nx.Graph()

    dictionnaires = df.to_dict('records')

    liste_noms = []
    liste_coordonnees = []
    node_pos = {}

    for dic in dictionnaires:

        G.add_node(dic['titres'], node_color = str(dic['couleur_cluster']))# pos = (dic['x'], dic['y']))

        liste_noms.append(dic['titres'])
        li = (dic['x'],dic['y'])
        liste_coordonnees.append(li)
        node_pos[dic['titres']] = np.array(li)

    i = 0
    liste_noeuds = []
    liste_attr = []

    for nom in liste_noms:

        list_temp = liste_noms.copy()
        list_temp_c = liste_coordonnees.copy()

        list_temp.remove(nom)

        c = liste_coordonnees[i]
        list_temp_c.remove(c)


        for t in list_temp:
            tup = (nom,t)
            connexion_noeud = tuple(tup)
            liste_noeuds.append(connexion_noeud)

        for ci in list_temp_c:

            a_ci = np.array(ci)
            a_c = np.array(c)
            d = np.linalg.norm(a_ci-a_c, ord=2)
            distance_ = {'attr': d}
            liste_attr.append(distance_)
        
        i += 1

    liste_pr_edges = []

    j = 0
    
    for e in liste_noeuds:

        ed = [e,liste_attr[j]]
        liste_pr_edges.append(ed)
        j += 1


    for edge in liste_pr_edges:

        nn = list(edge[0])

        at = (edge[1]).get('attr')

        G.add_edge(nn[0], nn[1], attr=at)

    nx.draw_networkx(G, pos = node_pos, with_labels=True)

    t = str(datetime.datetime.now()).replace(':','_')
    chm = 'GEXF/' + t[:-7] +'.gexf'

    file = open(chm, "w")

    for line in nx.generate_gexf(G):
        file.writelines(line)
    
    file.close()
    
    return chm

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

fen.title('NLPvs')
fen.configure(background = '#232323')
fen.geometry('1100x900')

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

log_o = Image.open('logo.png')
log_o_md = log_o.resize((250, 83),Image.ANTIALIAS)
logo = ImageTk.PhotoImage(log_o_md)
logo_label = Label(frame_input, image=logo, bg = '#232323' )
logo_label.place(x=0,y=0)
logo_label.pack(padx = 40)

credit = Label(frame_input,text='par Vincent DUBOIS',fg='#e6e6e6', bg ='#232323',font=fontcredits)
credit.pack()

frame_2 = Frame(frame_droite, borderwidth = 0, relief = FLAT, bg = '#232323')
frame_2.pack(pady=20)

frame_log = LabelFrame(frame_2, text = 'Documents clusterisés', borderwidth = 0, relief = FLAT, bg = '#232323')
frame_log.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
frame_log.pack(padx = 0)

frame_output = Listbox(frame_log, borderwidth = 0, bg = '#e6e6e6', fg = '#FF0000', width=300)
frame_output.pack(fill = X, padx=0)

frame_graph = Frame(frame_log, borderwidth=0, relief = FLAT, bg= '#e6e6e6')
frame_graph.pack(fill=X)

btn_graph = Button(frame_graph, relief = GROOVE,text='Montrer le graph',bg='#e6e6e6', command=(lambda : afficher_graph(frame_output)))
btn_graph.config(fg='#232323')
btn_graph.pack(pady=10)

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

r_test = Label(frame_btn_test, text = "PDF", bg = '#e6e6e6')
r_test.pack(fill=X, pady=10)

#choix type de fichier

pdf = BooleanVar()

train_true = Radiobutton(frame_btn_test, text="PDF", variable=pdf, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_true.pack(pady=5)
train_false = Radiobutton(frame_btn_test, text="HTML", variable=pdf, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_false.pack(pady=5)

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

#choix gpu/cpu
gpu = BooleanVar()

gpu_true = Radiobutton(frame_train, text="Accélération CUDA (GPU Nvidia requis)", variable=gpu, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
gpu_true.pack(pady=5)
gpu_false = Radiobutton(frame_train, text="CPU", variable=gpu, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
gpu_false.pack(pady=5)

#nombre de clusters

clusters = LabelFrame(frame_train, borderwidth=0, text='Nombre de clusters', relief = SUNKEN, bg = '#e6e6e6')
clusters.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
clusters.pack(fill=X, expand = 'yes', pady=20)

n_clusters = Spinbox(clusters, from_=3, to=50, bg= "#e6e6e6",fg='#232323')
n_clusters.pack(pady=10)

#nom personnalisé

nm = LabelFrame(frame_train, borderwidth=0, text='Nom personnalisé', relief = SUNKEN, bg = '#e6e6e6')
nm.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
nm.pack(fill=X, expand = 'yes', pady=20)

nm_entry = Entry(nm, bg= "#e6e6e6",fg='#232323')
nm_entry.insert(30,'Classification')
nm_entry.pack(fill=X, expand = 'yes', pady=10)

#data d'entraînement (désactivé)
'''
dir_src= LabelFrame(frame_train, borderwidth=0, text="Dossier contenant les fichiers d'entraînement", relief = SUNKEN, bg = '#e6e6e6')
dir_src.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_src.pack(fill=X, expand = 'yes', pady=20)

btn_src = Button(dir_src,text='Sélectionner un dossier', relief = GROOVE,bg='#e6e6e6',command=(lambda :chemin_dossier_src(r_src)))
btn_src.config(fg='#232323')
btn_src.pack(pady=10)

r_src = Label(dir_src, text = "PDF", bg = '#e6e6e6')
'''
r_src =  None
#r_src.pack(fill=X, pady=10)


#démarrer 

btn_start = Button(bench_et_start, text="Lancer la classification",height=3, bg = '#00C800',fg = '#E6E6E6', command=(lambda: start(frame_output, bench_txt, r_src, r_test, n_clusters, entrain=entrain, mdl = r_mdl, cl=cl_output, gpu_bool = gpu, pdf_bool=pdf,nm = nm_entry)))
btn_start.config(highlightthickness = 1, highlightbackground='#009100', highlightcolor='#009100')
btn_start.pack(pady= 20, fill=X)

quitter =  Frame(frame_droite, bg ='#232323', relief = FLAT)
quitter.pack(side = BOTTOM, pady = 20)

btn_quitter = Button(quitter, text="Quitter", command=fen.quit, bg='#C80000', fg = '#E6E6E6')
btn_quitter.config(highlightthickness = 1, highlightbackground='#910000', highlightcolor='#910000')
btn_quitter.pack()

fen.mainloop()