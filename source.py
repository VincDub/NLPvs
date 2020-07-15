## bibliothèques système
import codecs
import collections
import datetime
import os
import random
import re
import shutil
import string
import sys
import time
import tkinter.font as tkFont
import unicodedata
import webbrowser

## création d'interface graphique avec des widgets
from tkinter import filedialog
import tkinter as Tk

## gestion d'images pour le logo sur l'interface graphique
from PIL import Image, ImageTk

## génération des graphiques
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpld3
from mpld3.utils import get_id

## Gestion et manipulation des matrices
import numpy as np

## Création de tables de données
import pandas as pd

## parsing des fichiers HTML
from bs4 import BeautifulSoup

## fonction de racinisation pour le français
import nltk
from nltk.stem.snowball import FrenchStemmer

## parsing des ficchiers PDF
from pdfminer import high_level

## sauvegarde et chargement des modèles de classification
import joblib

## algorithmes ML pour la vectorisation, la clusterisation et la réduction de dimensions
import sklearn.cluster as skcl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

## algorithmes ML avec accélération CUDA
## import uniquement si cuml est installé sur un système compatible
try:
    import cuml.cluster as cucl
except ImportError:
    cucl = None
    print("impossible d'importer cuML : accélération GPU désactivée par défaut")


## Génération de couleurs aléatoires les plus distinctes possibles entre elles

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

## Récupération des chemins des fichiers à classer dans le répertoire défini

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


## Nettoyage des textes extraits des fichiers à classer
## Contient différentes opérations sur les chaînes de caractères

def nettoyage(txt):

    texte_clean = list()

    for texte in txt:
        texte = "".join(texte.splitlines())
        texte = unicodedata.normalize('NFD', texte).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        texte = texte.lower()
        rm_ponctuation = str.maketrans(string.punctuation, " "*len(string.punctuation))
        texte = texte.translate(rm_ponctuation)
        texte = "".join([i for i in texte if not i.isdigit()]) ## SUPPRESSION DES CHIFFRES
        texte = re.sub(u'\u2019',u"\u0027", texte)
        texte_clean.append(texte)

    return texte_clean

## Tokenisation du texte nettoyé de chaque fichier à classer

def tokenisation(txt):

    stop_words = []

    f = open(('stopwordsfr.txt'),"r")

    for ligne in f.readlines():

        stop_words.append(ligne[:-1])

    tk = [mot for mot in txt.split() if (len(mot)>3) and (mot not in stop_words)]

    blank = ""
    texte_tk = [token for token in tk if (token not in blank)]

    stemmer=FrenchStemmer()
    texte_tk_racin = [stemmer.stem(token) for token in texte_tk]

    texte_tk_racin_regroup = ""

    for n in texte_tk_racin:

        a = ' ' + n
        texte_tk_racin_regroup += a

    return texte_tk,texte_tk_racin,texte_tk_racin_regroup


## Extraction du texte sous forme d'une chaîne de caractère par document à classer
## pour les fichiers PDF

def extraction_texte_pdf(chemin):

    noms_fichiers = fichiers_cibles(chemin)
    noms_fichiers_sortie = []
    textes_fichiers = []
    
    for nom in noms_fichiers:

        try:
            textes_fichiers.append(high_level.extract_text(nom,"",None,0,True,'utf-8',None))
            noms_fichiers_sortie.append(nom)

        except TypeError:
            
            print('la structure du fichier ' + nom + ' est invalide !')

    fichiers_clean = nettoyage(textes_fichiers)

    return noms_fichiers_sortie, fichiers_clean


## Extraction du texte sous forme d'une chaîne de caractère par document à classer
## pour les fichiers HTML

def extraction_texte_html(chemin):

    noms_fichiers = fichiers_cibles(chemin)
    textes_fichiers = []
    noms_fichiers_sortie= []
    
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

            noms_fichiers_sortie.append(nom)
            textes_fichiers.append(texte_p) 

    fichiers_clean = nettoyage(textes_fichiers)

    return noms_fichiers_sortie, fichiers_clean


## Classification d'un corpus de documents en créant et sauvegardant un nouveau modèle

def classifier_corpus_nouveau_modele(dir_class,nombre_clusters,gpu_bool,pdf_bool):

    if pdf_bool==True:

        preprocess = extraction_texte_pdf(dir_class)
    else:

        preprocess = extraction_texte_html(dir_class)

    liste_textes = preprocess[1]
    noms_fichiers = preprocess[0]

    sans_racin = []
    racin = []
    liste_tokens = []

    for txt in liste_textes:

        token = tokenisation(txt)

        liste_tokens.append(token[2])
        sans_racin.extend(token[0])
        racin.extend(token[1])


    vocabulaire = pd.DataFrame({'words': sans_racin}, index = racin)

    vectoriseur = TfidfVectorizer(input = 'content',max_df=0.8, min_df=0.2,max_features = 9000000,token_pattern=r'\S+',preprocessor=None,lowercase=False, tokenizer=None)

    matrice_corpus = vectoriseur.fit_transform(liste_tokens)

    termes = vectoriseur.get_feature_names()

    if gpu_bool == False:

        modele_clusterisation = skcl.KMeans(n_clusters=nombre_clusters, init='k-means++').fit(matrice_corpus)
        suffixe = "-CPU"
    
    elif cucl == None:

        modele_clusterisation = skcl.KMeans(n_clusters=nombre_clusters, init='k-means++').fit(matrice_corpus)
        suffixe = "-CPU"

    else:

        modele_clusterisation = cucl.KMeans(n_clusters=nombre_clusters, init='k-means++').fit(matrice_corpus)
        suffixe = "-CUDA_(GPU Nvidia Requis)"


    modele_clusterisation.predict(matrice_corpus)
    
    classification = modele_clusterisation.labels_.tolist()
    
    resultats_classification = dict()


    i = 0
    for cl in classification:

        nom = (os.path.split(noms_fichiers[i]))[-1]
        if pdf_bool == True:

            nom_solo = nom[:-4]

        else:

            nom_solo = nom[:-5]

        resultats_classification[nom_solo] = cl
        i += 1

    clusters = extraction_features(modele_clusterisation,nombre_clusters,termes,vocabulaire)


    t = str(datetime.datetime.now()).replace(':','_')
    path= os.path.join('MODEL',t[:-7]+(suffixe))
    os.mkdir(path)
    
    joblib.dump(modele_clusterisation,(path + '/model.sav'))
    joblib.dump(vectoriseur,(path + '/vect.sav'))
    joblib.dump(nombre_clusters,(path + '/nb_clust.sav'))
    joblib.dump(vocabulaire,(path + '/vocab.sav'))
    joblib.dump(termes,(path + '/termes.sav'))

    return resultats_classification,clusters,matrice_corpus


## Classification d'un corpus de documents à partir d'un modèle existant

def classifier_corpus_modele_existant(dir_class,dir_mdl,pdf_bool):


    if pdf_bool==True:

        preprocess = extraction_texte_pdf(dir_class)
    else:

        preprocess = extraction_texte_html(dir_class)

    liste_textes = preprocess[1]
    noms_fichiers = preprocess[0]

    sans_racin = []
    racin = []
    liste_tokens = []

    for txt in liste_textes:

        token = tokenisation(txt)

        liste_tokens.append(token[2])
        sans_racin.extend(token[0])
        racin.extend(token[1])
    
    vocabulaire = joblib.load(dir_mdl + '/vocab.sav')
    vectoriseur = joblib.load(dir_mdl+'/vect.sav')
    termes = joblib.load(dir_mdl+'/termes.sav')
    nombre_clusters = joblib.load(dir_mdl+'/nb_clust.sav')
    modele_clusterisation = joblib.load(dir_mdl+'/model.sav')

    matrice_corpus = vectoriseur.fit_transform(liste_tokens)

    modele_clusterisation.predict(matrice_corpus)

    classification = modele_clusterisation.labels_.tolist()

    resultats_classification = dict()

    i = 0
    for cl in classification:

        nom = (os.path.split(noms_fichiers[i]))[-1]
        if pdf_bool == True:

            nom_solo = nom[:-4]

        else:

            nom_solo = nom[:-5]

        resultats_classification[nom_solo] = cl
        i += 1

    clusters = extraction_features(modele_clusterisation,nombre_clusters,termes,vocabulaire)

    return resultats_classification,clusters,matrice_corpus

## Fonction principale

def main(frame_resultats,frame_bench,frame_clusters,dir_class,dir_mdl,nombre_clusters,nom_perso,entrainement_bool,gpu_bool,pdf_bool):

    t_0 = time.time()

    gpu_bool = gpu_bool.get()
    pdf_bool = pdf_bool.get()
    dir_class = dir_class.cget("text")
    dir_mdl = dir_mdl.cget("text")
    nombre_clusters = int(nombre_clusters.get())
    nom_perso = nom_perso.get()
    entrainement_bool = entrainement_bool.get()

    if entrainement_bool == False:

        resultat_classification = classifier_corpus_modele_existant(dir_class,dir_mdl,pdf_bool)
        matrice_documents_cluster = resultat_classification[2]
        features_par_cluster = resultat_classification[1]
        log = resultat_classification[0]

        liste_noms = []
        liste_index = []
        
        for f in features_par_cluster:

            frame_clusters.insert(Tk.END,(str(f) + str(features_par_cluster[f])))
        
        for l in log:

            liste_noms.append(l)
            liste_index.append(log[l])

            frame_resultats.insert(Tk.END,(str(l) + ' affecté au cluster N° '+ str(log[l])))
        
    else:

        resultat_classification = classifier_corpus_nouveau_modele(dir_class,nombre_clusters,gpu_bool,pdf_bool)
        matrice_documents_cluster = resultat_classification[2]
        features_par_cluster = resultat_classification[1]
        log = resultat_classification[0]

        liste_noms = []
        liste_index = []
        
        for f in features_par_cluster:

            frame_clusters.insert(Tk.END,(str(f) + str(features_par_cluster[f])))
        
        for l in log:

            liste_noms.append(l)
            liste_index.append(log[l])

            frame_resultats.insert(Tk.END,(str(l) + ' affecté au cluster N° '+ str(log[l])))
    
    graph = produire_graph_2d(matrice_documents_cluster,features_par_cluster,liste_index,liste_noms,nom_perso)

    t_1 = time.time()

    frame_bench["text"] = "effectué en " + str((t_1 - t_0))[:5] + " secondes."
    msg = 'Graph exporté sous :' + graph

    frame_resultats.insert(Tk.END,' ')
    frame_resultats.insert(Tk.END,msg)

## Extraction des 4 principales features de chaque clusters dans un dictionnaire

def extraction_features(modele,nombre_clusters,termes,vocabulaire):

    centroides = modele.cluster_centers_.argsort()[:, ::-1]

    blank =['',',']
    tm = []

    for term in termes:
        term.split(' ')
        if term not in blank:
            tm.append(term)

    clusters = dict()

    for i in range(nombre_clusters):

        mots_par_clusters = []

        for ind in centroides[i, :4]:

            term_ind = tm[ind]

            mot = (vocabulaire.at[term_ind,'words'])[0]
            
            mots_par_clusters.append(mot)

        clusters['Cluster N° ' + str(i) +':'] = mots_par_clusters
    
    return clusters



## Génération d'un fichier HTML contenant la représentation graphique de la classification

def produire_graph_2d(matrice_documents_cluster,features_par_cluster,liste_index,liste_noms,nom_perso):


    distance = 1 - cosine_similarity(matrice_documents_cluster)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(distance)

    xs, ys = pos[:, 0], pos[:, 1]

    noms_clusters = dict()
    couleurs_clusters = dict()

    i = 0
    palette = list()

    for x in features_par_cluster.values():

        nom = ', '.join(x)
        palette.append(generate_new_color(palette,pastel_factor = 0.5))
        couleurs_clusters[i] = str(mpl.colors.to_hex(palette[i]))
        noms_clusters[i] = nom
        
        i += 1

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

    df = pd.DataFrame(dict(x=xs, y=ys, label=liste_index, titres=liste_noms))
    df.sort_values(by=['label'])

    groupes = df.groupby('label')

    fig, ax = plt.subplots(figsize=(17,9))
    ax.margins(0.01)


    elem_points = []
    titres_tot = []
    titres_clust = []

    for nom, groupe in groupes:

        titres = []

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

    affichage_noms= PointHTMLTooltipC(po[0],tot,voffset=1, hoffset=1, css=css)

    legende_interactive= InteractiveLegendPluginTxt(elem_points,titres_tot,titres_clust,alpha_unsel=0.0, alpha_over=1.2,start_visible=False)

    mpld3.plugins.connect(fig, affichage_noms,legende_interactive, TopToolbar())

    plt.subplots_adjust(right=0.72)

    cd = 'GRAPH'
    t = str(datetime.datetime.now()).replace(':','_')
    chemin = os.path.join(cd, (nom_perso + ' ' + t[:-7] + '.html'))

    ax.set_title((nom_perso +  ' - ' + t[:-16]), size=20, y=0, x = 1.2)

    mpld3.save_html(fig,chemin)

    return chemin

def afficher_graph(lst):

    path = (lst.get(Tk.END).split(' :'))[1]

    webbrowser.open(path, new=2)
    
'''
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
'''
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

fen = Tk.Tk()

fen.title('NLPvs')
fen.configure(background = '#232323')
fen.geometry('1100x900')

fontdemarrer = tkFont.Font(family='Arial',size=50)
fontcredits = tkFont.Font(family='Arial',size=8)

frame_input = Tk.Frame(fen, borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
frame_input.pack(side = Tk.LEFT, padx = 20)


ligne = Tk.Canvas(fen,bg= '#232323', height=700, borderwidth=0, width=50)
ligne.create_line(25,25,25,675,width=2,fill='#e6e6e6')
ligne.config(highlightthickness = 0, highlightbackground = "#232323", highlightcolor='#232323')
ligne.pack(side=Tk.LEFT, padx=70)


frame_droite = Tk.Frame(fen, borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
frame_droite.pack(side=Tk.LEFT, padx=20)

log_o = Image.open('logo.png')
log_o_md = log_o.resize((250, 83),Image.ANTIALIAS)
logo = ImageTk.PhotoImage(log_o_md)
logo_label = Tk.Label(frame_input, image=logo, bg = '#232323' )
logo_label.place(x=0,y=0)
logo_label.pack(padx = 40)

credit = Tk.Label(frame_input,text='par Vincent DUBOIS',fg='#e6e6e6', bg ='#232323',font=fontcredits)
credit.pack()

frame_2 = Tk.Frame(frame_droite, borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
frame_2.pack(pady=20)

frame_log = Tk.LabelFrame(frame_2, text = 'Documents clusterisés', borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
frame_log.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
frame_log.pack(padx = 0)

frame_output = Tk.Listbox(frame_log, borderwidth = 0, bg = '#e6e6e6', fg = '#FF0000', width=300)
frame_output.pack(fill = Tk.X, padx=0)

frame_graph = Tk.Frame(frame_log, borderwidth=0, relief = Tk.FLAT, bg= '#e6e6e6')
frame_graph.pack(fill=Tk.X)

btn_graph = Tk.Button(frame_graph, relief = Tk.GROOVE,text='Montrer le graph',bg='#e6e6e6', command=(lambda : afficher_graph(frame_output)))
btn_graph.config(fg='#232323')
btn_graph.pack(pady=10)

frame_cl = Tk.LabelFrame(frame_2, text = 'Clusters', borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
frame_cl.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
frame_cl.pack(fill=Tk.X, pady = 20)

cl_output = Tk.Listbox(frame_cl, borderwidth = 0, bg = '#e6e6e6', fg = '#FF0000',)
cl_output.pack(fill=Tk.X,padx=0)

bench_et_start = Tk.Frame(frame_2, relief = Tk.FLAT, bg = '#232323')
bench_et_start.pack(fill=Tk.X, padx=0)

bench = Tk.LabelFrame(bench_et_start,text = "Temps d'exécution",borderwidth = 0, relief = Tk.FLAT, bg = '#232323')
bench.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
bench.pack(pady=20, fill=Tk.X)

bench_txt = Tk.Label(bench, text="", bg='#e6e6e6',borderwidth=0)
bench_txt.pack(fill=Tk.X, padx=0, pady=0)

#data à clusteriser

dir_test = Tk.LabelFrame(frame_input, borderwidth=0, text='Dossier contenant les fichiers à classer', relief = Tk.FLAT, bg = '#232323')
dir_test.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_test.pack(fill=Tk.X, expand = 'yes', pady=20)

frame_btn_test = Tk.Frame(dir_test, borderwidth=0, relief = Tk.FLAT, bg= '#e6e6e6')
frame_btn_test.pack(fill=Tk.X)

btn_test = Tk.Button(frame_btn_test, relief = Tk.GROOVE,text='Sélectionner un dossier',bg='#e6e6e6', command=(lambda : chemin_dossier_test(r_test)))
btn_test.config(fg='#232323')
btn_test.pack(pady=10)

r_test = Tk.Label(frame_btn_test, text = "PDF", bg = '#e6e6e6')
r_test.pack(fill=Tk.X, pady=10)

#choix type de fichier

pdf = Tk.BooleanVar()

train_true = Tk.Radiobutton(frame_btn_test, text="PDF", variable=pdf, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_true.pack(pady=5)
train_false = Tk.Radiobutton(frame_btn_test, text="HTML", variable=pdf, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_false.pack(pady=5)

#selection du modele

dir_mdl = Tk.LabelFrame(frame_input, borderwidth=0, text='Sélection du modèle de classification', relief = Tk.FLAT, bg = '#232323')
dir_mdl.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
dir_mdl.pack(fill=Tk.X, expand = 'yes', pady=20)

frame_btn_mdl = Tk.Frame(dir_mdl, borderwidth=0, relief = Tk.FLAT, bg= '#e6e6e6')
frame_btn_mdl.pack(fill=Tk.X)

btn_mdl = Tk.Button(frame_btn_mdl, relief = Tk.GROOVE,text='Sélectionner un dossier',bg='#e6e6e6',command=(lambda : chemin_dossier_mdl(r_mdl)))
btn_mdl.config(fg='#232323')
btn_mdl.pack(pady=10)

r_mdl = Tk.Label(frame_btn_mdl, text = "MODEL/DEFAUT", bg = '#e6e6e6')
r_mdl.pack(fill=Tk.X, pady=10)

#entrainement

train = Tk.LabelFrame(frame_input, borderwidth=0, text='Entraîner un nouveau modèle', relief = Tk.FLAT, bg = '#232323')
train.config(fg = '#E6E6E6', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train.pack(fill=Tk.X, expand = 'yes', pady=20)

frame_train = Tk.Frame(train, borderwidth=0, relief = Tk.FLAT, bg= '#e6e6e6')
frame_train.pack(fill=Tk.X)

#choix
entrain = Tk.BooleanVar()

train_true = Tk.Radiobutton(frame_train, text="oui", variable=entrain, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_true.pack(pady=5)
train_false = Tk.Radiobutton(frame_train, text="non", variable=entrain, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
train_false.pack(pady=5)

#choix gpu/cpu
gpu = Tk.BooleanVar()

gpu_true = Tk.Radiobutton(frame_train, text="Accélération CUDA (GPU Nvidia requis)", variable=gpu, value=True,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
gpu_true.pack(pady=5)
gpu_false = Tk.Radiobutton(frame_train, text="CPU", variable=gpu, value=False,bg= '#e6e6e6',fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
gpu_false.pack(pady=5)

#nombre de clusters

clusters = Tk.LabelFrame(frame_train, borderwidth=0, text='Nombre de clusters', relief = Tk.SUNKEN, bg = '#e6e6e6')
clusters.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
clusters.pack(fill=Tk.X, expand = 'yes', pady=20)

n_clusters = Tk.Spinbox(clusters, from_=3, to=50, bg= "#e6e6e6",fg='#232323')
n_clusters.pack(pady=10)

#nom personnalisé

nm = Tk.LabelFrame(frame_train, borderwidth=0, text='Nom personnalisé', relief = Tk.SUNKEN, bg = '#e6e6e6')
nm.config(fg = '#232323', highlightthickness = 1, highlightbackground = "#E6E6E6", highlightcolor='#E6E6E6')
nm.pack(fill=Tk.X, expand = 'yes', pady=20)

nm_entry = Tk.Entry(nm, bg= "#e6e6e6",fg='#232323')
nm_entry.insert(30,'Classification')
nm_entry.pack(fill=Tk.X, expand = 'yes', pady=10)

#démarrer 

btn_start = Tk.Button(bench_et_start, text="Lancer la classification",height=3, bg = '#00C800',fg = '#E6E6E6', command=(lambda: main(frame_resultats=frame_output,frame_bench = bench_txt,frame_clusters=cl_output, dir_class= r_test, dir_mdl = r_mdl, nombre_clusters = n_clusters,nom_perso = nm_entry, entrainement_bool=entrain, gpu_bool = gpu, pdf_bool=pdf)))
btn_start.config(highlightthickness = 1, highlightbackground='#009100', highlightcolor='#009100')
btn_start.pack(pady= 20, fill=Tk.X)

quitter =  Tk.Frame(frame_droite, bg ='#232323', relief = Tk.FLAT)
quitter.pack(side = Tk.BOTTOM, pady = 20)

btn_quitter = Tk.Button(quitter, text="Quitter", command=fen.quit, bg='#C80000', fg = '#E6E6E6')
btn_quitter.config(highlightthickness = 1, highlightbackground='#910000', highlightcolor='#910000')
btn_quitter.pack()

fen.mainloop()
