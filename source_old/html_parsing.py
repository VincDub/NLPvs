'''# %%'''
import os
import requests #Accéder à un URL
from bs4 import BeautifulSoup #Traitement des balises XML et HTML
import re
import string
import timeit
import jupyter
import pickle

url_utilisateur = ["https://developer.nvidia.com/how-to-cuda-python", "https://pytorch.org/"]

def url_vers_txt(url):
    markup = requests.get(url).text #Récupérer le texte brut (avec balises, etc...)
    _doc = BeautifulSoup(markup,"lxml")
    titre = _doc.title.get_text()
    titre = titre.strip()
    paragraphes = _doc.find_all('p') #Récupérer les balises <p> et leur contenu
    l = len(paragraphes)-1
    texte=""
    for i in range (0,l): 
        texte += (paragraphes[i].get_text()) #Récupérer le texte des balises <p>
    return texte, titre
    
def enregistrer(texte, titre, chemin):
    fichier_txt = open(chemin + "/" + titre + ".txt", "w+")
    fichier_txt.write(str(texte))


def combiner_texte(liste):
    texte_add = ' '.join(liste)
    return texte_add

def retirer_syntaxe(texte):
    texte = texte.lower()
    texte = re.sub('\[.*?\]', '', texte)
    texte = re.sub('[%s]' % re.escape(string.punctuation), '', texte)
    texte = re.sub('\w*\d\w*', '', texte)
    texte.strip()
    return texte

'''# %%
#Benchmarking sans accélération GPU
%timeit retirer_syntaxe(url_vers_txt(url_utilisateur))'''

def transcrire_enregistrer(url, chemin):
    for i in range (0,len(url)):
        texte = retirer_syntaxe(url_vers_txt(url[i])[0])
        titre = url_vers_txt(url[i])[1]
        enregistrer(texte,titre,chemin)
        print("URL #" + str(i) +"traitée !")

transcrire_enregistrer(url_utilisateur,"export")