import pdfminer
from pdfminer import high_level
import os
import pandas as pd
import re
import nltk
from nltk import FreqDist, word_tokenize

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

    for entry in txt:
        entry = re.sub('\w*\d\w*', '', entry)
        entry = re.sub('\n', " ", entry)
        txt_n1.append(entry)
    return txt_n1


def trouver_hapax(txt):

    hapax_txt = list()

    for entry in txt:
        entry_tk = word_tokenize(entry)
        hapax = FreqDist.hapaxes(entry_tk)
        hapax_txt.append(hapax)

    return hapax_txt



def parser_pdf_masse(dir):

    noms_fichiers = fichiers_cibles(dir)
    textes_fichiers = list()
    
    for entry in noms_fichiers:
        textes_fichiers.append(pdfminer.high_level.extract_text(entry,"",None,0,True,'utf-8',None))
    
    textes_fichiers = nettoyage_1(textes_fichiers)
    #hapax_fichiers = trouver_hapax(textes_fichiers)


    pdf_dict = {'Titre':noms_fichiers, 'Texte' :textes_fichiers}
    pdf_df = pd.DataFrame(pdf_dict)
    return pdf_df
    

print(parser_pdf_masse("PDF"))