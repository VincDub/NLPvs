# Historique d'NLPvs


<h2>v2-alpha.6</h2>
<h3>27/07/2020</h3>

<ul>
<li>Ajout d'une barre de scrolling dans les sections "hapaxes" et "noms-propres"</li>
<ul>


<h2>v2-alpha.5</h2>
<h3>23/07/2020</h3>

<ul>
<li>Intégration des hapaxes dans les résultats de l'analyse</li>
<li>Intégration des noms communs présents dans le document dans les résultats de l'analyse </li>
<ul>


<h2>v2-alpha.4</h2>
<h3>22/07/2020</h3>

<ul>
<li>Détéction et suppression plus efficace des sections peu utiles sémantiquement dans les fichiers PDF (sommaires/bibliographies/iconographies/etc...)</li>
<li>Intégration du Post tagger de l'Université de Stanford, qui permet d'identifier la fonction de chaque token dans le texte</li>
<li>Filtrage de tous les tokens aux fonctions non-utiles sémantiquement (articles/pronoms/adverbes) remplaçant le filtrage par "stop-words"</li>
<li>Extraction d'hapaxes grandement améliorée</li>
<ul>


<h2>v2-alpha.3</h2>
<h3>20/07/2020</h3>

<ul>
<li>Remplacement du module <code>PDFminer</code> par <code>PDFplumber</code>, permettant d'avoir plus de données concernant les caractères parsés d'un fichier PDF</li>
<li>Pour les fichiers PDF: Détection des titres et des corps de texte permettant la filtration des pages trop peu fournies (couvertures), des pages d'abstract ainsi que des pages de références bibliographiques/iconographiques/etc...en fin de document</li>
<li>Filtration des "hapaxes" résiduels par comparaison avec le reste des tokens (légère réduction de performances)</li>
<li>Rennomage personnalisé ré-activé : si l'utilisateur n'en indique aucun, le rennomage par défaut prend le relais</li>
<ul>


<h2>v2-alpha.2</h2>
<h3>17/07/2020</h3>

<ul>
<li>Remplacement des opérateurs de la fonction <code>nettoyage</code> par des opérations regex (gain de performances)</li>
<li>Ajout d'un rennomage automatique du fichier HTML en fonction de la quantité de documents classifiés
<ul>
<li>Si un seul document est détécté, le fichier de résultats prendra son nom</li>
<li>Si plusieurs documents sont détéctés, le fichier de résultats prendra le nom du dossier dans lequel ils sont regroupés</li>
</ul>
</li>
<li>Si un seul document est détécté, la classification est remplacée par une analyse individuelle du document dans la restitution graphique</li>
</ul>


<h2>v2-alpha.1</h2>
<h3>15/07/2020</h3>

<ul>
<li>Ajout d'un fichier<code>log.md</code>rendant compte de l'historique du développement du projet</li>
<li>Clarification du code source :</li>
<li>Réécriture en profondeur de la fonction principale <code>start</code> et changement de son nom en <code>main</code></li>
<li>Suppression des fonctions  <code>tester_texte</code>,<code>vectoriser_avec_noms</code>,<code>vectoriser_avec_noms_diff</code> devenues inutiles</li>
<li>Suppression d'anciennes fonction mises en commentaires</li>
<li>Suppression de la dépendance aux modules <code>FrenchLefffLemmatizer</code> et <code>Networkx</code></li>
<li>Rennomage de certaines variables et fonctions pour une meilleur lisibilité</li>
<li>Ajout de commentaires pour chaque fonction et lors des imports pour clarifier leurs fonctions</li>
</ul>


<h2>v1-alpha.4</h2>
<h3>10/07/2020</h3>

<ul>
<li>Optimisation de l'appel au vectoriseur TF-IDF (légère amélioration en termes de performances)</li>
<li>Suppression des fonctions <code>racin_pour_tfidf</code> et <code>ss_racin_pour_tfidf</code> devenues inutiles</li>
<li>Réécriture du fichier <code>README.md</code> en Français</li>
<li>Ajout d'un fichier <code>Requirements.md</code> référençant tous les modules requis ainsi que leurs commandes d'installation respectives</li>
<li>Ajout d'un fichier <code>main.py</code> servant de base extérimentale pour la programmation d'un CNN dans le futur</li>
</ul>

<h2>v1-alpha.3</h2>
<h3>08/07/2020</h3>

<ul>
<li>Intégration d'une légende interactive dans la représentation graphique HTML permettant de rendre visible le titre de tous les documents appartenant à un même cluster</li>
<li>Intégration d'une fonction permettant d'afficher le titre d'un document lors du survol de sa pastille</li>
<li>Ajout sur l'interface d'une option de définition d'un titre personnalisé pour la représentation graphique</li>
<li>Suppression de la fonction permettant de charger un deuxième corpus pour classifier un autre corpus en une seule opération (simplifiant la procédure d'entraînement d'un nouveau modèle)</li>
</ul>


<h2>v1-alpha.2</h2>
<h3>06/07/2020</h3>

<ul>
<li>Ajout d'une option pour pouvoir classifier des fichiers HTML</li>
<li>Remplacement de la représentation graphique statique en .PNG par une représentation graphique au format .HTML, sauvegardée sous le même dossier<code>GRAPH</code></li>
<li>Suppression de la dépendance à la liste des "stop-words" français du module NLTK</li>
<li>Ajout d'un fichier texte contenant une liste ajustée des "stop-words" français se substituant à celle de NLTK</li>
</ul>


<h2>v1-alpha.1</h2>
<h3>18/05/2020</h3>

<ul>
<li>Désactivation de l'affichage des axes sur le graphique des résultats</li>
<li>Augmentation de la taille des titres des documents</li>
</ul>

<h2>v0-alpha.8</h2>
<h3>16/05/2020</h3>

<ul>
<li>Fix du bug qui provoquait un renvoi de coordonnées nulles dans la matrice TF-IDF d'un corpus si ce dernier était classifié à partir d'un modèle existant</li>
</ul>


<h2>v0-alpha.7</h2>
<h3>14/05/2020</h3>

<ul>
<li>Ajout d'une condition permettant de détécter la compatibilité CUDA et donc de la désactiver par défaut sur un système incompatible (RAPIDS Ne fonctionnant pas sous Windows)</li>
</ul>


<h2>v0-alpha.6</h2>
<h3>11/05/2020</h3>

<ul>
<li>Ajout d'une option pour pouvoir accélérer la classification en utilisant une version de Kmeans exploitant les coeurs CUDA (si l'utilisateur possède un GPU compatible)</li>
</ul>


<h2>v0-alpha.5</h2>
<h3>04/05/2020</h3>

<ul>
<li>Ajout d'une fonction pour extraire les features les plus significatives de chaque cluster</li>
<li>Ajout d'une fonction pour générer une représentation graphique des résultats de la classification et la sauvegarder automatiquement en un fichier PNG dans un dossier nommé<code>GRAPH</code></li>
<li>Ajout d'une fonction pour générer une représentation graphique des résultats de la classification et la sauvegarder automatiquement en un fichier GEXF dans un dossier nommé<code>GEXF</code></li>
</ul>

<h2>v0-alpha.4</h2>
<h3>01/05/2020</h3>

<ul>
<li>Ajout d'une fonction pour sauvegarder chaque modèle de classification généré (dans un dossier nommé<code>MODEL</code>) pour pouvoir les réutiliser sur d'autres corpus</li>
<li>Ajoutt d'une fonction permettant de charger un deuxième corpus pour classifier un autre corpus en une seule opération</li>
</ul>


<h2>v0-alpha.3</h2>
<h3>27/04/2020</h3>

<ul>
<li>Ajout d'une interface graphique pour gérer les paramètres de la classification et afficher les résultats</li>
<li>Ajout d'une fonction pour sauvegarder chaque modèle de classification généré pour pouvoir les réutiliser sur d'autres corpus</li>
</ul>

<h2>v0-alpha.2</h2>
<h3>20/04/2020</h3>

<ul>
<li>Ajout d'une fonction pour tokeniser les chaînes de caractères nettoyées</li>
<li>Ajout d'une fonction pour construire une matrice d'indices TF-IDF à partir de l'ensemble des documents, et les clusteriser grâce à l'algorithme Kmeans</li>
<li>Ajout d'une fonction principale pour restituer les résultats</li>
</ul>


<h2>v0-alpha.1</h2>
<h3>06/04/2020</h3>

<ul>
<li>Ajout d'une fonction pour récupérer les chemins de tous les fichiers présents dans un répertoire et ses sous-dossier</li>
<li>Ajout d'une fonction pour extraire le texte de chaque document PDF présent dans un répertoire en une liste de chaînes de caractères</li>
<li>Ajout d'une fonction pour nettoyer et formater les chaînes de caractères extraites</li>
</ul>