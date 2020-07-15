# Modules requis

<h3>Environnement conseillé : Anaconda</h3>

<ul style="list-style: none;">
    <li>PDFMiner</li>
    <code>pip install pdfminer.six</code>
    <li></li>
    <li>NLTK</li>
    <code>conda install NLTK</code>
    <li></li>
    <li>Pandas</li>
    <code>conda install pandas</code>
    <li></li>
    <li>Scikit Learn</li>
    <code>conda install scikit-learn</code>
    <li></li>
    <li>Numpy</li>
    <code>conda install -c anaconda numpy</code>      
    <li></li>
    <li>CuML  (/!\ Ne fonctionne pas sous Windows)</li>
    <code>conda install -c rapidsai -c nvidia -c conda-forge -c defaults cuml=0.13</code>
    <li></li>
    <li>Matplotlib</li>
    <code>conda install matplotlib</code>      
    <li></li>
    <li>Pillow</li>
    <code>conda install -c anaconda pillow</code>      
    <li></li>
    <li>Mpld3</li>
    <code>pip install mpld3</code>
    <li></li>
    <li>Bs4</li>
    <code>conda install bs4</code>
</ul>

<h3>/!\ Bug connu concernant VSCode sous Windows</h3>

<ul style="list-style: none;">
  <li>Visual Studio Code provoque un bug lors de
l’importation du module Numpy à la date du 10/07/2020 sous Windows 10. Dans ce cas là, PyCharm est une solide alternative</li>
</ul>