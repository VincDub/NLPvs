# NLPvs (06/04/2020 Update)

<h2>About NLPvs</h2>

NLPvs is a WIP student project aimed to parse a large ammount of text from PDF files. Although it has been imagined to crunch research-oriented documents written in French (such as thesis, etc...), its use could be easily extended to other fields.

# Current Status 

<h3>Features</h3>

<ul>
  <li>directory file-searching function (scans a given folder in the same directory as the source code to find all its files, their paths and will even search in subpaths)</li>
  <li>cleaning lvl 1 function (remove non-unicode caracters from the raw extracted text)</li>
  <li>Main function (creates a dataframe from all the extracted text from the files)</li>
</ul>

<h3>In progress</h3>

<ul>
  <li>Hapaxes-finding function</li>
</ul>

# Requirements to run the latest source file

<h3>Required</h3>

<ul>
  <li>Python (3.7.6, but should work on latest available version)</li>
  <li><a href=https://www.nltk.org/>NLTK</a>, a Natural Processing Language library</li>
  <li><a href=https://pandas.pydata.org/>Pandas</a>, used to create dataframes(tables) from the extracted data</li>
  <li><a href=https://github.com/pdfminer/pdfminer.six#pdfminersix>PDFMiner.six</a>, a PDF parsing library</li>
  <li>a folder in the same directory as the source code to dump all the PDF files to process (either call it <code>PDF</code> or change the name in the source code to whatever name you wish)</li>
  <li>Some PDF files (obviously with text in them)</li>
</ul>

<h3>Optional</h3>

<ul>
  <li><a href=https://www.anaconda.com/>Anaconda</a>, an easy to setup Python developement platform</li> 
</ul>

# Roadmap

<h3>Machine-learning part</h3>

<ul>
  <li>Polish the extraction part (make a quick GUI ?)</li>
  <li>Continue the user-defined analysis and classification (finding hapaxes, statistics, lexical richness index, etc..) by treating it through NLTK</li>
  <li>Vectorize the texts (with Numpy/Rapids) and accelerate the previous operations with CUDA (Access to the GPU required !)</li>
  <li>Benchmark to compare execution between CPU and GPU</li>
  <li>(Graph generating from data)</li>
</ul>

<h3>Deep-learning part</h3>

<ul>
  <li>Create an Pytorch Framework based envrionment on the GPU-equiped workstation (docker will be needed to add the pytorch distro from nvidia repos)</li>
  <li>Using Transformer equiped with BERT, GPT, FlauBERT and CamemBERT, it will be possible to do "transfer-learning", which is re-using existing pre-trained neural networks in order to use it in our specific case (camemBERT and FlauBERT have huge databases with great success rates) (customization seems possible, but I'll need more time to dig through it)</li>
  <li>Possible use cases would be : Features extraction, paraphrasing detection (meaning proximity between two sentences) or summarizing (within a given length) </li>
  <li>Benchmark to compare execution between CPU and GPU</li>
  <li>(Graph generating from data)</li>
</ul>

# Resources

(Old Resources available in the <code>old_resources</code> folder)

<ul>
  <li><a href=https://www.nltk.org/book/ch01.html>A quick look at basic NLTK features</a></li>
  <li><a href=https://www.nltk.org/book/ch06.html/>Text classification with NLTK</a></li>
  <li><a href=https://huggingface.co/transformers/usage.html>Huggingface's Tranformer possible usages</a></li>
  <li><a href=https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>Pytorch distro from NVIDIA NGC</a></li>
  <li><a href=https://github.com/getalp/Flaubert>FlauBERT GitHub repository</a></li>
  <li><a href=https://camembert-model.fr/>CamemBERT website</a></li>
  <li><a href=https://github.com/adashofdata/nlp-in-python-tutorial/>NLP with Python - Ipython Notebooks</a></li>
  <li><a href=https://www.youtube.com/watch?v=xvqsFTUsOmc>Conference : NLP in Python</a></li>
</ul>

# Additional Notes

<ul>
  <li>I have the choice between Tensorflow and Pytorch for deep learning. For the moment, I intend to use Pytorch as it's more familiar to me.</li>
</ul>
