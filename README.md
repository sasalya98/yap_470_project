470_proje.ipynb dosyasında yaptığım train, preprocessing, test kısımları var
470_test_proje.ipynb kısmında test ve metriklerden oluşan kısım var save lediğim modelleri load ederek ve preprocesslenmiş veri
setimde test edilip metriklerin ve matrixlerin bastırılmış hali var
demo.py kısmında terminal üzerinden trainlediğim modellerden birini seçerek text ve entity girdiğinizde size sentiment veren bir demo mevcut.


Kodlarını yazarken kullandığım dokümantasyonlar:

NLTK (tokenizasyon, stopword, stemmer)
https://www.nltk.org

Stanza (POS & dependency parsing)
https://stanfordnlp.github.io/stanza/

scikit-learn (TF–IDF, RandomForest, LogisticRegression, LinearSVC, MultinomialNB, MLP)
https://scikit-learn.org/stable/

imbalanced-learn (SMOTE)
https://imbalanced-learn.org/stable/

VADER SentimentIntensityAnalyzer
https://github.com/cjhutto/vaderSentiment

fuzzywuzzy (fuzzy string matching)
https://github.com/seatgeek/fuzzywuzzy

GloVe Pre-trained Embeddings
https://nlp.stanford.edu/projects/glove/

SciPy Sparse (csr_matrix, hstack)
https://docs.scipy.org/doc/scipy/reference/sparse.html

XGBoost
https://xgboost.readthedocs.io/

Pandas (veri yükleme/inceleme)
https://pandas.pydata.org/

Matplotlib (grafik çizimi)
https://matplotlib.org/
