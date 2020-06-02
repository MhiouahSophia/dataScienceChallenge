import nltk
nltk.download('wordnet')
# package allows to tokenize sentence into words
nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize
# package to retrieve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
# package allows to stem the text (retrieve only the root)
from nltk.stem.snowball import FrenchStemmer
# package allows to
nltk.download('punkt')
# package for Lemmatize the text (take only the root)
from nltk.stem import WordNetLemmatizer
# CountVectorizer for word embeddings - word count
from sklearn.feature_extraction.text import CountVectorizer
# TfidfVectorizer for word embeddings - tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer


def fr_stop_word():
    # List of stop word
    stopWords = set(stopwords.words('french'))
    stopWords = [i for i in stopWords]
    return stopWords


def stemmed_words(doc):
    # Create a stemmer function in order to keep only root words (racines)
    stemmer = FrenchStemmer()
    analyzer = CountVectorizer().build_analyzer()

    return (stemmer.stem(w) for w in analyzer(doc))


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# word count
def word_count(data):
    stopWords = fr_stop_word()
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(data).toarray()


# tfidf

def tfidf(data):
    stopWords = fr_stop_word()
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), analyzer=stemmed_words, max_features=1000,
                                 stop_words=stopWords, strip_accents='ascii')
    return vectorizer.fit_transform(data).toarray()


# fastText


#Load the model and Retrieve 10 dimensions instead of 300 dimensions

#load the model
ft = fasttext.load_model('cc.fr.300.bin')

#Reduce the dimension of the model
fasttext.util.reduce_model(ft, 100)

#Save the reduced model
ft.save_model('cc.fr.100.bin')

#load the model
ft = fasttext.load_model('./cc.fr.10.bin')
ft.get_dimension()

#Retrieve FastText vocabulary
vocab_ft = ft.get_words()

#ft.get_sentence_vector('syndic')
#feedback_train_clean[1]

#Mapping between the word in the corpus vs FastText vocab
X_train_ft = [[ft.get_sentence_vector(word) for word in word_tokenize(x) if word in vocab_ft]
               for x in feedback_train_clean]
X_test_ft = [[ft.get_sentence_vector(word) for word in word_tokenize(x) if word in vocab_ft]
               for x in feedback_val_clean]

#Remove empty fasttext train data
X_train_ft_clean=[]
theme_train_clean = []
for i in range(0,len(X_train_ft)):
    if len(X_train_ft[i])!=0:
        X_train_ft_clean.append(X_train_ft[i])
        theme_train_clean.append(theme_train_code_clean[i])

#Remove empty fasttext test data
X_test_ft_clean=[]
theme_val_clean = []
for i in range(0,len(X_test_ft)):
    if len(X_test_ft[i])!=0:
        X_test_ft_clean.append(X_test_ft[i])
        theme_val_clean.append(theme_val_code_clean[i])