import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def tokenize_lemma_stopwords(text):
    """[Standard text preprocessing using tokenization and lemmatization]

    Args:
        text ([str]): [Take a text as an input. Can be apply in a DataFrame using apply functions]

    Returns:
        [str]: [Return cleaned text]
    """

    # Initialize 
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sw = set(stopwords.words('english'))


    text = text.replace("\n", " ")
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text.lower())

    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]

    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    # tokens = [stemmer.stem(t) for t in tokens]

    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in sw] # remove stopwords
    cleaned_text = " ".join(tokens)

    return cleaned_text


def basic_stopwords(text):
    sw = set(stopwords.words('english'))
    text = text.replace("\n", " ")
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text.lower())

    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]

    # remove short words, they're probably not useful
    # tokens = [t for t in tokens if len(t) > 2]
    # tokens = [t for t in tokens if t not in sw] # remove stopwords
    cleaned_text = " ".join(tokens)
    
    return cleaned_text


def dataCleaning(df, tokenize=True):
    data = df.copy()
    if tokenize:
        data["Description"] = data["Description"].apply(tokenize_lemma_stopwords)
    else: 
        data["Description"] = data["Description"].apply(basic_stopwords)
    return data