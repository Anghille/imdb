from . import nlp #Own module
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import tensorflow as tf
import numpy as np
import pandas as pd

STOPWORDS = set(stopwords.words('english'))

def data_preprocessing(df, length=150, vocab_size=5000):
    
    max_length = length
    vocab_size = vocab_size
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = 'OOV'

    print("\nParameters:")
    print("   Max padding length:", length)
    print("   Vocabulary size:", vocab_size)
    print("   Using oov: True\n")
    print("-Preparing data")

    print("-Removing NaN Values from the restriction feature")
    # We take the dataframe without restriction_NaN because our goal is to predict those NaN using text classification method
    # cleanedData = df.loc[(df.Restriction == "PG13") | (df.Restriction == "Grated") | (df.Restriction == "Rrated"), ["Restriction","Description"]].reset_index(drop=True)
    cleanedData = df.loc[df["Restriction"].notna(), ["Restriction","Description"]].reset_index(drop=True)
    cleanedData = nlp.dataCleaning(cleanedData, tokenize=False)

    print("--Splitting into train-test set")
    # Splitting into train-test dataset
    cleanedTrainData , cleanedTestData = train_test_split(cleanedData, test_size=0.3, random_state=46, shuffle=True)
    cleanedValData, cleandedTestData = train_test_split(cleanedTestData, test_size=0.2, random_state=46, shuffle=True)

    print("---Preparing labels, train/test/val Restriction (labels) and description (articles)")
    # Preparing arrays
    labels = cleanedData["Restriction"]
    # Train arrays
    train_articles = cleanedTrainData["Description"].values
    train_labels = cleanedTrainData["Restriction"]
    # Validation arrays
    validation_articles = cleanedValData["Description"].values
    validation_labels = cleanedValData["Restriction"]
    # Test arrays 
    test_articles = cleandedTestData["Description"].values
    test_labels = cleandedTestData["Restriction"]

    print("----Tokenizing and padding description data")
    # Description Tokenizer
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index

    print("-----Tokenizing and padding labels data")
    # Label Tokenizer
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    print("------Converting words to sequence of numbers")
    # Train ,validation and test label sequence
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    test_labels_seq = np.array(label_tokenizer.texts_to_sequences(test_labels))


    print("-------Padding train/test/val data")
    # Train, validation and test padding
    train_sequences = tokenizer.texts_to_sequences(train_articles)
    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    test_sequences = tokenizer.texts_to_sequences(test_articles)

    test_padded = pad_sequences(validation_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, 
                                padding=padding_type, truncating=trunc_type)

    print("--------Reseting index")
    # Put sequence to start at index 0
    training_label_seq = training_label_seq-1
    validation_label_seq = validation_label_seq-1

    print("---------Preparing embedded words")
    # Preparing Pre-embeded glove words
    embeddings_index = {}

    f = open("D:/Onedrive/4 - Documents/1 - Formations/Kaggle/imdb/input/glove.6B.100d.txt", encoding="utf8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print("----------Creating Embbeded word-matrix")
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Returning: Labels, embedded matrix, X_train, Y_train, X_val, Y_val, X_test, Y_test")
    list_variables = [labels, embedding_matrix, train_padded, training_label_seq, 
                      validation_padded, validation_label_seq, test_padded, 
                      test_labels_seq]
    
    return list_variables