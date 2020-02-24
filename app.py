import pandas as pd
import numpy as np
import pickle
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC

df = pd.read_csv('./dataset-en.csv', delimiter=';')
stop_words = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
spell = SpellChecker()

df['source'] = df['source'].str.lower()


def identify_tokens(row):
    source = row['source']
    tokens = word_tokenize(source)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


df['source_'] = df.apply(identify_tokens, axis=1)


def remove_stops(row):
    source_tokenization = row['source_']
    stop = [w for w in source_tokenization if not w in stop_words]
    return (stop)


df['source_'] = df.apply(remove_stops, axis=1)


def spell_check(row):
    source_tokenization = row['source_']
    for i in range(len(source_tokenization)):
        source_tokenization[i] = spell.correction(source_tokenization[i])
    return (source_tokenization)


df['source__'] = df.apply(spell_check, axis=1)


def stem_lemma(row):
    my_list = row['source_']
    stemmed_list = [wordnet_lemmatizer.lemmatize(word) for word in my_list]
    return (stemmed_list)


df['source_'] = df.apply(stem_lemma, axis=1)


def stem_porter(row):
    my_list = row['source_']
    stemmed_list = [porter_stemmer.stem(word) for word in my_list]
    return (stemmed_list)


def stem_snowball(row):
    my_list = row['source_']
    stemmed_list = [snowball_stemmer.stem(word) for word in my_list]
    return (stemmed_list)


#df['source_'] = df.apply(stem_snowball, axis=1)
df['source_'] = df.apply(stem_porter, axis=1)


def rejoin_words(row):
    my_list = row['source_']
    joined_words = (" ".join(my_list))
    return joined_words


df['stem_meaningful'] = df.apply(rejoin_words, axis=1)

# print(df)

X = df['stem_meaningful']
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

# print(X_train)
# print(Y_train)
# print(X_test)

tfidf = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)

X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

le = preprocessing.LabelEncoder()

le.fit(list(Y_train))
Y_train_le = le.transform(list(Y_train))
Y_test_le = le.transform(list(Y_test))

# print(Y_train_le)

clf = SVC(kernel='sigmoid', C=10, gamma=0.5,
          decision_function_shape='ovo', probability=True)
#clf = CalibratedClassifierCV(clf, cv=5)
clf.fit(X_train_tf, Y_train_le)


classes = range(0, len(list(le.classes_)))
classes_label = le.inverse_transform(classes)

d = {}
for c in classes:
    d[c] = classes_label[c]

#pickle.dump(clf, open('clf_model.pkl', 'wb'))
#pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
#pickle.dump(d, open('label.pkl', 'wb'))
#pickle.load(open('label.pkl', 'rb'))

print(clf.score(X_test_tf, Y_test_le))

sentence = 'find loyalti program'

sentence = tfidf.transform([sentence])

# print(sentence)

result = clf.predict(sentence)
confidence = clf.predict_proba(sentence)[0]
#confidence = clf.decision_function(sentence)
print('>>>>>')

print("intent: " + d[result[0]])
print(confidence[result])
