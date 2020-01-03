# Wine naive bayes classifier

#%% Predicting variety on full dataset

import pandas as pd
import numpy as np
import re,string

import _collections
from _collections import defaultdict

import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk import precision
from nltk.tokenize import RegexpTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn

#prepare the lemmatizer and the stop list
lmtzr = WordNetLemmatizer()
stoplist = stopwords.words('english')
stoplist.extend(["drink", "now", "wine", "flavor", "flavors", "palate"])     # wine-specific stop words

## LOAD THE LARGE DATASET
df = pd.read_csv("/Users/oliveroerskov/Documents/Skole/CogSci kandidat/Natural Language Processing/Exam/Wine project/wine-reviews/NBC_wine.csv")

#some sanity check
print(df.head(10))
print(df.columns)
print(df.shape)

#dropping duplicate rows if any
df.drop_duplicates(keep=False, inplace=True)
df.drop(["country", "id", "color"], inplace=True, axis = 1)
df.dropna(how="any", inplace=True)
print(df.shape)

#visualizing
df['description'].head(10)

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
label_names = flatten([l.lower().split(" ") for l in list(set(df.variety))])

#save descriptions as a list of strings
descriptions = list(df.description)

# RE removing words with length <=2 characters
process_dsc = list(filter(None, [re.sub(r'\b\w{1,2}\b','', x) for x in descriptions]))
# RE removing punctuation
process_dsc = list(filter(None, [re.sub(r'[\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\&\$\"\/\%\+]+','', x) for x in process_dsc]))
print("punctuation removed")

# Tokenize and get rid of CAPITAL letters
process_dsc = [
    [word for word in document.lower().split()]
    for document in process_dsc]
print("tokenizing done")

#finally, lemmatize the tokens
cleaned_descriptions = [
    [lmtzr.lemmatize(word) for word in document if word not in stoplist]
    for document in process_dsc
]
print("lemmatizing done")

# remove instances of labels from text
cleaned_descriptions_without_labelnames = [
    [word for word in document if word.lower() not in label_names]
    for document in cleaned_descriptions
]
print("labels removed from text")

# put back into dataframe
df.description = cleaned_descriptions_without_labelnames
# create documents
documents = list(zip(df.description, df.variety))

# remove data with too few observations
MIN_DOCS = 100                            #The minimum number of occurences
labels = [c for d,c in documents]
label_freqs = nltk.FreqDist(labels)
small_labels = [l for l,f in label_freqs.items() if f < MIN_DOCS]
large_labels = [l for l,f in label_freqs.items() if f >= MIN_DOCS]
documents = [(d,c) for d,c in documents if c in large_labels]
labels = [l for l in labels if l in large_labels]
print("Removed: ", len(small_labels), small_labels)

indices = np.array(list(range(len(labels))))

# split in traning and test set
X_train, X_test, y_train, y_test = train_test_split(
    indices, 
    labels, 
    test_size = 0.05,                   # 95/5 split
    stratify = labels)

train_docs = [documents[i] for i in list(X_train)]
test_docs = [documents[i] for i in list(X_test)]


# Find training vocab
all_text = flatten([d for d,c in train_docs])

# design feature extractor/vectorizer
# only 10,000 most frequent words that are not in the labels (full token match)

all_word_freqs = nltk.FreqDist(all_text)
filtered_words_freqs = all_word_freqs
for k in label_names:
    filtered_words_freqs.pop(k, None)
word_features = list(filtered_words_freqs)[:10000]       ### Selecting the number of most frequent words to be included

# vectorize - put a 1 if the word in vocab occurs in word features - put a 0 if not
def document_features(document):
    document_words = set(document)
    features = [1 if word in document_words else 0 for word in word_features]
    return features

train_features = [document_features(d) for (d,c) in train_docs]  # creates feature sets with features and classes
train_labels = [c for (d,c) in train_docs]
test_features = [document_features(d) for (d,c) in test_docs]  # creates feature sets with features and classes
test_labels = [c for (d,c) in test_docs]

#%% TRAIN CLASSIFIER

## Bernoulli NB
clf_variety = BernoulliNB()
clf_variety.fit(train_features, train_labels)
test_predictions = clf_variety.predict(test_features)
accuracy = np.mean(test_predictions == test_labels)

print("Accuracy: %f" % (accuracy))
print(classification_report(test_labels, test_predictions))

#%%  trying with MULTINOMIAL NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

X_list = [(d) for d,c in documents]
X = [' '.join(word) for word in X_list]
y = [(c) for d,c in documents]
#y = [' '.join(word) for word in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state = 42)

mnb = Pipeline([('vect', TfidfVectorizer(sublinear_tf=True)),
               ('chi', SelectKBest(chi2, k=10000)),
               ('clf', MultinomialNB()),])
mnb.fit(X_train, y_train)

%time
from sklearn.metrics import classification_report
y_pred = mnb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


#%% MOST INFORMATIVE FEATURES

vectorizer = mnb.named_steps['vect']
chi = mnb.named_steps['chi']
clf = mnb.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

large_labels.sort()
target_names = large_labels
print("top 10 keywords per class")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))


#%% ENTERING NEW INPUT


# Examples of newin descriptions for classification
napa_red = 'This is a very young, concentrated and tannin-laced wine with great potential for aging, from Tim and Marcia Mondavi\'s Pritchard Hill winery estate. An almost black color is the first impression, then an aroma of blueberries, blackberries and black currants leads to very ripe black-fruit flavors and oak spices that are thoroughly held in check by the tight texture.'
beaujolais_blanc = 'This magnificent estate at the foot of Mont Brouilly shows a Burgundian approach. Judicious wood aging brings toastiness as well as acidity to the ripe yellow and citrus fruits. It would be worth aging the wine into 2020.'
port = 'The wine\'s fine perfumed black plum fruits give a wonderful jammy character while bringing out a fresh edge. These are balanced by the dry core of this beautiful wine with its rich, generous tannins. It will all come together from 2030 in a very fine, integrated wine.'
champagne = 'A grand wine, this is the 24th blend of Grand Siécle, the producer\'s top cuvée. With wonderful toastiness, the Champagne from grand cru vineyards is at a perfect moment of maturity. Almonds mingle with the white fruits and acidity in perfect balance. Drink now.'
aus_riesling = 'This vintage of Springvale is shining. Simultaneously delicate, dry and fruity, it sings bright tunes of lavender, lemon-lime, grapefruit, white peach, and touches of white spice. The palate feels like licking a lime Popsicle without the sweetness. It\'s buoyed by tingly acidity, with a soft talc-like texture and bright citrus right to the end where the spice trickles onto the tongue at the very close. Bright, focused and mouthwatering, this is a joyful wine that drinks easily now but will gain tons more complexity with a decade or more of cellaring.'
pinot_nero = 'The perfect example of a “fruity” aroma of red wine: raspberries, cherries, blackberries, strawberries and currants berries, are perceptible even to the less trained nose.'
romanee_conti = 'So complex and so intellectual. Truly focused and brilliant balance and energy. Certainly on the plateau. Adding new layers all the time, which makes Conti so unique. Cinnamon, smoke, orange peel, white flowers, morel, mushrooms, medicine closet, blood orange, vivid rose petals, moss covered forest floor, tiny red fruits, ginger, menthol, mint and ... and ... So rich in its very own way, master balance no less. Deep and still very light footed. Jaw dropping experience, that leaves you speechless.'
zin = 'tobacco, taste explosion, blackberries, wildberries, cherries, licorice, cranberries, jammy, deep, plum, soil'
alsace = 'gooseberry balanced aciddry high acid green apples unripe strawberries starfruit petrol'


# tokenize and lemmatize new input - simple
tokenizer = RegexpTokenizer(r'\w+')
wine = tokenizer.tokenize(aus_riesling)       #### <----- INSERT WINE DESCRIPTION HERE

wine = [lmtzr.lemmatize(word) for word in wine if word not in stoplist]
wine = [x.lower() for x in wine]

#array = np.asarray(wine)

def document_features_uni(document):
    document_words = set(document)
    features = [[1 if word in document_words else 0 for word in word_features]]
    return features

len(document_features_uni(wine))

prediction = clf_var.predict(document_features_uni(wine))
prediction_prob = clf_var.predict_proba(document_features_uni(wine))
labels_list = list(set(labels))
probs_list = list(prediction_prob[0])

# All this below just to get a pie chart and a list of 5 most probable varieties 
# dictionary
d = {'Variety':sorted(labels_list), 'Probability':probs_list}
results_df = pd.DataFrame(d)
results_df

results_df['Probability'] = round(pd.to_numeric(results_df['Probability']), 2)
results_df = results_df.sort_values(by=['Probability'], ascending = False)
results_df



# Pie plot of results

import matplotlib.pyplot as plt

# plot colors
cmap = plt.get_cmap("tab20")
colors = cmap(np.arange(3)*4)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labs = results_df['Variety'][0:5]
sizes = results_df['Probability'][0:5]
explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
fig1.legend(sizes, labs, loc="best")
patches, texts = plt.pie(sizes, shadow=True, startangle=90)
plt.legend(patches, labs, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

results_df



#### Can we generate new sentences
import nltk

# RE removing punctuation
descriptions_without_punctuation = list(filter(None, [re.sub(r'[\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\&\$\"\/\%\+]+','', x) for x in descriptions]))

bigtext = (', '.join(descriptions_without_punctuation))
bigtext = bigtext.split()
bigtext_1 = nltk.Text(bigtext)

bigtext_1.generate(length=30,text_seed=["This"], random_seed=300)
