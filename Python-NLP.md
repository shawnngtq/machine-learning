# Natural Language Processing Fundamentals in Python

Editor: Shawn Ng<br>
Content Author: **Katharine Jarmul**<br>
[Site](https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python)<br>

1. [Regular expressions & word tokenization](#1-regular-expressions-word-tokenization)
    - re.split() and re.findall()
    - Word tokenization with NLTK
    - re.search() and re.match()
    - Regex with NLTK tokenization
    - Non-ascii tokenization
    - Histogram of text length
2. [Simple topic identification](#2-simple-topic-identification)
    - Building a Counter with bag-of-words
    - Text preprocessing
    - Creating and querying a corpus with gensim
    - Gensim bag-of-words
    - Tf-idf
3. [Named-entity recognition](#3-named-entity-recognition)
    - NLTK
    - Pie chart
    - spaCy
    - Polyglot, French NER
    - NER via ensemble model
4. [Building a "fake news" classifier](#4-building-a-fake-news-classifier)
    - CountVectorizer for text classification
    - TfidfVectorizer for text classification
    - Inspecting the vectors
    - Training and testing the "fake news" model with CountVectorizer
    - Improving your model





## 1. Regular expressions & word tokenization
### re.split() and re.findall()
```python
import re

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.|?|!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))
```

### Word tokenization with NLTK
```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Split text into sentences: sentences
sentences = sent_tokenize(text)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the text: unique_tokens
unique_tokens = set(word_tokenize(text))

# Print the unique tokens result
print(unique_tokens)
```

### re.search() and re.match()
```python
# Search for the first occurrence of "keyword" in text: match
match = re.search(r"keyword", text)
print(match.start(), match.end())

# Search for anything in square brackets before \n
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, text))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
```

### Regex with NLTK tokenization
```python
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"

# Use the pattern on the first tweet in the tweets list
regexp_tokenize(tweets[0], pattern1)

# Write a pattern that matches both mentions and hashtags
pattern2 = r"([#|@]\w+)"

# Use the pattern on the last tweet in the tweets list
regexp_tokenize(tweets[-1], pattern2)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)
```

### Non-ascii tokenization
```python
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(text, emoji))
```

### Histogram of text length
```python
# Split the script into lines: lines
lines = text.split('\n')

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, r"\w+") for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)
plt.show()
```




## 2. Simple topic identification
### Building a Counter with bag-of-words
```python
from collections import Counter
tokens = word_tokenize(text)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))
```

### Text preprocessing
```python
# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all words in stop_words
no_stops = [t for t in alpha_only if t not in stop_words]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
```

### Creating and querying a corpus with gensim
```python
# Import Dictionary
from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])
```

### Gensim bag-of-words
```python
# Sort the text for frequency: bow_doc
bow_doc = sorted(text, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
  print(dictionary.get(word_id), word_count)
  
# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
  total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
  print(word_id, word_count)
```

### Tf-idf
```python
# Import TfidfModel
from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
  print(dictionary.get(term_id), weight)
```





## 3. Named-entity recognition
### NLTK
```python
# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
  for chunk in sent:
    if hasattr(chunk, "label") and chunk.label() == "NE":
      print(chunk)
```

### Pie chart
```python
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
  for chunk in sent:
    if hasattr(chunk, 'label'):
      ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()
```

### spaCy
```python
# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
  print(ent.label_, ent.text)
```

### Polyglot, French NER
```python
# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
  print(ent)

# first element is the entity tag, and the second element is the full string of the entity text
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]
```

### NER via ensemble model
```python
# Create a set of spaCy entities keeping only their text: spacy_ents
spacy_ents = {e.text for e in doc.ents} 

# Create a set of the intersection between the spacy and polyglot entities: ensemble_ents
# polyglot entities: poly_ents
ensemble_ents = spacy_ents.intersection(poly_ents)

# Print the common entities
print(ensemble_ents)

# Calculate the number of entities not included in the new ensemble set of entities: num_left_out
num_left_out = len(spacy_ents.union(poly_ents)) - len(ensemble_ents)
print(num_left_out)
```





## 4. Building a "fake news" classifier
### CountVectorizer for text classification
```python
# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])
```

### TfidfVectorizer for text classification
```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])
```

### Inspecting the vectors
```python
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df) - set(tfidf_df)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
```

### Training and testing the "fake news" model with CountVectorizer
```python
# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)
```

### Improving your model
```python
# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
  # Instantiate the classifier: nb_classifier
  nb_classifier = MultinomialNB(alpha=alpha)
  # Fit to the training data
  nb_classifier.fit(tfidf_train, y_train)
  # Predict the labels: pred
  pred = nb_classifier.predict(tfidf_test)
  # Compute accuracy: score
  score = metrics.accuracy_score(y_test, pred)
  return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
  print('Alpha: ', alpha)
  print('Score: ', train_and_predict(alpha))
  print()


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
```