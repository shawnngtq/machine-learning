# Natural Language Processing Fundamentals in Python

Editor: Shawn Ng<br>
Content Author: **Katharine Jarmul**<br>
Site: https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python<br>

1. Regular expressions & word tokenization
2. Simple topic identification
3. Named-entity recognition
4. Building a "fake news" classifier

## 1. Regular expressions & word tokenization
### re.split() and re.findall()
```python
# Import the regex module
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
# Import necessary modules
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Split data into sentences: sentences
sentences = sent_tokenize(data)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the data: unique_tokens
unique_tokens = set(word_tokenize(data))

# Print the unique tokens result
print(unique_tokens)
```

### re.search() and re.match()
```python
# Search for the first occurrence of "keyword" in data: match
match = re.search(r"keyword", data)
print(match.start(), match.end())

# Search for anything in square brackets before \n
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, data))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
```