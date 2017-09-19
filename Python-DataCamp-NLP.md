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