```python
!pip install bltk
```

Importing the libraries

Certainly! Here's an easy-to-understand explanation of the code you've provided:

### 1. **Import Libraries**

The code is importing several important libraries to help in building a machine learning model that works with text data. Here's what each one does:

- **`train_test_split`** (from `sklearn.model_selection`): This function helps in splitting the dataset into two parts: one for training the model and one for testing the model (usually a training set and a test set).
  
- **`Tokenizer`** (from `bltk.langtools`): This is a tool from the Bengali Natural Language Processing Toolkit (BLTK). It helps in breaking text into smaller units, called tokens (like words or phrases). It’s especially useful for text processing in Bengali.

- **`remove_stopwords`** (from `bltk.langtools`): Another function from BLTK. It removes common words that do not add much meaning (like "the", "is", "and"). These are called "stopwords". Removing them helps the model focus on important words.

- **`classification_report`, `confusion_matrix`, `plot_confusion_matrix`, `accuracy_score`** (from `sklearn.metrics`): These are used for evaluating the performance of your model.
    - `classification_report`: Gives a summary of the classification results (precision, recall, f1-score, etc.)
    - `confusion_matrix`: A table that shows the predicted vs. actual classification.
    - `plot_confusion_matrix`: A graphical version of the confusion matrix.
    - `accuracy_score`: Measures the percentage of correct predictions.

- **`make_scorer`, `roc_auc_score`** (from `sklearn.metrics`):
    - `make_scorer`: Used to create a custom scoring function.
    - `roc_auc_score`: Measures the quality of the classification model based on how well it distinguishes between classes.

- **`TfidfVectorizer`** (from `sklearn.feature_extraction.text`): Converts text data into a numerical format that can be fed into machine learning models. It gives more weight to rare words in the document and less to common ones.

- **`GridSearchCV`, `RandomizedSearchCV`** (from `sklearn.model_selection`): These are methods used to tune the hyperparameters of a machine learning model.
    - `GridSearchCV` tries every possible combination of parameters.
    - `RandomizedSearchCV` tries a random subset of combinations, making it faster.

- **`scipy.stats`**: This is a library for statistical functions. It's used for various statistical operations like hypothesis testing, probability distribution, etc.

- **`matplotlib.pyplot`**: A library for creating plots and graphs. It's used here for visualizing things like confusion matrices.

- **`sklearn.metrics`**: This is used for calculating various evaluation metrics like accuracy, precision, recall, etc., in machine learning.

- **`collections`**: Provides specialized container datatypes like `Counter`, which is useful for counting occurrences of items, especially in text data.

- **`nltk`**: The Natural Language Toolkit. It's another library for text processing. It's widely used for tokenizing, stemming, and other natural language tasks.

- **`numpy`**: A library for numerical operations. It's commonly used in machine learning for working with arrays, matrices, and performing mathematical operations.

- **`pandas`**: A library for working with structured data (like tables). It is used for reading, processing, and analyzing datasets.

- **`codecs`**: A module to handle file reading and writing, especially when dealing with non-ASCII characters (e.g., Bengali text).


```python
from sklearn.model_selection import train_test_split
from bltk.langtools import Tokenizer # BLTK: The Bengali Natural Language Processing Toolkit
from bltk.langtools import remove_stopwords
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import collections
import nltk
import numpy as np
import pandas as pd
import codecs
```

The code you've shared is importing specific items from a Python module named `bltk.langtools.banglachars`. Let's break down each part in simple terms:

### 1. **`from bltk.langtools.banglachars import`**

This part means that we are importing certain elements from the module `banglachars` that is inside the `langtools` package, which itself is part of the `bltk` package. In other words, we are accessing a set of tools designed to work with Bengali characters.

### 2. **What is `bltk.langtools.banglachars`?**

This is a Python module that contains data related to Bengali characters and scripts. It helps in handling Bengali text more easily by organizing various types of Bengali characters into categories.

### 3. **The items being imported:**

- **`vowels`**: This will likely be a collection (such as a list or set) of all the Bengali vowel characters. In Bengali, vowels are the basic sounds that form the building blocks of the language.

- **`vowel_signs`**: These are symbols in Bengali that modify the sound of a vowel when placed with consonants. In Bengali script, vowel signs are written around consonants to represent different vowel sounds.

- **`consonants`**: This is a collection of all the Bengali consonant characters. These characters, when combined with vowels, form syllables and words in Bengali.

- **`digits`**: This would be the set of Bengali numerals (digits), such as ১ (1), ২ (2), etc. It's different from Arabic numerals (0, 1, 2, etc.) that are commonly used in English.

- **`operators`**: This could include mathematical or linguistic operators used in Bengali texts, such as punctuation marks or symbols used in writing.

- **`punctuations`**: A collection of punctuation marks used in Bengali, like the Bengali comma (`,`) or full stop (`।`).

- **`others`**: This could be a miscellaneous collection, which may include other types of characters that don't fit into the categories above, such as special symbols, spaces, or perhaps characters used in specific contexts.


The code you've shared is importing specific items from a Python module named `bltk.langtools.banglachars`. Let's break down each part in simple terms:

### 1. **`from bltk.langtools.banglachars import`**

This part means that we are importing certain elements from the module `banglachars` that is inside the `langtools` package, which itself is part of the `bltk` package. In other words, we are accessing a set of tools designed to work with Bengali characters.

### 2. **What is `bltk.langtools.banglachars`?**

This is a Python module that contains data related to Bengali characters and scripts. It helps in handling Bengali text more easily by organizing various types of Bengali characters into categories.

### 3. **The items being imported:**

- **`vowels`**: This will likely be a collection (such as a list or set) of all the Bengali vowel characters. In Bengali, vowels are the basic sounds that form the building blocks of the language.

- **`vowel_signs`**: These are symbols in Bengali that modify the sound of a vowel when placed with consonants. In Bengali script, vowel signs are written around consonants to represent different vowel sounds.

- **`consonants`**: This is a collection of all the Bengali consonant characters. These characters, when combined with vowels, form syllables and words in Bengali.

- **`digits`**: This would be the set of Bengali numerals (digits), such as ১ (1), ২ (2), etc. It's different from Arabic numerals (0, 1, 2, etc.) that are commonly used in English.

- **`operators`**: This could include mathematical or linguistic operators used in Bengali texts, such as punctuation marks or symbols used in writing.

- **`punctuations`**: A collection of punctuation marks used in Bengali, like the Bengali comma (`,`) or full stop (`।`).

- **`others`**: This could be a miscellaneous collection, which may include other types of characters that don't fit into the categories above, such as special symbols, spaces, or perhaps characters used in specific contexts.


```python
from bltk.langtools.banglachars import (vowels,
                                        vowel_signs,
                                        consonants,
                                        digits,
                                        operators,
                                        punctuations,
                                        others)
```


```python
print(f'Vowels: {vowels}')
print(f'Vowel signs: {vowel_signs}')
print(f'Consonants: {consonants}')
print(f'Digits: {digits}')
print(f'Operators: {operators}')
print(f'Punctuation marks: {punctuations}')
print(f'Others: {others}')
```

    Vowels: ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ঌ', 'এ', 'ঐ', 'ও', 'ঔ']
    Vowel signs: ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ে', 'ৈ', 'ো', 'ৌ']
    Consonants: ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ']
    Digits: ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
    Operators: ['=', '+', '-', '*', '/', '%', '<', '>', '×', '÷']
    Punctuation marks: ['।', ',', ';', ':', '?', '!', "'", '.', '"', '-', '[', ']', '{', '}', '(', ')', '–', '—', '―', '~']
    Others: ['৳', '৺', '্', 'ঀ', 'ঽ', '#', '$']



```python
INPUT_FILE = "ecommerce_dataset.txt"
stopwords_list ="stopwords.txt"
```

## Dataset Preparation and Cleaning

Let's break down the code step by step and explain it in simple terms:

### 1. **Initial Setup**
```python
counter = collections.Counter()
tokenizer = Tokenizer()
maxlen = 0
xs, ys = [], []
bangla_stopwords = codecs.open(stopwords_list,'r',encoding='utf-8').read().split()
```

- **`counter = collections.Counter()`**: This creates a `Counter` object from the `collections` module. It will be used to count the frequency of words that appear in the sentences.
  
- **`tokenizer = Tokenizer()`**: This creates an instance of the `Tokenizer` class. This object will be responsible for breaking sentences into individual words (also known as tokenizing).

- **`maxlen = 0`**: This initializes a variable `maxlen` to 0. This will be used to track the longest sentence (in terms of number of words after removing stop words and punctuation).

- **`xs, ys = [], []`**: These are two empty lists where:
  - `xs` will store the cleaned sentences (i.e., sentences with stop words and punctuation removed).
  - `ys` will store the corresponding labels (or categories) of those sentences.

- **`bangla_stopwords = codecs.open(stopwords_list, 'r', encoding='utf-8').read().split()`**:
  - This line loads a list of Bangla (Bengali) stopwords (common words like "the", "and", "in", etc., that are usually removed during text processing because they don't add much meaning).
  - It reads the stopwords from a file (`stopwords_list`) and splits the content into a list of words.

### 2. **Reading the Input File**
```python
fin = codecs.open(INPUT_FILE, "r", encoding='utf-16')
for line in fin:
    _, sent = line.strip().split("\t")  # Stripping the dataset based on tab. That is stripping label from sentence
```

- **`fin = codecs.open(INPUT_FILE, "r", encoding='utf-16')`**: This opens the input file (`INPUT_FILE`) in read mode with UTF-16 encoding. It's a common encoding for non-English text, like Bengali.

- **`for line in fin:`**: This starts a loop that processes each line in the input file one by one.

- **`_, sent = line.strip().split("\t")`**: Each line of the file is expected to have a label (such as a category or class) and a sentence, separated by a tab (`\t`).
  - `line.strip()` removes any extra spaces or newlines.
  - `split("\t")` splits the line into two parts: the label (stored in `_`) and the sentence (stored in `sent`).
  - The `_` variable is used here to store the label, but it's not used further in the code (it's a convention to show that the value is intentionally ignored).

### 3. **Tokenizing the Sentence**
```python
words = tokenizer.word_tokenizer(sent)
print("After Tokenizing: ", words)
```

- **`words = tokenizer.word_tokenizer(sent)`**: This line breaks the sentence (`sent`) into individual words (tokens). The `word_tokenizer` method of the `Tokenizer` object handles this process.
  
- **`print("After Tokenizing: ", words)`**: This prints the list of words obtained from the sentence after tokenization.

### 4. **Removing Punctuation**
```python
wordsExcludingPunctuationMarks = [word for word in words if word not in punctuations]
print("Truncating punctuation:", wordsExcludingPunctuationMarks)
```

- **`wordsExcludingPunctuationMarks = [word for word in words if word not in punctuations]`**: This removes punctuation marks from the list of words.
  - It loops through the `words` list and keeps only those words that are **not** in the `punctuations` list (which is defined earlier in the code).

- **`print("Truncating punctuation:", wordsExcludingPunctuationMarks)`**: This prints the list of words after punctuation marks have been removed.

### 5. **Removing Stop Words**
```python
wordsExcludingStopWords = [word.strip() for word in wordsExcludingPunctuationMarks if word not in bangla_stopwords]
print("Truncating StopWords:", wordsExcludingStopWords)
```

- **`wordsExcludingStopWords = [word.strip() for word in wordsExcludingPunctuationMarks if word not in bangla_stopwords]`**: This removes stop words from the list of words (those that appear in the `bangla_stopwords` list).
  - It loops through the words, strips any extra spaces, and only keeps words that are **not** in the `bangla_stopwords` list.

- **`print("Truncating StopWords:", wordsExcludingStopWords)`**: This prints the list of words after stop words have been removed.

### 6. **Tracking the Longest Sentence**
```python
if len(wordsExcludingStopWords) > maxlen:  # For calculating the maximum number of words in a sentence
    maxlen = len(wordsExcludingStopWords)
```

- **`if len(wordsExcludingStopWords) > maxlen:`**: This checks if the current sentence (after removing stop words and punctuation) has more words than the longest sentence encountered so far.
  - If it does, it updates the `maxlen` variable to the length of this sentence.

### 7. **Counting Word Frequencies**
```python
for wordExcludingStopWords in wordsExcludingStopWords:
    counter[wordExcludingStopWords] += 1  # Putting the frequency of each word in a dictionary
```

- **`for wordExcludingStopWords in wordsExcludingStopWords:`**: This loops through each word in the sentence after stop words and punctuation have been removed.

- **`counter[wordExcludingStopWords] += 1`**: For each word, it increments its count in the `counter` (a dictionary-like object). This is counting how often each word appears in the sentences processed so far.

### 8. **Storing Cleaned Data**
```python
ys.append(int(_))
xs.append(' '.join(wordsExcludingStopWords))
```

- **`ys.append(int(_))`**: This adds the label (converted to an integer) to the `ys` list. This will be the output or target label for the machine learning model (e.g., the category or class for the sentence).

- **`xs.append(' '.join(wordsExcludingStopWords))`**: This joins the list of words back into a single string (with spaces between words) and adds it to the `xs` list. This will be the input (the cleaned sentence) for the machine learning model.

### 9. **Closing the Input File**
```python
fin.close()
```

- Finally, the input file is closed after all lines have been processed.



```python
counter = collections.Counter()
tokenizer = Tokenizer()
maxlen = 0
xs, ys = [], []
bangla_stopwords = codecs.open(stopwords_list,'r',encoding='utf-8').read().split()


fin = codecs.open(INPUT_FILE, "r", encoding='utf-16')
for line in fin:

    _, sent = line.strip().split("\t") #Stripping the dataset based on tab. That is stripping label from sentence
    print("Label: ", _)
    print("Sentence: ",sent)

    words = tokenizer.word_tokenizer(sent)
    print("Afert Tokenizing: ",words)

    wordsExcludingPunctuationMarks=[word for word in words if word not in punctuations]
    print("Truncating punctuation:", wordsExcludingPunctuationMarks)

    wordsExcludingStopWords = [word.strip() for word in wordsExcludingPunctuationMarks if word not in bangla_stopwords]
    print("Truncating StopWords:", wordsExcludingStopWords)

    if len(wordsExcludingStopWords) > maxlen: #For calculating the maximum number of words in a sentence
        maxlen = len(wordsExcludingStopWords)
    for wordExcludingStopWords in wordsExcludingStopWords:
        counter[wordExcludingStopWords] += 1 #Putting the frequency of each  word in a dictionary
    print("***************************************************************************************")

    ys.append(int(_))
    xs.append(' '.join(wordsExcludingStopWords))


fin.close()


```

    Label:  0
    Sentence:  অনেকগুলা অরডার আছে একটু দেখবেন
    Afert Tokenizing:  ['অনেকগুলা', 'অরডার', 'আছে', 'একটু', 'দেখবেন']
    Truncating punctuation: ['অনেকগুলা', 'অরডার', 'আছে', 'একটু', 'দেখবেন']
    Truncating StopWords: ['অনেকগুলা', 'অরডার', 'একটু', 'দেখবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালোবাসা রইল ইভ্যালির প্রতি
    Afert Tokenizing:  ['ভালোবাসা', 'রইল', 'ইভ্যালির', 'প্রতি']
    Truncating punctuation: ['ভালোবাসা', 'রইল', 'ইভ্যালির', 'প্রতি']
    Truncating StopWords: ['ভালোবাসা', 'রইল', 'ইভ্যালির']
    ***************************************************************************************
    Label:  0
    Sentence:  আগের প্রডাক্ট ক্লিয়ার করেন তারাতাড়ি
    Afert Tokenizing:  ['আগের', 'প্রডাক্ট', 'ক্লিয়ার', 'করেন', 'তারাতাড়ি']
    Truncating punctuation: ['আগের', 'প্রডাক্ট', 'ক্লিয়ার', 'করেন', 'তারাতাড়ি']
    Truncating StopWords: ['আগের', 'প্রডাক্ট', 'ক্লিয়ার', 'তারাতাড়ি']
    ***************************************************************************************
    Label:  0
    Sentence:  আর ভাল লাগতেছে না
    Afert Tokenizing:  ['আর', 'ভাল', 'লাগতেছে', 'না']
    Truncating punctuation: ['আর', 'ভাল', 'লাগতেছে', 'না']
    Truncating StopWords: ['ভাল', 'লাগতেছে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দয়া করে একটু বলেন ভাই কবে পাবো
    Afert Tokenizing:  ['দয়া', 'করে', 'একটু', 'বলেন', 'ভাই', 'কবে', 'পাবো']
    Truncating punctuation: ['দয়া', 'করে', 'একটু', 'বলেন', 'ভাই', 'কবে', 'পাবো']
    Truncating StopWords: ['দয়া', 'একটু', 'ভাই', 'পাবো']
    ***************************************************************************************
    Label:  0
    Sentence:  সঠিক তারিখে দিতেন তাহলে কেউ অভিযোগ দিত না।
    Afert Tokenizing:  ['সঠিক', 'তারিখে', 'দিতেন', 'তাহলে', 'কেউ', 'অভিযোগ', 'দিত', 'না', '।']
    Truncating punctuation: ['সঠিক', 'তারিখে', 'দিতেন', 'তাহলে', 'কেউ', 'অভিযোগ', 'দিত', 'না']
    Truncating StopWords: ['সঠিক', 'তারিখে', 'দিতেন', 'অভিযোগ', 'দিত', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ই কমার্সের নামে আপনারা সাধারণ মানুষের সাথে যা করতেছে, একদিন এর হিসাব আপনাদের কড়ায় ঘন্ডায় দিতে হবে
    Afert Tokenizing:  ['ই', 'কমার্সের', 'নামে', 'আপনারা', 'সাধারণ', 'মানুষের', 'সাথে', 'যা', 'করতেছে', ',', 'একদিন', 'এর', 'হিসাব', 'আপনাদের', 'কড়ায়', 'ঘন্ডায়', 'দিতে', 'হবে']
    Truncating punctuation: ['ই', 'কমার্সের', 'নামে', 'আপনারা', 'সাধারণ', 'মানুষের', 'সাথে', 'যা', 'করতেছে', 'একদিন', 'এর', 'হিসাব', 'আপনাদের', 'কড়ায়', 'ঘন্ডায়', 'দিতে', 'হবে']
    Truncating StopWords: ['কমার্সের', 'নামে', 'আপনারা', 'মানুষের', 'সাথে', 'করতেছে', 'একদিন', 'হিসাব', 'আপনাদের', 'কড়ায়', 'ঘন্ডায়']
    ***************************************************************************************
    Label:  0
    Sentence:  ফাইজলামি!!!
    Afert Tokenizing:  ['ফাইজলামি!!', '!']
    Truncating punctuation: ['ফাইজলামি!!']
    Truncating StopWords: ['ফাইজলামি!!']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার দীর্ঘ হায়াত কামনা করি
    Afert Tokenizing:  ['আপনার', 'দীর্ঘ', 'হায়াত', 'কামনা', 'করি']
    Truncating punctuation: ['আপনার', 'দীর্ঘ', 'হায়াত', 'কামনা', 'করি']
    Truncating StopWords: ['দীর্ঘ', 'হায়াত', 'কামনা']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আর কিছু অডার করার মত টাকা নাই তাই বলে কি আমার স্বপ্নের এবং খুবই প্রয়োজনীয় বাইকটা পাব না
    Afert Tokenizing:  ['ভাই', 'আর', 'কিছু', 'অডার', 'করার', 'মত', 'টাকা', 'নাই', 'তাই', 'বলে', 'কি', 'আমার', 'স্বপ্নের', 'এবং', 'খুবই', 'প্রয়োজনীয়', 'বাইকটা', 'পাব', 'না']
    Truncating punctuation: ['ভাই', 'আর', 'কিছু', 'অডার', 'করার', 'মত', 'টাকা', 'নাই', 'তাই', 'বলে', 'কি', 'আমার', 'স্বপ্নের', 'এবং', 'খুবই', 'প্রয়োজনীয়', 'বাইকটা', 'পাব', 'না']
    Truncating StopWords: ['ভাই', 'অডার', 'মত', 'টাকা', 'নাই', 'স্বপ্নের', 'খুবই', 'প্রয়োজনীয়', 'বাইকটা', 'পাব', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমার সামান্য গ্রোসারি আইটেম দিতে পারলেন না ৩ মাসে।
    Afert Tokenizing:  ['ভাই', 'আমার', 'সামান্য', 'গ্রোসারি', 'আইটেম', 'দিতে', 'পারলেন', 'না', '৩', 'মাসে', '।']
    Truncating punctuation: ['ভাই', 'আমার', 'সামান্য', 'গ্রোসারি', 'আইটেম', 'দিতে', 'পারলেন', 'না', '৩', 'মাসে']
    Truncating StopWords: ['ভাই', 'সামান্য', 'গ্রোসারি', 'আইটেম', 'পারলেন', 'না', '৩', 'মাসে']
    ***************************************************************************************
    Label:  1
    Sentence:   লক্ষ যুবকের স্বপ্ন বেঁচে পুরণ হোক
    Afert Tokenizing:  ['লক্ষ', 'যুবকের', 'স্বপ্ন', 'বেঁচে', 'পুরণ', 'হোক']
    Truncating punctuation: ['লক্ষ', 'যুবকের', 'স্বপ্ন', 'বেঁচে', 'পুরণ', 'হোক']
    Truncating StopWords: ['যুবকের', 'স্বপ্ন', 'বেঁচে', 'পুরণ']
    ***************************************************************************************
    Label:  1
    Sentence:  কথা কাজে মিল রাখলে গ্রাহক বারবে না হয় কমবে আশা করি দ্রুত সমাদান হবে ধন্যবাদ
    Afert Tokenizing:  ['কথা', 'কাজে', 'মিল', 'রাখলে', 'গ্রাহক', 'বারবে', 'না', 'হয়', 'কমবে', 'আশা', 'করি', 'দ্রুত', 'সমাদান', 'হবে', 'ধন্যবাদ']
    Truncating punctuation: ['কথা', 'কাজে', 'মিল', 'রাখলে', 'গ্রাহক', 'বারবে', 'না', 'হয়', 'কমবে', 'আশা', 'করি', 'দ্রুত', 'সমাদান', 'হবে', 'ধন্যবাদ']
    Truncating StopWords: ['কথা', 'মিল', 'রাখলে', 'গ্রাহক', 'বারবে', 'না', 'কমবে', 'আশা', 'দ্রুত', 'সমাদান', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমি আমার পন্য গুলো কবে পাবো,,,,,?
    Afert Tokenizing:  ['ভাই', 'আমি', 'আমার', 'পন্য', 'গুলো', 'কবে', 'পাবো,,,,,', '?']
    Truncating punctuation: ['ভাই', 'আমি', 'আমার', 'পন্য', 'গুলো', 'কবে', 'পাবো,,,,,']
    Truncating StopWords: ['ভাই', 'পন্য', 'গুলো', 'পাবো,,,,,']
    ***************************************************************************************
    Label:  1
    Sentence:  Daraz থেকে প্রতি মাসেই কিছু না কিছু কিনেছি। হোক সেটা বড় বা ছোট অর্ডার।
    Afert Tokenizing:  ['Daraz', 'থেকে', 'প্রতি', 'মাসেই', 'কিছু', 'না', 'কিছু', 'কিনেছি', '।', 'হোক', 'সেটা', 'বড়', 'বা', 'ছোট', 'অর্ডার', '।']
    Truncating punctuation: ['Daraz', 'থেকে', 'প্রতি', 'মাসেই', 'কিছু', 'না', 'কিছু', 'কিনেছি', 'হোক', 'সেটা', 'বড়', 'বা', 'ছোট', 'অর্ডার']
    Truncating StopWords: ['Daraz', 'মাসেই', 'না', 'কিনেছি', 'বড়', 'ছোট', 'অর্ডার']
    ***************************************************************************************
    Label:  0
    Sentence:  এই প্রডাক্ট গুলো কি পাওয়ার সম্ভাবনা আছে ভাই?
    Afert Tokenizing:  ['এই', 'প্রডাক্ট', 'গুলো', 'কি', 'পাওয়ার', 'সম্ভাবনা', 'আছে', 'ভাই', '?']
    Truncating punctuation: ['এই', 'প্রডাক্ট', 'গুলো', 'কি', 'পাওয়ার', 'সম্ভাবনা', 'আছে', 'ভাই']
    Truncating StopWords: ['প্রডাক্ট', 'গুলো', 'পাওয়ার', 'সম্ভাবনা', 'ভাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কোনো খবর বার্তা নাই, যত তাড়াতাড়ি পারেন ডেলিভারি দেন...
    Afert Tokenizing:  ['কোনো', 'খবর', 'বার্তা', 'নাই', ',', 'যত', 'তাড়াতাড়ি', 'পারেন', 'ডেলিভারি', 'দেন..', '.']
    Truncating punctuation: ['কোনো', 'খবর', 'বার্তা', 'নাই', 'যত', 'তাড়াতাড়ি', 'পারেন', 'ডেলিভারি', 'দেন..']
    Truncating StopWords: ['খবর', 'বার্তা', 'নাই', 'তাড়াতাড়ি', 'ডেলিভারি', 'দেন..']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনারা আর কখনো মানুষের আস্থা অর্জন করতে পারবেন না ১০০%!!!
    Afert Tokenizing:  ['আপনারা', 'আর', 'কখনো', 'মানুষের', 'আস্থা', 'অর্জন', 'করতে', 'পারবেন', 'না', '১০০%!!', '!']
    Truncating punctuation: ['আপনারা', 'আর', 'কখনো', 'মানুষের', 'আস্থা', 'অর্জন', 'করতে', 'পারবেন', 'না', '১০০%!!']
    Truncating StopWords: ['আপনারা', 'কখনো', 'মানুষের', 'আস্থা', 'অর্জন', 'পারবেন', 'না', '১০০%!!']
    ***************************************************************************************
    Label:  0
    Sentence:  কাষ্টমারের ভোগান্তি কমিয়ে, কথা অনুযায়ী কাজ করারও পরামর্শ রইলো
    Afert Tokenizing:  ['কাষ্টমারের', 'ভোগান্তি', 'কমিয়ে', ',', 'কথা', 'অনুযায়ী', 'কাজ', 'করারও', 'পরামর্শ', 'রইলো']
    Truncating punctuation: ['কাষ্টমারের', 'ভোগান্তি', 'কমিয়ে', 'কথা', 'অনুযায়ী', 'কাজ', 'করারও', 'পরামর্শ', 'রইলো']
    Truncating StopWords: ['কাষ্টমারের', 'ভোগান্তি', 'কমিয়ে', 'কথা', 'অনুযায়ী', 'করারও', 'পরামর্শ', 'রইলো']
    ***************************************************************************************
    Label:  0
    Sentence:  এমন ওয়াদা দিবেন না, যেটা রক্ষা করতে পারবেন না। ধন্যবাদ।
    Afert Tokenizing:  ['এমন', 'ওয়াদা', 'দিবেন', 'না', ',', 'যেটা', 'রক্ষা', 'করতে', 'পারবেন', 'না', '।', 'ধন্যবাদ', '।']
    Truncating punctuation: ['এমন', 'ওয়াদা', 'দিবেন', 'না', 'যেটা', 'রক্ষা', 'করতে', 'পারবেন', 'না', 'ধন্যবাদ']
    Truncating StopWords: ['ওয়াদা', 'দিবেন', 'না', 'যেটা', 'রক্ষা', 'পারবেন', 'না', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  সততা নিয়ে ব্যবসা করলে সফলতা আসবে ।
    Afert Tokenizing:  ['সততা', 'নিয়ে', 'ব্যবসা', 'করলে', 'সফলতা', 'আসবে', '', '।']
    Truncating punctuation: ['সততা', 'নিয়ে', 'ব্যবসা', 'করলে', 'সফলতা', 'আসবে', '']
    Truncating StopWords: ['সততা', 'ব্যবসা', 'সফলতা', 'আসবে', '']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনি ভালো থাকবেন ভালো রাখবেন আপনার গ্রাহক দের।
    Afert Tokenizing:  ['আপনি', 'ভালো', 'থাকবেন', 'ভালো', 'রাখবেন', 'আপনার', 'গ্রাহক', 'দের', '।']
    Truncating punctuation: ['আপনি', 'ভালো', 'থাকবেন', 'ভালো', 'রাখবেন', 'আপনার', 'গ্রাহক', 'দের']
    Truncating StopWords: ['ভালো', 'ভালো', 'রাখবেন', 'গ্রাহক', 'দের']
    ***************************************************************************************
    Label:  1
    Sentence:  বেস্ট অব লাক
    Afert Tokenizing:  ['বেস্ট', 'অব', 'লাক']
    Truncating punctuation: ['বেস্ট', 'অব', 'লাক']
    Truncating StopWords: ['বেস্ট', 'অব', 'লাক']
    ***************************************************************************************
    Label:  1
    Sentence:  এগিয়ে যাক ইভেলি আগামীর পথে স্বপ্ন পূরনে
    Afert Tokenizing:  ['এগিয়ে', 'যাক', 'ইভেলি', 'আগামীর', 'পথে', 'স্বপ্ন', 'পূরনে']
    Truncating punctuation: ['এগিয়ে', 'যাক', 'ইভেলি', 'আগামীর', 'পথে', 'স্বপ্ন', 'পূরনে']
    Truncating StopWords: ['এগিয়ে', 'যাক', 'ইভেলি', 'আগামীর', 'পথে', 'স্বপ্ন', 'পূরনে']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার বাকি অর্ডার গুলো দেরি হলেও এটা একটু দেওয়ার অনুরোধ করতেছি।
    Afert Tokenizing:  ['আমার', 'বাকি', 'অর্ডার', 'গুলো', 'দেরি', 'হলেও', 'এটা', 'একটু', 'দেওয়ার', 'অনুরোধ', 'করতেছি', '।']
    Truncating punctuation: ['আমার', 'বাকি', 'অর্ডার', 'গুলো', 'দেরি', 'হলেও', 'এটা', 'একটু', 'দেওয়ার', 'অনুরোধ', 'করতেছি']
    Truncating StopWords: ['বাকি', 'অর্ডার', 'গুলো', 'দেরি', 'একটু', 'দেওয়ার', 'অনুরোধ', 'করতেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  অনেকে অর্ডার করতে আসতে পারছে না কারণ যথাসময়ে পণ্য ডেলিভারি পাচ্ছে না।
    Afert Tokenizing:  ['অনেকে', 'অর্ডার', 'করতে', 'আসতে', 'পারছে', 'না', 'কারণ', 'যথাসময়ে', 'পণ্য', 'ডেলিভারি', 'পাচ্ছে', 'না', '।']
    Truncating punctuation: ['অনেকে', 'অর্ডার', 'করতে', 'আসতে', 'পারছে', 'না', 'কারণ', 'যথাসময়ে', 'পণ্য', 'ডেলিভারি', 'পাচ্ছে', 'না']
    Truncating StopWords: ['অর্ডার', 'আসতে', 'পারছে', 'না', 'যথাসময়ে', 'পণ্য', 'ডেলিভারি', 'পাচ্ছে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  যারা মানুষের কষ্টের টাকা নিয়ে ইনজয় করে তাদের মৃত্যু নেই।
    Afert Tokenizing:  ['যারা', 'মানুষের', 'কষ্টের', 'টাকা', 'নিয়ে', 'ইনজয়', 'করে', 'তাদের', 'মৃত্যু', 'নেই', '।']
    Truncating punctuation: ['যারা', 'মানুষের', 'কষ্টের', 'টাকা', 'নিয়ে', 'ইনজয়', 'করে', 'তাদের', 'মৃত্যু', 'নেই']
    Truncating StopWords: ['মানুষের', 'কষ্টের', 'টাকা', 'ইনজয়', 'মৃত্যু', 'নেই']
    ***************************************************************************************
    Label:  0
    Sentence:  সকল প্রাণীকে মৃত্যুর স্বাদ গ্রহণ করিতে হইবে।
    Afert Tokenizing:  ['সকল', 'প্রাণীকে', 'মৃত্যুর', 'স্বাদ', 'গ্রহণ', 'করিতে', 'হইবে', '।']
    Truncating punctuation: ['সকল', 'প্রাণীকে', 'মৃত্যুর', 'স্বাদ', 'গ্রহণ', 'করিতে', 'হইবে']
    Truncating StopWords: ['সকল', 'প্রাণীকে', 'মৃত্যুর', 'স্বাদ', 'গ্রহণ']
    ***************************************************************************************
    Label:  0
    Sentence:  ফটকাবাজী বাদ দেন
    Afert Tokenizing:  ['ফটকাবাজী', 'বাদ', 'দেন']
    Truncating punctuation: ['ফটকাবাজী', 'বাদ', 'দেন']
    Truncating StopWords: ['ফটকাবাজী', 'বাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  বেচে থাকুক ইভ্যালি, বেচে থাক হাজার মানুষের সপ্ন।
    Afert Tokenizing:  ['বেচে', 'থাকুক', 'ইভ্যালি', ',', 'বেচে', 'থাক', 'হাজার', 'মানুষের', 'সপ্ন', '।']
    Truncating punctuation: ['বেচে', 'থাকুক', 'ইভ্যালি', 'বেচে', 'থাক', 'হাজার', 'মানুষের', 'সপ্ন']
    Truncating StopWords: ['বেচে', 'থাকুক', 'ইভ্যালি', 'বেচে', 'থাক', 'মানুষের', 'সপ্ন']
    ***************************************************************************************
    Label:  0
    Sentence:  ওর্ডারকৃত টিভিটি ডেলিভারির জন্য ব্যবস্থা নেওয়ার জন্য তানাহলে আমি ভোক্তা অধিকার আইনে মামলা করব।
    Afert Tokenizing:  ['ওর্ডারকৃত', 'টিভিটি', 'ডেলিভারির', 'জন্য', 'ব্যবস্থা', 'নেওয়ার', 'জন্য', 'তানাহলে', 'আমি', 'ভোক্তা', 'অধিকার', 'আইনে', 'মামলা', 'করব', '।']
    Truncating punctuation: ['ওর্ডারকৃত', 'টিভিটি', 'ডেলিভারির', 'জন্য', 'ব্যবস্থা', 'নেওয়ার', 'জন্য', 'তানাহলে', 'আমি', 'ভোক্তা', 'অধিকার', 'আইনে', 'মামলা', 'করব']
    Truncating StopWords: ['ওর্ডারকৃত', 'টিভিটি', 'ডেলিভারির', 'ব্যবস্থা', 'নেওয়ার', 'তানাহলে', 'ভোক্তা', 'অধিকার', 'আইনে', 'মামলা', 'করব']
    ***************************************************************************************
    Label:  1
    Sentence:  ইভ্যালির জন্য সবসময় শুভকামনা।
    Afert Tokenizing:  ['ইভ্যালির', 'জন্য', 'সবসময়', 'শুভকামনা', '।']
    Truncating punctuation: ['ইভ্যালির', 'জন্য', 'সবসময়', 'শুভকামনা']
    Truncating StopWords: ['ইভ্যালির', 'সবসময়', 'শুভকামনা']
    ***************************************************************************************
    Label:  0
    Sentence:  জিনিস লাগবে না, আমার টাকা দেন, চলে যাই।
    Afert Tokenizing:  ['জিনিস', 'লাগবে', 'না', ',', 'আমার', 'টাকা', 'দেন', ',', 'চলে', 'যাই', '।']
    Truncating punctuation: ['জিনিস', 'লাগবে', 'না', 'আমার', 'টাকা', 'দেন', 'চলে', 'যাই']
    Truncating StopWords: ['জিনিস', 'লাগবে', 'না', 'টাকা', 'যাই']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট গুলো ডেলিভারি করবেন দ্রুত আশাবাদী।
    Afert Tokenizing:  ['প্রোডাক্ট', 'গুলো', 'ডেলিভারি', 'করবেন', 'দ্রুত', 'আশাবাদী', '।']
    Truncating punctuation: ['প্রোডাক্ট', 'গুলো', 'ডেলিভারি', 'করবেন', 'দ্রুত', 'আশাবাদী']
    Truncating StopWords: ['প্রোডাক্ট', 'গুলো', 'ডেলিভারি', 'দ্রুত', 'আশাবাদী']
    ***************************************************************************************
    Label:  1
    Sentence:   শুভ কামনা রইল
    Afert Tokenizing:  ['শুভ', 'কামনা', 'রইল']
    Truncating punctuation: ['শুভ', 'কামনা', 'রইল']
    Truncating StopWords: ['শুভ', 'কামনা', 'রইল']
    ***************************************************************************************
    Label:  0
    Sentence:  মার্চ এ অর্ডার করেও প্রোডাক্ট পাইনাই।
    Afert Tokenizing:  ['মার্চ', 'এ', 'অর্ডার', 'করেও', 'প্রোডাক্ট', 'পাইনাই', '।']
    Truncating punctuation: ['মার্চ', 'এ', 'অর্ডার', 'করেও', 'প্রোডাক্ট', 'পাইনাই']
    Truncating StopWords: ['মার্চ', 'অর্ডার', 'করেও', 'প্রোডাক্ট', 'পাইনাই']
    ***************************************************************************************
    Label:  1
    Sentence:  পাশে আছলাম, পাশে আছি, পাশে থাকমু।
    Afert Tokenizing:  ['পাশে', 'আছলাম', ',', 'পাশে', 'আছি', ',', 'পাশে', 'থাকমু', '।']
    Truncating punctuation: ['পাশে', 'আছলাম', 'পাশে', 'আছি', 'পাশে', 'থাকমু']
    Truncating StopWords: ['পাশে', 'আছলাম', 'পাশে', 'আছি', 'পাশে', 'থাকমু']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রতারক থেকে দুরে থাকুন
    Afert Tokenizing:  ['প্রতারক', 'থেকে', 'দুরে', 'থাকুন']
    Truncating punctuation: ['প্রতারক', 'থেকে', 'দুরে', 'থাকুন']
    Truncating StopWords: ['প্রতারক', 'দুরে', 'থাকুন']
    ***************************************************************************************
    Label:  0
    Sentence:  অরে বাটপার।
    Afert Tokenizing:  ['অরে', 'বাটপার', '।']
    Truncating punctuation: ['অরে', 'বাটপার']
    Truncating StopWords: ['অরে', 'বাটপার']
    ***************************************************************************************
    Label:  0
    Sentence:  তোদের লজ্জা শরম নাই৷ ৷
    Afert Tokenizing:  ['তোদের', 'লজ্জা', 'শরম', 'নাই৷', '৷']
    Truncating punctuation: ['তোদের', 'লজ্জা', 'শরম', 'নাই৷', '৷']
    Truncating StopWords: ['তোদের', 'লজ্জা', 'শরম', 'নাই৷', '৷']
    ***************************************************************************************
    Label:  0
    Sentence:  কাস্টমার কেয়ার কল করলে কেউ রিসিভ করে না কেনো,
    Afert Tokenizing:  ['কাস্টমার', 'কেয়ার', 'কল', 'করলে', 'কেউ', 'রিসিভ', 'করে', 'না', 'কেনো', ',']
    Truncating punctuation: ['কাস্টমার', 'কেয়ার', 'কল', 'করলে', 'কেউ', 'রিসিভ', 'করে', 'না', 'কেনো']
    Truncating StopWords: ['কাস্টমার', 'কেয়ার', 'কল', 'রিসিভ', 'না', 'কেনো']
    ***************************************************************************************
    Label:  0
    Sentence:  নাটক বন্ধ করেন
    Afert Tokenizing:  ['নাটক', 'বন্ধ', 'করেন']
    Truncating punctuation: ['নাটক', 'বন্ধ', 'করেন']
    Truncating StopWords: ['নাটক', 'বন্ধ']
    ***************************************************************************************
    Label:  0
    Sentence:  কবে দিবেন ভাই বুড়া হয়ে গেলে????
    Afert Tokenizing:  ['কবে', 'দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???', '?']
    Truncating punctuation: ['কবে', 'দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???']
    Truncating StopWords: ['দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???']
    ***************************************************************************************
    Label:  0
    Sentence:  কবে দিবেন ভাই বুড়া হয়ে গেলে????
    Afert Tokenizing:  ['কবে', 'দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???', '?']
    Truncating punctuation: ['কবে', 'দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???']
    Truncating StopWords: ['দিবেন', 'ভাই', 'বুড়া', 'হয়ে', 'গেলে???']
    ***************************************************************************************
    Label:  0
    Sentence:  ন্যাড়া একবারই বেল তলায় যায়
    Afert Tokenizing:  ['ন্যাড়া', 'একবারই', 'বেল', 'তলায়', 'যায়']
    Truncating punctuation: ['ন্যাড়া', 'একবারই', 'বেল', 'তলায়', 'যায়']
    Truncating StopWords: ['ন্যাড়া', 'একবারই', 'বেল', 'তলায়']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা না পাওয়া পর্যন্ত নতুন কোন অর্ডার করবো না
    Afert Tokenizing:  ['এটা', 'না', 'পাওয়া', 'পর্যন্ত', 'নতুন', 'কোন', 'অর্ডার', 'করবো', 'না']
    Truncating punctuation: ['এটা', 'না', 'পাওয়া', 'পর্যন্ত', 'নতুন', 'কোন', 'অর্ডার', 'করবো', 'না']
    Truncating StopWords: ['না', 'পাওয়া', 'অর্ডার', 'করবো', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  রাতে অর্ডার দিব
    Afert Tokenizing:  ['রাতে', 'অর্ডার', 'দিব']
    Truncating punctuation: ['রাতে', 'অর্ডার', 'দিব']
    Truncating StopWords: ['রাতে', 'অর্ডার', 'দিব']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি নিব কিন্তু লিংক কাজ করেনা কেন
    Afert Tokenizing:  ['আমি', 'নিব', 'কিন্তু', 'লিংক', 'কাজ', 'করেনা', 'কেন']
    Truncating punctuation: ['আমি', 'নিব', 'কিন্তু', 'লিংক', 'কাজ', 'করেনা', 'কেন']
    Truncating StopWords: ['নিব', 'লিংক', 'করেনা']
    ***************************************************************************************
    Label:  0
    Sentence:  এবার থাম রে তোরা
    Afert Tokenizing:  ['এবার', 'থাম', 'রে', 'তোরা']
    Truncating punctuation: ['এবার', 'থাম', 'রে', 'তোরা']
    Truncating StopWords: ['থাম', 'রে', 'তোরা']
    ***************************************************************************************
    Label:  0
    Sentence:  এখন কল রিসিভ করাও বন্ধ করে দিয়েছেন!
    Afert Tokenizing:  ['এখন', 'কল', 'রিসিভ', 'করাও', 'বন্ধ', 'করে', 'দিয়েছেন', '!']
    Truncating punctuation: ['এখন', 'কল', 'রিসিভ', 'করাও', 'বন্ধ', 'করে', 'দিয়েছেন']
    Truncating StopWords: ['কল', 'রিসিভ', 'করাও', 'বন্ধ', 'দিয়েছেন']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রতারককে কেউ বিশ্বাস করে না
    Afert Tokenizing:  ['প্রতারককে', 'কেউ', 'বিশ্বাস', 'করে', 'না']
    Truncating punctuation: ['প্রতারককে', 'কেউ', 'বিশ্বাস', 'করে', 'না']
    Truncating StopWords: ['প্রতারককে', 'বিশ্বাস', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  নিব আমি দাম টা বলেন
    Afert Tokenizing:  ['নিব', 'আমি', 'দাম', 'টা', 'বলেন']
    Truncating punctuation: ['নিব', 'আমি', 'দাম', 'টা', 'বলেন']
    Truncating StopWords: ['নিব', 'দাম', 'টা']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রতারণা ছাড়া কিছুই না
    Afert Tokenizing:  ['প্রতারণা', 'ছাড়া', 'কিছুই', 'না']
    Truncating punctuation: ['প্রতারণা', 'ছাড়া', 'কিছুই', 'না']
    Truncating StopWords: ['প্রতারণা', 'ছাড়া', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আগের টা ডেলিভারী করেন
    Afert Tokenizing:  ['আগের', 'টা', 'ডেলিভারী', 'করেন']
    Truncating punctuation: ['আগের', 'টা', 'ডেলিভারী', 'করেন']
    Truncating StopWords: ['আগের', 'টা', 'ডেলিভারী']
    ***************************************************************************************
    Label:  0
    Sentence:  ইহা এখন ফ্রী দিলেও নিবে না
    Afert Tokenizing:  ['ইহা', 'এখন', 'ফ্রী', 'দিলেও', 'নিবে', 'না']
    Truncating punctuation: ['ইহা', 'এখন', 'ফ্রী', 'দিলেও', 'নিবে', 'না']
    Truncating StopWords: ['ফ্রী', 'দিলেও', 'নিবে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  নির্লজ্জ
    Afert Tokenizing:  ['নির্লজ্জ']
    Truncating punctuation: ['নির্লজ্জ']
    Truncating StopWords: ['নির্লজ্জ']
    ***************************************************************************************
    Label:  0
    Sentence:  সালা ধান্দাবাজ
    Afert Tokenizing:  ['সালা', 'ধান্দাবাজ']
    Truncating punctuation: ['সালা', 'ধান্দাবাজ']
    Truncating StopWords: ['সালা', 'ধান্দাবাজ']
    ***************************************************************************************
    Label:  0
    Sentence:  ক্যাশ অন ডেলিভারিতে আপনাদের সমাস্যা কোথায় বুঝলাম না।
    Afert Tokenizing:  ['ক্যাশ', 'অন', 'ডেলিভারিতে', 'আপনাদের', 'সমাস্যা', 'কোথায়', 'বুঝলাম', 'না', '।']
    Truncating punctuation: ['ক্যাশ', 'অন', 'ডেলিভারিতে', 'আপনাদের', 'সমাস্যা', 'কোথায়', 'বুঝলাম', 'না']
    Truncating StopWords: ['ক্যাশ', 'অন', 'ডেলিভারিতে', 'আপনাদের', 'সমাস্যা', 'কোথায়', 'বুঝলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  এভাবেই এগিয়ে যেতে হবে। হার মানলে হবে না
    Afert Tokenizing:  ['এভাবেই', 'এগিয়ে', 'যেতে', 'হবে', '।', 'হার', 'মানলে', 'হবে', 'না']
    Truncating punctuation: ['এভাবেই', 'এগিয়ে', 'যেতে', 'হবে', 'হার', 'মানলে', 'হবে', 'না']
    Truncating StopWords: ['এভাবেই', 'এগিয়ে', 'হার', 'মানলে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ফেসবুক পোষ্ট ডিজাইন তো ভালই বানিয়েছেন।
    Afert Tokenizing:  ['ফেসবুক', 'পোষ্ট', 'ডিজাইন', 'তো', 'ভালই', 'বানিয়েছেন', '।']
    Truncating punctuation: ['ফেসবুক', 'পোষ্ট', 'ডিজাইন', 'তো', 'ভালই', 'বানিয়েছেন']
    Truncating StopWords: ['ফেসবুক', 'পোষ্ট', 'ডিজাইন', 'ভালই', 'বানিয়েছেন']
    ***************************************************************************************
    Label:  1
    Sentence:  বিজনেস চালিয়ে যান... ইনশাআল্লাহ বিজয় সুনিশ্চিত...
    Afert Tokenizing:  ['বিজনেস', 'চালিয়ে', 'যান..', '.', 'ইনশাআল্লাহ', 'বিজয়', 'সুনিশ্চিত..', '.']
    Truncating punctuation: ['বিজনেস', 'চালিয়ে', 'যান..', 'ইনশাআল্লাহ', 'বিজয়', 'সুনিশ্চিত..']
    Truncating StopWords: ['বিজনেস', 'চালিয়ে', 'যান..', 'ইনশাআল্লাহ', 'বিজয়', 'সুনিশ্চিত..']
    ***************************************************************************************
    Label:  0
    Sentence:  লাজ লজ্জা কি আর আসে।
    Afert Tokenizing:  ['লাজ', 'লজ্জা', 'কি', 'আর', 'আসে', '।']
    Truncating punctuation: ['লাজ', 'লজ্জা', 'কি', 'আর', 'আসে']
    Truncating StopWords: ['লাজ', 'লজ্জা', 'আসে']
    ***************************************************************************************
    Label:  0
    Sentence:  দোকানে দাম আনেক কম আপনাদের সপ থেকে।
    Afert Tokenizing:  ['দোকানে', 'দাম', 'আনেক', 'কম', 'আপনাদের', 'সপ', 'থেকে', '।']
    Truncating punctuation: ['দোকানে', 'দাম', 'আনেক', 'কম', 'আপনাদের', 'সপ', 'থেকে']
    Truncating StopWords: ['দোকানে', 'দাম', 'আনেক', 'কম', 'আপনাদের', 'সপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনারা জনগনের সাথে বাটপারি করতেছেন ডেলিভারি দিচ্ছেন না
    Afert Tokenizing:  ['আপনারা', 'জনগনের', 'সাথে', 'বাটপারি', 'করতেছেন', 'ডেলিভারি', 'দিচ্ছেন', 'না']
    Truncating punctuation: ['আপনারা', 'জনগনের', 'সাথে', 'বাটপারি', 'করতেছেন', 'ডেলিভারি', 'দিচ্ছেন', 'না']
    Truncating StopWords: ['আপনারা', 'জনগনের', 'সাথে', 'বাটপারি', 'করতেছেন', 'ডেলিভারি', 'দিচ্ছেন', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আর কত ছ্যাচরামি করবেন রে ভাই
    Afert Tokenizing:  ['আর', 'কত', 'ছ্যাচরামি', 'করবেন', 'রে', 'ভাই']
    Truncating punctuation: ['আর', 'কত', 'ছ্যাচরামি', 'করবেন', 'রে', 'ভাই']
    Truncating StopWords: ['ছ্যাচরামি', 'রে', 'ভাই']
    ***************************************************************************************
    Label:  0
    Sentence:  মঙ্গল গ্রহের মোবাইল?! জীবদ্দশায় ডেলিভারি হবে না।
    Afert Tokenizing:  ['মঙ্গল', 'গ্রহের', 'মোবাইল?', '!', 'জীবদ্দশায়', 'ডেলিভারি', 'হবে', 'না', '।']
    Truncating punctuation: ['মঙ্গল', 'গ্রহের', 'মোবাইল?', 'জীবদ্দশায়', 'ডেলিভারি', 'হবে', 'না']
    Truncating StopWords: ['মঙ্গল', 'গ্রহের', 'মোবাইল?', 'জীবদ্দশায়', 'ডেলিভারি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এখনো কিছু লোভী লোক আছে যারা সত্যিই অর্ডার করবে।
    Afert Tokenizing:  ['এখনো', 'কিছু', 'লোভী', 'লোক', 'আছে', 'যারা', 'সত্যিই', 'অর্ডার', 'করবে', '।']
    Truncating punctuation: ['এখনো', 'কিছু', 'লোভী', 'লোক', 'আছে', 'যারা', 'সত্যিই', 'অর্ডার', 'করবে']
    Truncating StopWords: ['এখনো', 'লোভী', 'লোক', 'সত্যিই', 'অর্ডার']
    ***************************************************************************************
    Label:  1
    Sentence:  কম দামে ভালো বাইক দিবে। অর্ডার করতে পারো
    Afert Tokenizing:  ['কম', 'দামে', 'ভালো', 'বাইক', 'দিবে', '।', 'অর্ডার', 'করতে', 'পারো']
    Truncating punctuation: ['কম', 'দামে', 'ভালো', 'বাইক', 'দিবে', 'অর্ডার', 'করতে', 'পারো']
    Truncating StopWords: ['কম', 'দামে', 'ভালো', 'বাইক', 'দিবে', 'অর্ডার', 'পারো']
    ***************************************************************************************
    Label:  0
    Sentence:  সব চিটিং বাজ
    Afert Tokenizing:  ['সব', 'চিটিং', 'বাজ']
    Truncating punctuation: ['সব', 'চিটিং', 'বাজ']
    Truncating StopWords: ['চিটিং', 'বাজ']
    ***************************************************************************************
    Label:  1
    Sentence:  লাইক দিয়ে পাশে থাকলাম
    Afert Tokenizing:  ['লাইক', 'দিয়ে', 'পাশে', 'থাকলাম']
    Truncating punctuation: ['লাইক', 'দিয়ে', 'পাশে', 'থাকলাম']
    Truncating StopWords: ['লাইক', 'দিয়ে', 'পাশে', 'থাকলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের কাস্টমার কেয়ারে আজকে 10/12 দিন কল দিয়েও কেউ ধরে না
    Afert Tokenizing:  ['আপনাদের', 'কাস্টমার', 'কেয়ারে', 'আজকে', '10/12', 'দিন', 'কল', 'দিয়েও', 'কেউ', 'ধরে', 'না']
    Truncating punctuation: ['আপনাদের', 'কাস্টমার', 'কেয়ারে', 'আজকে', '10/12', 'দিন', 'কল', 'দিয়েও', 'কেউ', 'ধরে', 'না']
    Truncating StopWords: ['আপনাদের', 'কাস্টমার', 'কেয়ারে', 'আজকে', '10/12', 'কল', 'দিয়েও', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি নিবো
    Afert Tokenizing:  ['আমি', 'নিবো']
    Truncating punctuation: ['আমি', 'নিবো']
    Truncating StopWords: ['নিবো']
    ***************************************************************************************
    Label:  1
    Sentence:  আচ্ছা আপনারা কি সুই বেচেন
    Afert Tokenizing:  ['আচ্ছা', 'আপনারা', 'কি', 'সুই', 'বেচেন']
    Truncating punctuation: ['আচ্ছা', 'আপনারা', 'কি', 'সুই', 'বেচেন']
    Truncating StopWords: ['আচ্ছা', 'আপনারা', 'সুই', 'বেচেন']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনারা ২ নাম্বার জিনিস দেন,গ্রাহকে হয়রানি করেন,
    Afert Tokenizing:  ['আপনারা', '২', 'নাম্বার', 'জিনিস', 'দেন,গ্রাহকে', 'হয়রানি', 'করেন', ',']
    Truncating punctuation: ['আপনারা', '২', 'নাম্বার', 'জিনিস', 'দেন,গ্রাহকে', 'হয়রানি', 'করেন']
    Truncating StopWords: ['আপনারা', '২', 'নাম্বার', 'জিনিস', 'দেন,গ্রাহকে', 'হয়রানি']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালোবাসা অবিরাম
    Afert Tokenizing:  ['ভালোবাসা', 'অবিরাম']
    Truncating punctuation: ['ভালোবাসা', 'অবিরাম']
    Truncating StopWords: ['ভালোবাসা', 'অবিরাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি নিব, লোকেশান প্লিজ
    Afert Tokenizing:  ['আমি', 'নিব', ',', 'লোকেশান', 'প্লিজ']
    Truncating punctuation: ['আমি', 'নিব', 'লোকেশান', 'প্লিজ']
    Truncating StopWords: ['নিব', 'লোকেশান', 'প্লিজ']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার এগিয়ে চলুক
    Afert Tokenizing:  ['চমৎকার', 'এগিয়ে', 'চলুক']
    Truncating punctuation: ['চমৎকার', 'এগিয়ে', 'চলুক']
    Truncating StopWords: ['চমৎকার', 'এগিয়ে', 'চলুক']
    ***************************************************************************************
    Label:  1
    Sentence:  শুভকামনা
    Afert Tokenizing:  ['শুভকামনা']
    Truncating punctuation: ['শুভকামনা']
    Truncating StopWords: ['শুভকামনা']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা হাস্যকর ব্যাপার এখনো আপনারা অফার দিচ্ছেন
    Afert Tokenizing:  ['এটা', 'হাস্যকর', 'ব্যাপার', 'এখনো', 'আপনারা', 'অফার', 'দিচ্ছেন']
    Truncating punctuation: ['এটা', 'হাস্যকর', 'ব্যাপার', 'এখনো', 'আপনারা', 'অফার', 'দিচ্ছেন']
    Truncating StopWords: ['হাস্যকর', 'ব্যাপার', 'এখনো', 'আপনারা', 'অফার', 'দিচ্ছেন']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাওতাবাজী এখনো ছাড়বা না?
    Afert Tokenizing:  ['ভাওতাবাজী', 'এখনো', 'ছাড়বা', 'না', '?']
    Truncating punctuation: ['ভাওতাবাজী', 'এখনো', 'ছাড়বা', 'না']
    Truncating StopWords: ['ভাওতাবাজী', 'এখনো', 'ছাড়বা', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  নাহ এইবার ঠিক আছে
    Afert Tokenizing:  ['নাহ', 'এইবার', 'ঠিক', 'আছে']
    Truncating punctuation: ['নাহ', 'এইবার', 'ঠিক', 'আছে']
    Truncating StopWords: ['নাহ', 'এইবার', 'ঠিক']
    ***************************************************************************************
    Label:  0
    Sentence:  মামলা আমিও করবো আমার চার লাখ টাকা যদি না পাই
    Afert Tokenizing:  ['মামলা', 'আমিও', 'করবো', 'আমার', 'চার', 'লাখ', 'টাকা', 'যদি', 'না', 'পাই']
    Truncating punctuation: ['মামলা', 'আমিও', 'করবো', 'আমার', 'চার', 'লাখ', 'টাকা', 'যদি', 'না', 'পাই']
    Truncating StopWords: ['মামলা', 'আমিও', 'করবো', 'লাখ', 'টাকা', 'না', 'পাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আশা ছেড়ে দাও
    Afert Tokenizing:  ['আশা', 'ছেড়ে', 'দাও']
    Truncating punctuation: ['আশা', 'ছেড়ে', 'দাও']
    Truncating StopWords: ['আশা', 'ছেড়ে', 'দাও']
    ***************************************************************************************
    Label:  0
    Sentence:  জীবনে কোনদিন ভাবি নাই যে এত বড় ধরা খাব
    Afert Tokenizing:  ['জীবনে', 'কোনদিন', 'ভাবি', 'নাই', 'যে', 'এত', 'বড়', 'ধরা', 'খাব']
    Truncating punctuation: ['জীবনে', 'কোনদিন', 'ভাবি', 'নাই', 'যে', 'এত', 'বড়', 'ধরা', 'খাব']
    Truncating StopWords: ['জীবনে', 'কোনদিন', 'ভাবি', 'নাই', 'বড়', 'খাব']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার একটা অর্ডার বাতিল কেন হয়েছে?
    Afert Tokenizing:  ['আমার', 'একটা', 'অর্ডার', 'বাতিল', 'কেন', 'হয়েছে', '?']
    Truncating punctuation: ['আমার', 'একটা', 'অর্ডার', 'বাতিল', 'কেন', 'হয়েছে']
    Truncating StopWords: ['একটা', 'অর্ডার', 'বাতিল', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  নতুন চোখে হোক না শুরু, শুরু হোক না বিশ্বাসে!!
    Afert Tokenizing:  ['নতুন', 'চোখে', 'হোক', 'না', 'শুরু', ',', 'শুরু', 'হোক', 'না', 'বিশ্বাসে!', '!']
    Truncating punctuation: ['নতুন', 'চোখে', 'হোক', 'না', 'শুরু', 'শুরু', 'হোক', 'না', 'বিশ্বাসে!']
    Truncating StopWords: ['চোখে', 'না', 'না', 'বিশ্বাসে!']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার একটা অর্ডার বাতিল কেন হয়েছে?
    Afert Tokenizing:  ['আমার', 'একটা', 'অর্ডার', 'বাতিল', 'কেন', 'হয়েছে', '?']
    Truncating punctuation: ['আমার', 'একটা', 'অর্ডার', 'বাতিল', 'কেন', 'হয়েছে']
    Truncating StopWords: ['একটা', 'অর্ডার', 'বাতিল', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার! এভাবে নিত্যনতুন অফার দিয়ে গ্রাহকদের আকৃষ্ট করুন
    Afert Tokenizing:  ['চমৎকার', '!', 'এভাবে', 'নিত্যনতুন', 'অফার', 'দিয়ে', 'গ্রাহকদের', 'আকৃষ্ট', 'করুন']
    Truncating punctuation: ['চমৎকার', 'এভাবে', 'নিত্যনতুন', 'অফার', 'দিয়ে', 'গ্রাহকদের', 'আকৃষ্ট', 'করুন']
    Truncating StopWords: ['চমৎকার', 'এভাবে', 'নিত্যনতুন', 'অফার', 'দিয়ে', 'গ্রাহকদের', 'আকৃষ্ট', 'করুন']
    ***************************************************************************************
    Label:  1
    Sentence:  শাড়ি টা আমার চাই ই চাই
    Afert Tokenizing:  ['শাড়ি', 'টা', 'আমার', 'চাই', 'ই', 'চাই']
    Truncating punctuation: ['শাড়ি', 'টা', 'আমার', 'চাই', 'ই', 'চাই']
    Truncating StopWords: ['শাড়ি', 'টা', 'চাই', 'চাই']
    ***************************************************************************************
    Label:  0
    Sentence:  এরা ভন্ড প্রতারক
    Afert Tokenizing:  ['এরা', 'ভন্ড', 'প্রতারক']
    Truncating punctuation: ['এরা', 'ভন্ড', 'প্রতারক']
    Truncating StopWords: ['ভন্ড', 'প্রতারক']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমার ২ টা ফোন এর অডার দিয়ে ছিলাম ৫ মাস হয়ে গেছে কোন খবর নাই ।
    Afert Tokenizing:  ['ভাই', 'আমার', '২', 'টা', 'ফোন', 'এর', 'অডার', 'দিয়ে', 'ছিলাম', '৫', 'মাস', 'হয়ে', 'গেছে', 'কোন', 'খবর', 'নাই', '', '।']
    Truncating punctuation: ['ভাই', 'আমার', '২', 'টা', 'ফোন', 'এর', 'অডার', 'দিয়ে', 'ছিলাম', '৫', 'মাস', 'হয়ে', 'গেছে', 'কোন', 'খবর', 'নাই', '']
    Truncating StopWords: ['ভাই', '২', 'টা', 'ফোন', 'অডার', 'দিয়ে', 'ছিলাম', '৫', 'মাস', 'হয়ে', 'খবর', 'নাই', '']
    ***************************************************************************************
    Label:  1
    Sentence:   অনন্য শপিং এক্সপেরিয়েন্স
    Afert Tokenizing:  ['অনন্য', 'শপিং', 'এক্সপেরিয়েন্স']
    Truncating punctuation: ['অনন্য', 'শপিং', 'এক্সপেরিয়েন্স']
    Truncating StopWords: ['অনন্য', 'শপিং', 'এক্সপেরিয়েন্স']
    ***************************************************************************************
    Label:  0
    Sentence:  আমাদের পুরাতন অর্ডার গুলো কি ডেলিভারি পাবো???
    Afert Tokenizing:  ['আমাদের', 'পুরাতন', 'অর্ডার', 'গুলো', 'কি', 'ডেলিভারি', 'পাবো??', '?']
    Truncating punctuation: ['আমাদের', 'পুরাতন', 'অর্ডার', 'গুলো', 'কি', 'ডেলিভারি', 'পাবো??']
    Truncating StopWords: ['পুরাতন', 'অর্ডার', 'গুলো', 'ডেলিভারি', 'পাবো??']
    ***************************************************************************************
    Label:  0
    Sentence:  পেমেন্ট করা যাচ্ছে না।
    Afert Tokenizing:  ['পেমেন্ট', 'করা', 'যাচ্ছে', 'না', '।']
    Truncating punctuation: ['পেমেন্ট', 'করা', 'যাচ্ছে', 'না']
    Truncating StopWords: ['পেমেন্ট', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আর কত দিন ঘোরানোর ইচ্ছা আছে?
    Afert Tokenizing:  ['আর', 'কত', 'দিন', 'ঘোরানোর', 'ইচ্ছা', 'আছে', '?']
    Truncating punctuation: ['আর', 'কত', 'দিন', 'ঘোরানোর', 'ইচ্ছা', 'আছে']
    Truncating StopWords: ['ঘোরানোর', 'ইচ্ছা']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ নিবে প্রোডাক্ট না দেয়ার সম্ভাবণা
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'নিবে', 'প্রোডাক্ট', 'না', 'দেয়ার', 'সম্ভাবণা']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'নিবে', 'প্রোডাক্ট', 'না', 'দেয়ার', 'সম্ভাবণা']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'নিবে', 'প্রোডাক্ট', 'না', 'দেয়ার', 'সম্ভাবণা']
    ***************************************************************************************
    Label:  1
    Sentence:  বাহ কি চমৎকার
    Afert Tokenizing:  ['বাহ', 'কি', 'চমৎকার']
    Truncating punctuation: ['বাহ', 'কি', 'চমৎকার']
    Truncating StopWords: ['বাহ', 'চমৎকার']
    ***************************************************************************************
    Label:  1
    Sentence:  দুর্বার গতিতে এগিয়ে যাও
    Afert Tokenizing:  ['দুর্বার', 'গতিতে', 'এগিয়ে', 'যাও']
    Truncating punctuation: ['দুর্বার', 'গতিতে', 'এগিয়ে', 'যাও']
    Truncating StopWords: ['দুর্বার', 'গতিতে', 'এগিয়ে', 'যাও']
    ***************************************************************************************
    Label:  1
    Sentence:  সাথে ছিলাম আছি থাকবো।
    Afert Tokenizing:  ['সাথে', 'ছিলাম', 'আছি', 'থাকবো', '।']
    Truncating punctuation: ['সাথে', 'ছিলাম', 'আছি', 'থাকবো']
    Truncating StopWords: ['সাথে', 'ছিলাম', 'আছি', 'থাকবো']
    ***************************************************************************************
    Label:  1
    Sentence:  ইভ্যালি আরও চাঙা হবে ইনশাআল্লাহ
    Afert Tokenizing:  ['ইভ্যালি', 'আরও', 'চাঙা', 'হবে', 'ইনশাআল্লাহ']
    Truncating punctuation: ['ইভ্যালি', 'আরও', 'চাঙা', 'হবে', 'ইনশাআল্লাহ']
    Truncating StopWords: ['ইভ্যালি', 'চাঙা', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  সম্পূর্ন ক্যাশ অন ডেলিভারি হলে অর্ডার দিবো, নাহলে দেয়ার ইচ্ছা নাই
    Afert Tokenizing:  ['সম্পূর্ন', 'ক্যাশ', 'অন', 'ডেলিভারি', 'হলে', 'অর্ডার', 'দিবো', ',', 'নাহলে', 'দেয়ার', 'ইচ্ছা', 'নাই']
    Truncating punctuation: ['সম্পূর্ন', 'ক্যাশ', 'অন', 'ডেলিভারি', 'হলে', 'অর্ডার', 'দিবো', 'নাহলে', 'দেয়ার', 'ইচ্ছা', 'নাই']
    Truncating StopWords: ['সম্পূর্ন', 'ক্যাশ', 'অন', 'ডেলিভারি', 'অর্ডার', 'দিবো', 'নাহলে', 'দেয়ার', 'ইচ্ছা', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  এগিয়ে যাও
    Afert Tokenizing:  ['এগিয়ে', 'যাও']
    Truncating punctuation: ['এগিয়ে', 'যাও']
    Truncating StopWords: ['এগিয়ে', 'যাও']
    ***************************************************************************************
    Label:  1
    Sentence:  এগিয়ে যাও
    Afert Tokenizing:  ['এগিয়ে', 'যাও']
    Truncating punctuation: ['এগিয়ে', 'যাও']
    Truncating StopWords: ['এগিয়ে', 'যাও']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকে প্রোডাক্ট পেয়েছি,ভালো লাগছে
    Afert Tokenizing:  ['আজকে', 'প্রোডাক্ট', 'পেয়েছি,ভালো', 'লাগছে']
    Truncating punctuation: ['আজকে', 'প্রোডাক্ট', 'পেয়েছি,ভালো', 'লাগছে']
    Truncating StopWords: ['আজকে', 'প্রোডাক্ট', 'পেয়েছি,ভালো', 'লাগছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ইনশাআল্লাহ খেলা হবে কাল
    Afert Tokenizing:  ['ইনশাআল্লাহ', 'খেলা', 'হবে', 'কাল']
    Truncating punctuation: ['ইনশাআল্লাহ', 'খেলা', 'হবে', 'কাল']
    Truncating StopWords: ['ইনশাআল্লাহ', 'খেলা', 'কাল']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম লিখতে সমস্যা কোথায়।
    Afert Tokenizing:  ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়', '।']
    Truncating punctuation: ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়']
    Truncating StopWords: ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম লিখতে সমস্যা কোথায়।
    Afert Tokenizing:  ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়', '।']
    Truncating punctuation: ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়']
    Truncating StopWords: ['দাম', 'লিখতে', 'সমস্যা', 'কোথায়']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাই আমি কিনতে আগ্রহী
    Afert Tokenizing:  ['ভাই', 'আমি', 'কিনতে', 'আগ্রহী']
    Truncating punctuation: ['ভাই', 'আমি', 'কিনতে', 'আগ্রহী']
    Truncating StopWords: ['ভাই', 'কিনতে', 'আগ্রহী']
    ***************************************************************************************
    Label:  1
    Sentence:  বেছে থাক ইভ্যালি হাজার বছর।
    Afert Tokenizing:  ['বেছে', 'থাক', 'ইভ্যালি', 'হাজার', 'বছর', '।']
    Truncating punctuation: ['বেছে', 'থাক', 'ইভ্যালি', 'হাজার', 'বছর']
    Truncating StopWords: ['বেছে', 'থাক', 'ইভ্যালি', 'বছর']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট দিলে খুশি হইতাম
    Afert Tokenizing:  ['প্রোডাক্ট', 'দিলে', 'খুশি', 'হইতাম']
    Truncating punctuation: ['প্রোডাক্ট', 'দিলে', 'খুশি', 'হইতাম']
    Truncating StopWords: ['প্রোডাক্ট', 'দিলে', 'খুশি', 'হইতাম']
    ***************************************************************************************
    Label:  0
    Sentence:  যাই হোক ম্যাসেজ এর রিপ্লাই তো দিবেন
    Afert Tokenizing:  ['যাই', 'হোক', 'ম্যাসেজ', 'এর', 'রিপ্লাই', 'তো', 'দিবেন']
    Truncating punctuation: ['যাই', 'হোক', 'ম্যাসেজ', 'এর', 'রিপ্লাই', 'তো', 'দিবেন']
    Truncating StopWords: ['যাই', 'ম্যাসেজ', 'রিপ্লাই', 'দিবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি কিনতে আগ্রহী
    Afert Tokenizing:  ['আমি', 'কিনতে', 'আগ্রহী']
    Truncating punctuation: ['আমি', 'কিনতে', 'আগ্রহী']
    Truncating StopWords: ['কিনতে', 'আগ্রহী']
    ***************************************************************************************
    Label:  1
    Sentence:  মোবাইল এর অফার দিলে বেশি সেল হবে, এবং প্রফিট ও হবে
    Afert Tokenizing:  ['মোবাইল', 'এর', 'অফার', 'দিলে', 'বেশি', 'সেল', 'হবে', ',', 'এবং', 'প্রফিট', 'ও', 'হবে']
    Truncating punctuation: ['মোবাইল', 'এর', 'অফার', 'দিলে', 'বেশি', 'সেল', 'হবে', 'এবং', 'প্রফিট', 'ও', 'হবে']
    Truncating StopWords: ['মোবাইল', 'অফার', 'দিলে', 'বেশি', 'সেল', 'প্রফিট']
    ***************************************************************************************
    Label:  1
    Sentence:  সাথে আছি
    Afert Tokenizing:  ['সাথে', 'আছি']
    Truncating punctuation: ['সাথে', 'আছি']
    Truncating StopWords: ['সাথে', 'আছি']
    ***************************************************************************************
    Label:  0
    Sentence:  এ কেমন রসিকতা
    Afert Tokenizing:  ['এ', 'কেমন', 'রসিকতা']
    Truncating punctuation: ['এ', 'কেমন', 'রসিকতা']
    Truncating StopWords: ['কেমন', 'রসিকতা']
    ***************************************************************************************
    Label:  1
    Sentence:  বেচে থাকুক ইভালী সেবা পাক হাজার মানুষ...!
    Afert Tokenizing:  ['বেচে', 'থাকুক', 'ইভালী', 'সেবা', 'পাক', 'হাজার', 'মানুষ...', '!']
    Truncating punctuation: ['বেচে', 'থাকুক', 'ইভালী', 'সেবা', 'পাক', 'হাজার', 'মানুষ...']
    Truncating StopWords: ['বেচে', 'থাকুক', 'ইভালী', 'সেবা', 'পাক', 'মানুষ...']
    ***************************************************************************************
    Label:  1
    Sentence:  সবার সব পাওনা বুঝিয়ে দিবেন। পাশেই আছি শুভ কামনা রইলো।
    Afert Tokenizing:  ['সবার', 'সব', 'পাওনা', 'বুঝিয়ে', 'দিবেন', '।', 'পাশেই', 'আছি', 'শুভ', 'কামনা', 'রইলো', '।']
    Truncating punctuation: ['সবার', 'সব', 'পাওনা', 'বুঝিয়ে', 'দিবেন', 'পাশেই', 'আছি', 'শুভ', 'কামনা', 'রইলো']
    Truncating StopWords: ['পাওনা', 'বুঝিয়ে', 'দিবেন', 'পাশেই', 'আছি', 'শুভ', 'কামনা', 'রইলো']
    ***************************************************************************************
    Label:  1
    Sentence:   দেশি প্রতিষ্ঠান বেচে থাকুক।
    Afert Tokenizing:  ['দেশি', 'প্রতিষ্ঠান', 'বেচে', 'থাকুক', '।']
    Truncating punctuation: ['দেশি', 'প্রতিষ্ঠান', 'বেচে', 'থাকুক']
    Truncating StopWords: ['দেশি', 'প্রতিষ্ঠান', 'বেচে', 'থাকুক']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি কিনতে চাচ্ছি কিন্তু পেমেন্ট করা যাচ্ছেনা
    Afert Tokenizing:  ['আমি', 'কিনতে', 'চাচ্ছি', 'কিন্তু', 'পেমেন্ট', 'করা', 'যাচ্ছেনা']
    Truncating punctuation: ['আমি', 'কিনতে', 'চাচ্ছি', 'কিন্তু', 'পেমেন্ট', 'করা', 'যাচ্ছেনা']
    Truncating StopWords: ['কিনতে', 'চাচ্ছি', 'পেমেন্ট', 'যাচ্ছেনা']
    ***************************************************************************************
    Label:  1
    Sentence:  ১০০০ অডার দিব দোকানের জন্য
    Afert Tokenizing:  ['১০০০', 'অডার', 'দিব', 'দোকানের', 'জন্য']
    Truncating punctuation: ['১০০০', 'অডার', 'দিব', 'দোকানের', 'জন্য']
    Truncating StopWords: ['১০০০', 'অডার', 'দিব', 'দোকানের']
    ***************************************************************************************
    Label:  1
    Sentence:  অজও অছি কালও থাকবো বিশ্বাস অছে এখনও
    Afert Tokenizing:  ['অজও', 'অছি', 'কালও', 'থাকবো', 'বিশ্বাস', 'অছে', 'এখনও']
    Truncating punctuation: ['অজও', 'অছি', 'কালও', 'থাকবো', 'বিশ্বাস', 'অছে', 'এখনও']
    Truncating StopWords: ['অজও', 'অছি', 'কালও', 'থাকবো', 'বিশ্বাস', 'অছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকে ১০টা বাইক অর্ডার করবো!
    Afert Tokenizing:  ['আজকে', '১০টা', 'বাইক', 'অর্ডার', 'করবো', '!']
    Truncating punctuation: ['আজকে', '১০টা', 'বাইক', 'অর্ডার', 'করবো']
    Truncating StopWords: ['আজকে', '১০টা', 'বাইক', 'অর্ডার', 'করবো']
    ***************************************************************************************
    Label:  1
    Sentence:  কি ভাবে যে ধন্যবাদ দিব আপনাদের ভাষা পাচ্ছি না
    Afert Tokenizing:  ['কি', 'ভাবে', 'যে', 'ধন্যবাদ', 'দিব', 'আপনাদের', 'ভাষা', 'পাচ্ছি', 'না']
    Truncating punctuation: ['কি', 'ভাবে', 'যে', 'ধন্যবাদ', 'দিব', 'আপনাদের', 'ভাষা', 'পাচ্ছি', 'না']
    Truncating StopWords: ['ধন্যবাদ', 'দিব', 'আপনাদের', 'ভাষা', 'পাচ্ছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এদের লজ্জা নাই।
    Afert Tokenizing:  ['এদের', 'লজ্জা', 'নাই', '।']
    Truncating punctuation: ['এদের', 'লজ্জা', 'নাই']
    Truncating StopWords: ['লজ্জা', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কয়মাসে 7 দিন হবে
    Afert Tokenizing:  ['কয়মাসে', '7', 'দিন', 'হবে']
    Truncating punctuation: ['কয়মাসে', '7', 'দিন', 'হবে']
    Truncating StopWords: ['কয়মাসে', '7']
    ***************************************************************************************
    Label:  0
    Sentence:  দয়া করে কেউ অর্ডার করবেন না।
    Afert Tokenizing:  ['দয়া', 'করে', 'কেউ', 'অর্ডার', 'করবেন', 'না', '।']
    Truncating punctuation: ['দয়া', 'করে', 'কেউ', 'অর্ডার', 'করবেন', 'না']
    Truncating StopWords: ['দয়া', 'অর্ডার', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ঘুরে দাড়াক ই-কমার্স
    Afert Tokenizing:  ['ঘুরে', 'দাড়াক', 'ই-কমার্স']
    Truncating punctuation: ['ঘুরে', 'দাড়াক', 'ই-কমার্স']
    Truncating StopWords: ['ঘুরে', 'দাড়াক', 'ই-কমার্স']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করলে পণ্য পাওয়ার নিশ্চয়তা নাই,এভাবে আর কয়দিন চলবে।
    Afert Tokenizing:  ['অর্ডার', 'করলে', 'পণ্য', 'পাওয়ার', 'নিশ্চয়তা', 'নাই,এভাবে', 'আর', 'কয়দিন', 'চলবে', '।']
    Truncating punctuation: ['অর্ডার', 'করলে', 'পণ্য', 'পাওয়ার', 'নিশ্চয়তা', 'নাই,এভাবে', 'আর', 'কয়দিন', 'চলবে']
    Truncating StopWords: ['অর্ডার', 'পণ্য', 'পাওয়ার', 'নিশ্চয়তা', 'নাই,এভাবে', 'কয়দিন', 'চলবে']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি ব্যাবহার করছি। সব মিলিয়ে দারুন একটা ফোন।
    Afert Tokenizing:  ['আমি', 'ব্যাবহার', 'করছি', '।', 'সব', 'মিলিয়ে', 'দারুন', 'একটা', 'ফোন', '।']
    Truncating punctuation: ['আমি', 'ব্যাবহার', 'করছি', 'সব', 'মিলিয়ে', 'দারুন', 'একটা', 'ফোন']
    Truncating StopWords: ['ব্যাবহার', 'করছি', 'মিলিয়ে', 'দারুন', 'একটা', 'ফোন']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি তো দেন না!!
    Afert Tokenizing:  ['ডেলিভারি', 'তো', 'দেন', 'না!', '!']
    Truncating punctuation: ['ডেলিভারি', 'তো', 'দেন', 'না!']
    Truncating StopWords: ['ডেলিভারি', 'না!']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করতে ইচ্ছে করে বার বার,কিন্তু কমেন্ট বক্স পড়ে ইচ্ছেটাই মাটি হয়ে যায়
    Afert Tokenizing:  ['অর্ডার', 'করতে', 'ইচ্ছে', 'করে', 'বার', 'বার,কিন্তু', 'কমেন্ট', 'বক্স', 'পড়ে', 'ইচ্ছেটাই', 'মাটি', 'হয়ে', 'যায়']
    Truncating punctuation: ['অর্ডার', 'করতে', 'ইচ্ছে', 'করে', 'বার', 'বার,কিন্তু', 'কমেন্ট', 'বক্স', 'পড়ে', 'ইচ্ছেটাই', 'মাটি', 'হয়ে', 'যায়']
    Truncating StopWords: ['অর্ডার', 'ইচ্ছে', 'বার,কিন্তু', 'কমেন্ট', 'বক্স', 'পড়ে', 'ইচ্ছেটাই', 'মাটি', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  অর কত দিন অপেক্ষা করতে হবে
    Afert Tokenizing:  ['অর', 'কত', 'দিন', 'অপেক্ষা', 'করতে', 'হবে']
    Truncating punctuation: ['অর', 'কত', 'দিন', 'অপেক্ষা', 'করতে', 'হবে']
    Truncating StopWords: ['অর', 'অপেক্ষা']
    ***************************************************************************************
    Label:  0
    Sentence:  আর ধৈর্য্য ধরে রাখা যাচ্ছে না।
    Afert Tokenizing:  ['আর', 'ধৈর্য্য', 'ধরে', 'রাখা', 'যাচ্ছে', 'না', '।']
    Truncating punctuation: ['আর', 'ধৈর্য্য', 'ধরে', 'রাখা', 'যাচ্ছে', 'না']
    Truncating StopWords: ['ধৈর্য্য', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আড়াই মাস হয়ে গেল এখনো পন্য পেলাম না
    Afert Tokenizing:  ['আড়াই', 'মাস', 'হয়ে', 'গেল', 'এখনো', 'পন্য', 'পেলাম', 'না']
    Truncating punctuation: ['আড়াই', 'মাস', 'হয়ে', 'গেল', 'এখনো', 'পন্য', 'পেলাম', 'না']
    Truncating StopWords: ['আড়াই', 'মাস', 'হয়ে', 'এখনো', 'পন্য', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ২৪ ঘন্টার মধ্যে দেবে। অবিশ্বাস্য
    Afert Tokenizing:  ['২৪', 'ঘন্টার', 'মধ্যে', 'দেবে', '।', 'অবিশ্বাস্য']
    Truncating punctuation: ['২৪', 'ঘন্টার', 'মধ্যে', 'দেবে', 'অবিশ্বাস্য']
    Truncating StopWords: ['২৪', 'ঘন্টার', 'দেবে', 'অবিশ্বাস্য']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের তো একটা রিভিও ভাল দেখলাম না…
    Afert Tokenizing:  ['আপনাদের', 'তো', 'একটা', 'রিভিও', 'ভাল', 'দেখলাম', 'না…']
    Truncating punctuation: ['আপনাদের', 'তো', 'একটা', 'রিভিও', 'ভাল', 'দেখলাম', 'না…']
    Truncating StopWords: ['আপনাদের', 'একটা', 'রিভিও', 'ভাল', 'দেখলাম', 'না…']
    ***************************************************************************************
    Label:  0
    Sentence:  জানুয়ারির ১৫ তারিখে অর্ডার করেছি, এখনো প্রোডাক্টটি পাই নাই,ইভ্যালি টাকা মেরে দিসে
    Afert Tokenizing:  ['জানুয়ারির', '১৫', 'তারিখে', 'অর্ডার', 'করেছি', ',', 'এখনো', 'প্রোডাক্টটি', 'পাই', 'নাই,ইভ্যালি', 'টাকা', 'মেরে', 'দিসে']
    Truncating punctuation: ['জানুয়ারির', '১৫', 'তারিখে', 'অর্ডার', 'করেছি', 'এখনো', 'প্রোডাক্টটি', 'পাই', 'নাই,ইভ্যালি', 'টাকা', 'মেরে', 'দিসে']
    Truncating StopWords: ['জানুয়ারির', '১৫', 'তারিখে', 'অর্ডার', 'করেছি', 'এখনো', 'প্রোডাক্টটি', 'পাই', 'নাই,ইভ্যালি', 'টাকা', 'মেরে', 'দিসে']
    ***************************************************************************************
    Label:  0
    Sentence:  আমিও প্রস্তুতি নিচ্ছি মামলা করার
    Afert Tokenizing:  ['আমিও', 'প্রস্তুতি', 'নিচ্ছি', 'মামলা', 'করার']
    Truncating punctuation: ['আমিও', 'প্রস্তুতি', 'নিচ্ছি', 'মামলা', 'করার']
    Truncating StopWords: ['আমিও', 'প্রস্তুতি', 'নিচ্ছি', 'মামলা']
    ***************************************************************************************
    Label:  1
    Sentence:  কুয়ালিটি আর প্রাইস ২ টাই বেস্ট
    Afert Tokenizing:  ['কুয়ালিটি', 'আর', 'প্রাইস', '২', 'টাই', 'বেস্ট']
    Truncating punctuation: ['কুয়ালিটি', 'আর', 'প্রাইস', '২', 'টাই', 'বেস্ট']
    Truncating StopWords: ['কুয়ালিটি', 'প্রাইস', '২', 'টাই', 'বেস্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  তোমাদের এই পথচলা কন্টকমুক্ত ও সুন্দর হোক।
    Afert Tokenizing:  ['তোমাদের', 'এই', 'পথচলা', 'কন্টকমুক্ত', 'ও', 'সুন্দর', 'হোক', '।']
    Truncating punctuation: ['তোমাদের', 'এই', 'পথচলা', 'কন্টকমুক্ত', 'ও', 'সুন্দর', 'হোক']
    Truncating StopWords: ['তোমাদের', 'পথচলা', 'কন্টকমুক্ত', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুণ ছিল জিনিসটা
    Afert Tokenizing:  ['দারুণ', 'ছিল', 'জিনিসটা']
    Truncating punctuation: ['দারুণ', 'ছিল', 'জিনিসটা']
    Truncating StopWords: ['দারুণ', 'জিনিসটা']
    ***************************************************************************************
    Label:  0
    Sentence:  আইনগত ব্যবস্থা নেয়া হবে
    Afert Tokenizing:  ['আইনগত', 'ব্যবস্থা', 'নেয়া', 'হবে']
    Truncating punctuation: ['আইনগত', 'ব্যবস্থা', 'নেয়া', 'হবে']
    Truncating StopWords: ['আইনগত', 'ব্যবস্থা', 'নেয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  আর কত অপেক্ষা করবো
    Afert Tokenizing:  ['আর', 'কত', 'অপেক্ষা', 'করবো']
    Truncating punctuation: ['আর', 'কত', 'অপেক্ষা', 'করবো']
    Truncating StopWords: ['অপেক্ষা', 'করবো']
    ***************************************************************************************
    Label:  0
    Sentence:  আর জীবনেও evally থেকে অর্ডার করব না।
    Afert Tokenizing:  ['আর', 'জীবনেও', 'evally', 'থেকে', 'অর্ডার', 'করব', 'না', '।']
    Truncating punctuation: ['আর', 'জীবনেও', 'evally', 'থেকে', 'অর্ডার', 'করব', 'না']
    Truncating StopWords: ['জীবনেও', 'evally', 'অর্ডার', 'করব', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  সুন্দর কালেকশন
    Afert Tokenizing:  ['সুন্দর', 'কালেকশন']
    Truncating punctuation: ['সুন্দর', 'কালেকশন']
    Truncating StopWords: ['সুন্দর', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর ড্রেস
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'ড্রেস']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'ড্রেস']
    Truncating StopWords: ['সুন্দর', 'ড্রেস']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও
    Afert Tokenizing:  ['ওয়াও']
    Truncating punctuation: ['ওয়াও']
    Truncating StopWords: ['ওয়াও']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও অসম্ভব সুন্দর
    Afert Tokenizing:  ['ওয়াও', 'অসম্ভব', 'সুন্দর']
    Truncating punctuation: ['ওয়াও', 'অসম্ভব', 'সুন্দর']
    Truncating StopWords: ['ওয়াও', 'অসম্ভব', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুন
    Afert Tokenizing:  ['দারুন']
    Truncating punctuation: ['দারুন']
    Truncating StopWords: ['দারুন']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর
    Afert Tokenizing:  ['খুব', 'সুন্দর']
    Truncating punctuation: ['খুব', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  শাড়ীর কালার টা খুবই সুন্দর
    Afert Tokenizing:  ['শাড়ীর', 'কালার', 'টা', 'খুবই', 'সুন্দর']
    Truncating punctuation: ['শাড়ীর', 'কালার', 'টা', 'খুবই', 'সুন্দর']
    Truncating StopWords: ['শাড়ীর', 'কালার', 'টা', 'খুবই', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর কালেকশন
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'কালেকশন']
    Truncating punctuation: ['খুব', 'সুন্দর', 'কালেকশন']
    Truncating StopWords: ['সুন্দর', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  বাহ কি চমৎকার
    Afert Tokenizing:  ['বাহ', 'কি', 'চমৎকার']
    Truncating punctuation: ['বাহ', 'কি', 'চমৎকার']
    Truncating StopWords: ['বাহ', 'চমৎকার']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ
    Afert Tokenizing:  ['আলহামদুলিল্লাহ']
    Truncating punctuation: ['আলহামদুলিল্লাহ']
    Truncating StopWords: ['আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুণ প্যাকেজিং
    Afert Tokenizing:  ['দারুণ', 'প্যাকেজিং']
    Truncating punctuation: ['দারুণ', 'প্যাকেজিং']
    Truncating StopWords: ['দারুণ', 'প্যাকেজিং']
    ***************************************************************************************
    Label:  1
    Sentence:  বিউটিফুল
    Afert Tokenizing:  ['বিউটিফুল']
    Truncating punctuation: ['বিউটিফুল']
    Truncating StopWords: ['বিউটিফুল']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশাআল্লাহ
    Afert Tokenizing:  ['মাশাআল্লাহ']
    Truncating punctuation: ['মাশাআল্লাহ']
    Truncating StopWords: ['মাশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশাআল্লাহ
    Afert Tokenizing:  ['মাশাআল্লাহ']
    Truncating punctuation: ['মাশাআল্লাহ']
    Truncating StopWords: ['মাশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও। প্রাইজ প্লিজ
    Afert Tokenizing:  ['ওয়াও', '।', 'প্রাইজ', 'প্লিজ']
    Truncating punctuation: ['ওয়াও', 'প্রাইজ', 'প্লিজ']
    Truncating StopWords: ['ওয়াও', 'প্রাইজ', 'প্লিজ']
    ***************************************************************************************
    Label:  1
    Sentence:  কি সুন্দর প্যাকেজ!!
    Afert Tokenizing:  ['কি', 'সুন্দর', 'প্যাকেজ!', '!']
    Truncating punctuation: ['কি', 'সুন্দর', 'প্যাকেজ!']
    Truncating StopWords: ['সুন্দর', 'প্যাকেজ!']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো সার্ভিস পেয়েছি, ধন্যবাদ
    Afert Tokenizing:  ['ভালো', 'সার্ভিস', 'পেয়েছি', ',', 'ধন্যবাদ']
    Truncating punctuation: ['ভালো', 'সার্ভিস', 'পেয়েছি', 'ধন্যবাদ']
    Truncating StopWords: ['ভালো', 'সার্ভিস', 'পেয়েছি', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশাআল্লাহ এগিয়ে যাও
    Afert Tokenizing:  ['মাশাআল্লাহ', 'এগিয়ে', 'যাও']
    Truncating punctuation: ['মাশাআল্লাহ', 'এগিয়ে', 'যাও']
    Truncating StopWords: ['মাশাআল্লাহ', 'এগিয়ে', 'যাও']
    ***************************************************************************************
    Label:  0
    Sentence:  নেট প্রাইজ থেকে আরো ১০০ টাকা বেশি চেয়েছেন
    Afert Tokenizing:  ['নেট', 'প্রাইজ', 'থেকে', 'আরো', '১০০', 'টাকা', 'বেশি', 'চেয়েছেন']
    Truncating punctuation: ['নেট', 'প্রাইজ', 'থেকে', 'আরো', '১০০', 'টাকা', 'বেশি', 'চেয়েছেন']
    Truncating StopWords: ['নেট', 'প্রাইজ', 'আরো', '১০০', 'টাকা', 'বেশি', 'চেয়েছেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আমের কোয়ালিটি খুব ভালো ছিলো কালারো সুন্দর ছিলো
    Afert Tokenizing:  ['আমের', 'কোয়ালিটি', 'খুব', 'ভালো', 'ছিলো', 'কালারো', 'সুন্দর', 'ছিলো']
    Truncating punctuation: ['আমের', 'কোয়ালিটি', 'খুব', 'ভালো', 'ছিলো', 'কালারো', 'সুন্দর', 'ছিলো']
    Truncating StopWords: ['আমের', 'কোয়ালিটি', 'ভালো', 'ছিলো', 'কালারো', 'সুন্দর', 'ছিলো']
    ***************************************************************************************
    Label:  0
    Sentence:  দুইবার আম পাঠাইছে একবারও ভালো পরলো না
    Afert Tokenizing:  ['দুইবার', 'আম', 'পাঠাইছে', 'একবারও', 'ভালো', 'পরলো', 'না']
    Truncating punctuation: ['দুইবার', 'আম', 'পাঠাইছে', 'একবারও', 'ভালো', 'পরলো', 'না']
    Truncating StopWords: ['দুইবার', 'আম', 'পাঠাইছে', 'একবারও', 'ভালো', 'পরলো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এসব বাটপারদের নামে মামলা করা উচিত।
    Afert Tokenizing:  ['এসব', 'বাটপারদের', 'নামে', 'মামলা', 'করা', 'উচিত', '।']
    Truncating punctuation: ['এসব', 'বাটপারদের', 'নামে', 'মামলা', 'করা', 'উচিত']
    Truncating StopWords: ['এসব', 'বাটপারদের', 'নামে', 'মামলা']
    ***************************************************************************************
    Label:  0
    Sentence:  এসব বাটপারদের নামে মামলা করা উচিত।
    Afert Tokenizing:  ['এসব', 'বাটপারদের', 'নামে', 'মামলা', 'করা', 'উচিত', '।']
    Truncating punctuation: ['এসব', 'বাটপারদের', 'নামে', 'মামলা', 'করা', 'উচিত']
    Truncating StopWords: ['এসব', 'বাটপারদের', 'নামে', 'মামলা']
    ***************************************************************************************
    Label:  1
    Sentence:  অর্ডার কনফার্ম হওয়ার পর থেকে প্রোডাক্ট হাতে পাওয়া পর্যন্ত পুরোটাই বেশ ভালো একটা অভিজ্ঞতা ছিল
    Afert Tokenizing:  ['অর্ডার', 'কনফার্ম', 'হওয়ার', 'পর', 'থেকে', 'প্রোডাক্ট', 'হাতে', 'পাওয়া', 'পর্যন্ত', 'পুরোটাই', 'বেশ', 'ভালো', 'একটা', 'অভিজ্ঞতা', 'ছিল']
    Truncating punctuation: ['অর্ডার', 'কনফার্ম', 'হওয়ার', 'পর', 'থেকে', 'প্রোডাক্ট', 'হাতে', 'পাওয়া', 'পর্যন্ত', 'পুরোটাই', 'বেশ', 'ভালো', 'একটা', 'অভিজ্ঞতা', 'ছিল']
    Truncating StopWords: ['অর্ডার', 'কনফার্ম', 'হওয়ার', 'প্রোডাক্ট', 'হাতে', 'পাওয়া', 'পুরোটাই', 'ভালো', 'একটা', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রাইস হিসেবে কোয়ালিটি নিয়ে আমি স্যাটিসফাইড
    Afert Tokenizing:  ['প্রাইস', 'হিসেবে', 'কোয়ালিটি', 'নিয়ে', 'আমি', 'স্যাটিসফাইড']
    Truncating punctuation: ['প্রাইস', 'হিসেবে', 'কোয়ালিটি', 'নিয়ে', 'আমি', 'স্যাটিসফাইড']
    Truncating StopWords: ['প্রাইস', 'হিসেবে', 'কোয়ালিটি', 'স্যাটিসফাইড']
    ***************************************************************************************
    Label:  1
    Sentence:  শপিং এক্সপেরিয়েন্স ছিল পুরোপুরি সন্তোষজনক
    Afert Tokenizing:  ['শপিং', 'এক্সপেরিয়েন্স', 'ছিল', 'পুরোপুরি', 'সন্তোষজনক']
    Truncating punctuation: ['শপিং', 'এক্সপেরিয়েন্স', 'ছিল', 'পুরোপুরি', 'সন্তোষজনক']
    Truncating StopWords: ['শপিং', 'এক্সপেরিয়েন্স', 'পুরোপুরি', 'সন্তোষজনক']
    ***************************************************************************************
    Label:  0
    Sentence:  এতো কম দামে ল্যাপটপ টেবিল পাবার কথা না।
    Afert Tokenizing:  ['এতো', 'কম', 'দামে', 'ল্যাপটপ', 'টেবিল', 'পাবার', 'কথা', 'না', '।']
    Truncating punctuation: ['এতো', 'কম', 'দামে', 'ল্যাপটপ', 'টেবিল', 'পাবার', 'কথা', 'না']
    Truncating StopWords: ['এতো', 'কম', 'দামে', 'ল্যাপটপ', 'টেবিল', 'পাবার', 'কথা', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ আলিশা মাঠ অনেকদিন পর অফার দিলেন
    Afert Tokenizing:  ['ধন্যবাদ', 'আলিশা', 'মাঠ', 'অনেকদিন', 'পর', 'অফার', 'দিলেন']
    Truncating punctuation: ['ধন্যবাদ', 'আলিশা', 'মাঠ', 'অনেকদিন', 'পর', 'অফার', 'দিলেন']
    Truncating StopWords: ['ধন্যবাদ', 'আলিশা', 'মাঠ', 'অনেকদিন', 'অফার']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ , এখন দিনে দিনে পাচ্ছি।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '', ',', 'এখন', 'দিনে', 'দিনে', 'পাচ্ছি', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', '', 'এখন', 'দিনে', 'দিনে', 'পাচ্ছি']
    Truncating StopWords: ['আলহামদুলিল্লাহ', '', 'দিনে', 'দিনে', 'পাচ্ছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ আজ আলেশা মার্ট এর বাইক ভেলিভারি পাইলাম।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'আজ', 'আলেশা', 'মার্ট', 'এর', 'বাইক', 'ভেলিভারি', 'পাইলাম', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আজ', 'আলেশা', 'মার্ট', 'এর', 'বাইক', 'ভেলিভারি', 'পাইলাম']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'আলেশা', 'মার্ট', 'বাইক', 'ভেলিভারি', 'পাইলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  সাবাস আলেশা মাট
    Afert Tokenizing:  ['সাবাস', 'আলেশা', 'মাট']
    Truncating punctuation: ['সাবাস', 'আলেশা', 'মাট']
    Truncating StopWords: ['সাবাস', 'আলেশা', 'মাট']
    ***************************************************************************************
    Label:  1
    Sentence:  অসংখ্য ধন্যবাদ। পণ্য টি পাওয়ার জন্য
    Afert Tokenizing:  ['অসংখ্য', 'ধন্যবাদ', '।', 'পণ্য', 'টি', 'পাওয়ার', 'জন্য']
    Truncating punctuation: ['অসংখ্য', 'ধন্যবাদ', 'পণ্য', 'টি', 'পাওয়ার', 'জন্য']
    Truncating StopWords: ['অসংখ্য', 'ধন্যবাদ', 'পণ্য', 'পাওয়ার']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার টা করেছি অনেক দিন হল এখনো প্রসেসিং অবস্থায় আছে একটু তাড়াতাড়ি দিলে উপকার হতো।
    Afert Tokenizing:  ['অর্ডার', 'টা', 'করেছি', 'অনেক', 'দিন', 'হল', 'এখনো', 'প্রসেসিং', 'অবস্থায়', 'আছে', 'একটু', 'তাড়াতাড়ি', 'দিলে', 'উপকার', 'হতো', '।']
    Truncating punctuation: ['অর্ডার', 'টা', 'করেছি', 'অনেক', 'দিন', 'হল', 'এখনো', 'প্রসেসিং', 'অবস্থায়', 'আছে', 'একটু', 'তাড়াতাড়ি', 'দিলে', 'উপকার', 'হতো']
    Truncating StopWords: ['অর্ডার', 'টা', 'করেছি', 'এখনো', 'প্রসেসিং', 'অবস্থায়', 'একটু', 'তাড়াতাড়ি', 'দিলে', 'উপকার', 'হতো']
    ***************************************************************************************
    Label:  1
    Sentence:  কখনো আলেশা মার্ট থেকে কিছু কিনে প্রোতারিতো হই নাই। এই জন্য ধন্যবাদ জানাই।
    Afert Tokenizing:  ['কখনো', 'আলেশা', 'মার্ট', 'থেকে', 'কিছু', 'কিনে', 'প্রোতারিতো', 'হই', 'নাই', '।', 'এই', 'জন্য', 'ধন্যবাদ', 'জানাই', '।']
    Truncating punctuation: ['কখনো', 'আলেশা', 'মার্ট', 'থেকে', 'কিছু', 'কিনে', 'প্রোতারিতো', 'হই', 'নাই', 'এই', 'জন্য', 'ধন্যবাদ', 'জানাই']
    Truncating StopWords: ['কখনো', 'আলেশা', 'মার্ট', 'কিনে', 'প্রোতারিতো', 'হই', 'নাই', 'ধন্যবাদ', 'জানাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার বাইক কই ৩ মাস হয়ে গেলো এখন ও খবর নাই
    Afert Tokenizing:  ['আমার', 'বাইক', 'কই', '৩', 'মাস', 'হয়ে', 'গেলো', 'এখন', 'ও', 'খবর', 'নাই']
    Truncating punctuation: ['আমার', 'বাইক', 'কই', '৩', 'মাস', 'হয়ে', 'গেলো', 'এখন', 'ও', 'খবর', 'নাই']
    Truncating StopWords: ['বাইক', 'কই', '৩', 'মাস', 'হয়ে', 'গেলো', 'খবর', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  আরো ভালো কিছু ফোন দেওয়ার চেষ্টা করেন।সাধু বাদ
    Afert Tokenizing:  ['আরো', 'ভালো', 'কিছু', 'ফোন', 'দেওয়ার', 'চেষ্টা', 'করেন।সাধু', 'বাদ']
    Truncating punctuation: ['আরো', 'ভালো', 'কিছু', 'ফোন', 'দেওয়ার', 'চেষ্টা', 'করেন।সাধু', 'বাদ']
    Truncating StopWords: ['আরো', 'ভালো', 'ফোন', 'চেষ্টা', 'করেন।সাধু', 'বাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  পন্য পেয়েছি কিন্তু উপহার হিসেবে টি-শার্ট পাইনি এইটা কোন কথা
    Afert Tokenizing:  ['পন্য', 'পেয়েছি', 'কিন্তু', 'উপহার', 'হিসেবে', 'টি-শার্ট', 'পাইনি', 'এইটা', 'কোন', 'কথা']
    Truncating punctuation: ['পন্য', 'পেয়েছি', 'কিন্তু', 'উপহার', 'হিসেবে', 'টি-শার্ট', 'পাইনি', 'এইটা', 'কোন', 'কথা']
    Truncating StopWords: ['পন্য', 'পেয়েছি', 'উপহার', 'হিসেবে', 'টি-শার্ট', 'পাইনি', 'এইটা', 'কথা']
    ***************************************************************************************
    Label:  1
    Sentence:  বাংলাদেশের একমাত্র বেস্ট ই-কমার্স সাইট ''আলেশা মার্ট
    Afert Tokenizing:  ['বাংলাদেশের', 'একমাত্র', 'বেস্ট', 'ই-কমার্স', 'সাইট', "'আলেশা", "'", 'মার্ট']
    Truncating punctuation: ['বাংলাদেশের', 'একমাত্র', 'বেস্ট', 'ই-কমার্স', 'সাইট', "'আলেশা", 'মার্ট']
    Truncating StopWords: ['বাংলাদেশের', 'একমাত্র', 'বেস্ট', 'ই-কমার্স', 'সাইট', "'আলেশা", 'মার্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  দেশী পন্য কিনে হবেন ধন্য
    Afert Tokenizing:  ['দেশী', 'পন্য', 'কিনে', 'হবেন', 'ধন্য']
    Truncating punctuation: ['দেশী', 'পন্য', 'কিনে', 'হবেন', 'ধন্য']
    Truncating StopWords: ['দেশী', 'পন্য', 'কিনে', 'ধন্য']
    ***************************************************************************************
    Label:  0
    Sentence:  বেশি খাইতে জায়েন না
    Afert Tokenizing:  ['বেশি', 'খাইতে', 'জায়েন', 'না']
    Truncating punctuation: ['বেশি', 'খাইতে', 'জায়েন', 'না']
    Truncating StopWords: ['বেশি', 'খাইতে', 'জায়েন', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  মনে হয় এ বছর আর পাব না
    Afert Tokenizing:  ['মনে', 'হয়', 'এ', 'বছর', 'আর', 'পাব', 'না']
    Truncating punctuation: ['মনে', 'হয়', 'এ', 'বছর', 'আর', 'পাব', 'না']
    Truncating StopWords: ['বছর', 'পাব', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপারের দল বাইকের লিস্ট দে
    Afert Tokenizing:  ['বাটপারের', 'দল', 'বাইকের', 'লিস্ট', 'দে']
    Truncating punctuation: ['বাটপারের', 'দল', 'বাইকের', 'লিস্ট', 'দে']
    Truncating StopWords: ['বাটপারের', 'দল', 'বাইকের', 'লিস্ট', 'দে']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশ এর চেয়ে বাজে আন লাইন একটিও নাই।
    Afert Tokenizing:  ['বাংলাদেশ', 'এর', 'চেয়ে', 'বাজে', 'আন', 'লাইন', 'একটিও', 'নাই', '।']
    Truncating punctuation: ['বাংলাদেশ', 'এর', 'চেয়ে', 'বাজে', 'আন', 'লাইন', 'একটিও', 'নাই']
    Truncating StopWords: ['বাংলাদেশ', 'চেয়ে', 'বাজে', 'আন', 'লাইন', 'একটিও', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  দেখে ভালো লাগলো
    Afert Tokenizing:  ['দেখে', 'ভালো', 'লাগলো']
    Truncating punctuation: ['দেখে', 'ভালো', 'লাগলো']
    Truncating StopWords: ['ভালো', 'লাগলো']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালোই বাটপার
    Afert Tokenizing:  ['ভালোই', 'বাটপার']
    Truncating punctuation: ['ভালোই', 'বাটপার']
    Truncating StopWords: ['ভালোই', 'বাটপার']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাই সাইকেল এখান থেকে নিতে পারেন
    Afert Tokenizing:  ['ভাই', 'সাইকেল', 'এখান', 'থেকে', 'নিতে', 'পারেন']
    Truncating punctuation: ['ভাই', 'সাইকেল', 'এখান', 'থেকে', 'নিতে', 'পারেন']
    Truncating StopWords: ['ভাই', 'সাইকেল', 'এখান']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের বিশ্বাস করা যায়
    Afert Tokenizing:  ['আপনাদের', 'বিশ্বাস', 'করা', 'যায়']
    Truncating punctuation: ['আপনাদের', 'বিশ্বাস', 'করা', 'যায়']
    Truncating StopWords: ['আপনাদের', 'বিশ্বাস', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশি
    Afert Tokenizing:  ['দাম', 'বেশি']
    Truncating punctuation: ['দাম', 'বেশি']
    Truncating StopWords: ['দাম', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনা‌দের এ কার্যক্রম আ‌রো এ‌গি‌য়ে যাক, এ প্রত‌্যাশা ক‌রি
    Afert Tokenizing:  ['আপনা\u200cদের', 'এ', 'কার্যক্রম', 'আ\u200cরো', 'এ\u200cগি\u200cয়ে', 'যাক', ',', 'এ', 'প্রত\u200c্যাশা', 'ক\u200cরি']
    Truncating punctuation: ['আপনা\u200cদের', 'এ', 'কার্যক্রম', 'আ\u200cরো', 'এ\u200cগি\u200cয়ে', 'যাক', 'এ', 'প্রত\u200c্যাশা', 'ক\u200cরি']
    Truncating StopWords: ['আপনা\u200cদের', 'কার্যক্রম', 'আ\u200cরো', 'এ\u200cগি\u200cয়ে', 'যাক', 'প্রত\u200c্যাশা', 'ক\u200cরি']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি চার্জ একটু কমানো দরকার।
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'একটু', 'কমানো', 'দরকার', '।']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'একটু', 'কমানো', 'দরকার']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'একটু', 'কমানো', 'দরকার']
    ***************************************************************************************
    Label:  1
    Sentence:  আলিশা মার্ট দেশ সেরা হবে ইনশাআল্লাহ একদিন
    Afert Tokenizing:  ['আলিশা', 'মার্ট', 'দেশ', 'সেরা', 'হবে', 'ইনশাআল্লাহ', 'একদিন']
    Truncating punctuation: ['আলিশা', 'মার্ট', 'দেশ', 'সেরা', 'হবে', 'ইনশাআল্লাহ', 'একদিন']
    Truncating StopWords: ['আলিশা', 'মার্ট', 'দেশ', 'সেরা', 'ইনশাআল্লাহ', 'একদিন']
    ***************************************************************************************
    Label:  0
    Sentence:  এরা কি আসলে প্রোডাক্ট দেয়???
    Afert Tokenizing:  ['এরা', 'কি', 'আসলে', 'প্রোডাক্ট', 'দেয়??', '?']
    Truncating punctuation: ['এরা', 'কি', 'আসলে', 'প্রোডাক্ট', 'দেয়??']
    Truncating StopWords: ['আসলে', 'প্রোডাক্ট', 'দেয়??']
    ***************************************************************************************
    Label:  0
    Sentence:  পুরাই চোর সালারা
    Afert Tokenizing:  ['পুরাই', 'চোর', 'সালারা']
    Truncating punctuation: ['পুরাই', 'চোর', 'সালারা']
    Truncating StopWords: ['পুরাই', 'চোর', 'সালারা']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ আলেশা_মার্ট আমার গাড়ী টি কনফার্ম করার জন্য
    Afert Tokenizing:  ['ধন্যবাদ', 'আলেশা_মার্ট', 'আমার', 'গাড়ী', 'টি', 'কনফার্ম', 'করার', 'জন্য']
    Truncating punctuation: ['ধন্যবাদ', 'আলেশা_মার্ট', 'আমার', 'গাড়ী', 'টি', 'কনফার্ম', 'করার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'আলেশা_মার্ট', 'গাড়ী', 'কনফার্ম']
    ***************************************************************************************
    Label:  0
    Sentence:  কেউ এই ফাঁদে পা বাড়াবেন না।
    Afert Tokenizing:  ['কেউ', 'এই', 'ফাঁদে', 'পা', 'বাড়াবেন', 'না', '।']
    Truncating punctuation: ['কেউ', 'এই', 'ফাঁদে', 'পা', 'বাড়াবেন', 'না']
    Truncating StopWords: ['ফাঁদে', 'পা', 'বাড়াবেন', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এদের থেকে কি কেউ কিছু পেয়েছেন আজ পর্যন্ত?
    Afert Tokenizing:  ['এদের', 'থেকে', 'কি', 'কেউ', 'কিছু', 'পেয়েছেন', 'আজ', 'পর্যন্ত', '?']
    Truncating punctuation: ['এদের', 'থেকে', 'কি', 'কেউ', 'কিছু', 'পেয়েছেন', 'আজ', 'পর্যন্ত']
    Truncating StopWords: ['পেয়েছেন']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবি খারাপ অবস্থা এদের
    Afert Tokenizing:  ['খুবি', 'খারাপ', 'অবস্থা', 'এদের']
    Truncating punctuation: ['খুবি', 'খারাপ', 'অবস্থা', 'এদের']
    Truncating StopWords: ['খুবি', 'খারাপ', 'অবস্থা']
    ***************************************************************************************
    Label:  1
    Sentence:   আশা করি ঠিক সময়ের মধ্যেই পাবো ।
    Afert Tokenizing:  ['আশা', 'করি', 'ঠিক', 'সময়ের', 'মধ্যেই', 'পাবো', '', '।']
    Truncating punctuation: ['আশা', 'করি', 'ঠিক', 'সময়ের', 'মধ্যেই', 'পাবো', '']
    Truncating StopWords: ['আশা', 'ঠিক', 'সময়ের', 'পাবো', '']
    ***************************************************************************************
    Label:  1
    Sentence:  স্বপ্ন পুরনের সারথি
    Afert Tokenizing:  ['স্বপ্ন', 'পুরনের', 'সারথি']
    Truncating punctuation: ['স্বপ্ন', 'পুরনের', 'সারথি']
    Truncating StopWords: ['স্বপ্ন', 'পুরনের', 'সারথি']
    ***************************************************************************************
    Label:  0
    Sentence:  টিভিতে বিজ্ঞাপন দিয়ে টাকা নষ্ট না করে অফার দেন
    Afert Tokenizing:  ['টিভিতে', 'বিজ্ঞাপন', 'দিয়ে', 'টাকা', 'নষ্ট', 'না', 'করে', 'অফার', 'দেন']
    Truncating punctuation: ['টিভিতে', 'বিজ্ঞাপন', 'দিয়ে', 'টাকা', 'নষ্ট', 'না', 'করে', 'অফার', 'দেন']
    Truncating StopWords: ['টিভিতে', 'বিজ্ঞাপন', 'দিয়ে', 'টাকা', 'নষ্ট', 'না', 'অফার']
    ***************************************************************************************
    Label:  0
    Sentence:  যে রকম দাম রাখছেন তাতে আপনারা আগামী ১ যুগেও উন্নতি করতে পারবেন না!!!
    Afert Tokenizing:  ['যে', 'রকম', 'দাম', 'রাখছেন', 'তাতে', 'আপনারা', 'আগামী', '১', 'যুগেও', 'উন্নতি', 'করতে', 'পারবেন', 'না!!', '!']
    Truncating punctuation: ['যে', 'রকম', 'দাম', 'রাখছেন', 'তাতে', 'আপনারা', 'আগামী', '১', 'যুগেও', 'উন্নতি', 'করতে', 'পারবেন', 'না!!']
    Truncating StopWords: ['দাম', 'রাখছেন', 'আপনারা', '১', 'যুগেও', 'উন্নতি', 'পারবেন', 'না!!']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের অ্যাপস আরো উন্নত করেন।
    Afert Tokenizing:  ['আপনাদের', 'অ্যাপস', 'আরো', 'উন্নত', 'করেন', '।']
    Truncating punctuation: ['আপনাদের', 'অ্যাপস', 'আরো', 'উন্নত', 'করেন']
    Truncating StopWords: ['আপনাদের', 'অ্যাপস', 'আরো', 'উন্নত']
    ***************************************************************************************
    Label:  0
    Sentence:  ইভ্যালির চাচাতো ভাই না হলেই হয়।
    Afert Tokenizing:  ['ইভ্যালির', 'চাচাতো', 'ভাই', 'না', 'হলেই', 'হয়', '।']
    Truncating punctuation: ['ইভ্যালির', 'চাচাতো', 'ভাই', 'না', 'হলেই', 'হয়']
    Truncating StopWords: ['ইভ্যালির', 'চাচাতো', 'ভাই', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ধান্ধাবাজির কিছুই বুঝলাম না
    Afert Tokenizing:  ['ধান্ধাবাজির', 'কিছুই', 'বুঝলাম', 'না']
    Truncating punctuation: ['ধান্ধাবাজির', 'কিছুই', 'বুঝলাম', 'না']
    Truncating StopWords: ['ধান্ধাবাজির', 'বুঝলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের সব থেকে খারাপ দিক হলো কেউ কিছু জানতে চাইতে উত্তর দেন না।
    Afert Tokenizing:  ['আপনাদের', 'সব', 'থেকে', 'খারাপ', 'দিক', 'হলো', 'কেউ', 'কিছু', 'জানতে', 'চাইতে', 'উত্তর', 'দেন', 'না', '।']
    Truncating punctuation: ['আপনাদের', 'সব', 'থেকে', 'খারাপ', 'দিক', 'হলো', 'কেউ', 'কিছু', 'জানতে', 'চাইতে', 'উত্তর', 'দেন', 'না']
    Truncating StopWords: ['আপনাদের', 'খারাপ', 'দিক', 'চাইতে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি নিবো এর আগেও এই সিরাম ইউজ করেছি
    Afert Tokenizing:  ['আমি', 'নিবো', 'এর', 'আগেও', 'এই', 'সিরাম', 'ইউজ', 'করেছি']
    Truncating punctuation: ['আমি', 'নিবো', 'এর', 'আগেও', 'এই', 'সিরাম', 'ইউজ', 'করেছি']
    Truncating StopWords: ['নিবো', 'আগেও', 'সিরাম', 'ইউজ', 'করেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  এবারের প্যাকেজিং টা অনেক বাজে ছিলো।
    Afert Tokenizing:  ['এবারের', 'প্যাকেজিং', 'টা', 'অনেক', 'বাজে', 'ছিলো', '।']
    Truncating punctuation: ['এবারের', 'প্যাকেজিং', 'টা', 'অনেক', 'বাজে', 'ছিলো']
    Truncating StopWords: ['এবারের', 'প্যাকেজিং', 'টা', 'বাজে', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ সব প্রডাক্ট ভালো ছিলো
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'সব', 'প্রডাক্ট', 'ভালো', 'ছিলো']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'সব', 'প্রডাক্ট', 'ভালো', 'ছিলো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'প্রডাক্ট', 'ভালো', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের সব কিছু ই ভাল।আমি যাদের কে আপনাদে পন্য নিতে বলেছি সবাই নিয়ে প্রশংসা করতে ছে আপনাদের
    Afert Tokenizing:  ['আপনাদের', 'সব', 'কিছু', 'ই', 'ভাল।আমি', 'যাদের', 'কে', 'আপনাদে', 'পন্য', 'নিতে', 'বলেছি', 'সবাই', 'নিয়ে', 'প্রশংসা', 'করতে', 'ছে', 'আপনাদের']
    Truncating punctuation: ['আপনাদের', 'সব', 'কিছু', 'ই', 'ভাল।আমি', 'যাদের', 'কে', 'আপনাদে', 'পন্য', 'নিতে', 'বলেছি', 'সবাই', 'নিয়ে', 'প্রশংসা', 'করতে', 'ছে', 'আপনাদের']
    Truncating StopWords: ['আপনাদের', 'ভাল।আমি', 'আপনাদে', 'পন্য', 'বলেছি', 'সবাই', 'প্রশংসা', 'ছে', 'আপনাদের']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ ৩দিন যাবত ব্যবহার করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম
    Afert Tokenizing:  ['আজ', '৩দিন', 'যাবত', 'ব্যবহার', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    Truncating punctuation: ['আজ', '৩দিন', 'যাবত', 'ব্যবহার', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    Truncating StopWords: ['৩দিন', 'যাবত', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি একে একে 4 টা নিয়েছি।খুবই ভালো প্রোডাক্ট
    Afert Tokenizing:  ['আমি', 'একে', 'একে', '4', 'টা', 'নিয়েছি।খুবই', 'ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['আমি', 'একে', 'একে', '4', 'টা', 'নিয়েছি।খুবই', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['4', 'টা', 'নিয়েছি।খুবই', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  নিতে চাই
    Afert Tokenizing:  ['নিতে', 'চাই']
    Truncating punctuation: ['নিতে', 'চাই']
    Truncating StopWords: ['চাই']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো প্রোডাক্ট
    Afert Tokenizing:  ['অনেক', 'ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['অনেক', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট ভালো কিন্তু দাম বেশি
    Afert Tokenizing:  ['প্রোডাক্ট', 'ভালো', 'কিন্তু', 'দাম', 'বেশি']
    Truncating punctuation: ['প্রোডাক্ট', 'ভালো', 'কিন্তু', 'দাম', 'বেশি']
    Truncating StopWords: ['প্রোডাক্ট', 'ভালো', 'দাম', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো কোয়ালিটি
    Afert Tokenizing:  ['অনেক', 'ভালো', 'কোয়ালিটি']
    Truncating punctuation: ['অনেক', 'ভালো', 'কোয়ালিটি']
    Truncating StopWords: ['ভালো', 'কোয়ালিটি']
    ***************************************************************************************
    Label:  1
    Sentence:  মাল হাতে পাইছি খুব ভালো
    Afert Tokenizing:  ['মাল', 'হাতে', 'পাইছি', 'খুব', 'ভালো']
    Truncating punctuation: ['মাল', 'হাতে', 'পাইছি', 'খুব', 'ভালো']
    Truncating StopWords: ['মাল', 'হাতে', 'পাইছি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবই ভাল কোয়ালিটি । পড়তেও আলাদা মজা পাওয়া যাইয়া
    Afert Tokenizing:  ['খুবই', 'ভাল', 'কোয়ালিটি', '', '।', 'পড়তেও', 'আলাদা', 'মজা', 'পাওয়া', 'যাইয়া']
    Truncating punctuation: ['খুবই', 'ভাল', 'কোয়ালিটি', '', 'পড়তেও', 'আলাদা', 'মজা', 'পাওয়া', 'যাইয়া']
    Truncating StopWords: ['খুবই', 'ভাল', 'কোয়ালিটি', '', 'পড়তেও', 'আলাদা', 'মজা', 'পাওয়া', 'যাইয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম গুলো বেশী মনে হচ্ছে
    Afert Tokenizing:  ['দাম', 'গুলো', 'বেশী', 'মনে', 'হচ্ছে']
    Truncating punctuation: ['দাম', 'গুলো', 'বেশী', 'মনে', 'হচ্ছে']
    Truncating StopWords: ['দাম', 'গুলো', 'বেশী']
    ***************************************************************************************
    Label:  1
    Sentence:  যেমনটা চেয়েছিলাম তেমনটা পেয়েছি
    Afert Tokenizing:  ['যেমনটা', 'চেয়েছিলাম', 'তেমনটা', 'পেয়েছি']
    Truncating punctuation: ['যেমনটা', 'চেয়েছিলাম', 'তেমনটা', 'পেয়েছি']
    Truncating StopWords: ['যেমনটা', 'চেয়েছিলাম', 'তেমনটা', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রোডাক্ট ভাল।
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'ভাল', '।']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'ভাল']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের এই মাক্স গুলো অনেক দিন যাবত ব্যবহার করছি,খুবই ভাল কোয়ালিটি,পড়তেও আরাম
    Afert Tokenizing:  ['আপনাদের', 'এই', 'মাক্স', 'গুলো', 'অনেক', 'দিন', 'যাবত', 'ব্যবহার', 'করছি,খুবই', 'ভাল', 'কোয়ালিটি,পড়তেও', 'আরাম']
    Truncating punctuation: ['আপনাদের', 'এই', 'মাক্স', 'গুলো', 'অনেক', 'দিন', 'যাবত', 'ব্যবহার', 'করছি,খুবই', 'ভাল', 'কোয়ালিটি,পড়তেও', 'আরাম']
    Truncating StopWords: ['আপনাদের', 'মাক্স', 'গুলো', 'যাবত', 'করছি,খুবই', 'ভাল', 'কোয়ালিটি,পড়তেও', 'আরাম']
    ***************************************************************************************
    Label:  1
    Sentence:  ১০% এন্ড ৫% দুইটাই পাইছি আলহামদুলিল্লাহ
    Afert Tokenizing:  ['১০%', 'এন্ড', '৫%', 'দুইটাই', 'পাইছি', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['১০%', 'এন্ড', '৫%', 'দুইটাই', 'পাইছি', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['১০%', 'এন্ড', '৫%', 'দুইটাই', 'পাইছি', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমিও বিকাশে পেমেন্ট করছিলাম কিন্তু ক্যাশব্যাক পাইনি
    Afert Tokenizing:  ['ভাই', 'আমিও', 'বিকাশে', 'পেমেন্ট', 'করছিলাম', 'কিন্তু', 'ক্যাশব্যাক', 'পাইনি']
    Truncating punctuation: ['ভাই', 'আমিও', 'বিকাশে', 'পেমেন্ট', 'করছিলাম', 'কিন্তু', 'ক্যাশব্যাক', 'পাইনি']
    Truncating StopWords: ['ভাই', 'আমিও', 'বিকাশে', 'পেমেন্ট', 'করছিলাম', 'ক্যাশব্যাক', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম জানার পর মনটা খুব খারাপ হয়ে গেলো।
    Afert Tokenizing:  ['দাম', 'জানার', 'পর', 'মনটা', 'খুব', 'খারাপ', 'হয়ে', 'গেলো', '।']
    Truncating punctuation: ['দাম', 'জানার', 'পর', 'মনটা', 'খুব', 'খারাপ', 'হয়ে', 'গেলো']
    Truncating StopWords: ['দাম', 'জানার', 'মনটা', 'খারাপ', 'হয়ে', 'গেলো']
    ***************************************************************************************
    Label:  0
    Sentence:  Quality অনুযায়ী দাম অনেক বেশী
    Afert Tokenizing:  ['Quality', 'অনুযায়ী', 'দাম', 'অনেক', 'বেশী']
    Truncating punctuation: ['Quality', 'অনুযায়ী', 'দাম', 'অনেক', 'বেশী']
    Truncating StopWords: ['Quality', 'অনুযায়ী', 'দাম', 'বেশী']
    ***************************************************************************************
    Label:  0
    Sentence:  কোয়ালিটি যা দেখান তা তো দেন না।
    Afert Tokenizing:  ['কোয়ালিটি', 'যা', 'দেখান', 'তা', 'তো', 'দেন', 'না', '।']
    Truncating punctuation: ['কোয়ালিটি', 'যা', 'দেখান', 'তা', 'তো', 'দেন', 'না']
    Truncating StopWords: ['কোয়ালিটি', 'দেখান', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রতিটি প্রোডাক্টের গুনগত মান খুবিই ভালো!
    Afert Tokenizing:  ['প্রতিটি', 'প্রোডাক্টের', 'গুনগত', 'মান', 'খুবিই', 'ভালো', '!']
    Truncating punctuation: ['প্রতিটি', 'প্রোডাক্টের', 'গুনগত', 'মান', 'খুবিই', 'ভালো']
    Truncating StopWords: ['প্রতিটি', 'প্রোডাক্টের', 'গুনগত', 'মান', 'খুবিই', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার পণ্যগুলো অনেক সুন্দর ও আরাম দায়ক
    Afert Tokenizing:  ['আপনার', 'পণ্যগুলো', 'অনেক', 'সুন্দর', 'ও', 'আরাম', 'দায়ক']
    Truncating punctuation: ['আপনার', 'পণ্যগুলো', 'অনেক', 'সুন্দর', 'ও', 'আরাম', 'দায়ক']
    Truncating StopWords: ['পণ্যগুলো', 'সুন্দর', 'আরাম', 'দায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  যথেষ্ট প্রিমিয়াম প্রোডাক্ট ।আমি আজকে আমার এবং আমার ছেলের জন্য টি-শার্ট নিয়েছি চমৎকার ফেব্রিক্স
    Afert Tokenizing:  ['যথেষ্ট', 'প্রিমিয়াম', 'প্রোডাক্ট', 'আমি', '।', 'আজকে', 'আমার', 'এবং', 'আমার', 'ছেলের', 'জন্য', 'টি-শার্ট', 'নিয়েছি', 'চমৎকার', 'ফেব্রিক্স']
    Truncating punctuation: ['যথেষ্ট', 'প্রিমিয়াম', 'প্রোডাক্ট', 'আমি', 'আজকে', 'আমার', 'এবং', 'আমার', 'ছেলের', 'জন্য', 'টি-শার্ট', 'নিয়েছি', 'চমৎকার', 'ফেব্রিক্স']
    Truncating StopWords: ['যথেষ্ট', 'প্রিমিয়াম', 'প্রোডাক্ট', 'আজকে', 'ছেলের', 'টি-শার্ট', 'নিয়েছি', 'চমৎকার', 'ফেব্রিক্স']
    ***************************************************************************************
    Label:  1
    Sentence:  সবই নিতে মন চায়..
    Afert Tokenizing:  ['সবই', 'নিতে', 'মন', 'চায়.', '.']
    Truncating punctuation: ['সবই', 'নিতে', 'মন', 'চায়.']
    Truncating StopWords: ['সবই', 'মন', 'চায়.']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম একটু বেশি হইলেও কাপড়ের মান এবং সৃজনশীলতা আপনাদের পন্যের একটি বড় গুণ।
    Afert Tokenizing:  ['দাম', 'একটু', 'বেশি', 'হইলেও', 'কাপড়ের', 'মান', 'এবং', 'সৃজনশীলতা', 'আপনাদের', 'পন্যের', 'একটি', 'বড়', 'গুণ', '।']
    Truncating punctuation: ['দাম', 'একটু', 'বেশি', 'হইলেও', 'কাপড়ের', 'মান', 'এবং', 'সৃজনশীলতা', 'আপনাদের', 'পন্যের', 'একটি', 'বড়', 'গুণ']
    Truncating StopWords: ['দাম', 'একটু', 'বেশি', 'হইলেও', 'কাপড়ের', 'মান', 'সৃজনশীলতা', 'আপনাদের', 'পন্যের', 'বড়', 'গুণ']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের পন্যের গুণমান চমৎকার। সত্যি ই আমি প্রেমে পরে গেছি,
    Afert Tokenizing:  ['আপনাদের', 'পন্যের', 'গুণমান', 'চমৎকার', '।', 'সত্যি', 'ই', 'আমি', 'প্রেমে', 'পরে', 'গেছি', ',']
    Truncating punctuation: ['আপনাদের', 'পন্যের', 'গুণমান', 'চমৎকার', 'সত্যি', 'ই', 'আমি', 'প্রেমে', 'পরে', 'গেছি']
    Truncating StopWords: ['আপনাদের', 'পন্যের', 'গুণমান', 'চমৎকার', 'সত্যি', 'প্রেমে', 'গেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাইজ টা আমার জন্য কিঞ্চিত হাই।
    Afert Tokenizing:  ['প্রাইজ', 'টা', 'আমার', 'জন্য', 'কিঞ্চিত', 'হাই', '।']
    Truncating punctuation: ['প্রাইজ', 'টা', 'আমার', 'জন্য', 'কিঞ্চিত', 'হাই']
    Truncating StopWords: ['প্রাইজ', 'টা', 'কিঞ্চিত', 'হাই']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম টা একটু কম রাখা যায় না
    Afert Tokenizing:  ['দাম', 'টা', 'একটু', 'কম', 'রাখা', 'যায়', 'না']
    Truncating punctuation: ['দাম', 'টা', 'একটু', 'কম', 'রাখা', 'যায়', 'না']
    Truncating StopWords: ['দাম', 'টা', 'একটু', 'কম', 'যায়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনারা উওর দেন না কেন ? ইনবক্স ক‌রে‌ছি , রেসপন্স নাই
    Afert Tokenizing:  ['আপনারা', 'উওর', 'দেন', 'না', 'কেন', '', '?', 'ইনবক্স', 'ক\u200cরে\u200cছি', '', ',', 'রেসপন্স', 'নাই']
    Truncating punctuation: ['আপনারা', 'উওর', 'দেন', 'না', 'কেন', '', 'ইনবক্স', 'ক\u200cরে\u200cছি', '', 'রেসপন্স', 'নাই']
    Truncating StopWords: ['আপনারা', 'উওর', 'না', '', 'ইনবক্স', 'ক\u200cরে\u200cছি', '', 'রেসপন্স', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় অসাধারণ
    Afert Tokenizing:  ['এক', 'কথায়', 'অসাধারণ']
    Truncating punctuation: ['এক', 'কথায়', 'অসাধারণ']
    Truncating StopWords: ['এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  একটি জিনিস আপনাদের খুবই ভালো সেটা হলো প্রাইস কমেন্টে অথবা স্ট্যাটাসে লিখে দেন নতুবা ইনবক্সে চেক করা লাগত।
    Afert Tokenizing:  ['একটি', 'জিনিস', 'আপনাদের', 'খুবই', 'ভালো', 'সেটা', 'হলো', 'প্রাইস', 'কমেন্টে', 'অথবা', 'স্ট্যাটাসে', 'লিখে', 'দেন', 'নতুবা', 'ইনবক্সে', 'চেক', 'করা', 'লাগত', '।']
    Truncating punctuation: ['একটি', 'জিনিস', 'আপনাদের', 'খুবই', 'ভালো', 'সেটা', 'হলো', 'প্রাইস', 'কমেন্টে', 'অথবা', 'স্ট্যাটাসে', 'লিখে', 'দেন', 'নতুবা', 'ইনবক্সে', 'চেক', 'করা', 'লাগত']
    Truncating StopWords: ['জিনিস', 'আপনাদের', 'খুবই', 'ভালো', 'প্রাইস', 'কমেন্টে', 'স্ট্যাটাসে', 'লিখে', 'নতুবা', 'ইনবক্সে', 'চেক', 'লাগত']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম কমালে আরেকটি অর্ডার করতাম
    Afert Tokenizing:  ['দাম', 'কমালে', 'আরেকটি', 'অর্ডার', 'করতাম']
    Truncating punctuation: ['দাম', 'কমালে', 'আরেকটি', 'অর্ডার', 'করতাম']
    Truncating StopWords: ['দাম', 'কমালে', 'আরেকটি', 'অর্ডার', 'করতাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রোডাক্ট গুলো অসাধারণ
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদে পন‍্য আমার অনেক ভালো লাগছে 100% কোয়ালিটি সপ্মন‍্য আমি এই পর্যন্ত সাতটা টি শার্ট আনলাম চোখবুজে বিশ্বাস করা যায় আমার আশা আপনারা সেই বিশ্বাস টুকু দরে রাখবেন।
    Afert Tokenizing:  ['আপনাদে', 'পন\u200d্য', 'আমার', 'অনেক', 'ভালো', 'লাগছে', '100%', 'কোয়ালিটি', 'সপ্মন\u200d্য', 'আমি', 'এই', 'পর্যন্ত', 'সাতটা', 'টি', 'শার্ট', 'আনলাম', 'চোখবুজে', 'বিশ্বাস', 'করা', 'যায়', 'আমার', 'আশা', 'আপনারা', 'সেই', 'বিশ্বাস', 'টুকু', 'দরে', 'রাখবেন', '।']
    Truncating punctuation: ['আপনাদে', 'পন\u200d্য', 'আমার', 'অনেক', 'ভালো', 'লাগছে', '100%', 'কোয়ালিটি', 'সপ্মন\u200d্য', 'আমি', 'এই', 'পর্যন্ত', 'সাতটা', 'টি', 'শার্ট', 'আনলাম', 'চোখবুজে', 'বিশ্বাস', 'করা', 'যায়', 'আমার', 'আশা', 'আপনারা', 'সেই', 'বিশ্বাস', 'টুকু', 'দরে', 'রাখবেন']
    Truncating StopWords: ['আপনাদে', 'পন\u200d্য', 'ভালো', 'লাগছে', '100%', 'কোয়ালিটি', 'সপ্মন\u200d্য', 'সাতটা', 'শার্ট', 'আনলাম', 'চোখবুজে', 'বিশ্বাস', 'যায়', 'আশা', 'আপনারা', 'বিশ্বাস', 'টুকু', 'দরে', 'রাখবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  ৩ পিস নিলাম, অনেক সুন্দর। ধন্যবাদ
    Afert Tokenizing:  ['৩', 'পিস', 'নিলাম', ',', 'অনেক', 'সুন্দর', '।', 'ধন্যবাদ']
    Truncating punctuation: ['৩', 'পিস', 'নিলাম', 'অনেক', 'সুন্দর', 'ধন্যবাদ']
    Truncating StopWords: ['৩', 'পিস', 'নিলাম', 'সুন্দর', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি একটা নিছি খুবই ভালো মানের প্রোডাক্ট, ইনশাল্লাহ আর একটা নেব।
    Afert Tokenizing:  ['আমি', 'একটা', 'নিছি', 'খুবই', 'ভালো', 'মানের', 'প্রোডাক্ট', ',', 'ইনশাল্লাহ', 'আর', 'একটা', 'নেব', '।']
    Truncating punctuation: ['আমি', 'একটা', 'নিছি', 'খুবই', 'ভালো', 'মানের', 'প্রোডাক্ট', 'ইনশাল্লাহ', 'আর', 'একটা', 'নেব']
    Truncating StopWords: ['একটা', 'নিছি', 'খুবই', 'ভালো', 'মানের', 'প্রোডাক্ট', 'ইনশাল্লাহ', 'একটা', 'নেব']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর। ভালো লাগছে
    Afert Tokenizing:  ['অনেক', 'সুন্দর', '।', 'ভালো', 'লাগছে']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'ভালো', 'লাগছে']
    Truncating StopWords: ['সুন্দর', 'ভালো', 'লাগছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আসলেই তাদের প্রোডাক্ট গুলা ভালো কোয়ালিটির।
    Afert Tokenizing:  ['আসলেই', 'তাদের', 'প্রোডাক্ট', 'গুলা', 'ভালো', 'কোয়ালিটির', '।']
    Truncating punctuation: ['আসলেই', 'তাদের', 'প্রোডাক্ট', 'গুলা', 'ভালো', 'কোয়ালিটির']
    Truncating StopWords: ['আসলেই', 'প্রোডাক্ট', 'গুলা', 'ভালো', 'কোয়ালিটির']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম কমান
    Afert Tokenizing:  ['দাম', 'কমান']
    Truncating punctuation: ['দাম', 'কমান']
    Truncating StopWords: ['দাম', 'কমান']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার গুলি পেলাম নাতো
    Afert Tokenizing:  ['আমার', 'গুলি', 'পেলাম', 'নাতো']
    Truncating punctuation: ['আমার', 'গুলি', 'পেলাম', 'নাতো']
    Truncating StopWords: ['পেলাম', 'নাতো']
    ***************************************************************************************
    Label:  1
    Sentence:  গতকাল সবুজ টি-শার্টটা অর্ডার দিয়ে, আজকেই পেয়ে গেলাম।
    Afert Tokenizing:  ['গতকাল', 'সবুজ', 'টি-শার্টটা', 'অর্ডার', 'দিয়ে', ',', 'আজকেই', 'পেয়ে', 'গেলাম', '।']
    Truncating punctuation: ['গতকাল', 'সবুজ', 'টি-শার্টটা', 'অর্ডার', 'দিয়ে', 'আজকেই', 'পেয়ে', 'গেলাম']
    Truncating StopWords: ['গতকাল', 'সবুজ', 'টি-শার্টটা', 'অর্ডার', 'দিয়ে', 'আজকেই', 'পেয়ে', 'গেলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইনে কেনাকাটা করা বড় বোকামি।পন্যটা মনের মতো হয় না।
    Afert Tokenizing:  ['অনলাইনে', 'কেনাকাটা', 'করা', 'বড়', 'বোকামি।পন্যটা', 'মনের', 'মতো', 'হয়', 'না', '।']
    Truncating punctuation: ['অনলাইনে', 'কেনাকাটা', 'করা', 'বড়', 'বোকামি।পন্যটা', 'মনের', 'মতো', 'হয়', 'না']
    Truncating StopWords: ['অনলাইনে', 'কেনাকাটা', 'বড়', 'বোকামি।পন্যটা', 'মনের', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি এক টা নিয়েছি অনেক ভালো পছন্দ হয়েছে । দাম টা আর একটু কম হলে মেরুন কালার টা নিতাম ।
    Afert Tokenizing:  ['আমি', 'এক', 'টা', 'নিয়েছি', 'অনেক', 'ভালো', 'পছন্দ', 'হয়েছে', '', '।', 'দাম', 'টা', 'আর', 'একটু', 'কম', 'হলে', 'মেরুন', 'কালার', 'টা', 'নিতাম', '', '।']
    Truncating punctuation: ['আমি', 'এক', 'টা', 'নিয়েছি', 'অনেক', 'ভালো', 'পছন্দ', 'হয়েছে', '', 'দাম', 'টা', 'আর', 'একটু', 'কম', 'হলে', 'মেরুন', 'কালার', 'টা', 'নিতাম', '']
    Truncating StopWords: ['এক', 'টা', 'নিয়েছি', 'ভালো', 'পছন্দ', 'হয়েছে', '', 'দাম', 'টা', 'একটু', 'কম', 'মেরুন', 'কালার', 'টা', 'নিতাম', '']
    ***************************************************************************************
    Label:  0
    Sentence:  চাইলাম ৩২ আপনারা দিয়ে দিলেন ৩৪। এখন এটা কিভাবে ব্যাবহার করবো?
    Afert Tokenizing:  ['চাইলাম', '৩২', 'আপনারা', 'দিয়ে', 'দিলেন', '৩৪', '।', 'এখন', 'এটা', 'কিভাবে', 'ব্যাবহার', 'করবো', '?']
    Truncating punctuation: ['চাইলাম', '৩২', 'আপনারা', 'দিয়ে', 'দিলেন', '৩৪', 'এখন', 'এটা', 'কিভাবে', 'ব্যাবহার', 'করবো']
    Truncating StopWords: ['চাইলাম', '৩২', 'আপনারা', 'দিয়ে', '৩৪', 'কিভাবে', 'ব্যাবহার', 'করবো']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি চট্টগ্রামে আজকে হাতে পেলাম অনেক, অনেক, ভাল যেই রকম চেয়েছি সেই রকম পেয়েছি, আলহামদুলিল্লাহ, কথা আর কাজে মিল আছে,
    Afert Tokenizing:  ['আমি', 'চট্টগ্রামে', 'আজকে', 'হাতে', 'পেলাম', 'অনেক', ',', 'অনেক', ',', 'ভাল', 'যেই', 'রকম', 'চেয়েছি', 'সেই', 'রকম', 'পেয়েছি', ',', 'আলহামদুলিল্লাহ', ',', 'কথা', 'আর', 'কাজে', 'মিল', 'আছে', ',']
    Truncating punctuation: ['আমি', 'চট্টগ্রামে', 'আজকে', 'হাতে', 'পেলাম', 'অনেক', 'অনেক', 'ভাল', 'যেই', 'রকম', 'চেয়েছি', 'সেই', 'রকম', 'পেয়েছি', 'আলহামদুলিল্লাহ', 'কথা', 'আর', 'কাজে', 'মিল', 'আছে']
    Truncating StopWords: ['চট্টগ্রামে', 'আজকে', 'হাতে', 'পেলাম', 'ভাল', 'যেই', 'চেয়েছি', 'পেয়েছি', 'আলহামদুলিল্লাহ', 'কথা', 'মিল']
    ***************************************************************************************
    Label:  1
    Sentence:  বেস্ট কোয়ালিটি
    Afert Tokenizing:  ['বেস্ট', 'কোয়ালিটি']
    Truncating punctuation: ['বেস্ট', 'কোয়ালিটি']
    Truncating StopWords: ['বেস্ট', 'কোয়ালিটি']
    ***************************************************************************************
    Label:  1
    Sentence:  জোস কালেকশন
    Afert Tokenizing:  ['জোস', 'কালেকশন']
    Truncating punctuation: ['জোস', 'কালেকশন']
    Truncating StopWords: ['জোস', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  এক্সক্লুসিভ কালেকশন
    Afert Tokenizing:  ['এক্সক্লুসিভ', 'কালেকশন']
    Truncating punctuation: ['এক্সক্লুসিভ', 'কালেকশন']
    Truncating StopWords: ['এক্সক্লুসিভ', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটিফুল কালেকশন অলওয়েজ
    Afert Tokenizing:  ['কোয়ালিটিফুল', 'কালেকশন', 'অলওয়েজ']
    Truncating punctuation: ['কোয়ালিটিফুল', 'কালেকশন', 'অলওয়েজ']
    Truncating StopWords: ['কোয়ালিটিফুল', 'কালেকশন', 'অলওয়েজ']
    ***************************************************************************************
    Label:  1
    Sentence:  "অনলাইনে এই প্রথম কিছু নেওয়া হলো। ভয়টা কেটে গেল। ধন্যবাদ "
    Afert Tokenizing:  ['অনলাইনে', '"', 'এই', 'প্রথম', 'কিছু', 'নেওয়া', 'হলো', '।', 'ভয়টা', 'কেটে', 'গেল', '।', 'ধন্যবাদ', '', '"']
    Truncating punctuation: ['অনলাইনে', 'এই', 'প্রথম', 'কিছু', 'নেওয়া', 'হলো', 'ভয়টা', 'কেটে', 'গেল', 'ধন্যবাদ', '']
    Truncating StopWords: ['অনলাইনে', 'ভয়টা', 'কেটে', 'ধন্যবাদ', '']
    ***************************************************************************************
    Label:  0
    Sentence:  ঈদের দিন বাটার ভাউচার অর্ডার দিয়েছিলাম, ৪৮ ঘন্টার মধ্যে ডেলিভারি দেয়ার কথা ছিল কিন্তু এখনো তো পেলাম না
    Afert Tokenizing:  ['ঈদের', 'দিন', 'বাটার', 'ভাউচার', 'অর্ডার', 'দিয়েছিলাম', ',', '৪৮', 'ঘন্টার', 'মধ্যে', 'ডেলিভারি', 'দেয়ার', 'কথা', 'ছিল', 'কিন্তু', 'এখনো', 'তো', 'পেলাম', 'না']
    Truncating punctuation: ['ঈদের', 'দিন', 'বাটার', 'ভাউচার', 'অর্ডার', 'দিয়েছিলাম', '৪৮', 'ঘন্টার', 'মধ্যে', 'ডেলিভারি', 'দেয়ার', 'কথা', 'ছিল', 'কিন্তু', 'এখনো', 'তো', 'পেলাম', 'না']
    Truncating StopWords: ['ঈদের', 'বাটার', 'ভাউচার', 'অর্ডার', 'দিয়েছিলাম', '৪৮', 'ঘন্টার', 'ডেলিভারি', 'দেয়ার', 'কথা', 'এখনো', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আমরা বাংলাদেশে ই-কমার্স সাইটের প্রসার চাই
    Afert Tokenizing:  ['আমরা', 'বাংলাদেশে', 'ই-কমার্স', 'সাইটের', 'প্রসার', 'চাই']
    Truncating punctuation: ['আমরা', 'বাংলাদেশে', 'ই-কমার্স', 'সাইটের', 'প্রসার', 'চাই']
    Truncating StopWords: ['বাংলাদেশে', 'ই-কমার্স', 'সাইটের', 'প্রসার', 'চাই']
    ***************************************************************************************
    Label:  1
    Sentence:  অনলাইন কেনাকাটায় বদলে দেবে আপনার একমাত্র অনলাইন নির্ভরশীল প্রতিষ্ঠান মোনার্ক মার্ঠ
    Afert Tokenizing:  ['অনলাইন', 'কেনাকাটায়', 'বদলে', 'দেবে', 'আপনার', 'একমাত্র', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'মোনার্ক', 'মার্ঠ']
    Truncating punctuation: ['অনলাইন', 'কেনাকাটায়', 'বদলে', 'দেবে', 'আপনার', 'একমাত্র', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'মোনার্ক', 'মার্ঠ']
    Truncating StopWords: ['অনলাইন', 'কেনাকাটায়', 'দেবে', 'একমাত্র', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'মোনার্ক', 'মার্ঠ']
    ***************************************************************************************
    Label:  0
    Sentence:  মিনিমাম ১০% ডিসকাউন্ট দেয়া উচিত ছিল
    Afert Tokenizing:  ['মিনিমাম', '১০%', 'ডিসকাউন্ট', 'দেয়া', 'উচিত', 'ছিল']
    Truncating punctuation: ['মিনিমাম', '১০%', 'ডিসকাউন্ট', 'দেয়া', 'উচিত', 'ছিল']
    Truncating StopWords: ['মিনিমাম', '১০%', 'ডিসকাউন্ট', 'দেয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  এক টাকাও তো ডিস্কাউন্ট দিলেন না?
    Afert Tokenizing:  ['এক', 'টাকাও', 'তো', 'ডিস্কাউন্ট', 'দিলেন', 'না', '?']
    Truncating punctuation: ['এক', 'টাকাও', 'তো', 'ডিস্কাউন্ট', 'দিলেন', 'না']
    Truncating StopWords: ['এক', 'টাকাও', 'ডিস্কাউন্ট', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো ও সময়োপযোগী পদক্ষেপ
    Afert Tokenizing:  ['ভালো', 'ও', 'সময়োপযোগী', 'পদক্ষেপ']
    Truncating punctuation: ['ভালো', 'ও', 'সময়োপযোগী', 'পদক্ষেপ']
    Truncating StopWords: ['ভালো', 'সময়োপযোগী', 'পদক্ষেপ']
    ***************************************************************************************
    Label:  1
    Sentence:  মোনার্ক মার্ঠ একটি অনলাইন নির্ভরশীল প্রতিষ্ঠান এটার মাধ্যমে যে কোন অর্ডার করা যায়
    Afert Tokenizing:  ['মোনার্ক', 'মার্ঠ', 'একটি', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'এটার', 'মাধ্যমে', 'যে', 'কোন', 'অর্ডার', 'করা', 'যায়']
    Truncating punctuation: ['মোনার্ক', 'মার্ঠ', 'একটি', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'এটার', 'মাধ্যমে', 'যে', 'কোন', 'অর্ডার', 'করা', 'যায়']
    Truncating StopWords: ['মোনার্ক', 'মার্ঠ', 'অনলাইন', 'নির্ভরশীল', 'প্রতিষ্ঠান', 'এটার', 'অর্ডার', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  বেশ ভালো পন্য
    Afert Tokenizing:  ['বেশ', 'ভালো', 'পন্য']
    Truncating punctuation: ['বেশ', 'ভালো', 'পন্য']
    Truncating StopWords: ['ভালো', 'পন্য']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুণ অফার
    Afert Tokenizing:  ['দারুণ', 'অফার']
    Truncating punctuation: ['দারুণ', 'অফার']
    Truncating StopWords: ['দারুণ', 'অফার']
    ***************************************************************************************
    Label:  0
    Sentence:  9 তারিখের অর্ডার করলাম এখনো পাচ্ছিনা
    Afert Tokenizing:  ['9', 'তারিখের', 'অর্ডার', 'করলাম', 'এখনো', 'পাচ্ছিনা']
    Truncating punctuation: ['9', 'তারিখের', 'অর্ডার', 'করলাম', 'এখনো', 'পাচ্ছিনা']
    Truncating StopWords: ['9', 'তারিখের', 'অর্ডার', 'করলাম', 'এখনো', 'পাচ্ছিনা']
    ***************************************************************************************
    Label:  1
    Sentence:  অফার দেখে আমি শিহরিত
    Afert Tokenizing:  ['অফার', 'দেখে', 'আমি', 'শিহরিত']
    Truncating punctuation: ['অফার', 'দেখে', 'আমি', 'শিহরিত']
    Truncating StopWords: ['অফার', 'শিহরিত']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের পন‍্য আর একটু দম কমালে ভালো হয়
    Afert Tokenizing:  ['আপনাদের', 'পন\u200d্য', 'আর', 'একটু', 'দম', 'কমালে', 'ভালো', 'হয়']
    Truncating punctuation: ['আপনাদের', 'পন\u200d্য', 'আর', 'একটু', 'দম', 'কমালে', 'ভালো', 'হয়']
    Truncating StopWords: ['আপনাদের', 'পন\u200d্য', 'একটু', 'দম', 'কমালে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  গুড
    Afert Tokenizing:  ['গুড']
    Truncating punctuation: ['গুড']
    Truncating StopWords: ['গুড']
    ***************************************************************************************
    Label:  0
    Sentence:  এভাবে ঠকানোর মানে কি? আশাকরি রিপ্লে দিবেন, অর্ডার ক্যন্সেল করতে চাই আমি।
    Afert Tokenizing:  ['এভাবে', 'ঠকানোর', 'মানে', 'কি', '?', 'আশাকরি', 'রিপ্লে', 'দিবেন', ',', 'অর্ডার', 'ক্যন্সেল', 'করতে', 'চাই', 'আমি', '।']
    Truncating punctuation: ['এভাবে', 'ঠকানোর', 'মানে', 'কি', 'আশাকরি', 'রিপ্লে', 'দিবেন', 'অর্ডার', 'ক্যন্সেল', 'করতে', 'চাই', 'আমি']
    Truncating StopWords: ['এভাবে', 'ঠকানোর', 'মানে', 'আশাকরি', 'রিপ্লে', 'দিবেন', 'অর্ডার', 'ক্যন্সেল', 'চাই']
    ***************************************************************************************
    Label:  0
    Sentence:  এইসব এর মানে কি ভাই শর্তাবলী অনুযায়ী অর্ডার করলাম কিন্তু কোনো ক্যাসব্যাক পেলাম না  তোরা শুধু প্রডাক্ট ভেলিভারি দিতে আয় বাইন্ধা রাখমু
    Afert Tokenizing:  ['এইসব', 'এর', 'মানে', 'কি', 'ভাই', 'শর্তাবলী', 'অনুযায়ী', 'অর্ডার', 'করলাম', 'কিন্তু', 'কোনো', 'ক্যাসব্যাক', 'পেলাম', 'না', 'তোরা', 'শুধু', 'প্রডাক্ট', 'ভেলিভারি', 'দিতে', 'আয়', 'বাইন্ধা', 'রাখমু']
    Truncating punctuation: ['এইসব', 'এর', 'মানে', 'কি', 'ভাই', 'শর্তাবলী', 'অনুযায়ী', 'অর্ডার', 'করলাম', 'কিন্তু', 'কোনো', 'ক্যাসব্যাক', 'পেলাম', 'না', 'তোরা', 'শুধু', 'প্রডাক্ট', 'ভেলিভারি', 'দিতে', 'আয়', 'বাইন্ধা', 'রাখমু']
    Truncating StopWords: ['এইসব', 'মানে', 'ভাই', 'শর্তাবলী', 'অর্ডার', 'করলাম', 'ক্যাসব্যাক', 'পেলাম', 'না', 'তোরা', 'শুধু', 'প্রডাক্ট', 'ভেলিভারি', 'আয়', 'বাইন্ধা', 'রাখমু']
    ***************************************************************************************
    Label:  0
    Sentence:  অফারের নামে এসব হয়রানি বন্ধ করেন, কী অফার দিয়েছেন তার কোন সঠিক তথ্য দিতে পারেন না।
    Afert Tokenizing:  ['অফারের', 'নামে', 'এসব', 'হয়রানি', 'বন্ধ', 'করেন', ',', 'কী', 'অফার', 'দিয়েছেন', 'তার', 'কোন', 'সঠিক', 'তথ্য', 'দিতে', 'পারেন', 'না', '।']
    Truncating punctuation: ['অফারের', 'নামে', 'এসব', 'হয়রানি', 'বন্ধ', 'করেন', 'কী', 'অফার', 'দিয়েছেন', 'তার', 'কোন', 'সঠিক', 'তথ্য', 'দিতে', 'পারেন', 'না']
    Truncating StopWords: ['অফারের', 'নামে', 'এসব', 'হয়রানি', 'বন্ধ', 'অফার', 'দিয়েছেন', 'সঠিক', 'তথ্য', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  শুধু ধোঁকা বাজি
    Afert Tokenizing:  ['শুধু', 'ধোঁকা', 'বাজি']
    Truncating punctuation: ['শুধু', 'ধোঁকা', 'বাজি']
    Truncating StopWords: ['শুধু', 'ধোঁকা', 'বাজি']
    ***************************************************************************************
    Label:  0
    Sentence:  নগদে পেমেন্ট করে লাভ কী, কোন ডিসকাউন্ট নাই
    Afert Tokenizing:  ['নগদে', 'পেমেন্ট', 'করে', 'লাভ', 'কী', ',', 'কোন', 'ডিসকাউন্ট', 'নাই']
    Truncating punctuation: ['নগদে', 'পেমেন্ট', 'করে', 'লাভ', 'কী', 'কোন', 'ডিসকাউন্ট', 'নাই']
    Truncating StopWords: ['নগদে', 'পেমেন্ট', 'লাভ', 'ডিসকাউন্ট', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  এ ধান্দা হলো ফ্রী অর্ডার নেওয়ার জন্য তারপর মানুষের টাকা আটকে রাখবে,
    Afert Tokenizing:  ['এ', 'ধান্দা', 'হলো', 'ফ্রী', 'অর্ডার', 'নেওয়ার', 'জন্য', 'তারপর', 'মানুষের', 'টাকা', 'আটকে', 'রাখবে', ',']
    Truncating punctuation: ['এ', 'ধান্দা', 'হলো', 'ফ্রী', 'অর্ডার', 'নেওয়ার', 'জন্য', 'তারপর', 'মানুষের', 'টাকা', 'আটকে', 'রাখবে']
    Truncating StopWords: ['ধান্দা', 'ফ্রী', 'অর্ডার', 'নেওয়ার', 'মানুষের', 'টাকা', 'আটকে', 'রাখবে']
    ***************************************************************************************
    Label:  0
    Sentence:  কেউ এখন আর অনলাইনে অর্ডার করতে চায়না সবার মনে একটাই ভয়
    Afert Tokenizing:  ['কেউ', 'এখন', 'আর', 'অনলাইনে', 'অর্ডার', 'করতে', 'চায়না', 'সবার', 'মনে', 'একটাই', 'ভয়']
    Truncating punctuation: ['কেউ', 'এখন', 'আর', 'অনলাইনে', 'অর্ডার', 'করতে', 'চায়না', 'সবার', 'মনে', 'একটাই', 'ভয়']
    Truncating StopWords: ['অনলাইনে', 'অর্ডার', 'চায়না', 'একটাই', 'ভয়']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনারা এই রকম কম দামের প্রোডাক্ট দিয়ে বেশী দাম লিখলে জনগন খাবে না।
    Afert Tokenizing:  ['আপনারা', 'এই', 'রকম', 'কম', 'দামের', 'প্রোডাক্ট', 'দিয়ে', 'বেশী', 'দাম', 'লিখলে', 'জনগন', 'খাবে', 'না', '।']
    Truncating punctuation: ['আপনারা', 'এই', 'রকম', 'কম', 'দামের', 'প্রোডাক্ট', 'দিয়ে', 'বেশী', 'দাম', 'লিখলে', 'জনগন', 'খাবে', 'না']
    Truncating StopWords: ['আপনারা', 'কম', 'দামের', 'প্রোডাক্ট', 'দিয়ে', 'বেশী', 'দাম', 'লিখলে', 'জনগন', 'খাবে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  অভার প্রাইজড
    Afert Tokenizing:  ['অভার', 'প্রাইজড']
    Truncating punctuation: ['অভার', 'প্রাইজড']
    Truncating StopWords: ['অভার', 'প্রাইজড']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম অনুযায়ী, তেমন ভালো না, সাইজও ঠিক দেয়নি
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', ',', 'তেমন', 'ভালো', 'না', ',', 'সাইজও', 'ঠিক', 'দেয়নি']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'তেমন', 'ভালো', 'না', 'সাইজও', 'ঠিক', 'দেয়নি']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'ভালো', 'না', 'সাইজও', 'ঠিক', 'দেয়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রোডাক্ট পেয়েছি, যেমন চেয়েছিলাম, ঠিক তেমনি পেয়েছি,ধন্যবাদ আপনাদের মনের মতো প্রোডাক্ট দিবার জন্য
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'পেয়েছি', ',', 'যেমন', 'চেয়েছিলাম', ',', 'ঠিক', 'তেমনি', 'পেয়েছি,ধন্যবাদ', 'আপনাদের', 'মনের', 'মতো', 'প্রোডাক্ট', 'দিবার', 'জন্য']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'পেয়েছি', 'যেমন', 'চেয়েছিলাম', 'ঠিক', 'তেমনি', 'পেয়েছি,ধন্যবাদ', 'আপনাদের', 'মনের', 'মতো', 'প্রোডাক্ট', 'দিবার', 'জন্য']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'পেয়েছি', 'চেয়েছিলাম', 'ঠিক', 'তেমনি', 'পেয়েছি,ধন্যবাদ', 'আপনাদের', 'মনের', 'প্রোডাক্ট', 'দিবার']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাইজ বেশি ভাই
    Afert Tokenizing:  ['প্রাইজ', 'বেশি', 'ভাই']
    Truncating punctuation: ['প্রাইজ', 'বেশি', 'ভাই']
    Truncating StopWords: ['প্রাইজ', 'বেশি', 'ভাই']
    ***************************************************************************************
    Label:  1
    Sentence:  বাহ্
    Afert Tokenizing:  ['বাহ্']
    Truncating punctuation: ['বাহ্']
    Truncating StopWords: ['বাহ্']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ সুন্দর প্রোডাক্টগুলা.. এক্কেবারে পারফেক্ট কাস্টমাইজড
    Afert Tokenizing:  ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', '.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    Truncating punctuation: ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    Truncating StopWords: ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    ***************************************************************************************
    Label:  1
    Sentence:  1st অর্ডার এই সন্তুষ্ট আমি ,, ধন্যবাদ
    Afert Tokenizing:  ['1st', 'অর্ডার', 'এই', 'সন্তুষ্ট', 'আমি', ',', ',', 'ধন্যবাদ']
    Truncating punctuation: ['1st', 'অর্ডার', 'এই', 'সন্তুষ্ট', 'আমি', 'ধন্যবাদ']
    Truncating StopWords: ['1st', 'অর্ডার', 'সন্তুষ্ট', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনারাও নিয়ে নিতে পারেন।
    Afert Tokenizing:  ['আপনারাও', 'নিয়ে', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['আপনারাও', 'নিয়ে', 'নিতে', 'পারেন']
    Truncating StopWords: ['আপনারাও']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকেই ডেলিভারি পেলাম। আলহামদুলিল্লাহ কোয়ালিটি সহ সবকিছুই ঠিকঠাক আছে। সবচেয়ে ভালো লেগেছে যেই বিষয়টা সেটা হলো পেইজের কথা এবং কাজে মিল আছে যেটা এখন পাওয়া খুবই দুষ্কর।
    Afert Tokenizing:  ['আজকেই', 'ডেলিভারি', 'পেলাম', '।', 'আলহামদুলিল্লাহ', 'কোয়ালিটি', 'সহ', 'সবকিছুই', 'ঠিকঠাক', 'আছে', '।', 'সবচেয়ে', 'ভালো', 'লেগেছে', 'যেই', 'বিষয়টা', 'সেটা', 'হলো', 'পেইজের', 'কথা', 'এবং', 'কাজে', 'মিল', 'আছে', 'যেটা', 'এখন', 'পাওয়া', 'খুবই', 'দুষ্কর', '।']
    Truncating punctuation: ['আজকেই', 'ডেলিভারি', 'পেলাম', 'আলহামদুলিল্লাহ', 'কোয়ালিটি', 'সহ', 'সবকিছুই', 'ঠিকঠাক', 'আছে', 'সবচেয়ে', 'ভালো', 'লেগেছে', 'যেই', 'বিষয়টা', 'সেটা', 'হলো', 'পেইজের', 'কথা', 'এবং', 'কাজে', 'মিল', 'আছে', 'যেটা', 'এখন', 'পাওয়া', 'খুবই', 'দুষ্কর']
    Truncating StopWords: ['আজকেই', 'ডেলিভারি', 'পেলাম', 'আলহামদুলিল্লাহ', 'কোয়ালিটি', 'সবকিছুই', 'ঠিকঠাক', 'সবচেয়ে', 'ভালো', 'লেগেছে', 'যেই', 'বিষয়টা', 'পেইজের', 'কথা', 'মিল', 'যেটা', 'পাওয়া', 'খুবই', 'দুষ্কর']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক অনেক ধন্যবাদ mr fiction কে
    Afert Tokenizing:  ['অনেক', 'অনেক', 'ধন্যবাদ', 'mr', 'fiction', 'কে']
    Truncating punctuation: ['অনেক', 'অনেক', 'ধন্যবাদ', 'mr', 'fiction', 'কে']
    Truncating StopWords: ['ধন্যবাদ', 'mr', 'fiction']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি অনেক ভালো, প্রিন্ট ও চোখে পড়ার মতো। আমি সন্তুষ্ট
    Afert Tokenizing:  ['কোয়ালিটি', 'অনেক', 'ভালো', ',', 'প্রিন্ট', 'ও', 'চোখে', 'পড়ার', 'মতো', '।', 'আমি', 'সন্তুষ্ট']
    Truncating punctuation: ['কোয়ালিটি', 'অনেক', 'ভালো', 'প্রিন্ট', 'ও', 'চোখে', 'পড়ার', 'মতো', 'আমি', 'সন্তুষ্ট']
    Truncating StopWords: ['কোয়ালিটি', 'ভালো', 'প্রিন্ট', 'চোখে', 'পড়ার', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  কিছুদিন ব্যবহার করে দেখলাম আল্লাহামদুলিল্লাহ কভারের মান অত্যান্ত ভাল
    Afert Tokenizing:  ['কিছুদিন', 'ব্যবহার', 'করে', 'দেখলাম', 'আল্লাহামদুলিল্লাহ', 'কভারের', 'মান', 'অত্যান্ত', 'ভাল']
    Truncating punctuation: ['কিছুদিন', 'ব্যবহার', 'করে', 'দেখলাম', 'আল্লাহামদুলিল্লাহ', 'কভারের', 'মান', 'অত্যান্ত', 'ভাল']
    Truncating StopWords: ['কিছুদিন', 'দেখলাম', 'আল্লাহামদুলিল্লাহ', 'কভারের', 'মান', 'অত্যান্ত', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  আল্লাহতালার রহমতে খুব ভাল প্রিমিয়াম কলেটির কভার পেয়েছি।ধন্যবাদ
    Afert Tokenizing:  ['আল্লাহতালার', 'রহমতে', 'খুব', 'ভাল', 'প্রিমিয়াম', 'কলেটির', 'কভার', 'পেয়েছি।ধন্যবাদ']
    Truncating punctuation: ['আল্লাহতালার', 'রহমতে', 'খুব', 'ভাল', 'প্রিমিয়াম', 'কলেটির', 'কভার', 'পেয়েছি।ধন্যবাদ']
    Truncating StopWords: ['আল্লাহতালার', 'রহমতে', 'ভাল', 'প্রিমিয়াম', 'কলেটির', 'কভার', 'পেয়েছি।ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশআল্লাহ  আল্লাহামদুলিল্লাহ
    Afert Tokenizing:  ['মাশআল্লাহ', 'আল্লাহামদুলিল্লাহ']
    Truncating punctuation: ['মাশআল্লাহ', 'আল্লাহামদুলিল্লাহ']
    Truncating StopWords: ['মাশআল্লাহ', 'আল্লাহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম অনুযায়ী কাভার গুলো ঠিক আসে।
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'কাভার', 'গুলো', 'ঠিক', 'আসে', '।']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'কাভার', 'গুলো', 'ঠিক', 'আসে']
    Truncating StopWords: ['দাম', 'কাভার', 'গুলো', 'ঠিক', 'আসে']
    ***************************************************************************************
    Label:  1
    Sentence:  দুই দিনের ভিতর হাতে পেয়েছি ৷ Mr. Fiction কে অসংখ্য ধন্যবাদ
    Afert Tokenizing:  ['দুই', 'দিনের', 'ভিতর', 'হাতে', 'পেয়েছি', '৷', 'Mr', '.', 'Fiction', 'কে', 'অসংখ্য', 'ধন্যবাদ']
    Truncating punctuation: ['দুই', 'দিনের', 'ভিতর', 'হাতে', 'পেয়েছি', '৷', 'Mr', 'Fiction', 'কে', 'অসংখ্য', 'ধন্যবাদ']
    Truncating StopWords: ['দিনের', 'ভিতর', 'হাতে', 'পেয়েছি', '৷', 'Mr', 'Fiction', 'অসংখ্য', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  দামে কম, মানে ভাল..
    Afert Tokenizing:  ['দামে', 'কম', ',', 'মানে', 'ভাল.', '.']
    Truncating punctuation: ['দামে', 'কম', 'মানে', 'ভাল.']
    Truncating StopWords: ['দামে', 'কম', 'মানে', 'ভাল.']
    ***************************************************************************************
    Label:  1
    Sentence:  অন্যরা ও কোন কিছু না ভেবে চট  করে অর্ডার দিয়ে দেন৷ আপনিও নিরাশ হবেন না  সেই গ্যারান্টি দিলাম। সব কিছুর জন্য  ধন্যবাদ
    Afert Tokenizing:  ['অন্যরা', 'ও', 'কোন', 'কিছু', 'না', 'ভেবে', 'চট', 'করে', 'অর্ডার', 'দিয়ে', 'দেন৷', 'আপনিও', 'নিরাশ', 'হবেন', 'না', 'সেই', 'গ্যারান্টি', 'দিলাম', '।', 'সব', 'কিছুর', 'জন্য', 'ধন্যবাদ']
    Truncating punctuation: ['অন্যরা', 'ও', 'কোন', 'কিছু', 'না', 'ভেবে', 'চট', 'করে', 'অর্ডার', 'দিয়ে', 'দেন৷', 'আপনিও', 'নিরাশ', 'হবেন', 'না', 'সেই', 'গ্যারান্টি', 'দিলাম', 'সব', 'কিছুর', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['অন্যরা', 'না', 'ভেবে', 'চট', 'অর্ডার', 'দিয়ে', 'দেন৷', 'আপনিও', 'নিরাশ', 'না', 'গ্যারান্টি', 'দিলাম', 'কিছুর', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  এত দামি খাতায় কি লিখবো
    Afert Tokenizing:  ['এত', 'দামি', 'খাতায়', 'কি', 'লিখবো']
    Truncating punctuation: ['এত', 'দামি', 'খাতায়', 'কি', 'লিখবো']
    Truncating StopWords: ['দামি', 'খাতায়', 'লিখবো']
    ***************************************************************************************
    Label:  0
    Sentence:  এত দাম কেও রাখে?
    Afert Tokenizing:  ['এত', 'দাম', 'কেও', 'রাখে', '?']
    Truncating punctuation: ['এত', 'দাম', 'কেও', 'রাখে']
    Truncating StopWords: ['দাম', 'কেও', 'রাখে']
    ***************************************************************************************
    Label:  1
    Sentence:  জিনিস সুন্দর কিন্তু দাম বেশি
    Afert Tokenizing:  ['জিনিস', 'সুন্দর', 'কিন্তু', 'দাম', 'বেশি']
    Truncating punctuation: ['জিনিস', 'সুন্দর', 'কিন্তু', 'দাম', 'বেশি']
    Truncating StopWords: ['জিনিস', 'সুন্দর', 'দাম', 'বেশি']
    ***************************************************************************************
    Label:  0
    Sentence:  ব্যাবসা করার তো একটা লিমিট থাকা উচিত।
    Afert Tokenizing:  ['ব্যাবসা', 'করার', 'তো', 'একটা', 'লিমিট', 'থাকা', 'উচিত', '।']
    Truncating punctuation: ['ব্যাবসা', 'করার', 'তো', 'একটা', 'লিমিট', 'থাকা', 'উচিত']
    Truncating StopWords: ['ব্যাবসা', 'একটা', 'লিমিট']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই বিজনেস করতেছেন ভালো কথা। তবে আপনাদের উচিত ক্রেতা + যারা কমেন্ট করতেছে তাদের মতামত গুলোকে প্রধান্য দেওয়া।
    Afert Tokenizing:  ['ভাই', 'বিজনেস', 'করতেছেন', 'ভালো', 'কথা', '।', 'তবে', 'আপনাদের', 'উচিত', 'ক্রেতা', '+', 'যারা', 'কমেন্ট', 'করতেছে', 'তাদের', 'মতামত', 'গুলোকে', 'প্রধান্য', 'দেওয়া', '।']
    Truncating punctuation: ['ভাই', 'বিজনেস', 'করতেছেন', 'ভালো', 'কথা', 'তবে', 'আপনাদের', 'উচিত', 'ক্রেতা', '+', 'যারা', 'কমেন্ট', 'করতেছে', 'তাদের', 'মতামত', 'গুলোকে', 'প্রধান্য', 'দেওয়া']
    Truncating StopWords: ['ভাই', 'বিজনেস', 'করতেছেন', 'ভালো', 'কথা', 'আপনাদের', 'ক্রেতা', '+', 'কমেন্ট', 'করতেছে', 'মতামত', 'গুলোকে', 'প্রধান্য']
    ***************************************************************************************
    Label:  0
    Sentence:  যে মূল্যটা দিয়েছেন সেটা অনেক বেশি।
    Afert Tokenizing:  ['যে', 'মূল্যটা', 'দিয়েছেন', 'সেটা', 'অনেক', 'বেশি', '।']
    Truncating punctuation: ['যে', 'মূল্যটা', 'দিয়েছেন', 'সেটা', 'অনেক', 'বেশি']
    Truncating StopWords: ['মূল্যটা', 'দিয়েছেন', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকে প্রডাক্টা পেলাম খুব সুন্দর হইছে ধন্যবাদ আপনাদের
    Afert Tokenizing:  ['আজকে', 'প্রডাক্টা', 'পেলাম', 'খুব', 'সুন্দর', 'হইছে', 'ধন্যবাদ', 'আপনাদের']
    Truncating punctuation: ['আজকে', 'প্রডাক্টা', 'পেলাম', 'খুব', 'সুন্দর', 'হইছে', 'ধন্যবাদ', 'আপনাদের']
    Truncating StopWords: ['আজকে', 'প্রডাক্টা', 'পেলাম', 'সুন্দর', 'হইছে', 'ধন্যবাদ', 'আপনাদের']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রথমটা পছন্দ হইসে বাট দামটা বেশি,,কিছু কমানো যায়না
    Afert Tokenizing:  ['প্রথমটা', 'পছন্দ', 'হইসে', 'বাট', 'দামটা', 'বেশি,,কিছু', 'কমানো', 'যায়না']
    Truncating punctuation: ['প্রথমটা', 'পছন্দ', 'হইসে', 'বাট', 'দামটা', 'বেশি,,কিছু', 'কমানো', 'যায়না']
    Truncating StopWords: ['প্রথমটা', 'পছন্দ', 'হইসে', 'বাট', 'দামটা', 'বেশি,,কিছু', 'কমানো', 'যায়না']
    ***************************************************************************************
    Label:  1
    Sentence:  অনলাইন প্রোডাক্ট হিসেবে অনেক ভালো। সুন্দর ফিটিং হয়েছে। রেটিং এ দশে দশ। চাইলে নিতে পারেন প্রোডাক্ট এর মানে অনেক ভালো।
    Afert Tokenizing:  ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'অনেক', 'ভালো', '।', 'সুন্দর', 'ফিটিং', 'হয়েছে', '।', 'রেটিং', 'এ', 'দশে', 'দশ', '।', 'চাইলে', 'নিতে', 'পারেন', 'প্রোডাক্ট', 'এর', 'মানে', 'অনেক', 'ভালো', '।']
    Truncating punctuation: ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'অনেক', 'ভালো', 'সুন্দর', 'ফিটিং', 'হয়েছে', 'রেটিং', 'এ', 'দশে', 'দশ', 'চাইলে', 'নিতে', 'পারেন', 'প্রোডাক্ট', 'এর', 'মানে', 'অনেক', 'ভালো']
    Truncating StopWords: ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'ভালো', 'সুন্দর', 'ফিটিং', 'রেটিং', 'দশে', 'দশ', 'চাইলে', 'প্রোডাক্ট', 'মানে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  "আমি দুইটা প্যান্ট অর্ডার করেছি, দুইটা প্যান্টই যথেষ্ট ভালো হয়েছে,মাপ একুরেট হয়েছে, অনেক অনেক ধন্যবাদ আপনাদেরকে"
    Afert Tokenizing:  ['আমি', '"', 'দুইটা', 'প্যান্ট', 'অর্ডার', 'করেছি', ',', 'দুইটা', 'প্যান্টই', 'যথেষ্ট', 'ভালো', 'হয়েছে,মাপ', 'একুরেট', 'হয়েছে', ',', 'অনেক', 'অনেক', 'ধন্যবাদ', 'আপনাদেরকে', '"']
    Truncating punctuation: ['আমি', 'দুইটা', 'প্যান্ট', 'অর্ডার', 'করেছি', 'দুইটা', 'প্যান্টই', 'যথেষ্ট', 'ভালো', 'হয়েছে,মাপ', 'একুরেট', 'হয়েছে', 'অনেক', 'অনেক', 'ধন্যবাদ', 'আপনাদেরকে']
    Truncating StopWords: ['দুইটা', 'প্যান্ট', 'অর্ডার', 'করেছি', 'দুইটা', 'প্যান্টই', 'যথেষ্ট', 'ভালো', 'হয়েছে,মাপ', 'একুরেট', 'হয়েছে', 'ধন্যবাদ', 'আপনাদেরকে']
    ***************************************************************************************
    Label:  1
    Sentence:  এত কম মূল্যে এতো কোয়ালিটি সম্পন্ন ভালো প্রোডাক্ট সত্যিই প্রত্যাশার অধিক। সেইসাথে দ্রুতগতির ডেলিভারি,প্যাকেজিং ও স্টাফদের ব্যাবহারে সত্যিই আমি মুগ্ধ
    Afert Tokenizing:  ['এত', 'কম', 'মূল্যে', 'এতো', 'কোয়ালিটি', 'সম্পন্ন', 'ভালো', 'প্রোডাক্ট', 'সত্যিই', 'প্রত্যাশার', 'অধিক', '।', 'সেইসাথে', 'দ্রুতগতির', 'ডেলিভারি,প্যাকেজিং', 'ও', 'স্টাফদের', 'ব্যাবহারে', 'সত্যিই', 'আমি', 'মুগ্ধ']
    Truncating punctuation: ['এত', 'কম', 'মূল্যে', 'এতো', 'কোয়ালিটি', 'সম্পন্ন', 'ভালো', 'প্রোডাক্ট', 'সত্যিই', 'প্রত্যাশার', 'অধিক', 'সেইসাথে', 'দ্রুতগতির', 'ডেলিভারি,প্যাকেজিং', 'ও', 'স্টাফদের', 'ব্যাবহারে', 'সত্যিই', 'আমি', 'মুগ্ধ']
    Truncating StopWords: ['কম', 'মূল্যে', 'এতো', 'কোয়ালিটি', 'সম্পন্ন', 'ভালো', 'প্রোডাক্ট', 'সত্যিই', 'প্রত্যাশার', 'অধিক', 'সেইসাথে', 'দ্রুতগতির', 'ডেলিভারি,প্যাকেজিং', 'স্টাফদের', 'ব্যাবহারে', 'সত্যিই', 'মুগ্ধ']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ থেকে আমি আপনাদের ফ্যান হয়ে গেলাম
    Afert Tokenizing:  ['আজ', 'থেকে', 'আমি', 'আপনাদের', 'ফ্যান', 'হয়ে', 'গেলাম']
    Truncating punctuation: ['আজ', 'থেকে', 'আমি', 'আপনাদের', 'ফ্যান', 'হয়ে', 'গেলাম']
    Truncating StopWords: ['আপনাদের', 'ফ্যান', 'হয়ে', 'গেলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ভালো জিনিস হাতে পেয়েছি। ধন্যবাদ...
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'ভালো', 'জিনিস', 'হাতে', 'পেয়েছি', '।', 'ধন্যবাদ..', '.']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ভালো', 'জিনিস', 'হাতে', 'পেয়েছি', 'ধন্যবাদ..']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ভালো', 'জিনিস', 'হাতে', 'পেয়েছি', 'ধন্যবাদ..']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যি বলতে, দাম এর সাথে তুলনা করলে, অসাধারণ মানের। শুভ কামনা রইলো।
    Afert Tokenizing:  ['সত্যি', 'বলতে', ',', 'দাম', 'এর', 'সাথে', 'তুলনা', 'করলে', ',', 'অসাধারণ', 'মানের', '।', 'শুভ', 'কামনা', 'রইলো', '।']
    Truncating punctuation: ['সত্যি', 'বলতে', 'দাম', 'এর', 'সাথে', 'তুলনা', 'করলে', 'অসাধারণ', 'মানের', 'শুভ', 'কামনা', 'রইলো']
    Truncating StopWords: ['সত্যি', 'দাম', 'সাথে', 'তুলনা', 'অসাধারণ', 'মানের', 'শুভ', 'কামনা', 'রইলো']
    ***************************************************************************************
    Label:  1
    Sentence:  আজেই আপনাদের পণ্য হাতে পেয়েছি পণ্য নের কোয়ালিটি খুবই ভালো এবং আরামদায়ক
    Afert Tokenizing:  ['আজেই', 'আপনাদের', 'পণ্য', 'হাতে', 'পেয়েছি', 'পণ্য', 'নের', 'কোয়ালিটি', 'খুবই', 'ভালো', 'এবং', 'আরামদায়ক']
    Truncating punctuation: ['আজেই', 'আপনাদের', 'পণ্য', 'হাতে', 'পেয়েছি', 'পণ্য', 'নের', 'কোয়ালিটি', 'খুবই', 'ভালো', 'এবং', 'আরামদায়ক']
    Truncating StopWords: ['আজেই', 'আপনাদের', 'পণ্য', 'হাতে', 'পেয়েছি', 'পণ্য', 'নের', 'কোয়ালিটি', 'খুবই', 'ভালো', 'আরামদায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  রিভিউ ৫ ভিতর ৫। সামনে হয়তো চোখ বন্ধ করে অর্ডার করবো
    Afert Tokenizing:  ['রিভিউ', '৫', 'ভিতর', '৫', '।', 'সামনে', 'হয়তো', 'চোখ', 'বন্ধ', 'করে', 'অর্ডার', 'করবো']
    Truncating punctuation: ['রিভিউ', '৫', 'ভিতর', '৫', 'সামনে', 'হয়তো', 'চোখ', 'বন্ধ', 'করে', 'অর্ডার', 'করবো']
    Truncating StopWords: ['রিভিউ', '৫', 'ভিতর', '৫', 'হয়তো', 'চোখ', 'বন্ধ', 'অর্ডার', 'করবো']
    ***************************************************************************************
    Label:  1
    Sentence:  গতকাল অর্ডার করেছিলাম আজকেই হাতে পেয়েছি এবং গুণগত মানের দিক দিয়ে বলতে হয় খুবই ভালো ধন্যবাদ জানাচ্ছি ও অনেক শুভকামনা রইলো।
    Afert Tokenizing:  ['গতকাল', 'অর্ডার', 'করেছিলাম', 'আজকেই', 'হাতে', 'পেয়েছি', 'এবং', 'গুণগত', 'মানের', 'দিক', 'দিয়ে', 'বলতে', 'হয়', 'খুবই', 'ভালো', 'ধন্যবাদ', 'জানাচ্ছি', 'ও', 'অনেক', 'শুভকামনা', 'রইলো', '।']
    Truncating punctuation: ['গতকাল', 'অর্ডার', 'করেছিলাম', 'আজকেই', 'হাতে', 'পেয়েছি', 'এবং', 'গুণগত', 'মানের', 'দিক', 'দিয়ে', 'বলতে', 'হয়', 'খুবই', 'ভালো', 'ধন্যবাদ', 'জানাচ্ছি', 'ও', 'অনেক', 'শুভকামনা', 'রইলো']
    Truncating StopWords: ['গতকাল', 'অর্ডার', 'করেছিলাম', 'আজকেই', 'হাতে', 'পেয়েছি', 'গুণগত', 'মানের', 'দিক', 'খুবই', 'ভালো', 'ধন্যবাদ', 'জানাচ্ছি', 'শুভকামনা', 'রইলো']
    ***************************************************************************************
    Label:  1
    Sentence:  আশা করি ভবিষ্যতে পন্যের মান ঠিক রেখে আরো বহুদূর এগিয়ে যাবেন।শুভ কামনা
    Afert Tokenizing:  ['আশা', 'করি', 'ভবিষ্যতে', 'পন্যের', 'মান', 'ঠিক', 'রেখে', 'আরো', 'বহুদূর', 'এগিয়ে', 'যাবেন।শুভ', 'কামনা']
    Truncating punctuation: ['আশা', 'করি', 'ভবিষ্যতে', 'পন্যের', 'মান', 'ঠিক', 'রেখে', 'আরো', 'বহুদূর', 'এগিয়ে', 'যাবেন।শুভ', 'কামনা']
    Truncating StopWords: ['আশা', 'ভবিষ্যতে', 'পন্যের', 'মান', 'ঠিক', 'আরো', 'বহুদূর', 'এগিয়ে', 'যাবেন।শুভ', 'কামনা']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ। পণ্যগুলো খুব দ্রুত হাতে পেয়েছি।
    Afert Tokenizing:  ['ধন্যবাদ', '।', 'পণ্যগুলো', 'খুব', 'দ্রুত', 'হাতে', 'পেয়েছি', '।']
    Truncating punctuation: ['ধন্যবাদ', 'পণ্যগুলো', 'খুব', 'দ্রুত', 'হাতে', 'পেয়েছি']
    Truncating StopWords: ['ধন্যবাদ', 'পণ্যগুলো', 'দ্রুত', 'হাতে', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ Deen এতো সুন্দর পণ্য আমাদের কাছে পৌছে দেওয়ার জন্য। এগিয়ে যাক সামনের দিকে এই কামনা করি।
    Afert Tokenizing:  ['ধন্যবাদ', 'Deen', 'এতো', 'সুন্দর', 'পণ্য', 'আমাদের', 'কাছে', 'পৌছে', 'দেওয়ার', 'জন্য', '।', 'এগিয়ে', 'যাক', 'সামনের', 'দিকে', 'এই', 'কামনা', 'করি', '।']
    Truncating punctuation: ['ধন্যবাদ', 'Deen', 'এতো', 'সুন্দর', 'পণ্য', 'আমাদের', 'কাছে', 'পৌছে', 'দেওয়ার', 'জন্য', 'এগিয়ে', 'যাক', 'সামনের', 'দিকে', 'এই', 'কামনা', 'করি']
    Truncating StopWords: ['ধন্যবাদ', 'Deen', 'এতো', 'সুন্দর', 'পণ্য', 'পৌছে', 'দেওয়ার', 'এগিয়ে', 'যাক', 'সামনের', 'কামনা']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ কিছুক্ষন আগে আমার অর্ডারকৃত পণ্যটি হাতে পেলাম।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'কিছুক্ষন', 'আগে', 'আমার', 'অর্ডারকৃত', 'পণ্যটি', 'হাতে', 'পেলাম', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'কিছুক্ষন', 'আগে', 'আমার', 'অর্ডারকৃত', 'পণ্যটি', 'হাতে', 'পেলাম']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'কিছুক্ষন', 'অর্ডারকৃত', 'পণ্যটি', 'হাতে', 'পেলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আমিও নিলাম। খুবই ভালো।
    Afert Tokenizing:  ['আমিও', 'নিলাম', '।', 'খুবই', 'ভালো', '।']
    Truncating punctuation: ['আমিও', 'নিলাম', 'খুবই', 'ভালো']
    Truncating StopWords: ['আমিও', 'নিলাম', 'খুবই', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেটিং খুব ভালো ও দ্রুত পণ্য ডেলিভারি দেয়া হয়েছে।
    Afert Tokenizing:  ['প্যাকেটিং', 'খুব', 'ভালো', 'ও', 'দ্রুত', 'পণ্য', 'ডেলিভারি', 'দেয়া', 'হয়েছে', '।']
    Truncating punctuation: ['প্যাকেটিং', 'খুব', 'ভালো', 'ও', 'দ্রুত', 'পণ্য', 'ডেলিভারি', 'দেয়া', 'হয়েছে']
    Truncating StopWords: ['প্যাকেটিং', 'ভালো', 'দ্রুত', 'পণ্য', 'ডেলিভারি', 'দেয়া', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ আপনাদের কে এত দ্রুত ডেলিভারী দেওয়ার জন্যে
    Afert Tokenizing:  ['ধন্যবাদ', 'আপনাদের', 'কে', 'এত', 'দ্রুত', 'ডেলিভারী', 'দেওয়ার', 'জন্যে']
    Truncating punctuation: ['ধন্যবাদ', 'আপনাদের', 'কে', 'এত', 'দ্রুত', 'ডেলিভারী', 'দেওয়ার', 'জন্যে']
    Truncating StopWords: ['ধন্যবাদ', 'আপনাদের', 'দ্রুত', 'ডেলিভারী', 'জন্যে']
    ***************************************************************************************
    Label:  1
    Sentence:  Deen  থেকে এত ভালো প্রোডাক্ট অনলাইন থেকে পাবো এটা আশা করিনি
    Afert Tokenizing:  ['Deen', 'থেকে', 'এত', 'ভালো', 'প্রোডাক্ট', 'অনলাইন', 'থেকে', 'পাবো', 'এটা', 'আশা', 'করিনি']
    Truncating punctuation: ['Deen', 'থেকে', 'এত', 'ভালো', 'প্রোডাক্ট', 'অনলাইন', 'থেকে', 'পাবো', 'এটা', 'আশা', 'করিনি']
    Truncating StopWords: ['Deen', 'ভালো', 'প্রোডাক্ট', 'অনলাইন', 'পাবো', 'আশা', 'করিনি']
    ***************************************************************************************
    Label:  1
    Sentence:  তারা অনেক ভালো
    Afert Tokenizing:  ['তারা', 'অনেক', 'ভালো']
    Truncating punctuation: ['তারা', 'অনেক', 'ভালো']
    Truncating StopWords: ['ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি, সার্ভিস খুব-ই ভালো। আরো নতুন নতুন কালেকশনের অপেক্ষায় থাকলাম। শুভকামনা।
    Afert Tokenizing:  ['কোয়ালিটি', ',', 'সার্ভিস', 'খুব-ই', 'ভালো', '।', 'আরো', 'নতুন', 'নতুন', 'কালেকশনের', 'অপেক্ষায়', 'থাকলাম', '।', 'শুভকামনা', '।']
    Truncating punctuation: ['কোয়ালিটি', 'সার্ভিস', 'খুব-ই', 'ভালো', 'আরো', 'নতুন', 'নতুন', 'কালেকশনের', 'অপেক্ষায়', 'থাকলাম', 'শুভকামনা']
    Truncating StopWords: ['কোয়ালিটি', 'সার্ভিস', 'খুব-ই', 'ভালো', 'আরো', 'কালেকশনের', 'অপেক্ষায়', 'থাকলাম', 'শুভকামনা']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম টাও ঠিকাছে মনে হয়েছে আমার কাছে। প্যাকেজিং ও সুন্দর।
    Afert Tokenizing:  ['দাম', 'টাও', 'ঠিকাছে', 'মনে', 'হয়েছে', 'আমার', 'কাছে', '।', 'প্যাকেজিং', 'ও', 'সুন্দর', '।']
    Truncating punctuation: ['দাম', 'টাও', 'ঠিকাছে', 'মনে', 'হয়েছে', 'আমার', 'কাছে', 'প্যাকেজিং', 'ও', 'সুন্দর']
    Truncating StopWords: ['দাম', 'টাও', 'ঠিকাছে', 'হয়েছে', 'প্যাকেজিং', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকিং,প্যান্ট কোয়ালিটি ও দ্রুত ডেলিভারী দেওয়ার জন্য ধন্যবাদ।
    Afert Tokenizing:  ['প্যাকিং,প্যান্ট', 'কোয়ালিটি', 'ও', 'দ্রুত', 'ডেলিভারী', 'দেওয়ার', 'জন্য', 'ধন্যবাদ', '।']
    Truncating punctuation: ['প্যাকিং,প্যান্ট', 'কোয়ালিটি', 'ও', 'দ্রুত', 'ডেলিভারী', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['প্যাকিং,প্যান্ট', 'কোয়ালিটি', 'দ্রুত', 'ডেলিভারী', 'দেওয়ার', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক তারাতাড়ি পেয়েছি।  একদম বাড়িতে দিয়ে গেল ভালো ছিল সব মিলিয়ে!
    Afert Tokenizing:  ['অনেক', 'তারাতাড়ি', 'পেয়েছি', '।', 'একদম', 'বাড়িতে', 'দিয়ে', 'গেল', 'ভালো', 'ছিল', 'সব', 'মিলিয়ে', '!']
    Truncating punctuation: ['অনেক', 'তারাতাড়ি', 'পেয়েছি', 'একদম', 'বাড়িতে', 'দিয়ে', 'গেল', 'ভালো', 'ছিল', 'সব', 'মিলিয়ে']
    Truncating StopWords: ['তারাতাড়ি', 'পেয়েছি', 'একদম', 'বাড়িতে', 'দিয়ে', 'ভালো', 'মিলিয়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ কাপড়ের মান অনেক ভালো... এই টা দিয়ে আমি তিনবার নিলাম ডিন থেকে.... মাশাআল্লাহ ডেলিভারিও পেয়েছি খুবই তাড়াতাড়ি...
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'কাপড়ের', 'মান', 'অনেক', 'ভালো..', '.', 'এই', 'টা', 'দিয়ে', 'আমি', 'তিনবার', 'নিলাম', 'ডিন', 'থেকে...', '.', 'মাশাআল্লাহ', 'ডেলিভারিও', 'পেয়েছি', 'খুবই', 'তাড়াতাড়ি..', '.']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'কাপড়ের', 'মান', 'অনেক', 'ভালো..', 'এই', 'টা', 'দিয়ে', 'আমি', 'তিনবার', 'নিলাম', 'ডিন', 'থেকে...', 'মাশাআল্লাহ', 'ডেলিভারিও', 'পেয়েছি', 'খুবই', 'তাড়াতাড়ি..']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'কাপড়ের', 'মান', 'ভালো..', 'টা', 'দিয়ে', 'তিনবার', 'নিলাম', 'ডিন', 'থেকে...', 'মাশাআল্লাহ', 'ডেলিভারিও', 'পেয়েছি', 'খুবই', 'তাড়াতাড়ি..']
    ***************************************************************************************
    Label:  1
    Sentence:  আগামীতে আরো কেনাকাটা হবে আপনাদের থেকে ইনশাআল্লাহ
    Afert Tokenizing:  ['আগামীতে', 'আরো', 'কেনাকাটা', 'হবে', 'আপনাদের', 'থেকে', 'ইনশাআল্লাহ']
    Truncating punctuation: ['আগামীতে', 'আরো', 'কেনাকাটা', 'হবে', 'আপনাদের', 'থেকে', 'ইনশাআল্লাহ']
    Truncating StopWords: ['আগামীতে', 'আরো', 'কেনাকাটা', 'আপনাদের', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  অসংখ্য ধন্যবাদ। পণ্য টি পাওয়ার জন্য
    Afert Tokenizing:  ['অসংখ্য', 'ধন্যবাদ', '।', 'পণ্য', 'টি', 'পাওয়ার', 'জন্য']
    Truncating punctuation: ['অসংখ্য', 'ধন্যবাদ', 'পণ্য', 'টি', 'পাওয়ার', 'জন্য']
    Truncating StopWords: ['অসংখ্য', 'ধন্যবাদ', 'পণ্য', 'পাওয়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  বর্তমান দুই নাম্বরি ই-কমার্স এর মধ্য থেকে ভালো ট্রাস্টেট একটা ই-কমার্স খুঁজে পাওয়া অনেক দুষ্কর
    Afert Tokenizing:  ['বর্তমান', 'দুই', 'নাম্বরি', 'ই-কমার্স', 'এর', 'মধ্য', 'থেকে', 'ভালো', 'ট্রাস্টেট', 'একটা', 'ই-কমার্স', 'খুঁজে', 'পাওয়া', 'অনেক', 'দুষ্কর']
    Truncating punctuation: ['বর্তমান', 'দুই', 'নাম্বরি', 'ই-কমার্স', 'এর', 'মধ্য', 'থেকে', 'ভালো', 'ট্রাস্টেট', 'একটা', 'ই-কমার্স', 'খুঁজে', 'পাওয়া', 'অনেক', 'দুষ্কর']
    Truncating StopWords: ['বর্তমান', 'নাম্বরি', 'ই-কমার্স', 'মধ্য', 'ভালো', 'ট্রাস্টেট', 'একটা', 'ই-কমার্স', 'খুঁজে', 'দুষ্কর']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট গুলো এইমাত্র হাতে পেলাম আর প্যাকেজিং টা ছিলো অসাধারণ
    Afert Tokenizing:  ['প্রোডাক্ট', 'গুলো', 'এইমাত্র', 'হাতে', 'পেলাম', 'আর', 'প্যাকেজিং', 'টা', 'ছিলো', 'অসাধারণ']
    Truncating punctuation: ['প্রোডাক্ট', 'গুলো', 'এইমাত্র', 'হাতে', 'পেলাম', 'আর', 'প্যাকেজিং', 'টা', 'ছিলো', 'অসাধারণ']
    Truncating StopWords: ['প্রোডাক্ট', 'গুলো', 'এইমাত্র', 'হাতে', 'পেলাম', 'প্যাকেজিং', 'টা', 'ছিলো', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  আশা করবো ভবিষ্যতে  আপনাদের কোয়ালিটি অক্ষুণ্ণ থাকবে
    Afert Tokenizing:  ['আশা', 'করবো', 'ভবিষ্যতে', 'আপনাদের', 'কোয়ালিটি', 'অক্ষুণ্ণ', 'থাকবে']
    Truncating punctuation: ['আশা', 'করবো', 'ভবিষ্যতে', 'আপনাদের', 'কোয়ালিটি', 'অক্ষুণ্ণ', 'থাকবে']
    Truncating StopWords: ['আশা', 'করবো', 'ভবিষ্যতে', 'আপনাদের', 'কোয়ালিটি', 'অক্ষুণ্ণ']
    ***************************************************************************************
    Label:  1
    Sentence:  দোয়া করি আপনাদের জন্য.প্যান্ট দুটি আমার মনের মত হয়েছে কাপরের মান ও ভালো
    Afert Tokenizing:  ['দোয়া', 'করি', 'আপনাদের', 'জন্য.প্যান্ট', 'দুটি', 'আমার', 'মনের', 'মত', 'হয়েছে', 'কাপরের', 'মান', 'ও', 'ভালো']
    Truncating punctuation: ['দোয়া', 'করি', 'আপনাদের', 'জন্য.প্যান্ট', 'দুটি', 'আমার', 'মনের', 'মত', 'হয়েছে', 'কাপরের', 'মান', 'ও', 'ভালো']
    Truncating StopWords: ['দোয়া', 'আপনাদের', 'জন্য.প্যান্ট', 'মনের', 'মত', 'হয়েছে', 'কাপরের', 'মান', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  ওনাদের সিস্টেম খুব ভালো। আমিও একবার পন্য চেইঞ্জ করেছিলাম। আরো অর্ডার দিবো আমার কাজের লোকদের জন্য।
    Afert Tokenizing:  ['ওনাদের', 'সিস্টেম', 'খুব', 'ভালো', '।', 'আমিও', 'একবার', 'পন্য', 'চেইঞ্জ', 'করেছিলাম', '।', 'আরো', 'অর্ডার', 'দিবো', 'আমার', 'কাজের', 'লোকদের', 'জন্য', '।']
    Truncating punctuation: ['ওনাদের', 'সিস্টেম', 'খুব', 'ভালো', 'আমিও', 'একবার', 'পন্য', 'চেইঞ্জ', 'করেছিলাম', 'আরো', 'অর্ডার', 'দিবো', 'আমার', 'কাজের', 'লোকদের', 'জন্য']
    Truncating StopWords: ['ওনাদের', 'সিস্টেম', 'ভালো', 'আমিও', 'পন্য', 'চেইঞ্জ', 'করেছিলাম', 'আরো', 'অর্ডার', 'দিবো', 'কাজের', 'লোকদের']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্যের গুণগত মান খুবই ভালো এত অল্প টাকায় এত ভালো জিনিস পাবো ভাবতেও পারিনি ☺ চোখ বন্ধ করে ১০ এ ১০ মার্ক দেয়ায় যায়
    Afert Tokenizing:  ['পণ্যের', 'গুণগত', 'মান', 'খুবই', 'ভালো', 'এত', 'অল্প', 'টাকায়', 'এত', 'ভালো', 'জিনিস', 'পাবো', 'ভাবতেও', 'পারিনি', '☺', 'চোখ', 'বন্ধ', 'করে', '১০', 'এ', '১০', 'মার্ক', 'দেয়ায়', 'যায়']
    Truncating punctuation: ['পণ্যের', 'গুণগত', 'মান', 'খুবই', 'ভালো', 'এত', 'অল্প', 'টাকায়', 'এত', 'ভালো', 'জিনিস', 'পাবো', 'ভাবতেও', 'পারিনি', '☺', 'চোখ', 'বন্ধ', 'করে', '১০', 'এ', '১০', 'মার্ক', 'দেয়ায়', 'যায়']
    Truncating StopWords: ['পণ্যের', 'গুণগত', 'মান', 'খুবই', 'ভালো', 'অল্প', 'টাকায়', 'ভালো', 'জিনিস', 'পাবো', 'ভাবতেও', 'পারিনি', '☺', 'চোখ', 'বন্ধ', '১০', '১০', 'মার্ক', 'দেয়ায়', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  এইগুলা চোর বাটপার
    Afert Tokenizing:  ['এইগুলা', 'চোর', 'বাটপার']
    Truncating punctuation: ['এইগুলা', 'চোর', 'বাটপার']
    Truncating StopWords: ['এইগুলা', 'চোর', 'বাটপার']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম ফালতু, 500টাকার প্যান্ট বেচে 1200টাকায়, ইসলামী নাম দেখে কেউ বিভ্রান্ত হয়েন্না প্লীজ।
    Afert Tokenizing:  ['একদম', 'ফালতু', ',', '500টাকার', 'প্যান্ট', 'বেচে', '1200টাকায়', ',', 'ইসলামী', 'নাম', 'দেখে', 'কেউ', 'বিভ্রান্ত', 'হয়েন্না', 'প্লীজ', '।']
    Truncating punctuation: ['একদম', 'ফালতু', '500টাকার', 'প্যান্ট', 'বেচে', '1200টাকায়', 'ইসলামী', 'নাম', 'দেখে', 'কেউ', 'বিভ্রান্ত', 'হয়েন্না', 'প্লীজ']
    Truncating StopWords: ['একদম', 'ফালতু', '500টাকার', 'প্যান্ট', 'বেচে', '1200টাকায়', 'ইসলামী', 'নাম', 'বিভ্রান্ত', 'হয়েন্না', 'প্লীজ']
    ***************************************************************************************
    Label:  0
    Sentence:  সমস্যা হচ্ছে লো কোয়ালিটি ম্যাটারিয়াল ইউস করেন আপনারা।
    Afert Tokenizing:  ['সমস্যা', 'হচ্ছে', 'লো', 'কোয়ালিটি', 'ম্যাটারিয়াল', 'ইউস', 'করেন', 'আপনারা', '।']
    Truncating punctuation: ['সমস্যা', 'হচ্ছে', 'লো', 'কোয়ালিটি', 'ম্যাটারিয়াল', 'ইউস', 'করেন', 'আপনারা']
    Truncating StopWords: ['সমস্যা', 'লো', 'কোয়ালিটি', 'ম্যাটারিয়াল', 'ইউস', 'আপনারা']
    ***************************************************************************************
    Label:  0
    Sentence:  মারকেটিং ভালো আপনাদের বাট কাপরের ভ্যারাইটি নাই।
    Afert Tokenizing:  ['মারকেটিং', 'ভালো', 'আপনাদের', 'বাট', 'কাপরের', 'ভ্যারাইটি', 'নাই', '।']
    Truncating punctuation: ['মারকেটিং', 'ভালো', 'আপনাদের', 'বাট', 'কাপরের', 'ভ্যারাইটি', 'নাই']
    Truncating StopWords: ['মারকেটিং', 'ভালো', 'আপনাদের', 'বাট', 'কাপরের', 'ভ্যারাইটি', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের থেকে কেনাকাটা করে এক বারো ঠকিনি। ধন্যবাদ আপনাদের
    Afert Tokenizing:  ['আপনাদের', 'থেকে', 'কেনাকাটা', 'করে', 'এক', 'বারো', 'ঠকিনি', '।', 'ধন্যবাদ', 'আপনাদের']
    Truncating punctuation: ['আপনাদের', 'থেকে', 'কেনাকাটা', 'করে', 'এক', 'বারো', 'ঠকিনি', 'ধন্যবাদ', 'আপনাদের']
    Truncating StopWords: ['আপনাদের', 'কেনাকাটা', 'এক', 'বারো', 'ঠকিনি', 'ধন্যবাদ', 'আপনাদের']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি মাধ্যমটা ভালো নয়
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'মাধ্যমটা', 'ভালো', 'নয়']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'মাধ্যমটা', 'ভালো', 'নয়']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'মাধ্যমটা', 'ভালো', 'নয়']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি আমার খুব পছন্দ হয়েছে। আপনাদের জন্য শুভকামনা রইল।
    Afert Tokenizing:  ['কোয়ালিটি', 'আমার', 'খুব', 'পছন্দ', 'হয়েছে', '।', 'আপনাদের', 'জন্য', 'শুভকামনা', 'রইল', '।']
    Truncating punctuation: ['কোয়ালিটি', 'আমার', 'খুব', 'পছন্দ', 'হয়েছে', 'আপনাদের', 'জন্য', 'শুভকামনা', 'রইল']
    Truncating StopWords: ['কোয়ালিটি', 'পছন্দ', 'হয়েছে', 'আপনাদের', 'শুভকামনা', 'রইল']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ যেরকম চেয়েছি ঠিক সে রকমই পেয়েছি
    Afert Tokenizing:  ['ধন্যবাদ', 'যেরকম', 'চেয়েছি', 'ঠিক', 'সে', 'রকমই', 'পেয়েছি']
    Truncating punctuation: ['ধন্যবাদ', 'যেরকম', 'চেয়েছি', 'ঠিক', 'সে', 'রকমই', 'পেয়েছি']
    Truncating StopWords: ['ধন্যবাদ', 'যেরকম', 'চেয়েছি', 'ঠিক', 'রকমই', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  সবচেয়ে ভালো দিক ছিল ডেলিভারি খুব দ্রুত এবং প্যাকেজিং খুবই সুন্দর আল্লাহ আপনাদের মঙ্গল করুক
    Afert Tokenizing:  ['সবচেয়ে', 'ভালো', 'দিক', 'ছিল', 'ডেলিভারি', 'খুব', 'দ্রুত', 'এবং', 'প্যাকেজিং', 'খুবই', 'সুন্দর', 'আল্লাহ', 'আপনাদের', 'মঙ্গল', 'করুক']
    Truncating punctuation: ['সবচেয়ে', 'ভালো', 'দিক', 'ছিল', 'ডেলিভারি', 'খুব', 'দ্রুত', 'এবং', 'প্যাকেজিং', 'খুবই', 'সুন্দর', 'আল্লাহ', 'আপনাদের', 'মঙ্গল', 'করুক']
    Truncating StopWords: ['সবচেয়ে', 'ভালো', 'দিক', 'ডেলিভারি', 'দ্রুত', 'প্যাকেজিং', 'খুবই', 'সুন্দর', 'আল্লাহ', 'আপনাদের', 'মঙ্গল', 'করুক']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটির জন্য ধন্যবাদ
    Afert Tokenizing:  ['কোয়ালিটির', 'জন্য', 'ধন্যবাদ']
    Truncating punctuation: ['কোয়ালিটির', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['কোয়ালিটির', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  পারফেক্ট
    Afert Tokenizing:  ['পারফেক্ট']
    Truncating punctuation: ['পারফেক্ট']
    Truncating StopWords: ['পারফেক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো মানের প্রোডাক্ট দেওয়ার জন্য অনেক অনেক ধন্যবাদ আপনাদেরকে,অনেক দূর এগিয়ে যাক
    Afert Tokenizing:  ['ভালো', 'মানের', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য', 'অনেক', 'অনেক', 'ধন্যবাদ', 'আপনাদেরকে,অনেক', 'দূর', 'এগিয়ে', 'যাক']
    Truncating punctuation: ['ভালো', 'মানের', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য', 'অনেক', 'অনেক', 'ধন্যবাদ', 'আপনাদেরকে,অনেক', 'দূর', 'এগিয়ে', 'যাক']
    Truncating StopWords: ['ভালো', 'মানের', 'প্রোডাক্ট', 'দেওয়ার', 'ধন্যবাদ', 'আপনাদেরকে,অনেক', 'দূর', 'এগিয়ে', 'যাক']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি প্রমান পেয়েছি, আপনাদের কথায় আর কাজে মিল আছে।আপনাদের মঙ্গল কামনা করি এবং সাথেই আছি,ধন্যবাদ।
    Afert Tokenizing:  ['আমি', 'প্রমান', 'পেয়েছি', ',', 'আপনাদের', 'কথায়', 'আর', 'কাজে', 'মিল', 'আছে।আপনাদের', 'মঙ্গল', 'কামনা', 'করি', 'এবং', 'সাথেই', 'আছি,ধন্যবাদ', '।']
    Truncating punctuation: ['আমি', 'প্রমান', 'পেয়েছি', 'আপনাদের', 'কথায়', 'আর', 'কাজে', 'মিল', 'আছে।আপনাদের', 'মঙ্গল', 'কামনা', 'করি', 'এবং', 'সাথেই', 'আছি,ধন্যবাদ']
    Truncating StopWords: ['প্রমান', 'পেয়েছি', 'আপনাদের', 'কথায়', 'মিল', 'আছে।আপনাদের', 'মঙ্গল', 'কামনা', 'সাথেই', 'আছি,ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  সবাই সুন্দর সুন্দর কথা বলে কিন্তু টাকার লোভ কেউ সামলাতে পারে না।
    Afert Tokenizing:  ['সবাই', 'সুন্দর', 'সুন্দর', 'কথা', 'বলে', 'কিন্তু', 'টাকার', 'লোভ', 'কেউ', 'সামলাতে', 'পারে', 'না', '।']
    Truncating punctuation: ['সবাই', 'সুন্দর', 'সুন্দর', 'কথা', 'বলে', 'কিন্তু', 'টাকার', 'লোভ', 'কেউ', 'সামলাতে', 'পারে', 'না']
    Truncating StopWords: ['সবাই', 'সুন্দর', 'সুন্দর', 'কথা', 'টাকার', 'লোভ', 'সামলাতে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যি রিস্ক নাই। আপনাদের সার্ভিস বেস্ট
    Afert Tokenizing:  ['সত্যি', 'রিস্ক', 'নাই', '।', 'আপনাদের', 'সার্ভিস', 'বেস্ট']
    Truncating punctuation: ['সত্যি', 'রিস্ক', 'নাই', 'আপনাদের', 'সার্ভিস', 'বেস্ট']
    Truncating StopWords: ['সত্যি', 'রিস্ক', 'নাই', 'আপনাদের', 'সার্ভিস', 'বেস্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম টা খুবই বেশি
    Afert Tokenizing:  ['দাম', 'টা', 'খুবই', 'বেশি']
    Truncating punctuation: ['দাম', 'টা', 'খুবই', 'বেশি']
    Truncating StopWords: ['দাম', 'টা', 'খুবই', 'বেশি']
    ***************************************************************************************
    Label:  0
    Sentence:  নতুন প্রোডাক্ট আনেন।
    Afert Tokenizing:  ['নতুন', 'প্রোডাক্ট', 'আনেন', '।']
    Truncating punctuation: ['নতুন', 'প্রোডাক্ট', 'আনেন']
    Truncating StopWords: ['প্রোডাক্ট', 'আনেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার কেনো যেনো অনলাইন প্রডাক্টে আস্থা হয় না।কিন্তু এই নিয়ে ২য় বার Deen থেকে টি-শার্ট  নিলাম আর আলহামদুলিল্লাহ নিরাশ হইনি।
    Afert Tokenizing:  ['আমার', 'কেনো', 'যেনো', 'অনলাইন', 'প্রডাক্টে', 'আস্থা', 'হয়', 'না।কিন্তু', 'এই', 'নিয়ে', '২য়', 'বার', 'Deen', 'থেকে', 'টি-শার্ট', 'নিলাম', 'আর', 'আলহামদুলিল্লাহ', 'নিরাশ', 'হইনি', '।']
    Truncating punctuation: ['আমার', 'কেনো', 'যেনো', 'অনলাইন', 'প্রডাক্টে', 'আস্থা', 'হয়', 'না।কিন্তু', 'এই', 'নিয়ে', '২য়', 'বার', 'Deen', 'থেকে', 'টি-শার্ট', 'নিলাম', 'আর', 'আলহামদুলিল্লাহ', 'নিরাশ', 'হইনি']
    Truncating StopWords: ['কেনো', 'যেনো', 'অনলাইন', 'প্রডাক্টে', 'আস্থা', 'না।কিন্তু', '২য়', 'Deen', 'টি-শার্ট', 'নিলাম', 'আলহামদুলিল্লাহ', 'নিরাশ', 'হইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ, কোয়ালিটি অনেক ভালো।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', ',', 'কোয়ালিটি', 'অনেক', 'ভালো', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'কোয়ালিটি', 'অনেক', 'ভালো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'কোয়ালিটি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রডাক্ট অনেক ভালো এবং মানসম্মত।ধারাবাহিক ভাবে আপনারা মানসম্মত প্রডাক্ট ডেলিভারি দিলে অচিরেই আপনাদের সাফল্য পেয়ে যাবেন।
    Afert Tokenizing:  ['আপনাদের', 'প্রডাক্ট', 'অনেক', 'ভালো', 'এবং', 'মানসম্মত।ধারাবাহিক', 'ভাবে', 'আপনারা', 'মানসম্মত', 'প্রডাক্ট', 'ডেলিভারি', 'দিলে', 'অচিরেই', 'আপনাদের', 'সাফল্য', 'পেয়ে', 'যাবেন', '।']
    Truncating punctuation: ['আপনাদের', 'প্রডাক্ট', 'অনেক', 'ভালো', 'এবং', 'মানসম্মত।ধারাবাহিক', 'ভাবে', 'আপনারা', 'মানসম্মত', 'প্রডাক্ট', 'ডেলিভারি', 'দিলে', 'অচিরেই', 'আপনাদের', 'সাফল্য', 'পেয়ে', 'যাবেন']
    Truncating StopWords: ['আপনাদের', 'প্রডাক্ট', 'ভালো', 'মানসম্মত।ধারাবাহিক', 'আপনারা', 'মানসম্মত', 'প্রডাক্ট', 'ডেলিভারি', 'দিলে', 'অচিরেই', 'আপনাদের', 'সাফল্য', 'পেয়ে', 'যাবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ২৪ ঘন্টারও কম সময়ের মধ্যে ডেলিভারি পেলাম। কোয়ালিটি নিয়ে সন্তুষ্ট, আলহামদুলিল্লাহ।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '২৪', 'ঘন্টারও', 'কম', 'সময়ের', 'মধ্যে', 'ডেলিভারি', 'পেলাম', '।', 'কোয়ালিটি', 'নিয়ে', 'সন্তুষ্ট', ',', 'আলহামদুলিল্লাহ', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', '২৪', 'ঘন্টারও', 'কম', 'সময়ের', 'মধ্যে', 'ডেলিভারি', 'পেলাম', 'কোয়ালিটি', 'নিয়ে', 'সন্তুষ্ট', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['আলহামদুলিল্লাহ', '২৪', 'ঘন্টারও', 'কম', 'সময়ের', 'ডেলিভারি', 'পেলাম', 'কোয়ালিটি', 'সন্তুষ্ট', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো লাগছে পন্য টি হাতে পেয়ে।।।
    Afert Tokenizing:  ['খুব', 'ভালো', 'লাগছে', 'পন্য', 'টি', 'হাতে', 'পেয়ে।।', '।']
    Truncating punctuation: ['খুব', 'ভালো', 'লাগছে', 'পন্য', 'টি', 'হাতে', 'পেয়ে।।']
    Truncating StopWords: ['ভালো', 'লাগছে', 'পন্য', 'হাতে', 'পেয়ে।।']
    ***************************************************************************************
    Label:  1
    Sentence:  অালহামদু-লিল্লাহ,,,,  সবগুলো প্রডাক্ট ঠিকঠাক পেয়েছি অামি সন্তুষ্ট....!!
    Afert Tokenizing:  ['অালহামদু-লিল্লাহ,,,', ',', 'সবগুলো', 'প্রডাক্ট', 'ঠিকঠাক', 'পেয়েছি', 'অামি', 'সন্তুষ্ট....!', '!']
    Truncating punctuation: ['অালহামদু-লিল্লাহ,,,', 'সবগুলো', 'প্রডাক্ট', 'ঠিকঠাক', 'পেয়েছি', 'অামি', 'সন্তুষ্ট....!']
    Truncating StopWords: ['অালহামদু-লিল্লাহ,,,', 'সবগুলো', 'প্রডাক্ট', 'ঠিকঠাক', 'পেয়েছি', 'অামি', 'সন্তুষ্ট....!']
    ***************************************************************************************
    Label:  0
    Sentence:  আজকে সারাদিন কল করে,মেসেঞ্জারে নক করেও আপনাদের সাড়া পেলাম না,বিষয়টি দুঃখজনক।
    Afert Tokenizing:  ['আজকে', 'সারাদিন', 'কল', 'করে,মেসেঞ্জারে', 'নক', 'করেও', 'আপনাদের', 'সাড়া', 'পেলাম', 'না,বিষয়টি', 'দুঃখজনক', '।']
    Truncating punctuation: ['আজকে', 'সারাদিন', 'কল', 'করে,মেসেঞ্জারে', 'নক', 'করেও', 'আপনাদের', 'সাড়া', 'পেলাম', 'না,বিষয়টি', 'দুঃখজনক']
    Truncating StopWords: ['আজকে', 'সারাদিন', 'কল', 'করে,মেসেঞ্জারে', 'নক', 'করেও', 'আপনাদের', 'সাড়া', 'পেলাম', 'না,বিষয়টি', 'দুঃখজনক']
    ***************************************************************************************
    Label:  1
    Sentence:  পরবর্তীতে আবারো অর্ডার করার ইচ্ছা আছে।ধন্যবাদ
    Afert Tokenizing:  ['পরবর্তীতে', 'আবারো', 'অর্ডার', 'করার', 'ইচ্ছা', 'আছে।ধন্যবাদ']
    Truncating punctuation: ['পরবর্তীতে', 'আবারো', 'অর্ডার', 'করার', 'ইচ্ছা', 'আছে।ধন্যবাদ']
    Truncating StopWords: ['পরবর্তীতে', 'আবারো', 'অর্ডার', 'ইচ্ছা', 'আছে।ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  এই নিয়ে দ্বিতীয়বারের মত কেনাকাটা। প্যাকিং ভাল আর দ্রুততম সময়ে ডেলিভারি আর প্রোডাক্ট কোয়ালিটি ভাল পেয়ে বরাবরের মতোই সন্তুষ্ট
    Afert Tokenizing:  ['এই', 'নিয়ে', 'দ্বিতীয়বারের', 'মত', 'কেনাকাটা', '।', 'প্যাকিং', 'ভাল', 'আর', 'দ্রুততম', 'সময়ে', 'ডেলিভারি', 'আর', 'প্রোডাক্ট', 'কোয়ালিটি', 'ভাল', 'পেয়ে', 'বরাবরের', 'মতোই', 'সন্তুষ্ট']
    Truncating punctuation: ['এই', 'নিয়ে', 'দ্বিতীয়বারের', 'মত', 'কেনাকাটা', 'প্যাকিং', 'ভাল', 'আর', 'দ্রুততম', 'সময়ে', 'ডেলিভারি', 'আর', 'প্রোডাক্ট', 'কোয়ালিটি', 'ভাল', 'পেয়ে', 'বরাবরের', 'মতোই', 'সন্তুষ্ট']
    Truncating StopWords: ['দ্বিতীয়বারের', 'মত', 'কেনাকাটা', 'প্যাকিং', 'ভাল', 'দ্রুততম', 'সময়ে', 'ডেলিভারি', 'প্রোডাক্ট', 'কোয়ালিটি', 'ভাল', 'পেয়ে', 'বরাবরের', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আপনাদের সাইটে তো ডুকাই যাচ্ছে না। প্রচুর লোডিং নিচ্ছে
    Afert Tokenizing:  ['ভাই', 'আপনাদের', 'সাইটে', 'তো', 'ডুকাই', 'যাচ্ছে', 'না', '।', 'প্রচুর', 'লোডিং', 'নিচ্ছে']
    Truncating punctuation: ['ভাই', 'আপনাদের', 'সাইটে', 'তো', 'ডুকাই', 'যাচ্ছে', 'না', 'প্রচুর', 'লোডিং', 'নিচ্ছে']
    Truncating StopWords: ['ভাই', 'আপনাদের', 'সাইটে', 'ডুকাই', 'না', 'প্রচুর', 'লোডিং', 'নিচ্ছে']
    ***************************************************************************************
    Label:  1
    Sentence:  বিশ্বাসের মাত্রাটা বেড়ে গেছে
    Afert Tokenizing:  ['বিশ্বাসের', 'মাত্রাটা', 'বেড়ে', 'গেছে']
    Truncating punctuation: ['বিশ্বাসের', 'মাত্রাটা', 'বেড়ে', 'গেছে']
    Truncating StopWords: ['বিশ্বাসের', 'মাত্রাটা', 'বেড়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ  ডেলিভারি খুব ফাস্ট পেয়েছি  প্রোডাক্ট ও খুব ভালো।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'ডেলিভারি', 'খুব', 'ফাস্ট', 'পেয়েছি', 'প্রোডাক্ট', 'ও', 'খুব', 'ভালো', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ডেলিভারি', 'খুব', 'ফাস্ট', 'পেয়েছি', 'প্রোডাক্ট', 'ও', 'খুব', 'ভালো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ডেলিভারি', 'ফাস্ট', 'পেয়েছি', 'প্রোডাক্ট', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  এই মাস্ক গুলো যদি একটা ৫০ টাকা হয় তাহলে আপনারা টাকা বেশি নিচ্ছেন বলে মনে হয় না আপনাদের কাছে ।এইটা কোন ধরনের প্রতারণা।
    Afert Tokenizing:  ['এই', 'মাস্ক', 'গুলো', 'যদি', 'একটা', '৫০', 'টাকা', 'হয়', 'তাহলে', 'আপনারা', 'টাকা', 'বেশি', 'নিচ্ছেন', 'বলে', 'মনে', 'হয়', 'না', 'আপনাদের', 'কাছে', 'এইটা', '।', 'কোন', 'ধরনের', 'প্রতারণা', '।']
    Truncating punctuation: ['এই', 'মাস্ক', 'গুলো', 'যদি', 'একটা', '৫০', 'টাকা', 'হয়', 'তাহলে', 'আপনারা', 'টাকা', 'বেশি', 'নিচ্ছেন', 'বলে', 'মনে', 'হয়', 'না', 'আপনাদের', 'কাছে', 'এইটা', 'কোন', 'ধরনের', 'প্রতারণা']
    Truncating StopWords: ['মাস্ক', 'গুলো', 'একটা', '৫০', 'টাকা', 'আপনারা', 'টাকা', 'বেশি', 'নিচ্ছেন', 'না', 'আপনাদের', 'এইটা', 'ধরনের', 'প্রতারণা']
    ***************************************************************************************
    Label:  1
    Sentence:  এ রকম সবগুলা ওয়েবসাইট যদি প্রডাক্ট সাপ্লাই দিত তাহলে আমরা কাষ্টমার যারা অনলাইনে ওর্ডার করে বার বার ঠকি তাদের এই ভ্রান্ত ধারনা দূর হত
    Afert Tokenizing:  ['এ', 'রকম', 'সবগুলা', 'ওয়েবসাইট', 'যদি', 'প্রডাক্ট', 'সাপ্লাই', 'দিত', 'তাহলে', 'আমরা', 'কাষ্টমার', 'যারা', 'অনলাইনে', 'ওর্ডার', 'করে', 'বার', 'বার', 'ঠকি', 'তাদের', 'এই', 'ভ্রান্ত', 'ধারনা', 'দূর', 'হত']
    Truncating punctuation: ['এ', 'রকম', 'সবগুলা', 'ওয়েবসাইট', 'যদি', 'প্রডাক্ট', 'সাপ্লাই', 'দিত', 'তাহলে', 'আমরা', 'কাষ্টমার', 'যারা', 'অনলাইনে', 'ওর্ডার', 'করে', 'বার', 'বার', 'ঠকি', 'তাদের', 'এই', 'ভ্রান্ত', 'ধারনা', 'দূর', 'হত']
    Truncating StopWords: ['সবগুলা', 'ওয়েবসাইট', 'প্রডাক্ট', 'সাপ্লাই', 'দিত', 'কাষ্টমার', 'অনলাইনে', 'ওর্ডার', 'ঠকি', 'ভ্রান্ত', 'ধারনা', 'দূর']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট এর পরিমান আরও বাড়াতে হবে। আইটেম একেবারেই কম।
    Afert Tokenizing:  ['প্রোডাক্ট', 'এর', 'পরিমান', 'আরও', 'বাড়াতে', 'হবে', '।', 'আইটেম', 'একেবারেই', 'কম', '।']
    Truncating punctuation: ['প্রোডাক্ট', 'এর', 'পরিমান', 'আরও', 'বাড়াতে', 'হবে', 'আইটেম', 'একেবারেই', 'কম']
    Truncating StopWords: ['প্রোডাক্ট', 'পরিমান', 'বাড়াতে', 'আইটেম', 'একেবারেই', 'কম']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ওয়েবসাইট টা একটু ইম্প্রুভ করেন। প্র-চ-ন্ড স্লো।
    Afert Tokenizing:  ['আপনাদের', 'ওয়েবসাইট', 'টা', 'একটু', 'ইম্প্রুভ', 'করেন', '।', 'প্র-চ-ন্ড', 'স্লো', '।']
    Truncating punctuation: ['আপনাদের', 'ওয়েবসাইট', 'টা', 'একটু', 'ইম্প্রুভ', 'করেন', 'প্র-চ-ন্ড', 'স্লো']
    Truncating StopWords: ['আপনাদের', 'ওয়েবসাইট', 'টা', 'একটু', 'ইম্প্রুভ', 'প্র-চ-ন্ড', 'স্লো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রথম অর্ডার ছিল তাই একটু ভয়ে ছিলাম, কিন্তু যখন প্রেডাক্ট হাতে পেলাম সব ভয় কেটে গেল।
    Afert Tokenizing:  ['প্রথম', 'অর্ডার', 'ছিল', 'তাই', 'একটু', 'ভয়ে', 'ছিলাম', ',', 'কিন্তু', 'যখন', 'প্রেডাক্ট', 'হাতে', 'পেলাম', 'সব', 'ভয়', 'কেটে', 'গেল', '।']
    Truncating punctuation: ['প্রথম', 'অর্ডার', 'ছিল', 'তাই', 'একটু', 'ভয়ে', 'ছিলাম', 'কিন্তু', 'যখন', 'প্রেডাক্ট', 'হাতে', 'পেলাম', 'সব', 'ভয়', 'কেটে', 'গেল']
    Truncating StopWords: ['অর্ডার', 'একটু', 'ভয়ে', 'ছিলাম', 'প্রেডাক্ট', 'হাতে', 'পেলাম', 'ভয়', 'কেটে']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম আরও একটু কম রাখা উচিৎ।
    Afert Tokenizing:  ['দাম', 'আরও', 'একটু', 'কম', 'রাখা', 'উচিৎ', '।']
    Truncating punctuation: ['দাম', 'আরও', 'একটু', 'কম', 'রাখা', 'উচিৎ']
    Truncating StopWords: ['দাম', 'একটু', 'কম', 'উচিৎ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ প্রোডাক্ট হাতে পেয়েছি ধন্যবাদ আপনাদেরকে।  কথা এবং কাজে আপনাদের অনেক মিল আছে আর প্রোডাক্টের  ব্যাপারে কি বলব এক কথায়  অসাধারণ।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'ধন্যবাদ', 'আপনাদেরকে', '।', 'কথা', 'এবং', 'কাজে', 'আপনাদের', 'অনেক', 'মিল', 'আছে', 'আর', 'প্রোডাক্টের', 'ব্যাপারে', 'কি', 'বলব', 'এক', 'কথায়', 'অসাধারণ', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'ধন্যবাদ', 'আপনাদেরকে', 'কথা', 'এবং', 'কাজে', 'আপনাদের', 'অনেক', 'মিল', 'আছে', 'আর', 'প্রোডাক্টের', 'ব্যাপারে', 'কি', 'বলব', 'এক', 'কথায়', 'অসাধারণ']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'ধন্যবাদ', 'আপনাদেরকে', 'কথা', 'আপনাদের', 'মিল', 'প্রোডাক্টের', 'বলব', 'এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় প্রাইজ রেঞ্জের মধ্যে খুব ভালো প্রডাক্ট। আপনাদের উজ্জ্বল ভবিষ্যৎ কামনা করি।
    Afert Tokenizing:  ['এক', 'কথায়', 'প্রাইজ', 'রেঞ্জের', 'মধ্যে', 'খুব', 'ভালো', 'প্রডাক্ট', '।', 'আপনাদের', 'উজ্জ্বল', 'ভবিষ্যৎ', 'কামনা', 'করি', '।']
    Truncating punctuation: ['এক', 'কথায়', 'প্রাইজ', 'রেঞ্জের', 'মধ্যে', 'খুব', 'ভালো', 'প্রডাক্ট', 'আপনাদের', 'উজ্জ্বল', 'ভবিষ্যৎ', 'কামনা', 'করি']
    Truncating StopWords: ['এক', 'কথায়', 'প্রাইজ', 'রেঞ্জের', 'ভালো', 'প্রডাক্ট', 'আপনাদের', 'উজ্জ্বল', 'ভবিষ্যৎ', 'কামনা']
    ***************************************************************************************
    Label:  1
    Sentence:  দ্রুত ডেলিভারি পেয়েছি।ধন্যবাদ
    Afert Tokenizing:  ['দ্রুত', 'ডেলিভারি', 'পেয়েছি।ধন্যবাদ']
    Truncating punctuation: ['দ্রুত', 'ডেলিভারি', 'পেয়েছি।ধন্যবাদ']
    Truncating StopWords: ['দ্রুত', 'ডেলিভারি', 'পেয়েছি।ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ  প্রোডাক্ট অনেক ভাল ছিল আমি যেমন ছিলাম ঠিক তেমনই পেয়েছি
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'অনেক', 'ভাল', 'ছিল', 'আমি', 'যেমন', 'ছিলাম', 'ঠিক', 'তেমনই', 'পেয়েছি']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'অনেক', 'ভাল', 'ছিল', 'আমি', 'যেমন', 'ছিলাম', 'ঠিক', 'তেমনই', 'পেয়েছি']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'প্রোডাক্ট', 'ভাল', 'ছিলাম', 'ঠিক', 'তেমনই', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাই আপনাদের পন্য ভালো আর আপনারা প্রতারক নন।
    Afert Tokenizing:  ['ভাই', 'আপনাদের', 'পন্য', 'ভালো', 'আর', 'আপনারা', 'প্রতারক', 'নন', '।']
    Truncating punctuation: ['ভাই', 'আপনাদের', 'পন্য', 'ভালো', 'আর', 'আপনারা', 'প্রতারক', 'নন']
    Truncating StopWords: ['ভাই', 'আপনাদের', 'পন্য', 'ভালো', 'আপনারা', 'প্রতারক', 'নন']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদে একটা অর্ডার দিছিলাম গত মাসে এখনো কোন খবর নাই,আমি প্রোডাক্ট টা কবে পেতে পারি একটু জানাবেন??
    Afert Tokenizing:  ['আপনাদে', 'একটা', 'অর্ডার', 'দিছিলাম', 'গত', 'মাসে', 'এখনো', 'কোন', 'খবর', 'নাই,আমি', 'প্রোডাক্ট', 'টা', 'কবে', 'পেতে', 'পারি', 'একটু', 'জানাবেন?', '?']
    Truncating punctuation: ['আপনাদে', 'একটা', 'অর্ডার', 'দিছিলাম', 'গত', 'মাসে', 'এখনো', 'কোন', 'খবর', 'নাই,আমি', 'প্রোডাক্ট', 'টা', 'কবে', 'পেতে', 'পারি', 'একটু', 'জানাবেন?']
    Truncating StopWords: ['আপনাদে', 'একটা', 'অর্ডার', 'দিছিলাম', 'গত', 'মাসে', 'এখনো', 'খবর', 'নাই,আমি', 'প্রোডাক্ট', 'টা', 'পেতে', 'একটু', 'জানাবেন?']
    ***************************************************************************************
    Label:  0
    Sentence:  জুনের ১৬ তারিখে অর্ডার দিয়েছি। আপনাদের এখনও কোন খবর নাই। এস‌এম‌এস করলে কোন রিপ্লাই দেন না।
    Afert Tokenizing:  ['জুনের', '১৬', 'তারিখে', 'অর্ডার', 'দিয়েছি', '।', 'আপনাদের', 'এখনও', 'কোন', 'খবর', 'নাই', '।', 'এস\u200cএম\u200cএস', 'করলে', 'কোন', 'রিপ্লাই', 'দেন', 'না', '।']
    Truncating punctuation: ['জুনের', '১৬', 'তারিখে', 'অর্ডার', 'দিয়েছি', 'আপনাদের', 'এখনও', 'কোন', 'খবর', 'নাই', 'এস\u200cএম\u200cএস', 'করলে', 'কোন', 'রিপ্লাই', 'দেন', 'না']
    Truncating StopWords: ['জুনের', '১৬', 'তারিখে', 'অর্ডার', 'দিয়েছি', 'আপনাদের', 'খবর', 'নাই', 'এস\u200cএম\u200cএস', 'রিপ্লাই', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  এত রিজেনবল প্রাইসে অরিজিনাল লেদার। আর সত্যি অনেক আরামদায়ক আপনাদের প্রডাক্ট
    Afert Tokenizing:  ['এত', 'রিজেনবল', 'প্রাইসে', 'অরিজিনাল', 'লেদার', '।', 'আর', 'সত্যি', 'অনেক', 'আরামদায়ক', 'আপনাদের', 'প্রডাক্ট']
    Truncating punctuation: ['এত', 'রিজেনবল', 'প্রাইসে', 'অরিজিনাল', 'লেদার', 'আর', 'সত্যি', 'অনেক', 'আরামদায়ক', 'আপনাদের', 'প্রডাক্ট']
    Truncating StopWords: ['রিজেনবল', 'প্রাইসে', 'অরিজিনাল', 'লেদার', 'সত্যি', 'আরামদায়ক', 'আপনাদের', 'প্রডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের লোকেশন অনুযায়ী গিয়ে প্রতারিত হয়েছি
    Afert Tokenizing:  ['আপনাদের', 'লোকেশন', 'অনুযায়ী', 'গিয়ে', 'প্রতারিত', 'হয়েছি']
    Truncating punctuation: ['আপনাদের', 'লোকেশন', 'অনুযায়ী', 'গিয়ে', 'প্রতারিত', 'হয়েছি']
    Truncating StopWords: ['আপনাদের', 'লোকেশন', 'অনুযায়ী', 'প্রতারিত', 'হয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যি অনেক সুন্দর প্রোডাক্ট
    Afert Tokenizing:  ['সত্যি', 'অনেক', 'সুন্দর', 'প্রোডাক্ট']
    Truncating punctuation: ['সত্যি', 'অনেক', 'সুন্দর', 'প্রোডাক্ট']
    Truncating StopWords: ['সত্যি', 'সুন্দর', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রোডাক্ট গুলো অসাধারণ
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'গুলো', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  সুন্দর কালেকশন
    Afert Tokenizing:  ['সুন্দর', 'কালেকশন']
    Truncating punctuation: ['সুন্দর', 'কালেকশন']
    Truncating StopWords: ['সুন্দর', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রতিটি পন্য অসম্ভব সুন্দর
    Afert Tokenizing:  ['প্রতিটি', 'পন্য', 'অসম্ভব', 'সুন্দর']
    Truncating punctuation: ['প্রতিটি', 'পন্য', 'অসম্ভব', 'সুন্দর']
    Truncating StopWords: ['প্রতিটি', 'পন্য', 'অসম্ভব', 'সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেড়ে গেছে।
    Afert Tokenizing:  ['দাম', 'বেড়ে', 'গেছে', '।']
    Truncating punctuation: ['দাম', 'বেড়ে', 'গেছে']
    Truncating StopWords: ['দাম', 'বেড়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  ইদে জুতা নিলাম এর এখন জুতার হিল নষ্ট হলো কেন।।
    Afert Tokenizing:  ['ইদে', 'জুতা', 'নিলাম', 'এর', 'এখন', 'জুতার', 'হিল', 'নষ্ট', 'হলো', 'কেন।', '।']
    Truncating punctuation: ['ইদে', 'জুতা', 'নিলাম', 'এর', 'এখন', 'জুতার', 'হিল', 'নষ্ট', 'হলো', 'কেন।']
    Truncating StopWords: ['ইদে', 'জুতা', 'নিলাম', 'জুতার', 'হিল', 'নষ্ট', 'কেন।']
    ***************************************************************************************
    Label:  0
    Sentence:  গিয়েছিলাম কোন কিছুই পাই নাই ভালো
    Afert Tokenizing:  ['গিয়েছিলাম', 'কোন', 'কিছুই', 'পাই', 'নাই', 'ভালো']
    Truncating punctuation: ['গিয়েছিলাম', 'কোন', 'কিছুই', 'পাই', 'নাই', 'ভালো']
    Truncating StopWords: ['গিয়েছিলাম', 'পাই', 'নাই', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ নিলাম, অনেক ভালো মানের জুতা
    Afert Tokenizing:  ['আজ', 'নিলাম', ',', 'অনেক', 'ভালো', 'মানের', 'জুতা']
    Truncating punctuation: ['আজ', 'নিলাম', 'অনেক', 'ভালো', 'মানের', 'জুতা']
    Truncating StopWords: ['নিলাম', 'ভালো', 'মানের', 'জুতা']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যি ই খুব সুন্দর। বেশি খুশি লেগেছে কালার একদম ছবির মতই
    Afert Tokenizing:  ['সত্যি', 'ই', 'খুব', 'সুন্দর', '।', 'বেশি', 'খুশি', 'লেগেছে', 'কালার', 'একদম', 'ছবির', 'মতই']
    Truncating punctuation: ['সত্যি', 'ই', 'খুব', 'সুন্দর', 'বেশি', 'খুশি', 'লেগেছে', 'কালার', 'একদম', 'ছবির', 'মতই']
    Truncating StopWords: ['সত্যি', 'সুন্দর', 'বেশি', 'খুশি', 'লেগেছে', 'কালার', 'একদম', 'ছবির', 'মতই']
    ***************************************************************************************
    Label:  1
    Sentence:  আসসালামু আলাইকুম আপু তোমার জামা অডার করেছি।
    Afert Tokenizing:  ['আসসালামু', 'আলাইকুম', 'আপু', 'তোমার', 'জামা', 'অডার', 'করেছি', '।']
    Truncating punctuation: ['আসসালামু', 'আলাইকুম', 'আপু', 'তোমার', 'জামা', 'অডার', 'করেছি']
    Truncating StopWords: ['আসসালামু', 'আলাইকুম', 'আপু', 'জামা', 'অডার', 'করেছি']
    ***************************************************************************************
    Label:  1
    Sentence:  তোমারদে জিনিস গুলো অনেক অনেক ভালো
    Afert Tokenizing:  ['তোমারদে', 'জিনিস', 'গুলো', 'অনেক', 'অনেক', 'ভালো']
    Truncating punctuation: ['তোমারদে', 'জিনিস', 'গুলো', 'অনেক', 'অনেক', 'ভালো']
    Truncating StopWords: ['তোমারদে', 'জিনিস', 'গুলো', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  জামা দুটি পেয়েছি আপু, অনেক সুন্দর হয়েছে
    Afert Tokenizing:  ['জামা', 'দুটি', 'পেয়েছি', 'আপু', ',', 'অনেক', 'সুন্দর', 'হয়েছে']
    Truncating punctuation: ['জামা', 'দুটি', 'পেয়েছি', 'আপু', 'অনেক', 'সুন্দর', 'হয়েছে']
    Truncating StopWords: ['জামা', 'পেয়েছি', 'আপু', 'সুন্দর', 'হয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:  দামটা বেশি হয়ে যায়
    Afert Tokenizing:  ['দামটা', 'বেশি', 'হয়ে', 'যায়']
    Truncating punctuation: ['দামটা', 'বেশি', 'হয়ে', 'যায়']
    Truncating StopWords: ['দামটা', 'বেশি', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  হাস্যকর দাম।
    Afert Tokenizing:  ['হাস্যকর', 'দাম', '।']
    Truncating punctuation: ['হাস্যকর', 'দাম']
    Truncating StopWords: ['হাস্যকর', 'দাম']
    ***************************************************************************************
    Label:  0
    Sentence:  এতো দাম
    Afert Tokenizing:  ['এতো', 'দাম']
    Truncating punctuation: ['এতো', 'দাম']
    Truncating StopWords: ['এতো', 'দাম']
    ***************************************************************************************
    Label:  0
    Sentence:  অনেক দাম কারন কি?
    Afert Tokenizing:  ['অনেক', 'দাম', 'কারন', 'কি', '?']
    Truncating punctuation: ['অনেক', 'দাম', 'কারন', 'কি']
    Truncating StopWords: ['দাম', 'কারন']
    ***************************************************************************************
    Label:  1
    Sentence:  কী সুন্দর
    Afert Tokenizing:  ['কী', 'সুন্দর']
    Truncating punctuation: ['কী', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  অসম্ভব সুন্দর, আরামদায়ক
    Afert Tokenizing:  ['অসম্ভব', 'সুন্দর', ',', 'আরামদায়ক']
    Truncating punctuation: ['অসম্ভব', 'সুন্দর', 'আরামদায়ক']
    Truncating StopWords: ['অসম্ভব', 'সুন্দর', 'আরামদায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর সুন্দর প্যান্ট।
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'সুন্দর', 'প্যান্ট', '।']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'সুন্দর', 'প্যান্ট']
    Truncating StopWords: ['সুন্দর', 'সুন্দর', 'প্যান্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি একটা নিয়েছি। ভাল প্যান্ট আরামদায়ক।
    Afert Tokenizing:  ['আমি', 'একটা', 'নিয়েছি', '।', 'ভাল', 'প্যান্ট', 'আরামদায়ক', '।']
    Truncating punctuation: ['আমি', 'একটা', 'নিয়েছি', 'ভাল', 'প্যান্ট', 'আরামদায়ক']
    Truncating StopWords: ['একটা', 'নিয়েছি', 'ভাল', 'প্যান্ট', 'আরামদায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  অসংখ্য ধন্যবাদ একদিনের মধ্যে পেয়ে গেলাম।
    Afert Tokenizing:  ['অসংখ্য', 'ধন্যবাদ', 'একদিনের', 'মধ্যে', 'পেয়ে', 'গেলাম', '।']
    Truncating punctuation: ['অসংখ্য', 'ধন্যবাদ', 'একদিনের', 'মধ্যে', 'পেয়ে', 'গেলাম']
    Truncating StopWords: ['অসংখ্য', 'ধন্যবাদ', 'একদিনের', 'গেলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের পাঞ্জাবির কাপড় বেশ ভালো মানের
    Afert Tokenizing:  ['আপনাদের', 'পাঞ্জাবির', 'কাপড়', 'বেশ', 'ভালো', 'মানের']
    Truncating punctuation: ['আপনাদের', 'পাঞ্জাবির', 'কাপড়', 'বেশ', 'ভালো', 'মানের']
    Truncating StopWords: ['আপনাদের', 'পাঞ্জাবির', 'কাপড়', 'ভালো', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকেই প্রডাক্ট টা হাতে পেলাম অসম্ভব সুন্দর
    Afert Tokenizing:  ['আজকেই', 'প্রডাক্ট', 'টা', 'হাতে', 'পেলাম', 'অসম্ভব', 'সুন্দর']
    Truncating punctuation: ['আজকেই', 'প্রডাক্ট', 'টা', 'হাতে', 'পেলাম', 'অসম্ভব', 'সুন্দর']
    Truncating StopWords: ['আজকেই', 'প্রডাক্ট', 'টা', 'হাতে', 'পেলাম', 'অসম্ভব', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  অনলাইন প্রোডাক্ট হিসেবে অনেক ভালো
    Afert Tokenizing:  ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'অনেক', 'ভালো']
    Truncating punctuation: ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'অনেক', 'ভালো']
    Truncating StopWords: ['অনলাইন', 'প্রোডাক্ট', 'হিসেবে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুণ কালেকশন
    Afert Tokenizing:  ['দারুণ', 'কালেকশন']
    Truncating punctuation: ['দারুণ', 'কালেকশন']
    Truncating StopWords: ['দারুণ', 'কালেকশন']
    ***************************************************************************************
    Label:  0
    Sentence:  দামটা একটু বেশি মনে হচ্ছে।
    Afert Tokenizing:  ['দামটা', 'একটু', 'বেশি', 'মনে', 'হচ্ছে', '।']
    Truncating punctuation: ['দামটা', 'একটু', 'বেশি', 'মনে', 'হচ্ছে']
    Truncating StopWords: ['দামটা', 'একটু', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ ছিল পরে অনেক কম্ফোর্ট পেয়েছি এবং খুবই আরামদায়ক
    Afert Tokenizing:  ['অসাধারণ', 'ছিল', 'পরে', 'অনেক', 'কম্ফোর্ট', 'পেয়েছি', 'এবং', 'খুবই', 'আরামদায়ক']
    Truncating punctuation: ['অসাধারণ', 'ছিল', 'পরে', 'অনেক', 'কম্ফোর্ট', 'পেয়েছি', 'এবং', 'খুবই', 'আরামদায়ক']
    Truncating StopWords: ['অসাধারণ', 'কম্ফোর্ট', 'পেয়েছি', 'খুবই', 'আরামদায়ক']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম টা ত বেশ চড়া।
    Afert Tokenizing:  ['দাম', 'টা', 'ত', 'বেশ', 'চড়া', '।']
    Truncating punctuation: ['দাম', 'টা', 'ত', 'বেশ', 'চড়া']
    Truncating StopWords: ['দাম', 'টা', 'ত', 'চড়া']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো। আরামদায়ক
    Afert Tokenizing:  ['অনেক', 'ভালো', '।', 'আরামদায়ক']
    Truncating punctuation: ['অনেক', 'ভালো', 'আরামদায়ক']
    Truncating StopWords: ['ভালো', 'আরামদায়ক']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি অর্ডার করেছি কেউ ফোন করে নি , অর্ডার কনফার্ম হয়েছে কি না বুঝতে পারছি না
    Afert Tokenizing:  ['আমি', 'অর্ডার', 'করেছি', 'কেউ', 'ফোন', 'করে', 'নি', '', ',', 'অর্ডার', 'কনফার্ম', 'হয়েছে', 'কি', 'না', 'বুঝতে', 'পারছি', 'না']
    Truncating punctuation: ['আমি', 'অর্ডার', 'করেছি', 'কেউ', 'ফোন', 'করে', 'নি', '', 'অর্ডার', 'কনফার্ম', 'হয়েছে', 'কি', 'না', 'বুঝতে', 'পারছি', 'না']
    Truncating StopWords: ['অর্ডার', 'করেছি', 'ফোন', 'নি', '', 'অর্ডার', 'কনফার্ম', 'না', 'বুঝতে', 'পারছি', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট হাতে পেলাম। খুবই ভালো মানের ছিলো
    Afert Tokenizing:  ['প্রডাক্ট', 'হাতে', 'পেলাম', '।', 'খুবই', 'ভালো', 'মানের', 'ছিলো']
    Truncating punctuation: ['প্রডাক্ট', 'হাতে', 'পেলাম', 'খুবই', 'ভালো', 'মানের', 'ছিলো']
    Truncating StopWords: ['প্রডাক্ট', 'হাতে', 'পেলাম', 'খুবই', 'ভালো', 'মানের', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর আমি নিছি
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'আমি', 'নিছি']
    Truncating punctuation: ['খুব', 'সুন্দর', 'আমি', 'নিছি']
    Truncating StopWords: ['সুন্দর', 'নিছি']
    ***************************************************************************************
    Label:  1
    Sentence:  চাইলে নিতে পারেন প্রোডাক্ট এর মানে অনেক ভালো।
    Afert Tokenizing:  ['চাইলে', 'নিতে', 'পারেন', 'প্রোডাক্ট', 'এর', 'মানে', 'অনেক', 'ভালো', '।']
    Truncating punctuation: ['চাইলে', 'নিতে', 'পারেন', 'প্রোডাক্ট', 'এর', 'মানে', 'অনেক', 'ভালো']
    Truncating StopWords: ['চাইলে', 'প্রোডাক্ট', 'মানে', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাইজটা অনেক বেশি
    Afert Tokenizing:  ['প্রাইজটা', 'অনেক', 'বেশি']
    Truncating punctuation: ['প্রাইজটা', 'অনেক', 'বেশি']
    Truncating StopWords: ['প্রাইজটা', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি নিয়েছি খুব ভালো প্রোডাক্ট
    Afert Tokenizing:  ['আমি', 'নিয়েছি', 'খুব', 'ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['আমি', 'নিয়েছি', 'খুব', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['নিয়েছি', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  টেক্সট দিয়ে রাখছি কোন রিপ্লাই নাই
    Afert Tokenizing:  ['টেক্সট', 'দিয়ে', 'রাখছি', 'কোন', 'রিপ্লাই', 'নাই']
    Truncating punctuation: ['টেক্সট', 'দিয়ে', 'রাখছি', 'কোন', 'রিপ্লাই', 'নাই']
    Truncating StopWords: ['টেক্সট', 'দিয়ে', 'রাখছি', 'রিপ্লাই', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশি চাচ্ছেন।
    Afert Tokenizing:  ['দাম', 'বেশি', 'চাচ্ছেন', '।']
    Truncating punctuation: ['দাম', 'বেশি', 'চাচ্ছেন']
    Truncating StopWords: ['দাম', 'বেশি', 'চাচ্ছেন']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইন থেকে নিব না দোকানে এসে নিব
    Afert Tokenizing:  ['অনলাইন', 'থেকে', 'নিব', 'না', 'দোকানে', 'এসে', 'নিব']
    Truncating punctuation: ['অনলাইন', 'থেকে', 'নিব', 'না', 'দোকানে', 'এসে', 'নিব']
    Truncating StopWords: ['অনলাইন', 'নিব', 'না', 'দোকানে', 'নিব']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি আমার অর্ডার করা প্রডাক্টস হাতে পেয়েছি, কালার এবং মান দুটোই ভাল, সত্যিই ভাল লেগেছে খুব
    Afert Tokenizing:  ['আমি', 'আমার', 'অর্ডার', 'করা', 'প্রডাক্টস', 'হাতে', 'পেয়েছি', ',', 'কালার', 'এবং', 'মান', 'দুটোই', 'ভাল', ',', 'সত্যিই', 'ভাল', 'লেগেছে', 'খুব']
    Truncating punctuation: ['আমি', 'আমার', 'অর্ডার', 'করা', 'প্রডাক্টস', 'হাতে', 'পেয়েছি', 'কালার', 'এবং', 'মান', 'দুটোই', 'ভাল', 'সত্যিই', 'ভাল', 'লেগেছে', 'খুব']
    Truncating StopWords: ['অর্ডার', 'প্রডাক্টস', 'হাতে', 'পেয়েছি', 'কালার', 'মান', 'দুটোই', 'ভাল', 'সত্যিই', 'ভাল', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  পন্য ক্রয় করার পর যদি দেখি সার্টের গুনগত মান খারাপ
    Afert Tokenizing:  ['পন্য', 'ক্রয়', 'করার', 'পর', 'যদি', 'দেখি', 'সার্টের', 'গুনগত', 'মান', 'খারাপ']
    Truncating punctuation: ['পন্য', 'ক্রয়', 'করার', 'পর', 'যদি', 'দেখি', 'সার্টের', 'গুনগত', 'মান', 'খারাপ']
    Truncating StopWords: ['পন্য', 'ক্রয়', 'দেখি', 'সার্টের', 'গুনগত', 'মান', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের শোরুমের ঠিকানা দিন, আমি সরাসরি কিনতে চাচ্ছি,
    Afert Tokenizing:  ['আপনাদের', 'শোরুমের', 'ঠিকানা', 'দিন', ',', 'আমি', 'সরাসরি', 'কিনতে', 'চাচ্ছি', ',']
    Truncating punctuation: ['আপনাদের', 'শোরুমের', 'ঠিকানা', 'দিন', 'আমি', 'সরাসরি', 'কিনতে', 'চাচ্ছি']
    Truncating StopWords: ['আপনাদের', 'শোরুমের', 'ঠিকানা', 'সরাসরি', 'কিনতে', 'চাচ্ছি']
    ***************************************************************************************
    Label:  1
    Sentence:  দামে কম মানে ভালো প্রোডাক্ট।
    Afert Tokenizing:  ['দামে', 'কম', 'মানে', 'ভালো', 'প্রোডাক্ট', '।']
    Truncating punctuation: ['দামে', 'কম', 'মানে', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['দামে', 'কম', 'মানে', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ সুন্দর প্রোডাক্টগুলা.. এক্কেবারে পারফেক্ট কাস্টমাইজড
    Afert Tokenizing:  ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', '.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    Truncating punctuation: ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    Truncating StopWords: ['অসাধারণ', 'সুন্দর', 'প্রোডাক্টগুলা.', 'এক্কেবারে', 'পারফেক্ট', 'কাস্টমাইজড']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ আলহামদুলিল্লাহ আলহামদুলিল্লাহ
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজের মত হবেনাতো অর্ডার দিলাম একটা দারাজ দিছে আরেকটা এমন হবে নাতো
    Afert Tokenizing:  ['দারাজের', 'মত', 'হবেনাতো', 'অর্ডার', 'দিলাম', 'একটা', 'দারাজ', 'দিছে', 'আরেকটা', 'এমন', 'হবে', 'নাতো']
    Truncating punctuation: ['দারাজের', 'মত', 'হবেনাতো', 'অর্ডার', 'দিলাম', 'একটা', 'দারাজ', 'দিছে', 'আরেকটা', 'এমন', 'হবে', 'নাতো']
    Truncating StopWords: ['দারাজের', 'মত', 'হবেনাতো', 'অর্ডার', 'দিলাম', 'একটা', 'দারাজ', 'দিছে', 'আরেকটা', 'নাতো']
    ***************************************************************************************
    Label:  1
    Sentence:  কাস্টমার সার্ভিস এ যারা আছেন প্রফেশনাল আমার কাছে মনে হয়েছে।
    Afert Tokenizing:  ['কাস্টমার', 'সার্ভিস', 'এ', 'যারা', 'আছেন', 'প্রফেশনাল', 'আমার', 'কাছে', 'মনে', 'হয়েছে', '।']
    Truncating punctuation: ['কাস্টমার', 'সার্ভিস', 'এ', 'যারা', 'আছেন', 'প্রফেশনাল', 'আমার', 'কাছে', 'মনে', 'হয়েছে']
    Truncating StopWords: ['কাস্টমার', 'সার্ভিস', 'আছেন', 'প্রফেশনাল', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের সার্ভিস অনেক ভালো
    Afert Tokenizing:  ['আপনাদের', 'সার্ভিস', 'অনেক', 'ভালো']
    Truncating punctuation: ['আপনাদের', 'সার্ভিস', 'অনেক', 'ভালো']
    Truncating StopWords: ['আপনাদের', 'সার্ভিস', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ ডেলিভারি করা হলো। আলহামদুলিল্লাহ, অনেক সুন্দর হয়েছে।
    Afert Tokenizing:  ['আজ', 'ডেলিভারি', 'করা', 'হলো', '।', 'আলহামদুলিল্লাহ', ',', 'অনেক', 'সুন্দর', 'হয়েছে', '।']
    Truncating punctuation: ['আজ', 'ডেলিভারি', 'করা', 'হলো', 'আলহামদুলিল্লাহ', 'অনেক', 'সুন্দর', 'হয়েছে']
    Truncating StopWords: ['ডেলিভারি', 'আলহামদুলিল্লাহ', 'সুন্দর', 'হয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:  পেন্ডিং অর্ডারগুলি পাইলে বাঁচি...
    Afert Tokenizing:  ['পেন্ডিং', 'অর্ডারগুলি', 'পাইলে', 'বাঁচি..', '.']
    Truncating punctuation: ['পেন্ডিং', 'অর্ডারগুলি', 'পাইলে', 'বাঁচি..']
    Truncating StopWords: ['পেন্ডিং', 'অর্ডারগুলি', 'পাইলে', 'বাঁচি..']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি গত তিন বছর ধরে তাদের কাছ থেকেই নিচ্ছি। আলহামদুলিল্লাহ, তাদের সার্ভিসে আমি সন্তুষ্ট।
    Afert Tokenizing:  ['আমি', 'গত', 'তিন', 'বছর', 'ধরে', 'তাদের', 'কাছ', 'থেকেই', 'নিচ্ছি', '।', 'আলহামদুলিল্লাহ', ',', 'তাদের', 'সার্ভিসে', 'আমি', 'সন্তুষ্ট', '।']
    Truncating punctuation: ['আমি', 'গত', 'তিন', 'বছর', 'ধরে', 'তাদের', 'কাছ', 'থেকেই', 'নিচ্ছি', 'আলহামদুলিল্লাহ', 'তাদের', 'সার্ভিসে', 'আমি', 'সন্তুষ্ট']
    Truncating StopWords: ['গত', 'তিন', 'বছর', 'নিচ্ছি', 'আলহামদুলিল্লাহ', 'সার্ভিসে', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  নির্ভেজাল আর নির্ভরযোগ্য সেবা পেয়ে যাচ্ছি।
    Afert Tokenizing:  ['নির্ভেজাল', 'আর', 'নির্ভরযোগ্য', 'সেবা', 'পেয়ে', 'যাচ্ছি', '।']
    Truncating punctuation: ['নির্ভেজাল', 'আর', 'নির্ভরযোগ্য', 'সেবা', 'পেয়ে', 'যাচ্ছি']
    Truncating StopWords: ['নির্ভেজাল', 'নির্ভরযোগ্য', 'সেবা', 'পেয়ে', 'যাচ্ছি']
    ***************************************************************************************
    Label:  1
    Sentence:  উনাদের ডেলিভারিও ফাস্ট
    Afert Tokenizing:  ['উনাদের', 'ডেলিভারিও', 'ফাস্ট']
    Truncating punctuation: ['উনাদের', 'ডেলিভারিও', 'ফাস্ট']
    Truncating StopWords: ['উনাদের', 'ডেলিভারিও', 'ফাস্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশে ই-কমার্স মানেই প্রতারণা ।
    Afert Tokenizing:  ['বাংলাদেশে', 'ই-কমার্স', 'মানেই', 'প্রতারণা', '', '।']
    Truncating punctuation: ['বাংলাদেশে', 'ই-কমার্স', 'মানেই', 'প্রতারণা', '']
    Truncating StopWords: ['বাংলাদেশে', 'ই-কমার্স', 'মানেই', 'প্রতারণা', '']
    ***************************************************************************************
    Label:  0
    Sentence:  চিটার বাটপার দের খোঁজ খবর নাই
    Afert Tokenizing:  ['চিটার', 'বাটপার', 'দের', 'খোঁজ', 'খবর', 'নাই']
    Truncating punctuation: ['চিটার', 'বাটপার', 'দের', 'খোঁজ', 'খবর', 'নাই']
    Truncating StopWords: ['চিটার', 'বাটপার', 'দের', 'খোঁজ', 'খবর', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপারটা জেলেই থাক !
    Afert Tokenizing:  ['বাটপারটা', 'জেলেই', 'থাক', '', '!']
    Truncating punctuation: ['বাটপারটা', 'জেলেই', 'থাক', '']
    Truncating StopWords: ['বাটপারটা', 'জেলেই', 'থাক', '']
    ***************************************************************************************
    Label:  0
    Sentence:  আমরা ভুক্তভুগীরা কি প্রোডাক্ট কি পাবো নাকি টাকাটাও ফেরত পাবো না ??
    Afert Tokenizing:  ['আমরা', 'ভুক্তভুগীরা', 'কি', 'প্রোডাক্ট', 'কি', 'পাবো', 'নাকি', 'টাকাটাও', 'ফেরত', 'পাবো', 'না', '?', '?']
    Truncating punctuation: ['আমরা', 'ভুক্তভুগীরা', 'কি', 'প্রোডাক্ট', 'কি', 'পাবো', 'নাকি', 'টাকাটাও', 'ফেরত', 'পাবো', 'না']
    Truncating StopWords: ['ভুক্তভুগীরা', 'প্রোডাক্ট', 'পাবো', 'টাকাটাও', 'ফেরত', 'পাবো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ওয়েব সাইট এ এখনো ঢুকা যায় না কেন??
    Afert Tokenizing:  ['ওয়েব', 'সাইট', 'এ', 'এখনো', 'ঢুকা', 'যায়', 'না', 'কেন?', '?']
    Truncating punctuation: ['ওয়েব', 'সাইট', 'এ', 'এখনো', 'ঢুকা', 'যায়', 'না', 'কেন?']
    Truncating StopWords: ['ওয়েব', 'সাইট', 'এখনো', 'ঢুকা', 'যায়', 'না', 'কেন?']
    ***************************************************************************************
    Label:  0
    Sentence:  বেশি দাম হয়ে যায়
    Afert Tokenizing:  ['বেশি', 'দাম', 'হয়ে', 'যায়']
    Truncating punctuation: ['বেশি', 'দাম', 'হয়ে', 'যায়']
    Truncating StopWords: ['বেশি', 'দাম', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের জন্য দোয়া রইল ভাই
    Afert Tokenizing:  ['আপনাদের', 'জন্য', 'দোয়া', 'রইল', 'ভাই']
    Truncating punctuation: ['আপনাদের', 'জন্য', 'দোয়া', 'রইল', 'ভাই']
    Truncating StopWords: ['আপনাদের', 'দোয়া', 'রইল', 'ভাই']
    ***************************************************************************************
    Label:  1
    Sentence:  অর্ডার করতে পারেন নিশ্চিন্তে
    Afert Tokenizing:  ['অর্ডার', 'করতে', 'পারেন', 'নিশ্চিন্তে']
    Truncating punctuation: ['অর্ডার', 'করতে', 'পারেন', 'নিশ্চিন্তে']
    Truncating StopWords: ['অর্ডার', 'নিশ্চিন্তে']
    ***************************************************************************************
    Label:  1
    Sentence:   খুবই মান সম্পন্ন ও আরামদায়ক জার্সি দেওয়ার জন্য ধন্যবাদ
    Afert Tokenizing:  ['খুবই', 'মান', 'সম্পন্ন', 'ও', 'আরামদায়ক', 'জার্সি', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating punctuation: ['খুবই', 'মান', 'সম্পন্ন', 'ও', 'আরামদায়ক', 'জার্সি', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['খুবই', 'মান', 'সম্পন্ন', 'আরামদায়ক', 'জার্সি', 'দেওয়ার', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:   অনেক অনেক ধন্যবাদ অনেক ভাল মানের কাপড়।
    Afert Tokenizing:  ['অনেক', 'অনেক', 'ধন্যবাদ', 'অনেক', 'ভাল', 'মানের', 'কাপড়', '।']
    Truncating punctuation: ['অনেক', 'অনেক', 'ধন্যবাদ', 'অনেক', 'ভাল', 'মানের', 'কাপড়']
    Truncating StopWords: ['ধন্যবাদ', 'ভাল', 'মানের', 'কাপড়']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ কম দামে সুন্দর একটা জার্সি উপহার দেওয়ার জন্য
    Afert Tokenizing:  ['ধন্যবাদ', 'কম', 'দামে', 'সুন্দর', 'একটা', 'জার্সি', 'উপহার', 'দেওয়ার', 'জন্য']
    Truncating punctuation: ['ধন্যবাদ', 'কম', 'দামে', 'সুন্দর', 'একটা', 'জার্সি', 'উপহার', 'দেওয়ার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'কম', 'দামে', 'সুন্দর', 'একটা', 'জার্সি', 'উপহার', 'দেওয়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড় কোয়ালিটি ১০০তে ১০০% ভালো।
    Afert Tokenizing:  ['কাপড়', 'কোয়ালিটি', '১০০তে', '১০০%', 'ভালো', '।']
    Truncating punctuation: ['কাপড়', 'কোয়ালিটি', '১০০তে', '১০০%', 'ভালো']
    Truncating StopWords: ['কাপড়', 'কোয়ালিটি', '১০০তে', '১০০%', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  আমাদের টাকা ফেরত চাই
    Afert Tokenizing:  ['আমাদের', 'টাকা', 'ফেরত', 'চাই']
    Truncating punctuation: ['আমাদের', 'টাকা', 'ফেরত', 'চাই']
    Truncating StopWords: ['টাকা', 'ফেরত', 'চাই']
    ***************************************************************************************
    Label:  1
    Sentence:  আশাকরি আমাদের জিনিশ গুলা পাবো,
    Afert Tokenizing:  ['আশাকরি', 'আমাদের', 'জিনিশ', 'গুলা', 'পাবো', ',']
    Truncating punctuation: ['আশাকরি', 'আমাদের', 'জিনিশ', 'গুলা', 'পাবো']
    Truncating StopWords: ['আশাকরি', 'জিনিশ', 'গুলা', 'পাবো']
    ***************************************************************************************
    Label:  1
    Sentence:  তাহলে পন্য গুলো পাবো ইনশাআল্লাহ
    Afert Tokenizing:  ['তাহলে', 'পন্য', 'গুলো', 'পাবো', 'ইনশাআল্লাহ']
    Truncating punctuation: ['তাহলে', 'পন্য', 'গুলো', 'পাবো', 'ইনশাআল্লাহ']
    Truncating StopWords: ['পন্য', 'গুলো', 'পাবো', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  মূল টাকাটা দেন জাস্ট। লাভ আর দরকার নেই আমার।
    Afert Tokenizing:  ['মূল', 'টাকাটা', 'দেন', 'জাস্ট', '।', 'লাভ', 'আর', 'দরকার', 'নেই', 'আমার', '।']
    Truncating punctuation: ['মূল', 'টাকাটা', 'দেন', 'জাস্ট', 'লাভ', 'আর', 'দরকার', 'নেই', 'আমার']
    Truncating StopWords: ['মূল', 'টাকাটা', 'জাস্ট', 'লাভ', 'দরকার', 'নেই']
    ***************************************************************************************
    Label:  0
    Sentence:  মূল টাকাটা দেন জাস্ট। লাভ আর দরকার নেই আমার।
    Afert Tokenizing:  ['মূল', 'টাকাটা', 'দেন', 'জাস্ট', '।', 'লাভ', 'আর', 'দরকার', 'নেই', 'আমার', '।']
    Truncating punctuation: ['মূল', 'টাকাটা', 'দেন', 'জাস্ট', 'লাভ', 'আর', 'দরকার', 'নেই', 'আমার']
    Truncating StopWords: ['মূল', 'টাকাটা', 'জাস্ট', 'লাভ', 'দরকার', 'নেই']
    ***************************************************************************************
    Label:  1
    Sentence:  এই পেজের পন্য গুলো অনেক ভালো ,আপনারা সবাই নিতে পারেন আমি দুইটি নিয়েছি সেম টু সেম দিয়েছে , ধন্যবাদ
    Afert Tokenizing:  ['এই', 'পেজের', 'পন্য', 'গুলো', 'অনেক', 'ভালো', 'আপনারা', ',', 'সবাই', 'নিতে', 'পারেন', 'আমি', 'দুইটি', 'নিয়েছি', 'সেম', 'টু', 'সেম', 'দিয়েছে', '', ',', 'ধন্যবাদ']
    Truncating punctuation: ['এই', 'পেজের', 'পন্য', 'গুলো', 'অনেক', 'ভালো', 'আপনারা', 'সবাই', 'নিতে', 'পারেন', 'আমি', 'দুইটি', 'নিয়েছি', 'সেম', 'টু', 'সেম', 'দিয়েছে', '', 'ধন্যবাদ']
    Truncating StopWords: ['পেজের', 'পন্য', 'গুলো', 'ভালো', 'আপনারা', 'সবাই', 'দুইটি', 'নিয়েছি', 'সেম', 'টু', 'সেম', '', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাইজ অনেক বেশি
    Afert Tokenizing:  ['প্রাইজ', 'অনেক', 'বেশি']
    Truncating punctuation: ['প্রাইজ', 'অনেক', 'বেশি']
    Truncating StopWords: ['প্রাইজ', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবি ভালো প্রোডাক্টটি ধন্যবাদ
    Afert Tokenizing:  ['খুবি', 'ভালো', 'প্রোডাক্টটি', 'ধন্যবাদ']
    Truncating punctuation: ['খুবি', 'ভালো', 'প্রোডাক্টটি', 'ধন্যবাদ']
    Truncating StopWords: ['খুবি', 'ভালো', 'প্রোডাক্টটি', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রত্যেকটা প্রোডাক্ট খুবই মানসম্মত
    Afert Tokenizing:  ['আপনাদের', 'প্রত্যেকটা', 'প্রোডাক্ট', 'খুবই', 'মানসম্মত']
    Truncating punctuation: ['আপনাদের', 'প্রত্যেকটা', 'প্রোডাক্ট', 'খুবই', 'মানসম্মত']
    Truncating StopWords: ['আপনাদের', 'প্রত্যেকটা', 'প্রোডাক্ট', 'খুবই', 'মানসম্মত']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ আপনাদের।।।। কথা রাখার জন্য।।।।।
    Afert Tokenizing:  ['ধন্যবাদ', 'আপনাদের।।।', '।', 'কথা', 'রাখার', 'জন্য।।।।', '।']
    Truncating punctuation: ['ধন্যবাদ', 'আপনাদের।।।', 'কথা', 'রাখার', 'জন্য।।।।']
    Truncating StopWords: ['ধন্যবাদ', 'আপনাদের।।।', 'কথা', 'রাখার', 'জন্য।।।।']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মোটামুটি চয়েছ হয়েছে কিন্তু দামটা বেশি মনে হচ্ছে,
    Afert Tokenizing:  ['আমার', 'মোটামুটি', 'চয়েছ', 'হয়েছে', 'কিন্তু', 'দামটা', 'বেশি', 'মনে', 'হচ্ছে', ',']
    Truncating punctuation: ['আমার', 'মোটামুটি', 'চয়েছ', 'হয়েছে', 'কিন্তু', 'দামটা', 'বেশি', 'মনে', 'হচ্ছে']
    Truncating StopWords: ['মোটামুটি', 'চয়েছ', 'হয়েছে', 'দামটা', 'বেশি']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশি হয়
    Afert Tokenizing:  ['দাম', 'বেশি', 'হয়']
    Truncating punctuation: ['দাম', 'বেশি', 'হয়']
    Truncating StopWords: ['দাম', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ, খুব দ্রুতই প্রোডাক্ট হাতে পেয়েছি এবং কাপড়ের মান অনেক ভালো,
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', ',', 'খুব', 'দ্রুতই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'এবং', 'কাপড়ের', 'মান', 'অনেক', 'ভালো', ',']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'খুব', 'দ্রুতই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'এবং', 'কাপড়ের', 'মান', 'অনেক', 'ভালো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'দ্রুতই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'কাপড়ের', 'মান', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার জিনিস কবে পাব
    Afert Tokenizing:  ['আমার', 'জিনিস', 'কবে', 'পাব']
    Truncating punctuation: ['আমার', 'জিনিস', 'কবে', 'পাব']
    Truncating StopWords: ['জিনিস', 'পাব']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার পচ্ছন্দ কিন্তু ডেলিভারি চার্জ আমাকে নিরুৎসাহিত করছে।
    Afert Tokenizing:  ['আমার', 'পচ্ছন্দ', 'কিন্তু', 'ডেলিভারি', 'চার্জ', 'আমাকে', 'নিরুৎসাহিত', 'করছে', '।']
    Truncating punctuation: ['আমার', 'পচ্ছন্দ', 'কিন্তু', 'ডেলিভারি', 'চার্জ', 'আমাকে', 'নিরুৎসাহিত', 'করছে']
    Truncating StopWords: ['পচ্ছন্দ', 'ডেলিভারি', 'চার্জ', 'নিরুৎসাহিত']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি একটা নিতে চাই।
    Afert Tokenizing:  ['আমি', 'একটা', 'নিতে', 'চাই', '।']
    Truncating punctuation: ['আমি', 'একটা', 'নিতে', 'চাই']
    Truncating StopWords: ['একটা', 'চাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের প্রোডাক্ট এর আগে আমি নিয়েছিলাম কয়মাস আগে,পুরো ফালতু
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'এর', 'আগে', 'আমি', 'নিয়েছিলাম', 'কয়মাস', 'আগে,পুরো', 'ফালতু']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'এর', 'আগে', 'আমি', 'নিয়েছিলাম', 'কয়মাস', 'আগে,পুরো', 'ফালতু']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'নিয়েছিলাম', 'কয়মাস', 'আগে,পুরো', 'ফালতু']
    ***************************************************************************************
    Label:  0
    Sentence:  এরা ফাজিল। অর্ডার করলে মাল পাঠায় না।
    Afert Tokenizing:  ['এরা', 'ফাজিল', '।', 'অর্ডার', 'করলে', 'মাল', 'পাঠায়', 'না', '।']
    Truncating punctuation: ['এরা', 'ফাজিল', 'অর্ডার', 'করলে', 'মাল', 'পাঠায়', 'না']
    Truncating StopWords: ['ফাজিল', 'অর্ডার', 'মাল', 'পাঠায়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম টা একটু বেশি চাইতেছে কেন
    Afert Tokenizing:  ['দাম', 'টা', 'একটু', 'বেশি', 'চাইতেছে', 'কেন']
    Truncating punctuation: ['দাম', 'টা', 'একটু', 'বেশি', 'চাইতেছে', 'কেন']
    Truncating StopWords: ['দাম', 'টা', 'একটু', 'বেশি', 'চাইতেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ কাপড় টা পরে ভালোই লাগলো।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'কাপড়', 'টা', 'পরে', 'ভালোই', 'লাগলো', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'কাপড়', 'টা', 'পরে', 'ভালোই', 'লাগলো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'কাপড়', 'টা', 'ভালোই', 'লাগলো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রোডাক্ট পেয়ে আমি সন্তুষ্ট। আরো নিবো শীঘ্রই ইনশাআল্লাহ। কোয়ালিটি ধরে রাখবেন আশা করি।
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্ট', 'পেয়ে', 'আমি', 'সন্তুষ্ট', '।', 'আরো', 'নিবো', 'শীঘ্রই', 'ইনশাআল্লাহ', '।', 'কোয়ালিটি', 'ধরে', 'রাখবেন', 'আশা', 'করি', '।']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্ট', 'পেয়ে', 'আমি', 'সন্তুষ্ট', 'আরো', 'নিবো', 'শীঘ্রই', 'ইনশাআল্লাহ', 'কোয়ালিটি', 'ধরে', 'রাখবেন', 'আশা', 'করি']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্ট', 'পেয়ে', 'সন্তুষ্ট', 'আরো', 'নিবো', 'শীঘ্রই', 'ইনশাআল্লাহ', 'কোয়ালিটি', 'রাখবেন', 'আশা']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম অনুযায়ী প্রোডাক্ট মানসম্মত, যে কেউ ক্রয় করতে পারেন।
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'প্রোডাক্ট', 'মানসম্মত', ',', 'যে', 'কেউ', 'ক্রয়', 'করতে', 'পারেন', '।']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'প্রোডাক্ট', 'মানসম্মত', 'যে', 'কেউ', 'ক্রয়', 'করতে', 'পারেন']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'প্রোডাক্ট', 'মানসম্মত', 'ক্রয়']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ আজ পেয়েছি পছন্দের জিনিস গুলো সব মিলিয়ে কম দামে অসাধারণ প্রোডাক্ট
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'আজ', 'পেয়েছি', 'পছন্দের', 'জিনিস', 'গুলো', 'সব', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আজ', 'পেয়েছি', 'পছন্দের', 'জিনিস', 'গুলো', 'সব', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'পেয়েছি', 'পছন্দের', 'জিনিস', 'গুলো', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো লাগছে থ্যাঙ্ক ইউ
    Afert Tokenizing:  ['ভালো', 'লাগছে', 'থ্যাঙ্ক', 'ইউ']
    Truncating punctuation: ['ভালো', 'লাগছে', 'থ্যাঙ্ক', 'ইউ']
    Truncating StopWords: ['ভালো', 'লাগছে', 'থ্যাঙ্ক', 'ইউ']
    ***************************************************************************************
    Label:  0
    Sentence:  সব মিলিয়ে এটি একটি আনপ্রফেশনাল পোস্ট।
    Afert Tokenizing:  ['সব', 'মিলিয়ে', 'এটি', 'একটি', 'আনপ্রফেশনাল', 'পোস্ট', '।']
    Truncating punctuation: ['সব', 'মিলিয়ে', 'এটি', 'একটি', 'আনপ্রফেশনাল', 'পোস্ট']
    Truncating StopWords: ['মিলিয়ে', 'আনপ্রফেশনাল', 'পোস্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  ১ দিনে প্রোডাক্ট পেয়েছি। কোয়ালিটি অসাধারণ। সবাই কিনতে পারেন
    Afert Tokenizing:  ['১', 'দিনে', 'প্রোডাক্ট', 'পেয়েছি', '।', 'কোয়ালিটি', 'অসাধারণ', '।', 'সবাই', 'কিনতে', 'পারেন']
    Truncating punctuation: ['১', 'দিনে', 'প্রোডাক্ট', 'পেয়েছি', 'কোয়ালিটি', 'অসাধারণ', 'সবাই', 'কিনতে', 'পারেন']
    Truncating StopWords: ['১', 'দিনে', 'প্রোডাক্ট', 'পেয়েছি', 'কোয়ালিটি', 'অসাধারণ', 'সবাই', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম অনুযায়ী মানসম্মত টি শার্ট
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'মানসম্মত', 'টি', 'শার্ট']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'মানসম্মত', 'টি', 'শার্ট']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'মানসম্মত', 'শার্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  পন্যের মান খুব ভালো।
    Afert Tokenizing:  ['পন্যের', 'মান', 'খুব', 'ভালো', '।']
    Truncating punctuation: ['পন্যের', 'মান', 'খুব', 'ভালো']
    Truncating StopWords: ['পন্যের', 'মান', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো কালেকশন
    Afert Tokenizing:  ['ভালো', 'কালেকশন']
    Truncating punctuation: ['ভালো', 'কালেকশন']
    Truncating StopWords: ['ভালো', 'কালেকশন']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাইজ বলবেন সরাসরি কমেন্ট এ বলবেন ইনবক্সে ডাকার কি আছে?
    Afert Tokenizing:  ['প্রাইজ', 'বলবেন', 'সরাসরি', 'কমেন্ট', 'এ', 'বলবেন', 'ইনবক্সে', 'ডাকার', 'কি', 'আছে', '?']
    Truncating punctuation: ['প্রাইজ', 'বলবেন', 'সরাসরি', 'কমেন্ট', 'এ', 'বলবেন', 'ইনবক্সে', 'ডাকার', 'কি', 'আছে']
    Truncating StopWords: ['প্রাইজ', 'বলবেন', 'সরাসরি', 'কমেন্ট', 'বলবেন', 'ইনবক্সে', 'ডাকার']
    ***************************************************************************************
    Label:  1
    Sentence:  অন্যান্য পেইজের তুলনায় দাম অনেকটাই কম এবং সার্ভিস অনেকটাই ভালো এবং ব্যবহার অনেকটাই ভালো তাদের ।
    Afert Tokenizing:  ['অন্যান্য', 'পেইজের', 'তুলনায়', 'দাম', 'অনেকটাই', 'কম', 'এবং', 'সার্ভিস', 'অনেকটাই', 'ভালো', 'এবং', 'ব্যবহার', 'অনেকটাই', 'ভালো', 'তাদের', '', '।']
    Truncating punctuation: ['অন্যান্য', 'পেইজের', 'তুলনায়', 'দাম', 'অনেকটাই', 'কম', 'এবং', 'সার্ভিস', 'অনেকটাই', 'ভালো', 'এবং', 'ব্যবহার', 'অনেকটাই', 'ভালো', 'তাদের', '']
    Truncating StopWords: ['অন্যান্য', 'পেইজের', 'তুলনায়', 'দাম', 'অনেকটাই', 'কম', 'সার্ভিস', 'অনেকটাই', 'ভালো', 'অনেকটাই', 'ভালো', '']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর প্রোডাক্ট,,দাম ও অন্যান্য পেজ থেকে কম,, ফিটিং অনেক সুন্দর হইছে,,স্যাটিসফাইড
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'প্রোডাক্ট,,দাম', 'ও', 'অন্যান্য', 'পেজ', 'থেকে', 'কম,', ',', 'ফিটিং', 'অনেক', 'সুন্দর', 'হইছে,,স্যাটিসফাইড']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'প্রোডাক্ট,,দাম', 'ও', 'অন্যান্য', 'পেজ', 'থেকে', 'কম,', 'ফিটিং', 'অনেক', 'সুন্দর', 'হইছে,,স্যাটিসফাইড']
    Truncating StopWords: ['সুন্দর', 'প্রোডাক্ট,,দাম', 'অন্যান্য', 'পেজ', 'কম,', 'ফিটিং', 'সুন্দর', 'হইছে,,স্যাটিসফাইড']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর তো
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'তো']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'তো']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  ২ দিন ব্যবহার করার পর রিভিউ দিলাম। আলহামদুলিল্লাহ অনেক ভালো প্রোডাক্ট।
    Afert Tokenizing:  ['২', 'দিন', 'ব্যবহার', 'করার', 'পর', 'রিভিউ', 'দিলাম', '।', 'আলহামদুলিল্লাহ', 'অনেক', 'ভালো', 'প্রোডাক্ট', '।']
    Truncating punctuation: ['২', 'দিন', 'ব্যবহার', 'করার', 'পর', 'রিভিউ', 'দিলাম', 'আলহামদুলিল্লাহ', 'অনেক', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['২', 'রিভিউ', 'দিলাম', 'আলহামদুলিল্লাহ', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ। প্রডাক্ট কোয়ালিটি এবং সার্ভিস খুবই ভালো।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '।', 'প্রডাক্ট', 'কোয়ালিটি', 'এবং', 'সার্ভিস', 'খুবই', 'ভালো', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'প্রডাক্ট', 'কোয়ালিটি', 'এবং', 'সার্ভিস', 'খুবই', 'ভালো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'প্রডাক্ট', 'কোয়ালিটি', 'সার্ভিস', 'খুবই', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট কোয়ালিটি অসাধারন।ডেলিভারি সিষ্টেম খুব ভালো।ব্যবহার যথেষ্ট ভালো।এককথায় আমার খুবই ভালো লেগেছে।
    Afert Tokenizing:  ['প্রোডাক্ট', 'কোয়ালিটি', 'অসাধারন।ডেলিভারি', 'সিষ্টেম', 'খুব', 'ভালো।ব্যবহার', 'যথেষ্ট', 'ভালো।এককথায়', 'আমার', 'খুবই', 'ভালো', 'লেগেছে', '।']
    Truncating punctuation: ['প্রোডাক্ট', 'কোয়ালিটি', 'অসাধারন।ডেলিভারি', 'সিষ্টেম', 'খুব', 'ভালো।ব্যবহার', 'যথেষ্ট', 'ভালো।এককথায়', 'আমার', 'খুবই', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['প্রোডাক্ট', 'কোয়ালিটি', 'অসাধারন।ডেলিভারি', 'সিষ্টেম', 'ভালো।ব্যবহার', 'যথেষ্ট', 'ভালো।এককথায়', 'খুবই', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম অনেক
    Afert Tokenizing:  ['দাম', 'অনেক']
    Truncating punctuation: ['দাম', 'অনেক']
    Truncating StopWords: ['দাম']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশের অনলাইন শপ মানে গলাকাটা দাম
    Afert Tokenizing:  ['বাংলাদেশের', 'অনলাইন', 'শপ', 'মানে', 'গলাকাটা', 'দাম']
    Truncating punctuation: ['বাংলাদেশের', 'অনলাইন', 'শপ', 'মানে', 'গলাকাটা', 'দাম']
    Truncating StopWords: ['বাংলাদেশের', 'অনলাইন', 'শপ', 'মানে', 'গলাকাটা', 'দাম']
    ***************************************************************************************
    Label:  0
    Sentence:   অর্ডার করার পর প্রাপ্ত প্রোডাক্ট আর ছবির সাথে মিল থাকে না
    Afert Tokenizing:  ['অর্ডার', 'করার', 'পর', 'প্রাপ্ত', 'প্রোডাক্ট', 'আর', 'ছবির', 'সাথে', 'মিল', 'থাকে', 'না']
    Truncating punctuation: ['অর্ডার', 'করার', 'পর', 'প্রাপ্ত', 'প্রোডাক্ট', 'আর', 'ছবির', 'সাথে', 'মিল', 'থাকে', 'না']
    Truncating StopWords: ['অর্ডার', 'প্রাপ্ত', 'প্রোডাক্ট', 'ছবির', 'সাথে', 'মিল', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশের অনলাইন পন্যের যে মান, যে একবার কেনে সে জীবনে আর কোন দিন কেনার ইচ্ছেও প্রকাশ করে না।পন্যের মান খুবই খারাপ
    Afert Tokenizing:  ['বাংলাদেশের', 'অনলাইন', 'পন্যের', 'যে', 'মান', ',', 'যে', 'একবার', 'কেনে', 'সে', 'জীবনে', 'আর', 'কোন', 'দিন', 'কেনার', 'ইচ্ছেও', 'প্রকাশ', 'করে', 'না।পন্যের', 'মান', 'খুবই', 'খারাপ']
    Truncating punctuation: ['বাংলাদেশের', 'অনলাইন', 'পন্যের', 'যে', 'মান', 'যে', 'একবার', 'কেনে', 'সে', 'জীবনে', 'আর', 'কোন', 'দিন', 'কেনার', 'ইচ্ছেও', 'প্রকাশ', 'করে', 'না।পন্যের', 'মান', 'খুবই', 'খারাপ']
    Truncating StopWords: ['বাংলাদেশের', 'অনলাইন', 'পন্যের', 'মান', 'কেনে', 'জীবনে', 'কেনার', 'ইচ্ছেও', 'প্রকাশ', 'না।পন্যের', 'মান', 'খুবই', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মনে হয় বাংলাদেশে যারা অনলাইন শপিং করেন তারা সবাই এইরকম ভাবে একবার হলেও ঠকেছেন।
    Afert Tokenizing:  ['আমার', 'মনে', 'হয়', 'বাংলাদেশে', 'যারা', 'অনলাইন', 'শপিং', 'করেন', 'তারা', 'সবাই', 'এইরকম', 'ভাবে', 'একবার', 'হলেও', 'ঠকেছেন', '।']
    Truncating punctuation: ['আমার', 'মনে', 'হয়', 'বাংলাদেশে', 'যারা', 'অনলাইন', 'শপিং', 'করেন', 'তারা', 'সবাই', 'এইরকম', 'ভাবে', 'একবার', 'হলেও', 'ঠকেছেন']
    Truncating StopWords: ['বাংলাদেশে', 'অনলাইন', 'শপিং', 'সবাই', 'এইরকম', 'ঠকেছেন']
    ***************************************************************************************
    Label:  0
    Sentence:  যত চিটার বাটপার আছে সব এখন এই অনলাইনে , ভালো যা আছে খুজে পাওয়া দুষ্কর
    Afert Tokenizing:  ['যত', 'চিটার', 'বাটপার', 'আছে', 'সব', 'এখন', 'এই', 'অনলাইনে', '', ',', 'ভালো', 'যা', 'আছে', 'খুজে', 'পাওয়া', 'দুষ্কর']
    Truncating punctuation: ['যত', 'চিটার', 'বাটপার', 'আছে', 'সব', 'এখন', 'এই', 'অনলাইনে', '', 'ভালো', 'যা', 'আছে', 'খুজে', 'পাওয়া', 'দুষ্কর']
    Truncating StopWords: ['চিটার', 'বাটপার', 'অনলাইনে', '', 'ভালো', 'খুজে', 'পাওয়া', 'দুষ্কর']
    ***************************************************************************************
    Label:  0
    Sentence:  আগে পণ্যের মান ভালো করা লাগবে, অনলাইন কেনাকাটায় দেখায় মুরগী , খাওয়ায় ডাইল
    Afert Tokenizing:  ['আগে', 'পণ্যের', 'মান', 'ভালো', 'করা', 'লাগবে', ',', 'অনলাইন', 'কেনাকাটায়', 'দেখায়', 'মুরগী', '', ',', 'খাওয়ায়', 'ডাইল']
    Truncating punctuation: ['আগে', 'পণ্যের', 'মান', 'ভালো', 'করা', 'লাগবে', 'অনলাইন', 'কেনাকাটায়', 'দেখায়', 'মুরগী', '', 'খাওয়ায়', 'ডাইল']
    Truncating StopWords: ['পণ্যের', 'মান', 'ভালো', 'লাগবে', 'অনলাইন', 'কেনাকাটায়', 'দেখায়', 'মুরগী', '', 'খাওয়ায়', 'ডাইল']
    ***************************************************************************************
    Label:  0
    Sentence:  যত বারই ক্রয় করেছি ততবারই ধরা।
    Afert Tokenizing:  ['যত', 'বারই', 'ক্রয়', 'করেছি', 'ততবারই', 'ধরা', '।']
    Truncating punctuation: ['যত', 'বারই', 'ক্রয়', 'করেছি', 'ততবারই', 'ধরা']
    Truncating StopWords: ['বারই', 'ক্রয়', 'করেছি', 'ততবারই']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইনে ঠকার আশংকা আছে।
    Afert Tokenizing:  ['অনলাইনে', 'ঠকার', 'আশংকা', 'আছে', '।']
    Truncating punctuation: ['অনলাইনে', 'ঠকার', 'আশংকা', 'আছে']
    Truncating StopWords: ['অনলাইনে', 'ঠকার', 'আশংকা']
    ***************************************************************************************
    Label:  0
    Sentence:  মাক্সিমাম মানুষ দেখবেন যাচাই বাছাই না করেই প্রডাক্ট কিনে এরপর দোষ দেয় অনলাইনের।
    Afert Tokenizing:  ['মাক্সিমাম', 'মানুষ', 'দেখবেন', 'যাচাই', 'বাছাই', 'না', 'করেই', 'প্রডাক্ট', 'কিনে', 'এরপর', 'দোষ', 'দেয়', 'অনলাইনের', '।']
    Truncating punctuation: ['মাক্সিমাম', 'মানুষ', 'দেখবেন', 'যাচাই', 'বাছাই', 'না', 'করেই', 'প্রডাক্ট', 'কিনে', 'এরপর', 'দোষ', 'দেয়', 'অনলাইনের']
    Truncating StopWords: ['মাক্সিমাম', 'মানুষ', 'দেখবেন', 'যাচাই', 'বাছাই', 'না', 'প্রডাক্ট', 'কিনে', 'এরপর', 'দোষ', 'দেয়', 'অনলাইনের']
    ***************************************************************************************
    Label:  0
    Sentence:   প্রধান উদ্দেশ্যই কাষ্টমার ঠকানো
    Afert Tokenizing:  ['প্রধান', 'উদ্দেশ্যই', 'কাষ্টমার', 'ঠকানো']
    Truncating punctuation: ['প্রধান', 'উদ্দেশ্যই', 'কাষ্টমার', 'ঠকানো']
    Truncating StopWords: ['প্রধান', 'উদ্দেশ্যই', 'কাষ্টমার', 'ঠকানো']
    ***************************************************************************************
    Label:  0
    Sentence:  কিনলেই ঠকতে হয়,
    Afert Tokenizing:  ['কিনলেই', 'ঠকতে', 'হয়', ',']
    Truncating punctuation: ['কিনলেই', 'ঠকতে', 'হয়']
    Truncating StopWords: ['কিনলেই', 'ঠকতে']
    ***************************************************************************************
    Label:  1
    Sentence:  লোভে পরে অর্ধেক দামে না কিনে বরং সঠিক দামে প্রোডাক্ট কিনুন অবশ্যই ভালো সেবা পাবেন
    Afert Tokenizing:  ['লোভে', 'পরে', 'অর্ধেক', 'দামে', 'না', 'কিনে', 'বরং', 'সঠিক', 'দামে', 'প্রোডাক্ট', 'কিনুন', 'অবশ্যই', 'ভালো', 'সেবা', 'পাবেন']
    Truncating punctuation: ['লোভে', 'পরে', 'অর্ধেক', 'দামে', 'না', 'কিনে', 'বরং', 'সঠিক', 'দামে', 'প্রোডাক্ট', 'কিনুন', 'অবশ্যই', 'ভালো', 'সেবা', 'পাবেন']
    Truncating StopWords: ['লোভে', 'অর্ধেক', 'দামে', 'না', 'কিনে', 'সঠিক', 'দামে', 'প্রোডাক্ট', 'কিনুন', 'অবশ্যই', 'ভালো', 'সেবা', 'পাবেন']
    ***************************************************************************************
    Label:  0
    Sentence:  কিভাবে কেনাকাটা করবো! যতবার করেছি ততবারই ঠকেছি
    Afert Tokenizing:  ['কিভাবে', 'কেনাকাটা', 'করবো', '!', 'যতবার', 'করেছি', 'ততবারই', 'ঠকেছি']
    Truncating punctuation: ['কিভাবে', 'কেনাকাটা', 'করবো', 'যতবার', 'করেছি', 'ততবারই', 'ঠকেছি']
    Truncating StopWords: ['কিভাবে', 'কেনাকাটা', 'করবো', 'যতবার', 'করেছি', 'ততবারই', 'ঠকেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  সঠিক পণ্যের মান নিশ্চয়তা নাই, কিভাবে কিনবে।
    Afert Tokenizing:  ['সঠিক', 'পণ্যের', 'মান', 'নিশ্চয়তা', 'নাই', ',', 'কিভাবে', 'কিনবে', '।']
    Truncating punctuation: ['সঠিক', 'পণ্যের', 'মান', 'নিশ্চয়তা', 'নাই', 'কিভাবে', 'কিনবে']
    Truncating StopWords: ['সঠিক', 'পণ্যের', 'মান', 'নিশ্চয়তা', 'নাই', 'কিভাবে', 'কিনবে']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশি দিয়ে অনলাইন থেকে কিনবো কেন?
    Afert Tokenizing:  ['দাম', 'বেশি', 'দিয়ে', 'অনলাইন', 'থেকে', 'কিনবো', 'কেন', '?']
    Truncating punctuation: ['দাম', 'বেশি', 'দিয়ে', 'অনলাইন', 'থেকে', 'কিনবো', 'কেন']
    Truncating StopWords: ['দাম', 'বেশি', 'দিয়ে', 'অনলাইন', 'কিনবো']
    ***************************************************************************************
    Label:  0
    Sentence:  "সকল পঁচা, নিম্নমানের ও বেশী দামে অনলাইনে দ্রব্য বিক্রি হচ্ছে। যতবার কিনেছি ততবার ধরা খাইছি।
    Afert Tokenizing:  ['সকল', '"', 'পঁচা', ',', 'নিম্নমানের', 'ও', 'বেশী', 'দামে', 'অনলাইনে', 'দ্রব্য', 'বিক্রি', 'হচ্ছে', '।', 'যতবার', 'কিনেছি', 'ততবার', 'ধরা', 'খাইছি', '।']
    Truncating punctuation: ['সকল', 'পঁচা', 'নিম্নমানের', 'ও', 'বেশী', 'দামে', 'অনলাইনে', 'দ্রব্য', 'বিক্রি', 'হচ্ছে', 'যতবার', 'কিনেছি', 'ততবার', 'ধরা', 'খাইছি']
    Truncating StopWords: ['সকল', 'পঁচা', 'নিম্নমানের', 'বেশী', 'দামে', 'অনলাইনে', 'দ্রব্য', 'বিক্রি', 'যতবার', 'কিনেছি', 'ততবার', 'খাইছি']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশে এখনো অধিকাংশ মানুষ স্ক্যামের স্বীকার হয়
    Afert Tokenizing:  ['বাংলাদেশে', 'এখনো', 'অধিকাংশ', 'মানুষ', 'স্ক্যামের', 'স্বীকার', 'হয়']
    Truncating punctuation: ['বাংলাদেশে', 'এখনো', 'অধিকাংশ', 'মানুষ', 'স্ক্যামের', 'স্বীকার', 'হয়']
    Truncating StopWords: ['বাংলাদেশে', 'এখনো', 'অধিকাংশ', 'মানুষ', 'স্ক্যামের', 'স্বীকার']
    ***************************************************************************************
    Label:  1
    Sentence:  মানুষ কে অনলাইনে কেনাকাটা করতে উৎসাহ করতে হবে ,
    Afert Tokenizing:  ['মানুষ', 'কে', 'অনলাইনে', 'কেনাকাটা', 'করতে', 'উৎসাহ', 'করতে', 'হবে', '', ',']
    Truncating punctuation: ['মানুষ', 'কে', 'অনলাইনে', 'কেনাকাটা', 'করতে', 'উৎসাহ', 'করতে', 'হবে', '']
    Truncating StopWords: ['মানুষ', 'অনলাইনে', 'কেনাকাটা', 'উৎসাহ', '']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো মানের অনলাইন শপ নাই
    Afert Tokenizing:  ['ভালো', 'মানের', 'অনলাইন', 'শপ', 'নাই']
    Truncating punctuation: ['ভালো', 'মানের', 'অনলাইন', 'শপ', 'নাই']
    Truncating StopWords: ['ভালো', 'মানের', 'অনলাইন', 'শপ', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশে অনলাইনে কেনাকাটা করে তার মধ্যে 90% মানুষ ঠকে
    Afert Tokenizing:  ['বাংলাদেশে', 'অনলাইনে', 'কেনাকাটা', 'করে', 'তার', 'মধ্যে', '90%', 'মানুষ', 'ঠকে']
    Truncating punctuation: ['বাংলাদেশে', 'অনলাইনে', 'কেনাকাটা', 'করে', 'তার', 'মধ্যে', '90%', 'মানুষ', 'ঠকে']
    Truncating StopWords: ['বাংলাদেশে', 'অনলাইনে', 'কেনাকাটা', '90%', 'মানুষ', 'ঠকে']
    ***************************************************************************************
    Label:  0
    Sentence:  সততার অভাব
    Afert Tokenizing:  ['সততার', 'অভাব']
    Truncating punctuation: ['সততার', 'অভাব']
    Truncating StopWords: ['সততার', 'অভাব']
    ***************************************************************************************
    Label:  0
    Sentence:  দুঃখজনক হলেও সত্যি যে, দেশে এখনও ভালো মানের কোন অনলাইন শপ নাই... যারা আছে তারা শুধুই গ্রাহকদের ঠকিয়ে কষ্ট দেয়।
    Afert Tokenizing:  ['দুঃখজনক', 'হলেও', 'সত্যি', 'যে', ',', 'দেশে', 'এখনও', 'ভালো', 'মানের', 'কোন', 'অনলাইন', 'শপ', 'নাই..', '.', 'যারা', 'আছে', 'তারা', 'শুধুই', 'গ্রাহকদের', 'ঠকিয়ে', 'কষ্ট', 'দেয়', '।']
    Truncating punctuation: ['দুঃখজনক', 'হলেও', 'সত্যি', 'যে', 'দেশে', 'এখনও', 'ভালো', 'মানের', 'কোন', 'অনলাইন', 'শপ', 'নাই..', 'যারা', 'আছে', 'তারা', 'শুধুই', 'গ্রাহকদের', 'ঠকিয়ে', 'কষ্ট', 'দেয়']
    Truncating StopWords: ['দুঃখজনক', 'সত্যি', 'দেশে', 'ভালো', 'মানের', 'অনলাইন', 'শপ', 'নাই..', 'শুধুই', 'গ্রাহকদের', 'ঠকিয়ে', 'কষ্ট', 'দেয়']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশের ই-কমার্স মানে কিনলেন ত ঠকলেন
    Afert Tokenizing:  ['বাংলাদেশের', 'ই-কমার্স', 'মানে', 'কিনলেন', 'ত', 'ঠকলেন']
    Truncating punctuation: ['বাংলাদেশের', 'ই-কমার্স', 'মানে', 'কিনলেন', 'ত', 'ঠকলেন']
    Truncating StopWords: ['বাংলাদেশের', 'ই-কমার্স', 'মানে', 'কিনলেন', 'ত', 'ঠকলেন']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইনের সার্ভিস খারাপ
    Afert Tokenizing:  ['অনলাইনের', 'সার্ভিস', 'খারাপ']
    Truncating punctuation: ['অনলাইনের', 'সার্ভিস', 'খারাপ']
    Truncating StopWords: ['অনলাইনের', 'সার্ভিস', 'খারাপ']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো প্রোডাক্ট
    Afert Tokenizing:  ['ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  সবকিছুর দাম বেড়ে গেছে
    Afert Tokenizing:  ['সবকিছুর', 'দাম', 'বেড়ে', 'গেছে']
    Truncating punctuation: ['সবকিছুর', 'দাম', 'বেড়ে', 'গেছে']
    Truncating StopWords: ['সবকিছুর', 'দাম', 'বেড়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুন অফার
    Afert Tokenizing:  ['দারুন', 'অফার']
    Truncating punctuation: ['দারুন', 'অফার']
    Truncating StopWords: ['দারুন', 'অফার']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার সবগুলো ই
    Afert Tokenizing:  ['চমৎকার', 'সবগুলো', 'ই']
    Truncating punctuation: ['চমৎকার', 'সবগুলো', 'ই']
    Truncating StopWords: ['চমৎকার', 'সবগুলো']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর কালেকশন
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'কালেকশন']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'কালেকশন']
    Truncating StopWords: ['সুন্দর', 'কালেকশন']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর।
    Afert Tokenizing:  ['অনেক', 'সুন্দর', '।']
    Truncating punctuation: ['অনেক', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার জন্য দোয়া এবং শুভকামনা রইল
    Afert Tokenizing:  ['আপনার', 'জন্য', 'দোয়া', 'এবং', 'শুভকামনা', 'রইল']
    Truncating punctuation: ['আপনার', 'জন্য', 'দোয়া', 'এবং', 'শুভকামনা', 'রইল']
    Truncating StopWords: ['দোয়া', 'শুভকামনা', 'রইল']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা আমাকে ১ বছরেও রিফান্ড দিল না
    Afert Tokenizing:  ['আমার', 'টাকা', 'আমাকে', '১', 'বছরেও', 'রিফান্ড', 'দিল', 'না']
    Truncating punctuation: ['আমার', 'টাকা', 'আমাকে', '১', 'বছরেও', 'রিফান্ড', 'দিল', 'না']
    Truncating StopWords: ['টাকা', '১', 'বছরেও', 'রিফান্ড', 'দিল', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এধরনের ঠকবাজি কর্মকান্ডের জন্যে বাংলাদেশে ইকমার্সে ঠিকমত গ্রো করতে পারছে না। ক্রেতা বিক্রেতাকে বিশ্বাস করতে পারছে না।
    Afert Tokenizing:  ['এধরনের', 'ঠকবাজি', 'কর্মকান্ডের', 'জন্যে', 'বাংলাদেশে', 'ইকমার্সে', 'ঠিকমত', 'গ্রো', 'করতে', 'পারছে', 'না', '।', 'ক্রেতা', 'বিক্রেতাকে', 'বিশ্বাস', 'করতে', 'পারছে', 'না', '।']
    Truncating punctuation: ['এধরনের', 'ঠকবাজি', 'কর্মকান্ডের', 'জন্যে', 'বাংলাদেশে', 'ইকমার্সে', 'ঠিকমত', 'গ্রো', 'করতে', 'পারছে', 'না', 'ক্রেতা', 'বিক্রেতাকে', 'বিশ্বাস', 'করতে', 'পারছে', 'না']
    Truncating StopWords: ['এধরনের', 'ঠকবাজি', 'কর্মকান্ডের', 'জন্যে', 'বাংলাদেশে', 'ইকমার্সে', 'ঠিকমত', 'গ্রো', 'পারছে', 'না', 'ক্রেতা', 'বিক্রেতাকে', 'বিশ্বাস', 'পারছে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ম্যাক্সিমাম মানহীন পন্য বাংলাদেশের অনলাইনে বিক্রি করে।
    Afert Tokenizing:  ['ম্যাক্সিমাম', 'মানহীন', 'পন্য', 'বাংলাদেশের', 'অনলাইনে', 'বিক্রি', 'করে', '।']
    Truncating punctuation: ['ম্যাক্সিমাম', 'মানহীন', 'পন্য', 'বাংলাদেশের', 'অনলাইনে', 'বিক্রি', 'করে']
    Truncating StopWords: ['ম্যাক্সিমাম', 'মানহীন', 'পন্য', 'বাংলাদেশের', 'অনলাইনে', 'বিক্রি']
    ***************************************************************************************
    Label:  0
    Sentence:  অন্ততপক্ষে বাংলাদেশ এ অনলাইন এ কেনাকাটা না করাই ভালো, দেখায় একটা , ডেলিভারি দেয় অন্যটা...
    Afert Tokenizing:  ['অন্ততপক্ষে', 'বাংলাদেশ', 'এ', 'অনলাইন', 'এ', 'কেনাকাটা', 'না', 'করাই', 'ভালো', ',', 'দেখায়', 'একটা', '', ',', 'ডেলিভারি', 'দেয়', 'অন্যটা..', '.']
    Truncating punctuation: ['অন্ততপক্ষে', 'বাংলাদেশ', 'এ', 'অনলাইন', 'এ', 'কেনাকাটা', 'না', 'করাই', 'ভালো', 'দেখায়', 'একটা', '', 'ডেলিভারি', 'দেয়', 'অন্যটা..']
    Truncating StopWords: ['অন্ততপক্ষে', 'বাংলাদেশ', 'অনলাইন', 'কেনাকাটা', 'না', 'ভালো', 'দেখায়', 'একটা', '', 'ডেলিভারি', 'দেয়', 'অন্যটা..']
    ***************************************************************************************
    Label:  1
    Sentence:  এরা খুবই ভালো আমার ৪ টা ফোন ডেলিভারি দিছে সময় মতো।
    Afert Tokenizing:  ['এরা', 'খুবই', 'ভালো', 'আমার', '৪', 'টা', 'ফোন', 'ডেলিভারি', 'দিছে', 'সময়', 'মতো', '।']
    Truncating punctuation: ['এরা', 'খুবই', 'ভালো', 'আমার', '৪', 'টা', 'ফোন', 'ডেলিভারি', 'দিছে', 'সময়', 'মতো']
    Truncating StopWords: ['খুবই', 'ভালো', '৪', 'টা', 'ফোন', 'ডেলিভারি', 'দিছে', 'সময়']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমিও ভুক্তভোগী আমাকে রিফান্ড ও দিচ্ছে না এরা
    Afert Tokenizing:  ['ভাই', 'আমিও', 'ভুক্তভোগী', 'আমাকে', 'রিফান্ড', 'ও', 'দিচ্ছে', 'না', 'এরা']
    Truncating punctuation: ['ভাই', 'আমিও', 'ভুক্তভোগী', 'আমাকে', 'রিফান্ড', 'ও', 'দিচ্ছে', 'না', 'এরা']
    Truncating StopWords: ['ভাই', 'আমিও', 'ভুক্তভোগী', 'রিফান্ড', 'দিচ্ছে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  বাংলাদেশে একমাত্র পিকাবু-ই ট্রাস্টেড।
    Afert Tokenizing:  ['বাংলাদেশে', 'একমাত্র', 'পিকাবু-ই', 'ট্রাস্টেড', '।']
    Truncating punctuation: ['বাংলাদেশে', 'একমাত্র', 'পিকাবু-ই', 'ট্রাস্টেড']
    Truncating StopWords: ['বাংলাদেশে', 'একমাত্র', 'পিকাবু-ই', 'ট্রাস্টেড']
    ***************************************************************************************
    Label:  1
    Sentence:  শতভাগ ট্রাস্টেড, ৪ লক্ষ টাকার কাছাকাছি আমার কয়েক জন রিলেটিভ পন্য ক্রয় করেছে। এবং ডেলিভারি ও ভালো
    Afert Tokenizing:  ['শতভাগ', 'ট্রাস্টেড', ',', '৪', 'লক্ষ', 'টাকার', 'কাছাকাছি', 'আমার', 'কয়েক', 'জন', 'রিলেটিভ', 'পন্য', 'ক্রয়', 'করেছে', '।', 'এবং', 'ডেলিভারি', 'ও', 'ভালো']
    Truncating punctuation: ['শতভাগ', 'ট্রাস্টেড', '৪', 'লক্ষ', 'টাকার', 'কাছাকাছি', 'আমার', 'কয়েক', 'জন', 'রিলেটিভ', 'পন্য', 'ক্রয়', 'করেছে', 'এবং', 'ডেলিভারি', 'ও', 'ভালো']
    Truncating StopWords: ['শতভাগ', 'ট্রাস্টেড', '৪', 'টাকার', 'কাছাকাছি', 'রিলেটিভ', 'পন্য', 'ক্রয়', 'ডেলিভারি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:   এক দুইদিন দেরি হলেও এরা ভাল প্রোডাক্টই দেয়
    Afert Tokenizing:  ['এক', 'দুইদিন', 'দেরি', 'হলেও', 'এরা', 'ভাল', 'প্রোডাক্টই', 'দেয়']
    Truncating punctuation: ['এক', 'দুইদিন', 'দেরি', 'হলেও', 'এরা', 'ভাল', 'প্রোডাক্টই', 'দেয়']
    Truncating StopWords: ['এক', 'দুইদিন', 'দেরি', 'ভাল', 'প্রোডাক্টই', 'দেয়']
    ***************************************************************************************
    Label:  1
    Sentence:  পিকাবো থেকে অনেক গুলা ডিভাইস কেনা হইছে খারাপ কিছু পাই নাই। সার্ভিস ও খুব ভালো।
    Afert Tokenizing:  ['পিকাবো', 'থেকে', 'অনেক', 'গুলা', 'ডিভাইস', 'কেনা', 'হইছে', 'খারাপ', 'কিছু', 'পাই', 'নাই', '।', 'সার্ভিস', 'ও', 'খুব', 'ভালো', '।']
    Truncating punctuation: ['পিকাবো', 'থেকে', 'অনেক', 'গুলা', 'ডিভাইস', 'কেনা', 'হইছে', 'খারাপ', 'কিছু', 'পাই', 'নাই', 'সার্ভিস', 'ও', 'খুব', 'ভালো']
    Truncating StopWords: ['পিকাবো', 'গুলা', 'ডিভাইস', 'কেনা', 'হইছে', 'খারাপ', 'পাই', 'নাই', 'সার্ভিস', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  এদের থেকে ফ্যন নিয়ে ধরা খায়ছি, চলে না
    Afert Tokenizing:  ['এদের', 'থেকে', 'ফ্যন', 'নিয়ে', 'ধরা', 'খায়ছি', ',', 'চলে', 'না']
    Truncating punctuation: ['এদের', 'থেকে', 'ফ্যন', 'নিয়ে', 'ধরা', 'খায়ছি', 'চলে', 'না']
    Truncating StopWords: ['ফ্যন', 'খায়ছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি পেতে কতদিন লাগবে বলা যায়না।১০-১২ দিনেও পেতে পারেন,২-৩ মাসও লাগতে পারে,আবার নাও পেতে পারেন।
    Afert Tokenizing:  ['ডেলিভারি', 'পেতে', 'কতদিন', 'লাগবে', 'বলা', 'যায়না।১০-১২', 'দিনেও', 'পেতে', 'পারেন,২-৩', 'মাসও', 'লাগতে', 'পারে,আবার', 'নাও', 'পেতে', 'পারেন', '।']
    Truncating punctuation: ['ডেলিভারি', 'পেতে', 'কতদিন', 'লাগবে', 'বলা', 'যায়না।১০-১২', 'দিনেও', 'পেতে', 'পারেন,২-৩', 'মাসও', 'লাগতে', 'পারে,আবার', 'নাও', 'পেতে', 'পারেন']
    Truncating StopWords: ['ডেলিভারি', 'পেতে', 'কতদিন', 'লাগবে', 'যায়না।১০-১২', 'দিনেও', 'পেতে', 'পারেন,২-৩', 'মাসও', 'লাগতে', 'পারে,আবার', 'নাও', 'পেতে']
    ***************************************************************************************
    Label:  0
    Sentence:  আকর্ষণীয় ছবি দেখিয়ে নকল / নিন্ম মানের পণ্য সরবরাহ
    Afert Tokenizing:  ['আকর্ষণীয়', 'ছবি', 'দেখিয়ে', 'নকল', '/', 'নিন্ম', 'মানের', 'পণ্য', 'সরবরাহ']
    Truncating punctuation: ['আকর্ষণীয়', 'ছবি', 'দেখিয়ে', 'নকল', '/', 'নিন্ম', 'মানের', 'পণ্য', 'সরবরাহ']
    Truncating StopWords: ['আকর্ষণীয়', 'ছবি', 'দেখিয়ে', 'নকল', '/', 'নিন্ম', 'মানের', 'পণ্য', 'সরবরাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  সেলার পণ্য আপলোডের সময় ওজন ভুল দিয়েছেন(গ্রাম ভেবে কেজিতে দিয়েছেন)।
    Afert Tokenizing:  ['সেলার', 'পণ্য', 'আপলোডের', 'সময়', 'ওজন', 'ভুল', 'দিয়েছেন(গ্রাম', 'ভেবে', 'কেজিতে', 'দিয়েছেন)', '।']
    Truncating punctuation: ['সেলার', 'পণ্য', 'আপলোডের', 'সময়', 'ওজন', 'ভুল', 'দিয়েছেন(গ্রাম', 'ভেবে', 'কেজিতে', 'দিয়েছেন)']
    Truncating StopWords: ['সেলার', 'পণ্য', 'আপলোডের', 'সময়', 'ওজন', 'ভুল', 'দিয়েছেন(গ্রাম', 'ভেবে', 'কেজিতে', 'দিয়েছেন)']
    ***************************************************************************************
    Label:  0
    Sentence:  ওরে চিটার
    Afert Tokenizing:  ['ওরে', 'চিটার']
    Truncating punctuation: ['ওরে', 'চিটার']
    Truncating StopWords: ['ওরে', 'চিটার']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপারি করেও অনায়াসে ব্যবসা করতেছে
    Afert Tokenizing:  ['বাটপারি', 'করেও', 'অনায়াসে', 'ব্যবসা', 'করতেছে']
    Truncating punctuation: ['বাটপারি', 'করেও', 'অনায়াসে', 'ব্যবসা', 'করতেছে']
    Truncating StopWords: ['বাটপারি', 'করেও', 'অনায়াসে', 'ব্যবসা', 'করতেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  ভোক্তা অধিকার এ মামলা করেন
    Afert Tokenizing:  ['ভোক্তা', 'অধিকার', 'এ', 'মামলা', 'করেন']
    Truncating punctuation: ['ভোক্তা', 'অধিকার', 'এ', 'মামলা', 'করেন']
    Truncating StopWords: ['ভোক্তা', 'অধিকার', 'মামলা']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি ও একই সমস্যার সম্মুখীন হয়েছি অনেক বেশি হয়রানি হয়েছি।
    Afert Tokenizing:  ['আমি', 'ও', 'একই', 'সমস্যার', 'সম্মুখীন', 'হয়েছি', 'অনেক', 'বেশি', 'হয়রানি', 'হয়েছি', '।']
    Truncating punctuation: ['আমি', 'ও', 'একই', 'সমস্যার', 'সম্মুখীন', 'হয়েছি', 'অনেক', 'বেশি', 'হয়রানি', 'হয়েছি']
    Truncating StopWords: ['সমস্যার', 'সম্মুখীন', 'হয়েছি', 'বেশি', 'হয়রানি', 'হয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  টাকা মার যাওয়ার সম্ভাবনা কিছুটা আছে
    Afert Tokenizing:  ['টাকা', 'মার', 'যাওয়ার', 'সম্ভাবনা', 'কিছুটা', 'আছে']
    Truncating punctuation: ['টাকা', 'মার', 'যাওয়ার', 'সম্ভাবনা', 'কিছুটা', 'আছে']
    Truncating StopWords: ['টাকা', 'মার', 'যাওয়ার', 'সম্ভাবনা', 'কিছুটা']
    ***************************************************************************************
    Label:  0
    Sentence:  অথবাতে অর্ডার করে আমি ভুক্তভুগি, ৬০ টি ডিম অর্ডার করেছিলাম। ডিমগুলো মারাত্মক তিতা স্বাদের
    Afert Tokenizing:  ['অথবাতে', 'অর্ডার', 'করে', 'আমি', 'ভুক্তভুগি', ',', '৬০', 'টি', 'ডিম', 'অর্ডার', 'করেছিলাম', '।', 'ডিমগুলো', 'মারাত্মক', 'তিতা', 'স্বাদের']
    Truncating punctuation: ['অথবাতে', 'অর্ডার', 'করে', 'আমি', 'ভুক্তভুগি', '৬০', 'টি', 'ডিম', 'অর্ডার', 'করেছিলাম', 'ডিমগুলো', 'মারাত্মক', 'তিতা', 'স্বাদের']
    Truncating StopWords: ['অথবাতে', 'অর্ডার', 'ভুক্তভুগি', '৬০', 'ডিম', 'অর্ডার', 'করেছিলাম', 'ডিমগুলো', 'মারাত্মক', 'তিতা', 'স্বাদের']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি তো ভাউচার দিয়ে অর্ডার করলাম,জিনিসও পেয়ে গেছি।
    Afert Tokenizing:  ['আমি', 'তো', 'ভাউচার', 'দিয়ে', 'অর্ডার', 'করলাম,জিনিসও', 'পেয়ে', 'গেছি', '।']
    Truncating punctuation: ['আমি', 'তো', 'ভাউচার', 'দিয়ে', 'অর্ডার', 'করলাম,জিনিসও', 'পেয়ে', 'গেছি']
    Truncating StopWords: ['ভাউচার', 'দিয়ে', 'অর্ডার', 'করলাম,জিনিসও', 'পেয়ে', 'গেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মতে ওনাদের জুতা ছাড়া কোনো কিছু ভালো না কারণ আমি নিজে 3 টা টি-শার্ট নিয়েছিলাম বলতে পারি থার্ডক্লাস কাপড়।
    Afert Tokenizing:  ['আমার', 'মতে', 'ওনাদের', 'জুতা', 'ছাড়া', 'কোনো', 'কিছু', 'ভালো', 'না', 'কারণ', 'আমি', 'নিজে', '3', 'টা', 'টি-শার্ট', 'নিয়েছিলাম', 'বলতে', 'পারি', 'থার্ডক্লাস', 'কাপড়', '।']
    Truncating punctuation: ['আমার', 'মতে', 'ওনাদের', 'জুতা', 'ছাড়া', 'কোনো', 'কিছু', 'ভালো', 'না', 'কারণ', 'আমি', 'নিজে', '3', 'টা', 'টি-শার্ট', 'নিয়েছিলাম', 'বলতে', 'পারি', 'থার্ডক্লাস', 'কাপড়']
    Truncating StopWords: ['মতে', 'ওনাদের', 'জুতা', 'ভালো', 'না', '3', 'টা', 'টি-শার্ট', 'নিয়েছিলাম', 'থার্ডক্লাস', 'কাপড়']
    ***************************************************************************************
    Label:  0
    Sentence:  এর জুতা লেদার এর ঠিকই কিন্তু নিম্নমানের সস্তা লেদার۔ জায়েজ এর জুতা ভালোনা দাম হিসেবে
    Afert Tokenizing:  ['এর', 'জুতা', 'লেদার', 'এর', 'ঠিকই', 'কিন্তু', 'নিম্নমানের', 'সস্তা', 'লেদার۔', 'জায়েজ', 'এর', 'জুতা', 'ভালোনা', 'দাম', 'হিসেবে']
    Truncating punctuation: ['এর', 'জুতা', 'লেদার', 'এর', 'ঠিকই', 'কিন্তু', 'নিম্নমানের', 'সস্তা', 'লেদার۔', 'জায়েজ', 'এর', 'জুতা', 'ভালোনা', 'দাম', 'হিসেবে']
    Truncating StopWords: ['জুতা', 'লেদার', 'ঠিকই', 'নিম্নমানের', 'সস্তা', 'লেদার۔', 'জায়েজ', 'জুতা', 'ভালোনা', 'দাম', 'হিসেবে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর রিভিউ। কম দামে ভালোই হয়েছে ফোনটি।
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'রিভিউ', '।', 'কম', 'দামে', 'ভালোই', 'হয়েছে', 'ফোনটি', '।']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'রিভিউ', 'কম', 'দামে', 'ভালোই', 'হয়েছে', 'ফোনটি']
    Truncating StopWords: ['সুন্দর', 'রিভিউ', 'কম', 'দামে', 'ভালোই', 'হয়েছে', 'ফোনটি']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো হয়েছে এই দামে। বাজার থেকে ৩০০+ টাকা কম পেয়েছেন।
    Afert Tokenizing:  ['অনেক', 'ভালো', 'হয়েছে', 'এই', 'দামে', '।', 'বাজার', 'থেকে', '৩০০+', 'টাকা', 'কম', 'পেয়েছেন', '।']
    Truncating punctuation: ['অনেক', 'ভালো', 'হয়েছে', 'এই', 'দামে', 'বাজার', 'থেকে', '৩০০+', 'টাকা', 'কম', 'পেয়েছেন']
    Truncating StopWords: ['ভালো', 'হয়েছে', 'দামে', 'বাজার', '৩০০+', 'টাকা', 'কম', 'পেয়েছেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ, E-Courier এর মাধ্যমে অনেক দ্রুতই ডেলিভারি পেলাম  । অর্ডার করা থেকে ডেলিভারি পর্যন্ত সকল সার্ভিস ভালো লেগেছে
    Afert Tokenizing:  ['আজ', ',', 'E-Courier', 'এর', 'মাধ্যমে', 'অনেক', 'দ্রুতই', 'ডেলিভারি', 'পেলাম', '', '।', 'অর্ডার', 'করা', 'থেকে', 'ডেলিভারি', 'পর্যন্ত', 'সকল', 'সার্ভিস', 'ভালো', 'লেগেছে']
    Truncating punctuation: ['আজ', 'E-Courier', 'এর', 'মাধ্যমে', 'অনেক', 'দ্রুতই', 'ডেলিভারি', 'পেলাম', '', 'অর্ডার', 'করা', 'থেকে', 'ডেলিভারি', 'পর্যন্ত', 'সকল', 'সার্ভিস', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['E-Courier', 'দ্রুতই', 'ডেলিভারি', 'পেলাম', '', 'অর্ডার', 'ডেলিভারি', 'সকল', 'সার্ভিস', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  দারাজে অনেক কমে পাওয়া যায় এবং ডেলিভারি অনেক ফাস্ট
    Afert Tokenizing:  ['দারাজে', 'অনেক', 'কমে', 'পাওয়া', 'যায়', 'এবং', 'ডেলিভারি', 'অনেক', 'ফাস্ট']
    Truncating punctuation: ['দারাজে', 'অনেক', 'কমে', 'পাওয়া', 'যায়', 'এবং', 'ডেলিভারি', 'অনেক', 'ফাস্ট']
    Truncating StopWords: ['দারাজে', 'কমে', 'পাওয়া', 'যায়', 'ডেলিভারি', 'ফাস্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  ট্রাই করেছি অনেকগুলো কিন্তু সার্ভিস ভালো না কারোই
    Afert Tokenizing:  ['ট্রাই', 'করেছি', 'অনেকগুলো', 'কিন্তু', 'সার্ভিস', 'ভালো', 'না', 'কারোই']
    Truncating punctuation: ['ট্রাই', 'করেছি', 'অনেকগুলো', 'কিন্তু', 'সার্ভিস', 'ভালো', 'না', 'কারোই']
    Truncating StopWords: ['ট্রাই', 'করেছি', 'অনেকগুলো', 'সার্ভিস', 'ভালো', 'না', 'কারোই']
    ***************************************************************************************
    Label:  1
    Sentence:  ইভ্যালির প্রতি আস্থা ছিলো আছে এবং থাকবে
    Afert Tokenizing:  ['ইভ্যালির', 'প্রতি', 'আস্থা', 'ছিলো', 'আছে', 'এবং', 'থাকবে']
    Truncating punctuation: ['ইভ্যালির', 'প্রতি', 'আস্থা', 'ছিলো', 'আছে', 'এবং', 'থাকবে']
    Truncating StopWords: ['ইভ্যালির', 'আস্থা', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  অবশ্যই পিকাবো, দ্রুত ও ট্রাস্টেড সার্ভিস
    Afert Tokenizing:  ['অবশ্যই', 'পিকাবো', ',', 'দ্রুত', 'ও', 'ট্রাস্টেড', 'সার্ভিস']
    Truncating punctuation: ['অবশ্যই', 'পিকাবো', 'দ্রুত', 'ও', 'ট্রাস্টেড', 'সার্ভিস']
    Truncating StopWords: ['অবশ্যই', 'পিকাবো', 'দ্রুত', 'ট্রাস্টেড', 'সার্ভিস']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার মনে হয় নিরাপত্তা, রিটার্ন ও রিফান্ডের মত বিষয়গুলো বিবেচনা করলে দারাজই সেরা।
    Afert Tokenizing:  ['আমার', 'মনে', 'হয়', 'নিরাপত্তা', ',', 'রিটার্ন', 'ও', 'রিফান্ডের', 'মত', 'বিষয়গুলো', 'বিবেচনা', 'করলে', 'দারাজই', 'সেরা', '।']
    Truncating punctuation: ['আমার', 'মনে', 'হয়', 'নিরাপত্তা', 'রিটার্ন', 'ও', 'রিফান্ডের', 'মত', 'বিষয়গুলো', 'বিবেচনা', 'করলে', 'দারাজই', 'সেরা']
    Truncating StopWords: ['নিরাপত্তা', 'রিটার্ন', 'রিফান্ডের', 'মত', 'বিষয়গুলো', 'বিবেচনা', 'দারাজই', 'সেরা']
    ***************************************************************************************
    Label:  1
    Sentence:  পিকাবু এর মতো স্মার্ট সার্ভিস মনে হয় না কোনো ই-কমার্স আছে। ইলেকট্রনিক গ্যাজেটস এর পিকাবু বেস্ট
    Afert Tokenizing:  ['পিকাবু', 'এর', 'মতো', 'স্মার্ট', 'সার্ভিস', 'মনে', 'হয়', 'না', 'কোনো', 'ই-কমার্স', 'আছে', '।', 'ইলেকট্রনিক', 'গ্যাজেটস', 'এর', 'পিকাবু', 'বেস্ট']
    Truncating punctuation: ['পিকাবু', 'এর', 'মতো', 'স্মার্ট', 'সার্ভিস', 'মনে', 'হয়', 'না', 'কোনো', 'ই-কমার্স', 'আছে', 'ইলেকট্রনিক', 'গ্যাজেটস', 'এর', 'পিকাবু', 'বেস্ট']
    Truncating StopWords: ['পিকাবু', 'স্মার্ট', 'সার্ভিস', 'না', 'ই-কমার্স', 'ইলেকট্রনিক', 'গ্যাজেটস', 'পিকাবু', 'বেস্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  তাদের সার্ভিস আসলেই ভালো
    Afert Tokenizing:  ['তাদের', 'সার্ভিস', 'আসলেই', 'ভালো']
    Truncating punctuation: ['তাদের', 'সার্ভিস', 'আসলেই', 'ভালো']
    Truncating StopWords: ['সার্ভিস', 'আসলেই', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  এমন ফ্রড প্রতিনিয়তই হচ্ছে।
    Afert Tokenizing:  ['এমন', 'ফ্রড', 'প্রতিনিয়তই', 'হচ্ছে', '।']
    Truncating punctuation: ['এমন', 'ফ্রড', 'প্রতিনিয়তই', 'হচ্ছে']
    Truncating StopWords: ['ফ্রড', 'প্রতিনিয়তই']
    ***************************************************************************************
    Label:  1
    Sentence:  এরা ধান্দাবাজ পার্টি।
    Afert Tokenizing:  ['এরা', 'ধান্দাবাজ', 'পার্টি', '।']
    Truncating punctuation: ['এরা', 'ধান্দাবাজ', 'পার্টি']
    Truncating StopWords: ['ধান্দাবাজ', 'পার্টি']
    ***************************************************************************************
    Label:  0
    Sentence:  এদের বিরুদ্ধে একশন নেওয়া দরকার
    Afert Tokenizing:  ['এদের', 'বিরুদ্ধে', 'একশন', 'নেওয়া', 'দরকার']
    Truncating punctuation: ['এদের', 'বিরুদ্ধে', 'একশন', 'নেওয়া', 'দরকার']
    Truncating StopWords: ['বিরুদ্ধে', 'একশন', 'দরকার']
    ***************************************************************************************
    Label:  0
    Sentence:  ভোক্তায় একটা অভিযোগ মেরে দেন, এদের শিক্ষা হওয়া উচিত।
    Afert Tokenizing:  ['ভোক্তায়', 'একটা', 'অভিযোগ', 'মেরে', 'দেন', ',', 'এদের', 'শিক্ষা', 'হওয়া', 'উচিত', '।']
    Truncating punctuation: ['ভোক্তায়', 'একটা', 'অভিযোগ', 'মেরে', 'দেন', 'এদের', 'শিক্ষা', 'হওয়া', 'উচিত']
    Truncating StopWords: ['ভোক্তায়', 'একটা', 'অভিযোগ', 'মেরে', 'শিক্ষা', 'হওয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  ব্র্যান্ড ছাড়া কাপড় অনলাইন থেকে কিনা উচিৎ না । অনলাইনে ভুক্তভুগিদের বেশির ভাগ মানুষ ই কাপড় কিনে ধরা খাইছে ।
    Afert Tokenizing:  ['ব্র্যান্ড', 'ছাড়া', 'কাপড়', 'অনলাইন', 'থেকে', 'কিনা', 'উচিৎ', 'না', '', '।', 'অনলাইনে', 'ভুক্তভুগিদের', 'বেশির', 'ভাগ', 'মানুষ', 'ই', 'কাপড়', 'কিনে', 'ধরা', 'খাইছে', '', '।']
    Truncating punctuation: ['ব্র্যান্ড', 'ছাড়া', 'কাপড়', 'অনলাইন', 'থেকে', 'কিনা', 'উচিৎ', 'না', '', 'অনলাইনে', 'ভুক্তভুগিদের', 'বেশির', 'ভাগ', 'মানুষ', 'ই', 'কাপড়', 'কিনে', 'ধরা', 'খাইছে', '']
    Truncating StopWords: ['ব্র্যান্ড', 'ছাড়া', 'কাপড়', 'অনলাইন', 'কিনা', 'উচিৎ', 'না', '', 'অনলাইনে', 'ভুক্তভুগিদের', 'বেশির', 'ভাগ', 'মানুষ', 'কাপড়', 'কিনে', 'খাইছে', '']
    ***************************************************************************************
    Label:  1
    Sentence:  ওদের প্রোডাক্টগুলো ভালোই।
    Afert Tokenizing:  ['ওদের', 'প্রোডাক্টগুলো', 'ভালোই', '।']
    Truncating punctuation: ['ওদের', 'প্রোডাক্টগুলো', 'ভালোই']
    Truncating StopWords: ['প্রোডাক্টগুলো', 'ভালোই']
    ***************************************************************************************
    Label:  1
    Sentence:  সব মিলিয়ে পান্ডামার্টের সার্ভিসে আমি খুব সন্তুষ্ট  পান্ডামার্ট এর জন্য শুভকামনা রইল
    Afert Tokenizing:  ['সব', 'মিলিয়ে', 'পান্ডামার্টের', 'সার্ভিসে', 'আমি', 'খুব', 'সন্তুষ্ট', 'পান্ডামার্ট', 'এর', 'জন্য', 'শুভকামনা', 'রইল']
    Truncating punctuation: ['সব', 'মিলিয়ে', 'পান্ডামার্টের', 'সার্ভিসে', 'আমি', 'খুব', 'সন্তুষ্ট', 'পান্ডামার্ট', 'এর', 'জন্য', 'শুভকামনা', 'রইল']
    Truncating StopWords: ['মিলিয়ে', 'পান্ডামার্টের', 'সার্ভিসে', 'সন্তুষ্ট', 'পান্ডামার্ট', 'শুভকামনা', 'রইল']
    ***************************************************************************************
    Label:  1
    Sentence:  দের সার্ভিস টিম যেমন ভাল তেমনি অথেনটিক প্রোডাক্ট এর ব্যাপারেও ওরা খুব সচেতন।
    Afert Tokenizing:  ['দের', 'সার্ভিস', 'টিম', 'যেমন', 'ভাল', 'তেমনি', 'অথেনটিক', 'প্রোডাক্ট', 'এর', 'ব্যাপারেও', 'ওরা', 'খুব', 'সচেতন', '।']
    Truncating punctuation: ['দের', 'সার্ভিস', 'টিম', 'যেমন', 'ভাল', 'তেমনি', 'অথেনটিক', 'প্রোডাক্ট', 'এর', 'ব্যাপারেও', 'ওরা', 'খুব', 'সচেতন']
    Truncating StopWords: ['দের', 'সার্ভিস', 'টিম', 'ভাল', 'তেমনি', 'অথেনটিক', 'প্রোডাক্ট', 'ব্যাপারেও', 'সচেতন']
    ***************************************************************************************
    Label:  0
    Sentence:   ফালতু সার্ভিস!
    Afert Tokenizing:  ['ফালতু', 'সার্ভিস', '!']
    Truncating punctuation: ['ফালতু', 'সার্ভিস']
    Truncating StopWords: ['ফালতু', 'সার্ভিস']
    ***************************************************************************************
    Label:  0
    Sentence:  মামলা করেন ভাই, কোনো সুযোগ দেওয়া যাবে না এদের।
    Afert Tokenizing:  ['মামলা', 'করেন', 'ভাই', ',', 'কোনো', 'সুযোগ', 'দেওয়া', 'যাবে', 'না', 'এদের', '।']
    Truncating punctuation: ['মামলা', 'করেন', 'ভাই', 'কোনো', 'সুযোগ', 'দেওয়া', 'যাবে', 'না', 'এদের']
    Truncating StopWords: ['মামলা', 'ভাই', 'সুযোগ', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  সকল ফ্রি প্রোডাক্ট এর প্যাকেজিং, প্রোডাক্ট কোয়ালিটি খুবই ভালো ছিল।
    Afert Tokenizing:  ['সকল', 'ফ্রি', 'প্রোডাক্ট', 'এর', 'প্যাকেজিং', ',', 'প্রোডাক্ট', 'কোয়ালিটি', 'খুবই', 'ভালো', 'ছিল', '।']
    Truncating punctuation: ['সকল', 'ফ্রি', 'প্রোডাক্ট', 'এর', 'প্যাকেজিং', 'প্রোডাক্ট', 'কোয়ালিটি', 'খুবই', 'ভালো', 'ছিল']
    Truncating StopWords: ['সকল', 'ফ্রি', 'প্রোডাক্ট', 'প্যাকেজিং', 'প্রোডাক্ট', 'কোয়ালিটি', 'খুবই', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু সার্ভিস, মনেহয় ডিসপ্লের পণ্য বিক্রি করে। প্লাস্টিক পণ্যে অনেক দাগ আর ধূলাবালি থাকে
    Afert Tokenizing:  ['ফালতু', 'সার্ভিস', ',', 'মনেহয়', 'ডিসপ্লের', 'পণ্য', 'বিক্রি', 'করে', '।', 'প্লাস্টিক', 'পণ্যে', 'অনেক', 'দাগ', 'আর', 'ধূলাবালি', 'থাকে']
    Truncating punctuation: ['ফালতু', 'সার্ভিস', 'মনেহয়', 'ডিসপ্লের', 'পণ্য', 'বিক্রি', 'করে', 'প্লাস্টিক', 'পণ্যে', 'অনেক', 'দাগ', 'আর', 'ধূলাবালি', 'থাকে']
    Truncating StopWords: ['ফালতু', 'সার্ভিস', 'মনেহয়', 'ডিসপ্লের', 'পণ্য', 'বিক্রি', 'প্লাস্টিক', 'পণ্যে', 'দাগ', 'ধূলাবালি']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশি রাখে
    Afert Tokenizing:  ['দাম', 'বেশি', 'রাখে']
    Truncating punctuation: ['দাম', 'বেশি', 'রাখে']
    Truncating StopWords: ['দাম', 'বেশি', 'রাখে']
    ***************************************************************************************
    Label:  0
    Sentence:  চাল ডাল ইদানিং পুরাই ফালতু সার্ভিস দেয়
    Afert Tokenizing:  ['চাল', 'ডাল', 'ইদানিং', 'পুরাই', 'ফালতু', 'সার্ভিস', 'দেয়']
    Truncating punctuation: ['চাল', 'ডাল', 'ইদানিং', 'পুরাই', 'ফালতু', 'সার্ভিস', 'দেয়']
    Truncating StopWords: ['চাল', 'ডাল', 'ইদানিং', 'পুরাই', 'ফালতু', 'সার্ভিস', 'দেয়']
    ***************************************************************************************
    Label:  0
    Sentence:  এদের নামে ভোক্তা অধিকারে মামলা করেছি কিছুদিন হয়। পোকা/পচা খেজুর দিয়েছিল।
    Afert Tokenizing:  ['এদের', 'নামে', 'ভোক্তা', 'অধিকারে', 'মামলা', 'করেছি', 'কিছুদিন', 'হয়', '।', 'পোকা/পচা', 'খেজুর', 'দিয়েছিল', '।']
    Truncating punctuation: ['এদের', 'নামে', 'ভোক্তা', 'অধিকারে', 'মামলা', 'করেছি', 'কিছুদিন', 'হয়', 'পোকা/পচা', 'খেজুর', 'দিয়েছিল']
    Truncating StopWords: ['নামে', 'ভোক্তা', 'অধিকারে', 'মামলা', 'করেছি', 'কিছুদিন', 'পোকা/পচা', 'খেজুর', 'দিয়েছিল']
    ***************************************************************************************
    Label:  1
    Sentence:  ই-কমার্স সেক্টর ঘুরে দাড়াচ্ছে আলহামদুলিল্লাহ
    Afert Tokenizing:  ['ই-কমার্স', 'সেক্টর', 'ঘুরে', 'দাড়াচ্ছে', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['ই-কমার্স', 'সেক্টর', 'ঘুরে', 'দাড়াচ্ছে', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['ই-কমার্স', 'সেক্টর', 'ঘুরে', 'দাড়াচ্ছে', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ আমিও রিফান্ড পাবো ইনশাআল্লাহ
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'আমিও', 'রিফান্ড', 'পাবো', 'ইনশাআল্লাহ']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আমিও', 'রিফান্ড', 'পাবো', 'ইনশাআল্লাহ']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'আমিও', 'রিফান্ড', 'পাবো', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা ফেরত দেয়নি কিউকম
    Afert Tokenizing:  ['আমার', 'টাকা', 'ফেরত', 'দেয়নি', 'কিউকম']
    Truncating punctuation: ['আমার', 'টাকা', 'ফেরত', 'দেয়নি', 'কিউকম']
    Truncating StopWords: ['টাকা', 'ফেরত', 'দেয়নি', 'কিউকম']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ই-কমার্স সেক্টর আবার ঘুরে দাড়াচ্ছে
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'ই-কমার্স', 'সেক্টর', 'আবার', 'ঘুরে', 'দাড়াচ্ছে']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ই-কমার্স', 'সেক্টর', 'আবার', 'ঘুরে', 'দাড়াচ্ছে']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ই-কমার্স', 'সেক্টর', 'ঘুরে', 'দাড়াচ্ছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ই-কমার্সের সুদিন ফিরতেছে আলহামদুলিল্লাহ
    Afert Tokenizing:  ['ই-কমার্সের', 'সুদিন', 'ফিরতেছে', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['ই-কমার্সের', 'সুদিন', 'ফিরতেছে', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['ই-কমার্সের', 'সুদিন', 'ফিরতেছে', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ। ইকমার্স সেক্টর টা ধীরে ধীরে ফিরতে শুরু করেছে
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '।', 'ইকমার্স', 'সেক্টর', 'টা', 'ধীরে', 'ধীরে', 'ফিরতে', 'শুরু', 'করেছে']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ইকমার্স', 'সেক্টর', 'টা', 'ধীরে', 'ধীরে', 'ফিরতে', 'শুরু', 'করেছে']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ইকমার্স', 'সেক্টর', 'টা', 'ধীরে', 'ধীরে', 'ফিরতে']
    ***************************************************************************************
    Label:  0
    Sentence:  ঐ সালা বাটপারের যদি একদিন মনের মত করে গালি দিতে পারতাম।
    Afert Tokenizing:  ['ঐ', 'সালা', 'বাটপারের', 'যদি', 'একদিন', 'মনের', 'মত', 'করে', 'গালি', 'দিতে', 'পারতাম', '।']
    Truncating punctuation: ['ঐ', 'সালা', 'বাটপারের', 'যদি', 'একদিন', 'মনের', 'মত', 'করে', 'গালি', 'দিতে', 'পারতাম']
    Truncating StopWords: ['সালা', 'বাটপারের', 'একদিন', 'মনের', 'মত', 'গালি', 'পারতাম']
    ***************************************************************************************
    Label:  0
    Sentence:   দারাজের বাটপারি দিন দিন বেড়েই চলছে।
    Afert Tokenizing:  ['দারাজের', 'বাটপারি', 'দিন', 'দিন', 'বেড়েই', 'চলছে', '।']
    Truncating punctuation: ['দারাজের', 'বাটপারি', 'দিন', 'দিন', 'বেড়েই', 'চলছে']
    Truncating StopWords: ['দারাজের', 'বাটপারি', 'বেড়েই', 'চলছে']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজে বর্তমানে ভোগান্তির শেষ নেই,আমার ৩ তারিখের কমপ্লেইনের কোন খোজখবর নেই
    Afert Tokenizing:  ['দারাজে', 'বর্তমানে', 'ভোগান্তির', 'শেষ', 'নেই,আমার', '৩', 'তারিখের', 'কমপ্লেইনের', 'কোন', 'খোজখবর', 'নেই']
    Truncating punctuation: ['দারাজে', 'বর্তমানে', 'ভোগান্তির', 'শেষ', 'নেই,আমার', '৩', 'তারিখের', 'কমপ্লেইনের', 'কোন', 'খোজখবর', 'নেই']
    Truncating StopWords: ['দারাজে', 'বর্তমানে', 'ভোগান্তির', 'শেষ', 'নেই,আমার', '৩', 'তারিখের', 'কমপ্লেইনের', 'খোজখবর', 'নেই']
    ***************************************************************************************
    Label:  0
    Sentence:  আল্লাহর দোহায় লাগে এই বিশ্ব বাটপারদের বিরুদ্ধে কিছু একটা করুন সবাই। এরা প্রকৃতপক্ষেই চিটার।
    Afert Tokenizing:  ['আল্লাহর', 'দোহায়', 'লাগে', 'এই', 'বিশ্ব', 'বাটপারদের', 'বিরুদ্ধে', 'কিছু', 'একটা', 'করুন', 'সবাই', '।', 'এরা', 'প্রকৃতপক্ষেই', 'চিটার', '।']
    Truncating punctuation: ['আল্লাহর', 'দোহায়', 'লাগে', 'এই', 'বিশ্ব', 'বাটপারদের', 'বিরুদ্ধে', 'কিছু', 'একটা', 'করুন', 'সবাই', 'এরা', 'প্রকৃতপক্ষেই', 'চিটার']
    Truncating StopWords: ['আল্লাহর', 'দোহায়', 'লাগে', 'বিশ্ব', 'বাটপারদের', 'বিরুদ্ধে', 'একটা', 'করুন', 'সবাই', 'প্রকৃতপক্ষেই', 'চিটার']
    ***************************************************************************************
    Label:  1
    Sentence:  এর মত ট্রাস্টেড ই-কমার্স সাইট বাংলাদেশে খুব কমই আছে!
    Afert Tokenizing:  ['এর', 'মত', 'ট্রাস্টেড', 'ই-কমার্স', 'সাইট', 'বাংলাদেশে', 'খুব', 'কমই', 'আছে', '!']
    Truncating punctuation: ['এর', 'মত', 'ট্রাস্টেড', 'ই-কমার্স', 'সাইট', 'বাংলাদেশে', 'খুব', 'কমই', 'আছে']
    Truncating StopWords: ['মত', 'ট্রাস্টেড', 'ই-কমার্স', 'সাইট', 'বাংলাদেশে', 'কমই']
    ***************************************************************************************
    Label:  1
    Sentence:  নিশ্চিন্তে নিয়ে নেন, আমি অলরেডি ইউজ করছি। এছাড়াও গত ৬-৭ বছরে অনেক প্রোডাক্ট নিয়েছি ওদের থেকে। সবগুলো ১০০% পারফেক্ট
    Afert Tokenizing:  ['নিশ্চিন্তে', 'নিয়ে', 'নেন', ',', 'আমি', 'অলরেডি', 'ইউজ', 'করছি', '।', 'এছাড়াও', 'গত', '৬-৭', 'বছরে', 'অনেক', 'প্রোডাক্ট', 'নিয়েছি', 'ওদের', 'থেকে', '।', 'সবগুলো', '১০০%', 'পারফেক্ট']
    Truncating punctuation: ['নিশ্চিন্তে', 'নিয়ে', 'নেন', 'আমি', 'অলরেডি', 'ইউজ', 'করছি', 'এছাড়াও', 'গত', '৬-৭', 'বছরে', 'অনেক', 'প্রোডাক্ট', 'নিয়েছি', 'ওদের', 'থেকে', 'সবগুলো', '১০০%', 'পারফেক্ট']
    Truncating StopWords: ['নিশ্চিন্তে', 'নেন', 'অলরেডি', 'ইউজ', 'করছি', 'এছাড়াও', 'গত', '৬-৭', 'বছরে', 'প্রোডাক্ট', 'নিয়েছি', 'সবগুলো', '১০০%', 'পারফেক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  আমিও ভুক্তভোগী, আসুন সবাই মিলে প্রতিবাদ করি
    Afert Tokenizing:  ['আমিও', 'ভুক্তভোগী', ',', 'আসুন', 'সবাই', 'মিলে', 'প্রতিবাদ', 'করি']
    Truncating punctuation: ['আমিও', 'ভুক্তভোগী', 'আসুন', 'সবাই', 'মিলে', 'প্রতিবাদ', 'করি']
    Truncating StopWords: ['আমিও', 'ভুক্তভোগী', 'আসুন', 'সবাই', 'মিলে', 'প্রতিবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  দেশের নাম্বার ওয়ান ধোকাবাজ
    Afert Tokenizing:  ['দেশের', 'নাম্বার', 'ওয়ান', 'ধোকাবাজ']
    Truncating punctuation: ['দেশের', 'নাম্বার', 'ওয়ান', 'ধোকাবাজ']
    Truncating StopWords: ['দেশের', 'নাম্বার', 'ওয়ান', 'ধোকাবাজ']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার রিভিউ দেখে আমিও অর্ডার করেছিলাম। বেশ ভাল স্বাদ। ধন্যবাদ ভাই।
    Afert Tokenizing:  ['আপনার', 'রিভিউ', 'দেখে', 'আমিও', 'অর্ডার', 'করেছিলাম', '।', 'বেশ', 'ভাল', 'স্বাদ', '।', 'ধন্যবাদ', 'ভাই', '।']
    Truncating punctuation: ['আপনার', 'রিভিউ', 'দেখে', 'আমিও', 'অর্ডার', 'করেছিলাম', 'বেশ', 'ভাল', 'স্বাদ', 'ধন্যবাদ', 'ভাই']
    Truncating StopWords: ['রিভিউ', 'আমিও', 'অর্ডার', 'করেছিলাম', 'ভাল', 'স্বাদ', 'ধন্যবাদ', 'ভাই']
    ***************************************************************************************
    Label:  1
    Sentence:  জুতা হাতে পেয়ে মনে হলো সত্যি জিতে গেছি, অনেক ভালো মানের জুতা বাকিটা ইউজ করে বুঝা যাবে
    Afert Tokenizing:  ['জুতা', 'হাতে', 'পেয়ে', 'মনে', 'হলো', 'সত্যি', 'জিতে', 'গেছি', ',', 'অনেক', 'ভালো', 'মানের', 'জুতা', 'বাকিটা', 'ইউজ', 'করে', 'বুঝা', 'যাবে']
    Truncating punctuation: ['জুতা', 'হাতে', 'পেয়ে', 'মনে', 'হলো', 'সত্যি', 'জিতে', 'গেছি', 'অনেক', 'ভালো', 'মানের', 'জুতা', 'বাকিটা', 'ইউজ', 'করে', 'বুঝা', 'যাবে']
    Truncating StopWords: ['জুতা', 'হাতে', 'পেয়ে', 'সত্যি', 'জিতে', 'গেছি', 'ভালো', 'মানের', 'জুতা', 'বাকিটা', 'ইউজ', 'বুঝা']
    ***************************************************************************************
    Label:  1
    Sentence:  এগিয়ে যাক আমাদের দেশের ই-কমার্স, বদলে যাক আমাদের ই-কমার্স অভিজ্ঞতা।
    Afert Tokenizing:  ['এগিয়ে', 'যাক', 'আমাদের', 'দেশের', 'ই-কমার্স', ',', 'বদলে', 'যাক', 'আমাদের', 'ই-কমার্স', 'অভিজ্ঞতা', '।']
    Truncating punctuation: ['এগিয়ে', 'যাক', 'আমাদের', 'দেশের', 'ই-কমার্স', 'বদলে', 'যাক', 'আমাদের', 'ই-কমার্স', 'অভিজ্ঞতা']
    Truncating StopWords: ['এগিয়ে', 'যাক', 'দেশের', 'ই-কমার্স', 'যাক', 'ই-কমার্স', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  0
    Sentence:  একে তো বক্স কেটে আসল প্রোডাক্ট সরিয়ে ভুল প্রোডাক্ট ঢুকিয়েছে তারপর এটাও নষ্ট প্রোডাক্ট
    Afert Tokenizing:  ['একে', 'তো', 'বক্স', 'কেটে', 'আসল', 'প্রোডাক্ট', 'সরিয়ে', 'ভুল', 'প্রোডাক্ট', 'ঢুকিয়েছে', 'তারপর', 'এটাও', 'নষ্ট', 'প্রোডাক্ট']
    Truncating punctuation: ['একে', 'তো', 'বক্স', 'কেটে', 'আসল', 'প্রোডাক্ট', 'সরিয়ে', 'ভুল', 'প্রোডাক্ট', 'ঢুকিয়েছে', 'তারপর', 'এটাও', 'নষ্ট', 'প্রোডাক্ট']
    Truncating StopWords: ['বক্স', 'কেটে', 'আসল', 'প্রোডাক্ট', 'সরিয়ে', 'ভুল', 'প্রোডাক্ট', 'ঢুকিয়েছে', 'এটাও', 'নষ্ট', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  তারা সারপ্রাইজ বক্স নামে কৌশল ভাবে প্রতারণা করে,আমি সরাসরি এর ভিক্টিম
    Afert Tokenizing:  ['তারা', 'সারপ্রাইজ', 'বক্স', 'নামে', 'কৌশল', 'ভাবে', 'প্রতারণা', 'করে,আমি', 'সরাসরি', 'এর', 'ভিক্টিম']
    Truncating punctuation: ['তারা', 'সারপ্রাইজ', 'বক্স', 'নামে', 'কৌশল', 'ভাবে', 'প্রতারণা', 'করে,আমি', 'সরাসরি', 'এর', 'ভিক্টিম']
    Truncating StopWords: ['সারপ্রাইজ', 'বক্স', 'নামে', 'কৌশল', 'প্রতারণা', 'করে,আমি', 'সরাসরি', 'ভিক্টিম']
    ***************************************************************************************
    Label:  0
    Sentence:  এরা আসলেই ধান্দাবাজ। এদের প্রডাক্ট কোয়ালিটিও বাজে।
    Afert Tokenizing:  ['এরা', 'আসলেই', 'ধান্দাবাজ', '।', 'এদের', 'প্রডাক্ট', 'কোয়ালিটিও', 'বাজে', '।']
    Truncating punctuation: ['এরা', 'আসলেই', 'ধান্দাবাজ', 'এদের', 'প্রডাক্ট', 'কোয়ালিটিও', 'বাজে']
    Truncating StopWords: ['আসলেই', 'ধান্দাবাজ', 'প্রডাক্ট', 'কোয়ালিটিও', 'বাজে']
    ***************************************************************************************
    Label:  0
    Sentence:  একটাও ভালো না
    Afert Tokenizing:  ['একটাও', 'ভালো', 'না']
    Truncating punctuation: ['একটাও', 'ভালো', 'না']
    Truncating StopWords: ['একটাও', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রডাক্ট  ডেলিভারীতে অনেক দেরি হয়েছে
    Afert Tokenizing:  ['প্রডাক্ট', 'ডেলিভারীতে', 'অনেক', 'দেরি', 'হয়েছে']
    Truncating punctuation: ['প্রডাক্ট', 'ডেলিভারীতে', 'অনেক', 'দেরি', 'হয়েছে']
    Truncating StopWords: ['প্রডাক্ট', 'ডেলিভারীতে', 'দেরি', 'হয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্টা প্যাকেজিং আরো ভালো হলে মনে হয় ভালো হতো
    Afert Tokenizing:  ['প্রোডাক্টা', 'প্যাকেজিং', 'আরো', 'ভালো', 'হলে', 'মনে', 'হয়', 'ভালো', 'হতো']
    Truncating punctuation: ['প্রোডাক্টা', 'প্যাকেজিং', 'আরো', 'ভালো', 'হলে', 'মনে', 'হয়', 'ভালো', 'হতো']
    Truncating StopWords: ['প্রোডাক্টা', 'প্যাকেজিং', 'আরো', 'ভালো', 'ভালো', 'হতো']
    ***************************************************************************************
    Label:  1
    Sentence:  মোনার্ক মার্ট থেকে ভালো এক্সপেরিয়েন্স ছিলো।
    Afert Tokenizing:  ['মোনার্ক', 'মার্ট', 'থেকে', 'ভালো', 'এক্সপেরিয়েন্স', 'ছিলো', '।']
    Truncating punctuation: ['মোনার্ক', 'মার্ট', 'থেকে', 'ভালো', 'এক্সপেরিয়েন্স', 'ছিলো']
    Truncating StopWords: ['মোনার্ক', 'মার্ট', 'ভালো', 'এক্সপেরিয়েন্স', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  অতিদ্রুত ও ক্যাশ অন ডেলিভারি পেয়েছি।
    Afert Tokenizing:  ['অতিদ্রুত', 'ও', 'ক্যাশ', 'অন', 'ডেলিভারি', 'পেয়েছি', '।']
    Truncating punctuation: ['অতিদ্রুত', 'ও', 'ক্যাশ', 'অন', 'ডেলিভারি', 'পেয়েছি']
    Truncating StopWords: ['অতিদ্রুত', 'ক্যাশ', 'অন', 'ডেলিভারি', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  সবাইকে বলব বাটপার থেকে দূরে থাকুন ২ টাকা বেশি গেলেও বাইরে থেকে কিনুন এরকম বাটপার ইর্কমাস থেকে দূরে থাকুন।
    Afert Tokenizing:  ['সবাইকে', 'বলব', 'বাটপার', 'থেকে', 'দূরে', 'থাকুন', '২', 'টাকা', 'বেশি', 'গেলেও', 'বাইরে', 'থেকে', 'কিনুন', 'এরকম', 'বাটপার', 'ইর্কমাস', 'থেকে', 'দূরে', 'থাকুন', '।']
    Truncating punctuation: ['সবাইকে', 'বলব', 'বাটপার', 'থেকে', 'দূরে', 'থাকুন', '২', 'টাকা', 'বেশি', 'গেলেও', 'বাইরে', 'থেকে', 'কিনুন', 'এরকম', 'বাটপার', 'ইর্কমাস', 'থেকে', 'দূরে', 'থাকুন']
    Truncating StopWords: ['সবাইকে', 'বলব', 'বাটপার', 'দূরে', 'থাকুন', '২', 'টাকা', 'বেশি', 'গেলেও', 'বাইরে', 'কিনুন', 'এরকম', 'বাটপার', 'ইর্কমাস', 'দূরে', 'থাকুন']
    ***************************************************************************************
    Label:  1
    Sentence:  আজ ডেলিভারী পেলাম। হাতে পেয়ে মুগ্ধ হয়েছি। এখন পর্যন্ত আমের জন্য এটি সেরা প্যাকেজিং বলতে হবে
    Afert Tokenizing:  ['আজ', 'ডেলিভারী', 'পেলাম', '।', 'হাতে', 'পেয়ে', 'মুগ্ধ', 'হয়েছি', '।', 'এখন', 'পর্যন্ত', 'আমের', 'জন্য', 'এটি', 'সেরা', 'প্যাকেজিং', 'বলতে', 'হবে']
    Truncating punctuation: ['আজ', 'ডেলিভারী', 'পেলাম', 'হাতে', 'পেয়ে', 'মুগ্ধ', 'হয়েছি', 'এখন', 'পর্যন্ত', 'আমের', 'জন্য', 'এটি', 'সেরা', 'প্যাকেজিং', 'বলতে', 'হবে']
    Truncating StopWords: ['ডেলিভারী', 'পেলাম', 'হাতে', 'পেয়ে', 'মুগ্ধ', 'হয়েছি', 'আমের', 'সেরা', 'প্যাকেজিং']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ এ ভুল প্রডাক্ট দিয়েছে এবং রিটার্ন নিচ্ছে না রিফান্ড ও করছে না, সেলার মেসেজ দেখেও কোন রিপ্লাই দিচ্ছে না,
    Afert Tokenizing:  ['দারাজ', 'এ', 'ভুল', 'প্রডাক্ট', 'দিয়েছে', 'এবং', 'রিটার্ন', 'নিচ্ছে', 'না', 'রিফান্ড', 'ও', 'করছে', 'না', ',', 'সেলার', 'মেসেজ', 'দেখেও', 'কোন', 'রিপ্লাই', 'দিচ্ছে', 'না', ',']
    Truncating punctuation: ['দারাজ', 'এ', 'ভুল', 'প্রডাক্ট', 'দিয়েছে', 'এবং', 'রিটার্ন', 'নিচ্ছে', 'না', 'রিফান্ড', 'ও', 'করছে', 'না', 'সেলার', 'মেসেজ', 'দেখেও', 'কোন', 'রিপ্লাই', 'দিচ্ছে', 'না']
    Truncating StopWords: ['দারাজ', 'ভুল', 'প্রডাক্ট', 'দিয়েছে', 'রিটার্ন', 'নিচ্ছে', 'না', 'রিফান্ড', 'না', 'সেলার', 'মেসেজ', 'দেখেও', 'রিপ্লাই', 'দিচ্ছে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেজিংটা সুন্দর
    Afert Tokenizing:  ['প্যাকেজিংটা', 'সুন্দর']
    Truncating punctuation: ['প্যাকেজিংটা', 'সুন্দর']
    Truncating StopWords: ['প্যাকেজিংটা', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ এতো সুন্দর একটা প্রডাক্ট দেয়ার জন্য।
    Afert Tokenizing:  ['ধন্যবাদ', 'এতো', 'সুন্দর', 'একটা', 'প্রডাক্ট', 'দেয়ার', 'জন্য', '।']
    Truncating punctuation: ['ধন্যবাদ', 'এতো', 'সুন্দর', 'একটা', 'প্রডাক্ট', 'দেয়ার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'এতো', 'সুন্দর', 'একটা', 'প্রডাক্ট', 'দেয়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  উনাদের ব্যবহার অত্যন্ত ভাল। শুভ কামনা রইলো আপনাদের জন্য।
    Afert Tokenizing:  ['উনাদের', 'ব্যবহার', 'অত্যন্ত', 'ভাল', '।', 'শুভ', 'কামনা', 'রইলো', 'আপনাদের', 'জন্য', '।']
    Truncating punctuation: ['উনাদের', 'ব্যবহার', 'অত্যন্ত', 'ভাল', 'শুভ', 'কামনা', 'রইলো', 'আপনাদের', 'জন্য']
    Truncating StopWords: ['উনাদের', 'অত্যন্ত', 'ভাল', 'শুভ', 'কামনা', 'রইলো', 'আপনাদের']
    ***************************************************************************************
    Label:  1
    Sentence:  মাত্রই পাঠাও কুরিয়ার আমাকে আপনাদের স্পোর্টস জার্সি দিয়ে গিলো অবশ্যই কোয়ালিটি যথেষ্ট ভালো
    Afert Tokenizing:  ['মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আমাকে', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    Truncating punctuation: ['মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আমাকে', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    Truncating StopWords: ['মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো লাগছে ভাই। যা দিয়েছেন আলহামদুলিল্লাহ
    Afert Tokenizing:  ['অনেক', 'ভালো', 'লাগছে', 'ভাই', '।', 'যা', 'দিয়েছেন', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['অনেক', 'ভালো', 'লাগছে', 'ভাই', 'যা', 'দিয়েছেন', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['ভালো', 'লাগছে', 'ভাই', 'দিয়েছেন', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  সঠিক সময়ে হাতে পেয়েছি। কাপড়ের মান ভালো, যা লেখা ছিল তাই দিয়েছে। নির্দ্বিধায় নিতে পারেন।
    Afert Tokenizing:  ['সঠিক', 'সময়ে', 'হাতে', 'পেয়েছি', '।', 'কাপড়ের', 'মান', 'ভালো', ',', 'যা', 'লেখা', 'ছিল', 'তাই', 'দিয়েছে', '।', 'নির্দ্বিধায়', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['সঠিক', 'সময়ে', 'হাতে', 'পেয়েছি', 'কাপড়ের', 'মান', 'ভালো', 'যা', 'লেখা', 'ছিল', 'তাই', 'দিয়েছে', 'নির্দ্বিধায়', 'নিতে', 'পারেন']
    Truncating StopWords: ['সঠিক', 'সময়ে', 'হাতে', 'পেয়েছি', 'কাপড়ের', 'মান', 'ভালো', 'লেখা', 'দিয়েছে', 'নির্দ্বিধায়']
    ***************************************************************************************
    Label:  1
    Sentence:  টি-শার্ট গুলো  পছন্দের  সব মিলিয়ে  কম দামে অসাধারণ     প্রোডাক্ট
    Afert Tokenizing:  ['টি-শার্ট', 'গুলো', 'পছন্দের', 'সব', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    Truncating punctuation: ['টি-শার্ট', 'গুলো', 'পছন্দের', 'সব', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    Truncating StopWords: ['টি-শার্ট', 'গুলো', 'পছন্দের', 'মিলিয়ে', 'কম', 'দামে', 'অসাধারণ', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  কাজের সাথে কথায় মিল পেয়েছি। আল্লাহ পাক আপনাদের বরকত দান করুক। জাযাকাল্লাহ খাইরান
    Afert Tokenizing:  ['কাজের', 'সাথে', 'কথায়', 'মিল', 'পেয়েছি', '।', 'আল্লাহ', 'পাক', 'আপনাদের', 'বরকত', 'দান', 'করুক', '।', 'জাযাকাল্লাহ', 'খাইরান']
    Truncating punctuation: ['কাজের', 'সাথে', 'কথায়', 'মিল', 'পেয়েছি', 'আল্লাহ', 'পাক', 'আপনাদের', 'বরকত', 'দান', 'করুক', 'জাযাকাল্লাহ', 'খাইরান']
    Truncating StopWords: ['কাজের', 'সাথে', 'কথায়', 'মিল', 'পেয়েছি', 'আল্লাহ', 'পাক', 'আপনাদের', 'বরকত', 'দান', 'করুক', 'জাযাকাল্লাহ', 'খাইরান']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ কথা এবং কাজে মিল রাখার জন্য।
    Afert Tokenizing:  ['ধন্যবাদ', 'কথা', 'এবং', 'কাজে', 'মিল', 'রাখার', 'জন্য', '।']
    Truncating punctuation: ['ধন্যবাদ', 'কথা', 'এবং', 'কাজে', 'মিল', 'রাখার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'কথা', 'মিল', 'রাখার']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট কোয়ালিটি আলহামদুলিল্লাহ  ডেলিভারী টাইমিং এবং পেইজের রেসপন্স ও অনেক ভালো। পার্সোনালি আমার অনেক ভাল্লাগসে
    Afert Tokenizing:  ['প্রোডাক্ট', 'কোয়ালিটি', 'আলহামদুলিল্লাহ', 'ডেলিভারী', 'টাইমিং', 'এবং', 'পেইজের', 'রেসপন্স', 'ও', 'অনেক', 'ভালো', '।', 'পার্সোনালি', 'আমার', 'অনেক', 'ভাল্লাগসে']
    Truncating punctuation: ['প্রোডাক্ট', 'কোয়ালিটি', 'আলহামদুলিল্লাহ', 'ডেলিভারী', 'টাইমিং', 'এবং', 'পেইজের', 'রেসপন্স', 'ও', 'অনেক', 'ভালো', 'পার্সোনালি', 'আমার', 'অনেক', 'ভাল্লাগসে']
    Truncating StopWords: ['প্রোডাক্ট', 'কোয়ালিটি', 'আলহামদুলিল্লাহ', 'ডেলিভারী', 'টাইমিং', 'পেইজের', 'রেসপন্স', 'ভালো', 'পার্সোনালি', 'ভাল্লাগসে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো লাগলো আপনাদের কাছ থেকে শপিং করে
    Afert Tokenizing:  ['অনেক', 'ভালো', 'লাগলো', 'আপনাদের', 'কাছ', 'থেকে', 'শপিং', 'করে']
    Truncating punctuation: ['অনেক', 'ভালো', 'লাগলো', 'আপনাদের', 'কাছ', 'থেকে', 'শপিং', 'করে']
    Truncating StopWords: ['ভালো', 'লাগলো', 'আপনাদের', 'শপিং']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার নিয়েছেন কিন্তু কখন পাবো বলছেন না, ইনবক্সে কেউ রিপ্লাই করছেন না
    Afert Tokenizing:  ['অর্ডার', 'নিয়েছেন', 'কিন্তু', 'কখন', 'পাবো', 'বলছেন', 'না', ',', 'ইনবক্সে', 'কেউ', 'রিপ্লাই', 'করছেন', 'না']
    Truncating punctuation: ['অর্ডার', 'নিয়েছেন', 'কিন্তু', 'কখন', 'পাবো', 'বলছেন', 'না', 'ইনবক্সে', 'কেউ', 'রিপ্লাই', 'করছেন', 'না']
    Truncating StopWords: ['অর্ডার', 'নিয়েছেন', 'কখন', 'পাবো', 'বলছেন', 'না', 'ইনবক্সে', 'রিপ্লাই', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ইনবক্স সার্ভিস একেবারেই খারাপ
    Afert Tokenizing:  ['আপনাদের', 'ইনবক্স', 'সার্ভিস', 'একেবারেই', 'খারাপ']
    Truncating punctuation: ['আপনাদের', 'ইনবক্স', 'সার্ভিস', 'একেবারেই', 'খারাপ']
    Truncating StopWords: ['আপনাদের', 'ইনবক্স', 'সার্ভিস', 'একেবারেই', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের নিয়ম মেনে আমি তিনদিন আগে অডার কনফার্ম করছি। এখনো পণ্য পাই নি। কবে পাবো?
    Afert Tokenizing:  ['আপনাদের', 'নিয়ম', 'মেনে', 'আমি', 'তিনদিন', 'আগে', 'অডার', 'কনফার্ম', 'করছি', '।', 'এখনো', 'পণ্য', 'পাই', 'নি', '।', 'কবে', 'পাবো', '?']
    Truncating punctuation: ['আপনাদের', 'নিয়ম', 'মেনে', 'আমি', 'তিনদিন', 'আগে', 'অডার', 'কনফার্ম', 'করছি', 'এখনো', 'পণ্য', 'পাই', 'নি', 'কবে', 'পাবো']
    Truncating StopWords: ['আপনাদের', 'নিয়ম', 'মেনে', 'তিনদিন', 'অডার', 'কনফার্ম', 'করছি', 'এখনো', 'পণ্য', 'পাই', 'নি', 'পাবো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের পণ্য গুলো অনেক ভাল। ছবির সাথে পুরো মিল আছে।
    Afert Tokenizing:  ['আপনাদের', 'পণ্য', 'গুলো', 'অনেক', 'ভাল', '।', 'ছবির', 'সাথে', 'পুরো', 'মিল', 'আছে', '।']
    Truncating punctuation: ['আপনাদের', 'পণ্য', 'গুলো', 'অনেক', 'ভাল', 'ছবির', 'সাথে', 'পুরো', 'মিল', 'আছে']
    Truncating StopWords: ['আপনাদের', 'পণ্য', 'গুলো', 'ভাল', 'ছবির', 'সাথে', 'পুরো', 'মিল']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের কাজ অনেক ভালো লেগেছে,  এই নিয়ে দুইবার নিলাম এবং সেটিসফাইড। শুভকামনা জানবেন
    Afert Tokenizing:  ['আপনাদের', 'কাজ', 'অনেক', 'ভালো', 'লেগেছে', ',', 'এই', 'নিয়ে', 'দুইবার', 'নিলাম', 'এবং', 'সেটিসফাইড', '।', 'শুভকামনা', 'জানবেন']
    Truncating punctuation: ['আপনাদের', 'কাজ', 'অনেক', 'ভালো', 'লেগেছে', 'এই', 'নিয়ে', 'দুইবার', 'নিলাম', 'এবং', 'সেটিসফাইড', 'শুভকামনা', 'জানবেন']
    Truncating StopWords: ['আপনাদের', 'ভালো', 'লেগেছে', 'দুইবার', 'নিলাম', 'সেটিসফাইড', 'শুভকামনা', 'জানবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের থেকে বুকিং দেয়া প্রডাক্ট এর কোয়ালিটি খুবই ভাল লেগেছে!
    Afert Tokenizing:  ['আপনাদের', 'থেকে', 'বুকিং', 'দেয়া', 'প্রডাক্ট', 'এর', 'কোয়ালিটি', 'খুবই', 'ভাল', 'লেগেছে', '!']
    Truncating punctuation: ['আপনাদের', 'থেকে', 'বুকিং', 'দেয়া', 'প্রডাক্ট', 'এর', 'কোয়ালিটি', 'খুবই', 'ভাল', 'লেগেছে']
    Truncating StopWords: ['আপনাদের', 'বুকিং', 'দেয়া', 'প্রডাক্ট', 'কোয়ালিটি', 'খুবই', 'ভাল', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আল্লাহামদুল্লাহ আজকে হাতে পেলাম। আপনাকে অনেক ধন্যবাদ
    Afert Tokenizing:  ['আল্লাহামদুল্লাহ', 'আজকে', 'হাতে', 'পেলাম', '।', 'আপনাকে', 'অনেক', 'ধন্যবাদ']
    Truncating punctuation: ['আল্লাহামদুল্লাহ', 'আজকে', 'হাতে', 'পেলাম', 'আপনাকে', 'অনেক', 'ধন্যবাদ']
    Truncating StopWords: ['আল্লাহামদুল্লাহ', 'আজকে', 'হাতে', 'পেলাম', 'আপনাকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো একটা প্রডাক্ট  সত্যিই এটা আমার খুব ভালো লেগেছে
    Afert Tokenizing:  ['খুব', 'ভালো', 'একটা', 'প্রডাক্ট', 'সত্যিই', 'এটা', 'আমার', 'খুব', 'ভালো', 'লেগেছে']
    Truncating punctuation: ['খুব', 'ভালো', 'একটা', 'প্রডাক্ট', 'সত্যিই', 'এটা', 'আমার', 'খুব', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['ভালো', 'একটা', 'প্রডাক্ট', 'সত্যিই', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর। যেটা চেয়েছি সেটাই পেয়েছি
    Afert Tokenizing:  ['অনেক', 'সুন্দর', '।', 'যেটা', 'চেয়েছি', 'সেটাই', 'পেয়েছি']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'যেটা', 'চেয়েছি', 'সেটাই', 'পেয়েছি']
    Truncating StopWords: ['সুন্দর', 'যেটা', 'চেয়েছি', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  ধূর, খুবই বাজে পণ্য।
    Afert Tokenizing:  ['ধূর', ',', 'খুবই', 'বাজে', 'পণ্য', '।']
    Truncating punctuation: ['ধূর', 'খুবই', 'বাজে', 'পণ্য']
    Truncating StopWords: ['ধূর', 'খুবই', 'বাজে', 'পণ্য']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ মাশা-আল্লাহ অনেক সুন্দর এবং খুবি ভালো মানের পণ্য।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'মাশা-আল্লাহ', 'অনেক', 'সুন্দর', 'এবং', 'খুবি', 'ভালো', 'মানের', 'পণ্য', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'মাশা-আল্লাহ', 'অনেক', 'সুন্দর', 'এবং', 'খুবি', 'ভালো', 'মানের', 'পণ্য']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'মাশা-আল্লাহ', 'সুন্দর', 'খুবি', 'ভালো', 'মানের', 'পণ্য']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি অনেক ভালো এর আগেও আমি কিনেছিলাম কিন্তু রিভিউ দিতে পারি নাই। সেলারকে ধন্যবাদ অসাধারণ কোয়ালিটি দেওয়ার জন্য।
    Afert Tokenizing:  ['কোয়ালিটি', 'অনেক', 'ভালো', 'এর', 'আগেও', 'আমি', 'কিনেছিলাম', 'কিন্তু', 'রিভিউ', 'দিতে', 'পারি', 'নাই', '।', 'সেলারকে', 'ধন্যবাদ', 'অসাধারণ', 'কোয়ালিটি', 'দেওয়ার', 'জন্য', '।']
    Truncating punctuation: ['কোয়ালিটি', 'অনেক', 'ভালো', 'এর', 'আগেও', 'আমি', 'কিনেছিলাম', 'কিন্তু', 'রিভিউ', 'দিতে', 'পারি', 'নাই', 'সেলারকে', 'ধন্যবাদ', 'অসাধারণ', 'কোয়ালিটি', 'দেওয়ার', 'জন্য']
    Truncating StopWords: ['কোয়ালিটি', 'ভালো', 'আগেও', 'কিনেছিলাম', 'রিভিউ', 'নাই', 'সেলারকে', 'ধন্যবাদ', 'অসাধারণ', 'কোয়ালিটি']
    ***************************************************************************************
    Label:  1
    Sentence:  মান ভালো, অন্য সবাই কিনতে পারেন।
    Afert Tokenizing:  ['মান', 'ভালো', ',', 'অন্য', 'সবাই', 'কিনতে', 'পারেন', '।']
    Truncating punctuation: ['মান', 'ভালো', 'অন্য', 'সবাই', 'কিনতে', 'পারেন']
    Truncating StopWords: ['মান', 'ভালো', 'সবাই', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:   ধন্যবাদ ভালো প্রোডাক্ট দেওয়ার জন্য।
    Afert Tokenizing:  ['ধন্যবাদ', 'ভালো', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য', '।']
    Truncating punctuation: ['ধন্যবাদ', 'ভালো', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'ভালো', 'প্রোডাক্ট', 'দেওয়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  পাঞ্জাবির কোয়ালিটি খুব ভালো আর তাদের সার্ভিসও খুব ভালো ছিলো
    Afert Tokenizing:  ['পাঞ্জাবির', 'কোয়ালিটি', 'খুব', 'ভালো', 'আর', 'তাদের', 'সার্ভিসও', 'খুব', 'ভালো', 'ছিলো']
    Truncating punctuation: ['পাঞ্জাবির', 'কোয়ালিটি', 'খুব', 'ভালো', 'আর', 'তাদের', 'সার্ভিসও', 'খুব', 'ভালো', 'ছিলো']
    Truncating StopWords: ['পাঞ্জাবির', 'কোয়ালিটি', 'ভালো', 'সার্ভিসও', 'ভালো', 'ছিলো']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবি বাজে কোয়ালিটি ভাই! ডিজাইন ভালো লাগছিল তাই কিনছিলাম এত দাম দিয়েও। বাট পুরাই বাজে
    Afert Tokenizing:  ['খুবি', 'বাজে', 'কোয়ালিটি', 'ভাই', '!', 'ডিজাইন', 'ভালো', 'লাগছিল', 'তাই', 'কিনছিলাম', 'এত', 'দাম', 'দিয়েও', '।', 'বাট', 'পুরাই', 'বাজে']
    Truncating punctuation: ['খুবি', 'বাজে', 'কোয়ালিটি', 'ভাই', 'ডিজাইন', 'ভালো', 'লাগছিল', 'তাই', 'কিনছিলাম', 'এত', 'দাম', 'দিয়েও', 'বাট', 'পুরাই', 'বাজে']
    Truncating StopWords: ['খুবি', 'বাজে', 'কোয়ালিটি', 'ভাই', 'ডিজাইন', 'ভালো', 'লাগছিল', 'কিনছিলাম', 'দাম', 'দিয়েও', 'বাট', 'পুরাই', 'বাজে']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই বাজে সার্ভিস। ধরা খেতে চাইলে অর্ডার দিন
    Afert Tokenizing:  ['খুবই', 'বাজে', 'সার্ভিস', '।', 'ধরা', 'খেতে', 'চাইলে', 'অর্ডার', 'দিন']
    Truncating punctuation: ['খুবই', 'বাজে', 'সার্ভিস', 'ধরা', 'খেতে', 'চাইলে', 'অর্ডার', 'দিন']
    Truncating StopWords: ['খুবই', 'বাজে', 'সার্ভিস', 'খেতে', 'চাইলে', 'অর্ডার']
    ***************************************************************************************
    Label:  0
    Sentence:  এত বাজে কাপড়!! ডেলিভারী সার্ভিস তো একটুও ভাল না। অর্ডার দেই একটা,পাই আরেকটা। চেঞ্জ করে দেওয়ার নামে ফটকামী। একচুয়ালি দে ডোন্ট নো হাও টু রান আ  বিজনেস!!
    Afert Tokenizing:  ['এত', 'বাজে', 'কাপড়!', '!', 'ডেলিভারী', 'সার্ভিস', 'তো', 'একটুও', 'ভাল', 'না', '।', 'অর্ডার', 'দেই', 'একটা,পাই', 'আরেকটা', '।', 'চেঞ্জ', 'করে', 'দেওয়ার', 'নামে', 'ফটকামী', '।', 'একচুয়ালি', 'দে', 'ডোন্ট', 'নো', 'হাও', 'টু', 'রান', 'আ', 'বিজনেস!', '!']
    Truncating punctuation: ['এত', 'বাজে', 'কাপড়!', 'ডেলিভারী', 'সার্ভিস', 'তো', 'একটুও', 'ভাল', 'না', 'অর্ডার', 'দেই', 'একটা,পাই', 'আরেকটা', 'চেঞ্জ', 'করে', 'দেওয়ার', 'নামে', 'ফটকামী', 'একচুয়ালি', 'দে', 'ডোন্ট', 'নো', 'হাও', 'টু', 'রান', 'আ', 'বিজনেস!']
    Truncating StopWords: ['বাজে', 'কাপড়!', 'ডেলিভারী', 'সার্ভিস', 'একটুও', 'ভাল', 'না', 'অর্ডার', 'দেই', 'একটা,পাই', 'আরেকটা', 'চেঞ্জ', 'দেওয়ার', 'নামে', 'ফটকামী', 'একচুয়ালি', 'দে', 'ডোন্ট', 'নো', 'হাও', 'টু', 'রান', 'আ', 'বিজনেস!']
    ***************************************************************************************
    Label:  1
    Sentence:  যদি এই পেজ এবং প্রোডাক্ট  নিয়ে কিছু বলতে হয় এক কথায় অসাধারণ। দুইবার এখান থেকে প্রোডাক্ট নিয়েছি দুই বারই খুব তারাতারি  পেয়েছি এবং প্রোডাক্টও এক কথায় যেমন ছবিতে দেওয়া আছে তেমন ই
    Afert Tokenizing:  ['যদি', 'এই', 'পেজ', 'এবং', 'প্রোডাক্ট', 'নিয়ে', 'কিছু', 'বলতে', 'হয়', 'এক', 'কথায়', 'অসাধারণ', '।', 'দুইবার', 'এখান', 'থেকে', 'প্রোডাক্ট', 'নিয়েছি', 'দুই', 'বারই', 'খুব', 'তারাতারি', 'পেয়েছি', 'এবং', 'প্রোডাক্টও', 'এক', 'কথায়', 'যেমন', 'ছবিতে', 'দেওয়া', 'আছে', 'তেমন', 'ই']
    Truncating punctuation: ['যদি', 'এই', 'পেজ', 'এবং', 'প্রোডাক্ট', 'নিয়ে', 'কিছু', 'বলতে', 'হয়', 'এক', 'কথায়', 'অসাধারণ', 'দুইবার', 'এখান', 'থেকে', 'প্রোডাক্ট', 'নিয়েছি', 'দুই', 'বারই', 'খুব', 'তারাতারি', 'পেয়েছি', 'এবং', 'প্রোডাক্টও', 'এক', 'কথায়', 'যেমন', 'ছবিতে', 'দেওয়া', 'আছে', 'তেমন', 'ই']
    Truncating StopWords: ['পেজ', 'প্রোডাক্ট', 'এক', 'কথায়', 'অসাধারণ', 'দুইবার', 'এখান', 'প্রোডাক্ট', 'নিয়েছি', 'বারই', 'তারাতারি', 'পেয়েছি', 'প্রোডাক্টও', 'এক', 'কথায়', 'ছবিতে']
    ***************************************************************************************
    Label:  1
    Sentence:  গ্রেট  একদম পারফেক্ট সার্ভিস!!
    Afert Tokenizing:  ['গ্রেট', 'একদম', 'পারফেক্ট', 'সার্ভিস!', '!']
    Truncating punctuation: ['গ্রেট', 'একদম', 'পারফেক্ট', 'সার্ভিস!']
    Truncating StopWords: ['গ্রেট', 'একদম', 'পারফেক্ট', 'সার্ভিস!']
    ***************************************************************************************
    Label:  1
    Sentence:  পন্যের মান খুব ভাল । প্রাইস একটু হাই মনে হতে পারে কিন্তু পন্যের মান দেখলে সেটাকে যথাযথই মনে হবে ।
    Afert Tokenizing:  ['পন্যের', 'মান', 'খুব', 'ভাল', '', '।', 'প্রাইস', 'একটু', 'হাই', 'মনে', 'হতে', 'পারে', 'কিন্তু', 'পন্যের', 'মান', 'দেখলে', 'সেটাকে', 'যথাযথই', 'মনে', 'হবে', '', '।']
    Truncating punctuation: ['পন্যের', 'মান', 'খুব', 'ভাল', '', 'প্রাইস', 'একটু', 'হাই', 'মনে', 'হতে', 'পারে', 'কিন্তু', 'পন্যের', 'মান', 'দেখলে', 'সেটাকে', 'যথাযথই', 'মনে', 'হবে', '']
    Truncating StopWords: ['পন্যের', 'মান', 'ভাল', '', 'প্রাইস', 'একটু', 'হাই', 'পন্যের', 'মান', 'দেখলে', 'সেটাকে', 'যথাযথই', '']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর এবং অনেক ভালো মানের ক্যাপ! চাইলে আপনারা ও উনাদের থেকে ক্যাপ নিতে পারেন কোনো টেনশন ছাডা
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'এবং', 'অনেক', 'ভালো', 'মানের', 'ক্যাপ', '!', 'চাইলে', 'আপনারা', 'ও', 'উনাদের', 'থেকে', 'ক্যাপ', 'নিতে', 'পারেন', 'কোনো', 'টেনশন', 'ছাডা']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'এবং', 'অনেক', 'ভালো', 'মানের', 'ক্যাপ', 'চাইলে', 'আপনারা', 'ও', 'উনাদের', 'থেকে', 'ক্যাপ', 'নিতে', 'পারেন', 'কোনো', 'টেনশন', 'ছাডা']
    Truncating StopWords: ['সুন্দর', 'ভালো', 'মানের', 'ক্যাপ', 'চাইলে', 'আপনারা', 'উনাদের', 'ক্যাপ', 'টেনশন', 'ছাডা']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর ক্যাপ,,,একশো তে একশো! আমি একটা নিয়েছি
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'ক্যাপ,,,একশো', 'তে', 'একশো', '!', 'আমি', 'একটা', 'নিয়েছি']
    Truncating punctuation: ['খুব', 'সুন্দর', 'ক্যাপ,,,একশো', 'তে', 'একশো', 'আমি', 'একটা', 'নিয়েছি']
    Truncating StopWords: ['সুন্দর', 'ক্যাপ,,,একশো', 'তে', 'একশো', 'একটা', 'নিয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রিমিয়াম প্যাকেজিং
    Afert Tokenizing:  ['প্রিমিয়াম', 'প্যাকেজিং']
    Truncating punctuation: ['প্রিমিয়াম', 'প্যাকেজিং']
    Truncating StopWords: ['প্রিমিয়াম', 'প্যাকেজিং']
    ***************************************************************************************
    Label:  1
    Sentence:  ওনাদের প্রোডাক্ট কোয়ালিটি যথেষ্ট ভালো। সময়মতো প্রোডাক্ট টি হাতে এসে পৌঁছেছে। যারা ভালো কোয়ালিটির ক্যাপ খুঁজছেন। তাদের প্রোডাক্ট গুলি ট্রাই করতে পারেন।
    Afert Tokenizing:  ['ওনাদের', 'প্রোডাক্ট', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো', '।', 'সময়মতো', 'প্রোডাক্ট', 'টি', 'হাতে', 'এসে', 'পৌঁছেছে', '।', 'যারা', 'ভালো', 'কোয়ালিটির', 'ক্যাপ', 'খুঁজছেন', '।', 'তাদের', 'প্রোডাক্ট', 'গুলি', 'ট্রাই', 'করতে', 'পারেন', '।']
    Truncating punctuation: ['ওনাদের', 'প্রোডাক্ট', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো', 'সময়মতো', 'প্রোডাক্ট', 'টি', 'হাতে', 'এসে', 'পৌঁছেছে', 'যারা', 'ভালো', 'কোয়ালিটির', 'ক্যাপ', 'খুঁজছেন', 'তাদের', 'প্রোডাক্ট', 'গুলি', 'ট্রাই', 'করতে', 'পারেন']
    Truncating StopWords: ['ওনাদের', 'প্রোডাক্ট', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো', 'সময়মতো', 'প্রোডাক্ট', 'হাতে', 'পৌঁছেছে', 'ভালো', 'কোয়ালিটির', 'ক্যাপ', 'খুঁজছেন', 'প্রোডাক্ট', 'ট্রাই']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ তোমাদেরকে
    Afert Tokenizing:  ['ধন্যবাদ', 'তোমাদেরকে']
    Truncating punctuation: ['ধন্যবাদ', 'তোমাদেরকে']
    Truncating StopWords: ['ধন্যবাদ', 'তোমাদেরকে']
    ***************************************************************************************
    Label:  1
    Sentence:  তোমাদের মাধ্যমে আমি অনেক কিছু কেনা-বেচার সক্ষম হয়েছি
    Afert Tokenizing:  ['তোমাদের', 'মাধ্যমে', 'আমি', 'অনেক', 'কিছু', 'কেনা-বেচার', 'সক্ষম', 'হয়েছি']
    Truncating punctuation: ['তোমাদের', 'মাধ্যমে', 'আমি', 'অনেক', 'কিছু', 'কেনা-বেচার', 'সক্ষম', 'হয়েছি']
    Truncating StopWords: ['তোমাদের', 'কেনা-বেচার', 'সক্ষম', 'হয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি প্রতারিত হয়েছিলাম, এই রকম ভাবে
    Afert Tokenizing:  ['আমি', 'প্রতারিত', 'হয়েছিলাম', ',', 'এই', 'রকম', 'ভাবে']
    Truncating punctuation: ['আমি', 'প্রতারিত', 'হয়েছিলাম', 'এই', 'রকম', 'ভাবে']
    Truncating StopWords: ['প্রতারিত', 'হয়েছিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব উপকার হইলো
    Afert Tokenizing:  ['খুব', 'উপকার', 'হইলো']
    Truncating punctuation: ['খুব', 'উপকার', 'হইলো']
    Truncating StopWords: ['উপকার', 'হইলো']
    ***************************************************************************************
    Label:  1
    Sentence:  মাস্ল আজ ৩দিন যাবত ব্যবহার করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম
    Afert Tokenizing:  ['মাস্ল', 'আজ', '৩দিন', 'যাবত', 'ব্যবহার', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    Truncating punctuation: ['মাস্ল', 'আজ', '৩দিন', 'যাবত', 'ব্যবহার', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    Truncating StopWords: ['মাস্ল', '৩দিন', 'যাবত', 'করছি।।খুবই', 'ভাল', 'কোয়ালিটি।।পড়তেও', 'আরাম']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি খুব ভালো। এটা পেয়ে খুশি।
    Afert Tokenizing:  ['কোয়ালিটি', 'খুব', 'ভালো', '।', 'এটা', 'পেয়ে', 'খুশি', '।']
    Truncating punctuation: ['কোয়ালিটি', 'খুব', 'ভালো', 'এটা', 'পেয়ে', 'খুশি']
    Truncating StopWords: ['কোয়ালিটি', 'ভালো', 'খুশি']
    ***************************************************************************************
    Label:  1
    Sentence:  পরিতৃপ্ত
    Afert Tokenizing:  ['পরিতৃপ্ত']
    Truncating punctuation: ['পরিতৃপ্ত']
    Truncating StopWords: ['পরিতৃপ্ত']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর মুখোশ। আমি কয়েকটি জিনিস সংগ্রহ করেছি। এটি ব্যবহারে সত্যিই খুব দীর্ঘস্থায়ী।
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'মুখোশ', '।', 'আমি', 'কয়েকটি', 'জিনিস', 'সংগ্রহ', 'করেছি', '।', 'এটি', 'ব্যবহারে', 'সত্যিই', 'খুব', 'দীর্ঘস্থায়ী', '।']
    Truncating punctuation: ['খুব', 'সুন্দর', 'মুখোশ', 'আমি', 'কয়েকটি', 'জিনিস', 'সংগ্রহ', 'করেছি', 'এটি', 'ব্যবহারে', 'সত্যিই', 'খুব', 'দীর্ঘস্থায়ী']
    Truncating StopWords: ['সুন্দর', 'মুখোশ', 'জিনিস', 'সংগ্রহ', 'করেছি', 'ব্যবহারে', 'সত্যিই', 'দীর্ঘস্থায়ী']
    ***************************************************************************************
    Label:  1
    Sentence:  এটা ভালো. আমি একটি ব্যবহার করছি.
    Afert Tokenizing:  ['এটা', 'ভালো', '.', 'আমি', 'একটি', 'ব্যবহার', 'করছি', '.']
    Truncating punctuation: ['এটা', 'ভালো', 'আমি', 'একটি', 'ব্যবহার', 'করছি']
    Truncating StopWords: ['ভালো', 'করছি']
    ***************************************************************************************
    Label:  1
    Sentence:  দুর্দান্ত সৃজনশীল দল.....অসাধারণ কাজ
    Afert Tokenizing:  ['দুর্দান্ত', 'সৃজনশীল', 'দল.....অসাধারণ', 'কাজ']
    Truncating punctuation: ['দুর্দান্ত', 'সৃজনশীল', 'দল.....অসাধারণ', 'কাজ']
    Truncating StopWords: ['দুর্দান্ত', 'সৃজনশীল', 'দল.....অসাধারণ']
    ***************************************************************************************
    Label:  0
    Sentence:  এ সমস্যা কেন...?
    Afert Tokenizing:  ['এ', 'সমস্যা', 'কেন...', '?']
    Truncating punctuation: ['এ', 'সমস্যা', 'কেন...']
    Truncating StopWords: ['সমস্যা', 'কেন...']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি মাত্র 2 টি-শার্ট অর্ডার করেছি এবং বিকাশের মাধ্যমে পেমেন্ট করেছি। কিন্তু কোনো নগদ ফেরত পাননি। এটা আমার প্রথম অর্ডার. তাই আগে নগদ ফেরত পাওয়ার কোনো উপায় নেই।
    Afert Tokenizing:  ['আমি', 'মাত্র', '2', 'টি-শার্ট', 'অর্ডার', 'করেছি', 'এবং', 'বিকাশের', 'মাধ্যমে', 'পেমেন্ট', 'করেছি', '।', 'কিন্তু', 'কোনো', 'নগদ', 'ফেরত', 'পাননি', '।', 'এটা', 'আমার', 'প্রথম', 'অর্ডার', '.', 'তাই', 'আগে', 'নগদ', 'ফেরত', 'পাওয়ার', 'কোনো', 'উপায়', 'নেই', '।']
    Truncating punctuation: ['আমি', 'মাত্র', '2', 'টি-শার্ট', 'অর্ডার', 'করেছি', 'এবং', 'বিকাশের', 'মাধ্যমে', 'পেমেন্ট', 'করেছি', 'কিন্তু', 'কোনো', 'নগদ', 'ফেরত', 'পাননি', 'এটা', 'আমার', 'প্রথম', 'অর্ডার', 'তাই', 'আগে', 'নগদ', 'ফেরত', 'পাওয়ার', 'কোনো', 'উপায়', 'নেই']
    Truncating StopWords: ['2', 'টি-শার্ট', 'অর্ডার', 'করেছি', 'বিকাশের', 'পেমেন্ট', 'করেছি', 'নগদ', 'ফেরত', 'পাননি', 'অর্ডার', 'নগদ', 'ফেরত', 'পাওয়ার', 'উপায়', 'নেই']
    ***************************************************************************************
    Label:  1
    Sentence:  এইটা ডিসাইন করে টি-শার্ট চলবে ভালো
    Afert Tokenizing:  ['এইটা', 'ডিসাইন', 'করে', 'টি-শার্ট', 'চলবে', 'ভালো']
    Truncating punctuation: ['এইটা', 'ডিসাইন', 'করে', 'টি-শার্ট', 'চলবে', 'ভালো']
    Truncating StopWords: ['এইটা', 'ডিসাইন', 'টি-শার্ট', 'চলবে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রিয় ব্র্যান্ড
    Afert Tokenizing:  ['প্রিয়', 'ব্র্যান্ড']
    Truncating punctuation: ['প্রিয়', 'ব্র্যান্ড']
    Truncating StopWords: ['প্রিয়', 'ব্র্যান্ড']
    ***************************************************************************************
    Label:  1
    Sentence:  এটার ফ্রি ডেলিভারি আগে জানলে অবশ্যই এটাই দিতাম
    Afert Tokenizing:  ['এটার', 'ফ্রি', 'ডেলিভারি', 'আগে', 'জানলে', 'অবশ্যই', 'এটাই', 'দিতাম']
    Truncating punctuation: ['এটার', 'ফ্রি', 'ডেলিভারি', 'আগে', 'জানলে', 'অবশ্যই', 'এটাই', 'দিতাম']
    Truncating StopWords: ['এটার', 'ফ্রি', 'ডেলিভারি', 'জানলে', 'অবশ্যই', 'দিতাম']
    ***************************************************************************************
    Label:  1
    Sentence:  ফেব্রিলাইভ এটা আগে বলা দরকার ছিলো আপনাদের পক্ষ থেকে  মাত্রই পাঠাও কুরিয়ার আমাকে আপনাদের স্পোর্টস জার্সি দিয়ে গিলো অবশ্যই কোয়ালিটি যথেষ্ট ভালো
    Afert Tokenizing:  ['ফেব্রিলাইভ', 'এটা', 'আগে', 'বলা', 'দরকার', 'ছিলো', 'আপনাদের', 'পক্ষ', 'থেকে', 'মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আমাকে', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    Truncating punctuation: ['ফেব্রিলাইভ', 'এটা', 'আগে', 'বলা', 'দরকার', 'ছিলো', 'আপনাদের', 'পক্ষ', 'থেকে', 'মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আমাকে', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    Truncating StopWords: ['ফেব্রিলাইভ', 'দরকার', 'ছিলো', 'আপনাদের', 'পক্ষ', 'মাত্রই', 'পাঠাও', 'কুরিয়ার', 'আপনাদের', 'স্পোর্টস', 'জার্সি', 'দিয়ে', 'গিলো', 'অবশ্যই', 'কোয়ালিটি', 'যথেষ্ট', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করতে গেলে ডেলিবারী চার্জ দেখায় নাহ।
    Afert Tokenizing:  ['অর্ডার', 'করতে', 'গেলে', 'ডেলিবারী', 'চার্জ', 'দেখায়', 'নাহ', '।']
    Truncating punctuation: ['অর্ডার', 'করতে', 'গেলে', 'ডেলিবারী', 'চার্জ', 'দেখায়', 'নাহ']
    Truncating StopWords: ['অর্ডার', 'ডেলিবারী', 'চার্জ', 'দেখায়', 'নাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি এটা কিনেছি. সত্যিই খুব সুন্দর.
    Afert Tokenizing:  ['আমি', 'এটা', 'কিনেছি', '.', 'সত্যিই', 'খুব', 'সুন্দর', '.']
    Truncating punctuation: ['আমি', 'এটা', 'কিনেছি', 'সত্যিই', 'খুব', 'সুন্দর']
    Truncating StopWords: ['কিনেছি', 'সত্যিই', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  গুণমান দুর্দান্ত।
    Afert Tokenizing:  ['গুণমান', 'দুর্দান্ত', '।']
    Truncating punctuation: ['গুণমান', 'দুর্দান্ত']
    Truncating StopWords: ['গুণমান', 'দুর্দান্ত']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ.. যা চাইসি তার থেকেও অনেক ভালো.. কাপড় তা কোয়ালিটিফুল
    Afert Tokenizing:  ['আলহামদুলিল্লাহ.', '.', 'যা', 'চাইসি', 'তার', 'থেকেও', 'অনেক', 'ভালো.', '.', 'কাপড়', 'তা', 'কোয়ালিটিফুল']
    Truncating punctuation: ['আলহামদুলিল্লাহ.', 'যা', 'চাইসি', 'তার', 'থেকেও', 'অনেক', 'ভালো.', 'কাপড়', 'তা', 'কোয়ালিটিফুল']
    Truncating StopWords: ['আলহামদুলিল্লাহ.', 'চাইসি', 'ভালো.', 'কাপড়', 'কোয়ালিটিফুল']
    ***************************************************************************************
    Label:  1
    Sentence:   ডেলিভারি চার্জ ফ্রি
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'ফ্রি']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'ফ্রি']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'ফ্রি']
    ***************************************************************************************
    Label:  0
    Sentence:  কাপড়ের মূল্য 585tk নয়। আজকে নিয়ে এসে হতাশ হয়েছি
    Afert Tokenizing:  ['কাপড়ের', 'মূল্য', '585tk', 'নয়', '।', 'আজকে', 'নিয়ে', 'এসে', 'হতাশ', 'হয়েছি']
    Truncating punctuation: ['কাপড়ের', 'মূল্য', '585tk', 'নয়', 'আজকে', 'নিয়ে', 'এসে', 'হতাশ', 'হয়েছি']
    Truncating StopWords: ['কাপড়ের', 'মূল্য', '585tk', 'নয়', 'আজকে', 'হতাশ', 'হয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি Fabrilife এর একজন বড় ভক্ত
    Afert Tokenizing:  ['আমি', 'Fabrilife', 'এর', 'একজন', 'বড়', 'ভক্ত']
    Truncating punctuation: ['আমি', 'Fabrilife', 'এর', 'একজন', 'বড়', 'ভক্ত']
    Truncating StopWords: ['Fabrilife', 'একজন', 'বড়', 'ভক্ত']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ ডিজাইন
    Afert Tokenizing:  ['অসাধারণ', 'ডিজাইন']
    Truncating punctuation: ['অসাধারণ', 'ডিজাইন']
    Truncating StopWords: ['অসাধারণ', 'ডিজাইন']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার প্রশংসার জন্য আপনাকে অনেক ধন্যবাদ
    Afert Tokenizing:  ['আপনার', 'প্রশংসার', 'জন্য', 'আপনাকে', 'অনেক', 'ধন্যবাদ']
    Truncating punctuation: ['আপনার', 'প্রশংসার', 'জন্য', 'আপনাকে', 'অনেক', 'ধন্যবাদ']
    Truncating StopWords: ['প্রশংসার', 'আপনাকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার সুস্বাদু নকশা। ভালবাসা
    Afert Tokenizing:  ['চমৎকার', 'সুস্বাদু', 'নকশা', '।', 'ভালবাসা']
    Truncating punctuation: ['চমৎকার', 'সুস্বাদু', 'নকশা', 'ভালবাসা']
    Truncating StopWords: ['চমৎকার', 'সুস্বাদু', 'নকশা', 'ভালবাসা']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রতিটি নকশা পছন্দ. প্রশংসা করা
    Afert Tokenizing:  ['প্রতিটি', 'নকশা', 'পছন্দ', '.', 'প্রশংসা', 'করা']
    Truncating punctuation: ['প্রতিটি', 'নকশা', 'পছন্দ', 'প্রশংসা', 'করা']
    Truncating StopWords: ['প্রতিটি', 'নকশা', 'পছন্দ', 'প্রশংসা']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার সংগ্রহ
    Afert Tokenizing:  ['চমৎকার', 'সংগ্রহ']
    Truncating punctuation: ['চমৎকার', 'সংগ্রহ']
    Truncating StopWords: ['চমৎকার', 'সংগ্রহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি কিছু পণ্য অর্ডার. আজ আমি এটা পেয়েছিলাম. এটা সত্যিই যে ডেলিভারি অন্যান্য শোরুমের তুলনায় খুব দ্রুত। গুণমান ভাল বলে মনে হচ্ছে। কিন্তু দাম আমাদের মধ্যে যুক্তিসঙ্গত নয়।
    Afert Tokenizing:  ['আমি', 'কিছু', 'পণ্য', 'অর্ডার', '.', 'আজ', 'আমি', 'এটা', 'পেয়েছিলাম', '.', 'এটা', 'সত্যিই', 'যে', 'ডেলিভারি', 'অন্যান্য', 'শোরুমের', 'তুলনায়', 'খুব', 'দ্রুত', '।', 'গুণমান', 'ভাল', 'বলে', 'মনে', 'হচ্ছে', '।', 'কিন্তু', 'দাম', 'আমাদের', 'মধ্যে', 'যুক্তিসঙ্গত', 'নয়', '।']
    Truncating punctuation: ['আমি', 'কিছু', 'পণ্য', 'অর্ডার', 'আজ', 'আমি', 'এটা', 'পেয়েছিলাম', 'এটা', 'সত্যিই', 'যে', 'ডেলিভারি', 'অন্যান্য', 'শোরুমের', 'তুলনায়', 'খুব', 'দ্রুত', 'গুণমান', 'ভাল', 'বলে', 'মনে', 'হচ্ছে', 'কিন্তু', 'দাম', 'আমাদের', 'মধ্যে', 'যুক্তিসঙ্গত', 'নয়']
    Truncating StopWords: ['পণ্য', 'অর্ডার', 'পেয়েছিলাম', 'সত্যিই', 'ডেলিভারি', 'অন্যান্য', 'শোরুমের', 'তুলনায়', 'দ্রুত', 'গুণমান', 'ভাল', 'দাম', 'যুক্তিসঙ্গত', 'নয়']
    ***************************************************************************************
    Label:  1
    Sentence:  বেশ ভালো তবে দামটা একটু বেশি মনে হচ্ছে
    Afert Tokenizing:  ['বেশ', 'ভালো', 'তবে', 'দামটা', 'একটু', 'বেশি', 'মনে', 'হচ্ছে']
    Truncating punctuation: ['বেশ', 'ভালো', 'তবে', 'দামটা', 'একটু', 'বেশি', 'মনে', 'হচ্ছে']
    Truncating StopWords: ['ভালো', 'দামটা', 'একটু', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্যের গুণমান এবং পরিষেবা অনুসারে মূল্য একেবারে যুক্তিসঙ্গত।
    Afert Tokenizing:  ['পণ্যের', 'গুণমান', 'এবং', 'পরিষেবা', 'অনুসারে', 'মূল্য', 'একেবারে', 'যুক্তিসঙ্গত', '।']
    Truncating punctuation: ['পণ্যের', 'গুণমান', 'এবং', 'পরিষেবা', 'অনুসারে', 'মূল্য', 'একেবারে', 'যুক্তিসঙ্গত']
    Truncating StopWords: ['পণ্যের', 'গুণমান', 'পরিষেবা', 'অনুসারে', 'মূল্য', 'একেবারে', 'যুক্তিসঙ্গত']
    ***************************************************************************************
    Label:  1
    Sentence:  হতে পারে! কিন্তু এটা সত্য যে আপনার পণ্যের গুণমান সত্যিই ভিন্ন।
    Afert Tokenizing:  ['হতে', 'পারে', '!', 'কিন্তু', 'এটা', 'সত্য', 'যে', 'আপনার', 'পণ্যের', 'গুণমান', 'সত্যিই', 'ভিন্ন', '।']
    Truncating punctuation: ['হতে', 'পারে', 'কিন্তু', 'এটা', 'সত্য', 'যে', 'আপনার', 'পণ্যের', 'গুণমান', 'সত্যিই', 'ভিন্ন']
    Truncating StopWords: ['সত্য', 'পণ্যের', 'গুণমান', 'সত্যিই', 'ভিন্ন']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার ফ্যাশন পছন্দ চমত্কার
    Afert Tokenizing:  ['আপনার', 'ফ্যাশন', 'পছন্দ', 'চমত্কার']
    Truncating punctuation: ['আপনার', 'ফ্যাশন', 'পছন্দ', 'চমত্কার']
    Truncating StopWords: ['ফ্যাশন', 'পছন্দ', 'চমত্কার']
    ***************************************************************************************
    Label:  1
    Sentence:  আকর্ষণীয় ডিজাইন
    Afert Tokenizing:  ['আকর্ষণীয়', 'ডিজাইন']
    Truncating punctuation: ['আকর্ষণীয়', 'ডিজাইন']
    Truncating StopWords: ['আকর্ষণীয়', 'ডিজাইন']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই চমৎকার কাপড়
    Afert Tokenizing:  ['সত্যিই', 'চমৎকার', 'কাপড়']
    Truncating punctuation: ['সত্যিই', 'চমৎকার', 'কাপড়']
    Truncating StopWords: ['সত্যিই', 'চমৎকার', 'কাপড়']
    ***************************************************************************************
    Label:  1
    Sentence:  কিন্তু সামগ্রিকভাবে, এটা সত্যিই ভাল
    Afert Tokenizing:  ['কিন্তু', 'সামগ্রিকভাবে', ',', 'এটা', 'সত্যিই', 'ভাল']
    Truncating punctuation: ['কিন্তু', 'সামগ্রিকভাবে', 'এটা', 'সত্যিই', 'ভাল']
    Truncating StopWords: ['সামগ্রিকভাবে', 'সত্যিই', 'ভাল']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনার যদি বিকল্প থাকে তবে এখান থেকে কেনাকাটা করবেন না। প্রচুর প্রতারক বিক্রেতা রয়েছে এবং তাদের পদ্ধতিগুলি খুব চতুর।
    Afert Tokenizing:  ['আপনার', 'যদি', 'বিকল্প', 'থাকে', 'তবে', 'এখান', 'থেকে', 'কেনাকাটা', 'করবেন', 'না', '।', 'প্রচুর', 'প্রতারক', 'বিক্রেতা', 'রয়েছে', 'এবং', 'তাদের', 'পদ্ধতিগুলি', 'খুব', 'চতুর', '।']
    Truncating punctuation: ['আপনার', 'যদি', 'বিকল্প', 'থাকে', 'তবে', 'এখান', 'থেকে', 'কেনাকাটা', 'করবেন', 'না', 'প্রচুর', 'প্রতারক', 'বিক্রেতা', 'রয়েছে', 'এবং', 'তাদের', 'পদ্ধতিগুলি', 'খুব', 'চতুর']
    Truncating StopWords: ['বিকল্প', 'এখান', 'কেনাকাটা', 'না', 'প্রচুর', 'প্রতারক', 'বিক্রেতা', 'পদ্ধতিগুলি', 'চতুর']
    ***************************************************************************************
    Label:  1
    Sentence:  নকশা সত্যিই ভাল. আমি এটা পছন্দ করি .
    Afert Tokenizing:  ['নকশা', 'সত্যিই', 'ভাল', '.', 'আমি', 'এটা', 'পছন্দ', 'করি', '', '.']
    Truncating punctuation: ['নকশা', 'সত্যিই', 'ভাল', 'আমি', 'এটা', 'পছন্দ', 'করি', '']
    Truncating StopWords: ['নকশা', 'সত্যিই', 'ভাল', 'পছন্দ', '']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি এই বছরের শুরুতে 2টি ভিন্ন সময়ে দুটি ছোট আইটেম অর্ডার করেছি। আমি তাদের গ্রহণ করিনি এবং তারা কোনো ম্যাসেজের উত্তর দিচ্ছে না
    Afert Tokenizing:  ['আমি', 'এই', 'বছরের', 'শুরুতে', '2টি', 'ভিন্ন', 'সময়ে', 'দুটি', 'ছোট', 'আইটেম', 'অর্ডার', 'করেছি', '।', 'আমি', 'তাদের', 'গ্রহণ', 'করিনি', 'এবং', 'তারা', 'কোনো', 'ম্যাসেজের', 'উত্তর', 'দিচ্ছে', 'না']
    Truncating punctuation: ['আমি', 'এই', 'বছরের', 'শুরুতে', '2টি', 'ভিন্ন', 'সময়ে', 'দুটি', 'ছোট', 'আইটেম', 'অর্ডার', 'করেছি', 'আমি', 'তাদের', 'গ্রহণ', 'করিনি', 'এবং', 'তারা', 'কোনো', 'ম্যাসেজের', 'উত্তর', 'দিচ্ছে', 'না']
    Truncating StopWords: ['বছরের', 'শুরুতে', '2টি', 'ভিন্ন', 'সময়ে', 'ছোট', 'আইটেম', 'অর্ডার', 'করেছি', 'গ্রহণ', 'করিনি', 'ম্যাসেজের', 'দিচ্ছে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব হতাশাজনক তাই আমি এখন অন্য কোথাও কেনাকাটা করছি।
    Afert Tokenizing:  ['খুব', 'হতাশাজনক', 'তাই', 'আমি', 'এখন', 'অন্য', 'কোথাও', 'কেনাকাটা', 'করছি', '।']
    Truncating punctuation: ['খুব', 'হতাশাজনক', 'তাই', 'আমি', 'এখন', 'অন্য', 'কোথাও', 'কেনাকাটা', 'করছি']
    Truncating StopWords: ['হতাশাজনক', 'কোথাও', 'কেনাকাটা', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  আলীএক্সপ্রেস আর সস্তা নয়, অনেক আইটেম নরওয়ের চেয়ে বেশি দামি কিন্তু গুণমান সবসময় ভালো নয় তা কল্পনা করুন!!!
    Afert Tokenizing:  ['আলীএক্সপ্রেস', 'আর', 'সস্তা', 'নয়', ',', 'অনেক', 'আইটেম', 'নরওয়ের', 'চেয়ে', 'বেশি', 'দামি', 'কিন্তু', 'গুণমান', 'সবসময়', 'ভালো', 'নয়', 'তা', 'কল্পনা', 'করুন!!', '!']
    Truncating punctuation: ['আলীএক্সপ্রেস', 'আর', 'সস্তা', 'নয়', 'অনেক', 'আইটেম', 'নরওয়ের', 'চেয়ে', 'বেশি', 'দামি', 'কিন্তু', 'গুণমান', 'সবসময়', 'ভালো', 'নয়', 'তা', 'কল্পনা', 'করুন!!']
    Truncating StopWords: ['আলীএক্সপ্রেস', 'সস্তা', 'নয়', 'আইটেম', 'নরওয়ের', 'বেশি', 'দামি', 'গুণমান', 'সবসময়', 'ভালো', 'নয়', 'কল্পনা', 'করুন!!']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি আমার জিনিসপত্র ফেরত দিতে পারি না। পোকামাকড়ের আক্রমনের দায়ে সেগুলো জব্দ করেছে কাস্টমস।
    Afert Tokenizing:  ['আমি', 'আমার', 'জিনিসপত্র', 'ফেরত', 'দিতে', 'পারি', 'না', '।', 'পোকামাকড়ের', 'আক্রমনের', 'দায়ে', 'সেগুলো', 'জব্দ', 'করেছে', 'কাস্টমস', '।']
    Truncating punctuation: ['আমি', 'আমার', 'জিনিসপত্র', 'ফেরত', 'দিতে', 'পারি', 'না', 'পোকামাকড়ের', 'আক্রমনের', 'দায়ে', 'সেগুলো', 'জব্দ', 'করেছে', 'কাস্টমস']
    Truncating StopWords: ['জিনিসপত্র', 'ফেরত', 'না', 'পোকামাকড়ের', 'আক্রমনের', 'দায়ে', 'সেগুলো', 'জব্দ', 'কাস্টমস']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ একটা জার্সি
    Afert Tokenizing:  ['অসাধারণ', 'একটা', 'জার্সি']
    Truncating punctuation: ['অসাধারণ', 'একটা', 'জার্সি']
    Truncating StopWords: ['অসাধারণ', 'একটা', 'জার্সি']
    ***************************************************************************************
    Label:  0
    Sentence:  জিনিসটি বাস্তবে ভালো না
    Afert Tokenizing:  ['জিনিসটি', 'বাস্তবে', 'ভালো', 'না']
    Truncating punctuation: ['জিনিসটি', 'বাস্তবে', 'ভালো', 'না']
    Truncating StopWords: ['জিনিসটি', 'বাস্তবে', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  সব চিটার বাটপার
    Afert Tokenizing:  ['সব', 'চিটার', 'বাটপার']
    Truncating punctuation: ['সব', 'চিটার', 'বাটপার']
    Truncating StopWords: ['চিটার', 'বাটপার']
    ***************************************************************************************
    Label:  0
    Sentence:  দশ পার্সেন্ট ভাটের কথা আগে বলা হয়নি
    Afert Tokenizing:  ['দশ', 'পার্সেন্ট', 'ভাটের', 'কথা', 'আগে', 'বলা', 'হয়নি']
    Truncating punctuation: ['দশ', 'পার্সেন্ট', 'ভাটের', 'কথা', 'আগে', 'বলা', 'হয়নি']
    Truncating StopWords: ['দশ', 'পার্সেন্ট', 'ভাটের', 'কথা']
    ***************************************************************************************
    Label:  0
    Sentence:  সাউন্ড বক্সের দাম অনেক বেশি
    Afert Tokenizing:  ['সাউন্ড', 'বক্সের', 'দাম', 'অনেক', 'বেশি']
    Truncating punctuation: ['সাউন্ড', 'বক্সের', 'দাম', 'অনেক', 'বেশি']
    Truncating StopWords: ['সাউন্ড', 'বক্সের', 'দাম', 'বেশি']
    ***************************************************************************************
    Label:  0
    Sentence:  হেলমেটটা খুব নিম্নমানের
    Afert Tokenizing:  ['হেলমেটটা', 'খুব', 'নিম্নমানের']
    Truncating punctuation: ['হেলমেটটা', 'খুব', 'নিম্নমানের']
    Truncating StopWords: ['হেলমেটটা', 'নিম্নমানের']
    ***************************************************************************************
    Label:  0
    Sentence:  হেলমেটটা ছবিতে সুন্দর কিন্তু বাস্তবে ভালো লাগে নাই
    Afert Tokenizing:  ['হেলমেটটা', 'ছবিতে', 'সুন্দর', 'কিন্তু', 'বাস্তবে', 'ভালো', 'লাগে', 'নাই']
    Truncating punctuation: ['হেলমেটটা', 'ছবিতে', 'সুন্দর', 'কিন্তু', 'বাস্তবে', 'ভালো', 'লাগে', 'নাই']
    Truncating StopWords: ['হেলমেটটা', 'ছবিতে', 'সুন্দর', 'বাস্তবে', 'ভালো', 'লাগে', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  হেলমেটে ফাইবার লেখা থাকলেও হেলমেটটি ফাইবারের না
    Afert Tokenizing:  ['হেলমেটে', 'ফাইবার', 'লেখা', 'থাকলেও', 'হেলমেটটি', 'ফাইবারের', 'না']
    Truncating punctuation: ['হেলমেটে', 'ফাইবার', 'লেখা', 'থাকলেও', 'হেলমেটটি', 'ফাইবারের', 'না']
    Truncating StopWords: ['হেলমেটে', 'ফাইবার', 'লেখা', 'থাকলেও', 'হেলমেটটি', 'ফাইবারের', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  বাচ্চার হেলমেটটি সুন্দরী আছে
    Afert Tokenizing:  ['বাচ্চার', 'হেলমেটটি', 'সুন্দরী', 'আছে']
    Truncating punctuation: ['বাচ্চার', 'হেলমেটটি', 'সুন্দরী', 'আছে']
    Truncating StopWords: ['বাচ্চার', 'হেলমেটটি', 'সুন্দরী']
    ***************************************************************************************
    Label:  1
    Sentence:  জিনিসটা আছে মোটামুটি
    Afert Tokenizing:  ['জিনিসটা', 'আছে', 'মোটামুটি']
    Truncating punctuation: ['জিনিসটা', 'আছে', 'মোটামুটি']
    Truncating StopWords: ['জিনিসটা', 'মোটামুটি']
    ***************************************************************************************
    Label:  1
    Sentence:  আরো সুন্দর আশা করেছিলাম
    Afert Tokenizing:  ['আরো', 'সুন্দর', 'আশা', 'করেছিলাম']
    Truncating punctuation: ['আরো', 'সুন্দর', 'আশা', 'করেছিলাম']
    Truncating StopWords: ['আরো', 'সুন্দর', 'আশা', 'করেছিলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  বেল্ট লেদার বললেও লেদার না আর্টিফিশিয়াল লেদার
    Afert Tokenizing:  ['বেল্ট', 'লেদার', 'বললেও', 'লেদার', 'না', 'আর্টিফিশিয়াল', 'লেদার']
    Truncating punctuation: ['বেল্ট', 'লেদার', 'বললেও', 'লেদার', 'না', 'আর্টিফিশিয়াল', 'লেদার']
    Truncating StopWords: ['বেল্ট', 'লেদার', 'বললেও', 'লেদার', 'না', 'আর্টিফিশিয়াল', 'লেদার']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের জিনিসের দাম আর একটু কমানো উচিত
    Afert Tokenizing:  ['আপনাদের', 'জিনিসের', 'দাম', 'আর', 'একটু', 'কমানো', 'উচিত']
    Truncating punctuation: ['আপনাদের', 'জিনিসের', 'দাম', 'আর', 'একটু', 'কমানো', 'উচিত']
    Truncating StopWords: ['আপনাদের', 'জিনিসের', 'দাম', 'একটু', 'কমানো']
    ***************************************************************************************
    Label:  0
    Sentence:  একদমই ভালো নয়
    Afert Tokenizing:  ['একদমই', 'ভালো', 'নয়']
    Truncating punctuation: ['একদমই', 'ভালো', 'নয়']
    Truncating StopWords: ['একদমই', 'ভালো', 'নয়']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইনে দাম অনেক বেশি
    Afert Tokenizing:  ['অনলাইনে', 'দাম', 'অনেক', 'বেশি']
    Truncating punctuation: ['অনলাইনে', 'দাম', 'অনেক', 'বেশি']
    Truncating StopWords: ['অনলাইনে', 'দাম', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের সার্ভিসে আমি খুব সন্তুষ্ট
    Afert Tokenizing:  ['আপনাদের', 'সার্ভিসে', 'আমি', 'খুব', 'সন্তুষ্ট']
    Truncating punctuation: ['আপনাদের', 'সার্ভিসে', 'আমি', 'খুব', 'সন্তুষ্ট']
    Truncating StopWords: ['আপনাদের', 'সার্ভিসে', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  জুতার মান খুবই খারাপ
    Afert Tokenizing:  ['জুতার', 'মান', 'খুবই', 'খারাপ']
    Truncating punctuation: ['জুতার', 'মান', 'খুবই', 'খারাপ']
    Truncating StopWords: ['জুতার', 'মান', 'খুবই', 'খারাপ']
    ***************************************************************************************
    Label:  1
    Sentence:  ছবিতে যেমন দেখেছি তখনই পেয়েছি
    Afert Tokenizing:  ['ছবিতে', 'যেমন', 'দেখেছি', 'তখনই', 'পেয়েছি']
    Truncating punctuation: ['ছবিতে', 'যেমন', 'দেখেছি', 'তখনই', 'পেয়েছি']
    Truncating StopWords: ['ছবিতে', 'দেখেছি', 'তখনই', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি ম্যান ভাল ছিলনা
    Afert Tokenizing:  ['ডেলিভারি', 'ম্যান', 'ভাল', 'ছিলনা']
    Truncating punctuation: ['ডেলিভারি', 'ম্যান', 'ভাল', 'ছিলনা']
    Truncating StopWords: ['ডেলিভারি', 'ম্যান', 'ভাল', 'ছিলনা']
    ***************************************************************************************
    Label:  1
    Sentence:  বা গেঞ্জিটা খুব সুন্দর তো
    Afert Tokenizing:  ['বা', 'গেঞ্জিটা', 'খুব', 'সুন্দর', 'তো']
    Truncating punctuation: ['বা', 'গেঞ্জিটা', 'খুব', 'সুন্দর', 'তো']
    Truncating StopWords: ['গেঞ্জিটা', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  এক্সক্লুসিভ কালেকশন
    Afert Tokenizing:  ['এক্সক্লুসিভ', 'কালেকশন']
    Truncating punctuation: ['এক্সক্লুসিভ', 'কালেকশন']
    Truncating StopWords: ['এক্সক্লুসিভ', 'কালেকশন']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করেছি তিন দিন আগে এখনো হাতে পায়নি
    Afert Tokenizing:  ['অর্ডার', 'করেছি', 'তিন', 'দিন', 'আগে', 'এখনো', 'হাতে', 'পায়নি']
    Truncating punctuation: ['অর্ডার', 'করেছি', 'তিন', 'দিন', 'আগে', 'এখনো', 'হাতে', 'পায়নি']
    Truncating StopWords: ['অর্ডার', 'করেছি', 'তিন', 'এখনো', 'হাতে', 'পায়নি']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি দিতে এত দেরি কেন হয়
    Afert Tokenizing:  ['ডেলিভারি', 'দিতে', 'এত', 'দেরি', 'কেন', 'হয়']
    Truncating punctuation: ['ডেলিভারি', 'দিতে', 'এত', 'দেরি', 'কেন', 'হয়']
    Truncating StopWords: ['ডেলিভারি', 'দেরি']
    ***************************************************************************************
    Label:  0
    Sentence:  দুইদিন পর পণ্য হাতে পেলাম
    Afert Tokenizing:  ['দুইদিন', 'পর', 'পণ্য', 'হাতে', 'পেলাম']
    Truncating punctuation: ['দুইদিন', 'পর', 'পণ্য', 'হাতে', 'পেলাম']
    Truncating StopWords: ['দুইদিন', 'পণ্য', 'হাতে', 'পেলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  এমটি হেলমেট কালেকশন আরো বাড়াতে হবে
    Afert Tokenizing:  ['এমটি', 'হেলমেট', 'কালেকশন', 'আরো', 'বাড়াতে', 'হবে']
    Truncating punctuation: ['এমটি', 'হেলমেট', 'কালেকশন', 'আরো', 'বাড়াতে', 'হবে']
    Truncating StopWords: ['এমটি', 'হেলমেট', 'কালেকশন', 'আরো', 'বাড়াতে']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রথম বাইকের জিনিসপত্র অনলাইন থেকে কিনলাম
    Afert Tokenizing:  ['প্রথম', 'বাইকের', 'জিনিসপত্র', 'অনলাইন', 'থেকে', 'কিনলাম']
    Truncating punctuation: ['প্রথম', 'বাইকের', 'জিনিসপত্র', 'অনলাইন', 'থেকে', 'কিনলাম']
    Truncating StopWords: ['বাইকের', 'জিনিসপত্র', 'অনলাইন', 'কিনলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  জাবের ভাইয়ের অনলাইনের দোকান অনেক ভালো
    Afert Tokenizing:  ['জাবের', 'ভাইয়ের', 'অনলাইনের', 'দোকান', 'অনেক', 'ভালো']
    Truncating punctuation: ['জাবের', 'ভাইয়ের', 'অনলাইনের', 'দোকান', 'অনেক', 'ভালো']
    Truncating StopWords: ['জাবের', 'ভাইয়ের', 'অনলাইনের', 'দোকান', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  শুভকামনা রইলো আমার ভাই
    Afert Tokenizing:  ['শুভকামনা', 'রইলো', 'আমার', 'ভাই']
    Truncating punctuation: ['শুভকামনা', 'রইলো', 'আমার', 'ভাই']
    Truncating StopWords: ['শুভকামনা', 'রইলো', 'ভাই']
    ***************************************************************************************
    Label:  1
    Sentence:  সেরা বাংলা সেরা পন্য
    Afert Tokenizing:  ['সেরা', 'বাংলা', 'সেরা', 'পন্য']
    Truncating punctuation: ['সেরা', 'বাংলা', 'সেরা', 'পন্য']
    Truncating StopWords: ['সেরা', 'বাংলা', 'সেরা', 'পন্য']
    ***************************************************************************************
    Label:  1
    Sentence:  ম্যাম আপনার মতামত জানানোর জন্য ধন্যবাদ
    Afert Tokenizing:  ['ম্যাম', 'আপনার', 'মতামত', 'জানানোর', 'জন্য', 'ধন্যবাদ']
    Truncating punctuation: ['ম্যাম', 'আপনার', 'মতামত', 'জানানোর', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['ম্যাম', 'মতামত', 'জানানোর', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  দামে কম মানে ভালো
    Afert Tokenizing:  ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating punctuation: ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating StopWords: ['দামে', 'কম', 'মানে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  এই শার্টটি খুবই সুন্দর গুলশন এর  পাশে শোরুম ঠাকলে আমক জানাবে
    Afert Tokenizing:  ['এই', 'শার্টটি', 'খুবই', 'সুন্দর', 'গুলশন', 'এর', 'পাশে', 'শোরুম', 'ঠাকলে', 'আমক', 'জানাবে']
    Truncating punctuation: ['এই', 'শার্টটি', 'খুবই', 'সুন্দর', 'গুলশন', 'এর', 'পাশে', 'শোরুম', 'ঠাকলে', 'আমক', 'জানাবে']
    Truncating StopWords: ['শার্টটি', 'খুবই', 'সুন্দর', 'গুলশন', 'পাশে', 'শোরুম', 'ঠাকলে', 'আমক', 'জানাবে']
    ***************************************************************************************
    Label:  0
    Sentence:  এগুলো ক কালার গেরান্টি। ওয়াস করলে কালার চলে যায় কেন?
    Afert Tokenizing:  ['এগুলো', 'ক', 'কালার', 'গেরান্টি', '।', 'ওয়াস', 'করলে', 'কালার', 'চলে', 'যায়', 'কেন', '?']
    Truncating punctuation: ['এগুলো', 'ক', 'কালার', 'গেরান্টি', 'ওয়াস', 'করলে', 'কালার', 'চলে', 'যায়', 'কেন']
    Truncating StopWords: ['এগুলো', 'ক', 'কালার', 'গেরান্টি', 'ওয়াস', 'কালার', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড় অনেক ভালো
    Afert Tokenizing:  ['কাপড়', 'অনেক', 'ভালো']
    Truncating punctuation: ['কাপড়', 'অনেক', 'ভালো']
    Truncating StopWords: ['কাপড়', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  সহজ কিন্তু মার্জিত
    Afert Tokenizing:  ['সহজ', 'কিন্তু', 'মার্জিত']
    Truncating punctuation: ['সহজ', 'কিন্তু', 'মার্জিত']
    Truncating StopWords: ['সহজ', 'মার্জিত']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল্লাগছে
    Afert Tokenizing:  ['ভাল্লাগছে']
    Truncating punctuation: ['ভাল্লাগছে']
    Truncating StopWords: ['ভাল্লাগছে']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভাল মানের পণ্য
    Afert Tokenizing:  ['খুব', 'ভাল', 'মানের', 'পণ্য']
    Truncating punctuation: ['খুব', 'ভাল', 'মানের', 'পণ্য']
    Truncating StopWords: ['ভাল', 'মানের', 'পণ্য']
    ***************************************************************************************
    Label:  1
    Sentence:  সুপার কলেকশন
    Afert Tokenizing:  ['সুপার', 'কলেকশন']
    Truncating punctuation: ['সুপার', 'কলেকশন']
    Truncating StopWords: ['সুপার', 'কলেকশন']
    ***************************************************************************************
    Label:  0
    Sentence:  এগুলো ক কালার গেরান্টি। ওয়াস করলে কালার চলে যায় কেন?
    Afert Tokenizing:  ['এগুলো', 'ক', 'কালার', 'গেরান্টি', '।', 'ওয়াস', 'করলে', 'কালার', 'চলে', 'যায়', 'কেন', '?']
    Truncating punctuation: ['এগুলো', 'ক', 'কালার', 'গেরান্টি', 'ওয়াস', 'করলে', 'কালার', 'চলে', 'যায়', 'কেন']
    Truncating StopWords: ['এগুলো', 'ক', 'কালার', 'গেরান্টি', 'ওয়াস', 'কালার', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  সহজ আমার প্রিয় ব্র্যান্ড বাংলাদেশে.
    Afert Tokenizing:  ['সহজ', 'আমার', 'প্রিয়', 'ব্র্যান্ড', 'বাংলাদেশে', '.']
    Truncating punctuation: ['সহজ', 'আমার', 'প্রিয়', 'ব্র্যান্ড', 'বাংলাদেশে']
    Truncating StopWords: ['সহজ', 'প্রিয়', 'ব্র্যান্ড', 'বাংলাদেশে']
    ***************************************************************************************
    Label:  1
    Sentence:  অধিকাংশই ভালো মানের।
    Afert Tokenizing:  ['অধিকাংশই', 'ভালো', 'মানের', '।']
    Truncating punctuation: ['অধিকাংশই', 'ভালো', 'মানের']
    Truncating StopWords: ['অধিকাংশই', 'ভালো', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  এত সুন্দর সংগ্রহ
    Afert Tokenizing:  ['এত', 'সুন্দর', 'সংগ্রহ']
    Truncating punctuation: ['এত', 'সুন্দর', 'সংগ্রহ']
    Truncating StopWords: ['সুন্দর', 'সংগ্রহ']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর পাঞ্জাবি
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'পাঞ্জাবি']
    Truncating punctuation: ['খুব', 'সুন্দর', 'পাঞ্জাবি']
    Truncating StopWords: ['সুন্দর', 'পাঞ্জাবি']
    ***************************************************************************************
    Label:  0
    Sentence:  ভোক্তা অধিকারে মামলা করবো আপনাদের নামে-ফালতু জিনিস সব রং জ্বলে যায় ১ মাসে সমস্যা হলে পরিবর্তন না করে দিয়ে হয়রানী করে পাবনা শোরুমের সকল স্টাপ আপনাদের বেয়াদব
    Afert Tokenizing:  ['ভোক্তা', 'অধিকারে', 'মামলা', 'করবো', 'আপনাদের', 'নামে-ফালতু', 'জিনিস', 'সব', 'রং', 'জ্বলে', 'যায়', '১', 'মাসে', 'সমস্যা', 'হলে', 'পরিবর্তন', 'না', 'করে', 'দিয়ে', 'হয়রানী', 'করে', 'পাবনা', 'শোরুমের', 'সকল', 'স্টাপ', 'আপনাদের', 'বেয়াদব']
    Truncating punctuation: ['ভোক্তা', 'অধিকারে', 'মামলা', 'করবো', 'আপনাদের', 'নামে-ফালতু', 'জিনিস', 'সব', 'রং', 'জ্বলে', 'যায়', '১', 'মাসে', 'সমস্যা', 'হলে', 'পরিবর্তন', 'না', 'করে', 'দিয়ে', 'হয়রানী', 'করে', 'পাবনা', 'শোরুমের', 'সকল', 'স্টাপ', 'আপনাদের', 'বেয়াদব']
    Truncating StopWords: ['ভোক্তা', 'অধিকারে', 'মামলা', 'করবো', 'আপনাদের', 'নামে-ফালতু', 'জিনিস', 'রং', 'জ্বলে', 'যায়', '১', 'মাসে', 'সমস্যা', 'পরিবর্তন', 'না', 'দিয়ে', 'হয়রানী', 'পাবনা', 'শোরুমের', 'সকল', 'স্টাপ', 'আপনাদের', 'বেয়াদব']
    ***************************************************************************************
    Label:  1
    Sentence:  এই ডিজাইন গুলো বড়দের হলে ভালো চলত
    Afert Tokenizing:  ['এই', 'ডিজাইন', 'গুলো', 'বড়দের', 'হলে', 'ভালো', 'চলত']
    Truncating punctuation: ['এই', 'ডিজাইন', 'গুলো', 'বড়দের', 'হলে', 'ভালো', 'চলত']
    Truncating StopWords: ['ডিজাইন', 'গুলো', 'বড়দের', 'ভালো', 'চলত']
    ***************************************************************************************
    Label:  1
    Sentence:  সকল শুভ কামনা
    Afert Tokenizing:  ['সকল', 'শুভ', 'কামনা']
    Truncating punctuation: ['সকল', 'শুভ', 'কামনা']
    Truncating StopWords: ['সকল', 'শুভ', 'কামনা']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আপনাদের টিশার্ট গুলো একটু ভালো করেন । আপনাদের টি-শার্টগুলো অনেক লম্বা হয়ে যায় পরে আর পড়া সম্ভব হয় না
    Afert Tokenizing:  ['ভাই', 'আপনাদের', 'টিশার্ট', 'গুলো', 'একটু', 'ভালো', 'করেন', '', '।', 'আপনাদের', 'টি-শার্টগুলো', 'অনেক', 'লম্বা', 'হয়ে', 'যায়', 'পরে', 'আর', 'পড়া', 'সম্ভব', 'হয়', 'না']
    Truncating punctuation: ['ভাই', 'আপনাদের', 'টিশার্ট', 'গুলো', 'একটু', 'ভালো', 'করেন', '', 'আপনাদের', 'টি-শার্টগুলো', 'অনেক', 'লম্বা', 'হয়ে', 'যায়', 'পরে', 'আর', 'পড়া', 'সম্ভব', 'হয়', 'না']
    Truncating StopWords: ['ভাই', 'আপনাদের', 'টিশার্ট', 'গুলো', 'একটু', 'ভালো', '', 'আপনাদের', 'টি-শার্টগুলো', 'লম্বা', 'পড়া', 'সম্ভব', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাংলাদেশের একমাত্র ফালতু ব্যান্ড থাকলে ইজি ফ্যাশন
    Afert Tokenizing:  ['বাংলাদেশের', 'একমাত্র', 'ফালতু', 'ব্যান্ড', 'থাকলে', 'ইজি', 'ফ্যাশন']
    Truncating punctuation: ['বাংলাদেশের', 'একমাত্র', 'ফালতু', 'ব্যান্ড', 'থাকলে', 'ইজি', 'ফ্যাশন']
    Truncating StopWords: ['বাংলাদেশের', 'একমাত্র', 'ফালতু', 'ব্যান্ড', 'থাকলে', 'ইজি', 'ফ্যাশন']
    ***************************************************************************************
    Label:  1
    Sentence:  ২০১৪ সাল থেকে শুরু করে এখন পযন্ত ব্যবহার করা আমার খুব পছন্দের একটি ব্রান্ড ইজি ফ্যাশন লিমিটেড
    Afert Tokenizing:  ['২০১৪', 'সাল', 'থেকে', 'শুরু', 'করে', 'এখন', 'পযন্ত', 'ব্যবহার', 'করা', 'আমার', 'খুব', 'পছন্দের', 'একটি', 'ব্রান্ড', 'ইজি', 'ফ্যাশন', 'লিমিটেড']
    Truncating punctuation: ['২০১৪', 'সাল', 'থেকে', 'শুরু', 'করে', 'এখন', 'পযন্ত', 'ব্যবহার', 'করা', 'আমার', 'খুব', 'পছন্দের', 'একটি', 'ব্রান্ড', 'ইজি', 'ফ্যাশন', 'লিমিটেড']
    Truncating StopWords: ['২০১৪', 'সাল', 'পযন্ত', 'পছন্দের', 'ব্রান্ড', 'ইজি', 'ফ্যাশন', 'লিমিটেড']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি নেক্সট থেকে প্রচুর জামাকাপড় এবং মাতালান থেকে ২ টি জিন্স প্যান্ট কিনেছি। কিন্তু সেই দোকানগুলিতে কোনও মান এবং আমার পছন্দের যোগ্য শার্ট খুঁজে পাইনি।
    Afert Tokenizing:  ['আমি', 'নেক্সট', 'থেকে', 'প্রচুর', 'জামাকাপড়', 'এবং', 'মাতালান', 'থেকে', '২', 'টি', 'জিন্স', 'প্যান্ট', 'কিনেছি', '।', 'কিন্তু', 'সেই', 'দোকানগুলিতে', 'কোনও', 'মান', 'এবং', 'আমার', 'পছন্দের', 'যোগ্য', 'শার্ট', 'খুঁজে', 'পাইনি', '।']
    Truncating punctuation: ['আমি', 'নেক্সট', 'থেকে', 'প্রচুর', 'জামাকাপড়', 'এবং', 'মাতালান', 'থেকে', '২', 'টি', 'জিন্স', 'প্যান্ট', 'কিনেছি', 'কিন্তু', 'সেই', 'দোকানগুলিতে', 'কোনও', 'মান', 'এবং', 'আমার', 'পছন্দের', 'যোগ্য', 'শার্ট', 'খুঁজে', 'পাইনি']
    Truncating StopWords: ['নেক্সট', 'প্রচুর', 'জামাকাপড়', 'মাতালান', '২', 'জিন্স', 'প্যান্ট', 'কিনেছি', 'দোকানগুলিতে', 'মান', 'পছন্দের', 'যোগ্য', 'শার্ট', 'খুঁজে', 'পাইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর
    Afert Tokenizing:  ['খুব', 'সুন্দর']
    Truncating punctuation: ['খুব', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  ফুটপাত কোয়ালিটি তোদের চাইতে ভালো
    Afert Tokenizing:  ['ফুটপাত', 'কোয়ালিটি', 'তোদের', 'চাইতে', 'ভালো']
    Truncating punctuation: ['ফুটপাত', 'কোয়ালিটি', 'তোদের', 'চাইতে', 'ভালো']
    Truncating StopWords: ['ফুটপাত', 'কোয়ালিটি', 'তোদের', 'চাইতে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  এই পাঞ্জাবি পরে অনেক শান্তি
    Afert Tokenizing:  ['এই', 'পাঞ্জাবি', 'পরে', 'অনেক', 'শান্তি']
    Truncating punctuation: ['এই', 'পাঞ্জাবি', 'পরে', 'অনেক', 'শান্তি']
    Truncating StopWords: ['পাঞ্জাবি', 'শান্তি']
    ***************************************************************************************
    Label:  1
    Sentence:  বিউটিফুল সমাস
    Afert Tokenizing:  ['বিউটিফুল', 'সমাস']
    Truncating punctuation: ['বিউটিফুল', 'সমাস']
    Truncating StopWords: ['বিউটিফুল', 'সমাস']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার খুব পছন্দের ব্রান্ড ইজি।হোম ডিস্ট্রিক সিলেট।
    Afert Tokenizing:  ['আমার', 'খুব', 'পছন্দের', 'ব্রান্ড', 'ইজি।হোম', 'ডিস্ট্রিক', 'সিলেট', '।']
    Truncating punctuation: ['আমার', 'খুব', 'পছন্দের', 'ব্রান্ড', 'ইজি।হোম', 'ডিস্ট্রিক', 'সিলেট']
    Truncating StopWords: ['পছন্দের', 'ব্রান্ড', 'ইজি।হোম', 'ডিস্ট্রিক', 'সিলেট']
    ***************************************************************************************
    Label:  1
    Sentence:  আসসালামু আলাইকুম আমি দুটি টি-শার্ট অর্ডার করেছিলাম আজ হাতে পেলাম। প্রথমে ভেবেছিলাম কাপড়ের মান ভালো হবে কিনা, কিন্তু টি-শার্ট হাতে পেয়ে মন ভরে গেল।
    Afert Tokenizing:  ['আসসালামু', 'আলাইকুম', 'আমি', 'দুটি', 'টি-শার্ট', 'অর্ডার', 'করেছিলাম', 'আজ', 'হাতে', 'পেলাম', '।', 'প্রথমে', 'ভেবেছিলাম', 'কাপড়ের', 'মান', 'ভালো', 'হবে', 'কিনা', ',', 'কিন্তু', 'টি-শার্ট', 'হাতে', 'পেয়ে', 'মন', 'ভরে', 'গেল', '।']
    Truncating punctuation: ['আসসালামু', 'আলাইকুম', 'আমি', 'দুটি', 'টি-শার্ট', 'অর্ডার', 'করেছিলাম', 'আজ', 'হাতে', 'পেলাম', 'প্রথমে', 'ভেবেছিলাম', 'কাপড়ের', 'মান', 'ভালো', 'হবে', 'কিনা', 'কিন্তু', 'টি-শার্ট', 'হাতে', 'পেয়ে', 'মন', 'ভরে', 'গেল']
    Truncating StopWords: ['আসসালামু', 'আলাইকুম', 'টি-শার্ট', 'অর্ডার', 'করেছিলাম', 'হাতে', 'পেলাম', 'প্রথমে', 'ভেবেছিলাম', 'কাপড়ের', 'মান', 'ভালো', 'কিনা', 'টি-শার্ট', 'হাতে', 'মন', 'ভরে']
    ***************************************************************************************
    Label:  1
    Sentence:  এই প্রথম অনলাইনে বিশ্বস্ত একটা কোম্পানি পেলাম আমি ফ্রান্সে থাকি আমি আরো কিছু টি-শার্ট অর্ডার করবো ইনশাআল্লাহ। যারা নিতে চান, নিতে পারেন নিঃসন্দেহে।
    Afert Tokenizing:  ['এই', 'প্রথম', 'অনলাইনে', 'বিশ্বস্ত', 'একটা', 'কোম্পানি', 'পেলাম', 'আমি', 'ফ্রান্সে', 'থাকি', 'আমি', 'আরো', 'কিছু', 'টি-শার্ট', 'অর্ডার', 'করবো', 'ইনশাআল্লাহ', '।', 'যারা', 'নিতে', 'চান', ',', 'নিতে', 'পারেন', 'নিঃসন্দেহে', '।']
    Truncating punctuation: ['এই', 'প্রথম', 'অনলাইনে', 'বিশ্বস্ত', 'একটা', 'কোম্পানি', 'পেলাম', 'আমি', 'ফ্রান্সে', 'থাকি', 'আমি', 'আরো', 'কিছু', 'টি-শার্ট', 'অর্ডার', 'করবো', 'ইনশাআল্লাহ', 'যারা', 'নিতে', 'চান', 'নিতে', 'পারেন', 'নিঃসন্দেহে']
    Truncating StopWords: ['অনলাইনে', 'বিশ্বস্ত', 'একটা', 'কোম্পানি', 'পেলাম', 'ফ্রান্সে', 'থাকি', 'আরো', 'টি-শার্ট', 'অর্ডার', 'করবো', 'ইনশাআল্লাহ', 'নিঃসন্দেহে']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ দেখতে
    Afert Tokenizing:  ['অসাধারণ', 'দেখতে']
    Truncating punctuation: ['অসাধারণ', 'দেখতে']
    Truncating StopWords: ['অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  শার্ট গুলো অনেক সুন্দর তবে আমার ওগুলো কেনার সাধ্য নাই
    Afert Tokenizing:  ['শার্ট', 'গুলো', 'অনেক', 'সুন্দর', 'তবে', 'আমার', 'ওগুলো', 'কেনার', 'সাধ্য', 'নাই']
    Truncating punctuation: ['শার্ট', 'গুলো', 'অনেক', 'সুন্দর', 'তবে', 'আমার', 'ওগুলো', 'কেনার', 'সাধ্য', 'নাই']
    Truncating StopWords: ['শার্ট', 'গুলো', 'সুন্দর', 'ওগুলো', 'কেনার', 'সাধ্য', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর লাকছে
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'লাকছে']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'লাকছে']
    Truncating StopWords: ['সুন্দর', 'লাকছে']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট ভালো না কালার শাদাশে হয়ে যায়
    Afert Tokenizing:  ['প্রোডাক্ট', 'ভালো', 'না', 'কালার', 'শাদাশে', 'হয়ে', 'যায়']
    Truncating punctuation: ['প্রোডাক্ট', 'ভালো', 'না', 'কালার', 'শাদাশে', 'হয়ে', 'যায়']
    Truncating StopWords: ['প্রোডাক্ট', 'ভালো', 'না', 'কালার', 'শাদাশে', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  নাইচ
    Afert Tokenizing:  ['নাইচ']
    Truncating punctuation: ['নাইচ']
    Truncating StopWords: ['নাইচ']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের গোল গলা গেঞ্জি গুলো লক সেলাই খুলে খুলে যায়, আমার অনেক গেঞ্জি নষ্ট হইছে
    Afert Tokenizing:  ['আপনাদের', 'গোল', 'গলা', 'গেঞ্জি', 'গুলো', 'লক', 'সেলাই', 'খুলে', 'খুলে', 'যায়', ',', 'আমার', 'অনেক', 'গেঞ্জি', 'নষ্ট', 'হইছে']
    Truncating punctuation: ['আপনাদের', 'গোল', 'গলা', 'গেঞ্জি', 'গুলো', 'লক', 'সেলাই', 'খুলে', 'খুলে', 'যায়', 'আমার', 'অনেক', 'গেঞ্জি', 'নষ্ট', 'হইছে']
    Truncating StopWords: ['আপনাদের', 'গোল', 'গলা', 'গেঞ্জি', 'গুলো', 'লক', 'সেলাই', 'খুলে', 'খুলে', 'যায়', 'গেঞ্জি', 'নষ্ট', 'হইছে']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি কিছু দিন আগে ১০৯৪ টাকা প্রাইজ একটা টি সাট নিছিলাম।কিন্তু একবার পড়তেই রং শেষ। একটা চেঞ্জ করে দিছে ঐ টাও একি
    Afert Tokenizing:  ['আমি', 'কিছু', 'দিন', 'আগে', '১০৯৪', 'টাকা', 'প্রাইজ', 'একটা', 'টি', 'সাট', 'নিছিলাম।কিন্তু', 'একবার', 'পড়তেই', 'রং', 'শেষ', '।', 'একটা', 'চেঞ্জ', 'করে', 'দিছে', 'ঐ', 'টাও', 'একি']
    Truncating punctuation: ['আমি', 'কিছু', 'দিন', 'আগে', '১০৯৪', 'টাকা', 'প্রাইজ', 'একটা', 'টি', 'সাট', 'নিছিলাম।কিন্তু', 'একবার', 'পড়তেই', 'রং', 'শেষ', 'একটা', 'চেঞ্জ', 'করে', 'দিছে', 'ঐ', 'টাও', 'একি']
    Truncating StopWords: ['১০৯৪', 'টাকা', 'প্রাইজ', 'একটা', 'সাট', 'নিছিলাম।কিন্তু', 'পড়তেই', 'রং', 'শেষ', 'একটা', 'চেঞ্জ', 'দিছে', 'টাও', 'একি']
    ***************************************************************************************
    Label:  1
    Sentence:  ঈদ উল ফিতর এর থেকে এবারের কালেকশন সত্যি ভালো
    Afert Tokenizing:  ['ঈদ', 'উল', 'ফিতর', 'এর', 'থেকে', 'এবারের', 'কালেকশন', 'সত্যি', 'ভালো']
    Truncating punctuation: ['ঈদ', 'উল', 'ফিতর', 'এর', 'থেকে', 'এবারের', 'কালেকশন', 'সত্যি', 'ভালো']
    Truncating StopWords: ['ঈদ', 'উল', 'ফিতর', 'এবারের', 'কালেকশন', 'সত্যি', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  মেসেঞ্জারে অর্ডার দিলে অর্ডার কনফার্ম করেনা
    Afert Tokenizing:  ['মেসেঞ্জারে', 'অর্ডার', 'দিলে', 'অর্ডার', 'কনফার্ম', 'করেনা']
    Truncating punctuation: ['মেসেঞ্জারে', 'অর্ডার', 'দিলে', 'অর্ডার', 'কনফার্ম', 'করেনা']
    Truncating StopWords: ['মেসেঞ্জারে', 'অর্ডার', 'দিলে', 'অর্ডার', 'কনফার্ম', 'করেনা']
    ***************************************************************************************
    Label:  1
    Sentence:  বাহ্ অসাধারন
    Afert Tokenizing:  ['বাহ্', 'অসাধারন']
    Truncating punctuation: ['বাহ্', 'অসাধারন']
    Truncating StopWords: ['বাহ্', 'অসাধারন']
    ***************************************************************************************
    Label:  0
    Sentence:  এখন একদম বাজে হয়ে গেছে
    Afert Tokenizing:  ['এখন', 'একদম', 'বাজে', 'হয়ে', 'গেছে']
    Truncating punctuation: ['এখন', 'একদম', 'বাজে', 'হয়ে', 'গেছে']
    Truncating StopWords: ['একদম', 'বাজে', 'হয়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  কাপড়ের মান খুবই নিম্নমানের কয়েকদিন ব্যবহার করার পর রং নষ্ট হয়ে যায়
    Afert Tokenizing:  ['কাপড়ের', 'মান', 'খুবই', 'নিম্নমানের', 'কয়েকদিন', 'ব্যবহার', 'করার', 'পর', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    Truncating punctuation: ['কাপড়ের', 'মান', 'খুবই', 'নিম্নমানের', 'কয়েকদিন', 'ব্যবহার', 'করার', 'পর', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    Truncating StopWords: ['কাপড়ের', 'মান', 'খুবই', 'নিম্নমানের', 'কয়েকদিন', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  ২০১৪ থেকে সাথে আছি।কিছু অভিযোগ ছিলো করলাম না
    Afert Tokenizing:  ['২০১৪', 'থেকে', 'সাথে', 'আছি।কিছু', 'অভিযোগ', 'ছিলো', 'করলাম', 'না']
    Truncating punctuation: ['২০১৪', 'থেকে', 'সাথে', 'আছি।কিছু', 'অভিযোগ', 'ছিলো', 'করলাম', 'না']
    Truncating StopWords: ['২০১৪', 'সাথে', 'আছি।কিছু', 'অভিযোগ', 'ছিলো', 'করলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  কাপড় গুলো ফালতু করে দাম করছে আকাশ চুম্বী তাই আর ইজি পরি না
    Afert Tokenizing:  ['কাপড়', 'গুলো', 'ফালতু', 'করে', 'দাম', 'করছে', 'আকাশ', 'চুম্বী', 'তাই', 'আর', 'ইজি', 'পরি', 'না']
    Truncating punctuation: ['কাপড়', 'গুলো', 'ফালতু', 'করে', 'দাম', 'করছে', 'আকাশ', 'চুম্বী', 'তাই', 'আর', 'ইজি', 'পরি', 'না']
    Truncating StopWords: ['কাপড়', 'গুলো', 'ফালতু', 'দাম', 'আকাশ', 'চুম্বী', 'ইজি', 'পরি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু কালেকসন
    Afert Tokenizing:  ['ফালতু', 'কালেকসন']
    Truncating punctuation: ['ফালতু', 'কালেকসন']
    Truncating StopWords: ['ফালতু', 'কালেকসন']
    ***************************************************************************************
    Label:  0
    Sentence:  একটা প্যান্ট কিনে বাসায় এসে দেখি, কোমোরের বোতাম ভাংগা, শব স্টীকার লাগানো আছে এখনো, কিন্ত মেমোটা ওরা ভুল করে দেয় নাই, আমিও খেয়াল করি নাই।
    Afert Tokenizing:  ['একটা', 'প্যান্ট', 'কিনে', 'বাসায়', 'এসে', 'দেখি', ',', 'কোমোরের', 'বোতাম', 'ভাংগা', ',', 'শব', 'স্টীকার', 'লাগানো', 'আছে', 'এখনো', ',', 'কিন্ত', 'মেমোটা', 'ওরা', 'ভুল', 'করে', 'দেয়', 'নাই', ',', 'আমিও', 'খেয়াল', 'করি', 'নাই', '।']
    Truncating punctuation: ['একটা', 'প্যান্ট', 'কিনে', 'বাসায়', 'এসে', 'দেখি', 'কোমোরের', 'বোতাম', 'ভাংগা', 'শব', 'স্টীকার', 'লাগানো', 'আছে', 'এখনো', 'কিন্ত', 'মেমোটা', 'ওরা', 'ভুল', 'করে', 'দেয়', 'নাই', 'আমিও', 'খেয়াল', 'করি', 'নাই']
    Truncating StopWords: ['একটা', 'প্যান্ট', 'কিনে', 'বাসায়', 'দেখি', 'কোমোরের', 'বোতাম', 'ভাংগা', 'শব', 'স্টীকার', 'লাগানো', 'এখনো', 'কিন্ত', 'মেমোটা', 'ভুল', 'দেয়', 'নাই', 'আমিও', 'খেয়াল', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কাপড়ের মান টা দিন দিন খারাপ করছে কিন্তু দাম বাড়াচ্ছে
    Afert Tokenizing:  ['কাপড়ের', 'মান', 'টা', 'দিন', 'দিন', 'খারাপ', 'করছে', 'কিন্তু', 'দাম', 'বাড়াচ্ছে']
    Truncating punctuation: ['কাপড়ের', 'মান', 'টা', 'দিন', 'দিন', 'খারাপ', 'করছে', 'কিন্তু', 'দাম', 'বাড়াচ্ছে']
    Truncating StopWords: ['কাপড়ের', 'মান', 'টা', 'খারাপ', 'দাম', 'বাড়াচ্ছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ইজি ফ্যাশন সব সময় অসহায় মানুষের পাশে থাকবে এই প্রত্যাশা করি দোয়া রইল ইজি ফ্যাশন এর জন্য
    Afert Tokenizing:  ['ইজি', 'ফ্যাশন', 'সব', 'সময়', 'অসহায়', 'মানুষের', 'পাশে', 'থাকবে', 'এই', 'প্রত্যাশা', 'করি', 'দোয়া', 'রইল', 'ইজি', 'ফ্যাশন', 'এর', 'জন্য']
    Truncating punctuation: ['ইজি', 'ফ্যাশন', 'সব', 'সময়', 'অসহায়', 'মানুষের', 'পাশে', 'থাকবে', 'এই', 'প্রত্যাশা', 'করি', 'দোয়া', 'রইল', 'ইজি', 'ফ্যাশন', 'এর', 'জন্য']
    Truncating StopWords: ['ইজি', 'ফ্যাশন', 'সময়', 'অসহায়', 'মানুষের', 'পাশে', 'প্রত্যাশা', 'দোয়া', 'রইল', 'ইজি', 'ফ্যাশন']
    ***************************************************************************************
    Label:  0
    Sentence:  রমজানের আগে টি-শার্ট এর দাম ছিল ৪৯৫ টাকা। ঈদের আগে তা বেড়ে হয় ৫৯৫ টাকা। সবচেয়ে বড় কথা হল এক সপ্তার মধ্যে কালার চলে গেল।
    Afert Tokenizing:  ['রমজানের', 'আগে', 'টি-শার্ট', 'এর', 'দাম', 'ছিল', '৪৯৫', 'টাকা', '।', 'ঈদের', 'আগে', 'তা', 'বেড়ে', 'হয়', '৫৯৫', 'টাকা', '।', 'সবচেয়ে', 'বড়', 'কথা', 'হল', 'এক', 'সপ্তার', 'মধ্যে', 'কালার', 'চলে', 'গেল', '।']
    Truncating punctuation: ['রমজানের', 'আগে', 'টি-শার্ট', 'এর', 'দাম', 'ছিল', '৪৯৫', 'টাকা', 'ঈদের', 'আগে', 'তা', 'বেড়ে', 'হয়', '৫৯৫', 'টাকা', 'সবচেয়ে', 'বড়', 'কথা', 'হল', 'এক', 'সপ্তার', 'মধ্যে', 'কালার', 'চলে', 'গেল']
    Truncating StopWords: ['রমজানের', 'টি-শার্ট', 'দাম', '৪৯৫', 'টাকা', 'ঈদের', 'বেড়ে', '৫৯৫', 'টাকা', 'সবচেয়ে', 'বড়', 'কথা', 'এক', 'সপ্তার', 'কালার']
    ***************************************************************************************
    Label:  0
    Sentence:  বর্তমানে ইজি ফ্যাশন এর গোল গলার টি-শারট গুলোর মডেল একটাও ভালো না
    Afert Tokenizing:  ['বর্তমানে', 'ইজি', 'ফ্যাশন', 'এর', 'গোল', 'গলার', 'টি-শারট', 'গুলোর', 'মডেল', 'একটাও', 'ভালো', 'না']
    Truncating punctuation: ['বর্তমানে', 'ইজি', 'ফ্যাশন', 'এর', 'গোল', 'গলার', 'টি-শারট', 'গুলোর', 'মডেল', 'একটাও', 'ভালো', 'না']
    Truncating StopWords: ['বর্তমানে', 'ইজি', 'ফ্যাশন', 'গোল', 'গলার', 'টি-শারট', 'গুলোর', 'মডেল', 'একটাও', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  সব চেয়ে বাজে ব্রান্ড কিন্তু কি আর করার আমাদের শহরে এটাই আছে
    Afert Tokenizing:  ['সব', 'চেয়ে', 'বাজে', 'ব্রান্ড', 'কিন্তু', 'কি', 'আর', 'করার', 'আমাদের', 'শহরে', 'এটাই', 'আছে']
    Truncating punctuation: ['সব', 'চেয়ে', 'বাজে', 'ব্রান্ড', 'কিন্তু', 'কি', 'আর', 'করার', 'আমাদের', 'শহরে', 'এটাই', 'আছে']
    Truncating StopWords: ['চেয়ে', 'বাজে', 'ব্রান্ড', 'শহরে']
    ***************************************************************************************
    Label:  0
    Sentence:  পছন্দ হয়েছে তবে দাম অনেক বেশি।
    Afert Tokenizing:  ['পছন্দ', 'হয়েছে', 'তবে', 'দাম', 'অনেক', 'বেশি', '।']
    Truncating punctuation: ['পছন্দ', 'হয়েছে', 'তবে', 'দাম', 'অনেক', 'বেশি']
    Truncating StopWords: ['পছন্দ', 'হয়েছে', 'দাম', 'বেশি']
    ***************************************************************************************
    Label:  0
    Sentence:  একছের দাম
    Afert Tokenizing:  ['একছের', 'দাম']
    Truncating punctuation: ['একছের', 'দাম']
    Truncating StopWords: ['একছের', 'দাম']
    ***************************************************************************************
    Label:  0
    Sentence:  পণ্যের দাম বেশি দেখিয়ে তারপরে ডিসকাউন্ট দেয়া হয় ।এটা একটা শুভঙ্করের ফাঁকি
    Afert Tokenizing:  ['পণ্যের', 'দাম', 'বেশি', 'দেখিয়ে', 'তারপরে', 'ডিসকাউন্ট', 'দেয়া', 'হয়', 'এটা', '।', 'একটা', 'শুভঙ্করের', 'ফাঁকি']
    Truncating punctuation: ['পণ্যের', 'দাম', 'বেশি', 'দেখিয়ে', 'তারপরে', 'ডিসকাউন্ট', 'দেয়া', 'হয়', 'এটা', 'একটা', 'শুভঙ্করের', 'ফাঁকি']
    Truncating StopWords: ['পণ্যের', 'দাম', 'বেশি', 'দেখিয়ে', 'তারপরে', 'ডিসকাউন্ট', 'দেয়া', 'একটা', 'শুভঙ্করের', 'ফাঁকি']
    ***************************************************************************************
    Label:  0
    Sentence:  ধোক্কাবাজী, ধাপ্পাবাজী অফার. উৎসবের সময়ে নগদে নেয় না,ক্যাশ নেয় হ্যান্ডক্যাশ.নগদে নেওয়ার সময় নাই.
    Afert Tokenizing:  ['ধোক্কাবাজী', ',', 'ধাপ্পাবাজী', 'অফার', '.', 'উৎসবের', 'সময়ে', 'নগদে', 'নেয়', 'না,ক্যাশ', 'নেয়', 'হ্যান্ডক্যাশ.নগদে', 'নেওয়ার', 'সময়', 'নাই', '.']
    Truncating punctuation: ['ধোক্কাবাজী', 'ধাপ্পাবাজী', 'অফার', 'উৎসবের', 'সময়ে', 'নগদে', 'নেয়', 'না,ক্যাশ', 'নেয়', 'হ্যান্ডক্যাশ.নগদে', 'নেওয়ার', 'সময়', 'নাই']
    Truncating StopWords: ['ধোক্কাবাজী', 'ধাপ্পাবাজী', 'অফার', 'উৎসবের', 'সময়ে', 'নগদে', 'নেয়', 'না,ক্যাশ', 'নেয়', 'হ্যান্ডক্যাশ.নগদে', 'নেওয়ার', 'সময়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  ইজির প্রোডাক্ট ভালো না। সালারা ২০০৳ এর টি-শার্ট ৬০০ টাকায় বিক্রি করে
    Afert Tokenizing:  ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', '।', 'সালারা', '২০০৳', 'এর', 'টি-শার্ট', '৬০০', 'টাকায়', 'বিক্রি', 'করে']
    Truncating punctuation: ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', 'সালারা', '২০০৳', 'এর', 'টি-শার্ট', '৬০০', 'টাকায়', 'বিক্রি', 'করে']
    Truncating StopWords: ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', 'সালারা', '২০০৳', 'টি-শার্ট', '৬০০', 'টাকায়', 'বিক্রি']
    ***************************************************************************************
    Label:  0
    Sentence:  টি শার্ট এর রং নষ্ট হয়ে যায়
    Afert Tokenizing:  ['টি', 'শার্ট', 'এর', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    Truncating punctuation: ['টি', 'শার্ট', 'এর', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    Truncating StopWords: ['শার্ট', 'রং', 'নষ্ট', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  ইজি খুব বাজে একটা ব্যন্ড। কালার থাকেনা।টি সার্ট অথবা সার্ট
    Afert Tokenizing:  ['ইজি', 'খুব', 'বাজে', 'একটা', 'ব্যন্ড', '।', 'কালার', 'থাকেনা।টি', 'সার্ট', 'অথবা', 'সার্ট']
    Truncating punctuation: ['ইজি', 'খুব', 'বাজে', 'একটা', 'ব্যন্ড', 'কালার', 'থাকেনা।টি', 'সার্ট', 'অথবা', 'সার্ট']
    Truncating StopWords: ['ইজি', 'বাজে', 'একটা', 'ব্যন্ড', 'কালার', 'থাকেনা।টি', 'সার্ট', 'সার্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  আস্তে আস্তে ফালতু ব্র্যান্ড হয়ে যাচ্ছে, কাপড় এর গুনগত মান ভালো না এখন
    Afert Tokenizing:  ['আস্তে', 'আস্তে', 'ফালতু', 'ব্র্যান্ড', 'হয়ে', 'যাচ্ছে', ',', 'কাপড়', 'এর', 'গুনগত', 'মান', 'ভালো', 'না', 'এখন']
    Truncating punctuation: ['আস্তে', 'আস্তে', 'ফালতু', 'ব্র্যান্ড', 'হয়ে', 'যাচ্ছে', 'কাপড়', 'এর', 'গুনগত', 'মান', 'ভালো', 'না', 'এখন']
    Truncating StopWords: ['আস্তে', 'আস্তে', 'ফালতু', 'ব্র্যান্ড', 'কাপড়', 'গুনগত', 'মান', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের সেলাই এর মান ভালো করতে হবে নিচের সেলাই কয় এক দিন পর খুলে যায়
    Afert Tokenizing:  ['আপনাদের', 'সেলাই', 'এর', 'মান', 'ভালো', 'করতে', 'হবে', 'নিচের', 'সেলাই', 'কয়', 'এক', 'দিন', 'পর', 'খুলে', 'যায়']
    Truncating punctuation: ['আপনাদের', 'সেলাই', 'এর', 'মান', 'ভালো', 'করতে', 'হবে', 'নিচের', 'সেলাই', 'কয়', 'এক', 'দিন', 'পর', 'খুলে', 'যায়']
    Truncating StopWords: ['আপনাদের', 'সেলাই', 'মান', 'ভালো', 'নিচের', 'সেলাই', 'কয়', 'এক', 'খুলে', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  ইজির প্রোডাক্ট ভালো না। ফালতু।
    Afert Tokenizing:  ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', '।', 'ফালতু', '।']
    Truncating punctuation: ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', 'ফালতু']
    Truncating StopWords: ['ইজির', 'প্রোডাক্ট', 'ভালো', 'না', 'ফালতু']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই শার্ট এর কালার ১ মাসের মধ্যেই নষ্ট হয়ে যায় কেন
    Afert Tokenizing:  ['ভাই', 'শার্ট', 'এর', 'কালার', '১', 'মাসের', 'মধ্যেই', 'নষ্ট', 'হয়ে', 'যায়', 'কেন']
    Truncating punctuation: ['ভাই', 'শার্ট', 'এর', 'কালার', '১', 'মাসের', 'মধ্যেই', 'নষ্ট', 'হয়ে', 'যায়', 'কেন']
    Truncating StopWords: ['ভাই', 'শার্ট', 'কালার', '১', 'মাসের', 'নষ্ট', 'হয়ে', 'যায়']
    ***************************************************************************************
    Label:  1
    Sentence:  শার্টটা কিন্তু হেব্বি সুন্দর
    Afert Tokenizing:  ['শার্টটা', 'কিন্তু', 'হেব্বি', 'সুন্দর']
    Truncating punctuation: ['শার্টটা', 'কিন্তু', 'হেব্বি', 'সুন্দর']
    Truncating StopWords: ['শার্টটা', 'হেব্বি', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ।  সব ভ্যালো মোটো পেইচি...।  প্যাকেজিং খুব ভ্যালো সাইলো
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '', '।', 'সব', 'ভ্যালো', 'মোটো', 'পেইচি...', '।', 'প্যাকেজিং', 'খুব', 'ভ্যালো', 'সাইলো']
    Truncating punctuation: ['আলহামদুলিল্লাহ', '', 'সব', 'ভ্যালো', 'মোটো', 'পেইচি...', 'প্যাকেজিং', 'খুব', 'ভ্যালো', 'সাইলো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', '', 'ভ্যালো', 'মোটো', 'পেইচি...', 'প্যাকেজিং', 'ভ্যালো', 'সাইলো']
    ***************************************************************************************
    Label:  1
    Sentence:  ছবি এবং পণ্য নিখুঁত। দাম taw অমর কাচ সাশ্রয়ী মূল্যের lagche. বক্সিং তা ভালো পাইচি, রাইডার আচরণ তা বেশি ভালো চিলো জেটা অমর খুবি ভালো ল্যাচচে
    Afert Tokenizing:  ['ছবি', 'এবং', 'পণ্য', 'নিখুঁত', '।', 'দাম', 'taw', 'অমর', 'কাচ', 'সাশ্রয়ী', 'মূল্যের', 'lagche', '.', 'বক্সিং', 'তা', 'ভালো', 'পাইচি', ',', 'রাইডার', 'আচরণ', 'তা', 'বেশি', 'ভালো', 'চিলো', 'জেটা', 'অমর', 'খুবি', 'ভালো', 'ল্যাচচে']
    Truncating punctuation: ['ছবি', 'এবং', 'পণ্য', 'নিখুঁত', 'দাম', 'taw', 'অমর', 'কাচ', 'সাশ্রয়ী', 'মূল্যের', 'lagche', 'বক্সিং', 'তা', 'ভালো', 'পাইচি', 'রাইডার', 'আচরণ', 'তা', 'বেশি', 'ভালো', 'চিলো', 'জেটা', 'অমর', 'খুবি', 'ভালো', 'ল্যাচচে']
    Truncating StopWords: ['ছবি', 'পণ্য', 'নিখুঁত', 'দাম', 'taw', 'অমর', 'কাচ', 'সাশ্রয়ী', 'মূল্যের', 'lagche', 'বক্সিং', 'ভালো', 'পাইচি', 'রাইডার', 'আচরণ', 'বেশি', 'ভালো', 'চিলো', 'জেটা', 'অমর', 'খুবি', 'ভালো', 'ল্যাচচে']
    ***************************************************************************************
    Label:  0
    Sentence:  আর চার্জ সর্বোচ্চ এক থেকে দেড় ঘণ্টা থাকে। আসলে কমদামি প্রোডাক্ট যেমন হওয়ার কথা এই ইয়ারফোন টি এর ব্যতিক্রম নয়। আসলে যেটা সত্যি সেটাই বললাম।
    Afert Tokenizing:  ['আর', 'চার্জ', 'সর্বোচ্চ', 'এক', 'থেকে', 'দেড়', 'ঘণ্টা', 'থাকে', '।', 'আসলে', 'কমদামি', 'প্রোডাক্ট', 'যেমন', 'হওয়ার', 'কথা', 'এই', 'ইয়ারফোন', 'টি', 'এর', 'ব্যতিক্রম', 'নয়', '।', 'আসলে', 'যেটা', 'সত্যি', 'সেটাই', 'বললাম', '।']
    Truncating punctuation: ['আর', 'চার্জ', 'সর্বোচ্চ', 'এক', 'থেকে', 'দেড়', 'ঘণ্টা', 'থাকে', 'আসলে', 'কমদামি', 'প্রোডাক্ট', 'যেমন', 'হওয়ার', 'কথা', 'এই', 'ইয়ারফোন', 'টি', 'এর', 'ব্যতিক্রম', 'নয়', 'আসলে', 'যেটা', 'সত্যি', 'সেটাই', 'বললাম']
    Truncating StopWords: ['চার্জ', 'সর্বোচ্চ', 'এক', 'দেড়', 'ঘণ্টা', 'আসলে', 'কমদামি', 'প্রোডাক্ট', 'কথা', 'ইয়ারফোন', 'ব্যতিক্রম', 'নয়', 'আসলে', 'যেটা', 'সত্যি', 'বললাম']
    ***************************************************************************************
    Label:  1
    Sentence:  মানিব্যাগটা অনেক সুন্দর 🥰 এবং এটা গুণগতমান অনেক ভালো। এটা আসল চামড়ার তৈরি। এবং ওয়ালেট টা অনেক সফট
    Afert Tokenizing:  ['মানিব্যাগটা', 'অনেক', 'সুন্দর', '🥰', 'এবং', 'এটা', 'গুণগতমান', 'অনেক', 'ভালো', '।', 'এটা', 'আসল', 'চামড়ার', 'তৈরি', '।', 'এবং', 'ওয়ালেট', 'টা', 'অনেক', 'সফট']
    Truncating punctuation: ['মানিব্যাগটা', 'অনেক', 'সুন্দর', '🥰', 'এবং', 'এটা', 'গুণগতমান', 'অনেক', 'ভালো', 'এটা', 'আসল', 'চামড়ার', 'তৈরি', 'এবং', 'ওয়ালেট', 'টা', 'অনেক', 'সফট']
    Truncating StopWords: ['মানিব্যাগটা', 'সুন্দর', '🥰', 'গুণগতমান', 'ভালো', 'আসল', 'চামড়ার', 'তৈরি', 'ওয়ালেট', 'টা', 'সফট']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই এই প্রডাক্টটা অনেক সুন্দর পুরোটাই চামড়া
    Afert Tokenizing:  ['সত্যিই', 'এই', 'প্রডাক্টটা', 'অনেক', 'সুন্দর', 'পুরোটাই', 'চামড়া']
    Truncating punctuation: ['সত্যিই', 'এই', 'প্রডাক্টটা', 'অনেক', 'সুন্দর', 'পুরোটাই', 'চামড়া']
    Truncating StopWords: ['সত্যিই', 'প্রডাক্টটা', 'সুন্দর', 'পুরোটাই', 'চামড়া']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট কোয়ালিটি অসাধারণ। শেলাই গুলা অনেক সুন্দর।
    Afert Tokenizing:  ['প্রডাক্ট', 'কোয়ালিটি', 'অসাধারণ', '।', 'শেলাই', 'গুলা', 'অনেক', 'সুন্দর', '।']
    Truncating punctuation: ['প্রডাক্ট', 'কোয়ালিটি', 'অসাধারণ', 'শেলাই', 'গুলা', 'অনেক', 'সুন্দর']
    Truncating StopWords: ['প্রডাক্ট', 'কোয়ালিটি', 'অসাধারণ', 'শেলাই', 'গুলা', 'সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  চামড়া কেন জানি আমার একটু লোকোয়ালির মনে হচ্ছে
    Afert Tokenizing:  ['চামড়া', 'কেন', 'জানি', 'আমার', 'একটু', 'লোকোয়ালির', 'মনে', 'হচ্ছে']
    Truncating punctuation: ['চামড়া', 'কেন', 'জানি', 'আমার', 'একটু', 'লোকোয়ালির', 'মনে', 'হচ্ছে']
    Truncating StopWords: ['চামড়া', 'জানি', 'একটু', 'লোকোয়ালির']
    ***************************************************************************************
    Label:  0
    Sentence:  কিছু দিন ব্যবহার করার পরে কাপড় নষ্ট হয়ে যায়।যদি কিনি L সাইজ কিছু দিন পরে ঐইটা হয়ে যায় XL
    Afert Tokenizing:  ['কিছু', 'দিন', 'ব্যবহার', 'করার', 'পরে', 'কাপড়', 'নষ্ট', 'হয়ে', 'যায়।যদি', 'কিনি', 'L', 'সাইজ', 'কিছু', 'দিন', 'পরে', 'ঐইটা', 'হয়ে', 'যায়', 'XL']
    Truncating punctuation: ['কিছু', 'দিন', 'ব্যবহার', 'করার', 'পরে', 'কাপড়', 'নষ্ট', 'হয়ে', 'যায়।যদি', 'কিনি', 'L', 'সাইজ', 'কিছু', 'দিন', 'পরে', 'ঐইটা', 'হয়ে', 'যায়', 'XL']
    Truncating StopWords: ['কাপড়', 'নষ্ট', 'হয়ে', 'যায়।যদি', 'কিনি', 'L', 'সাইজ', 'ঐইটা', 'হয়ে', 'যায়', 'XL']
    ***************************************************************************************
    Label:  1
    Sentence:  ক কথায় অসাধারন! এখানে ৫ ষ্টারের বেশি দেয়া জায়না  নয়ত আরো বেশি দিতাম! এত কম দামে এত ভালো মধু দেয়ার জন্য সেলার ও দারাজকে ধন্যবাদ
    Afert Tokenizing:  ['ক', 'কথায়', 'অসাধারন', '!', 'এখানে', '৫', 'ষ্টারের', 'বেশি', 'দেয়া', 'জায়না', 'নয়ত', 'আরো', 'বেশি', 'দিতাম', '!', 'এত', 'কম', 'দামে', 'এত', 'ভালো', 'মধু', 'দেয়ার', 'জন্য', 'সেলার', 'ও', 'দারাজকে', 'ধন্যবাদ']
    Truncating punctuation: ['ক', 'কথায়', 'অসাধারন', 'এখানে', '৫', 'ষ্টারের', 'বেশি', 'দেয়া', 'জায়না', 'নয়ত', 'আরো', 'বেশি', 'দিতাম', 'এত', 'কম', 'দামে', 'এত', 'ভালো', 'মধু', 'দেয়ার', 'জন্য', 'সেলার', 'ও', 'দারাজকে', 'ধন্যবাদ']
    Truncating StopWords: ['ক', 'কথায়', 'অসাধারন', '৫', 'ষ্টারের', 'বেশি', 'দেয়া', 'জায়না', 'নয়ত', 'আরো', 'বেশি', 'দিতাম', 'কম', 'দামে', 'ভালো', 'মধু', 'দেয়ার', 'সেলার', 'দারাজকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো মধু।  চোখ বন্ধ করে নিতে পারেন
    Afert Tokenizing:  ['ভালো', 'মধু', '।', 'চোখ', 'বন্ধ', 'করে', 'নিতে', 'পারেন']
    Truncating punctuation: ['ভালো', 'মধু', 'চোখ', 'বন্ধ', 'করে', 'নিতে', 'পারেন']
    Truncating StopWords: ['ভালো', 'মধু', 'চোখ', 'বন্ধ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ্‌ ঠকিনাই! যেমনটা আশা করেছি তার চাইতে অনেক ভালো মধু
    Afert Tokenizing:  ['আলহামদুলিল্লাহ্\u200c', 'ঠকিনাই', '!', 'যেমনটা', 'আশা', 'করেছি', 'তার', 'চাইতে', 'অনেক', 'ভালো', 'মধু']
    Truncating punctuation: ['আলহামদুলিল্লাহ্\u200c', 'ঠকিনাই', 'যেমনটা', 'আশা', 'করেছি', 'তার', 'চাইতে', 'অনেক', 'ভালো', 'মধু']
    Truncating StopWords: ['আলহামদুলিল্লাহ্\u200c', 'ঠকিনাই', 'যেমনটা', 'আশা', 'করেছি', 'চাইতে', 'ভালো', 'মধু']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্য পেয়ে আমি সন্তুষ্ট। সবকিছু ঠিক আছে, এবং যে দ্বিতীয় শেষ ছবিটি এটির ভিতরে ফোন রেখে তোলা হয়েছিল, যে কেউ চায়, সে এটা কিনতে পারে।
    Afert Tokenizing:  ['পণ্য', 'পেয়ে', 'আমি', 'সন্তুষ্ট', '।', 'সবকিছু', 'ঠিক', 'আছে', ',', 'এবং', 'যে', 'দ্বিতীয়', 'শেষ', 'ছবিটি', 'এটির', 'ভিতরে', 'ফোন', 'রেখে', 'তোলা', 'হয়েছিল', ',', 'যে', 'কেউ', 'চায়', ',', 'সে', 'এটা', 'কিনতে', 'পারে', '।']
    Truncating punctuation: ['পণ্য', 'পেয়ে', 'আমি', 'সন্তুষ্ট', 'সবকিছু', 'ঠিক', 'আছে', 'এবং', 'যে', 'দ্বিতীয়', 'শেষ', 'ছবিটি', 'এটির', 'ভিতরে', 'ফোন', 'রেখে', 'তোলা', 'হয়েছিল', 'যে', 'কেউ', 'চায়', 'সে', 'এটা', 'কিনতে', 'পারে']
    Truncating StopWords: ['পণ্য', 'সন্তুষ্ট', 'সবকিছু', 'ঠিক', 'দ্বিতীয়', 'শেষ', 'ছবিটি', 'এটির', 'ভিতরে', 'ফোন', 'তোলা', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট টা অনেক অনেক ভালো নিরদ্বিধায় নিতে পারেন সবাই,আমি বন্ধুদের সাথে নদিতে,পুখুরে কাটানো মজার সময় গুলোর ছবি তুলার জন্য নিছি
    Afert Tokenizing:  ['প্রডাক্ট', 'টা', 'অনেক', 'অনেক', 'ভালো', 'নিরদ্বিধায়', 'নিতে', 'পারেন', 'সবাই,আমি', 'বন্ধুদের', 'সাথে', 'নদিতে,পুখুরে', 'কাটানো', 'মজার', 'সময়', 'গুলোর', 'ছবি', 'তুলার', 'জন্য', 'নিছি']
    Truncating punctuation: ['প্রডাক্ট', 'টা', 'অনেক', 'অনেক', 'ভালো', 'নিরদ্বিধায়', 'নিতে', 'পারেন', 'সবাই,আমি', 'বন্ধুদের', 'সাথে', 'নদিতে,পুখুরে', 'কাটানো', 'মজার', 'সময়', 'গুলোর', 'ছবি', 'তুলার', 'জন্য', 'নিছি']
    Truncating StopWords: ['প্রডাক্ট', 'টা', 'ভালো', 'নিরদ্বিধায়', 'সবাই,আমি', 'বন্ধুদের', 'সাথে', 'নদিতে,পুখুরে', 'কাটানো', 'মজার', 'সময়', 'গুলোর', 'ছবি', 'তুলার', 'নিছি']
    ***************************************************************************************
    Label:  0
    Sentence:  পন্যটা আমার কাছে আজকে এসে পৌঁছেছে..অনেকদিন লেট  করে আসছে পন্যটা
    Afert Tokenizing:  ['পন্যটা', 'আমার', 'কাছে', 'আজকে', 'এসে', 'পৌঁছেছে..অনেকদিন', 'লেট', 'করে', 'আসছে', 'পন্যটা']
    Truncating punctuation: ['পন্যটা', 'আমার', 'কাছে', 'আজকে', 'এসে', 'পৌঁছেছে..অনেকদিন', 'লেট', 'করে', 'আসছে', 'পন্যটা']
    Truncating StopWords: ['পন্যটা', 'আজকে', 'পৌঁছেছে..অনেকদিন', 'লেট', 'আসছে', 'পন্যটা']
    ***************************************************************************************
    Label:  1
    Sentence:  ছবির মতো ই পেয়েছি। খুব পছন্দ হয়েছে
    Afert Tokenizing:  ['ছবির', 'মতো', 'ই', 'পেয়েছি', '।', 'খুব', 'পছন্দ', 'হয়েছে']
    Truncating punctuation: ['ছবির', 'মতো', 'ই', 'পেয়েছি', 'খুব', 'পছন্দ', 'হয়েছে']
    Truncating StopWords: ['ছবির', 'পেয়েছি', 'পছন্দ', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  তুলনামুলক দাম একটু বেশি হয়েছে, কিন্তু এই দামে বাসায় বসে প্রডাক্ট হাতে পেয়ে আমি অত্যন্ত খুশি
    Afert Tokenizing:  ['তুলনামুলক', 'দাম', 'একটু', 'বেশি', 'হয়েছে', ',', 'কিন্তু', 'এই', 'দামে', 'বাসায়', 'বসে', 'প্রডাক্ট', 'হাতে', 'পেয়ে', 'আমি', 'অত্যন্ত', 'খুশি']
    Truncating punctuation: ['তুলনামুলক', 'দাম', 'একটু', 'বেশি', 'হয়েছে', 'কিন্তু', 'এই', 'দামে', 'বাসায়', 'বসে', 'প্রডাক্ট', 'হাতে', 'পেয়ে', 'আমি', 'অত্যন্ত', 'খুশি']
    Truncating StopWords: ['তুলনামুলক', 'দাম', 'একটু', 'বেশি', 'হয়েছে', 'দামে', 'বাসায়', 'প্রডাক্ট', 'হাতে', 'পেয়ে', 'অত্যন্ত', 'খুশি']
    ***************************************************************************************
    Label:  1
    Sentence:  এককথায় এই দামে প্রডাক্ট আমার অনেক পছন্দ হয়েছে, আপনারা চাইলে নিতে পারেন
    Afert Tokenizing:  ['এককথায়', 'এই', 'দামে', 'প্রডাক্ট', 'আমার', 'অনেক', 'পছন্দ', 'হয়েছে', ',', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['এককথায়', 'এই', 'দামে', 'প্রডাক্ট', 'আমার', 'অনেক', 'পছন্দ', 'হয়েছে', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['এককথায়', 'দামে', 'প্রডাক্ট', 'পছন্দ', 'হয়েছে', 'আপনারা', 'চাইলে']
    ***************************************************************************************
    Label:  0
    Sentence:  কেউ নিবেন না চাইছি কি আর দিসে কি রিটার্ন নিয়ে যান টাকা টায় জলে গেলো
    Afert Tokenizing:  ['কেউ', 'নিবেন', 'না', 'চাইছি', 'কি', 'আর', 'দিসে', 'কি', 'রিটার্ন', 'নিয়ে', 'যান', 'টাকা', 'টায়', 'জলে', 'গেলো']
    Truncating punctuation: ['কেউ', 'নিবেন', 'না', 'চাইছি', 'কি', 'আর', 'দিসে', 'কি', 'রিটার্ন', 'নিয়ে', 'যান', 'টাকা', 'টায়', 'জলে', 'গেলো']
    Truncating StopWords: ['নিবেন', 'না', 'চাইছি', 'দিসে', 'রিটার্ন', 'টাকা', 'টায়', 'জলে', 'গেলো']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি খুব ভালো, দেখতেও বেশ। তবে সাইজে একটু বড়, তাই যারা অর্ডার করবেন এক সাইজ ছোট অর্ডার কিরবেন
    Afert Tokenizing:  ['কোয়ালিটি', 'খুব', 'ভালো', ',', 'দেখতেও', 'বেশ', '।', 'তবে', 'সাইজে', 'একটু', 'বড়', ',', 'তাই', 'যারা', 'অর্ডার', 'করবেন', 'এক', 'সাইজ', 'ছোট', 'অর্ডার', 'কিরবেন']
    Truncating punctuation: ['কোয়ালিটি', 'খুব', 'ভালো', 'দেখতেও', 'বেশ', 'তবে', 'সাইজে', 'একটু', 'বড়', 'তাই', 'যারা', 'অর্ডার', 'করবেন', 'এক', 'সাইজ', 'ছোট', 'অর্ডার', 'কিরবেন']
    Truncating StopWords: ['কোয়ালিটি', 'ভালো', 'দেখতেও', 'সাইজে', 'একটু', 'বড়', 'অর্ডার', 'এক', 'সাইজ', 'ছোট', 'অর্ডার', 'কিরবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  ৫০০ টাকা বাজেটে সেরা একটি ট্রিমার, ৪৫ মিনিটের মত ব্যাকআপ।
    Afert Tokenizing:  ['৫০০', 'টাকা', 'বাজেটে', 'সেরা', 'একটি', 'ট্রিমার', ',', '৪৫', 'মিনিটের', 'মত', 'ব্যাকআপ', '।']
    Truncating punctuation: ['৫০০', 'টাকা', 'বাজেটে', 'সেরা', 'একটি', 'ট্রিমার', '৪৫', 'মিনিটের', 'মত', 'ব্যাকআপ']
    Truncating StopWords: ['৫০০', 'টাকা', 'বাজেটে', 'সেরা', 'ট্রিমার', '৪৫', 'মিনিটের', 'মত', 'ব্যাকআপ']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেজিং খুবই নরমাল ছিল বাট প্রোডাক্টটির কোন ক্ষতি হয়নি
    Afert Tokenizing:  ['প্যাকেজিং', 'খুবই', 'নরমাল', 'ছিল', 'বাট', 'প্রোডাক্টটির', 'কোন', 'ক্ষতি', 'হয়নি']
    Truncating punctuation: ['প্যাকেজিং', 'খুবই', 'নরমাল', 'ছিল', 'বাট', 'প্রোডাক্টটির', 'কোন', 'ক্ষতি', 'হয়নি']
    Truncating StopWords: ['প্যাকেজিং', 'খুবই', 'নরমাল', 'বাট', 'প্রোডাক্টটির', 'ক্ষতি', 'হয়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট একদম ইনটেক পেয়েছি।আসা করছি ভালোই হবে। যার প্রয়োজন নিতে পারেন
    Afert Tokenizing:  ['প্রডাক্ট', 'একদম', 'ইনটেক', 'পেয়েছি।আসা', 'করছি', 'ভালোই', 'হবে', '।', 'যার', 'প্রয়োজন', 'নিতে', 'পারেন']
    Truncating punctuation: ['প্রডাক্ট', 'একদম', 'ইনটেক', 'পেয়েছি।আসা', 'করছি', 'ভালোই', 'হবে', 'যার', 'প্রয়োজন', 'নিতে', 'পারেন']
    Truncating StopWords: ['প্রডাক্ট', 'একদম', 'ইনটেক', 'পেয়েছি।আসা', 'করছি', 'ভালোই', 'প্রয়োজন']
    ***************************************************************************************
    Label:  1
    Sentence:  গেঞ্জির রং চেয়েছিলাম যে রকম ছবিতে আছে তেমনটাই পেয়েছি দুটো লাল একটি নীল
    Afert Tokenizing:  ['গেঞ্জির', 'রং', 'চেয়েছিলাম', 'যে', 'রকম', 'ছবিতে', 'আছে', 'তেমনটাই', 'পেয়েছি', 'দুটো', 'লাল', 'একটি', 'নীল']
    Truncating punctuation: ['গেঞ্জির', 'রং', 'চেয়েছিলাম', 'যে', 'রকম', 'ছবিতে', 'আছে', 'তেমনটাই', 'পেয়েছি', 'দুটো', 'লাল', 'একটি', 'নীল']
    Truncating StopWords: ['গেঞ্জির', 'রং', 'চেয়েছিলাম', 'ছবিতে', 'তেমনটাই', 'পেয়েছি', 'লাল', 'নীল']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্টা খুব ভাল,কম দামে এতো ভাল প্রোডাক্ট পাব ভাবিনি, কাপড়টা ও ভালো, কেউ চাইলে নিতে পারেন
    Afert Tokenizing:  ['প্রোডাক্টা', 'খুব', 'ভাল,কম', 'দামে', 'এতো', 'ভাল', 'প্রোডাক্ট', 'পাব', 'ভাবিনি', ',', 'কাপড়টা', 'ও', 'ভালো', ',', 'কেউ', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['প্রোডাক্টা', 'খুব', 'ভাল,কম', 'দামে', 'এতো', 'ভাল', 'প্রোডাক্ট', 'পাব', 'ভাবিনি', 'কাপড়টা', 'ও', 'ভালো', 'কেউ', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['প্রোডাক্টা', 'ভাল,কম', 'দামে', 'এতো', 'ভাল', 'প্রোডাক্ট', 'পাব', 'ভাবিনি', 'কাপড়টা', 'ভালো', 'চাইলে']
    ***************************************************************************************
    Label:  1
    Sentence:  টি--শার্ট একদম ঠিক আছে।দাম অনুযায়ী অনেক বেশিকিছু
    Afert Tokenizing:  ['টি--শার্ট', 'একদম', 'ঠিক', 'আছে।দাম', 'অনুযায়ী', 'অনেক', 'বেশিকিছু']
    Truncating punctuation: ['টি--শার্ট', 'একদম', 'ঠিক', 'আছে।দাম', 'অনুযায়ী', 'অনেক', 'বেশিকিছু']
    Truncating StopWords: ['টি--শার্ট', 'একদম', 'ঠিক', 'আছে।দাম', 'অনুযায়ী', 'বেশিকিছু']
    ***************************************************************************************
    Label:  1
    Sentence:  ব্যবহার করলাম ২৪ ঘন্টা ,, ভালোই কাজ করেছ + সব কিছু ভালোই ঠিক টাক আছে
    Afert Tokenizing:  ['ব্যবহার', 'করলাম', '২৪', 'ঘন্টা', ',', ',', 'ভালোই', 'কাজ', 'করেছ', '+', 'সব', 'কিছু', 'ভালোই', 'ঠিক', 'টাক', 'আছে']
    Truncating punctuation: ['ব্যবহার', 'করলাম', '২৪', 'ঘন্টা', 'ভালোই', 'কাজ', 'করেছ', '+', 'সব', 'কিছু', 'ভালোই', 'ঠিক', 'টাক', 'আছে']
    Truncating StopWords: ['করলাম', '২৪', 'ঘন্টা', 'ভালোই', 'করেছ', '+', 'ভালোই', 'ঠিক', 'টাক']
    ***************************************************************************************
    Label:  0
    Sentence:  এটি একটি খুব দরকারী অ্যালার্ম ঘড়ি তবে অ্যালার্মের শব্দটি যদি আরও কিছুটা বেশি হত তবে এটি আরও ভাল হত
    Afert Tokenizing:  ['এটি', 'একটি', 'খুব', 'দরকারী', 'অ্যালার্ম', 'ঘড়ি', 'তবে', 'অ্যালার্মের', 'শব্দটি', 'যদি', 'আরও', 'কিছুটা', 'বেশি', 'হত', 'তবে', 'এটি', 'আরও', 'ভাল', 'হত']
    Truncating punctuation: ['এটি', 'একটি', 'খুব', 'দরকারী', 'অ্যালার্ম', 'ঘড়ি', 'তবে', 'অ্যালার্মের', 'শব্দটি', 'যদি', 'আরও', 'কিছুটা', 'বেশি', 'হত', 'তবে', 'এটি', 'আরও', 'ভাল', 'হত']
    Truncating StopWords: ['দরকারী', 'অ্যালার্ম', 'ঘড়ি', 'অ্যালার্মের', 'শব্দটি', 'কিছুটা', 'বেশি', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্যটি আশ্চর্যজনক। আমি তো এটা আমার মায়ের জন্য কিনেছি
    Afert Tokenizing:  ['পণ্যটি', 'আশ্চর্যজনক', '।', 'আমি', 'তো', 'এটা', 'আমার', 'মায়ের', 'জন্য', 'কিনেছি']
    Truncating punctuation: ['পণ্যটি', 'আশ্চর্যজনক', 'আমি', 'তো', 'এটা', 'আমার', 'মায়ের', 'জন্য', 'কিনেছি']
    Truncating StopWords: ['পণ্যটি', 'আশ্চর্যজনক', 'মায়ের', 'কিনেছি']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম হিসেবে খারাপ না ভালোই...তবে চশমার ফ্রেমের লাল ডিজাইনটা রাবাবের,কিছুদিন পর উঠে যাওয়ার সম্ভাবনা আছে
    Afert Tokenizing:  ['দাম', 'হিসেবে', 'খারাপ', 'না', 'ভালোই...তবে', 'চশমার', 'ফ্রেমের', 'লাল', 'ডিজাইনটা', 'রাবাবের,কিছুদিন', 'পর', 'উঠে', 'যাওয়ার', 'সম্ভাবনা', 'আছে']
    Truncating punctuation: ['দাম', 'হিসেবে', 'খারাপ', 'না', 'ভালোই...তবে', 'চশমার', 'ফ্রেমের', 'লাল', 'ডিজাইনটা', 'রাবাবের,কিছুদিন', 'পর', 'উঠে', 'যাওয়ার', 'সম্ভাবনা', 'আছে']
    Truncating StopWords: ['দাম', 'হিসেবে', 'খারাপ', 'না', 'ভালোই...তবে', 'চশমার', 'ফ্রেমের', 'লাল', 'ডিজাইনটা', 'রাবাবের,কিছুদিন', 'উঠে', 'যাওয়ার', 'সম্ভাবনা']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব অল্প সময়ে পেয়েছি কিন্তু কোন কাভার বা বক্স দেওয়া হয়নি। আগে জানলে ওডার টা করতাম না তাই দুই স্রার দিলাম
    Afert Tokenizing:  ['খুব', 'অল্প', 'সময়ে', 'পেয়েছি', 'কিন্তু', 'কোন', 'কাভার', 'বা', 'বক্স', 'দেওয়া', 'হয়নি', '।', 'আগে', 'জানলে', 'ওডার', 'টা', 'করতাম', 'না', 'তাই', 'দুই', 'স্রার', 'দিলাম']
    Truncating punctuation: ['খুব', 'অল্প', 'সময়ে', 'পেয়েছি', 'কিন্তু', 'কোন', 'কাভার', 'বা', 'বক্স', 'দেওয়া', 'হয়নি', 'আগে', 'জানলে', 'ওডার', 'টা', 'করতাম', 'না', 'তাই', 'দুই', 'স্রার', 'দিলাম']
    Truncating StopWords: ['অল্প', 'সময়ে', 'পেয়েছি', 'কাভার', 'বক্স', 'হয়নি', 'জানলে', 'ওডার', 'টা', 'করতাম', 'না', 'স্রার', 'দিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভাল ছিল প্রতিটা মাক্স ☺️। পাশাপাশি KN95 মাক্স টা  থাকায় সম্পূর্ণ প্যাকেজটি এই মূল্যে অসাধারন ছিল।
    Afert Tokenizing:  ['অনেক', 'ভাল', 'ছিল', 'প্রতিটা', 'মাক্স', '☺️', '।', 'পাশাপাশি', 'KN95', 'মাক্স', 'টা', 'থাকায়', 'সম্পূর্ণ', 'প্যাকেজটি', 'এই', 'মূল্যে', 'অসাধারন', 'ছিল', '।']
    Truncating punctuation: ['অনেক', 'ভাল', 'ছিল', 'প্রতিটা', 'মাক্স', '☺️', 'পাশাপাশি', 'KN95', 'মাক্স', 'টা', 'থাকায়', 'সম্পূর্ণ', 'প্যাকেজটি', 'এই', 'মূল্যে', 'অসাধারন', 'ছিল']
    Truncating StopWords: ['ভাল', 'প্রতিটা', 'মাক্স', '☺️', 'পাশাপাশি', 'KN95', 'মাক্স', 'টা', 'সম্পূর্ণ', 'প্যাকেজটি', 'মূল্যে', 'অসাধারন']
    ***************************************************************************************
    Label:  1
    Sentence:  সব কিছু ঠিক আছে কিন্তু ভিতরে মোটা সুতার কাপর দেওয়া হয়েছে। দাম অনুযায়ী ঠিক আছে
    Afert Tokenizing:  ['সব', 'কিছু', 'ঠিক', 'আছে', 'কিন্তু', 'ভিতরে', 'মোটা', 'সুতার', 'কাপর', 'দেওয়া', 'হয়েছে', '।', 'দাম', 'অনুযায়ী', 'ঠিক', 'আছে']
    Truncating punctuation: ['সব', 'কিছু', 'ঠিক', 'আছে', 'কিন্তু', 'ভিতরে', 'মোটা', 'সুতার', 'কাপর', 'দেওয়া', 'হয়েছে', 'দাম', 'অনুযায়ী', 'ঠিক', 'আছে']
    Truncating StopWords: ['ঠিক', 'ভিতরে', 'মোটা', 'সুতার', 'কাপর', 'হয়েছে', 'দাম', 'অনুযায়ী', 'ঠিক']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ঠিক যেমন তা চাইছি তেমন পাইছি আর কালার তো সেম পাইছি
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'ঠিক', 'যেমন', 'তা', 'চাইছি', 'তেমন', 'পাইছি', 'আর', 'কালার', 'তো', 'সেম', 'পাইছি']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ঠিক', 'যেমন', 'তা', 'চাইছি', 'তেমন', 'পাইছি', 'আর', 'কালার', 'তো', 'সেম', 'পাইছি']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ঠিক', 'চাইছি', 'পাইছি', 'কালার', 'সেম', 'পাইছি']
    ***************************************************************************************
    Label:  1
    Sentence:  জিপার সিকিউরিটির কার্ড হোল্ডারটি সত্যিই খুব সুন্দর এবং আউটলুক ডিজাইন চমৎকার।
    Afert Tokenizing:  ['জিপার', 'সিকিউরিটির', 'কার্ড', 'হোল্ডারটি', 'সত্যিই', 'খুব', 'সুন্দর', 'এবং', 'আউটলুক', 'ডিজাইন', 'চমৎকার', '।']
    Truncating punctuation: ['জিপার', 'সিকিউরিটির', 'কার্ড', 'হোল্ডারটি', 'সত্যিই', 'খুব', 'সুন্দর', 'এবং', 'আউটলুক', 'ডিজাইন', 'চমৎকার']
    Truncating StopWords: ['জিপার', 'সিকিউরিটির', 'কার্ড', 'হোল্ডারটি', 'সত্যিই', 'সুন্দর', 'আউটলুক', 'ডিজাইন', 'চমৎকার']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশাআল্লাহ অনেক ভালো টি-সার্ট টা
    Afert Tokenizing:  ['মাশাআল্লাহ', 'অনেক', 'ভালো', 'টি-সার্ট', 'টা']
    Truncating punctuation: ['মাশাআল্লাহ', 'অনেক', 'ভালো', 'টি-সার্ট', 'টা']
    Truncating StopWords: ['মাশাআল্লাহ', 'ভালো', 'টি-সার্ট', 'টা']
    ***************************************************************************************
    Label:  1
    Sentence:  যেরকম দেখেছিলাম ঠিক সেরকম পেয়েছি।কাপড়ের কোয়ালিটি ও অনেক ভালো।
    Afert Tokenizing:  ['যেরকম', 'দেখেছিলাম', 'ঠিক', 'সেরকম', 'পেয়েছি।কাপড়ের', 'কোয়ালিটি', 'ও', 'অনেক', 'ভালো', '।']
    Truncating punctuation: ['যেরকম', 'দেখেছিলাম', 'ঠিক', 'সেরকম', 'পেয়েছি।কাপড়ের', 'কোয়ালিটি', 'ও', 'অনেক', 'ভালো']
    Truncating StopWords: ['যেরকম', 'দেখেছিলাম', 'ঠিক', 'সেরকম', 'পেয়েছি।কাপড়ের', 'কোয়ালিটি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  সব ঠিকঠাক পেয়েছি খুব সুন্দর হয়েছে ধন্যবাদ সেলারকে আমার কাছে 100% ওয়াটারপ্রুফ মনে হয়েছে আমার ভালো লেগেছে আপনারা চাইলে নিতে পারেন
    Afert Tokenizing:  ['সব', 'ঠিকঠাক', 'পেয়েছি', 'খুব', 'সুন্দর', 'হয়েছে', 'ধন্যবাদ', 'সেলারকে', 'আমার', 'কাছে', '100%', 'ওয়াটারপ্রুফ', 'মনে', 'হয়েছে', 'আমার', 'ভালো', 'লেগেছে', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['সব', 'ঠিকঠাক', 'পেয়েছি', 'খুব', 'সুন্দর', 'হয়েছে', 'ধন্যবাদ', 'সেলারকে', 'আমার', 'কাছে', '100%', 'ওয়াটারপ্রুফ', 'মনে', 'হয়েছে', 'আমার', 'ভালো', 'লেগেছে', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['ঠিকঠাক', 'পেয়েছি', 'সুন্দর', 'ধন্যবাদ', 'সেলারকে', '100%', 'ওয়াটারপ্রুফ', 'ভালো', 'লেগেছে', 'আপনারা', 'চাইলে']
    ***************************************************************************************
    Label:  0
    Sentence:  এটার সেলাই এর বিষয়ে সেলারকে আর ও একটু সচেতন হতে হবে। কাপড় সেলাই এর মত বডার দিলে অনেক ভাল হবে।
    Afert Tokenizing:  ['এটার', 'সেলাই', 'এর', 'বিষয়ে', 'সেলারকে', 'আর', 'ও', 'একটু', 'সচেতন', 'হতে', 'হবে', '।', 'কাপড়', 'সেলাই', 'এর', 'মত', 'বডার', 'দিলে', 'অনেক', 'ভাল', 'হবে', '।']
    Truncating punctuation: ['এটার', 'সেলাই', 'এর', 'বিষয়ে', 'সেলারকে', 'আর', 'ও', 'একটু', 'সচেতন', 'হতে', 'হবে', 'কাপড়', 'সেলাই', 'এর', 'মত', 'বডার', 'দিলে', 'অনেক', 'ভাল', 'হবে']
    Truncating StopWords: ['এটার', 'সেলাই', 'বিষয়ে', 'সেলারকে', 'একটু', 'সচেতন', 'কাপড়', 'সেলাই', 'মত', 'বডার', 'দিলে', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  এছাড়া দাম অনুযায়ী কাপড়ের মান মোটামুটি ভাল তবে  ভাবছিলাম কাপড় একটু মোটা হবে।
    Afert Tokenizing:  ['এছাড়া', 'দাম', 'অনুযায়ী', 'কাপড়ের', 'মান', 'মোটামুটি', 'ভাল', 'তবে', 'ভাবছিলাম', 'কাপড়', 'একটু', 'মোটা', 'হবে', '।']
    Truncating punctuation: ['এছাড়া', 'দাম', 'অনুযায়ী', 'কাপড়ের', 'মান', 'মোটামুটি', 'ভাল', 'তবে', 'ভাবছিলাম', 'কাপড়', 'একটু', 'মোটা', 'হবে']
    Truncating StopWords: ['এছাড়া', 'দাম', 'অনুযায়ী', 'কাপড়ের', 'মান', 'মোটামুটি', 'ভাল', 'ভাবছিলাম', 'কাপড়', 'একটু', 'মোটা']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেজিং ভালো ছিল। প্রোডাক্ট ও ভালো ওয়াটার প্রুফ।  বলতে গেলে খুব তাড়াতাড়িই প্রোডাক্ট হাতে পেয়েছি। সবমিলিয়ে সন্তোষজনক
    Afert Tokenizing:  ['প্যাকেজিং', 'ভালো', 'ছিল', '।', 'প্রোডাক্ট', 'ও', 'ভালো', 'ওয়াটার', 'প্রুফ', '।', 'বলতে', 'গেলে', 'খুব', 'তাড়াতাড়িই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', '।', 'সবমিলিয়ে', 'সন্তোষজনক']
    Truncating punctuation: ['প্যাকেজিং', 'ভালো', 'ছিল', 'প্রোডাক্ট', 'ও', 'ভালো', 'ওয়াটার', 'প্রুফ', 'বলতে', 'গেলে', 'খুব', 'তাড়াতাড়িই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'সবমিলিয়ে', 'সন্তোষজনক']
    Truncating StopWords: ['প্যাকেজিং', 'ভালো', 'প্রোডাক্ট', 'ভালো', 'ওয়াটার', 'প্রুফ', 'তাড়াতাড়িই', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'সবমিলিয়ে', 'সন্তোষজনক']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট টা খুবই ভালো প্যাকেটিং টা অনেক সুন্দর ছিল ঘড়িটা অনেক সুন্দর এক কথায় বলতে গেলে অসাধারণ
    Afert Tokenizing:  ['প্রোডাক্ট', 'টা', 'খুবই', 'ভালো', 'প্যাকেটিং', 'টা', 'অনেক', 'সুন্দর', 'ছিল', 'ঘড়িটা', 'অনেক', 'সুন্দর', 'এক', 'কথায়', 'বলতে', 'গেলে', 'অসাধারণ']
    Truncating punctuation: ['প্রোডাক্ট', 'টা', 'খুবই', 'ভালো', 'প্যাকেটিং', 'টা', 'অনেক', 'সুন্দর', 'ছিল', 'ঘড়িটা', 'অনেক', 'সুন্দর', 'এক', 'কথায়', 'বলতে', 'গেলে', 'অসাধারণ']
    Truncating StopWords: ['প্রোডাক্ট', 'টা', 'খুবই', 'ভালো', 'প্যাকেটিং', 'টা', 'সুন্দর', 'ঘড়িটা', 'সুন্দর', 'এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবই চমৎকার । সত্যিই আমি অবাক এই প্রথম অনলাইন থেকে কোন কিছু কিনলাম এবং আমি সম্পূর্ণভাবে স্যাটিস্ফাইড
    Afert Tokenizing:  ['খুবই', 'চমৎকার', '', '।', 'সত্যিই', 'আমি', 'অবাক', 'এই', 'প্রথম', 'অনলাইন', 'থেকে', 'কোন', 'কিছু', 'কিনলাম', 'এবং', 'আমি', 'সম্পূর্ণভাবে', 'স্যাটিস্ফাইড']
    Truncating punctuation: ['খুবই', 'চমৎকার', '', 'সত্যিই', 'আমি', 'অবাক', 'এই', 'প্রথম', 'অনলাইন', 'থেকে', 'কোন', 'কিছু', 'কিনলাম', 'এবং', 'আমি', 'সম্পূর্ণভাবে', 'স্যাটিস্ফাইড']
    Truncating StopWords: ['খুবই', 'চমৎকার', '', 'সত্যিই', 'অবাক', 'অনলাইন', 'কিনলাম', 'সম্পূর্ণভাবে', 'স্যাটিস্ফাইড']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ যেরকমটা চেয়েছি সেরকমই পেয়েছি! ২ টা অর্ডার দিছিলাম আর ২ টাই পেয়েছি
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'যেরকমটা', 'চেয়েছি', 'সেরকমই', 'পেয়েছি', '!', '২', 'টা', 'অর্ডার', 'দিছিলাম', 'আর', '২', 'টাই', 'পেয়েছি']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'যেরকমটা', 'চেয়েছি', 'সেরকমই', 'পেয়েছি', '২', 'টা', 'অর্ডার', 'দিছিলাম', 'আর', '২', 'টাই', 'পেয়েছি']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'যেরকমটা', 'চেয়েছি', 'সেরকমই', 'পেয়েছি', '২', 'টা', 'অর্ডার', 'দিছিলাম', '২', 'টাই', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  অল্প দামের মধ্যে ভালোই। ব্যায়াম করার জন্য
    Afert Tokenizing:  ['অল্প', 'দামের', 'মধ্যে', 'ভালোই', '।', 'ব্যায়াম', 'করার', 'জন্য']
    Truncating punctuation: ['অল্প', 'দামের', 'মধ্যে', 'ভালোই', 'ব্যায়াম', 'করার', 'জন্য']
    Truncating StopWords: ['অল্প', 'দামের', 'ভালোই', 'ব্যায়াম']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই বাজে মানের পন্য, এক দিন ও ব্যবহার করা যায় নি তার আগেই ভেঙে গেছে, খুবই নিম্নমানের প্লাস্টিকের ব্যবহার হয়েছে পন্য টি তে। সবাই কে না নেওয়ার অনুরোধ করব।
    Afert Tokenizing:  ['খুবই', 'বাজে', 'মানের', 'পন্য', ',', 'এক', 'দিন', 'ও', 'ব্যবহার', 'করা', 'যায়', 'নি', 'তার', 'আগেই', 'ভেঙে', 'গেছে', ',', 'খুবই', 'নিম্নমানের', 'প্লাস্টিকের', 'ব্যবহার', 'হয়েছে', 'পন্য', 'টি', 'তে', '।', 'সবাই', 'কে', 'না', 'নেওয়ার', 'অনুরোধ', 'করব', '।']
    Truncating punctuation: ['খুবই', 'বাজে', 'মানের', 'পন্য', 'এক', 'দিন', 'ও', 'ব্যবহার', 'করা', 'যায়', 'নি', 'তার', 'আগেই', 'ভেঙে', 'গেছে', 'খুবই', 'নিম্নমানের', 'প্লাস্টিকের', 'ব্যবহার', 'হয়েছে', 'পন্য', 'টি', 'তে', 'সবাই', 'কে', 'না', 'নেওয়ার', 'অনুরোধ', 'করব']
    Truncating StopWords: ['খুবই', 'বাজে', 'মানের', 'পন্য', 'এক', 'যায়', 'নি', 'ভেঙে', 'খুবই', 'নিম্নমানের', 'প্লাস্টিকের', 'হয়েছে', 'পন্য', 'তে', 'সবাই', 'না', 'নেওয়ার', 'অনুরোধ', 'করব']
    ***************************************************************************************
    Label:  1
    Sentence:  আমাকে একটি অবিশ্বাস্যভাবে সুন্দর ব্যাগ দেওয়ার জন্য আপনাকে অনেক ধন্যবাদ।
    Afert Tokenizing:  ['আমাকে', 'একটি', 'অবিশ্বাস্যভাবে', 'সুন্দর', 'ব্যাগ', 'দেওয়ার', 'জন্য', 'আপনাকে', 'অনেক', 'ধন্যবাদ', '।']
    Truncating punctuation: ['আমাকে', 'একটি', 'অবিশ্বাস্যভাবে', 'সুন্দর', 'ব্যাগ', 'দেওয়ার', 'জন্য', 'আপনাকে', 'অনেক', 'ধন্যবাদ']
    Truncating StopWords: ['অবিশ্বাস্যভাবে', 'সুন্দর', 'ব্যাগ', 'আপনাকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  ব্যাগটি খুবই ভালো মানের। পণ্যনিয়ে খুব খুশি।
    Afert Tokenizing:  ['ব্যাগটি', 'খুবই', 'ভালো', 'মানের', '।', 'পণ্যনিয়ে', 'খুব', 'খুশি', '।']
    Truncating punctuation: ['ব্যাগটি', 'খুবই', 'ভালো', 'মানের', 'পণ্যনিয়ে', 'খুব', 'খুশি']
    Truncating StopWords: ['ব্যাগটি', 'খুবই', 'ভালো', 'মানের', 'পণ্যনিয়ে', 'খুশি']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি অনেক ভালো। ১৫০০ টাকা প্রইজে অনেক ভালো ব্যাগ পেয়েছি। লোকাল শপে সেইম ব্যাগ দিগুন দাম
    Afert Tokenizing:  ['কোয়ালিটি', 'অনেক', 'ভালো', '।', '১৫০০', 'টাকা', 'প্রইজে', 'অনেক', 'ভালো', 'ব্যাগ', 'পেয়েছি', '।', 'লোকাল', 'শপে', 'সেইম', 'ব্যাগ', 'দিগুন', 'দাম']
    Truncating punctuation: ['কোয়ালিটি', 'অনেক', 'ভালো', '১৫০০', 'টাকা', 'প্রইজে', 'অনেক', 'ভালো', 'ব্যাগ', 'পেয়েছি', 'লোকাল', 'শপে', 'সেইম', 'ব্যাগ', 'দিগুন', 'দাম']
    Truncating StopWords: ['কোয়ালিটি', 'ভালো', '১৫০০', 'টাকা', 'প্রইজে', 'ভালো', 'ব্যাগ', 'পেয়েছি', 'লোকাল', 'শপে', 'সেইম', 'ব্যাগ', 'দিগুন', 'দাম']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট ভালো ছিল , ঝাল ছিল অনেক
    Afert Tokenizing:  ['প্রোডাক্ট', 'ভালো', 'ছিল', '', ',', 'ঝাল', 'ছিল', 'অনেক']
    Truncating punctuation: ['প্রোডাক্ট', 'ভালো', 'ছিল', '', 'ঝাল', 'ছিল', 'অনেক']
    Truncating StopWords: ['প্রোডাক্ট', 'ভালো', '', 'ঝাল']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট এর অবস্থা খুবই খারাপ। ইদুরের কাটা জিনিস দিয়ে দিছে। প‍্যাকেট সস সব খারাপ অবস্থা।
    Afert Tokenizing:  ['প্রোডাক্ট', 'এর', 'অবস্থা', 'খুবই', 'খারাপ', '।', 'ইদুরের', 'কাটা', 'জিনিস', 'দিয়ে', 'দিছে', '।', 'প\u200d্যাকেট', 'সস', 'সব', 'খারাপ', 'অবস্থা', '।']
    Truncating punctuation: ['প্রোডাক্ট', 'এর', 'অবস্থা', 'খুবই', 'খারাপ', 'ইদুরের', 'কাটা', 'জিনিস', 'দিয়ে', 'দিছে', 'প\u200d্যাকেট', 'সস', 'সব', 'খারাপ', 'অবস্থা']
    Truncating StopWords: ['প্রোডাক্ট', 'অবস্থা', 'খুবই', 'খারাপ', 'ইদুরের', 'কাটা', 'জিনিস', 'দিয়ে', 'দিছে', 'প\u200d্যাকেট', 'সস', 'খারাপ', 'অবস্থা']
    ***************************************************************************************
    Label:  1
    Sentence:  সর্বোত্তম মূল্যে ভাল মানের
    Afert Tokenizing:  ['সর্বোত্তম', 'মূল্যে', 'ভাল', 'মানের']
    Truncating punctuation: ['সর্বোত্তম', 'মূল্যে', 'ভাল', 'মানের']
    Truncating StopWords: ['সর্বোত্তম', 'মূল্যে', 'ভাল', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ই প্রয়োজনীয় প্রডাক্ট...সেলার কে ধন্যবাদ...ভাউচার দিয়ে মাত্র ২৪ টাকায় কিনেছি....আপনারা ও কিনতে পারেন
    Afert Tokenizing:  ['খুব', 'ই', 'প্রয়োজনীয়', 'প্রডাক্ট...সেলার', 'কে', 'ধন্যবাদ...ভাউচার', 'দিয়ে', 'মাত্র', '২৪', 'টাকায়', 'কিনেছি....আপনারা', 'ও', 'কিনতে', 'পারেন']
    Truncating punctuation: ['খুব', 'ই', 'প্রয়োজনীয়', 'প্রডাক্ট...সেলার', 'কে', 'ধন্যবাদ...ভাউচার', 'দিয়ে', 'মাত্র', '২৪', 'টাকায়', 'কিনেছি....আপনারা', 'ও', 'কিনতে', 'পারেন']
    Truncating StopWords: ['প্রয়োজনীয়', 'প্রডাক্ট...সেলার', 'ধন্যবাদ...ভাউচার', 'দিয়ে', '২৪', 'টাকায়', 'কিনেছি....আপনারা', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবব ই কাজার জিনিশ। কোয়ালিটিও খুব ভালো।
    Afert Tokenizing:  ['খুবব', 'ই', 'কাজার', 'জিনিশ', '।', 'কোয়ালিটিও', 'খুব', 'ভালো', '।']
    Truncating punctuation: ['খুবব', 'ই', 'কাজার', 'জিনিশ', 'কোয়ালিটিও', 'খুব', 'ভালো']
    Truncating StopWords: ['খুবব', 'কাজার', 'জিনিশ', 'কোয়ালিটিও', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো প্রডাক্ট .ব্যশ কম দামেই পেয়েছি  আবারো কিনবো ইনশাআল্লাহ
    Afert Tokenizing:  ['অনেক', 'ভালো', 'প্রডাক্ট', 'ব্যশ', '.', 'কম', 'দামেই', 'পেয়েছি', 'আবারো', 'কিনবো', 'ইনশাআল্লাহ']
    Truncating punctuation: ['অনেক', 'ভালো', 'প্রডাক্ট', 'ব্যশ', 'কম', 'দামেই', 'পেয়েছি', 'আবারো', 'কিনবো', 'ইনশাআল্লাহ']
    Truncating StopWords: ['ভালো', 'প্রডাক্ট', 'ব্যশ', 'কম', 'দামেই', 'পেয়েছি', 'আবারো', 'কিনবো', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ অক্ষত অবস্থায় পেয়েছি। এর আগেও নিয়েছিলাম আবারও নিলাম।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'অক্ষত', 'অবস্থায়', 'পেয়েছি', '।', 'এর', 'আগেও', 'নিয়েছিলাম', 'আবারও', 'নিলাম', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'অক্ষত', 'অবস্থায়', 'পেয়েছি', 'এর', 'আগেও', 'নিয়েছিলাম', 'আবারও', 'নিলাম']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'অক্ষত', 'অবস্থায়', 'পেয়েছি', 'আগেও', 'নিয়েছিলাম', 'আবারও', 'নিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্যের মানও ভালো। তবে আশা করছি দাম কমে পাবো।
    Afert Tokenizing:  ['পণ্যের', 'মানও', 'ভালো', '।', 'তবে', 'আশা', 'করছি', 'দাম', 'কমে', 'পাবো', '।']
    Truncating punctuation: ['পণ্যের', 'মানও', 'ভালো', 'তবে', 'আশা', 'করছি', 'দাম', 'কমে', 'পাবো']
    Truncating StopWords: ['পণ্যের', 'মানও', 'ভালো', 'আশা', 'করছি', 'দাম', 'কমে', 'পাবো']
    ***************************************************************************************
    Label:  1
    Sentence:  সব ঠিক ভাবে পেয়েছি।সাথে gift ও পেয়েছি
    Afert Tokenizing:  ['সব', 'ঠিক', 'ভাবে', 'পেয়েছি।সাথে', 'gift', 'ও', 'পেয়েছি']
    Truncating punctuation: ['সব', 'ঠিক', 'ভাবে', 'পেয়েছি।সাথে', 'gift', 'ও', 'পেয়েছি']
    Truncating StopWords: ['ঠিক', 'পেয়েছি।সাথে', 'gift', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  বেল্ট এর মান টা দাম হিসেবে খুবি ভালো    পাঁচ দিনের মধ্যে ডেলিভারি পেয়েছি
    Afert Tokenizing:  ['বেল্ট', 'এর', 'মান', 'টা', 'দাম', 'হিসেবে', 'খুবি', 'ভালো', 'পাঁচ', 'দিনের', 'মধ্যে', 'ডেলিভারি', 'পেয়েছি']
    Truncating punctuation: ['বেল্ট', 'এর', 'মান', 'টা', 'দাম', 'হিসেবে', 'খুবি', 'ভালো', 'পাঁচ', 'দিনের', 'মধ্যে', 'ডেলিভারি', 'পেয়েছি']
    Truncating StopWords: ['বেল্ট', 'মান', 'টা', 'দাম', 'হিসেবে', 'খুবি', 'ভালো', 'পাঁচ', 'দিনের', 'ডেলিভারি', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো জিনিছ যেমন চেয়েচিলাম তএমন পেয়েছি
    Afert Tokenizing:  ['খুব', 'ভালো', 'জিনিছ', 'যেমন', 'চেয়েচিলাম', 'তএমন', 'পেয়েছি']
    Truncating punctuation: ['খুব', 'ভালো', 'জিনিছ', 'যেমন', 'চেয়েচিলাম', 'তএমন', 'পেয়েছি']
    Truncating StopWords: ['ভালো', 'জিনিছ', 'চেয়েচিলাম', 'তএমন', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম অনুযায়ী প্রডাক্ট টা অনেক ভালো ছিল। সেলার ভাইয়ের কাছে যেটা চাইছি সেটাই দিয়েছেন। সাইজ, কালার, সেলাই সবকিছু ঠিক ছিল। সেলার ভাইকে অনেক অনেক ধন্যবাদ!
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'প্রডাক্ট', 'টা', 'অনেক', 'ভালো', 'ছিল', '।', 'সেলার', 'ভাইয়ের', 'কাছে', 'যেটা', 'চাইছি', 'সেটাই', 'দিয়েছেন', '।', 'সাইজ', ',', 'কালার', ',', 'সেলাই', 'সবকিছু', 'ঠিক', 'ছিল', '।', 'সেলার', 'ভাইকে', 'অনেক', 'অনেক', 'ধন্যবাদ', '!']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'প্রডাক্ট', 'টা', 'অনেক', 'ভালো', 'ছিল', 'সেলার', 'ভাইয়ের', 'কাছে', 'যেটা', 'চাইছি', 'সেটাই', 'দিয়েছেন', 'সাইজ', 'কালার', 'সেলাই', 'সবকিছু', 'ঠিক', 'ছিল', 'সেলার', 'ভাইকে', 'অনেক', 'অনেক', 'ধন্যবাদ']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'প্রডাক্ট', 'টা', 'ভালো', 'সেলার', 'ভাইয়ের', 'যেটা', 'চাইছি', 'দিয়েছেন', 'সাইজ', 'কালার', 'সেলাই', 'সবকিছু', 'ঠিক', 'সেলার', 'ভাইকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ, প্রোডাক্ট কোয়ালিটি অনেক ভালো ছিলো, সেলাই ভালো ছিলো, এক কথায় এই দামে প্রোডাক্ট পেয়ে আমি পুরোপুরি সন্তুষ্ট।
    Afert Tokenizing:  ['ধন্যবাদ', ',', 'প্রোডাক্ট', 'কোয়ালিটি', 'অনেক', 'ভালো', 'ছিলো', ',', 'সেলাই', 'ভালো', 'ছিলো', ',', 'এক', 'কথায়', 'এই', 'দামে', 'প্রোডাক্ট', 'পেয়ে', 'আমি', 'পুরোপুরি', 'সন্তুষ্ট', '।']
    Truncating punctuation: ['ধন্যবাদ', 'প্রোডাক্ট', 'কোয়ালিটি', 'অনেক', 'ভালো', 'ছিলো', 'সেলাই', 'ভালো', 'ছিলো', 'এক', 'কথায়', 'এই', 'দামে', 'প্রোডাক্ট', 'পেয়ে', 'আমি', 'পুরোপুরি', 'সন্তুষ্ট']
    Truncating StopWords: ['ধন্যবাদ', 'প্রোডাক্ট', 'কোয়ালিটি', 'ভালো', 'ছিলো', 'সেলাই', 'ভালো', 'ছিলো', 'এক', 'কথায়', 'দামে', 'প্রোডাক্ট', 'পেয়ে', 'পুরোপুরি', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড় - জার্সির কাপড় যেমন টি description এ দেওয়া আছে তেমন টি পেয়েছি । সেলার অনেস্ট ।
    Afert Tokenizing:  ['কাপড়', '-', 'জার্সির', 'কাপড়', 'যেমন', 'টি', 'description', 'এ', 'দেওয়া', 'আছে', 'তেমন', 'টি', 'পেয়েছি', '', '।', 'সেলার', 'অনেস্ট', '', '।']
    Truncating punctuation: ['কাপড়', 'জার্সির', 'কাপড়', 'যেমন', 'টি', 'description', 'এ', 'দেওয়া', 'আছে', 'তেমন', 'টি', 'পেয়েছি', '', 'সেলার', 'অনেস্ট', '']
    Truncating StopWords: ['কাপড়', 'জার্সির', 'কাপড়', 'description', 'পেয়েছি', '', 'সেলার', 'অনেস্ট', '']
    ***************************************************************************************
    Label:  1
    Sentence:  সাইজ পারফেক্ট,  কাপড় টা ও জোস রেকোমেন্ডেড চাইলে নিতে পারেন
    Afert Tokenizing:  ['সাইজ', 'পারফেক্ট', ',', 'কাপড়', 'টা', 'ও', 'জোস', 'রেকোমেন্ডেড', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['সাইজ', 'পারফেক্ট', 'কাপড়', 'টা', 'ও', 'জোস', 'রেকোমেন্ডেড', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['সাইজ', 'পারফেক্ট', 'কাপড়', 'টা', 'জোস', 'রেকোমেন্ডেড', 'চাইলে']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালভাবেই ফিট হয়েছে।ডেলিভারিও পেয়েছি দ্রুত
    Afert Tokenizing:  ['ভালভাবেই', 'ফিট', 'হয়েছে।ডেলিভারিও', 'পেয়েছি', 'দ্রুত']
    Truncating punctuation: ['ভালভাবেই', 'ফিট', 'হয়েছে।ডেলিভারিও', 'পেয়েছি', 'দ্রুত']
    Truncating StopWords: ['ভালভাবেই', 'ফিট', 'হয়েছে।ডেলিভারিও', 'পেয়েছি', 'দ্রুত']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রথমে ধন্যবাদ সেলার কে, কারণ ব্ল্যাক চাইছিলাম ও পাইছি এক কথায়, কম দামে ভালো পণ্য, ব্যাটারি চার্জ ও সাউন্ড কোয়ালিটি ভালো
    Afert Tokenizing:  ['প্রথমে', 'ধন্যবাদ', 'সেলার', 'কে', ',', 'কারণ', 'ব্ল্যাক', 'চাইছিলাম', 'ও', 'পাইছি', 'এক', 'কথায়', ',', 'কম', 'দামে', 'ভালো', 'পণ্য', ',', 'ব্যাটারি', 'চার্জ', 'ও', 'সাউন্ড', 'কোয়ালিটি', 'ভালো']
    Truncating punctuation: ['প্রথমে', 'ধন্যবাদ', 'সেলার', 'কে', 'কারণ', 'ব্ল্যাক', 'চাইছিলাম', 'ও', 'পাইছি', 'এক', 'কথায়', 'কম', 'দামে', 'ভালো', 'পণ্য', 'ব্যাটারি', 'চার্জ', 'ও', 'সাউন্ড', 'কোয়ালিটি', 'ভালো']
    Truncating StopWords: ['প্রথমে', 'ধন্যবাদ', 'সেলার', 'ব্ল্যাক', 'চাইছিলাম', 'পাইছি', 'এক', 'কথায়', 'কম', 'দামে', 'ভালো', 'পণ্য', 'ব্যাটারি', 'চার্জ', 'সাউন্ড', 'কোয়ালিটি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট কোয়ালিটি  অনেক ভালো ছিলো যেমন আশা করছিলাম তেমন পেয়েছি । সাউন্ড কোয়ালিটি আমার কাছে খুবই ভালো লাগছে। এক কথায় 10/10 কোন কুমতি পাইনি
    Afert Tokenizing:  ['প্রডাক্ট', 'কোয়ালিটি', 'অনেক', 'ভালো', 'ছিলো', 'যেমন', 'আশা', 'করছিলাম', 'তেমন', 'পেয়েছি', '', '।', 'সাউন্ড', 'কোয়ালিটি', 'আমার', 'কাছে', 'খুবই', 'ভালো', 'লাগছে', '।', 'এক', 'কথায়', '10/10', 'কোন', 'কুমতি', 'পাইনি']
    Truncating punctuation: ['প্রডাক্ট', 'কোয়ালিটি', 'অনেক', 'ভালো', 'ছিলো', 'যেমন', 'আশা', 'করছিলাম', 'তেমন', 'পেয়েছি', '', 'সাউন্ড', 'কোয়ালিটি', 'আমার', 'কাছে', 'খুবই', 'ভালো', 'লাগছে', 'এক', 'কথায়', '10/10', 'কোন', 'কুমতি', 'পাইনি']
    Truncating StopWords: ['প্রডাক্ট', 'কোয়ালিটি', 'ভালো', 'ছিলো', 'আশা', 'করছিলাম', 'পেয়েছি', '', 'সাউন্ড', 'কোয়ালিটি', 'খুবই', 'ভালো', 'লাগছে', 'এক', 'কথায়', '10/10', 'কুমতি', 'পাইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  সেলারের ব্যবহার আমার কাছে খুবই ভালো লাগছে যে কালার চেয়েছি সেটাই পেয়েছি।
    Afert Tokenizing:  ['সেলারের', 'ব্যবহার', 'আমার', 'কাছে', 'খুবই', 'ভালো', 'লাগছে', 'যে', 'কালার', 'চেয়েছি', 'সেটাই', 'পেয়েছি', '।']
    Truncating punctuation: ['সেলারের', 'ব্যবহার', 'আমার', 'কাছে', 'খুবই', 'ভালো', 'লাগছে', 'যে', 'কালার', 'চেয়েছি', 'সেটাই', 'পেয়েছি']
    Truncating StopWords: ['সেলারের', 'খুবই', 'ভালো', 'লাগছে', 'কালার', 'চেয়েছি', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ একটি লাইট,আমি এক টানা ৪০ মিনিট জালিয়ে রেখেছি আরো রাখা যেত ইচ্ছে করে জালাই নি। আমার খুব ই পছন্দ হয়েছে
    Afert Tokenizing:  ['অসাধারণ', 'একটি', 'লাইট,আমি', 'এক', 'টানা', '৪০', 'মিনিট', 'জালিয়ে', 'রেখেছি', 'আরো', 'রাখা', 'যেত', 'ইচ্ছে', 'করে', 'জালাই', 'নি', '।', 'আমার', 'খুব', 'ই', 'পছন্দ', 'হয়েছে']
    Truncating punctuation: ['অসাধারণ', 'একটি', 'লাইট,আমি', 'এক', 'টানা', '৪০', 'মিনিট', 'জালিয়ে', 'রেখেছি', 'আরো', 'রাখা', 'যেত', 'ইচ্ছে', 'করে', 'জালাই', 'নি', 'আমার', 'খুব', 'ই', 'পছন্দ', 'হয়েছে']
    Truncating StopWords: ['অসাধারণ', 'লাইট,আমি', 'এক', 'টানা', '৪০', 'মিনিট', 'জালিয়ে', 'রেখেছি', 'আরো', 'যেত', 'ইচ্ছে', 'জালাই', 'নি', 'পছন্দ', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  লাইটা ছোট হলে দেখতে সুন্দর। দাম ভালো।আলো অনেক বেশী চাজ ভালোই থাকে।
    Afert Tokenizing:  ['লাইটা', 'ছোট', 'হলে', 'দেখতে', 'সুন্দর', '।', 'দাম', 'ভালো।আলো', 'অনেক', 'বেশী', 'চাজ', 'ভালোই', 'থাকে', '।']
    Truncating punctuation: ['লাইটা', 'ছোট', 'হলে', 'দেখতে', 'সুন্দর', 'দাম', 'ভালো।আলো', 'অনেক', 'বেশী', 'চাজ', 'ভালোই', 'থাকে']
    Truncating StopWords: ['লাইটা', 'ছোট', 'সুন্দর', 'দাম', 'ভালো।আলো', 'বেশী', 'চাজ', 'ভালোই']
    ***************************************************************************************
    Label:  1
    Sentence:  চাইলে নিতে পারেন অনেক ভালো একটা পণ্য কম দামে
    Afert Tokenizing:  ['চাইলে', 'নিতে', 'পারেন', 'অনেক', 'ভালো', 'একটা', 'পণ্য', 'কম', 'দামে']
    Truncating punctuation: ['চাইলে', 'নিতে', 'পারেন', 'অনেক', 'ভালো', 'একটা', 'পণ্য', 'কম', 'দামে']
    Truncating StopWords: ['চাইলে', 'ভালো', 'একটা', 'পণ্য', 'কম', 'দামে']
    ***************************************************************************************
    Label:  0
    Sentence:  যেমনটা মনে করছিলাম তেমন না। মিস ফায়ার হয়
    Afert Tokenizing:  ['যেমনটা', 'মনে', 'করছিলাম', 'তেমন', 'না', '।', 'মিস', 'ফায়ার', 'হয়']
    Truncating punctuation: ['যেমনটা', 'মনে', 'করছিলাম', 'তেমন', 'না', 'মিস', 'ফায়ার', 'হয়']
    Truncating StopWords: ['যেমনটা', 'করছিলাম', 'না', 'মিস', 'ফায়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  বিজ্ঞাপনের মতো একই
    Afert Tokenizing:  ['বিজ্ঞাপনের', 'মতো', 'একই']
    Truncating punctuation: ['বিজ্ঞাপনের', 'মতো', 'একই']
    Truncating StopWords: ['বিজ্ঞাপনের']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল পণ্য সহজ ব্যবহার।
    Afert Tokenizing:  ['ভাল', 'পণ্য', 'সহজ', 'ব্যবহার', '।']
    Truncating punctuation: ['ভাল', 'পণ্য', 'সহজ', 'ব্যবহার']
    Truncating StopWords: ['ভাল', 'পণ্য', 'সহজ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুল্লিহা ভালো product পেয়েছি।সেলার service ভালো।তারা খুব ভালো chat response করে।T-Shirt টা খুব পছন্দ হয়েছে।Fabric quality যথেষ্ট ভালো।
    Afert Tokenizing:  ['আলহামদুল্লিহা', 'ভালো', 'product', 'পেয়েছি।সেলার', 'service', 'ভালো।তারা', 'খুব', 'ভালো', 'chat', 'response', 'করে।T-Shirt', 'টা', 'খুব', 'পছন্দ', 'হয়েছে।Fabric', 'quality', 'যথেষ্ট', 'ভালো', '।']
    Truncating punctuation: ['আলহামদুল্লিহা', 'ভালো', 'product', 'পেয়েছি।সেলার', 'service', 'ভালো।তারা', 'খুব', 'ভালো', 'chat', 'response', 'করে।T-Shirt', 'টা', 'খুব', 'পছন্দ', 'হয়েছে।Fabric', 'quality', 'যথেষ্ট', 'ভালো']
    Truncating StopWords: ['আলহামদুল্লিহা', 'ভালো', 'product', 'পেয়েছি।সেলার', 'service', 'ভালো।তারা', 'ভালো', 'chat', 'response', 'করে।T-Shirt', 'টা', 'পছন্দ', 'হয়েছে।Fabric', 'quality', 'যথেষ্ট', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেকদিন ব্যবহার করার পর রিভিউ দিলাম।T-shirt টি অনেক সুন্দর।খুব আরামদায়ক।কাপড়ের কোয়ালিটি ভালো।Showroom এর তুলনায় অনেক কম দামে T-shirt পেয়েছি।
    Afert Tokenizing:  ['অনেকদিন', 'ব্যবহার', 'করার', 'পর', 'রিভিউ', 'দিলাম।T-shirt', 'টি', 'অনেক', 'সুন্দর।খুব', 'আরামদায়ক।কাপড়ের', 'কোয়ালিটি', 'ভালো।Showroom', 'এর', 'তুলনায়', 'অনেক', 'কম', 'দামে', 'T-shirt', 'পেয়েছি', '।']
    Truncating punctuation: ['অনেকদিন', 'ব্যবহার', 'করার', 'পর', 'রিভিউ', 'দিলাম।T-shirt', 'টি', 'অনেক', 'সুন্দর।খুব', 'আরামদায়ক।কাপড়ের', 'কোয়ালিটি', 'ভালো।Showroom', 'এর', 'তুলনায়', 'অনেক', 'কম', 'দামে', 'T-shirt', 'পেয়েছি']
    Truncating StopWords: ['অনেকদিন', 'রিভিউ', 'দিলাম।T-shirt', 'সুন্দর।খুব', 'আরামদায়ক।কাপড়ের', 'কোয়ালিটি', 'ভালো।Showroom', 'তুলনায়', 'কম', 'দামে', 'T-shirt', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  দ্রুত সময়ে ডেলিভারি পেলাম . মূলত লোটো এবং kub আরামদায়ক
    Afert Tokenizing:  ['দ্রুত', 'সময়ে', 'ডেলিভারি', 'পেলাম', '', '.', 'মূলত', 'লোটো', 'এবং', 'kub', 'আরামদায়ক']
    Truncating punctuation: ['দ্রুত', 'সময়ে', 'ডেলিভারি', 'পেলাম', '', 'মূলত', 'লোটো', 'এবং', 'kub', 'আরামদায়ক']
    Truncating StopWords: ['দ্রুত', 'সময়ে', 'ডেলিভারি', 'পেলাম', '', 'মূলত', 'লোটো', 'kub', 'আরামদায়ক']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব সাবধানে ওডার দিবেন , ওডার দিছি xl তাহারা আমারে দিছে S সাইজের টি শার্ট
    Afert Tokenizing:  ['খুব', 'সাবধানে', 'ওডার', 'দিবেন', '', ',', 'ওডার', 'দিছি', 'xl', 'তাহারা', 'আমারে', 'দিছে', 'S', 'সাইজের', 'টি', 'শার্ট']
    Truncating punctuation: ['খুব', 'সাবধানে', 'ওডার', 'দিবেন', '', 'ওডার', 'দিছি', 'xl', 'তাহারা', 'আমারে', 'দিছে', 'S', 'সাইজের', 'টি', 'শার্ট']
    Truncating StopWords: ['সাবধানে', 'ওডার', 'দিবেন', '', 'ওডার', 'দিছি', 'xl', 'তাহারা', 'আমারে', 'দিছে', 'S', 'সাইজের', 'শার্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  নির্দিষ্ট সময়ের ভিতরেই সঠিক এবং গুনগত মানসম্পন্ন পন্যটি দেয়ার জন্য সেলার,এবং রাইডারকে ধন্যবাদ।
    Afert Tokenizing:  ['নির্দিষ্ট', 'সময়ের', 'ভিতরেই', 'সঠিক', 'এবং', 'গুনগত', 'মানসম্পন্ন', 'পন্যটি', 'দেয়ার', 'জন্য', 'সেলার,এবং', 'রাইডারকে', 'ধন্যবাদ', '।']
    Truncating punctuation: ['নির্দিষ্ট', 'সময়ের', 'ভিতরেই', 'সঠিক', 'এবং', 'গুনগত', 'মানসম্পন্ন', 'পন্যটি', 'দেয়ার', 'জন্য', 'সেলার,এবং', 'রাইডারকে', 'ধন্যবাদ']
    Truncating StopWords: ['নির্দিষ্ট', 'সময়ের', 'ভিতরেই', 'সঠিক', 'গুনগত', 'মানসম্পন্ন', 'পন্যটি', 'দেয়ার', 'সেলার,এবং', 'রাইডারকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ  একটা ঘড়ি  , পানির দামে শরবত পাইলাম তাও আবার সাথে ফ্রি চুলের বেন্ড ।  ধন্যবাদ সেলার
    Afert Tokenizing:  ['অসাধারণ', 'একটা', 'ঘড়ি', '', ',', 'পানির', 'দামে', 'শরবত', 'পাইলাম', 'তাও', 'আবার', 'সাথে', 'ফ্রি', 'চুলের', 'বেন্ড', '', '।', 'ধন্যবাদ', 'সেলার']
    Truncating punctuation: ['অসাধারণ', 'একটা', 'ঘড়ি', '', 'পানির', 'দামে', 'শরবত', 'পাইলাম', 'তাও', 'আবার', 'সাথে', 'ফ্রি', 'চুলের', 'বেন্ড', '', 'ধন্যবাদ', 'সেলার']
    Truncating StopWords: ['অসাধারণ', 'একটা', 'ঘড়ি', '', 'পানির', 'দামে', 'শরবত', 'পাইলাম', 'সাথে', 'ফ্রি', 'চুলের', 'বেন্ড', '', 'ধন্যবাদ', 'সেলার']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার কাছে একদম জোস,চাইলে কিনথে পারেন
    Afert Tokenizing:  ['আমার', 'কাছে', 'একদম', 'জোস,চাইলে', 'কিনথে', 'পারেন']
    Truncating punctuation: ['আমার', 'কাছে', 'একদম', 'জোস,চাইলে', 'কিনথে', 'পারেন']
    Truncating StopWords: ['একদম', 'জোস,চাইলে', 'কিনথে']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ যেটা অডা্র করেছিলাম সেটাই পেয়েছি। চাইলে আপনারা নিতে পারেন
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'যেটা', 'অডা্র', 'করেছিলাম', 'সেটাই', 'পেয়েছি', '।', 'চাইলে', 'আপনারা', 'নিতে', 'পারেন']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'যেটা', 'অডা্র', 'করেছিলাম', 'সেটাই', 'পেয়েছি', 'চাইলে', 'আপনারা', 'নিতে', 'পারেন']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'যেটা', 'অডা্র', 'করেছিলাম', 'পেয়েছি', 'চাইলে', 'আপনারা']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ টেবিলটা অনেক সুন্দর।  আমার ছেলে এটা পেয়ে অনেকটাই খুশি। সবাই নিতে পারেন।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'টেবিলটা', 'অনেক', 'সুন্দর', '।', 'আমার', 'ছেলে', 'এটা', 'পেয়ে', 'অনেকটাই', 'খুশি', '।', 'সবাই', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'টেবিলটা', 'অনেক', 'সুন্দর', 'আমার', 'ছেলে', 'এটা', 'পেয়ে', 'অনেকটাই', 'খুশি', 'সবাই', 'নিতে', 'পারেন']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'টেবিলটা', 'সুন্দর', 'ছেলে', 'পেয়ে', 'অনেকটাই', 'খুশি', 'সবাই']
    ***************************************************************************************
    Label:  1
    Sentence:  আমরা বাচ্চাটা লেখা পড়ার প্রতি আগ্রহ পাইছে এই টেবিল টার জন্য। ধন্যবাদ সেলারকে সময় মত ভালো পন্যটা দেয়ার জন্য।
    Afert Tokenizing:  ['আমরা', 'বাচ্চাটা', 'লেখা', 'পড়ার', 'প্রতি', 'আগ্রহ', 'পাইছে', 'এই', 'টেবিল', 'টার', 'জন্য', '।', 'ধন্যবাদ', 'সেলারকে', 'সময়', 'মত', 'ভালো', 'পন্যটা', 'দেয়ার', 'জন্য', '।']
    Truncating punctuation: ['আমরা', 'বাচ্চাটা', 'লেখা', 'পড়ার', 'প্রতি', 'আগ্রহ', 'পাইছে', 'এই', 'টেবিল', 'টার', 'জন্য', 'ধন্যবাদ', 'সেলারকে', 'সময়', 'মত', 'ভালো', 'পন্যটা', 'দেয়ার', 'জন্য']
    Truncating StopWords: ['বাচ্চাটা', 'লেখা', 'পড়ার', 'আগ্রহ', 'পাইছে', 'টেবিল', 'টার', 'ধন্যবাদ', 'সেলারকে', 'সময়', 'মত', 'ভালো', 'পন্যটা', 'দেয়ার']
    ***************************************************************************************
    Label:  1
    Sentence:  এই স্মার্ট ওয়াচ টি অসাধরণ পারফম কর‌ছে। মাই‌ক্রোফ‌নের সাউন্ড কোয়া‌লি‌টি খুব ভা‌লো। চোখ বন্ধ ক‌রে নিতে পা‌রেন।
    Afert Tokenizing:  ['এই', 'স্মার্ট', 'ওয়াচ', 'টি', 'অসাধরণ', 'পারফম', 'কর\u200cছে', '।', 'মাই\u200cক্রোফ\u200cনের', 'সাউন্ড', 'কোয়া\u200cলি\u200cটি', 'খুব', 'ভা\u200cলো', '।', 'চোখ', 'বন্ধ', 'ক\u200cরে', 'নিতে', 'পা\u200cরেন', '।']
    Truncating punctuation: ['এই', 'স্মার্ট', 'ওয়াচ', 'টি', 'অসাধরণ', 'পারফম', 'কর\u200cছে', 'মাই\u200cক্রোফ\u200cনের', 'সাউন্ড', 'কোয়া\u200cলি\u200cটি', 'খুব', 'ভা\u200cলো', 'চোখ', 'বন্ধ', 'ক\u200cরে', 'নিতে', 'পা\u200cরেন']
    Truncating StopWords: ['স্মার্ট', 'ওয়াচ', 'অসাধরণ', 'পারফম', 'কর\u200cছে', 'মাই\u200cক্রোফ\u200cনের', 'সাউন্ড', 'কোয়া\u200cলি\u200cটি', 'ভা\u200cলো', 'চোখ', 'বন্ধ', 'ক\u200cরে', 'পা\u200cরেন']
    ***************************************************************************************
    Label:  1
    Sentence:  পন্যটি ভালো ছিলো যেমন টা চেয়েছি তেমন টাই পেয়েছি। কিন্তু ইস্কিনপটেক্টর টা দিয়ে দিলে ভালো হত
    Afert Tokenizing:  ['পন্যটি', 'ভালো', 'ছিলো', 'যেমন', 'টা', 'চেয়েছি', 'তেমন', 'টাই', 'পেয়েছি', '।', 'কিন্তু', 'ইস্কিনপটেক্টর', 'টা', 'দিয়ে', 'দিলে', 'ভালো', 'হত']
    Truncating punctuation: ['পন্যটি', 'ভালো', 'ছিলো', 'যেমন', 'টা', 'চেয়েছি', 'তেমন', 'টাই', 'পেয়েছি', 'কিন্তু', 'ইস্কিনপটেক্টর', 'টা', 'দিয়ে', 'দিলে', 'ভালো', 'হত']
    Truncating StopWords: ['পন্যটি', 'ভালো', 'ছিলো', 'টা', 'চেয়েছি', 'টাই', 'পেয়েছি', 'ইস্কিনপটেক্টর', 'টা', 'দিয়ে', 'দিলে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ অনেক ভালো প্রোডাক্ট । দেড় দিন চালানোর পর রিভিউ দিচ্ছি । ব্যাটারি ব্যাকআপ ও অনেক ভালো । সবাই চাইলে নিতে পারেন
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'অনেক', 'ভালো', 'প্রোডাক্ট', '', '।', 'দেড়', 'দিন', 'চালানোর', 'পর', 'রিভিউ', 'দিচ্ছি', '', '।', 'ব্যাটারি', 'ব্যাকআপ', 'ও', 'অনেক', 'ভালো', '', '।', 'সবাই', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'অনেক', 'ভালো', 'প্রোডাক্ট', '', 'দেড়', 'দিন', 'চালানোর', 'পর', 'রিভিউ', 'দিচ্ছি', '', 'ব্যাটারি', 'ব্যাকআপ', 'ও', 'অনেক', 'ভালো', '', 'সবাই', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ভালো', 'প্রোডাক্ট', '', 'দেড়', 'চালানোর', 'রিভিউ', 'দিচ্ছি', '', 'ব্যাটারি', 'ব্যাকআপ', 'ভালো', '', 'সবাই', 'চাইলে']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই উপকারী স্মার্ট ওয়াচ...। কিন্তু আমি তো এর সাথে কিছু সমস্যার মুখোমুখি হয়েছি
    Afert Tokenizing:  ['খুবই', 'উপকারী', 'স্মার্ট', 'ওয়াচ...', '।', 'কিন্তু', 'আমি', 'তো', 'এর', 'সাথে', 'কিছু', 'সমস্যার', 'মুখোমুখি', 'হয়েছি']
    Truncating punctuation: ['খুবই', 'উপকারী', 'স্মার্ট', 'ওয়াচ...', 'কিন্তু', 'আমি', 'তো', 'এর', 'সাথে', 'কিছু', 'সমস্যার', 'মুখোমুখি', 'হয়েছি']
    Truncating StopWords: ['খুবই', 'উপকারী', 'স্মার্ট', 'ওয়াচ...', 'সাথে', 'সমস্যার', 'মুখোমুখি', 'হয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ, পাঁচ টা অর্ডার করেছিলাম, পাঁচ টাই হাতে পেয়েছি, সে জন্য খুব ভালো লাগছে, আর আমি কি বলবো, সত্যিই খুব অসাধারণ একটি প্রডাক্ট, আমি কোথাও কোন সমস্যা পাইনি
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', ',', 'পাঁচ', 'টা', 'অর্ডার', 'করেছিলাম', ',', 'পাঁচ', 'টাই', 'হাতে', 'পেয়েছি', ',', 'সে', 'জন্য', 'খুব', 'ভালো', 'লাগছে', ',', 'আর', 'আমি', 'কি', 'বলবো', ',', 'সত্যিই', 'খুব', 'অসাধারণ', 'একটি', 'প্রডাক্ট', ',', 'আমি', 'কোথাও', 'কোন', 'সমস্যা', 'পাইনি']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'পাঁচ', 'টা', 'অর্ডার', 'করেছিলাম', 'পাঁচ', 'টাই', 'হাতে', 'পেয়েছি', 'সে', 'জন্য', 'খুব', 'ভালো', 'লাগছে', 'আর', 'আমি', 'কি', 'বলবো', 'সত্যিই', 'খুব', 'অসাধারণ', 'একটি', 'প্রডাক্ট', 'আমি', 'কোথাও', 'কোন', 'সমস্যা', 'পাইনি']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'পাঁচ', 'টা', 'অর্ডার', 'করেছিলাম', 'পাঁচ', 'টাই', 'হাতে', 'পেয়েছি', 'ভালো', 'লাগছে', 'বলবো', 'সত্যিই', 'অসাধারণ', 'প্রডাক্ট', 'কোথাও', 'সমস্যা', 'পাইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই দারুণ আমি ভাবি নাই যে এত ভালো হবে প্রচন্ড ভালো হয়েছে আমি যে দুটি জামা নিয়েছি দুটি ভালো এসেছে আমি তাতে প্রচন্ড খুশি আপনারা চাইলে নিতে পারেন ধন্যবাদ দারাজ
    Afert Tokenizing:  ['সত্যিই', 'দারুণ', 'আমি', 'ভাবি', 'নাই', 'যে', 'এত', 'ভালো', 'হবে', 'প্রচন্ড', 'ভালো', 'হয়েছে', 'আমি', 'যে', 'দুটি', 'জামা', 'নিয়েছি', 'দুটি', 'ভালো', 'এসেছে', 'আমি', 'তাতে', 'প্রচন্ড', 'খুশি', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন', 'ধন্যবাদ', 'দারাজ']
    Truncating punctuation: ['সত্যিই', 'দারুণ', 'আমি', 'ভাবি', 'নাই', 'যে', 'এত', 'ভালো', 'হবে', 'প্রচন্ড', 'ভালো', 'হয়েছে', 'আমি', 'যে', 'দুটি', 'জামা', 'নিয়েছি', 'দুটি', 'ভালো', 'এসেছে', 'আমি', 'তাতে', 'প্রচন্ড', 'খুশি', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন', 'ধন্যবাদ', 'দারাজ']
    Truncating StopWords: ['সত্যিই', 'দারুণ', 'ভাবি', 'নাই', 'ভালো', 'প্রচন্ড', 'ভালো', 'জামা', 'নিয়েছি', 'ভালো', 'এসেছে', 'প্রচন্ড', 'খুশি', 'আপনারা', 'চাইলে', 'ধন্যবাদ', 'দারাজ']
    ***************************************************************************************
    Label:  1
    Sentence:  জামাটা ছবিতে যেরকম দেখছি তার থেকে অনেক সুন্দর । কাপড়ে মান অনেক ভালো।  এতো ভালো  মানের পোশাক  দেওয়ার জন্য অসংখ্য ধন্যবাদ
    Afert Tokenizing:  ['জামাটা', 'ছবিতে', 'যেরকম', 'দেখছি', 'তার', 'থেকে', 'অনেক', 'সুন্দর', '', '।', 'কাপড়ে', 'মান', 'অনেক', 'ভালো', '।', 'এতো', 'ভালো', 'মানের', 'পোশাক', 'দেওয়ার', 'জন্য', 'অসংখ্য', 'ধন্যবাদ']
    Truncating punctuation: ['জামাটা', 'ছবিতে', 'যেরকম', 'দেখছি', 'তার', 'থেকে', 'অনেক', 'সুন্দর', '', 'কাপড়ে', 'মান', 'অনেক', 'ভালো', 'এতো', 'ভালো', 'মানের', 'পোশাক', 'দেওয়ার', 'জন্য', 'অসংখ্য', 'ধন্যবাদ']
    Truncating StopWords: ['জামাটা', 'ছবিতে', 'যেরকম', 'দেখছি', 'সুন্দর', '', 'কাপড়ে', 'মান', 'ভালো', 'এতো', 'ভালো', 'মানের', 'পোশাক', 'দেওয়ার', 'অসংখ্য', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  যেমনটা দেখেছিলাম তেমনটাই মোটামুটি ভালো আছে
    Afert Tokenizing:  ['যেমনটা', 'দেখেছিলাম', 'তেমনটাই', 'মোটামুটি', 'ভালো', 'আছে']
    Truncating punctuation: ['যেমনটা', 'দেখেছিলাম', 'তেমনটাই', 'মোটামুটি', 'ভালো', 'আছে']
    Truncating StopWords: ['যেমনটা', 'দেখেছিলাম', 'তেমনটাই', 'মোটামুটি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম অনুযায়ী ঠিকঠাক ই আছে কিন্তু এরিপ্রক্ট এর কালার চেঞ্জ হয়ে গেছে! তবে ভালোই!!
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'ঠিকঠাক', 'ই', 'আছে', 'কিন্তু', 'এরিপ্রক্ট', 'এর', 'কালার', 'চেঞ্জ', 'হয়ে', 'গেছে', '!', 'তবে', 'ভালোই!', '!']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'ঠিকঠাক', 'ই', 'আছে', 'কিন্তু', 'এরিপ্রক্ট', 'এর', 'কালার', 'চেঞ্জ', 'হয়ে', 'গেছে', 'তবে', 'ভালোই!']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'ঠিকঠাক', 'এরিপ্রক্ট', 'কালার', 'চেঞ্জ', 'হয়ে', 'ভালোই!']
    ***************************************************************************************
    Label:  1
    Sentence:  খারাপ না,আমার কাছে ভালো মনে হয়েছে,অরিজিনাল ছবি দিলাম কেউ নিলে নিতে পারেন।
    Afert Tokenizing:  ['খারাপ', 'না,আমার', 'কাছে', 'ভালো', 'মনে', 'হয়েছে,অরিজিনাল', 'ছবি', 'দিলাম', 'কেউ', 'নিলে', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['খারাপ', 'না,আমার', 'কাছে', 'ভালো', 'মনে', 'হয়েছে,অরিজিনাল', 'ছবি', 'দিলাম', 'কেউ', 'নিলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['খারাপ', 'না,আমার', 'ভালো', 'হয়েছে,অরিজিনাল', 'ছবি', 'দিলাম', 'নিলে']
    ***************************************************************************************
    Label:  1
    Sentence:  এখন ইউস করে ডুরাবিলিটি দেখা লাগবে। ভাইয়াকে অনেক ধন্যবাদ সততার জন্য
    Afert Tokenizing:  ['এখন', 'ইউস', 'করে', 'ডুরাবিলিটি', 'দেখা', 'লাগবে', '।', 'ভাইয়াকে', 'অনেক', 'ধন্যবাদ', 'সততার', 'জন্য']
    Truncating punctuation: ['এখন', 'ইউস', 'করে', 'ডুরাবিলিটি', 'দেখা', 'লাগবে', 'ভাইয়াকে', 'অনেক', 'ধন্যবাদ', 'সততার', 'জন্য']
    Truncating StopWords: ['ইউস', 'ডুরাবিলিটি', 'লাগবে', 'ভাইয়াকে', 'ধন্যবাদ', 'সততার']
    ***************************************************************************************
    Label:  1
    Sentence:  কি বলবো বল পুরাই অস্থির একটা জিনিস, প্যাকেজিং  ঠিকঠাক ছিল অনেক সুন্দর, চাইলে আপনারাও কিনতে পারেন
    Afert Tokenizing:  ['কি', 'বলবো', 'বল', 'পুরাই', 'অস্থির', 'একটা', 'জিনিস', ',', 'প্যাকেজিং', 'ঠিকঠাক', 'ছিল', 'অনেক', 'সুন্দর', ',', 'চাইলে', 'আপনারাও', 'কিনতে', 'পারেন']
    Truncating punctuation: ['কি', 'বলবো', 'বল', 'পুরাই', 'অস্থির', 'একটা', 'জিনিস', 'প্যাকেজিং', 'ঠিকঠাক', 'ছিল', 'অনেক', 'সুন্দর', 'চাইলে', 'আপনারাও', 'কিনতে', 'পারেন']
    Truncating StopWords: ['বলবো', 'বল', 'পুরাই', 'অস্থির', 'একটা', 'জিনিস', 'প্যাকেজিং', 'ঠিকঠাক', 'সুন্দর', 'চাইলে', 'আপনারাও', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট  টা ভালো। শুধু খারাপ লাগছে স্টেন্ড এ রেখে ফোন এ টাচ করলে সেকিং করে।যেটা খুব ই বিরক্তিকর।  এছাড়া আর কোন প্রবলেম ফেইস করি নি
    Afert Tokenizing:  ['প্রোডাক্ট', 'টা', 'ভালো', '।', 'শুধু', 'খারাপ', 'লাগছে', 'স্টেন্ড', 'এ', 'রেখে', 'ফোন', 'এ', 'টাচ', 'করলে', 'সেকিং', 'করে।যেটা', 'খুব', 'ই', 'বিরক্তিকর', '।', 'এছাড়া', 'আর', 'কোন', 'প্রবলেম', 'ফেইস', 'করি', 'নি']
    Truncating punctuation: ['প্রোডাক্ট', 'টা', 'ভালো', 'শুধু', 'খারাপ', 'লাগছে', 'স্টেন্ড', 'এ', 'রেখে', 'ফোন', 'এ', 'টাচ', 'করলে', 'সেকিং', 'করে।যেটা', 'খুব', 'ই', 'বিরক্তিকর', 'এছাড়া', 'আর', 'কোন', 'প্রবলেম', 'ফেইস', 'করি', 'নি']
    Truncating StopWords: ['প্রোডাক্ট', 'টা', 'ভালো', 'শুধু', 'খারাপ', 'লাগছে', 'স্টেন্ড', 'ফোন', 'টাচ', 'সেকিং', 'করে।যেটা', 'বিরক্তিকর', 'এছাড়া', 'প্রবলেম', 'ফেইস', 'নি']
    ***************************************************************************************
    Label:  0
    Sentence:  পণ্যটি আমি ছবিতে যে রকম দেখেছিলাম দেখে মনে হচ্ছিল অনেক বড়।  কিন্তু হাতে পাওয়ার পর দেখলাম এটা খুবই সাধারণ একটি জিনিস দেখলে মনে হয় খুবই কম দামি।
    Afert Tokenizing:  ['পণ্যটি', 'আমি', 'ছবিতে', 'যে', 'রকম', 'দেখেছিলাম', 'দেখে', 'মনে', 'হচ্ছিল', 'অনেক', 'বড়', '।', 'কিন্তু', 'হাতে', 'পাওয়ার', 'পর', 'দেখলাম', 'এটা', 'খুবই', 'সাধারণ', 'একটি', 'জিনিস', 'দেখলে', 'মনে', 'হয়', 'খুবই', 'কম', 'দামি', '।']
    Truncating punctuation: ['পণ্যটি', 'আমি', 'ছবিতে', 'যে', 'রকম', 'দেখেছিলাম', 'দেখে', 'মনে', 'হচ্ছিল', 'অনেক', 'বড়', 'কিন্তু', 'হাতে', 'পাওয়ার', 'পর', 'দেখলাম', 'এটা', 'খুবই', 'সাধারণ', 'একটি', 'জিনিস', 'দেখলে', 'মনে', 'হয়', 'খুবই', 'কম', 'দামি']
    Truncating StopWords: ['পণ্যটি', 'ছবিতে', 'দেখেছিলাম', 'হচ্ছিল', 'বড়', 'হাতে', 'পাওয়ার', 'দেখলাম', 'খুবই', 'জিনিস', 'দেখলে', 'খুবই', 'কম', 'দামি']
    ***************************************************************************************
    Label:  0
    Sentence:  চাইছি একটা পাইছি আরেকটা,পাইছি তাও আবার নষ্ট, এগুলো কেউ নিবেন না
    Afert Tokenizing:  ['চাইছি', 'একটা', 'পাইছি', 'আরেকটা,পাইছি', 'তাও', 'আবার', 'নষ্ট', ',', 'এগুলো', 'কেউ', 'নিবেন', 'না']
    Truncating punctuation: ['চাইছি', 'একটা', 'পাইছি', 'আরেকটা,পাইছি', 'তাও', 'আবার', 'নষ্ট', 'এগুলো', 'কেউ', 'নিবেন', 'না']
    Truncating StopWords: ['চাইছি', 'একটা', 'পাইছি', 'আরেকটা,পাইছি', 'নষ্ট', 'এগুলো', 'নিবেন', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  কম প্রাইজ হিসেবে অনেক ভালোই আছে... কত দিন সার্ভিস দেয় এটাই এখন দেখার বিষয়
    Afert Tokenizing:  ['কম', 'প্রাইজ', 'হিসেবে', 'অনেক', 'ভালোই', 'আছে..', '.', 'কত', 'দিন', 'সার্ভিস', 'দেয়', 'এটাই', 'এখন', 'দেখার', 'বিষয়']
    Truncating punctuation: ['কম', 'প্রাইজ', 'হিসেবে', 'অনেক', 'ভালোই', 'আছে..', 'কত', 'দিন', 'সার্ভিস', 'দেয়', 'এটাই', 'এখন', 'দেখার', 'বিষয়']
    Truncating StopWords: ['কম', 'প্রাইজ', 'হিসেবে', 'ভালোই', 'আছে..', 'সার্ভিস', 'দেয়', 'দেখার', 'বিষয়']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক অনেক সুন্দর। আমার অনেক ভালো লেগেছে। অসাধারণ লেগেছে
    Afert Tokenizing:  ['অনেক', 'অনেক', 'সুন্দর', '।', 'আমার', 'অনেক', 'ভালো', 'লেগেছে', '।', 'অসাধারণ', 'লেগেছে']
    Truncating punctuation: ['অনেক', 'অনেক', 'সুন্দর', 'আমার', 'অনেক', 'ভালো', 'লেগেছে', 'অসাধারণ', 'লেগেছে']
    Truncating StopWords: ['সুন্দর', 'ভালো', 'লেগেছে', 'অসাধারণ', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  মাত্র 2 দিনের মধ্যে এটি পেয়েছি. দেখায় প্রিমিয়াম এবং মোটরটি আপনার প্রতিক্রিয়ার জন্য শক্তিশালী.ধন্যবাদ বিক্রেতা এবং ভাল পণ্যের জন্য এটি মাত্র 2 দিনের মধ্যে পেয়েছে।
    Afert Tokenizing:  ['মাত্র', '2', 'দিনের', 'মধ্যে', 'এটি', 'পেয়েছি', '.', 'দেখায়', 'প্রিমিয়াম', 'এবং', 'মোটরটি', 'আপনার', 'প্রতিক্রিয়ার', 'জন্য', 'শক্তিশালী.ধন্যবাদ', 'বিক্রেতা', 'এবং', 'ভাল', 'পণ্যের', 'জন্য', 'এটি', 'মাত্র', '2', 'দিনের', 'মধ্যে', 'পেয়েছে', '।']
    Truncating punctuation: ['মাত্র', '2', 'দিনের', 'মধ্যে', 'এটি', 'পেয়েছি', 'দেখায়', 'প্রিমিয়াম', 'এবং', 'মোটরটি', 'আপনার', 'প্রতিক্রিয়ার', 'জন্য', 'শক্তিশালী.ধন্যবাদ', 'বিক্রেতা', 'এবং', 'ভাল', 'পণ্যের', 'জন্য', 'এটি', 'মাত্র', '2', 'দিনের', 'মধ্যে', 'পেয়েছে']
    Truncating StopWords: ['2', 'দিনের', 'পেয়েছি', 'দেখায়', 'প্রিমিয়াম', 'মোটরটি', 'প্রতিক্রিয়ার', 'শক্তিশালী.ধন্যবাদ', 'বিক্রেতা', 'ভাল', 'পণ্যের', '2', 'দিনের', 'পেয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  পেন্টের কাপর টা ঠিক আছে,,কিন্তু গেঞ্জির কাপর টা পাতলা,,but অনেক আরামদায়ক,,
    Afert Tokenizing:  ['পেন্টের', 'কাপর', 'টা', 'ঠিক', 'আছে,,কিন্তু', 'গেঞ্জির', 'কাপর', 'টা', 'পাতলা,,but', 'অনেক', 'আরামদায়ক,', ',']
    Truncating punctuation: ['পেন্টের', 'কাপর', 'টা', 'ঠিক', 'আছে,,কিন্তু', 'গেঞ্জির', 'কাপর', 'টা', 'পাতলা,,but', 'অনেক', 'আরামদায়ক,']
    Truncating StopWords: ['পেন্টের', 'কাপর', 'টা', 'ঠিক', 'আছে,,কিন্তু', 'গেঞ্জির', 'কাপর', 'টা', 'পাতলা,,but', 'আরামদায়ক,']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার সাইজের সমস্যা এর চেয়ে ১ সাইজ বড়  দরকার ছিল, তবে সঠিক পরামর্শ পাই নাই, ফেরত ও দিতে পারি নাই।
    Afert Tokenizing:  ['আমার', 'সাইজের', 'সমস্যা', 'এর', 'চেয়ে', '১', 'সাইজ', 'বড়', 'দরকার', 'ছিল', ',', 'তবে', 'সঠিক', 'পরামর্শ', 'পাই', 'নাই', ',', 'ফেরত', 'ও', 'দিতে', 'পারি', 'নাই', '।']
    Truncating punctuation: ['আমার', 'সাইজের', 'সমস্যা', 'এর', 'চেয়ে', '১', 'সাইজ', 'বড়', 'দরকার', 'ছিল', 'তবে', 'সঠিক', 'পরামর্শ', 'পাই', 'নাই', 'ফেরত', 'ও', 'দিতে', 'পারি', 'নাই']
    Truncating StopWords: ['সাইজের', 'সমস্যা', 'চেয়ে', '১', 'সাইজ', 'বড়', 'দরকার', 'সঠিক', 'পরামর্শ', 'পাই', 'নাই', 'ফেরত', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  নেওয়ার জন্য বলবো, এই প্রাইজে এরকম জিনিস পারফেক্ট
    Afert Tokenizing:  ['নেওয়ার', 'জন্য', 'বলবো', ',', 'এই', 'প্রাইজে', 'এরকম', 'জিনিস', 'পারফেক্ট']
    Truncating punctuation: ['নেওয়ার', 'জন্য', 'বলবো', 'এই', 'প্রাইজে', 'এরকম', 'জিনিস', 'পারফেক্ট']
    Truncating StopWords: ['নেওয়ার', 'বলবো', 'প্রাইজে', 'এরকম', 'জিনিস', 'পারফেক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  মোটামুটি ভালো পণ্য ~ 24 টাকায় কিনেছি । আশা করি ভবিষ্যতে আরো কম দামে পাব
    Afert Tokenizing:  ['মোটামুটি', 'ভালো', 'পণ্য', '', '~', '24', 'টাকায়', 'কিনেছি', '', '।', 'আশা', 'করি', 'ভবিষ্যতে', 'আরো', 'কম', 'দামে', 'পাব']
    Truncating punctuation: ['মোটামুটি', 'ভালো', 'পণ্য', '', '24', 'টাকায়', 'কিনেছি', '', 'আশা', 'করি', 'ভবিষ্যতে', 'আরো', 'কম', 'দামে', 'পাব']
    Truncating StopWords: ['মোটামুটি', 'ভালো', 'পণ্য', '', '24', 'টাকায়', 'কিনেছি', '', 'আশা', 'ভবিষ্যতে', 'আরো', 'কম', 'দামে', 'পাব']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারটি খেতে অনেক সুস্বাদু  এবং কোয়ালিটি খুবই ভাল। কেউ খেতে চাইলে অর্ডার করতে পারেন
    Afert Tokenizing:  ['খাবারটি', 'খেতে', 'অনেক', 'সুস্বাদু', 'এবং', 'কোয়ালিটি', 'খুবই', 'ভাল', '।', 'কেউ', 'খেতে', 'চাইলে', 'অর্ডার', 'করতে', 'পারেন']
    Truncating punctuation: ['খাবারটি', 'খেতে', 'অনেক', 'সুস্বাদু', 'এবং', 'কোয়ালিটি', 'খুবই', 'ভাল', 'কেউ', 'খেতে', 'চাইলে', 'অর্ডার', 'করতে', 'পারেন']
    Truncating StopWords: ['খাবারটি', 'খেতে', 'সুস্বাদু', 'কোয়ালিটি', 'খুবই', 'ভাল', 'খেতে', 'চাইলে', 'অর্ডার']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ, সময় মতো হাতে পেয়েছি। এবং প্রোডাক্টি অনেক সুন্দর ছিল। এটি অনেক সুস্বাদু
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', ',', 'সময়', 'মতো', 'হাতে', 'পেয়েছি', '।', 'এবং', 'প্রোডাক্টি', 'অনেক', 'সুন্দর', 'ছিল', '।', 'এটি', 'অনেক', 'সুস্বাদু']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'সময়', 'মতো', 'হাতে', 'পেয়েছি', 'এবং', 'প্রোডাক্টি', 'অনেক', 'সুন্দর', 'ছিল', 'এটি', 'অনেক', 'সুস্বাদু']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'সময়', 'হাতে', 'পেয়েছি', 'প্রোডাক্টি', 'সুন্দর', 'সুস্বাদু']
    ***************************************************************************************
    Label:  1
    Sentence:  জিনিসটা ও অনেক সুন্দর। কিন্তু ভাঙ্গা ছিল ডেলিভারি ম্যান খুব যত্নে নিয়ে আসতে পারেনি
    Afert Tokenizing:  ['জিনিসটা', 'ও', 'অনেক', 'সুন্দর', '।', 'কিন্তু', 'ভাঙ্গা', 'ছিল', 'ডেলিভারি', 'ম্যান', 'খুব', 'যত্নে', 'নিয়ে', 'আসতে', 'পারেনি']
    Truncating punctuation: ['জিনিসটা', 'ও', 'অনেক', 'সুন্দর', 'কিন্তু', 'ভাঙ্গা', 'ছিল', 'ডেলিভারি', 'ম্যান', 'খুব', 'যত্নে', 'নিয়ে', 'আসতে', 'পারেনি']
    Truncating StopWords: ['জিনিসটা', 'সুন্দর', 'ভাঙ্গা', 'ডেলিভারি', 'ম্যান', 'যত্নে', 'আসতে', 'পারেনি']
    ***************************************************************************************
    Label:  1
    Sentence:  রাধুনি মানেই বেস্ট এতো কম দামে পাব ভাবতেই পারি নাই আরও কিনবো ইনশাআল্লাহ
    Afert Tokenizing:  ['রাধুনি', 'মানেই', 'বেস্ট', 'এতো', 'কম', 'দামে', 'পাব', 'ভাবতেই', 'পারি', 'নাই', 'আরও', 'কিনবো', 'ইনশাআল্লাহ']
    Truncating punctuation: ['রাধুনি', 'মানেই', 'বেস্ট', 'এতো', 'কম', 'দামে', 'পাব', 'ভাবতেই', 'পারি', 'নাই', 'আরও', 'কিনবো', 'ইনশাআল্লাহ']
    Truncating StopWords: ['রাধুনি', 'মানেই', 'বেস্ট', 'এতো', 'কম', 'দামে', 'পাব', 'ভাবতেই', 'নাই', 'কিনবো', 'ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  তারগুলো মোটামুটি ভালোই , কারোর লাগলে নিঃসন্দেহে নিতে পারেন।
    Afert Tokenizing:  ['তারগুলো', 'মোটামুটি', 'ভালোই', '', ',', 'কারোর', 'লাগলে', 'নিঃসন্দেহে', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['তারগুলো', 'মোটামুটি', 'ভালোই', '', 'কারোর', 'লাগলে', 'নিঃসন্দেহে', 'নিতে', 'পারেন']
    Truncating StopWords: ['তারগুলো', 'মোটামুটি', 'ভালোই', '', 'কারোর', 'লাগলে', 'নিঃসন্দেহে']
    ***************************************************************************************
    Label:  1
    Sentence:  তার ওডার করে ভাবছিলাম এত কম দামে কি তার দেব ভালো হবে কি না তা নিয়ে সন্দহে ছিলাম কিন্তু হতে পয়ে দেখি অনেক ভালো তার এবং সাথে একটি টেস্টার ফ্রি এত কম দামে ভালো তার দেওয়ার জন্য ধন্যবাদ।
    Afert Tokenizing:  ['তার', 'ওডার', 'করে', 'ভাবছিলাম', 'এত', 'কম', 'দামে', 'কি', 'তার', 'দেব', 'ভালো', 'হবে', 'কি', 'না', 'তা', 'নিয়ে', 'সন্দহে', 'ছিলাম', 'কিন্তু', 'হতে', 'পয়ে', 'দেখি', 'অনেক', 'ভালো', 'তার', 'এবং', 'সাথে', 'একটি', 'টেস্টার', 'ফ্রি', 'এত', 'কম', 'দামে', 'ভালো', 'তার', 'দেওয়ার', 'জন্য', 'ধন্যবাদ', '।']
    Truncating punctuation: ['তার', 'ওডার', 'করে', 'ভাবছিলাম', 'এত', 'কম', 'দামে', 'কি', 'তার', 'দেব', 'ভালো', 'হবে', 'কি', 'না', 'তা', 'নিয়ে', 'সন্দহে', 'ছিলাম', 'কিন্তু', 'হতে', 'পয়ে', 'দেখি', 'অনেক', 'ভালো', 'তার', 'এবং', 'সাথে', 'একটি', 'টেস্টার', 'ফ্রি', 'এত', 'কম', 'দামে', 'ভালো', 'তার', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['ওডার', 'ভাবছিলাম', 'কম', 'দামে', 'দেব', 'ভালো', 'না', 'সন্দহে', 'ছিলাম', 'পয়ে', 'দেখি', 'ভালো', 'সাথে', 'টেস্টার', 'ফ্রি', 'কম', 'দামে', 'ভালো', 'দেওয়ার', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  কেবল ভাল, সাথে টেস্টার ফ্রি দিছে। ধন্যবাদ
    Afert Tokenizing:  ['কেবল', 'ভাল', ',', 'সাথে', 'টেস্টার', 'ফ্রি', 'দিছে', '।', 'ধন্যবাদ']
    Truncating punctuation: ['কেবল', 'ভাল', 'সাথে', 'টেস্টার', 'ফ্রি', 'দিছে', 'ধন্যবাদ']
    Truncating StopWords: ['কেবল', 'ভাল', 'সাথে', 'টেস্টার', 'ফ্রি', 'দিছে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু প্রোডাক্ট! ৭০ টাকার ১২ ভোল্ট মোটরের বাতাস এর চেয়ে অনেক বেশি!
    Afert Tokenizing:  ['ফালতু', 'প্রোডাক্ট', '!', '৭০', 'টাকার', '১২', 'ভোল্ট', 'মোটরের', 'বাতাস', 'এর', 'চেয়ে', 'অনেক', 'বেশি', '!']
    Truncating punctuation: ['ফালতু', 'প্রোডাক্ট', '৭০', 'টাকার', '১২', 'ভোল্ট', 'মোটরের', 'বাতাস', 'এর', 'চেয়ে', 'অনেক', 'বেশি']
    Truncating StopWords: ['ফালতু', 'প্রোডাক্ট', '৭০', 'টাকার', '১২', 'ভোল্ট', 'মোটরের', 'বাতাস', 'চেয়ে', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  ফ্যানটার স্পীড বেশ ভালই। ছোট ফ্যান হিসেবে খারাপ নয়। স্পীড কমানো বাড়ানোর কোন অপশন নেই। আর সব কিছু ঠিক ছিল। আর এটা কত ওয়াটের ফ্যান description লেখা নেই। ধন্যবাদ সেলার ভাই কে।
    Afert Tokenizing:  ['ফ্যানটার', 'স্পীড', 'বেশ', 'ভালই', '।', 'ছোট', 'ফ্যান', 'হিসেবে', 'খারাপ', 'নয়', '।', 'স্পীড', 'কমানো', 'বাড়ানোর', 'কোন', 'অপশন', 'নেই', '।', 'আর', 'সব', 'কিছু', 'ঠিক', 'ছিল', '।', 'আর', 'এটা', 'কত', 'ওয়াটের', 'ফ্যান', 'description', 'লেখা', 'নেই', '।', 'ধন্যবাদ', 'সেলার', 'ভাই', 'কে', '।']
    Truncating punctuation: ['ফ্যানটার', 'স্পীড', 'বেশ', 'ভালই', 'ছোট', 'ফ্যান', 'হিসেবে', 'খারাপ', 'নয়', 'স্পীড', 'কমানো', 'বাড়ানোর', 'কোন', 'অপশন', 'নেই', 'আর', 'সব', 'কিছু', 'ঠিক', 'ছিল', 'আর', 'এটা', 'কত', 'ওয়াটের', 'ফ্যান', 'description', 'লেখা', 'নেই', 'ধন্যবাদ', 'সেলার', 'ভাই', 'কে']
    Truncating StopWords: ['ফ্যানটার', 'স্পীড', 'ভালই', 'ছোট', 'ফ্যান', 'হিসেবে', 'খারাপ', 'নয়', 'স্পীড', 'কমানো', 'বাড়ানোর', 'অপশন', 'নেই', 'ঠিক', 'ওয়াটের', 'ফ্যান', 'description', 'লেখা', 'নেই', 'ধন্যবাদ', 'সেলার', 'ভাই']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় অসাধারণ।  সেলার ভাই এত দ্রুত ডেলিভারি দিবে কল্পনাও করিনি
    Afert Tokenizing:  ['এক', 'কথায়', 'অসাধারণ', '।', 'সেলার', 'ভাই', 'এত', 'দ্রুত', 'ডেলিভারি', 'দিবে', 'কল্পনাও', 'করিনি']
    Truncating punctuation: ['এক', 'কথায়', 'অসাধারণ', 'সেলার', 'ভাই', 'এত', 'দ্রুত', 'ডেলিভারি', 'দিবে', 'কল্পনাও', 'করিনি']
    Truncating StopWords: ['এক', 'কথায়', 'অসাধারণ', 'সেলার', 'ভাই', 'দ্রুত', 'ডেলিভারি', 'দিবে', 'কল্পনাও', 'করিনি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি যেমন টা চাইছি ঠিক তেমন টাই পাইছি। অনেক ভাল  বেল্ট। টাকা হিসাবে অনেক ভাল হইছে
    Afert Tokenizing:  ['আমি', 'যেমন', 'টা', 'চাইছি', 'ঠিক', 'তেমন', 'টাই', 'পাইছি', '।', 'অনেক', 'ভাল', 'বেল্ট', '।', 'টাকা', 'হিসাবে', 'অনেক', 'ভাল', 'হইছে']
    Truncating punctuation: ['আমি', 'যেমন', 'টা', 'চাইছি', 'ঠিক', 'তেমন', 'টাই', 'পাইছি', 'অনেক', 'ভাল', 'বেল্ট', 'টাকা', 'হিসাবে', 'অনেক', 'ভাল', 'হইছে']
    Truncating StopWords: ['টা', 'চাইছি', 'ঠিক', 'টাই', 'পাইছি', 'ভাল', 'বেল্ট', 'টাকা', 'ভাল', 'হইছে']
    ***************************************************************************************
    Label:  1
    Sentence:  জিনিস  টা ভালোই বাট বড় মনে করছিলাম কিন্তু এটা একদম ছোট তার পরেও ভালোই
    Afert Tokenizing:  ['জিনিস', 'টা', 'ভালোই', 'বাট', 'বড়', 'মনে', 'করছিলাম', 'কিন্তু', 'এটা', 'একদম', 'ছোট', 'তার', 'পরেও', 'ভালোই']
    Truncating punctuation: ['জিনিস', 'টা', 'ভালোই', 'বাট', 'বড়', 'মনে', 'করছিলাম', 'কিন্তু', 'এটা', 'একদম', 'ছোট', 'তার', 'পরেও', 'ভালোই']
    Truncating StopWords: ['জিনিস', 'টা', 'ভালোই', 'বাট', 'বড়', 'করছিলাম', 'একদম', 'ছোট', 'ভালোই']
    ***************************************************************************************
    Label:  0
    Sentence:  কোয়ালিটি এভারেজ, সেলাইগুলো দুর্বল মনে হয়েছে। তবে ব্যাগটা দেখতে সুন্দর, বাচ্চাদের সাথে অনেক মানানসই। ভারী কিছু না বহন করাই বুদ্ধিমানের কাজ হবে।
    Afert Tokenizing:  ['কোয়ালিটি', 'এভারেজ', ',', 'সেলাইগুলো', 'দুর্বল', 'মনে', 'হয়েছে', '।', 'তবে', 'ব্যাগটা', 'দেখতে', 'সুন্দর', ',', 'বাচ্চাদের', 'সাথে', 'অনেক', 'মানানসই', '।', 'ভারী', 'কিছু', 'না', 'বহন', 'করাই', 'বুদ্ধিমানের', 'কাজ', 'হবে', '।']
    Truncating punctuation: ['কোয়ালিটি', 'এভারেজ', 'সেলাইগুলো', 'দুর্বল', 'মনে', 'হয়েছে', 'তবে', 'ব্যাগটা', 'দেখতে', 'সুন্দর', 'বাচ্চাদের', 'সাথে', 'অনেক', 'মানানসই', 'ভারী', 'কিছু', 'না', 'বহন', 'করাই', 'বুদ্ধিমানের', 'কাজ', 'হবে']
    Truncating StopWords: ['কোয়ালিটি', 'এভারেজ', 'সেলাইগুলো', 'দুর্বল', 'ব্যাগটা', 'সুন্দর', 'বাচ্চাদের', 'সাথে', 'মানানসই', 'ভারী', 'না', 'বহন', 'বুদ্ধিমানের']
    ***************************************************************************************
    Label:  1
    Sentence:  এতো অল্প দামে অসাধারণ পণ্য, আগেও নিলাম তবে এটা বেস্ট আগের শপ থেকে।
    Afert Tokenizing:  ['এতো', 'অল্প', 'দামে', 'অসাধারণ', 'পণ্য', ',', 'আগেও', 'নিলাম', 'তবে', 'এটা', 'বেস্ট', 'আগের', 'শপ', 'থেকে', '।']
    Truncating punctuation: ['এতো', 'অল্প', 'দামে', 'অসাধারণ', 'পণ্য', 'আগেও', 'নিলাম', 'তবে', 'এটা', 'বেস্ট', 'আগের', 'শপ', 'থেকে']
    Truncating StopWords: ['এতো', 'অল্প', 'দামে', 'অসাধারণ', 'পণ্য', 'আগেও', 'নিলাম', 'বেস্ট', 'আগের', 'শপ']
    ***************************************************************************************
    Label:  0
    Sentence:  Underwear অর্ডার দিলাম xxl size (41-44) ২টা..কিন্তু দিলেন একটা ছোট সাইজ। পিক দেখলেই বুঝবেন যে একটা কত ছোট দিছেন!! তারপর একটা  H&M Brand এর অর্ডার ছিলো। দিলেন ২টাই US Polo. তাও সাইজ ছোট। ফালতু পুরাই!!
    Afert Tokenizing:  ['Underwear', 'অর্ডার', 'দিলাম', 'xxl', 'size', '(41-44', ')', '২টা..কিন্তু', 'দিলেন', 'একটা', 'ছোট', 'সাইজ', '।', 'পিক', 'দেখলেই', 'বুঝবেন', 'যে', 'একটা', 'কত', 'ছোট', 'দিছেন!', '!', 'তারপর', 'একটা', 'H&M', 'Brand', 'এর', 'অর্ডার', 'ছিলো', '।', 'দিলেন', '২টাই', 'US', 'Polo', '.', 'তাও', 'সাইজ', 'ছোট', '।', 'ফালতু', 'পুরাই!', '!']
    Truncating punctuation: ['Underwear', 'অর্ডার', 'দিলাম', 'xxl', 'size', '(41-44', '২টা..কিন্তু', 'দিলেন', 'একটা', 'ছোট', 'সাইজ', 'পিক', 'দেখলেই', 'বুঝবেন', 'যে', 'একটা', 'কত', 'ছোট', 'দিছেন!', 'তারপর', 'একটা', 'H&M', 'Brand', 'এর', 'অর্ডার', 'ছিলো', 'দিলেন', '২টাই', 'US', 'Polo', 'তাও', 'সাইজ', 'ছোট', 'ফালতু', 'পুরাই!']
    Truncating StopWords: ['Underwear', 'অর্ডার', 'দিলাম', 'xxl', 'size', '(41-44', '২টা..কিন্তু', 'একটা', 'ছোট', 'সাইজ', 'পিক', 'দেখলেই', 'বুঝবেন', 'একটা', 'ছোট', 'দিছেন!', 'একটা', 'H&M', 'Brand', 'অর্ডার', 'ছিলো', '২টাই', 'US', 'Polo', 'সাইজ', 'ছোট', 'ফালতু', 'পুরাই!']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করেছিলাম এক্সেল কিন্তু পেয়েছি একদম ছোট সাইজ! এবং দুইটা দুই সাইজের! ভণ্ডামির একটা সিমা থাকা দরকার!
    Afert Tokenizing:  ['অর্ডার', 'করেছিলাম', 'এক্সেল', 'কিন্তু', 'পেয়েছি', 'একদম', 'ছোট', 'সাইজ', '!', 'এবং', 'দুইটা', 'দুই', 'সাইজের', '!', 'ভণ্ডামির', 'একটা', 'সিমা', 'থাকা', 'দরকার', '!']
    Truncating punctuation: ['অর্ডার', 'করেছিলাম', 'এক্সেল', 'কিন্তু', 'পেয়েছি', 'একদম', 'ছোট', 'সাইজ', 'এবং', 'দুইটা', 'দুই', 'সাইজের', 'ভণ্ডামির', 'একটা', 'সিমা', 'থাকা', 'দরকার']
    Truncating StopWords: ['অর্ডার', 'করেছিলাম', 'এক্সেল', 'পেয়েছি', 'একদম', 'ছোট', 'সাইজ', 'দুইটা', 'সাইজের', 'ভণ্ডামির', 'একটা', 'সিমা', 'দরকার']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার ছিল লার্জ সাইজের, নেভি ব্লু কালার। হাতে পেলাম মিডিয়াম সাইজের, সাদা রং। খারাপ লাগলো। এখন আবার রিটার্নের ঝামেলা।
    Afert Tokenizing:  ['অর্ডার', 'ছিল', 'লার্জ', 'সাইজের', ',', 'নেভি', 'ব্লু', 'কালার', '।', 'হাতে', 'পেলাম', 'মিডিয়াম', 'সাইজের', ',', 'সাদা', 'রং', '।', 'খারাপ', 'লাগলো', '।', 'এখন', 'আবার', 'রিটার্নের', 'ঝামেলা', '।']
    Truncating punctuation: ['অর্ডার', 'ছিল', 'লার্জ', 'সাইজের', 'নেভি', 'ব্লু', 'কালার', 'হাতে', 'পেলাম', 'মিডিয়াম', 'সাইজের', 'সাদা', 'রং', 'খারাপ', 'লাগলো', 'এখন', 'আবার', 'রিটার্নের', 'ঝামেলা']
    Truncating StopWords: ['অর্ডার', 'লার্জ', 'সাইজের', 'নেভি', 'ব্লু', 'কালার', 'হাতে', 'পেলাম', 'মিডিয়াম', 'সাইজের', 'সাদা', 'রং', 'খারাপ', 'লাগলো', 'রিটার্নের', 'ঝামেলা']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথাই দারুণ , পড়লে কিছুই বুঝা যায় না আর সেলার ও খুব আন্তরিক । এর আগেও আমি এই সেলারের কাছে পন্য কিনেছি
    Afert Tokenizing:  ['এক', 'কথাই', 'দারুণ', '', ',', 'পড়লে', 'কিছুই', 'বুঝা', 'যায়', 'না', 'আর', 'সেলার', 'ও', 'খুব', 'আন্তরিক', '', '।', 'এর', 'আগেও', 'আমি', 'এই', 'সেলারের', 'কাছে', 'পন্য', 'কিনেছি']
    Truncating punctuation: ['এক', 'কথাই', 'দারুণ', '', 'পড়লে', 'কিছুই', 'বুঝা', 'যায়', 'না', 'আর', 'সেলার', 'ও', 'খুব', 'আন্তরিক', '', 'এর', 'আগেও', 'আমি', 'এই', 'সেলারের', 'কাছে', 'পন্য', 'কিনেছি']
    Truncating StopWords: ['এক', 'কথাই', 'দারুণ', '', 'পড়লে', 'বুঝা', 'যায়', 'না', 'সেলার', 'আন্তরিক', '', 'আগেও', 'সেলারের', 'পন্য', 'কিনেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  বুঝতে পারলাম না একপ্যকেটে ২টা থাকার কথা কিন্তু আছে ১টা
    Afert Tokenizing:  ['বুঝতে', 'পারলাম', 'না', 'একপ্যকেটে', '২টা', 'থাকার', 'কথা', 'কিন্তু', 'আছে', '১টা']
    Truncating punctuation: ['বুঝতে', 'পারলাম', 'না', 'একপ্যকেটে', '২টা', 'থাকার', 'কথা', 'কিন্তু', 'আছে', '১টা']
    Truncating StopWords: ['বুঝতে', 'পারলাম', 'না', 'একপ্যকেটে', '২টা', 'থাকার', 'কথা', '১টা']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপাতদৃষ্টিতে দেখে ভালই মনে হচ্ছে। কতটা মজবুত সেটা এখনো পরীক্ষা করে দেখে নি। অর্ডার করার তিন দিনের মাথায় প্রোডাক্ট হাতে পেয়েছি। কিন্তু বক্সের ভিতরে স্ক্রু ড্রাইভার হাতলের গার্ড , ভাঙ্গা ছিল।
    Afert Tokenizing:  ['আপাতদৃষ্টিতে', '"', 'দেখে', 'ভালই', 'মনে', 'হচ্ছে', '।', 'কতটা', 'মজবুত', 'সেটা', 'এখনো', 'পরীক্ষা', 'করে', 'দেখে', 'নি', '।', 'অর্ডার', 'করার', 'তিন', 'দিনের', 'মাথায়', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', '।', 'কিন্তু', 'বক্সের', 'ভিতরে', 'স্ক্রু', 'ড্রাইভার', 'হাতলের', 'গার্ড', '', ',', 'ভাঙ্গা', 'ছিল', '।']
    Truncating punctuation: ['আপাতদৃষ্টিতে', 'দেখে', 'ভালই', 'মনে', 'হচ্ছে', 'কতটা', 'মজবুত', 'সেটা', 'এখনো', 'পরীক্ষা', 'করে', 'দেখে', 'নি', 'অর্ডার', 'করার', 'তিন', 'দিনের', 'মাথায়', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'কিন্তু', 'বক্সের', 'ভিতরে', 'স্ক্রু', 'ড্রাইভার', 'হাতলের', 'গার্ড', '', 'ভাঙ্গা', 'ছিল']
    Truncating StopWords: ['আপাতদৃষ্টিতে', 'ভালই', 'কতটা', 'মজবুত', 'এখনো', 'পরীক্ষা', 'নি', 'অর্ডার', 'তিন', 'দিনের', 'মাথায়', 'প্রোডাক্ট', 'হাতে', 'পেয়েছি', 'বক্সের', 'ভিতরে', 'স্ক্রু', 'ড্রাইভার', 'হাতলের', 'গার্ড', '', 'ভাঙ্গা']
    ***************************************************************************************
    Label:  1
    Sentence:  "দাম অনুযায়ী ভালো প্রডাক্ট, ছোট খাটো কাজ চালিয়ে নেয়া যাবে।  প্রডাক্ট ঠিক পেয়েছি,  কিন্তু স্কু বিটের পাশে প্রতিটি বিটের নাম লেখা থাকার কথা, কিন্তু আমার এটার মধ্যে নাম নেই!"
    Afert Tokenizing:  ['দাম', '"', 'অনুযায়ী', 'ভালো', 'প্রডাক্ট', ',', 'ছোট', 'খাটো', 'কাজ', 'চালিয়ে', 'নেয়া', 'যাবে', '।', 'প্রডাক্ট', 'ঠিক', 'পেয়েছি', ',', 'কিন্তু', 'স্কু', 'বিটের', 'পাশে', 'প্রতিটি', 'বিটের', 'নাম', 'লেখা', 'থাকার', 'কথা', ',', 'কিন্তু', 'আমার', 'এটার', 'মধ্যে', 'নাম', 'নেই!', '"']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'ভালো', 'প্রডাক্ট', 'ছোট', 'খাটো', 'কাজ', 'চালিয়ে', 'নেয়া', 'যাবে', 'প্রডাক্ট', 'ঠিক', 'পেয়েছি', 'কিন্তু', 'স্কু', 'বিটের', 'পাশে', 'প্রতিটি', 'বিটের', 'নাম', 'লেখা', 'থাকার', 'কথা', 'কিন্তু', 'আমার', 'এটার', 'মধ্যে', 'নাম', 'নেই!']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'ভালো', 'প্রডাক্ট', 'ছোট', 'খাটো', 'চালিয়ে', 'নেয়া', 'প্রডাক্ট', 'ঠিক', 'পেয়েছি', 'স্কু', 'বিটের', 'পাশে', 'প্রতিটি', 'বিটের', 'নাম', 'লেখা', 'থাকার', 'কথা', 'এটার', 'নাম', 'নেই!']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশাআল্লাহ অনেক সুন্দর একটি Product. দাম হিসেবে ঠিক আছে। সবাই নিতে পারেন
    Afert Tokenizing:  ['মাশাআল্লাহ', 'অনেক', 'সুন্দর', 'একটি', 'Product', '.', 'দাম', 'হিসেবে', 'ঠিক', 'আছে', '।', 'সবাই', 'নিতে', 'পারেন']
    Truncating punctuation: ['মাশাআল্লাহ', 'অনেক', 'সুন্দর', 'একটি', 'Product', 'দাম', 'হিসেবে', 'ঠিক', 'আছে', 'সবাই', 'নিতে', 'পারেন']
    Truncating StopWords: ['মাশাআল্লাহ', 'সুন্দর', 'Product', 'দাম', 'হিসেবে', 'ঠিক', 'সবাই']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট টিক আছে ডেলিভারি সময় মত কিন্তু কালারটা পিংক
    Afert Tokenizing:  ['প্রোডাক্ট', 'টিক', 'আছে', 'ডেলিভারি', 'সময়', 'মত', 'কিন্তু', 'কালারটা', 'পিংক']
    Truncating punctuation: ['প্রোডাক্ট', 'টিক', 'আছে', 'ডেলিভারি', 'সময়', 'মত', 'কিন্তু', 'কালারটা', 'পিংক']
    Truncating StopWords: ['প্রোডাক্ট', 'টিক', 'ডেলিভারি', 'সময়', 'মত', 'কালারটা', 'পিংক']
    ***************************************************************************************
    Label:  1
    Sentence:  "জিনিসটা অনেক ভালো হইছে, আমি যার জন্য প্রোডাক্টটা নিছি সে অনেক পছন্দ করছে"
    Afert Tokenizing:  ['জিনিসটা', '"', 'অনেক', 'ভালো', 'হইছে', ',', 'আমি', 'যার', 'জন্য', 'প্রোডাক্টটা', 'নিছি', 'সে', 'অনেক', 'পছন্দ', 'করছে', '"']
    Truncating punctuation: ['জিনিসটা', 'অনেক', 'ভালো', 'হইছে', 'আমি', 'যার', 'জন্য', 'প্রোডাক্টটা', 'নিছি', 'সে', 'অনেক', 'পছন্দ', 'করছে']
    Truncating StopWords: ['জিনিসটা', 'ভালো', 'হইছে', 'প্রোডাক্টটা', 'নিছি', 'পছন্দ']
    ***************************************************************************************
    Label:  1
    Sentence:  পারফেক্ট সাইজ ও আরামদায়ক ৷ সন্তুষ্ট এটা পেয়ে আমি ৷ আপনারা ও নিশ্চিন্তে এটি কিনতে পারেন
    Afert Tokenizing:  ['পারফেক্ট', 'সাইজ', 'ও', 'আরামদায়ক', '৷', 'সন্তুষ্ট', 'এটা', 'পেয়ে', 'আমি', '৷', 'আপনারা', 'ও', 'নিশ্চিন্তে', 'এটি', 'কিনতে', 'পারেন']
    Truncating punctuation: ['পারফেক্ট', 'সাইজ', 'ও', 'আরামদায়ক', '৷', 'সন্তুষ্ট', 'এটা', 'পেয়ে', 'আমি', '৷', 'আপনারা', 'ও', 'নিশ্চিন্তে', 'এটি', 'কিনতে', 'পারেন']
    Truncating StopWords: ['পারফেক্ট', 'সাইজ', 'আরামদায়ক', '৷', 'সন্তুষ্ট', '৷', 'আপনারা', 'নিশ্চিন্তে', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  ফা গুলা মোটামুটি ভালই। দাম হিসাবে ঠিক আছে। সোফা পেয়ে সবাই খুবই আনন্দ!এখন দেখা যাবে কত দিন যায়।
    Afert Tokenizing:  ['ফা', 'গুলা', 'মোটামুটি', 'ভালই', '।', 'দাম', 'হিসাবে', 'ঠিক', 'আছে', '।', 'সোফা', 'পেয়ে', 'সবাই', 'খুবই', 'আনন্দ!এখন', 'দেখা', 'যাবে', 'কত', 'দিন', 'যায়', '।']
    Truncating punctuation: ['ফা', 'গুলা', 'মোটামুটি', 'ভালই', 'দাম', 'হিসাবে', 'ঠিক', 'আছে', 'সোফা', 'পেয়ে', 'সবাই', 'খুবই', 'আনন্দ!এখন', 'দেখা', 'যাবে', 'কত', 'দিন', 'যায়']
    Truncating StopWords: ['ফা', 'গুলা', 'মোটামুটি', 'ভালই', 'দাম', 'ঠিক', 'সোফা', 'সবাই', 'খুবই', 'আনন্দ!এখন']
    ***************************************************************************************
    Label:  0
    Sentence:  মোটামুটি ভালো. .কিন্তু আমার সোফা র ফিনিশিং ভালো হয় নি
    Afert Tokenizing:  ['মোটামুটি', 'ভালো', '.', 'কিন্তু', '.', 'আমার', 'সোফা', 'র', 'ফিনিশিং', 'ভালো', 'হয়', 'নি']
    Truncating punctuation: ['মোটামুটি', 'ভালো', 'কিন্তু', 'আমার', 'সোফা', 'র', 'ফিনিশিং', 'ভালো', 'হয়', 'নি']
    Truncating StopWords: ['মোটামুটি', 'ভালো', 'সোফা', 'ফিনিশিং', 'ভালো', 'নি']
    ***************************************************************************************
    Label:  1
    Sentence:  A লেভেল এর একটা  জিনিস   সেলারকে বলবো এই রকম মানসম্পন্ন প্রোডাক্ট সেল করবেন
    Afert Tokenizing:  ['A', 'লেভেল', 'এর', 'একটা', 'জিনিস', 'সেলারকে', 'বলবো', 'এই', 'রকম', 'মানসম্পন্ন', 'প্রোডাক্ট', 'সেল', 'করবেন']
    Truncating punctuation: ['A', 'লেভেল', 'এর', 'একটা', 'জিনিস', 'সেলারকে', 'বলবো', 'এই', 'রকম', 'মানসম্পন্ন', 'প্রোডাক্ট', 'সেল', 'করবেন']
    Truncating StopWords: ['A', 'লেভেল', 'একটা', 'জিনিস', 'সেলারকে', 'বলবো', 'মানসম্পন্ন', 'প্রোডাক্ট', 'সেল']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেজিং + উপরের মোড়ক এক কথায় অসাধারণ ছিল
    Afert Tokenizing:  ['প্যাকেজিং', '+', 'উপরের', 'মোড়ক', 'এক', 'কথায়', 'অসাধারণ', 'ছিল']
    Truncating punctuation: ['প্যাকেজিং', '+', 'উপরের', 'মোড়ক', 'এক', 'কথায়', 'অসাধারণ', 'ছিল']
    Truncating StopWords: ['প্যাকেজিং', '+', 'উপরের', 'মোড়ক', 'এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  "আপাতত সব ঠিকঠাক আছে,  ব্যবহার করে দেখি কেমন । প্যাকেট ও সুন্দর ভাবে পাইছি।"
    Afert Tokenizing:  ['আপাতত', '"', 'সব', 'ঠিকঠাক', 'আছে', ',', 'ব্যবহার', 'করে', 'দেখি', 'কেমন', '', '।', 'প্যাকেট', 'ও', 'সুন্দর', 'ভাবে', 'পাইছি।', '"']
    Truncating punctuation: ['আপাতত', 'সব', 'ঠিকঠাক', 'আছে', 'ব্যবহার', 'করে', 'দেখি', 'কেমন', '', 'প্যাকেট', 'ও', 'সুন্দর', 'ভাবে', 'পাইছি।']
    Truncating StopWords: ['আপাতত', 'ঠিকঠাক', 'দেখি', 'কেমন', '', 'প্যাকেট', 'সুন্দর', 'পাইছি।']
    ***************************************************************************************
    Label:  1
    Sentence:  "দাম কম দেখে ভাবছিলাম প্রোডাক্টই ভাল হবে না, আসলে দাম অনুযায়ী অনেক ভালো "
    Afert Tokenizing:  ['দাম', '"', 'কম', 'দেখে', 'ভাবছিলাম', 'প্রোডাক্টই', 'ভাল', 'হবে', 'না', ',', 'আসলে', 'দাম', 'অনুযায়ী', 'অনেক', 'ভালো', '', '"']
    Truncating punctuation: ['দাম', 'কম', 'দেখে', 'ভাবছিলাম', 'প্রোডাক্টই', 'ভাল', 'হবে', 'না', 'আসলে', 'দাম', 'অনুযায়ী', 'অনেক', 'ভালো', '']
    Truncating StopWords: ['দাম', 'কম', 'ভাবছিলাম', 'প্রোডাক্টই', 'ভাল', 'না', 'আসলে', 'দাম', 'ভালো', '']
    ***************************************************************************************
    Label:  1
    Sentence:  "খুবই ভালো এবং মজবুত,  রিজনাবল"
    Afert Tokenizing:  ['খুবই', '"', 'ভালো', 'এবং', 'মজবুত', ',', 'রিজনাবল', '"']
    Truncating punctuation: ['খুবই', 'ভালো', 'এবং', 'মজবুত', 'রিজনাবল']
    Truncating StopWords: ['খুবই', 'ভালো', 'মজবুত', 'রিজনাবল']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ আগের দিন অর্ডার পরের দিন ডেলিভারি মাস্কগুলোর তুলনা হয়না।দেখতে ফ্রেস,ব্যবহারেও খুবই আরামদায়ক"
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', '"', 'আগের', 'দিন', 'অর্ডার', 'পরের', 'দিন', 'ডেলিভারি', 'মাস্কগুলোর', 'তুলনা', 'হয়না।দেখতে', 'ফ্রেস,ব্যবহারেও', 'খুবই', 'আরামদায়ক', '"']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আগের', 'দিন', 'অর্ডার', 'পরের', 'দিন', 'ডেলিভারি', 'মাস্কগুলোর', 'তুলনা', 'হয়না।দেখতে', 'ফ্রেস,ব্যবহারেও', 'খুবই', 'আরামদায়ক']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'আগের', 'অর্ডার', 'পরের', 'ডেলিভারি', 'মাস্কগুলোর', 'তুলনা', 'হয়না।দেখতে', 'ফ্রেস,ব্যবহারেও', 'খুবই', 'আরামদায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  "ভাল একটি অ্যান্ড্রয়েড ফোন অল্প দামের মধ্যেই, আমার পছন্দ হয়েছে, আপনারা চাইলে নিতে পারেন"
    Afert Tokenizing:  ['ভাল', '"', 'একটি', 'অ্যান্ড্রয়েড', 'ফোন', 'অল্প', 'দামের', 'মধ্যেই', ',', 'আমার', 'পছন্দ', 'হয়েছে', ',', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন', '"']
    Truncating punctuation: ['ভাল', 'একটি', 'অ্যান্ড্রয়েড', 'ফোন', 'অল্প', 'দামের', 'মধ্যেই', 'আমার', 'পছন্দ', 'হয়েছে', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['ভাল', 'অ্যান্ড্রয়েড', 'ফোন', 'অল্প', 'দামের', 'পছন্দ', 'হয়েছে', 'আপনারা', 'চাইলে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনকে দ্রুত পেয়েছি রাতে অর্ডার দিয়ে সকাল ১১টায় ডেলিভারি পাইছি প্রোডাক্ট অরিজিনাল মনে হচ্ছে বাকিটা ইউস করার পর বুজা যাবে
    Afert Tokenizing:  ['অনকে', 'দ্রুত', 'পেয়েছি', 'রাতে', 'অর্ডার', 'দিয়ে', 'সকাল', '১১টায়', 'ডেলিভারি', 'পাইছি', 'প্রোডাক্ট', 'অরিজিনাল', 'মনে', 'হচ্ছে', 'বাকিটা', 'ইউস', 'করার', 'পর', 'বুজা', 'যাবে']
    Truncating punctuation: ['অনকে', 'দ্রুত', 'পেয়েছি', 'রাতে', 'অর্ডার', 'দিয়ে', 'সকাল', '১১টায়', 'ডেলিভারি', 'পাইছি', 'প্রোডাক্ট', 'অরিজিনাল', 'মনে', 'হচ্ছে', 'বাকিটা', 'ইউস', 'করার', 'পর', 'বুজা', 'যাবে']
    Truncating StopWords: ['অনকে', 'দ্রুত', 'পেয়েছি', 'রাতে', 'অর্ডার', 'দিয়ে', 'সকাল', '১১টায়', 'ডেলিভারি', 'পাইছি', 'প্রোডাক্ট', 'অরিজিনাল', 'বাকিটা', 'ইউস', 'বুজা']
    ***************************************************************************************
    Label:  1
    Sentence:  মোবাইল পেয়েছি ঠিক ঠাক পেকেজিং টাও ভালো ছিলো
    Afert Tokenizing:  ['মোবাইল', 'পেয়েছি', 'ঠিক', 'ঠাক', 'পেকেজিং', 'টাও', 'ভালো', 'ছিলো']
    Truncating punctuation: ['মোবাইল', 'পেয়েছি', 'ঠিক', 'ঠাক', 'পেকেজিং', 'টাও', 'ভালো', 'ছিলো']
    Truncating StopWords: ['মোবাইল', 'পেয়েছি', 'ঠিক', 'ঠাক', 'পেকেজিং', 'টাও', 'ভালো', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারটি খেতে অনেক সুস্বাদু  এবং কোয়ালিটি খুবই ভাল। কেউ খেতে চাইলে অর্ডার করতে পারেন।
    Afert Tokenizing:  ['খাবারটি', 'খেতে', 'অনেক', 'সুস্বাদু', 'এবং', 'কোয়ালিটি', 'খুবই', 'ভাল', '।', 'কেউ', 'খেতে', 'চাইলে', 'অর্ডার', 'করতে', 'পারেন', '।']
    Truncating punctuation: ['খাবারটি', 'খেতে', 'অনেক', 'সুস্বাদু', 'এবং', 'কোয়ালিটি', 'খুবই', 'ভাল', 'কেউ', 'খেতে', 'চাইলে', 'অর্ডার', 'করতে', 'পারেন']
    Truncating StopWords: ['খাবারটি', 'খেতে', 'সুস্বাদু', 'কোয়ালিটি', 'খুবই', 'ভাল', 'খেতে', 'চাইলে', 'অর্ডার']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ, সময় মতো হাতে পেয়েছি। এবং প্রোডাক্টি অনেক সুন্দর ছিল। এটি অনেক সুস্বাদু।"
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ', ',', 'সময়', 'মতো', 'হাতে', 'পেয়েছি', '।', 'এবং', 'প্রোডাক্টি', 'অনেক', 'সুন্দর', 'ছিল', '।', 'এটি', 'অনেক', 'সুস্বাদু।', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ', 'সময়', 'মতো', 'হাতে', 'পেয়েছি', 'এবং', 'প্রোডাক্টি', 'অনেক', 'সুন্দর', 'ছিল', 'এটি', 'অনেক', 'সুস্বাদু।']
    Truncating StopWords: ['"আলহামদুলিল্লাহ', 'সময়', 'হাতে', 'পেয়েছি', 'প্রোডাক্টি', 'সুন্দর', 'সুস্বাদু।']
    ***************************************************************************************
    Label:  1
    Sentence:  চাইছি সাদা কিন্তু পাইছি নীল
    Afert Tokenizing:  ['চাইছি', 'সাদা', 'কিন্তু', 'পাইছি', 'নীল']
    Truncating punctuation: ['চাইছি', 'সাদা', 'কিন্তু', 'পাইছি', 'নীল']
    Truncating StopWords: ['চাইছি', 'সাদা', 'পাইছি', 'নীল']
    ***************************************************************************************
    Label:  1
    Sentence:  "অনেক ভালে কোয়ালিটি। আঠা বেশ স্ট্রং। সেলার অনেক রেস্পন্সিবল। সবাই নিতে পারেন।
    Afert Tokenizing:  ['অনেক', '"', 'ভালে', 'কোয়ালিটি', '।', 'আঠা', 'বেশ', 'স্ট্রং', '।', 'সেলার', 'অনেক', 'রেস্পন্সিবল', '।', 'সবাই', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['অনেক', 'ভালে', 'কোয়ালিটি', 'আঠা', 'বেশ', 'স্ট্রং', 'সেলার', 'অনেক', 'রেস্পন্সিবল', 'সবাই', 'নিতে', 'পারেন']
    Truncating StopWords: ['ভালে', 'কোয়ালিটি', 'আঠা', 'স্ট্রং', 'সেলার', 'রেস্পন্সিবল', 'সবাই']
    ***************************************************************************************
    Label:  1
    Sentence:  "কি বোলবো অর্ডার কোরলাম সাদা দিলো ব্লু,এটা ঠিকনা,এছারা সব ঠিক আছে প্যাকেটিং ভাল ছিল,নতুন মাল দিয়েছে,সাউন্ড ভালোই।"
    Afert Tokenizing:  ['কি', '"', 'বোলবো', 'অর্ডার', 'কোরলাম', 'সাদা', 'দিলো', 'ব্লু,এটা', 'ঠিকনা,এছারা', 'সব', 'ঠিক', 'আছে', 'প্যাকেটিং', 'ভাল', 'ছিল,নতুন', 'মাল', 'দিয়েছে,সাউন্ড', 'ভালোই।', '"']
    Truncating punctuation: ['কি', 'বোলবো', 'অর্ডার', 'কোরলাম', 'সাদা', 'দিলো', 'ব্লু,এটা', 'ঠিকনা,এছারা', 'সব', 'ঠিক', 'আছে', 'প্যাকেটিং', 'ভাল', 'ছিল,নতুন', 'মাল', 'দিয়েছে,সাউন্ড', 'ভালোই।']
    Truncating StopWords: ['বোলবো', 'অর্ডার', 'কোরলাম', 'সাদা', 'দিলো', 'ব্লু,এটা', 'ঠিকনা,এছারা', 'ঠিক', 'প্যাকেটিং', 'ভাল', 'ছিল,নতুন', 'মাল', 'দিয়েছে,সাউন্ড', 'ভালোই।']
    ***************************************************************************************
    Label:  1
    Sentence:  যে টেপ দেওয়া হয়েছে তা পূরাতন এক পাশে ময়লা লাগানো । অন্য পাশ ঠিক আছে
    Afert Tokenizing:  ['যে', 'টেপ', 'দেওয়া', 'হয়েছে', 'তা', 'পূরাতন', 'এক', 'পাশে', 'ময়লা', 'লাগানো', '', '।', 'অন্য', 'পাশ', 'ঠিক', 'আছে']
    Truncating punctuation: ['যে', 'টেপ', 'দেওয়া', 'হয়েছে', 'তা', 'পূরাতন', 'এক', 'পাশে', 'ময়লা', 'লাগানো', '', 'অন্য', 'পাশ', 'ঠিক', 'আছে']
    Truncating StopWords: ['টেপ', 'পূরাতন', 'এক', 'পাশে', 'ময়লা', 'লাগানো', '', 'পাশ', 'ঠিক']
    ***************************************************************************************
    Label:  1
    Sentence:  "ভালো প্রোডাক্ট  ছিলো। প্যেকেজিং ও খুব ভালো ছিলো। রিকমেন্ডড "
    Afert Tokenizing:  ['ভালো', '"', 'প্রোডাক্ট', 'ছিলো', '।', 'প্যেকেজিং', 'ও', 'খুব', 'ভালো', 'ছিলো', '।', 'রিকমেন্ডড', '', '"']
    Truncating punctuation: ['ভালো', 'প্রোডাক্ট', 'ছিলো', 'প্যেকেজিং', 'ও', 'খুব', 'ভালো', 'ছিলো', 'রিকমেন্ডড', '']
    Truncating StopWords: ['ভালো', 'প্রোডাক্ট', 'ছিলো', 'প্যেকেজিং', 'ভালো', 'ছিলো', 'রিকমেন্ডড', '']
    ***************************************************************************************
    Label:  1
    Sentence:  "দামের তুলনায় অনেক ভালো কোয়ালিটির একটা গ্লাস, প্যাকেজিং টাও সুন্দর ছিল, ২ দিনেই প্রোডাক্ট হাতে পেয়ে গেছি। সেলারকে অনেক ধন্যবাদ।
    Afert Tokenizing:  ['দামের', '"', 'তুলনায়', 'অনেক', 'ভালো', 'কোয়ালিটির', 'একটা', 'গ্লাস', ',', 'প্যাকেজিং', 'টাও', 'সুন্দর', 'ছিল', ',', '২', 'দিনেই', 'প্রোডাক্ট', 'হাতে', 'পেয়ে', 'গেছি', '।', 'সেলারকে', 'অনেক', 'ধন্যবাদ', '।']
    Truncating punctuation: ['দামের', 'তুলনায়', 'অনেক', 'ভালো', 'কোয়ালিটির', 'একটা', 'গ্লাস', 'প্যাকেজিং', 'টাও', 'সুন্দর', 'ছিল', '২', 'দিনেই', 'প্রোডাক্ট', 'হাতে', 'পেয়ে', 'গেছি', 'সেলারকে', 'অনেক', 'ধন্যবাদ']
    Truncating StopWords: ['দামের', 'তুলনায়', 'ভালো', 'কোয়ালিটির', 'একটা', 'গ্লাস', 'প্যাকেজিং', 'টাও', 'সুন্দর', '২', 'দিনেই', 'প্রোডাক্ট', 'হাতে', 'পেয়ে', 'গেছি', 'সেলারকে', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ একটি পণ্য।১৭৭ টাকায় এমন পণ্য বাজারে পাওয়া অসম্ভব।সাথে পেয়েছি ৩ দিনে হোম ডেলিভারি
    Afert Tokenizing:  ['অসাধারণ', 'একটি', 'পণ্য।১৭৭', 'টাকায়', 'এমন', 'পণ্য', 'বাজারে', 'পাওয়া', 'অসম্ভব।সাথে', 'পেয়েছি', '৩', 'দিনে', 'হোম', 'ডেলিভারি']
    Truncating punctuation: ['অসাধারণ', 'একটি', 'পণ্য।১৭৭', 'টাকায়', 'এমন', 'পণ্য', 'বাজারে', 'পাওয়া', 'অসম্ভব।সাথে', 'পেয়েছি', '৩', 'দিনে', 'হোম', 'ডেলিভারি']
    Truncating StopWords: ['অসাধারণ', 'পণ্য।১৭৭', 'টাকায়', 'পণ্য', 'বাজারে', 'পাওয়া', 'অসম্ভব।সাথে', 'পেয়েছি', '৩', 'দিনে', 'হোম', 'ডেলিভারি']
    ***************************************************************************************
    Label:  1
    Sentence:  "কোয়ালিটি সব ঠিক আছে, কিন্তু সাইজ এত বড় দিয়েছে যে আমার একদম হয়না। পুরা মুখই ঢেকে যায়। এখন কাউকে দিয়ে দিতে হবে"
    Afert Tokenizing:  ['কোয়ালিটি', '"', 'সব', 'ঠিক', 'আছে', ',', 'কিন্তু', 'সাইজ', 'এত', 'বড়', 'দিয়েছে', 'যে', 'আমার', 'একদম', 'হয়না', '।', 'পুরা', 'মুখই', 'ঢেকে', 'যায়', '।', 'এখন', 'কাউকে', 'দিয়ে', 'দিতে', 'হবে', '"']
    Truncating punctuation: ['কোয়ালিটি', 'সব', 'ঠিক', 'আছে', 'কিন্তু', 'সাইজ', 'এত', 'বড়', 'দিয়েছে', 'যে', 'আমার', 'একদম', 'হয়না', 'পুরা', 'মুখই', 'ঢেকে', 'যায়', 'এখন', 'কাউকে', 'দিয়ে', 'দিতে', 'হবে']
    Truncating StopWords: ['কোয়ালিটি', 'ঠিক', 'সাইজ', 'বড়', 'দিয়েছে', 'একদম', 'হয়না', 'পুরা', 'মুখই', 'ঢেকে', 'যায়', 'দিয়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  এত কম দামে একটি ভাল চসমা পেলাম খুব ভালো লাগতেছে।সার্ভিসম্যান অনেক ভালো ছিলো।
    Afert Tokenizing:  ['এত', 'কম', 'দামে', 'একটি', 'ভাল', 'চসমা', 'পেলাম', 'খুব', 'ভালো', 'লাগতেছে।সার্ভিসম্যান', 'অনেক', 'ভালো', 'ছিলো', '।']
    Truncating punctuation: ['এত', 'কম', 'দামে', 'একটি', 'ভাল', 'চসমা', 'পেলাম', 'খুব', 'ভালো', 'লাগতেছে।সার্ভিসম্যান', 'অনেক', 'ভালো', 'ছিলো']
    Truncating StopWords: ['কম', 'দামে', 'ভাল', 'চসমা', 'পেলাম', 'ভালো', 'লাগতেছে।সার্ভিসম্যান', 'ভালো', 'ছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  যেরকম আশা করেছিলাম তার চেয়েও বেশি ভালো
    Afert Tokenizing:  ['যেরকম', 'আশা', 'করেছিলাম', 'তার', 'চেয়েও', 'বেশি', 'ভালো']
    Truncating punctuation: ['যেরকম', 'আশা', 'করেছিলাম', 'তার', 'চেয়েও', 'বেশি', 'ভালো']
    Truncating StopWords: ['যেরকম', 'আশা', 'করেছিলাম', 'চেয়েও', 'বেশি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  "মেশ কাপড়ের।  পড়েও আরাম। দেখতেও খারাপ না। শুধু কাপড় পাতলা। "
    Afert Tokenizing:  ['মেশ', '"', 'কাপড়ের', '।', 'পড়েও', 'আরাম', '।', 'দেখতেও', 'খারাপ', 'না', '।', 'শুধু', 'কাপড়', 'পাতলা', '।', '', '"']
    Truncating punctuation: ['মেশ', 'কাপড়ের', 'পড়েও', 'আরাম', 'দেখতেও', 'খারাপ', 'না', 'শুধু', 'কাপড়', 'পাতলা', '']
    Truncating StopWords: ['মেশ', 'কাপড়ের', 'পড়েও', 'আরাম', 'দেখতেও', 'খারাপ', 'না', 'শুধু', 'কাপড়', 'পাতলা', '']
    ***************************************************************************************
    Label:  0
    Sentence:  "কেনার এক্সপেরিয়েন্স টা খুব বাজে ছিল, কারণ নির্দিষ্ট সময় থেকো তিন চার দিন দেরি,জিনিসটা দেখে ভালই লাগলো, ভালোই, ভেবেছিলাম লেস এমবোটারি করা পরে  দেখি প্রিন্ট করা"
    Afert Tokenizing:  ['কেনার', '"', 'এক্সপেরিয়েন্স', 'টা', 'খুব', 'বাজে', 'ছিল', ',', 'কারণ', 'নির্দিষ্ট', 'সময়', 'থেকো', 'তিন', 'চার', 'দিন', 'দেরি,জিনিসটা', 'দেখে', 'ভালই', 'লাগলো', ',', 'ভালোই', ',', 'ভেবেছিলাম', 'লেস', 'এমবোটারি', 'করা', 'পরে', 'দেখি', 'প্রিন্ট', 'করা', '"']
    Truncating punctuation: ['কেনার', 'এক্সপেরিয়েন্স', 'টা', 'খুব', 'বাজে', 'ছিল', 'কারণ', 'নির্দিষ্ট', 'সময়', 'থেকো', 'তিন', 'চার', 'দিন', 'দেরি,জিনিসটা', 'দেখে', 'ভালই', 'লাগলো', 'ভালোই', 'ভেবেছিলাম', 'লেস', 'এমবোটারি', 'করা', 'পরে', 'দেখি', 'প্রিন্ট', 'করা']
    Truncating StopWords: ['কেনার', 'এক্সপেরিয়েন্স', 'টা', 'বাজে', 'নির্দিষ্ট', 'সময়', 'থেকো', 'তিন', 'দেরি,জিনিসটা', 'ভালই', 'লাগলো', 'ভালোই', 'ভেবেছিলাম', 'লেস', 'এমবোটারি', 'দেখি', 'প্রিন্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যি বলতে মূল্য অনুযায়ী প্রোডাক্ট মাশাল্লাহ অনেক সুন্দর হয়েছে
    Afert Tokenizing:  ['সত্যি', 'বলতে', 'মূল্য', 'অনুযায়ী', 'প্রোডাক্ট', 'মাশাল্লাহ', 'অনেক', 'সুন্দর', 'হয়েছে']
    Truncating punctuation: ['সত্যি', 'বলতে', 'মূল্য', 'অনুযায়ী', 'প্রোডাক্ট', 'মাশাল্লাহ', 'অনেক', 'সুন্দর', 'হয়েছে']
    Truncating StopWords: ['সত্যি', 'মূল্য', 'প্রোডাক্ট', 'মাশাল্লাহ', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি সম্পন্ন প্রোডাক্ট দেওয়ার জন্য ধন্যবাদ
    Afert Tokenizing:  ['কোয়ালিটি', 'সম্পন্ন', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating punctuation: ['কোয়ালিটি', 'সম্পন্ন', 'প্রোডাক্ট', 'দেওয়ার', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['কোয়ালিটি', 'সম্পন্ন', 'প্রোডাক্ট', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  "মানের পণ্য, চমৎকার জিনিসপত্র। প্যাকেজিংও দুর্দান্ত ছিল"
    Afert Tokenizing:  ['মানের', '"', 'পণ্য', ',', 'চমৎকার', 'জিনিসপত্র', '।', 'প্যাকেজিংও', 'দুর্দান্ত', 'ছিল', '"']
    Truncating punctuation: ['মানের', 'পণ্য', 'চমৎকার', 'জিনিসপত্র', 'প্যাকেজিংও', 'দুর্দান্ত', 'ছিল']
    Truncating StopWords: ['মানের', 'পণ্য', 'চমৎকার', 'জিনিসপত্র', 'প্যাকেজিংও', 'দুর্দান্ত']
    ***************************************************************************************
    Label:  1
    Sentence:  স্নোবল ফেয়ারি লাইটস টা সত্যিই অসম্ভব সুন্দর। অনেক কোয়ালিটি ফুল। প্রতিটি লাইট ঠিক আছে
    Afert Tokenizing:  ['স্নোবল', 'ফেয়ারি', 'লাইটস', 'টা', 'সত্যিই', 'অসম্ভব', 'সুন্দর', '।', 'অনেক', 'কোয়ালিটি', 'ফুল', '।', 'প্রতিটি', 'লাইট', 'ঠিক', 'আছে']
    Truncating punctuation: ['স্নোবল', 'ফেয়ারি', 'লাইটস', 'টা', 'সত্যিই', 'অসম্ভব', 'সুন্দর', 'অনেক', 'কোয়ালিটি', 'ফুল', 'প্রতিটি', 'লাইট', 'ঠিক', 'আছে']
    Truncating StopWords: ['স্নোবল', 'ফেয়ারি', 'লাইটস', 'টা', 'সত্যিই', 'অসম্ভব', 'সুন্দর', 'কোয়ালিটি', 'ফুল', 'প্রতিটি', 'লাইট', 'ঠিক']
    ***************************************************************************************
    Label:  1
    Sentence:  "বাহ,অনেক ভালো লাগতেছে,য়েমন দেখেছি,তার চেয়েও সুন্দর লাগতেছে, এবং অনেকটা লম্বাও"
    Afert Tokenizing:  ['বাহ,অনেক', '"', 'ভালো', 'লাগতেছে,য়েমন', 'দেখেছি,তার', 'চেয়েও', 'সুন্দর', 'লাগতেছে', ',', 'এবং', 'অনেকটা', 'লম্বাও', '"']
    Truncating punctuation: ['বাহ,অনেক', 'ভালো', 'লাগতেছে,য়েমন', 'দেখেছি,তার', 'চেয়েও', 'সুন্দর', 'লাগতেছে', 'এবং', 'অনেকটা', 'লম্বাও']
    Truncating StopWords: ['বাহ,অনেক', 'ভালো', 'লাগতেছে,য়েমন', 'দেখেছি,তার', 'চেয়েও', 'সুন্দর', 'লাগতেছে', 'অনেকটা', 'লম্বাও']
    ***************************************************************************************
    Label:  1
    Sentence:  দামে কম মানে ভালো
    Afert Tokenizing:  ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating punctuation: ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating StopWords: ['দামে', 'কম', 'মানে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  জামাটি খুব ভালো গরমের সিজন ও দাম অনুযায়ি আপনারা চাইলে কিনতে পারেন
    Afert Tokenizing:  ['জামাটি', 'খুব', 'ভালো', 'গরমের', 'সিজন', 'ও', 'দাম', 'অনুযায়ি', 'আপনারা', 'চাইলে', 'কিনতে', 'পারেন']
    Truncating punctuation: ['জামাটি', 'খুব', 'ভালো', 'গরমের', 'সিজন', 'ও', 'দাম', 'অনুযায়ি', 'আপনারা', 'চাইলে', 'কিনতে', 'পারেন']
    Truncating StopWords: ['জামাটি', 'ভালো', 'গরমের', 'সিজন', 'দাম', 'অনুযায়ি', 'আপনারা', 'চাইলে', 'কিনতে']
    ***************************************************************************************
    Label:  1
    Sentence:  যেমনটা ছবিতে ঠিক তেমনই পেয়েছি
    Afert Tokenizing:  ['যেমনটা', 'ছবিতে', 'ঠিক', 'তেমনই', 'পেয়েছি']
    Truncating punctuation: ['যেমনটা', 'ছবিতে', 'ঠিক', 'তেমনই', 'পেয়েছি']
    Truncating StopWords: ['যেমনটা', 'ছবিতে', 'ঠিক', 'তেমনই', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি দুইটা নিয়েছি অনেক সুন্দর
    Afert Tokenizing:  ['আমি', 'দুইটা', 'নিয়েছি', 'অনেক', 'সুন্দর']
    Truncating punctuation: ['আমি', 'দুইটা', 'নিয়েছি', 'অনেক', 'সুন্দর']
    Truncating StopWords: ['দুইটা', 'নিয়েছি', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম হিসাবে। পাঞ্জাবিটা অনেক সুন্দর। হয়তো একটু গরম লাগে। তাতে কোনো সমস্যা নাই। দেখতে ভালো।
    Afert Tokenizing:  ['দাম', 'হিসাবে', '।', 'পাঞ্জাবিটা', 'অনেক', 'সুন্দর', '।', 'হয়তো', 'একটু', 'গরম', 'লাগে', '।', 'তাতে', 'কোনো', 'সমস্যা', 'নাই', '।', 'দেখতে', 'ভালো', '।']
    Truncating punctuation: ['দাম', 'হিসাবে', 'পাঞ্জাবিটা', 'অনেক', 'সুন্দর', 'হয়তো', 'একটু', 'গরম', 'লাগে', 'তাতে', 'কোনো', 'সমস্যা', 'নাই', 'দেখতে', 'ভালো']
    Truncating StopWords: ['দাম', 'পাঞ্জাবিটা', 'সুন্দর', 'একটু', 'গরম', 'লাগে', 'সমস্যা', 'নাই', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড়ের কোয়ালিটি ভাল ধাম আনুয়ায়ী সব কিছুই ঠিক আছে যে সাইজ চাইছিলাম তাই পাইছী
    Afert Tokenizing:  ['কাপড়ের', 'কোয়ালিটি', 'ভাল', 'ধাম', 'আনুয়ায়ী', 'সব', 'কিছুই', 'ঠিক', 'আছে', 'যে', 'সাইজ', 'চাইছিলাম', 'তাই', 'পাইছী']
    Truncating punctuation: ['কাপড়ের', 'কোয়ালিটি', 'ভাল', 'ধাম', 'আনুয়ায়ী', 'সব', 'কিছুই', 'ঠিক', 'আছে', 'যে', 'সাইজ', 'চাইছিলাম', 'তাই', 'পাইছী']
    Truncating StopWords: ['কাপড়ের', 'কোয়ালিটি', 'ভাল', 'ধাম', 'আনুয়ায়ী', 'ঠিক', 'সাইজ', 'চাইছিলাম', 'পাইছী']
    ***************************************************************************************
    Label:  1
    Sentence:  "সত্যি ২৯৯ টাকা তে অসাধারণ।  মার্কেটে একই জিনিস ৫০০/৬০০ চাইবে অনেক ভালো হয়েছে
    Afert Tokenizing:  ['সত্যি', '"', '২৯৯', 'টাকা', 'তে', 'অসাধারণ', '।', 'মার্কেটে', 'একই', 'জিনিস', '৫০০/৬০০', 'চাইবে', 'অনেক', 'ভালো', 'হয়েছে']
    Truncating punctuation: ['সত্যি', '২৯৯', 'টাকা', 'তে', 'অসাধারণ', 'মার্কেটে', 'একই', 'জিনিস', '৫০০/৬০০', 'চাইবে', 'অনেক', 'ভালো', 'হয়েছে']
    Truncating StopWords: ['সত্যি', '২৯৯', 'টাকা', 'তে', 'অসাধারণ', 'মার্কেটে', 'জিনিস', '৫০০/৬০০', 'চাইবে', 'ভালো', 'হয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:  একবারের বেশি পরতে পারিনি বাজে জিনিস
    Afert Tokenizing:  ['একবারের', 'বেশি', 'পরতে', 'পারিনি', 'বাজে', 'জিনিস']
    Truncating punctuation: ['একবারের', 'বেশি', 'পরতে', 'পারিনি', 'বাজে', 'জিনিস']
    Truncating StopWords: ['একবারের', 'বেশি', 'পরতে', 'পারিনি', 'বাজে', 'জিনিস']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম হিসেবে কাপড় এবং স্টাইল খুব ভালো
    Afert Tokenizing:  ['দাম', 'হিসেবে', 'কাপড়', 'এবং', 'স্টাইল', 'খুব', 'ভালো']
    Truncating punctuation: ['দাম', 'হিসেবে', 'কাপড়', 'এবং', 'স্টাইল', 'খুব', 'ভালো']
    Truncating StopWords: ['দাম', 'হিসেবে', 'কাপড়', 'স্টাইল', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মনে হয় যে মডেল দিয়েছে সেটা ২/৩ বার ধোয়ার পর উঠে যাওয়ার সম্ভাবনা আছে
    Afert Tokenizing:  ['আমার', 'মনে', 'হয়', 'যে', 'মডেল', 'দিয়েছে', 'সেটা', '২/৩', 'বার', 'ধোয়ার', 'পর', 'উঠে', 'যাওয়ার', 'সম্ভাবনা', 'আছে']
    Truncating punctuation: ['আমার', 'মনে', 'হয়', 'যে', 'মডেল', 'দিয়েছে', 'সেটা', '২/৩', 'বার', 'ধোয়ার', 'পর', 'উঠে', 'যাওয়ার', 'সম্ভাবনা', 'আছে']
    Truncating StopWords: ['মডেল', '২/৩', 'ধোয়ার', 'উঠে', 'সম্ভাবনা']
    ***************************************************************************************
    Label:  1
    Sentence:  ৩০০ টাকার মধ্যে এর থেকে ভাল কিছু পাওয়া যায় না। সুন্দর হয়েছে। আপনারা চাইলে নিতে পারেন
    Afert Tokenizing:  ['৩০০', 'টাকার', 'মধ্যে', 'এর', 'থেকে', 'ভাল', 'কিছু', 'পাওয়া', 'যায়', 'না', '।', 'সুন্দর', 'হয়েছে', '।', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating punctuation: ['৩০০', 'টাকার', 'মধ্যে', 'এর', 'থেকে', 'ভাল', 'কিছু', 'পাওয়া', 'যায়', 'না', 'সুন্দর', 'হয়েছে', 'আপনারা', 'চাইলে', 'নিতে', 'পারেন']
    Truncating StopWords: ['৩০০', 'টাকার', 'ভাল', 'না', 'সুন্দর', 'আপনারা', 'চাইলে']
    ***************************************************************************************
    Label:  0
    Sentence:  "মুটামুটি, পরতে গিয়ে বাটন খুলে গেসে  সেলায় ভালো সিল না"
    Afert Tokenizing:  ['"মুটামুটি', ',', 'পরতে', 'গিয়ে', 'বাটন', 'খুলে', 'গেসে', 'সেলায়', 'ভালো', 'সিল', 'না', '"']
    Truncating punctuation: ['"মুটামুটি', 'পরতে', 'গিয়ে', 'বাটন', 'খুলে', 'গেসে', 'সেলায়', 'ভালো', 'সিল', 'না']
    Truncating StopWords: ['"মুটামুটি', 'পরতে', 'বাটন', 'খুলে', 'গেসে', 'সেলায়', 'ভালো', 'সিল', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  "অনেক ভাল ছিল পান্জাবি টা,কম দামে ভাল জিনিস"
    Afert Tokenizing:  ['অনেক', '"', 'ভাল', 'ছিল', 'পান্জাবি', 'টা,কম', 'দামে', 'ভাল', 'জিনিস', '"']
    Truncating punctuation: ['অনেক', 'ভাল', 'ছিল', 'পান্জাবি', 'টা,কম', 'দামে', 'ভাল', 'জিনিস']
    Truncating StopWords: ['ভাল', 'পান্জাবি', 'টা,কম', 'দামে', 'ভাল', 'জিনিস']
    ***************************************************************************************
    Label:  1
    Sentence:  "জানিনা কার কাছে কেমন লাগবে, তবে আমার কাছে খুবই পারফেক্ট।"
    Afert Tokenizing:  ['জানিনা', '"', 'কার', 'কাছে', 'কেমন', 'লাগবে', ',', 'তবে', 'আমার', 'কাছে', 'খুবই', 'পারফেক্ট।', '"']
    Truncating punctuation: ['জানিনা', 'কার', 'কাছে', 'কেমন', 'লাগবে', 'তবে', 'আমার', 'কাছে', 'খুবই', 'পারফেক্ট।']
    Truncating StopWords: ['জানিনা', 'কার', 'কেমন', 'লাগবে', 'খুবই', 'পারফেক্ট।']
    ***************************************************************************************
    Label:  1
    Sentence:  "মানান সই, নিতে পারেন"
    Afert Tokenizing:  ['মানান', '"', 'সই', ',', 'নিতে', 'পারেন', '"']
    Truncating punctuation: ['মানান', 'সই', 'নিতে', 'পারেন']
    Truncating StopWords: ['মানান', 'সই']
    ***************************************************************************************
    Label:  1
    Sentence:  কোয়ালিটি দাম অনুযায়ী ঠিকই আছে
    Afert Tokenizing:  ['কোয়ালিটি', 'দাম', 'অনুযায়ী', 'ঠিকই', 'আছে']
    Truncating punctuation: ['কোয়ালিটি', 'দাম', 'অনুযায়ী', 'ঠিকই', 'আছে']
    Truncating StopWords: ['কোয়ালিটি', 'দাম', 'ঠিকই']
    ***************************************************************************************
    Label:  1
    Sentence:  "বকিছুই ঠিকঠাক আছে, সাইজটা উল্টাপাল্টা হয়ে গিয়েছে, যে সাইজটা পাঠানো হয়েছে এটা স্মল সাইজ, যদিও ট্যাগ লাগানো ছিল এল সাইজের।"
    Afert Tokenizing:  ['বকিছুই', '"', 'ঠিকঠাক', 'আছে', ',', 'সাইজটা', 'উল্টাপাল্টা', 'হয়ে', 'গিয়েছে', ',', 'যে', 'সাইজটা', 'পাঠানো', 'হয়েছে', 'এটা', 'স্মল', 'সাইজ', ',', 'যদিও', 'ট্যাগ', 'লাগানো', 'ছিল', 'এল', 'সাইজের।', '"']
    Truncating punctuation: ['বকিছুই', 'ঠিকঠাক', 'আছে', 'সাইজটা', 'উল্টাপাল্টা', 'হয়ে', 'গিয়েছে', 'যে', 'সাইজটা', 'পাঠানো', 'হয়েছে', 'এটা', 'স্মল', 'সাইজ', 'যদিও', 'ট্যাগ', 'লাগানো', 'ছিল', 'এল', 'সাইজের।']
    Truncating StopWords: ['বকিছুই', 'ঠিকঠাক', 'সাইজটা', 'উল্টাপাল্টা', 'সাইজটা', 'পাঠানো', 'স্মল', 'সাইজ', 'ট্যাগ', 'লাগানো', 'সাইজের।']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি কি অর্ডার করছি আর কি পেলাম।  মেজাজটাই খারাপ হয়ে গেছে
    Afert Tokenizing:  ['আমি', 'কি', 'অর্ডার', 'করছি', 'আর', 'কি', 'পেলাম', '।', 'মেজাজটাই', 'খারাপ', 'হয়ে', 'গেছে']
    Truncating punctuation: ['আমি', 'কি', 'অর্ডার', 'করছি', 'আর', 'কি', 'পেলাম', 'মেজাজটাই', 'খারাপ', 'হয়ে', 'গেছে']
    Truncating StopWords: ['অর্ডার', 'করছি', 'পেলাম', 'মেজাজটাই', 'খারাপ', 'হয়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ, যেটা ওর্ডার করছি সেটাই পাইছি।অল্প দামে খুবই ভালো একটা প্রোডাক্ট"
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ', ',', 'যেটা', 'ওর্ডার', 'করছি', 'সেটাই', 'পাইছি।অল্প', 'দামে', 'খুবই', 'ভালো', 'একটা', 'প্রোডাক্ট', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ', 'যেটা', 'ওর্ডার', 'করছি', 'সেটাই', 'পাইছি।অল্প', 'দামে', 'খুবই', 'ভালো', 'একটা', 'প্রোডাক্ট']
    Truncating StopWords: ['"আলহামদুলিল্লাহ', 'যেটা', 'ওর্ডার', 'করছি', 'পাইছি।অল্প', 'দামে', 'খুবই', 'ভালো', 'একটা', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার দিলাম কনভার্স দিলো লোফার কেমন হলো বলেন ভুল অর্ডার দিয়ে গেল কেন
    Afert Tokenizing:  ['অর্ডার', 'দিলাম', 'কনভার্স', 'দিলো', 'লোফার', 'কেমন', 'হলো', 'বলেন', 'ভুল', 'অর্ডার', 'দিয়ে', 'গেল', 'কেন']
    Truncating punctuation: ['অর্ডার', 'দিলাম', 'কনভার্স', 'দিলো', 'লোফার', 'কেমন', 'হলো', 'বলেন', 'ভুল', 'অর্ডার', 'দিয়ে', 'গেল', 'কেন']
    Truncating StopWords: ['অর্ডার', 'দিলাম', 'কনভার্স', 'দিলো', 'লোফার', 'কেমন', 'ভুল', 'অর্ডার']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম খারাপ প্রডাক্ট দিছে। ছবির সাথে কোনো মিল নাই
    Afert Tokenizing:  ['একদম', 'খারাপ', 'প্রডাক্ট', 'দিছে', '।', 'ছবির', 'সাথে', 'কোনো', 'মিল', 'নাই']
    Truncating punctuation: ['একদম', 'খারাপ', 'প্রডাক্ট', 'দিছে', 'ছবির', 'সাথে', 'কোনো', 'মিল', 'নাই']
    Truncating StopWords: ['একদম', 'খারাপ', 'প্রডাক্ট', 'দিছে', 'ছবির', 'সাথে', 'মিল', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:   আমি যেটা  পেয়েছি ওইটার কোনু মিল নেই আমি ৩৯ অর্ডার  দিছি আর ৪০ পাইছি
    Afert Tokenizing:  ['আমি', 'যেটা', 'পেয়েছি', 'ওইটার', 'কোনু', 'মিল', 'নেই', 'আমি', '৩৯', 'অর্ডার', 'দিছি', 'আর', '৪০', 'পাইছি']
    Truncating punctuation: ['আমি', 'যেটা', 'পেয়েছি', 'ওইটার', 'কোনু', 'মিল', 'নেই', 'আমি', '৩৯', 'অর্ডার', 'দিছি', 'আর', '৪০', 'পাইছি']
    Truncating StopWords: ['যেটা', 'পেয়েছি', 'ওইটার', 'কোনু', 'মিল', 'নেই', '৩৯', 'অর্ডার', 'দিছি', '৪০', 'পাইছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "বাটপার,চাইছি এক জিনিস দিয়েছে অন্য জিনিস,আমার টাকা টাই নষ্ট"
    Afert Tokenizing:  ['বাটপার,চাইছি', '"', 'এক', 'জিনিস', 'দিয়েছে', 'অন্য', 'জিনিস,আমার', 'টাকা', 'টাই', 'নষ্ট', '"']
    Truncating punctuation: ['বাটপার,চাইছি', 'এক', 'জিনিস', 'দিয়েছে', 'অন্য', 'জিনিস,আমার', 'টাকা', 'টাই', 'নষ্ট']
    Truncating StopWords: ['বাটপার,চাইছি', 'এক', 'জিনিস', 'দিয়েছে', 'জিনিস,আমার', 'টাকা', 'টাই', 'নষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  আমাকে অন্য একটি জুতা দেওয়া হয়েছে
    Afert Tokenizing:  ['আমাকে', 'অন্য', 'একটি', 'জুতা', 'দেওয়া', 'হয়েছে']
    Truncating punctuation: ['আমাকে', 'অন্য', 'একটি', 'জুতা', 'দেওয়া', 'হয়েছে']
    Truncating StopWords: ['জুতা', 'হয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই চাইছি হলুদ দিছে কালো এইটা কোন কাজ হইলো আমার হলুদ জুতা লাগবে
    Afert Tokenizing:  ['ভাই', 'চাইছি', 'হলুদ', 'দিছে', 'কালো', 'এইটা', 'কোন', 'কাজ', 'হইলো', 'আমার', 'হলুদ', 'জুতা', 'লাগবে']
    Truncating punctuation: ['ভাই', 'চাইছি', 'হলুদ', 'দিছে', 'কালো', 'এইটা', 'কোন', 'কাজ', 'হইলো', 'আমার', 'হলুদ', 'জুতা', 'লাগবে']
    Truncating StopWords: ['ভাই', 'চাইছি', 'হলুদ', 'দিছে', 'কালো', 'এইটা', 'হইলো', 'হলুদ', 'জুতা', 'লাগবে']
    ***************************************************************************************
    Label:  0
    Sentence:  জুতা ভালই হয়েছে কিন্তু যে গুলো চাইছিলাম অইগুলো দেয় নাই
    Afert Tokenizing:  ['জুতা', 'ভালই', 'হয়েছে', 'কিন্তু', 'যে', 'গুলো', 'চাইছিলাম', 'অইগুলো', 'দেয়', 'নাই']
    Truncating punctuation: ['জুতা', 'ভালই', 'হয়েছে', 'কিন্তু', 'যে', 'গুলো', 'চাইছিলাম', 'অইগুলো', 'দেয়', 'নাই']
    Truncating StopWords: ['জুতা', 'ভালই', 'হয়েছে', 'গুলো', 'চাইছিলাম', 'অইগুলো', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  চাইলাম কি আর পাইলাম কি খুব খারাপ একটা অভিজ্ঞতা
    Afert Tokenizing:  ['চাইলাম', 'কি', 'আর', 'পাইলাম', 'কি', 'খুব', 'খারাপ', 'একটা', 'অভিজ্ঞতা']
    Truncating punctuation: ['চাইলাম', 'কি', 'আর', 'পাইলাম', 'কি', 'খুব', 'খারাপ', 'একটা', 'অভিজ্ঞতা']
    Truncating StopWords: ['চাইলাম', 'পাইলাম', 'খারাপ', 'একটা', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি যে কালার দিয়েছি সেটা পাই নাই সাইজ টাও টিক না
    Afert Tokenizing:  ['আমি', 'যে', 'কালার', 'দিয়েছি', 'সেটা', 'পাই', 'নাই', 'সাইজ', 'টাও', 'টিক', 'না']
    Truncating punctuation: ['আমি', 'যে', 'কালার', 'দিয়েছি', 'সেটা', 'পাই', 'নাই', 'সাইজ', 'টাও', 'টিক', 'না']
    Truncating StopWords: ['কালার', 'দিয়েছি', 'পাই', 'নাই', 'সাইজ', 'টাও', 'টিক', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দিলাম সাদা সোল আর সাদা ফিতার আর আসলো পুরা কালো। এটা কোনো কথা হ্যা। পুরাই পালতু
    Afert Tokenizing:  ['দিলাম', 'সাদা', 'সোল', 'আর', 'সাদা', 'ফিতার', 'আর', 'আসলো', 'পুরা', 'কালো', '।', 'এটা', 'কোনো', 'কথা', 'হ্যা', '।', 'পুরাই', 'পালতু']
    Truncating punctuation: ['দিলাম', 'সাদা', 'সোল', 'আর', 'সাদা', 'ফিতার', 'আর', 'আসলো', 'পুরা', 'কালো', 'এটা', 'কোনো', 'কথা', 'হ্যা', 'পুরাই', 'পালতু']
    Truncating StopWords: ['দিলাম', 'সাদা', 'সোল', 'সাদা', 'ফিতার', 'আসলো', 'পুরা', 'কালো', 'কথা', 'হ্যা', 'পুরাই', 'পালতু']
    ***************************************************************************************
    Label:  0
    Sentence:  একেবারে ফালতু। তার উপর চেয়েছি ৪২ সাইজের   পাঠিয়েছে ৪৩ সাইজ
    Afert Tokenizing:  ['একেবারে', 'ফালতু', '।', 'তার', 'উপর', 'চেয়েছি', '৪২', 'সাইজের', 'পাঠিয়েছে', '৪৩', 'সাইজ']
    Truncating punctuation: ['একেবারে', 'ফালতু', 'তার', 'উপর', 'চেয়েছি', '৪২', 'সাইজের', 'পাঠিয়েছে', '৪৩', 'সাইজ']
    Truncating StopWords: ['একেবারে', 'ফালতু', 'চেয়েছি', '৪২', 'সাইজের', 'পাঠিয়েছে', '৪৩', 'সাইজ']
    ***************************************************************************************
    Label:  0
    Sentence:  "একটা অর্ডার করেছি অন্য একটা দিছে, মান খুবই খারাপ"
    Afert Tokenizing:  ['একটা', '"', 'অর্ডার', 'করেছি', 'অন্য', 'একটা', 'দিছে', ',', 'মান', 'খুবই', 'খারাপ', '"']
    Truncating punctuation: ['একটা', 'অর্ডার', 'করেছি', 'অন্য', 'একটা', 'দিছে', 'মান', 'খুবই', 'খারাপ']
    Truncating StopWords: ['একটা', 'অর্ডার', 'করেছি', 'একটা', 'দিছে', 'মান', 'খুবই', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে একটা পিরত দিয়ে দিছি এখনো টাকা দেয় নাই
    Afert Tokenizing:  ['বাজে', 'একটা', 'পিরত', 'দিয়ে', 'দিছি', 'এখনো', 'টাকা', 'দেয়', 'নাই']
    Truncating punctuation: ['বাজে', 'একটা', 'পিরত', 'দিয়ে', 'দিছি', 'এখনো', 'টাকা', 'দেয়', 'নাই']
    Truncating StopWords: ['বাজে', 'একটা', 'পিরত', 'দিয়ে', 'দিছি', 'এখনো', 'টাকা', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কি দিছেন চাইছি লাল দিছেন হলুদ এটাকি হল
    Afert Tokenizing:  ['কি', 'দিছেন', 'চাইছি', 'লাল', 'দিছেন', 'হলুদ', 'এটাকি', 'হল']
    Truncating punctuation: ['কি', 'দিছেন', 'চাইছি', 'লাল', 'দিছেন', 'হলুদ', 'এটাকি', 'হল']
    Truncating StopWords: ['দিছেন', 'চাইছি', 'লাল', 'দিছেন', 'হলুদ', 'এটাকি']
    ***************************************************************************************
    Label:  1
    Sentence:   দানের তুলনায় খুব ভালো জিনিস
    Afert Tokenizing:  ['দানের', 'তুলনায়', 'খুব', 'ভালো', 'জিনিস']
    Truncating punctuation: ['দানের', 'তুলনায়', 'খুব', 'ভালো', 'জিনিস']
    Truncating StopWords: ['দানের', 'তুলনায়', 'ভালো', 'জিনিস']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি তো আমার পণ্য ফেরত  দিতে চাই
    Afert Tokenizing:  ['আমি', 'তো', 'আমার', 'পণ্য', 'ফেরত', 'দিতে', 'চাই']
    Truncating punctuation: ['আমি', 'তো', 'আমার', 'পণ্য', 'ফেরত', 'দিতে', 'চাই']
    Truncating StopWords: ['পণ্য', 'ফেরত', 'চাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কালার ভুল দিয়েছেন
    Afert Tokenizing:  ['কালার', 'ভুল', 'দিয়েছেন']
    Truncating punctuation: ['কালার', 'ভুল', 'দিয়েছেন']
    Truncating StopWords: ['কালার', 'ভুল']
    ***************************************************************************************
    Label:  0
    Sentence:  কাজটা খুব খারাপ করছে  আমি আর কোন কিছু কিনবো নাহ ওডার দিছি কী আর দিছে কী
    Afert Tokenizing:  ['কাজটা', 'খুব', 'খারাপ', 'করছে', 'আমি', 'আর', 'কোন', 'কিছু', 'কিনবো', 'নাহ', 'ওডার', 'দিছি', 'কী', 'আর', 'দিছে', 'কী']
    Truncating punctuation: ['কাজটা', 'খুব', 'খারাপ', 'করছে', 'আমি', 'আর', 'কোন', 'কিছু', 'কিনবো', 'নাহ', 'ওডার', 'দিছি', 'কী', 'আর', 'দিছে', 'কী']
    Truncating StopWords: ['কাজটা', 'খারাপ', 'কিনবো', 'নাহ', 'ওডার', 'দিছি', 'দিছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকে প্রডাক্টটা পেয়ে আমি খুবই খুশি। কারণ এ বিক্রেতার  প্রডাক্ট এর প্যাকেজিং মান খুব ভাল ছিল
    Afert Tokenizing:  ['আজকে', 'প্রডাক্টটা', 'পেয়ে', 'আমি', 'খুবই', 'খুশি', '।', 'কারণ', 'এ', 'বিক্রেতার', 'প্রডাক্ট', 'এর', 'প্যাকেজিং', 'মান', 'খুব', 'ভাল', 'ছিল']
    Truncating punctuation: ['আজকে', 'প্রডাক্টটা', 'পেয়ে', 'আমি', 'খুবই', 'খুশি', 'কারণ', 'এ', 'বিক্রেতার', 'প্রডাক্ট', 'এর', 'প্যাকেজিং', 'মান', 'খুব', 'ভাল', 'ছিল']
    Truncating StopWords: ['আজকে', 'প্রডাক্টটা', 'পেয়ে', 'খুবই', 'খুশি', 'বিক্রেতার', 'প্রডাক্ট', 'প্যাকেজিং', 'মান', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  যেমনটা দেখেছি তেমনটাই পেয়েছি। কোনরকম প্রবলেম হয়নি
    Afert Tokenizing:  ['যেমনটা', 'দেখেছি', 'তেমনটাই', 'পেয়েছি', '।', 'কোনরকম', 'প্রবলেম', 'হয়নি']
    Truncating punctuation: ['যেমনটা', 'দেখেছি', 'তেমনটাই', 'পেয়েছি', 'কোনরকম', 'প্রবলেম', 'হয়নি']
    Truncating StopWords: ['যেমনটা', 'দেখেছি', 'তেমনটাই', 'পেয়েছি', 'কোনরকম', 'প্রবলেম']
    ***************************************************************************************
    Label:  1
    Sentence:  "বেশ ভাল, বোতামগুলি কিছুটা আলগা, এবং কিছুটা শক্তিশালী সুন্দর হবে"
    Afert Tokenizing:  ['বেশ', '"', 'ভাল', ',', 'বোতামগুলি', 'কিছুটা', 'আলগা', ',', 'এবং', 'কিছুটা', 'শক্তিশালী', 'সুন্দর', 'হবে', '"']
    Truncating punctuation: ['বেশ', 'ভাল', 'বোতামগুলি', 'কিছুটা', 'আলগা', 'এবং', 'কিছুটা', 'শক্তিশালী', 'সুন্দর', 'হবে']
    Truncating StopWords: ['ভাল', 'বোতামগুলি', 'কিছুটা', 'আলগা', 'কিছুটা', 'শক্তিশালী', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  "প্রোডাক্টটি অর্ডার করে অনেক কনফিউশনে ছিলাম যে কিনা কি দিয়ে দেয়, আসলে আমি যেমনটা ভেবেছিলাম সেরকম না প্রোডাক্টটা অনেক ভালো,"
    Afert Tokenizing:  ['প্রোডাক্টটি', '"', 'অর্ডার', 'করে', 'অনেক', 'কনফিউশনে', 'ছিলাম', 'যে', 'কিনা', 'কি', 'দিয়ে', 'দেয়', ',', 'আসলে', 'আমি', 'যেমনটা', 'ভেবেছিলাম', 'সেরকম', 'না', 'প্রোডাক্টটা', 'অনেক', 'ভালো,', '"']
    Truncating punctuation: ['প্রোডাক্টটি', 'অর্ডার', 'করে', 'অনেক', 'কনফিউশনে', 'ছিলাম', 'যে', 'কিনা', 'কি', 'দিয়ে', 'দেয়', 'আসলে', 'আমি', 'যেমনটা', 'ভেবেছিলাম', 'সেরকম', 'না', 'প্রোডাক্টটা', 'অনেক', 'ভালো,']
    Truncating StopWords: ['প্রোডাক্টটি', 'অর্ডার', 'কনফিউশনে', 'ছিলাম', 'কিনা', 'আসলে', 'যেমনটা', 'ভেবেছিলাম', 'সেরকম', 'না', 'প্রোডাক্টটা', 'ভালো,']
    ***************************************************************************************
    Label:  1
    Sentence:  "আমি নিজে পছন্দ করে কিছু কিনলে ঠকে যাই, এই ঘড়িটি হাতে পেয়ে মনে হলো এই প্রথম আমি কিছু কিনে জিতলাম"
    Afert Tokenizing:  ['আমি', '"', 'নিজে', 'পছন্দ', 'করে', 'কিছু', 'কিনলে', 'ঠকে', 'যাই', ',', 'এই', 'ঘড়িটি', 'হাতে', 'পেয়ে', 'মনে', 'হলো', 'এই', 'প্রথম', 'আমি', 'কিছু', 'কিনে', 'জিতলাম', '"']
    Truncating punctuation: ['আমি', 'নিজে', 'পছন্দ', 'করে', 'কিছু', 'কিনলে', 'ঠকে', 'যাই', 'এই', 'ঘড়িটি', 'হাতে', 'পেয়ে', 'মনে', 'হলো', 'এই', 'প্রথম', 'আমি', 'কিছু', 'কিনে', 'জিতলাম']
    Truncating StopWords: ['পছন্দ', 'কিনলে', 'ঠকে', 'যাই', 'ঘড়িটি', 'হাতে', 'কিনে', 'জিতলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  "ছোট ভাইকে গিফট দেওয়ার জন্য নিয়েছিলাম এবং তাকে দিয়েছি তার অনেক পছন্দ হয়েছে, ধন্যবাদ সেরার"
    Afert Tokenizing:  ['ছোট', '"', 'ভাইকে', 'গিফট', 'দেওয়ার', 'জন্য', 'নিয়েছিলাম', 'এবং', 'তাকে', 'দিয়েছি', 'তার', 'অনেক', 'পছন্দ', 'হয়েছে', ',', 'ধন্যবাদ', 'সেরার', '"']
    Truncating punctuation: ['ছোট', 'ভাইকে', 'গিফট', 'দেওয়ার', 'জন্য', 'নিয়েছিলাম', 'এবং', 'তাকে', 'দিয়েছি', 'তার', 'অনেক', 'পছন্দ', 'হয়েছে', 'ধন্যবাদ', 'সেরার']
    Truncating StopWords: ['ছোট', 'ভাইকে', 'গিফট', 'নিয়েছিলাম', 'দিয়েছি', 'পছন্দ', 'ধন্যবাদ', 'সেরার']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো প্রোডাক্ট টা এক কথায় অসাধারন। এই দামে এই ঘড়ি টা বেস্ট চয়েস হবে বলে আমি মনে করি
    Afert Tokenizing:  ['ভালো', 'প্রোডাক্ট', 'টা', 'এক', 'কথায়', 'অসাধারন', '।', 'এই', 'দামে', 'এই', 'ঘড়ি', 'টা', 'বেস্ট', 'চয়েস', 'হবে', 'বলে', 'আমি', 'মনে', 'করি']
    Truncating punctuation: ['ভালো', 'প্রোডাক্ট', 'টা', 'এক', 'কথায়', 'অসাধারন', 'এই', 'দামে', 'এই', 'ঘড়ি', 'টা', 'বেস্ট', 'চয়েস', 'হবে', 'বলে', 'আমি', 'মনে', 'করি']
    Truncating StopWords: ['ভালো', 'প্রোডাক্ট', 'টা', 'এক', 'কথায়', 'অসাধারন', 'দামে', 'ঘড়ি', 'টা', 'বেস্ট', 'চয়েস']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্যাকেট থেকে খোলার পরই উপরের কাচ খুলে পরে গেছে। খুবই হতাশাজনক"
    Afert Tokenizing:  ['প্যাকেট', '"', 'থেকে', 'খোলার', 'পরই', 'উপরের', 'কাচ', 'খুলে', 'পরে', 'গেছে', '।', 'খুবই', 'হতাশাজনক', '"']
    Truncating punctuation: ['প্যাকেট', 'থেকে', 'খোলার', 'পরই', 'উপরের', 'কাচ', 'খুলে', 'পরে', 'গেছে', 'খুবই', 'হতাশাজনক']
    Truncating StopWords: ['প্যাকেট', 'খোলার', 'পরই', 'উপরের', 'কাচ', 'খুলে', 'খুবই', 'হতাশাজনক']
    ***************************************************************************************
    Label:  0
    Sentence:  "দেখতে সুন্দর।হাত নাড়া চাড়া করলে ভিতরে ফিটিংস দূর্বলতা থাকার কারনে শব্দ হয়।
    Afert Tokenizing:  ['দেখতে', '"', 'সুন্দর।হাত', 'নাড়া', 'চাড়া', 'করলে', 'ভিতরে', 'ফিটিংস', 'দূর্বলতা', 'থাকার', 'কারনে', 'শব্দ', 'হয়', '।']
    Truncating punctuation: ['দেখতে', 'সুন্দর।হাত', 'নাড়া', 'চাড়া', 'করলে', 'ভিতরে', 'ফিটিংস', 'দূর্বলতা', 'থাকার', 'কারনে', 'শব্দ', 'হয়']
    Truncating StopWords: ['সুন্দর।হাত', 'নাড়া', 'চাড়া', 'ভিতরে', 'ফিটিংস', 'দূর্বলতা', 'থাকার', 'কারনে', 'শব্দ']
    ***************************************************************************************
    Label:  0
    Sentence:  "ঘড়িটা ভালো কিন্তু উপরের গেলাসটা লাগানো ছিলোনা, এটা আসা করিনি"
    Afert Tokenizing:  ['ঘড়িটা', '"', 'ভালো', 'কিন্তু', 'উপরের', 'গেলাসটা', 'লাগানো', 'ছিলোনা', ',', 'এটা', 'আসা', 'করিনি', '"']
    Truncating punctuation: ['ঘড়িটা', 'ভালো', 'কিন্তু', 'উপরের', 'গেলাসটা', 'লাগানো', 'ছিলোনা', 'এটা', 'আসা', 'করিনি']
    Truncating StopWords: ['ঘড়িটা', 'ভালো', 'উপরের', 'গেলাসটা', 'লাগানো', 'ছিলোনা', 'আসা', 'করিনি']
    ***************************************************************************************
    Label:  0
    Sentence:  কি ঘড়ি দিছেন ভাই টাইম দেখা যায়না।এখন কি করব বলেন
    Afert Tokenizing:  ['কি', 'ঘড়ি', 'দিছেন', 'ভাই', 'টাইম', 'দেখা', 'যায়না।এখন', 'কি', 'করব', 'বলেন']
    Truncating punctuation: ['কি', 'ঘড়ি', 'দিছেন', 'ভাই', 'টাইম', 'দেখা', 'যায়না।এখন', 'কি', 'করব', 'বলেন']
    Truncating StopWords: ['ঘড়ি', 'দিছেন', 'ভাই', 'টাইম', 'যায়না।এখন', 'করব']
    ***************************************************************************************
    Label:  1
    Sentence:  বেশ ভালো এবং মানসম্মত ঘড়ি
    Afert Tokenizing:  ['বেশ', 'ভালো', 'এবং', 'মানসম্মত', 'ঘড়ি']
    Truncating punctuation: ['বেশ', 'ভালো', 'এবং', 'মানসম্মত', 'ঘড়ি']
    Truncating StopWords: ['ভালো', 'মানসম্মত', 'ঘড়ি']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্টিভ হুবুহুব একদম ছবির সাথে মিল আছে পর একটি পেয়ে আমি অনেক খুশি
    Afert Tokenizing:  ['প্রডাক্টিভ', 'হুবুহুব', 'একদম', 'ছবির', 'সাথে', 'মিল', 'আছে', 'পর', 'একটি', 'পেয়ে', 'আমি', 'অনেক', 'খুশি']
    Truncating punctuation: ['প্রডাক্টিভ', 'হুবুহুব', 'একদম', 'ছবির', 'সাথে', 'মিল', 'আছে', 'পর', 'একটি', 'পেয়ে', 'আমি', 'অনেক', 'খুশি']
    Truncating StopWords: ['প্রডাক্টিভ', 'হুবুহুব', 'একদম', 'ছবির', 'সাথে', 'মিল', 'খুশি']
    ***************************************************************************************
    Label:  0
    Sentence:  চিটিং বাজ এরা। ২টা প্রোডাক্ট অর্ডার ছিলো। শুধু ১ টা প্রোডাক্ট এসেছে তাও নিম্নমানের
    Afert Tokenizing:  ['চিটিং', 'বাজ', 'এরা', '।', '২টা', 'প্রোডাক্ট', 'অর্ডার', 'ছিলো', '।', 'শুধু', '১', 'টা', 'প্রোডাক্ট', 'এসেছে', 'তাও', 'নিম্নমানের']
    Truncating punctuation: ['চিটিং', 'বাজ', 'এরা', '২টা', 'প্রোডাক্ট', 'অর্ডার', 'ছিলো', 'শুধু', '১', 'টা', 'প্রোডাক্ট', 'এসেছে', 'তাও', 'নিম্নমানের']
    Truncating StopWords: ['চিটিং', 'বাজ', '২টা', 'প্রোডাক্ট', 'অর্ডার', 'ছিলো', 'শুধু', '১', 'টা', 'প্রোডাক্ট', 'এসেছে', 'নিম্নমানের']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম  অনুযায়ী  হাতঘড়িটা  বেশ ভালো
    Afert Tokenizing:  ['দাম', 'অনুযায়ী', 'হাতঘড়িটা', 'বেশ', 'ভালো']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'হাতঘড়িটা', 'বেশ', 'ভালো']
    Truncating StopWords: ['দাম', 'অনুযায়ী', 'হাতঘড়িটা', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  "ভাই ঘড়ির মান অত্যান্ত খারাপ অবস্থা,সমস্থ পার্টস খুলা
    Afert Tokenizing:  ['ভাই', '"', 'ঘড়ির', 'মান', 'অত্যান্ত', 'খারাপ', 'অবস্থা,সমস্থ', 'পার্টস', 'খুলা']
    Truncating punctuation: ['ভাই', 'ঘড়ির', 'মান', 'অত্যান্ত', 'খারাপ', 'অবস্থা,সমস্থ', 'পার্টস', 'খুলা']
    Truncating StopWords: ['ভাই', 'ঘড়ির', 'মান', 'অত্যান্ত', 'খারাপ', 'অবস্থা,সমস্থ', 'পার্টস', 'খুলা']
    ***************************************************************************************
    Label:  1
    Sentence:  এই হেডফোনগুলি দুর্দান্ত। দাম অনুযায়ী সাউন্ড কোয়ালিটি খুব ভাল
    Afert Tokenizing:  ['এই', 'হেডফোনগুলি', 'দুর্দান্ত', '।', 'দাম', 'অনুযায়ী', 'সাউন্ড', 'কোয়ালিটি', 'খুব', 'ভাল']
    Truncating punctuation: ['এই', 'হেডফোনগুলি', 'দুর্দান্ত', 'দাম', 'অনুযায়ী', 'সাউন্ড', 'কোয়ালিটি', 'খুব', 'ভাল']
    Truncating StopWords: ['হেডফোনগুলি', 'দুর্দান্ত', 'দাম', 'সাউন্ড', 'কোয়ালিটি', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  যেমন দেখেছিলাম তেমনটাই পেয়েছি আর সাউন্ড সিস্টেম সব কিছুই ঠিক আছে একদম পারফেক্ট
    Afert Tokenizing:  ['যেমন', 'দেখেছিলাম', 'তেমনটাই', 'পেয়েছি', 'আর', 'সাউন্ড', 'সিস্টেম', 'সব', 'কিছুই', 'ঠিক', 'আছে', 'একদম', 'পারফেক্ট']
    Truncating punctuation: ['যেমন', 'দেখেছিলাম', 'তেমনটাই', 'পেয়েছি', 'আর', 'সাউন্ড', 'সিস্টেম', 'সব', 'কিছুই', 'ঠিক', 'আছে', 'একদম', 'পারফেক্ট']
    Truncating StopWords: ['দেখেছিলাম', 'তেমনটাই', 'পেয়েছি', 'সাউন্ড', 'সিস্টেম', 'ঠিক', 'একদম', 'পারফেক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে জিনিস ছাউন্ড ক্লাসিক তেমন ভাল না
    Afert Tokenizing:  ['বাজে', 'জিনিস', 'ছাউন্ড', 'ক্লাসিক', 'তেমন', 'ভাল', 'না']
    Truncating punctuation: ['বাজে', 'জিনিস', 'ছাউন্ড', 'ক্লাসিক', 'তেমন', 'ভাল', 'না']
    Truncating StopWords: ['বাজে', 'জিনিস', 'ছাউন্ড', 'ক্লাসিক', 'ভাল', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  একসেট বা দুই সেট কাপড় বহনের জন্য পারফেক্ট।
    Afert Tokenizing:  ['একসেট', 'বা', 'দুই', 'সেট', 'কাপড়', 'বহনের', 'জন্য', 'পারফেক্ট', '।']
    Truncating punctuation: ['একসেট', 'বা', 'দুই', 'সেট', 'কাপড়', 'বহনের', 'জন্য', 'পারফেক্ট']
    Truncating StopWords: ['একসেট', 'সেট', 'কাপড়', 'বহনের', 'পারফেক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ, পণ্যটি খুব ভাল। তবে এটি খুব বেশি স্থিতিস্থাপক নয়...।"
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ', ',', 'পণ্যটি', 'খুব', 'ভাল', '।', 'তবে', 'এটি', 'খুব', 'বেশি', 'স্থিতিস্থাপক', 'নয়...।', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ', 'পণ্যটি', 'খুব', 'ভাল', 'তবে', 'এটি', 'খুব', 'বেশি', 'স্থিতিস্থাপক', 'নয়...।']
    Truncating StopWords: ['"আলহামদুলিল্লাহ', 'পণ্যটি', 'ভাল', 'বেশি', 'স্থিতিস্থাপক', 'নয়...।']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ এবং খুব ই কমফোর্টেবল এবং ঠিক সাইজ পেয়েছি। এছাড়া ডেলিভারিও খুব দ্রুত পেয়েছি
    Afert Tokenizing:  ['অসাধারণ', 'এবং', 'খুব', 'ই', 'কমফোর্টেবল', 'এবং', 'ঠিক', 'সাইজ', 'পেয়েছি', '।', 'এছাড়া', 'ডেলিভারিও', 'খুব', 'দ্রুত', 'পেয়েছি']
    Truncating punctuation: ['অসাধারণ', 'এবং', 'খুব', 'ই', 'কমফোর্টেবল', 'এবং', 'ঠিক', 'সাইজ', 'পেয়েছি', 'এছাড়া', 'ডেলিভারিও', 'খুব', 'দ্রুত', 'পেয়েছি']
    Truncating StopWords: ['অসাধারণ', 'কমফোর্টেবল', 'ঠিক', 'সাইজ', 'পেয়েছি', 'এছাড়া', 'ডেলিভারিও', 'দ্রুত', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ই ভালো লেগেছে আমার কাছে
    Afert Tokenizing:  ['খুব', 'ই', 'ভালো', 'লেগেছে', 'আমার', 'কাছে']
    Truncating punctuation: ['খুব', 'ই', 'ভালো', 'লেগেছে', 'আমার', 'কাছে']
    Truncating StopWords: ['ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা বাজে মাল
    Afert Tokenizing:  ['এটা', 'বাজে', 'মাল']
    Truncating punctuation: ['এটা', 'বাজে', 'মাল']
    Truncating StopWords: ['বাজে', 'মাল']
    ***************************************************************************************
    Label:  1
    Sentence:  টাওয়েল টা ভাল লেগেছে।
    Afert Tokenizing:  ['টাওয়েল', 'টা', 'ভাল', 'লেগেছে', '।']
    Truncating punctuation: ['টাওয়েল', 'টা', 'ভাল', 'লেগেছে']
    Truncating StopWords: ['টাওয়েল', 'টা', 'ভাল', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ঠিক যেমনটা বিজ্ঞাপন দেওয়া হয়েছে। কোনো ত্রুটি নেই... গুণমান গড়
    Afert Tokenizing:  ['ঠিক', 'যেমনটা', 'বিজ্ঞাপন', 'দেওয়া', 'হয়েছে', '।', 'কোনো', 'ত্রুটি', 'নেই..', '.', 'গুণমান', 'গড়']
    Truncating punctuation: ['ঠিক', 'যেমনটা', 'বিজ্ঞাপন', 'দেওয়া', 'হয়েছে', 'কোনো', 'ত্রুটি', 'নেই..', 'গুণমান', 'গড়']
    Truncating StopWords: ['ঠিক', 'যেমনটা', 'বিজ্ঞাপন', 'ত্রুটি', 'নেই..', 'গুণমান', 'গড়']
    ***************************************************************************************
    Label:  1
    Sentence:  ছবি দিতে পারলাম না ব্যবহার করে ফেলছি কিন্তু অভিযোগ করার কোন কারন নাই ছবি তে যে রকম ঠিক সে রকম ই পাইছি
    Afert Tokenizing:  ['ছবি', 'দিতে', 'পারলাম', 'না', 'ব্যবহার', 'করে', 'ফেলছি', 'কিন্তু', 'অভিযোগ', 'করার', 'কোন', 'কারন', 'নাই', 'ছবি', 'তে', 'যে', 'রকম', 'ঠিক', 'সে', 'রকম', 'ই', 'পাইছি']
    Truncating punctuation: ['ছবি', 'দিতে', 'পারলাম', 'না', 'ব্যবহার', 'করে', 'ফেলছি', 'কিন্তু', 'অভিযোগ', 'করার', 'কোন', 'কারন', 'নাই', 'ছবি', 'তে', 'যে', 'রকম', 'ঠিক', 'সে', 'রকম', 'ই', 'পাইছি']
    Truncating StopWords: ['ছবি', 'পারলাম', 'না', 'ফেলছি', 'অভিযোগ', 'কারন', 'নাই', 'ছবি', 'তে', 'ঠিক', 'পাইছি']
    ***************************************************************************************
    Label:  1
    Sentence:  "ছেলে টা অনেক ভালো,, ব্যাবহার অনেক ভালো, সময় মতোন পন্য হাতে পৌছে দেয়।"
    Afert Tokenizing:  ['ছেলে', '"', 'টা', 'অনেক', 'ভালো,', ',', 'ব্যাবহার', 'অনেক', 'ভালো', ',', 'সময়', 'মতোন', 'পন্য', 'হাতে', 'পৌছে', 'দেয়।', '"']
    Truncating punctuation: ['ছেলে', 'টা', 'অনেক', 'ভালো,', 'ব্যাবহার', 'অনেক', 'ভালো', 'সময়', 'মতোন', 'পন্য', 'হাতে', 'পৌছে', 'দেয়।']
    Truncating StopWords: ['ছেলে', 'টা', 'ভালো,', 'ব্যাবহার', 'ভালো', 'সময়', 'মতোন', 'পন্য', 'হাতে', 'পৌছে', 'দেয়।']
    ***************************************************************************************
    Label:  1
    Sentence:  এই তাওয়াল টি অনেক ভালো বিশেষ করে গাড়ি মোছার জন্য।
    Afert Tokenizing:  ['এই', 'তাওয়াল', 'টি', 'অনেক', 'ভালো', 'বিশেষ', 'করে', 'গাড়ি', 'মোছার', 'জন্য', '।']
    Truncating punctuation: ['এই', 'তাওয়াল', 'টি', 'অনেক', 'ভালো', 'বিশেষ', 'করে', 'গাড়ি', 'মোছার', 'জন্য']
    Truncating StopWords: ['তাওয়াল', 'ভালো', 'বিশেষ', 'গাড়ি', 'মোছার']
    ***************************************************************************************
    Label:  0
    Sentence:  "একদম বাজে মাল খুভই পাতলা এবং কোয়ালিটি খারাপ"
    Afert Tokenizing:  ['একদম', '"', 'বাজে', 'মাল', 'খুভই', 'পাতলা', 'এবং', 'কোয়ালিটি', 'খারাপ', '"']
    Truncating punctuation: ['একদম', 'বাজে', 'মাল', 'খুভই', 'পাতলা', 'এবং', 'কোয়ালিটি', 'খারাপ']
    Truncating StopWords: ['একদম', 'বাজে', 'মাল', 'খুভই', 'পাতলা', 'কোয়ালিটি', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো কিন্তু অন্য কালার ডেলিভারি দিছে
    Afert Tokenizing:  ['ভালো', 'কিন্তু', 'অন্য', 'কালার', 'ডেলিভারি', 'দিছে']
    Truncating punctuation: ['ভালো', 'কিন্তু', 'অন্য', 'কালার', 'ডেলিভারি', 'দিছে']
    Truncating StopWords: ['ভালো', 'কালার', 'ডেলিভারি', 'দিছে']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো পানি এবং ময়লা পরিষ্কার করে
    Afert Tokenizing:  ['খুব', 'ভালো', 'পানি', 'এবং', 'ময়লা', 'পরিষ্কার', 'করে']
    Truncating punctuation: ['খুব', 'ভালো', 'পানি', 'এবং', 'ময়লা', 'পরিষ্কার', 'করে']
    Truncating StopWords: ['ভালো', 'পানি', 'ময়লা', 'পরিষ্কার']
    ***************************************************************************************
    Label:  1
    Sentence:  "সাইজ ছোট,মানে ভালোই"
    Afert Tokenizing:  ['সাইজ', '"', 'ছোট,মানে', 'ভালোই', '"']
    Truncating punctuation: ['সাইজ', 'ছোট,মানে', 'ভালোই']
    Truncating StopWords: ['সাইজ', 'ছোট,মানে', 'ভালোই']
    ***************************************************************************************
    Label:  0
    Sentence:  তেমন ভালো না।
    Afert Tokenizing:  ['তেমন', 'ভালো', 'না', '।']
    Truncating punctuation: ['তেমন', 'ভালো', 'না']
    Truncating StopWords: ['ভালো', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  "পণ্য সবসময়ের মতই খাঁটি। একটি সাবান বিনামূল্যে পেয়েছি। এটি ছাড়ের মূল্যে পেয়েছে, তাই দাম যুক্তিসঙ্গত ছিল এবং ডেলিভারি দ্রুত ছিল।"
    Afert Tokenizing:  ['পণ্য', '"', 'সবসময়ের', 'মতই', 'খাঁটি', '।', 'একটি', 'সাবান', 'বিনামূল্যে', 'পেয়েছি', '।', 'এটি', 'ছাড়ের', 'মূল্যে', 'পেয়েছে', ',', 'তাই', 'দাম', 'যুক্তিসঙ্গত', 'ছিল', 'এবং', 'ডেলিভারি', 'দ্রুত', 'ছিল।', '"']
    Truncating punctuation: ['পণ্য', 'সবসময়ের', 'মতই', 'খাঁটি', 'একটি', 'সাবান', 'বিনামূল্যে', 'পেয়েছি', 'এটি', 'ছাড়ের', 'মূল্যে', 'পেয়েছে', 'তাই', 'দাম', 'যুক্তিসঙ্গত', 'ছিল', 'এবং', 'ডেলিভারি', 'দ্রুত', 'ছিল।']
    Truncating StopWords: ['পণ্য', 'সবসময়ের', 'মতই', 'খাঁটি', 'সাবান', 'বিনামূল্যে', 'পেয়েছি', 'ছাড়ের', 'মূল্যে', 'পেয়েছে', 'দাম', 'যুক্তিসঙ্গত', 'ডেলিভারি', 'দ্রুত', 'ছিল।']
    ***************************************************************************************
    Label:  1
    Sentence:  বক্স করে অনেক ভালো করে সুন্দর করে দিয়েছে একটুও তেল পরেনি আর সাথে একটা সাবান গিফট ছিল।
    Afert Tokenizing:  ['বক্স', 'করে', 'অনেক', 'ভালো', 'করে', 'সুন্দর', 'করে', 'দিয়েছে', 'একটুও', 'তেল', 'পরেনি', 'আর', 'সাথে', 'একটা', 'সাবান', 'গিফট', 'ছিল', '।']
    Truncating punctuation: ['বক্স', 'করে', 'অনেক', 'ভালো', 'করে', 'সুন্দর', 'করে', 'দিয়েছে', 'একটুও', 'তেল', 'পরেনি', 'আর', 'সাথে', 'একটা', 'সাবান', 'গিফট', 'ছিল']
    Truncating StopWords: ['বক্স', 'ভালো', 'সুন্দর', 'একটুও', 'তেল', 'পরেনি', 'সাথে', 'একটা', 'সাবান', 'গিফট']
    ***************************************************************************************
    Label:  1
    Sentence:  "কোনও অভিযোগ নেই। বিনামূল্যে সাবান সহ সবাই ঠিক আছে, যদিও সব একই গন্ধ ছিল।"
    Afert Tokenizing:  ['কোনও', '"', 'অভিযোগ', 'নেই', '।', 'বিনামূল্যে', 'সাবান', 'সহ', 'সবাই', 'ঠিক', 'আছে', ',', 'যদিও', 'সব', 'একই', 'গন্ধ', 'ছিল।', '"']
    Truncating punctuation: ['কোনও', 'অভিযোগ', 'নেই', 'বিনামূল্যে', 'সাবান', 'সহ', 'সবাই', 'ঠিক', 'আছে', 'যদিও', 'সব', 'একই', 'গন্ধ', 'ছিল।']
    Truncating StopWords: ['অভিযোগ', 'নেই', 'বিনামূল্যে', 'সাবান', 'সবাই', 'ঠিক', 'গন্ধ', 'ছিল।']
    ***************************************************************************************
    Label:  1
    Sentence:  "৩য় বার অডার করলাম, সুন্দর করে প্যাকেট করে পাঠানো জন্য ধন্যবাদ"
    Afert Tokenizing:  ['৩য়', '"', 'বার', 'অডার', 'করলাম', ',', 'সুন্দর', 'করে', 'প্যাকেট', 'করে', 'পাঠানো', 'জন্য', 'ধন্যবাদ', '"']
    Truncating punctuation: ['৩য়', 'বার', 'অডার', 'করলাম', 'সুন্দর', 'করে', 'প্যাকেট', 'করে', 'পাঠানো', 'জন্য', 'ধন্যবাদ']
    Truncating StopWords: ['৩য়', 'অডার', 'করলাম', 'সুন্দর', 'প্যাকেট', 'পাঠানো', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  "সুন্দর প্যাকিং, ভালো অবস্থা য় পেয়েছি"
    Afert Tokenizing:  ['সুন্দর', '"', 'প্যাকিং', ',', 'ভালো', 'অবস্থা', 'য়', 'পেয়েছি', '"']
    Truncating punctuation: ['সুন্দর', 'প্যাকিং', 'ভালো', 'অবস্থা', 'য়', 'পেয়েছি']
    Truncating StopWords: ['সুন্দর', 'প্যাকিং', 'ভালো', 'অবস্থা', 'য়', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:   দাম টা আগের থেকে এটু বেশি
    Afert Tokenizing:  ['দাম', 'টা', 'আগের', 'থেকে', 'এটু', 'বেশি']
    Truncating punctuation: ['দাম', 'টা', 'আগের', 'থেকে', 'এটু', 'বেশি']
    Truncating StopWords: ['দাম', 'টা', 'আগের', 'এটু', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  "ভালভাবে পেলাম,ব্যবহার করা হয়নি"
    Afert Tokenizing:  ['ভালভাবে', '"', 'পেলাম,ব্যবহার', 'করা', 'হয়নি', '"']
    Truncating punctuation: ['ভালভাবে', 'পেলাম,ব্যবহার', 'করা', 'হয়নি']
    Truncating StopWords: ['ভালভাবে', 'পেলাম,ব্যবহার', 'হয়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  "অনেক ভালো, আবার নিবো"
    Afert Tokenizing:  ['অনেক', '"', 'ভালো', ',', 'আবার', 'নিবো', '"']
    Truncating punctuation: ['অনেক', 'ভালো', 'আবার', 'নিবো']
    Truncating StopWords: ['ভালো', 'নিবো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকেজিং ভালো ছিলো সাথে ফ্রী ডেলিভারি
    Afert Tokenizing:  ['প্যাকেজিং', 'ভালো', 'ছিলো', 'সাথে', 'ফ্রী', 'ডেলিভারি']
    Truncating punctuation: ['প্যাকেজিং', 'ভালো', 'ছিলো', 'সাথে', 'ফ্রী', 'ডেলিভারি']
    Truncating StopWords: ['প্যাকেজিং', 'ভালো', 'ছিলো', 'সাথে', 'ফ্রী', 'ডেলিভারি']
    ***************************************************************************************
    Label:  1
    Sentence:   লোকাল বাজার থেকে কম দামে পেয়েছি ডেলিভারি ও দ্রুত পেয়েছি
    Afert Tokenizing:  ['লোকাল', 'বাজার', 'থেকে', 'কম', 'দামে', 'পেয়েছি', 'ডেলিভারি', 'ও', 'দ্রুত', 'পেয়েছি']
    Truncating punctuation: ['লোকাল', 'বাজার', 'থেকে', 'কম', 'দামে', 'পেয়েছি', 'ডেলিভারি', 'ও', 'দ্রুত', 'পেয়েছি']
    Truncating StopWords: ['লোকাল', 'বাজার', 'কম', 'দামে', 'পেয়েছি', 'ডেলিভারি', 'দ্রুত', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "এটা কিনবেন না, মশা মারার জন্য কাজ করবেন না। সম্পূর্ণ অর্থের অপচয়।"
    Afert Tokenizing:  ['এটা', '"', 'কিনবেন', 'না', ',', 'মশা', 'মারার', 'জন্য', 'কাজ', 'করবেন', 'না', '।', 'সম্পূর্ণ', 'অর্থের', 'অপচয়।', '"']
    Truncating punctuation: ['এটা', 'কিনবেন', 'না', 'মশা', 'মারার', 'জন্য', 'কাজ', 'করবেন', 'না', 'সম্পূর্ণ', 'অর্থের', 'অপচয়।']
    Truncating StopWords: ['কিনবেন', 'না', 'মশা', 'মারার', 'না', 'সম্পূর্ণ', 'অর্থের', 'অপচয়।']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল প্রোডাক্ট পেয়েছি। ইন্ট্যাক্ট প্যাকেট এবং প্যাকেজিং ভাল ছিল।
    Afert Tokenizing:  ['ভাল', 'প্রোডাক্ট', 'পেয়েছি', '।', 'ইন্ট্যাক্ট', 'প্যাকেট', 'এবং', 'প্যাকেজিং', 'ভাল', 'ছিল', '।']
    Truncating punctuation: ['ভাল', 'প্রোডাক্ট', 'পেয়েছি', 'ইন্ট্যাক্ট', 'প্যাকেট', 'এবং', 'প্যাকেজিং', 'ভাল', 'ছিল']
    Truncating StopWords: ['ভাল', 'প্রোডাক্ট', 'পেয়েছি', 'ইন্ট্যাক্ট', 'প্যাকেট', 'প্যাকেজিং', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবই ভালো প্রোডাক্ট অরজিনাল প্রোডাক্ট দিয়েছে।
    Afert Tokenizing:  ['খুবই', 'ভালো', 'প্রোডাক্ট', 'অরজিনাল', 'প্রোডাক্ট', 'দিয়েছে', '।']
    Truncating punctuation: ['খুবই', 'ভালো', 'প্রোডাক্ট', 'অরজিনাল', 'প্রোডাক্ট', 'দিয়েছে']
    Truncating StopWords: ['খুবই', 'ভালো', 'প্রোডাক্ট', 'অরজিনাল', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  "শালারা ধোকাবাজ,,,খালি প্যাকেট দিয়েছ"
    Afert Tokenizing:  ['শালারা', '"', 'ধোকাবাজ,,,খালি', 'প্যাকেট', 'দিয়েছ', '"']
    Truncating punctuation: ['শালারা', 'ধোকাবাজ,,,খালি', 'প্যাকেট', 'দিয়েছ']
    Truncating StopWords: ['শালারা', 'ধোকাবাজ,,,খালি', 'প্যাকেট', 'দিয়েছ']
    ***************************************************************************************
    Label:  1
    Sentence:  জেনুইন প্রডাক্ট। ৩য় বার নিলাম
    Afert Tokenizing:  ['জেনুইন', 'প্রডাক্ট', '।', '৩য়', 'বার', 'নিলাম']
    Truncating punctuation: ['জেনুইন', 'প্রডাক্ট', '৩য়', 'বার', 'নিলাম']
    Truncating StopWords: ['জেনুইন', 'প্রডাক্ট', '৩য়', 'নিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  দামের তুলনায় বেশি ভালো
    Afert Tokenizing:  ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    Truncating punctuation: ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    Truncating StopWords: ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দামে কম মানে ভালো
    Afert Tokenizing:  ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating punctuation: ['দামে', 'কম', 'মানে', 'ভালো']
    Truncating StopWords: ['দামে', 'কম', 'মানে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দামের তুলনায় বেশি ভালো
    Afert Tokenizing:  ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    Truncating punctuation: ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    Truncating StopWords: ['দামের', 'তুলনায়', 'বেশি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  "দারুন ভালো জিনিস , দাম অনুসারে অনেক ভালো"
    Afert Tokenizing:  ['দারুন', '"', 'ভালো', 'জিনিস', '', ',', 'দাম', 'অনুসারে', 'অনেক', 'ভালো', '"']
    Truncating punctuation: ['দারুন', 'ভালো', 'জিনিস', '', 'দাম', 'অনুসারে', 'অনেক', 'ভালো']
    Truncating StopWords: ['দারুন', 'ভালো', 'জিনিস', '', 'দাম', 'অনুসারে', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  সেম্পল পন্যের সাথে আমি যা রিসিভ করলাম তার কি কোন মিল আছে?! আমি খুব মর্মাহত হয়েছি এতো নিম্ন মানের পন্য পেয়ে
    Afert Tokenizing:  ['সেম্পল', 'পন্যের', 'সাথে', 'আমি', 'যা', 'রিসিভ', 'করলাম', 'তার', 'কি', 'কোন', 'মিল', 'আছে?', '!', 'আমি', 'খুব', 'মর্মাহত', 'হয়েছি', 'এতো', 'নিম্ন', 'মানের', 'পন্য', 'পেয়ে']
    Truncating punctuation: ['সেম্পল', 'পন্যের', 'সাথে', 'আমি', 'যা', 'রিসিভ', 'করলাম', 'তার', 'কি', 'কোন', 'মিল', 'আছে?', 'আমি', 'খুব', 'মর্মাহত', 'হয়েছি', 'এতো', 'নিম্ন', 'মানের', 'পন্য', 'পেয়ে']
    Truncating StopWords: ['সেম্পল', 'পন্যের', 'সাথে', 'রিসিভ', 'করলাম', 'মিল', 'আছে?', 'মর্মাহত', 'হয়েছি', 'এতো', 'নিম্ন', 'মানের', 'পন্য', 'পেয়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  যেমন টা ছবিতে দেখানো হয়েছে মুজা গুলো তেমন না। যেটা দেওয়া হয়েছে এগুলো তেমন ভালো না।
    Afert Tokenizing:  ['যেমন', 'টা', 'ছবিতে', 'দেখানো', 'হয়েছে', 'মুজা', 'গুলো', 'তেমন', 'না', '।', 'যেটা', 'দেওয়া', 'হয়েছে', 'এগুলো', 'তেমন', 'ভালো', 'না', '।']
    Truncating punctuation: ['যেমন', 'টা', 'ছবিতে', 'দেখানো', 'হয়েছে', 'মুজা', 'গুলো', 'তেমন', 'না', 'যেটা', 'দেওয়া', 'হয়েছে', 'এগুলো', 'তেমন', 'ভালো', 'না']
    Truncating StopWords: ['টা', 'ছবিতে', 'দেখানো', 'মুজা', 'গুলো', 'না', 'যেটা', 'এগুলো', 'ভালো', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  "কালারও সেইম, মানও ভালো। আপনারাও নিতে পারেন।"
    Afert Tokenizing:  ['কালারও', '"', 'সেইম', ',', 'মানও', 'ভালো', '।', 'আপনারাও', 'নিতে', 'পারেন।', '"']
    Truncating punctuation: ['কালারও', 'সেইম', 'মানও', 'ভালো', 'আপনারাও', 'নিতে', 'পারেন।']
    Truncating StopWords: ['কালারও', 'সেইম', 'মানও', 'ভালো', 'আপনারাও', 'পারেন।']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপার । সব সময় এক রকম হয়/ ছবিতে দেখায় একরকম দেয় আরেক রকম।
    Afert Tokenizing:  ['বাটপার', '', '।', 'সব', 'সময়', 'এক', 'রকম', 'হয়/', 'ছবিতে', 'দেখায়', 'একরকম', 'দেয়', 'আরেক', 'রকম', '।']
    Truncating punctuation: ['বাটপার', '', 'সব', 'সময়', 'এক', 'রকম', 'হয়/', 'ছবিতে', 'দেখায়', 'একরকম', 'দেয়', 'আরেক', 'রকম']
    Truncating StopWords: ['বাটপার', '', 'সময়', 'এক', 'হয়/', 'ছবিতে', 'দেখায়', 'একরকম', 'আরেক']
    ***************************************************************************************
    Label:  1
    Sentence:  "সুন্দর জিনিস , চাইলে নিতে পারেন সবাই ।"
    Afert Tokenizing:  ['সুন্দর', '"', 'জিনিস', '', ',', 'চাইলে', 'নিতে', 'পারেন', 'সবাই', '।', '"']
    Truncating punctuation: ['সুন্দর', 'জিনিস', '', 'চাইলে', 'নিতে', 'পারেন', 'সবাই']
    Truncating StopWords: ['সুন্দর', 'জিনিস', '', 'চাইলে', 'সবাই']
    ***************************************************************************************
    Label:  1
    Sentence:  বিশ্বস্ত এবং গুনগত মানের । ১৩৫ টাকায় এমন জিনিস পাওয়া যাচ্ছে ভাবা যায় তাও আবার ৫ জোড়া ।
    Afert Tokenizing:  ['বিশ্বস্ত', 'এবং', 'গুনগত', 'মানের', '', '।', '১৩৫', 'টাকায়', 'এমন', 'জিনিস', 'পাওয়া', 'যাচ্ছে', 'ভাবা', 'যায়', 'তাও', 'আবার', '৫', 'জোড়া', '', '।']
    Truncating punctuation: ['বিশ্বস্ত', 'এবং', 'গুনগত', 'মানের', '', '১৩৫', 'টাকায়', 'এমন', 'জিনিস', 'পাওয়া', 'যাচ্ছে', 'ভাবা', 'যায়', 'তাও', 'আবার', '৫', 'জোড়া', '']
    Truncating StopWords: ['বিশ্বস্ত', 'গুনগত', 'মানের', '', '১৩৫', 'টাকায়', 'জিনিস', 'পাওয়া', 'ভাবা', 'যায়', '৫', 'জোড়া', '']
    ***************************************************************************************
    Label:  0
    Sentence:  যেমন কালার বিজ্ঞাপন দিলেন সেই রকম  পাইনি এগুলো এলোমেলো মাল দিয়ে কাষ্টমার সঙ্গে  বেঈমানী করা  সেটা ঠিক  করলেন আপনারা
    Afert Tokenizing:  ['যেমন', 'কালার', 'বিজ্ঞাপন', 'দিলেন', 'সেই', 'রকম', 'পাইনি', 'এগুলো', 'এলোমেলো', 'মাল', 'দিয়ে', 'কাষ্টমার', 'সঙ্গে', 'বেঈমানী', 'করা', 'সেটা', 'ঠিক', 'করলেন', 'আপনারা']
    Truncating punctuation: ['যেমন', 'কালার', 'বিজ্ঞাপন', 'দিলেন', 'সেই', 'রকম', 'পাইনি', 'এগুলো', 'এলোমেলো', 'মাল', 'দিয়ে', 'কাষ্টমার', 'সঙ্গে', 'বেঈমানী', 'করা', 'সেটা', 'ঠিক', 'করলেন', 'আপনারা']
    Truncating StopWords: ['কালার', 'বিজ্ঞাপন', 'পাইনি', 'এগুলো', 'এলোমেলো', 'মাল', 'দিয়ে', 'কাষ্টমার', 'বেঈমানী', 'ঠিক', 'আপনারা']
    ***************************************************************************************
    Label:  1
    Sentence:  দীর্ঘ দুই মাস পর ব্যবহার করার পরে আমি  রিভিউ শেয়ার করলাম। অনেক ভালমানের একটা প্রোডাক্ট
    Afert Tokenizing:  ['দীর্ঘ', 'দুই', 'মাস', 'পর', 'ব্যবহার', 'করার', 'পরে', 'আমি', 'রিভিউ', 'শেয়ার', 'করলাম', '।', 'অনেক', 'ভালমানের', 'একটা', 'প্রোডাক্ট']
    Truncating punctuation: ['দীর্ঘ', 'দুই', 'মাস', 'পর', 'ব্যবহার', 'করার', 'পরে', 'আমি', 'রিভিউ', 'শেয়ার', 'করলাম', 'অনেক', 'ভালমানের', 'একটা', 'প্রোডাক্ট']
    Truncating StopWords: ['দীর্ঘ', 'মাস', 'রিভিউ', 'শেয়ার', 'করলাম', 'ভালমানের', 'একটা', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রাপ্ত পণ্য কিন্তু নিম্নমানের এবং ত্রুটিপূর্ণ 5 পিস সেটের মধ্যে 2 জোড়া
    Afert Tokenizing:  ['প্রাপ্ত', 'পণ্য', 'কিন্তু', 'নিম্নমানের', 'এবং', 'ত্রুটিপূর্ণ', '5', 'পিস', 'সেটের', 'মধ্যে', '2', 'জোড়া']
    Truncating punctuation: ['প্রাপ্ত', 'পণ্য', 'কিন্তু', 'নিম্নমানের', 'এবং', 'ত্রুটিপূর্ণ', '5', 'পিস', 'সেটের', 'মধ্যে', '2', 'জোড়া']
    Truncating StopWords: ['প্রাপ্ত', 'পণ্য', 'নিম্নমানের', 'ত্রুটিপূর্ণ', '5', 'পিস', 'সেটের', '2', 'জোড়া']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম হিসাবে অনেক সুন্দর হয়েছে অনেক ভালো প্যাকিং ও ভালো ছিল সবদিক থেকে ভালো আছে আপনারা নিতে পারেন।
    Afert Tokenizing:  ['দাম', 'হিসাবে', 'অনেক', 'সুন্দর', 'হয়েছে', 'অনেক', 'ভালো', 'প্যাকিং', 'ও', 'ভালো', 'ছিল', 'সবদিক', 'থেকে', 'ভালো', 'আছে', 'আপনারা', 'নিতে', 'পারেন', '।']
    Truncating punctuation: ['দাম', 'হিসাবে', 'অনেক', 'সুন্দর', 'হয়েছে', 'অনেক', 'ভালো', 'প্যাকিং', 'ও', 'ভালো', 'ছিল', 'সবদিক', 'থেকে', 'ভালো', 'আছে', 'আপনারা', 'নিতে', 'পারেন']
    Truncating StopWords: ['দাম', 'সুন্দর', 'ভালো', 'প্যাকিং', 'ভালো', 'সবদিক', 'ভালো', 'আপনারা']
    ***************************************************************************************
    Label:  1
    Sentence:  "প্রায় এক মাস পরে রিভিউ দিলাম, দাম অনুযায়ী মোজা গুলো ভালো ছিল।"
    Afert Tokenizing:  ['প্রায়', '"', 'এক', 'মাস', 'পরে', 'রিভিউ', 'দিলাম', ',', 'দাম', 'অনুযায়ী', 'মোজা', 'গুলো', 'ভালো', 'ছিল।', '"']
    Truncating punctuation: ['প্রায়', 'এক', 'মাস', 'পরে', 'রিভিউ', 'দিলাম', 'দাম', 'অনুযায়ী', 'মোজা', 'গুলো', 'ভালো', 'ছিল।']
    Truncating StopWords: ['এক', 'মাস', 'রিভিউ', 'দিলাম', 'দাম', 'মোজা', 'গুলো', 'ভালো', 'ছিল।']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনারা নিতে পারেন চোখ বন্ধ করে
    Afert Tokenizing:  ['আপনারা', 'নিতে', 'পারেন', 'চোখ', 'বন্ধ', 'করে']
    Truncating punctuation: ['আপনারা', 'নিতে', 'পারেন', 'চোখ', 'বন্ধ', 'করে']
    Truncating StopWords: ['আপনারা', 'চোখ', 'বন্ধ']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্রোডাক্টের কোয়ালিটি ভালো না, সেলার প্রডাক্ট আমাকে চেক করে দেয়নি রিজেক্ট পেয়েছি"
    Afert Tokenizing:  ['প্রোডাক্টের', '"', 'কোয়ালিটি', 'ভালো', 'না', ',', 'সেলার', 'প্রডাক্ট', 'আমাকে', 'চেক', 'করে', 'দেয়নি', 'রিজেক্ট', 'পেয়েছি', '"']
    Truncating punctuation: ['প্রোডাক্টের', 'কোয়ালিটি', 'ভালো', 'না', 'সেলার', 'প্রডাক্ট', 'আমাকে', 'চেক', 'করে', 'দেয়নি', 'রিজেক্ট', 'পেয়েছি']
    Truncating StopWords: ['প্রোডাক্টের', 'কোয়ালিটি', 'ভালো', 'না', 'সেলার', 'প্রডাক্ট', 'চেক', 'দেয়নি', 'রিজেক্ট', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  "একদম অরিজিনাল,অনেক ভাল পোডাক্ট ।         "
    Afert Tokenizing:  ['একদম', '"', 'অরিজিনাল,অনেক', 'ভাল', 'পোডাক্ট', '', '।', '', '"']
    Truncating punctuation: ['একদম', 'অরিজিনাল,অনেক', 'ভাল', 'পোডাক্ট', '', '']
    Truncating StopWords: ['একদম', 'অরিজিনাল,অনেক', 'ভাল', 'পোডাক্ট', '', '']
    ***************************************************************************************
    Label:  1
    Sentence:  এটাকে বলা হয় সততার ব্যবসা। আপনাকে অনেক বিক্রেতা এবং পিকাবো ধন্যবাদ
    Afert Tokenizing:  ['এটাকে', 'বলা', 'হয়', 'সততার', 'ব্যবসা', '।', 'আপনাকে', 'অনেক', 'বিক্রেতা', 'এবং', 'পিকাবো', 'ধন্যবাদ']
    Truncating punctuation: ['এটাকে', 'বলা', 'হয়', 'সততার', 'ব্যবসা', 'আপনাকে', 'অনেক', 'বিক্রেতা', 'এবং', 'পিকাবো', 'ধন্যবাদ']
    Truncating StopWords: ['এটাকে', 'সততার', 'ব্যবসা', 'আপনাকে', 'বিক্রেতা', 'পিকাবো', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  এটা আমার প্রত্যাশা হচ্ছে. যে দাম সঙ্গে যেমন গুণ অবিশ্বাস্য
    Afert Tokenizing:  ['এটা', 'আমার', 'প্রত্যাশা', 'হচ্ছে', '.', 'যে', 'দাম', 'সঙ্গে', 'যেমন', 'গুণ', 'অবিশ্বাস্য']
    Truncating punctuation: ['এটা', 'আমার', 'প্রত্যাশা', 'হচ্ছে', 'যে', 'দাম', 'সঙ্গে', 'যেমন', 'গুণ', 'অবিশ্বাস্য']
    Truncating StopWords: ['প্রত্যাশা', 'দাম', 'গুণ', 'অবিশ্বাস্য']
    ***************************************************************************************
    Label:  1
    Sentence:  "মুজাগুলা খুব সুন্দর মাশাল্লাহ। ধন্যবাদ সেলার ভাইয়াকে পিকাবো ধন্যবাদ"
    Afert Tokenizing:  ['মুজাগুলা', '"', 'খুব', 'সুন্দর', 'মাশাল্লাহ', '।', 'ধন্যবাদ', 'সেলার', 'ভাইয়াকে', 'পিকাবো', 'ধন্যবাদ', '"']
    Truncating punctuation: ['মুজাগুলা', 'খুব', 'সুন্দর', 'মাশাল্লাহ', 'ধন্যবাদ', 'সেলার', 'ভাইয়াকে', 'পিকাবো', 'ধন্যবাদ']
    Truncating StopWords: ['মুজাগুলা', 'সুন্দর', 'মাশাল্লাহ', 'ধন্যবাদ', 'সেলার', 'ভাইয়াকে', 'পিকাবো', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম কনসিডার করলে জোড়া মাত্র ৩০ টাকা করে পড়েছে যা খুব ই ভালো
    Afert Tokenizing:  ['দাম', 'কনসিডার', 'করলে', 'জোড়া', 'মাত্র', '৩০', 'টাকা', 'করে', 'পড়েছে', 'যা', 'খুব', 'ই', 'ভালো']
    Truncating punctuation: ['দাম', 'কনসিডার', 'করলে', 'জোড়া', 'মাত্র', '৩০', 'টাকা', 'করে', 'পড়েছে', 'যা', 'খুব', 'ই', 'ভালো']
    Truncating StopWords: ['দাম', 'কনসিডার', 'জোড়া', '৩০', 'টাকা', 'পড়েছে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দেখতে তো ভালোই মনে হয়।ব্যবহারে পরে বুঝা যাবে বাকিটা।
    Afert Tokenizing:  ['দেখতে', 'তো', 'ভালোই', 'মনে', 'হয়।ব্যবহারে', 'পরে', 'বুঝা', 'যাবে', 'বাকিটা', '।']
    Truncating punctuation: ['দেখতে', 'তো', 'ভালোই', 'মনে', 'হয়।ব্যবহারে', 'পরে', 'বুঝা', 'যাবে', 'বাকিটা']
    Truncating StopWords: ['ভালোই', 'হয়।ব্যবহারে', 'বুঝা', 'বাকিটা']
    ***************************************************************************************
    Label:  1
    Sentence:  মোজা গুলো ভালই হয়েছে । আমার কাছে খুব ভালো লাগছে
    Afert Tokenizing:  ['মোজা', 'গুলো', 'ভালই', 'হয়েছে', '', '।', 'আমার', 'কাছে', 'খুব', 'ভালো', 'লাগছে']
    Truncating punctuation: ['মোজা', 'গুলো', 'ভালই', 'হয়েছে', '', 'আমার', 'কাছে', 'খুব', 'ভালো', 'লাগছে']
    Truncating StopWords: ['মোজা', 'গুলো', 'ভালই', '', 'ভালো', 'লাগছে']
    ***************************************************************************************
    Label:  1
    Sentence:  "ভালো,,কিন্তু আমার পায়ে লুজ হয়"
    Afert Tokenizing:  ['ভালো,,কিন্তু', '"', 'আমার', 'পায়ে', 'লুজ', 'হয়', '"']
    Truncating punctuation: ['ভালো,,কিন্তু', 'আমার', 'পায়ে', 'লুজ', 'হয়']
    Truncating StopWords: ['ভালো,,কিন্তু', 'পায়ে', 'লুজ']
    ***************************************************************************************
    Label:  0
    Sentence:  ভুয়া কোম্পানি কেউ টাকা দিয়ে ঠকবেন না
    Afert Tokenizing:  ['ভুয়া', 'কোম্পানি', 'কেউ', 'টাকা', 'দিয়ে', 'ঠকবেন', 'না']
    Truncating punctuation: ['ভুয়া', 'কোম্পানি', 'কেউ', 'টাকা', 'দিয়ে', 'ঠকবেন', 'না']
    Truncating StopWords: ['ভুয়া', 'কোম্পানি', 'টাকা', 'ঠকবেন', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে পণ্য অর্ডার করে আমার টাকা শেষ হয়ে গেল
    Afert Tokenizing:  ['বাজে', 'পণ্য', 'অর্ডার', 'করে', 'আমার', 'টাকা', 'শেষ', 'হয়ে', 'গেল']
    Truncating punctuation: ['বাজে', 'পণ্য', 'অর্ডার', 'করে', 'আমার', 'টাকা', 'শেষ', 'হয়ে', 'গেল']
    Truncating StopWords: ['বাজে', 'পণ্য', 'অর্ডার', 'টাকা', 'শেষ']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের অনলাইন সাইট থেকে না ফ্যান নষ্ট হয়ে গেছে
    Afert Tokenizing:  ['আপনাদের', 'অনলাইন', 'সাইট', 'থেকে', 'না', 'ফ্যান', 'নষ্ট', 'হয়ে', 'গেছে']
    Truncating punctuation: ['আপনাদের', 'অনলাইন', 'সাইট', 'থেকে', 'না', 'ফ্যান', 'নষ্ট', 'হয়ে', 'গেছে']
    Truncating StopWords: ['আপনাদের', 'অনলাইন', 'সাইট', 'না', 'ফ্যান', 'নষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা ফেরত দাও এবং নষ্ট করলে ফেরত দাও
    Afert Tokenizing:  ['আমার', 'টাকা', 'ফেরত', 'দাও', 'এবং', 'নষ্ট', 'করলে', 'ফেরত', 'দাও']
    Truncating punctuation: ['আমার', 'টাকা', 'ফেরত', 'দাও', 'এবং', 'নষ্ট', 'করলে', 'ফেরত', 'দাও']
    Truncating StopWords: ['টাকা', 'ফেরত', 'দাও', 'নষ্ট', 'ফেরত', 'দাও']
    ***************************************************************************************
    Label:  0
    Sentence:  এই বিজনেস ছেড়ে মাছের টা করেন ভুয়া
    Afert Tokenizing:  ['এই', 'বিজনেস', 'ছেড়ে', 'মাছের', 'টা', 'করেন', 'ভুয়া']
    Truncating punctuation: ['এই', 'বিজনেস', 'ছেড়ে', 'মাছের', 'টা', 'করেন', 'ভুয়া']
    Truncating StopWords: ['বিজনেস', 'ছেড়ে', 'মাছের', 'টা', 'ভুয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম হিসাবে এক কথায় জঘন্য
    Afert Tokenizing:  ['দাম', 'হিসাবে', 'এক', 'কথায়', 'জঘন্য']
    Truncating punctuation: ['দাম', 'হিসাবে', 'এক', 'কথায়', 'জঘন্য']
    Truncating StopWords: ['দাম', 'এক', 'কথায়', 'জঘন্য']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইন থেকে ফ্যান কিনে প্রতারিত হয়েছি
    Afert Tokenizing:  ['অনলাইন', 'থেকে', 'ফ্যান', 'কিনে', 'প্রতারিত', 'হয়েছি']
    Truncating punctuation: ['অনলাইন', 'থেকে', 'ফ্যান', 'কিনে', 'প্রতারিত', 'হয়েছি']
    Truncating StopWords: ['অনলাইন', 'ফ্যান', 'কিনে', 'প্রতারিত', 'হয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  সাত দিন হয়ে গেল এখনো ডেলিভারি দেন নাই আবার টাকা ফেরত দেন
    Afert Tokenizing:  ['সাত', 'দিন', 'হয়ে', 'গেল', 'এখনো', 'ডেলিভারি', 'দেন', 'নাই', 'আবার', 'টাকা', 'ফেরত', 'দেন']
    Truncating punctuation: ['সাত', 'দিন', 'হয়ে', 'গেল', 'এখনো', 'ডেলিভারি', 'দেন', 'নাই', 'আবার', 'টাকা', 'ফেরত', 'দেন']
    Truncating StopWords: ['সাত', 'এখনো', 'ডেলিভারি', 'নাই', 'টাকা', 'ফেরত']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড়ের মান খুবই নরম এবং পরতে আরামদায়ক।
    Afert Tokenizing:  ['কাপড়ের', 'মান', 'খুবই', 'নরম', 'এবং', 'পরতে', 'আরামদায়ক', '।']
    Truncating punctuation: ['কাপড়ের', 'মান', 'খুবই', 'নরম', 'এবং', 'পরতে', 'আরামদায়ক']
    Truncating StopWords: ['কাপড়ের', 'মান', 'খুবই', 'নরম', 'পরতে', 'আরামদায়ক']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রডাক্টের মান খুবই বাজে।  পিকাবোমানুষের পকেট মারা ধরছে। ভক্তা অধিদপ্তর কে জানাবো
    Afert Tokenizing:  ['প্রডাক্টের', 'মান', 'খুবই', 'বাজে', '।', 'পিকাবোমানুষের', 'পকেট', 'মারা', 'ধরছে', '।', 'ভক্তা', 'অধিদপ্তর', 'কে', 'জানাবো']
    Truncating punctuation: ['প্রডাক্টের', 'মান', 'খুবই', 'বাজে', 'পিকাবোমানুষের', 'পকেট', 'মারা', 'ধরছে', 'ভক্তা', 'অধিদপ্তর', 'কে', 'জানাবো']
    Truncating StopWords: ['প্রডাক্টের', 'মান', 'খুবই', 'বাজে', 'পিকাবোমানুষের', 'পকেট', 'মারা', 'ধরছে', 'ভক্তা', 'অধিদপ্তর', 'জানাবো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকিং ভালো ছিল ডেট কোডিং ভিতরে এবং বাহিরে মিল আছে প্যাকেজিং ভালো হয়েছে
    Afert Tokenizing:  ['প্যাকিং', 'ভালো', 'ছিল', 'ডেট', 'কোডিং', 'ভিতরে', 'এবং', 'বাহিরে', 'মিল', 'আছে', 'প্যাকেজিং', 'ভালো', 'হয়েছে']
    Truncating punctuation: ['প্যাকিং', 'ভালো', 'ছিল', 'ডেট', 'কোডিং', 'ভিতরে', 'এবং', 'বাহিরে', 'মিল', 'আছে', 'প্যাকেজিং', 'ভালো', 'হয়েছে']
    Truncating StopWords: ['প্যাকিং', 'ভালো', 'ডেট', 'কোডিং', 'ভিতরে', 'বাহিরে', 'মিল', 'প্যাকেজিং', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব বাজে প্রডাক্ট এখানে অর্ডার করার চেয়ে  আমাদের গাজিপুরে ফুটপাতে ভাল পাওয়া যায় প্রডাক্টে ময়লা মহিলাদের চুল  এমনটা আশা করি নাই
    Afert Tokenizing:  ['খুব', 'বাজে', 'প্রডাক্ট', 'এখানে', 'অর্ডার', 'করার', 'চেয়ে', 'আমাদের', 'গাজিপুরে', 'ফুটপাতে', 'ভাল', 'পাওয়া', 'যায়', 'প্রডাক্টে', 'ময়লা', 'মহিলাদের', 'চুল', 'এমনটা', 'আশা', 'করি', 'নাই']
    Truncating punctuation: ['খুব', 'বাজে', 'প্রডাক্ট', 'এখানে', 'অর্ডার', 'করার', 'চেয়ে', 'আমাদের', 'গাজিপুরে', 'ফুটপাতে', 'ভাল', 'পাওয়া', 'যায়', 'প্রডাক্টে', 'ময়লা', 'মহিলাদের', 'চুল', 'এমনটা', 'আশা', 'করি', 'নাই']
    Truncating StopWords: ['বাজে', 'প্রডাক্ট', 'অর্ডার', 'চেয়ে', 'গাজিপুরে', 'ফুটপাতে', 'ভাল', 'পাওয়া', 'যায়', 'প্রডাক্টে', 'ময়লা', 'মহিলাদের', 'চুল', 'এমনটা', 'আশা', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  "দাম অনুযায়ী পন্য অনেক ভালো ,সেলাই অনেক ভালো ছিল পরে দেখলাম সাইজ একদম পারফেক্ট ও খুব আরামদায়ক"
    Afert Tokenizing:  ['দাম', '"', 'অনুযায়ী', 'পন্য', 'অনেক', 'ভালো', 'সেলাই', ',', 'অনেক', 'ভালো', 'ছিল', 'পরে', 'দেখলাম', 'সাইজ', 'একদম', 'পারফেক্ট', 'ও', 'খুব', 'আরামদায়ক', '"']
    Truncating punctuation: ['দাম', 'অনুযায়ী', 'পন্য', 'অনেক', 'ভালো', 'সেলাই', 'অনেক', 'ভালো', 'ছিল', 'পরে', 'দেখলাম', 'সাইজ', 'একদম', 'পারফেক্ট', 'ও', 'খুব', 'আরামদায়ক']
    Truncating StopWords: ['দাম', 'পন্য', 'ভালো', 'সেলাই', 'ভালো', 'দেখলাম', 'সাইজ', 'একদম', 'পারফেক্ট', 'আরামদায়ক']
    ***************************************************************************************
    Label:  1
    Sentence:  মোটামুটি। ডেলিভারি চার্জ সহ ২২০ টাকা লাগসে টোটাল। এই বাজেটে ভালই
    Afert Tokenizing:  ['মোটামুটি', '।', 'ডেলিভারি', 'চার্জ', 'সহ', '২২০', 'টাকা', 'লাগসে', 'টোটাল', '।', 'এই', 'বাজেটে', 'ভালই']
    Truncating punctuation: ['মোটামুটি', 'ডেলিভারি', 'চার্জ', 'সহ', '২২০', 'টাকা', 'লাগসে', 'টোটাল', 'এই', 'বাজেটে', 'ভালই']
    Truncating StopWords: ['মোটামুটি', 'ডেলিভারি', 'চার্জ', '২২০', 'টাকা', 'লাগসে', 'টোটাল', 'বাজেটে', 'ভালই']
    ***************************************************************************************
    Label:  0
    Sentence:  ছবির সাথে মিল নেই ছবিতে দেখা যাচ্ছে পুরোটাই কালো দিসে সুইচ গুলো দেখি গোলাপী
    Afert Tokenizing:  ['ছবির', 'সাথে', 'মিল', 'নেই', 'ছবিতে', 'দেখা', 'যাচ্ছে', 'পুরোটাই', 'কালো', 'দিসে', 'সুইচ', 'গুলো', 'দেখি', 'গোলাপী']
    Truncating punctuation: ['ছবির', 'সাথে', 'মিল', 'নেই', 'ছবিতে', 'দেখা', 'যাচ্ছে', 'পুরোটাই', 'কালো', 'দিসে', 'সুইচ', 'গুলো', 'দেখি', 'গোলাপী']
    Truncating StopWords: ['ছবির', 'সাথে', 'মিল', 'নেই', 'ছবিতে', 'পুরোটাই', 'কালো', 'দিসে', 'সুইচ', 'গুলো', 'দেখি', 'গোলাপী']
    ***************************************************************************************
    Label:  1
    Sentence:  বলার কোনো ভাষা নাই অসাধারণ হইছে  কম টাকায় খুব ভালো একটি পোডাক্
    Afert Tokenizing:  ['বলার', 'কোনো', 'ভাষা', 'নাই', 'অসাধারণ', 'হইছে', 'কম', 'টাকায়', 'খুব', 'ভালো', 'একটি', 'পোডাক্']
    Truncating punctuation: ['বলার', 'কোনো', 'ভাষা', 'নাই', 'অসাধারণ', 'হইছে', 'কম', 'টাকায়', 'খুব', 'ভালো', 'একটি', 'পোডাক্']
    Truncating StopWords: ['বলার', 'ভাষা', 'নাই', 'অসাধারণ', 'হইছে', 'কম', 'টাকায়', 'ভালো', 'পোডাক্']
    ***************************************************************************************
    Label:  1
    Sentence:  এই মূল্য সীমার সাথে তুলনা করুন পণ্যের গুণমান যতটা ভাল ততটা।
    Afert Tokenizing:  ['এই', 'মূল্য', 'সীমার', 'সাথে', 'তুলনা', 'করুন', 'পণ্যের', 'গুণমান', 'যতটা', 'ভাল', 'ততটা', '।']
    Truncating punctuation: ['এই', 'মূল্য', 'সীমার', 'সাথে', 'তুলনা', 'করুন', 'পণ্যের', 'গুণমান', 'যতটা', 'ভাল', 'ততটা']
    Truncating StopWords: ['মূল্য', 'সীমার', 'সাথে', 'তুলনা', 'করুন', 'পণ্যের', 'গুণমান', 'ভাল', 'ততটা']
    ***************************************************************************************
    Label:  0
    Sentence:  বর্তমানে অনলাইন শপিং এ বিশ্বাস ই করা যাইনা। একটু দ্বিধা নিয়েই অর্ডার টা করেছিলাম।
    Afert Tokenizing:  ['বর্তমানে', 'অনলাইন', 'শপিং', 'এ', 'বিশ্বাস', 'ই', 'করা', 'যাইনা', '।', 'একটু', 'দ্বিধা', 'নিয়েই', 'অর্ডার', 'টা', 'করেছিলাম', '।']
    Truncating punctuation: ['বর্তমানে', 'অনলাইন', 'শপিং', 'এ', 'বিশ্বাস', 'ই', 'করা', 'যাইনা', 'একটু', 'দ্বিধা', 'নিয়েই', 'অর্ডার', 'টা', 'করেছিলাম']
    Truncating StopWords: ['বর্তমানে', 'অনলাইন', 'শপিং', 'বিশ্বাস', 'যাইনা', 'একটু', 'দ্বিধা', 'নিয়েই', 'অর্ডার', 'টা', 'করেছিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ ভালোই কিন্তু আরো এক সাইজ বড় অর্ডার দেয়া উচিত ছিলো।
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'ভালোই', 'কিন্তু', 'আরো', 'এক', 'সাইজ', 'বড়', 'অর্ডার', 'দেয়া', 'উচিত', 'ছিলো', '।']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'ভালোই', 'কিন্তু', 'আরো', 'এক', 'সাইজ', 'বড়', 'অর্ডার', 'দেয়া', 'উচিত', 'ছিলো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ভালোই', 'আরো', 'এক', 'সাইজ', 'বড়', 'অর্ডার', 'দেয়া', 'ছিলো']
    ***************************************************************************************
    Label:  0
    Sentence:   তবে একটার সাইজ চেঞ্জ করার জনু সেলারের সাথে অনেক বার কন্ট্রাক্ট করেও কোন রিপ্লাই পায়নি
    Afert Tokenizing:  ['তবে', 'একটার', 'সাইজ', 'চেঞ্জ', 'করার', 'জনু', 'সেলারের', 'সাথে', 'অনেক', 'বার', 'কন্ট্রাক্ট', 'করেও', 'কোন', 'রিপ্লাই', 'পায়নি']
    Truncating punctuation: ['তবে', 'একটার', 'সাইজ', 'চেঞ্জ', 'করার', 'জনু', 'সেলারের', 'সাথে', 'অনেক', 'বার', 'কন্ট্রাক্ট', 'করেও', 'কোন', 'রিপ্লাই', 'পায়নি']
    Truncating StopWords: ['একটার', 'সাইজ', 'চেঞ্জ', 'জনু', 'সেলারের', 'সাথে', 'কন্ট্রাক্ট', 'করেও', 'রিপ্লাই', 'পায়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  বেল্টা অনেক ভাল।
    Afert Tokenizing:  ['বেল্টা', 'অনেক', 'ভাল', '।']
    Truncating punctuation: ['বেল্টা', 'অনেক', 'ভাল']
    Truncating StopWords: ['বেল্টা', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  সামগ্রিক পণ্যের মান ভাল ছিল।
    Afert Tokenizing:  ['সামগ্রিক', 'পণ্যের', 'মান', 'ভাল', 'ছিল', '।']
    Truncating punctuation: ['সামগ্রিক', 'পণ্যের', 'মান', 'ভাল', 'ছিল']
    Truncating StopWords: ['সামগ্রিক', 'পণ্যের', 'মান', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনি যদি এটি কিনে থাকেন তবে আপনার জয়ী হওয়া উচিত।
    Afert Tokenizing:  ['আপনি', 'যদি', 'এটি', 'কিনে', 'থাকেন', 'তবে', 'আপনার', 'জয়ী', 'হওয়া', 'উচিত', '।']
    Truncating punctuation: ['আপনি', 'যদি', 'এটি', 'কিনে', 'থাকেন', 'তবে', 'আপনার', 'জয়ী', 'হওয়া', 'উচিত']
    Truncating StopWords: ['কিনে', 'জয়ী']
    ***************************************************************************************
    Label:  1
    Sentence:  গুণমান ভাল কিন্তু প্যাকেজ খারাপভাবে ক্ষতিগ্রস্ত হয়েছে এবং খড় বিচ্ছিন্ন করা হয়েছে
    Afert Tokenizing:  ['গুণমান', 'ভাল', 'কিন্তু', 'প্যাকেজ', 'খারাপভাবে', 'ক্ষতিগ্রস্ত', 'হয়েছে', 'এবং', 'খড়', 'বিচ্ছিন্ন', 'করা', 'হয়েছে']
    Truncating punctuation: ['গুণমান', 'ভাল', 'কিন্তু', 'প্যাকেজ', 'খারাপভাবে', 'ক্ষতিগ্রস্ত', 'হয়েছে', 'এবং', 'খড়', 'বিচ্ছিন্ন', 'করা', 'হয়েছে']
    Truncating StopWords: ['গুণমান', 'ভাল', 'প্যাকেজ', 'খারাপভাবে', 'ক্ষতিগ্রস্ত', 'খড়', 'বিচ্ছিন্ন']
    ***************************************************************************************
    Label:  1
    Sentence:  এইটা অন্য রকম জুস খেতে আমার কাছে ভালোই লাগে আবার ও স্টকে চাই
    Afert Tokenizing:  ['এইটা', 'অন্য', 'রকম', 'জুস', 'খেতে', 'আমার', 'কাছে', 'ভালোই', 'লাগে', 'আবার', 'ও', 'স্টকে', 'চাই']
    Truncating punctuation: ['এইটা', 'অন্য', 'রকম', 'জুস', 'খেতে', 'আমার', 'কাছে', 'ভালোই', 'লাগে', 'আবার', 'ও', 'স্টকে', 'চাই']
    Truncating StopWords: ['এইটা', 'জুস', 'খেতে', 'ভালোই', 'লাগে', 'স্টকে', 'চাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "সত্যি বলতে কি, আমার মতে এর স্বাদ ভালো লাগে না।"
    Afert Tokenizing:  ['সত্যি', '"', 'বলতে', 'কি', ',', 'আমার', 'মতে', 'এর', 'স্বাদ', 'ভালো', 'লাগে', 'না।', '"']
    Truncating punctuation: ['সত্যি', 'বলতে', 'কি', 'আমার', 'মতে', 'এর', 'স্বাদ', 'ভালো', 'লাগে', 'না।']
    Truncating StopWords: ['সত্যি', 'মতে', 'স্বাদ', 'ভালো', 'লাগে', 'না।']
    ***************************************************************************************
    Label:  0
    Sentence:  এইটা আবার কেরকম হএলো।কথা কাজে মীল নাই
    Afert Tokenizing:  ['এইটা', 'আবার', 'কেরকম', 'হএলো।কথা', 'কাজে', 'মীল', 'নাই']
    Truncating punctuation: ['এইটা', 'আবার', 'কেরকম', 'হএলো।কথা', 'কাজে', 'মীল', 'নাই']
    Truncating StopWords: ['এইটা', 'কেরকম', 'হএলো।কথা', 'মীল', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  ভেবেছিলাম মেয়াদ উত্তীর্ণ হওয়ার তারিখ নিয়া হয়তো ইস্যু হয় পারে কিন্তু একডম তাজা পণ্য পেইছি।
    Afert Tokenizing:  ['ভেবেছিলাম', 'মেয়াদ', 'উত্তীর্ণ', 'হওয়ার', 'তারিখ', 'নিয়া', 'হয়তো', 'ইস্যু', 'হয়', 'পারে', 'কিন্তু', 'একডম', 'তাজা', 'পণ্য', 'পেইছি', '।']
    Truncating punctuation: ['ভেবেছিলাম', 'মেয়াদ', 'উত্তীর্ণ', 'হওয়ার', 'তারিখ', 'নিয়া', 'হয়তো', 'ইস্যু', 'হয়', 'পারে', 'কিন্তু', 'একডম', 'তাজা', 'পণ্য', 'পেইছি']
    Truncating StopWords: ['ভেবেছিলাম', 'মেয়াদ', 'উত্তীর্ণ', 'তারিখ', 'নিয়া', 'ইস্যু', 'একডম', 'তাজা', 'পণ্য', 'পেইছি']
    ***************************************************************************************
    Label:  1
    Sentence:  একদম তাজা পণ্য পেইছি।
    Afert Tokenizing:  ['একদম', 'তাজা', 'পণ্য', 'পেইছি', '।']
    Truncating punctuation: ['একদম', 'তাজা', 'পণ্য', 'পেইছি']
    Truncating StopWords: ['একদম', 'তাজা', 'পণ্য', 'পেইছি']
    ***************************************************************************************
    Label:  0
    Sentence:  প্যাকেজিং চরম বাজে ছিল পণ্যটি দুমড়ে মুচড়ে গিয়েছ
    Afert Tokenizing:  ['প্যাকেজিং', 'চরম', 'বাজে', 'ছিল', 'পণ্যটি', 'দুমড়ে', 'মুচড়ে', 'গিয়েছ']
    Truncating punctuation: ['প্যাকেজিং', 'চরম', 'বাজে', 'ছিল', 'পণ্যটি', 'দুমড়ে', 'মুচড়ে', 'গিয়েছ']
    Truncating StopWords: ['প্যাকেজিং', 'চরম', 'বাজে', 'পণ্যটি', 'দুমড়ে', 'মুচড়ে', 'গিয়েছ']
    ***************************************************************************************
    Label:  1
    Sentence:  বাংলাদেশের সেরা পানীয়
    Afert Tokenizing:  ['বাংলাদেশের', 'সেরা', 'পানীয়']
    Truncating punctuation: ['বাংলাদেশের', 'সেরা', 'পানীয়']
    Truncating StopWords: ['বাংলাদেশের', 'সেরা', 'পানীয়']
    ***************************************************************************************
    Label:  1
    Sentence:  জুস গুলো খুব ভালো লাগে স্টক নাই নয়তো আর নিতাম.
    Afert Tokenizing:  ['জুস', 'গুলো', 'খুব', 'ভালো', 'লাগে', 'স্টক', 'নাই', 'নয়তো', 'আর', 'নিতাম', '.']
    Truncating punctuation: ['জুস', 'গুলো', 'খুব', 'ভালো', 'লাগে', 'স্টক', 'নাই', 'নয়তো', 'আর', 'নিতাম']
    Truncating StopWords: ['জুস', 'গুলো', 'ভালো', 'লাগে', 'স্টক', 'নাই', 'নয়তো', 'নিতাম']
    ***************************************************************************************
    Label:  1
    Sentence:  লেমন ফ্লেইভর এনার্জি ড্রিংকস ভালই ধন্যবাদ
    Afert Tokenizing:  ['লেমন', 'ফ্লেইভর', 'এনার্জি', 'ড্রিংকস', 'ভালই', 'ধন্যবাদ']
    Truncating punctuation: ['লেমন', 'ফ্লেইভর', 'এনার্জি', 'ড্রিংকস', 'ভালই', 'ধন্যবাদ']
    Truncating StopWords: ['লেমন', 'ফ্লেইভর', 'এনার্জি', 'ড্রিংকস', 'ভালই', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাকিং জোস ছিলো! ২ টাকা কমে পেয়েছি
    Afert Tokenizing:  ['প্যাকিং', 'জোস', 'ছিলো', '!', '২', 'টাকা', 'কমে', 'পেয়েছি']
    Truncating punctuation: ['প্যাকিং', 'জোস', 'ছিলো', '২', 'টাকা', 'কমে', 'পেয়েছি']
    Truncating StopWords: ['প্যাকিং', 'জোস', 'ছিলো', '২', 'টাকা', 'কমে', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  পণ্য নিয়মিত বা বাজারে পাওয়া যায় না দুর্বল বিতরণ চ্যানেল
    Afert Tokenizing:  ['পণ্য', 'নিয়মিত', 'বা', 'বাজারে', 'পাওয়া', 'যায়', 'না', 'দুর্বল', 'বিতরণ', 'চ্যানেল']
    Truncating punctuation: ['পণ্য', 'নিয়মিত', 'বা', 'বাজারে', 'পাওয়া', 'যায়', 'না', 'দুর্বল', 'বিতরণ', 'চ্যানেল']
    Truncating StopWords: ['পণ্য', 'নিয়মিত', 'বাজারে', 'না', 'দুর্বল', 'বিতরণ', 'চ্যানেল']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি হতাশ যে তারা কেনার সীমা সীমাবদ্ধ করেছে এবং আমি আমার অর্ডার দিতে পারছি না
    Afert Tokenizing:  ['আমি', 'হতাশ', 'যে', 'তারা', 'কেনার', 'সীমা', 'সীমাবদ্ধ', 'করেছে', 'এবং', 'আমি', 'আমার', 'অর্ডার', 'দিতে', 'পারছি', 'না']
    Truncating punctuation: ['আমি', 'হতাশ', 'যে', 'তারা', 'কেনার', 'সীমা', 'সীমাবদ্ধ', 'করেছে', 'এবং', 'আমি', 'আমার', 'অর্ডার', 'দিতে', 'পারছি', 'না']
    Truncating StopWords: ['হতাশ', 'কেনার', 'সীমা', 'সীমাবদ্ধ', 'অর্ডার', 'পারছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রতিবার পান্ডা মার্ট থেকে কিছু কিনলে প্যাকেজিং টা সবচেয়ে বাজে পাই
    Afert Tokenizing:  ['প্রতিবার', 'পান্ডা', 'মার্ট', 'থেকে', 'কিছু', 'কিনলে', 'প্যাকেজিং', 'টা', 'সবচেয়ে', 'বাজে', 'পাই']
    Truncating punctuation: ['প্রতিবার', 'পান্ডা', 'মার্ট', 'থেকে', 'কিছু', 'কিনলে', 'প্যাকেজিং', 'টা', 'সবচেয়ে', 'বাজে', 'পাই']
    Truncating StopWords: ['প্রতিবার', 'পান্ডা', 'মার্ট', 'কিনলে', 'প্যাকেজিং', 'টা', 'সবচেয়ে', 'বাজে', 'পাই']
    ***************************************************************************************
    Label:  0
    Sentence:  কিন্তু প্রডাক্ট ডেলিভারি জঘন্য
    Afert Tokenizing:  ['কিন্তু', 'প্রডাক্ট', 'ডেলিভারি', 'জঘন্য']
    Truncating punctuation: ['কিন্তু', 'প্রডাক্ট', 'ডেলিভারি', 'জঘন্য']
    Truncating StopWords: ['প্রডাক্ট', 'ডেলিভারি', 'জঘন্য']
    ***************************************************************************************
    Label:  0
    Sentence:  " তারা প্রডাক্ট ডিলা মেরে মেরে রাখে, প্যাকেজ ছিড়েবিড়ে শেষ"
    Afert Tokenizing:  ['', '"', 'তারা', 'প্রডাক্ট', 'ডিলা', 'মেরে', 'মেরে', 'রাখে', ',', 'প্যাকেজ', 'ছিড়েবিড়ে', 'শেষ', '"']
    Truncating punctuation: ['', 'তারা', 'প্রডাক্ট', 'ডিলা', 'মেরে', 'মেরে', 'রাখে', 'প্যাকেজ', 'ছিড়েবিড়ে', 'শেষ']
    Truncating StopWords: ['', 'প্রডাক্ট', 'ডিলা', 'মেরে', 'মেরে', 'রাখে', 'প্যাকেজ', 'ছিড়েবিড়ে', 'শেষ']
    ***************************************************************************************
    Label:  0
    Sentence:  "১৫ টা অর্ডার করেছিলাম, ১১ টা দিয়েছে বাকিগুলা যদি না থাকে, তাহলে টাকা ফেরত দিয়ে দিন"
    Afert Tokenizing:  ['১৫', '"', 'টা', 'অর্ডার', 'করেছিলাম', ',', '১১', 'টা', 'দিয়েছে', 'বাকিগুলা', 'যদি', 'না', 'থাকে', ',', 'তাহলে', 'টাকা', 'ফেরত', 'দিয়ে', 'দিন', '"']
    Truncating punctuation: ['১৫', 'টা', 'অর্ডার', 'করেছিলাম', '১১', 'টা', 'দিয়েছে', 'বাকিগুলা', 'যদি', 'না', 'থাকে', 'তাহলে', 'টাকা', 'ফেরত', 'দিয়ে', 'দিন']
    Truncating StopWords: ['১৫', 'টা', 'অর্ডার', 'করেছিলাম', '১১', 'টা', 'দিয়েছে', 'বাকিগুলা', 'না', 'টাকা', 'ফেরত', 'দিয়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  ট্রাউজার হাতে পেলাম দাম হিসেবে কাপড় অনেক ভাল মানের
    Afert Tokenizing:  ['ট্রাউজার', 'হাতে', 'পেলাম', 'দাম', 'হিসেবে', 'কাপড়', 'অনেক', 'ভাল', 'মানের']
    Truncating punctuation: ['ট্রাউজার', 'হাতে', 'পেলাম', 'দাম', 'হিসেবে', 'কাপড়', 'অনেক', 'ভাল', 'মানের']
    Truncating StopWords: ['ট্রাউজার', 'হাতে', 'পেলাম', 'দাম', 'হিসেবে', 'কাপড়', 'ভাল', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  সেলাররা এরকম প্রোডাক্ট প্রোভাইড করলে আশা করি মানুষদের অনলাইনে প্রোডাক্ট কেনার ইচ্ছে আরো বাড়বে
    Afert Tokenizing:  ['সেলাররা', 'এরকম', 'প্রোডাক্ট', 'প্রোভাইড', 'করলে', 'আশা', 'করি', 'মানুষদের', 'অনলাইনে', 'প্রোডাক্ট', 'কেনার', 'ইচ্ছে', 'আরো', 'বাড়বে']
    Truncating punctuation: ['সেলাররা', 'এরকম', 'প্রোডাক্ট', 'প্রোভাইড', 'করলে', 'আশা', 'করি', 'মানুষদের', 'অনলাইনে', 'প্রোডাক্ট', 'কেনার', 'ইচ্ছে', 'আরো', 'বাড়বে']
    Truncating StopWords: ['সেলাররা', 'এরকম', 'প্রোডাক্ট', 'প্রোভাইড', 'আশা', 'মানুষদের', 'অনলাইনে', 'প্রোডাক্ট', 'কেনার', 'ইচ্ছে', 'আরো', 'বাড়বে']
    ***************************************************************************************
    Label:  1
    Sentence:  "ট্রাউজারটা ভালই, কাপড়ও ভালোই সেলাইয়ের কোথাও ত্রুটি দেখতে পায়নি"
    Afert Tokenizing:  ['ট্রাউজারটা', '"', 'ভালই', ',', 'কাপড়ও', 'ভালোই', 'সেলাইয়ের', 'কোথাও', 'ত্রুটি', 'দেখতে', 'পায়নি', '"']
    Truncating punctuation: ['ট্রাউজারটা', 'ভালই', 'কাপড়ও', 'ভালোই', 'সেলাইয়ের', 'কোথাও', 'ত্রুটি', 'দেখতে', 'পায়নি']
    Truncating StopWords: ['ট্রাউজারটা', 'ভালই', 'কাপড়ও', 'ভালোই', 'সেলাইয়ের', 'কোথাও', 'ত্রুটি', 'পায়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  কালার ঠিক ছিলো আর দাম হিসাবে প্রডাক্টটি ভালো
    Afert Tokenizing:  ['কালার', 'ঠিক', 'ছিলো', 'আর', 'দাম', 'হিসাবে', 'প্রডাক্টটি', 'ভালো']
    Truncating punctuation: ['কালার', 'ঠিক', 'ছিলো', 'আর', 'দাম', 'হিসাবে', 'প্রডাক্টটি', 'ভালো']
    Truncating StopWords: ['কালার', 'ঠিক', 'ছিলো', 'দাম', 'প্রডাক্টটি', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:   আমার মনে হয় কোয়ালিটি টা আর একটু ইম্প্রুভ করা যেত
    Afert Tokenizing:  ['আমার', 'মনে', 'হয়', 'কোয়ালিটি', 'টা', 'আর', 'একটু', 'ইম্প্রুভ', 'করা', 'যেত']
    Truncating punctuation: ['আমার', 'মনে', 'হয়', 'কোয়ালিটি', 'টা', 'আর', 'একটু', 'ইম্প্রুভ', 'করা', 'যেত']
    Truncating StopWords: ['কোয়ালিটি', 'টা', 'একটু', 'ইম্প্রুভ', 'যেত']
    ***************************************************************************************
    Label:  0
    Sentence:   কাপড়ের কোয়ালিটি খুব বেশি ভালো না
    Afert Tokenizing:  ['কাপড়ের', 'কোয়ালিটি', 'খুব', 'বেশি', 'ভালো', 'না']
    Truncating punctuation: ['কাপড়ের', 'কোয়ালিটি', 'খুব', 'বেশি', 'ভালো', 'না']
    Truncating StopWords: ['কাপড়ের', 'কোয়ালিটি', 'বেশি', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  সেলার ছেরা মাল দেয়
    Afert Tokenizing:  ['সেলার', 'ছেরা', 'মাল', 'দেয়']
    Truncating punctuation: ['সেলার', 'ছেরা', 'মাল', 'দেয়']
    Truncating StopWords: ['সেলার', 'ছেরা', 'মাল', 'দেয়']
    ***************************************************************************************
    Label:  0
    Sentence:  কোন কোয়ালিটি নেই
    Afert Tokenizing:  ['কোন', 'কোয়ালিটি', 'নেই']
    Truncating punctuation: ['কোন', 'কোয়ালিটি', 'নেই']
    Truncating StopWords: ['কোয়ালিটি', 'নেই']
    ***************************************************************************************
    Label:  0
    Sentence:  ছেড়া জিনিস দিয়েছেনএা আশা করি নাই
    Afert Tokenizing:  ['ছেড়া', 'জিনিস', 'দিয়েছেনএা', 'আশা', 'করি', 'নাই']
    Truncating punctuation: ['ছেড়া', 'জিনিস', 'দিয়েছেনএা', 'আশা', 'করি', 'নাই']
    Truncating StopWords: ['ছেড়া', 'জিনিস', 'দিয়েছেনএা', 'আশা', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি সার্ভিসে একেবারেই অসন্তুষ্ট
    Afert Tokenizing:  ['ডেলিভারি', 'সার্ভিসে', 'একেবারেই', 'অসন্তুষ্ট']
    Truncating punctuation: ['ডেলিভারি', 'সার্ভিসে', 'একেবারেই', 'অসন্তুষ্ট']
    Truncating StopWords: ['ডেলিভারি', 'সার্ভিসে', 'একেবারেই', 'অসন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  দামও বাজারদরের চেয়ে কম
    Afert Tokenizing:  ['দামও', 'বাজারদরের', 'চেয়ে', 'কম']
    Truncating punctuation: ['দামও', 'বাজারদরের', 'চেয়ে', 'কম']
    Truncating StopWords: ['দামও', 'বাজারদরের', 'কম']
    ***************************************************************************************
    Label:  1
    Sentence:  "অসম্ভব ভালো, ফেস ওয়াস "
    Afert Tokenizing:  ['অসম্ভব', '"', 'ভালো', ',', 'ফেস', 'ওয়াস', '', '"']
    Truncating punctuation: ['অসম্ভব', 'ভালো', 'ফেস', 'ওয়াস', '']
    Truncating StopWords: ['অসম্ভব', 'ভালো', 'ফেস', 'ওয়াস', '']
    ***************************************************************************************
    Label:  1
    Sentence:  চুলের জেলটা অরিজিনাল
    Afert Tokenizing:  ['চুলের', 'জেলটা', 'অরিজিনাল']
    Truncating punctuation: ['চুলের', 'জেলটা', 'অরিজিনাল']
    Truncating StopWords: ['চুলের', 'জেলটা', 'অরিজিনাল']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট এর কার্যকরীতা তো কোম্পানির উপর ডিপেন্ড করে বাট সেলার বিশ্বাস এর মর্যাদা রেখেছে
    Afert Tokenizing:  ['প্রোডাক্ট', 'এর', 'কার্যকরীতা', 'তো', 'কোম্পানির', 'উপর', 'ডিপেন্ড', 'করে', 'বাট', 'সেলার', 'বিশ্বাস', 'এর', 'মর্যাদা', 'রেখেছে']
    Truncating punctuation: ['প্রোডাক্ট', 'এর', 'কার্যকরীতা', 'তো', 'কোম্পানির', 'উপর', 'ডিপেন্ড', 'করে', 'বাট', 'সেলার', 'বিশ্বাস', 'এর', 'মর্যাদা', 'রেখেছে']
    Truncating StopWords: ['প্রোডাক্ট', 'কার্যকরীতা', 'কোম্পানির', 'ডিপেন্ড', 'বাট', 'সেলার', 'বিশ্বাস', 'মর্যাদা', 'রেখেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ১২০ টাকা ছাড় অথচ মেয়াদ পর্যাপ্ত খুবই আনন্দিত হয়েছি
    Afert Tokenizing:  ['১২০', 'টাকা', 'ছাড়', 'অথচ', 'মেয়াদ', 'পর্যাপ্ত', 'খুবই', 'আনন্দিত', 'হয়েছি']
    Truncating punctuation: ['১২০', 'টাকা', 'ছাড়', 'অথচ', 'মেয়াদ', 'পর্যাপ্ত', 'খুবই', 'আনন্দিত', 'হয়েছি']
    Truncating StopWords: ['১২০', 'টাকা', 'ছাড়', 'মেয়াদ', 'পর্যাপ্ত', 'খুবই', 'আনন্দিত', 'হয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  অরডার দিলাম কি আর দিলো কি
    Afert Tokenizing:  ['অরডার', 'দিলাম', 'কি', 'আর', 'দিলো', 'কি']
    Truncating punctuation: ['অরডার', 'দিলাম', 'কি', 'আর', 'দিলো', 'কি']
    Truncating StopWords: ['অরডার', 'দিলাম', 'দিলো']
    ***************************************************************************************
    Label:  0
    Sentence:   পান্ডামার্ট এর প্রতি বিশ্বাসটা উঠে গেলো
    Afert Tokenizing:  ['পান্ডামার্ট', 'এর', 'প্রতি', 'বিশ্বাসটা', 'উঠে', 'গেলো']
    Truncating punctuation: ['পান্ডামার্ট', 'এর', 'প্রতি', 'বিশ্বাসটা', 'উঠে', 'গেলো']
    Truncating StopWords: ['পান্ডামার্ট', 'বিশ্বাসটা', 'উঠে', 'গেলো']
    ***************************************************************************************
    Label:  0
    Sentence:  আমাকে ১ বছরের পুরনো প্রোডাক্ট দেওয়া হয়েছে
    Afert Tokenizing:  ['আমাকে', '১', 'বছরের', 'পুরনো', 'প্রোডাক্ট', 'দেওয়া', 'হয়েছে']
    Truncating punctuation: ['আমাকে', '১', 'বছরের', 'পুরনো', 'প্রোডাক্ট', 'দেওয়া', 'হয়েছে']
    Truncating StopWords: ['১', 'বছরের', 'পুরনো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:   প্রোডাক্ট দেখে ই মন টা উদাসীন হয়ে গেল
    Afert Tokenizing:  ['প্রোডাক্ট', 'দেখে', 'ই', 'মন', 'টা', 'উদাসীন', 'হয়ে', 'গেল']
    Truncating punctuation: ['প্রোডাক্ট', 'দেখে', 'ই', 'মন', 'টা', 'উদাসীন', 'হয়ে', 'গেল']
    Truncating StopWords: ['প্রোডাক্ট', 'মন', 'টা', 'উদাসীন']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট এর বাইরে ধূলার আস্তরণ পরে গেছে
    Afert Tokenizing:  ['প্রোডাক্ট', 'এর', 'বাইরে', 'ধূলার', 'আস্তরণ', 'পরে', 'গেছে']
    Truncating punctuation: ['প্রোডাক্ট', 'এর', 'বাইরে', 'ধূলার', 'আস্তরণ', 'পরে', 'গেছে']
    Truncating StopWords: ['প্রোডাক্ট', 'বাইরে', 'ধূলার', 'আস্তরণ']
    ***************************************************************************************
    Label:  1
    Sentence:  নিঃসন্দেহে অনেক ভালো প্রোডাক্ট
    Afert Tokenizing:  ['নিঃসন্দেহে', 'অনেক', 'ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['নিঃসন্দেহে', 'অনেক', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['নিঃসন্দেহে', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  সব ঠিকঠাক মেয়াদ অনেক দিন আছে
    Afert Tokenizing:  ['সব', 'ঠিকঠাক', 'মেয়াদ', 'অনেক', 'দিন', 'আছে']
    Truncating punctuation: ['সব', 'ঠিকঠাক', 'মেয়াদ', 'অনেক', 'দিন', 'আছে']
    Truncating StopWords: ['ঠিকঠাক', 'মেয়াদ']
    ***************************************************************************************
    Label:  1
    Sentence:  "আর জেল টা অনেক হার্ড, ঠিক যেমনটা চেয়েছিলাম"
    Afert Tokenizing:  ['আর', '"', 'জেল', 'টা', 'অনেক', 'হার্ড', ',', 'ঠিক', 'যেমনটা', 'চেয়েছিলাম', '"']
    Truncating punctuation: ['আর', 'জেল', 'টা', 'অনেক', 'হার্ড', 'ঠিক', 'যেমনটা', 'চেয়েছিলাম']
    Truncating StopWords: ['জেল', 'টা', 'হার্ড', 'ঠিক', 'যেমনটা', 'চেয়েছিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  স্টুডিও এক্স সাশ্রয়ী মূল্যে সেরা মানের পুরুষ পণ্য সরবরাহ করে
    Afert Tokenizing:  ['স্টুডিও', 'এক্স', 'সাশ্রয়ী', 'মূল্যে', 'সেরা', 'মানের', 'পুরুষ', 'পণ্য', 'সরবরাহ', 'করে']
    Truncating punctuation: ['স্টুডিও', 'এক্স', 'সাশ্রয়ী', 'মূল্যে', 'সেরা', 'মানের', 'পুরুষ', 'পণ্য', 'সরবরাহ', 'করে']
    Truncating StopWords: ['স্টুডিও', 'এক্স', 'সাশ্রয়ী', 'মূল্যে', 'সেরা', 'মানের', 'পুরুষ', 'পণ্য', 'সরবরাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট ড্যামেজ ছিল
    Afert Tokenizing:  ['প্রোডাক্ট', 'ড্যামেজ', 'ছিল']
    Truncating punctuation: ['প্রোডাক্ট', 'ড্যামেজ', 'ছিল']
    Truncating StopWords: ['প্রোডাক্ট', 'ড্যামেজ']
    ***************************************************************************************
    Label:  1
    Sentence:  "ডিস্কাউন্টে কিনেছিলাম, অথেন্টিক প্রোডাক্ট পেয়েছি"
    Afert Tokenizing:  ['ডিস্কাউন্টে', '"', 'কিনেছিলাম', ',', 'অথেন্টিক', 'প্রোডাক্ট', 'পেয়েছি', '"']
    Truncating punctuation: ['ডিস্কাউন্টে', 'কিনেছিলাম', 'অথেন্টিক', 'প্রোডাক্ট', 'পেয়েছি']
    Truncating StopWords: ['ডিস্কাউন্টে', 'কিনেছিলাম', 'অথেন্টিক', 'প্রোডাক্ট', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম বাজে একটা স্প্রে ডেট নাই
    Afert Tokenizing:  ['একদম', 'বাজে', 'একটা', 'স্প্রে', 'ডেট', 'নাই']
    Truncating punctuation: ['একদম', 'বাজে', 'একটা', 'স্প্রে', 'ডেট', 'নাই']
    Truncating StopWords: ['একদম', 'বাজে', 'একটা', 'স্প্রে', 'ডেট', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা স্প্রে নাকি নর্দামার পানি ভাই?
    Afert Tokenizing:  ['এটা', 'স্প্রে', 'নাকি', 'নর্দামার', 'পানি', 'ভাই', '?']
    Truncating punctuation: ['এটা', 'স্প্রে', 'নাকি', 'নর্দামার', 'পানি', 'ভাই']
    Truncating StopWords: ['স্প্রে', 'নর্দামার', 'পানি', 'ভাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মত সাধারণ মানুষের টাকা এই ভাবে মেরে না খাইলেও পারতেন
    Afert Tokenizing:  ['আমার', 'মত', 'সাধারণ', 'মানুষের', 'টাকা', 'এই', 'ভাবে', 'মেরে', 'না', 'খাইলেও', 'পারতেন']
    Truncating punctuation: ['আমার', 'মত', 'সাধারণ', 'মানুষের', 'টাকা', 'এই', 'ভাবে', 'মেরে', 'না', 'খাইলেও', 'পারতেন']
    Truncating StopWords: ['মত', 'মানুষের', 'টাকা', 'মেরে', 'না', 'খাইলেও', 'পারতেন']
    ***************************************************************************************
    Label:  0
    Sentence:  ক্যান চেপ্টা খাওয়া এবং প্যাকেট সিলড করা ছিলনা মনে হচ্ছে ব্যবহৃত প্রোডক্ট দিয়েছে
    Afert Tokenizing:  ['ক্যান', 'চেপ্টা', 'খাওয়া', 'এবং', 'প্যাকেট', 'সিলড', 'করা', 'ছিলনা', 'মনে', 'হচ্ছে', 'ব্যবহৃত', 'প্রোডক্ট', 'দিয়েছে']
    Truncating punctuation: ['ক্যান', 'চেপ্টা', 'খাওয়া', 'এবং', 'প্যাকেট', 'সিলড', 'করা', 'ছিলনা', 'মনে', 'হচ্ছে', 'ব্যবহৃত', 'প্রোডক্ট', 'দিয়েছে']
    Truncating StopWords: ['ক্যান', 'চেপ্টা', 'খাওয়া', 'প্যাকেট', 'সিলড', 'ছিলনা', 'ব্যবহৃত', 'প্রোডক্ট', 'দিয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  দোকানের থেকে অনেক কম মূল্যে পেয়ছি
    Afert Tokenizing:  ['দোকানের', 'থেকে', 'অনেক', 'কম', 'মূল্যে', 'পেয়ছি']
    Truncating punctuation: ['দোকানের', 'থেকে', 'অনেক', 'কম', 'মূল্যে', 'পেয়ছি']
    Truncating StopWords: ['দোকানের', 'কম', 'মূল্যে', 'পেয়ছি']
    ***************************************************************************************
    Label:  1
    Sentence:  "গুনে, মানে এবং সুগন্ধিতে ভরপুর"
    Afert Tokenizing:  ['"গুনে', ',', 'মানে', 'এবং', 'সুগন্ধিতে', 'ভরপুর', '"']
    Truncating punctuation: ['"গুনে', 'মানে', 'এবং', 'সুগন্ধিতে', 'ভরপুর']
    Truncating StopWords: ['"গুনে', 'মানে', 'সুগন্ধিতে', 'ভরপুর']
    ***************************************************************************************
    Label:  1
    Sentence:  মেরিকোর প্রাডাক্ট মান সম্পর্কে আলাদাকরে বলার কিছুই না
    Afert Tokenizing:  ['মেরিকোর', 'প্রাডাক্ট', 'মান', 'সম্পর্কে', 'আলাদাকরে', 'বলার', 'কিছুই', 'না']
    Truncating punctuation: ['মেরিকোর', 'প্রাডাক্ট', 'মান', 'সম্পর্কে', 'আলাদাকরে', 'বলার', 'কিছুই', 'না']
    Truncating StopWords: ['মেরিকোর', 'প্রাডাক্ট', 'মান', 'সম্পর্কে', 'আলাদাকরে', 'বলার', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:   বড় কোম্পানির সার্ভিস যদি এতো নিম্নমানের হয় তাহলে কেমনে চলে
    Afert Tokenizing:  ['বড়', 'কোম্পানির', 'সার্ভিস', 'যদি', 'এতো', 'নিম্নমানের', 'হয়', 'তাহলে', 'কেমনে', 'চলে']
    Truncating punctuation: ['বড়', 'কোম্পানির', 'সার্ভিস', 'যদি', 'এতো', 'নিম্নমানের', 'হয়', 'তাহলে', 'কেমনে', 'চলে']
    Truncating StopWords: ['বড়', 'কোম্পানির', 'সার্ভিস', 'এতো', 'নিম্নমানের', 'কেমনে']
    ***************************************************************************************
    Label:  0
    Sentence:  দোকানের ড্যামেজ যেগুলো রির্টান দিছে সেগুলো অনলাইন কাস্টমারদের দিচ্ছি
    Afert Tokenizing:  ['দোকানের', 'ড্যামেজ', 'যেগুলো', 'রির্টান', 'দিছে', 'সেগুলো', 'অনলাইন', 'কাস্টমারদের', 'দিচ্ছি']
    Truncating punctuation: ['দোকানের', 'ড্যামেজ', 'যেগুলো', 'রির্টান', 'দিছে', 'সেগুলো', 'অনলাইন', 'কাস্টমারদের', 'দিচ্ছি']
    Truncating StopWords: ['দোকানের', 'ড্যামেজ', 'যেগুলো', 'রির্টান', 'দিছে', 'সেগুলো', 'অনলাইন', 'কাস্টমারদের', 'দিচ্ছি']
    ***************************************************************************************
    Label:  1
    Sentence:  বরাবরের মতো খাঁটি এবং দুর্দান্ত পারফিউম স্প্রে
    Afert Tokenizing:  ['বরাবরের', 'মতো', 'খাঁটি', 'এবং', 'দুর্দান্ত', 'পারফিউম', 'স্প্রে']
    Truncating punctuation: ['বরাবরের', 'মতো', 'খাঁটি', 'এবং', 'দুর্দান্ত', 'পারফিউম', 'স্প্রে']
    Truncating StopWords: ['বরাবরের', 'খাঁটি', 'দুর্দান্ত', 'পারফিউম', 'স্প্রে']
    ***************************************************************************************
    Label:  1
    Sentence:  "স্মেল ভালো লেগেছে,ডেটও আছে অনেক"
    Afert Tokenizing:  ['স্মেল', '"', 'ভালো', 'লেগেছে,ডেটও', 'আছে', 'অনেক', '"']
    Truncating punctuation: ['স্মেল', 'ভালো', 'লেগেছে,ডেটও', 'আছে', 'অনেক']
    Truncating StopWords: ['স্মেল', 'ভালো', 'লেগেছে,ডেটও']
    ***************************************************************************************
    Label:  0
    Sentence:  "২০২০ এর পন্য দিয়েছেন, অনেক পুরানো"
    Afert Tokenizing:  ['২০২০', '"', 'এর', 'পন্য', 'দিয়েছেন', ',', 'অনেক', 'পুরানো', '"']
    Truncating punctuation: ['২০২০', 'এর', 'পন্য', 'দিয়েছেন', 'অনেক', 'পুরানো']
    Truncating StopWords: ['২০২০', 'পন্য', 'দিয়েছেন', 'পুরানো']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রোডাক্ট গুলো তে অনেক ধুলাবালু ছিল ও ক্র্যাচ ছিল
    Afert Tokenizing:  ['প্রোডাক্ট', 'গুলো', 'তে', 'অনেক', 'ধুলাবালু', 'ছিল', 'ও', 'ক্র্যাচ', 'ছিল']
    Truncating punctuation: ['প্রোডাক্ট', 'গুলো', 'তে', 'অনেক', 'ধুলাবালু', 'ছিল', 'ও', 'ক্র্যাচ', 'ছিল']
    Truncating StopWords: ['প্রোডাক্ট', 'গুলো', 'তে', 'ধুলাবালু', 'ক্র্যাচ']
    ***************************************************************************************
    Label:  0
    Sentence:  ব্লাক চেয়েছিলাম  গ্রিন দিয়েছে
    Afert Tokenizing:  ['ব্লাক', 'চেয়েছিলাম', 'গ্রিন', 'দিয়েছে']
    Truncating punctuation: ['ব্লাক', 'চেয়েছিলাম', 'গ্রিন', 'দিয়েছে']
    Truncating StopWords: ['ব্লাক', 'চেয়েছিলাম', 'গ্রিন', 'দিয়েছে']
    ***************************************************************************************
    Label:  0
    Sentence:   অরডার করলাম ব্ল্যাক  কালার কিন্তু  ফালতু  একটা কালার দিসে
    Afert Tokenizing:  ['অরডার', 'করলাম', 'ব্ল্যাক', 'কালার', 'কিন্তু', 'ফালতু', 'একটা', 'কালার', 'দিসে']
    Truncating punctuation: ['অরডার', 'করলাম', 'ব্ল্যাক', 'কালার', 'কিন্তু', 'ফালতু', 'একটা', 'কালার', 'দিসে']
    Truncating StopWords: ['অরডার', 'করলাম', 'ব্ল্যাক', 'কালার', 'ফালতু', 'একটা', 'কালার', 'দিসে']
    ***************************************************************************************
    Label:  0
    Sentence:   তবে প্রডাক্ট কোন কাজ করে না
    Afert Tokenizing:  ['তবে', 'প্রডাক্ট', 'কোন', 'কাজ', 'করে', 'না']
    Truncating punctuation: ['তবে', 'প্রডাক্ট', 'কোন', 'কাজ', 'করে', 'না']
    Truncating StopWords: ['প্রডাক্ট', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই ছবি তে দেওয়া আছে একটা আমাকে দিছে আরেক
    Afert Tokenizing:  ['ভাই', 'ছবি', 'তে', 'দেওয়া', 'আছে', 'একটা', 'আমাকে', 'দিছে', 'আরেক']
    Truncating punctuation: ['ভাই', 'ছবি', 'তে', 'দেওয়া', 'আছে', 'একটা', 'আমাকে', 'দিছে', 'আরেক']
    Truncating StopWords: ['ভাই', 'ছবি', 'তে', 'একটা', 'দিছে', 'আরেক']
    ***************************************************************************************
    Label:  1
    Sentence:  টাকা অনুযায়ী ছাতাটা অনেক ভালো
    Afert Tokenizing:  ['টাকা', 'অনুযায়ী', 'ছাতাটা', 'অনেক', 'ভালো']
    Truncating punctuation: ['টাকা', 'অনুযায়ী', 'ছাতাটা', 'অনেক', 'ভালো']
    Truncating StopWords: ['টাকা', 'অনুযায়ী', 'ছাতাটা', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্টি ভালো কফি মিক্সড করার জন্য
    Afert Tokenizing:  ['প্রডাক্টি', 'ভালো', 'কফি', 'মিক্সড', 'করার', 'জন্য']
    Truncating punctuation: ['প্রডাক্টি', 'ভালো', 'কফি', 'মিক্সড', 'করার', 'জন্য']
    Truncating StopWords: ['প্রডাক্টি', 'ভালো', 'কফি', 'মিক্সড']
    ***************************************************************************************
    Label:  1
    Sentence:  ব্যাবহার করার মতোই পন্য
    Afert Tokenizing:  ['ব্যাবহার', 'করার', 'মতোই', 'পন্য']
    Truncating punctuation: ['ব্যাবহার', 'করার', 'মতোই', 'পন্য']
    Truncating StopWords: ['ব্যাবহার', 'পন্য']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব একটা ভালো না মাথা বাকা থাকায় অনেক বেশি কাপে
    Afert Tokenizing:  ['খুব', 'একটা', 'ভালো', 'না', 'মাথা', 'বাকা', 'থাকায়', 'অনেক', 'বেশি', 'কাপে']
    Truncating punctuation: ['খুব', 'একটা', 'ভালো', 'না', 'মাথা', 'বাকা', 'থাকায়', 'অনেক', 'বেশি', 'কাপে']
    Truncating StopWords: ['একটা', 'ভালো', 'না', 'মাথা', 'বাকা', 'থাকায়', 'বেশি', 'কাপে']
    ***************************************************************************************
    Label:  1
    Sentence:  "প্রোডাক্টটি বেশ ভালোভাবেই, দ্রুত পেয়েছি"
    Afert Tokenizing:  ['প্রোডাক্টটি', '"', 'বেশ', 'ভালোভাবেই', ',', 'দ্রুত', 'পেয়েছি', '"']
    Truncating punctuation: ['প্রোডাক্টটি', 'বেশ', 'ভালোভাবেই', 'দ্রুত', 'পেয়েছি']
    Truncating StopWords: ['প্রোডাক্টটি', 'ভালোভাবেই', 'দ্রুত', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো কফি মিক্সার এইদামে
    Afert Tokenizing:  ['খুব', 'ভালো', 'কফি', 'মিক্সার', 'এইদামে']
    Truncating punctuation: ['খুব', 'ভালো', 'কফি', 'মিক্সার', 'এইদামে']
    Truncating StopWords: ['ভালো', 'কফি', 'মিক্সার', 'এইদামে']
    ***************************************************************************************
    Label:  1
    Sentence:  দাম হিসাবে পণ্য ঠিক আছে কিন্তু কফি ফোম হচ্ছে না
    Afert Tokenizing:  ['দাম', 'হিসাবে', 'পণ্য', 'ঠিক', 'আছে', 'কিন্তু', 'কফি', 'ফোম', 'হচ্ছে', 'না']
    Truncating punctuation: ['দাম', 'হিসাবে', 'পণ্য', 'ঠিক', 'আছে', 'কিন্তু', 'কফি', 'ফোম', 'হচ্ছে', 'না']
    Truncating StopWords: ['দাম', 'পণ্য', 'ঠিক', 'কফি', 'ফোম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  মিক্সার টি ভালই  নিতে পারেন
    Afert Tokenizing:  ['মিক্সার', 'টি', 'ভালই', 'নিতে', 'পারেন']
    Truncating punctuation: ['মিক্সার', 'টি', 'ভালই', 'নিতে', 'পারেন']
    Truncating StopWords: ['মিক্সার', 'ভালই']
    ***************************************************************************************
    Label:  1
    Sentence:  ৯০ টাকায় জোস প্রোডাক্ট
    Afert Tokenizing:  ['৯০', 'টাকায়', 'জোস', 'প্রোডাক্ট']
    Truncating punctuation: ['৯০', 'টাকায়', 'জোস', 'প্রোডাক্ট']
    Truncating StopWords: ['৯০', 'টাকায়', 'জোস', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাঙা জিনিস আসছে খুবই বাজে অবস্থা
    Afert Tokenizing:  ['ভাঙা', 'জিনিস', 'আসছে', 'খুবই', 'বাজে', 'অবস্থা']
    Truncating punctuation: ['ভাঙা', 'জিনিস', 'আসছে', 'খুবই', 'বাজে', 'অবস্থা']
    Truncating StopWords: ['ভাঙা', 'জিনিস', 'আসছে', 'খুবই', 'বাজে', 'অবস্থা']
    ***************************************************************************************
    Label:  1
    Sentence:  কম দামে পেয়েছি চালিয়ে দেখলা ঠিকই আছে
    Afert Tokenizing:  ['কম', 'দামে', 'পেয়েছি', 'চালিয়ে', 'দেখলা', 'ঠিকই', 'আছে']
    Truncating punctuation: ['কম', 'দামে', 'পেয়েছি', 'চালিয়ে', 'দেখলা', 'ঠিকই', 'আছে']
    Truncating StopWords: ['কম', 'দামে', 'পেয়েছি', 'চালিয়ে', 'দেখলা', 'ঠিকই']
    ***************************************************************************************
    Label:  0
    Sentence:  ব্যাটারি দিয়ে চালানো বেশি ব্যয়বহুল
    Afert Tokenizing:  ['ব্যাটারি', 'দিয়ে', 'চালানো', 'বেশি', 'ব্যয়বহুল']
    Truncating punctuation: ['ব্যাটারি', 'দিয়ে', 'চালানো', 'বেশি', 'ব্যয়বহুল']
    Truncating StopWords: ['ব্যাটারি', 'দিয়ে', 'চালানো', 'বেশি', 'ব্যয়বহুল']
    ***************************************************************************************
    Label:  1
    Sentence:  কোন কম্প্লেইন নাই
    Afert Tokenizing:  ['কোন', 'কম্প্লেইন', 'নাই']
    Truncating punctuation: ['কোন', 'কম্প্লেইন', 'নাই']
    Truncating StopWords: ['কম্প্লেইন', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "দরিদ্র মানের, মোটর শক্তি পূর্ণ নয়"
    Afert Tokenizing:  ['দরিদ্র', '"', 'মানের', ',', 'মোটর', 'শক্তি', 'পূর্ণ', 'নয়', '"']
    Truncating punctuation: ['দরিদ্র', 'মানের', 'মোটর', 'শক্তি', 'পূর্ণ', 'নয়']
    Truncating StopWords: ['দরিদ্র', 'মানের', 'মোটর', 'শক্তি', 'পূর্ণ', 'নয়']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো লাগে নাই চলে আবার চলে না হুদাই কিনলাম
    Afert Tokenizing:  ['ভালো', 'লাগে', 'নাই', 'চলে', 'আবার', 'চলে', 'না', 'হুদাই', 'কিনলাম']
    Truncating punctuation: ['ভালো', 'লাগে', 'নাই', 'চলে', 'আবার', 'চলে', 'না', 'হুদাই', 'কিনলাম']
    Truncating StopWords: ['ভালো', 'লাগে', 'নাই', 'না', 'হুদাই', 'কিনলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  অত স্পিড নাই
    Afert Tokenizing:  ['অত', 'স্পিড', 'নাই']
    Truncating punctuation: ['অত', 'স্পিড', 'নাই']
    Truncating StopWords: ['অত', 'স্পিড', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  যতেষ্ট ভালো কোয়ালিটি
    Afert Tokenizing:  ['যতেষ্ট', 'ভালো', 'কোয়ালিটি']
    Truncating punctuation: ['যতেষ্ট', 'ভালো', 'কোয়ালিটি']
    Truncating StopWords: ['যতেষ্ট', 'ভালো', 'কোয়ালিটি']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রোডাক্ট মানসম্মত
    Afert Tokenizing:  ['প্রোডাক্ট', 'মানসম্মত']
    Truncating punctuation: ['প্রোডাক্ট', 'মানসম্মত']
    Truncating StopWords: ['প্রোডাক্ট', 'মানসম্মত']
    ***************************************************************************************
    Label:  0
    Sentence:  ৫ মিনিট চলার পর আর চলে না
    Afert Tokenizing:  ['৫', 'মিনিট', 'চলার', 'পর', 'আর', 'চলে', 'না']
    Truncating punctuation: ['৫', 'মিনিট', 'চলার', 'পর', 'আর', 'চলে', 'না']
    Truncating StopWords: ['৫', 'মিনিট', 'চলার', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  অনেক রকম ট্রাই করেছি কিন্তু আর চলে না
    Afert Tokenizing:  ['অনেক', 'রকম', 'ট্রাই', 'করেছি', 'কিন্তু', 'আর', 'চলে', 'না']
    Truncating punctuation: ['অনেক', 'রকম', 'ট্রাই', 'করেছি', 'কিন্তু', 'আর', 'চলে', 'না']
    Truncating StopWords: ['ট্রাই', 'করেছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম বাজে প্রোডাক্ট
    Afert Tokenizing:  ['একদম', 'বাজে', 'প্রোডাক্ট']
    Truncating punctuation: ['একদম', 'বাজে', 'প্রোডাক্ট']
    Truncating StopWords: ['একদম', 'বাজে', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  কফির কাপে দিলে আর ঘুরতে পারে না
    Afert Tokenizing:  ['কফির', 'কাপে', 'দিলে', 'আর', 'ঘুরতে', 'পারে', 'না']
    Truncating punctuation: ['কফির', 'কাপে', 'দিলে', 'আর', 'ঘুরতে', 'পারে', 'না']
    Truncating StopWords: ['কফির', 'কাপে', 'দিলে', 'ঘুরতে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দুইবার নষ্ট হয়ে গেলেও কাজ হচ্ছে না
    Afert Tokenizing:  ['দুইবার', 'নষ্ট', 'হয়ে', 'গেলেও', 'কাজ', 'হচ্ছে', 'না']
    Truncating punctuation: ['দুইবার', 'নষ্ট', 'হয়ে', 'গেলেও', 'কাজ', 'হচ্ছে', 'না']
    Truncating StopWords: ['দুইবার', 'নষ্ট', 'গেলেও', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  মটরটা আরো পাওয়ারফুল হলে ভালো হতো
    Afert Tokenizing:  ['মটরটা', 'আরো', 'পাওয়ারফুল', 'হলে', 'ভালো', 'হতো']
    Truncating punctuation: ['মটরটা', 'আরো', 'পাওয়ারফুল', 'হলে', 'ভালো', 'হতো']
    Truncating StopWords: ['মটরটা', 'আরো', 'পাওয়ারফুল', 'ভালো', 'হতো']
    ***************************************************************************************
    Label:  0
    Sentence:  ৭ দিনও হয় নাই ফ্যানটা চালাচ্ছি আজ হঠাৎ করে বন্ধ হয়ে গেলো
    Afert Tokenizing:  ['৭', 'দিনও', 'হয়', 'নাই', 'ফ্যানটা', 'চালাচ্ছি', 'আজ', 'হঠাৎ', 'করে', 'বন্ধ', 'হয়ে', 'গেলো']
    Truncating punctuation: ['৭', 'দিনও', 'হয়', 'নাই', 'ফ্যানটা', 'চালাচ্ছি', 'আজ', 'হঠাৎ', 'করে', 'বন্ধ', 'হয়ে', 'গেলো']
    Truncating StopWords: ['৭', 'দিনও', 'নাই', 'ফ্যানটা', 'চালাচ্ছি', 'হঠাৎ', 'বন্ধ', 'হয়ে', 'গেলো']
    ***************************************************************************************
    Label:  0
    Sentence:  এরকম খারাপ অভিজ্ঞতা অন্য আর কোনো প্রোডাক্ট  এর ক্ষেত্রে হয়নি
    Afert Tokenizing:  ['এরকম', 'খারাপ', 'অভিজ্ঞতা', 'অন্য', 'আর', 'কোনো', 'প্রোডাক্ট', 'এর', 'ক্ষেত্রে', 'হয়নি']
    Truncating punctuation: ['এরকম', 'খারাপ', 'অভিজ্ঞতা', 'অন্য', 'আর', 'কোনো', 'প্রোডাক্ট', 'এর', 'ক্ষেত্রে', 'হয়নি']
    Truncating StopWords: ['এরকম', 'খারাপ', 'অভিজ্ঞতা', 'প্রোডাক্ট', 'হয়নি']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্রোডাক্ট  ভাঙ্গাচোরা, এবং থেতলানো"
    Afert Tokenizing:  ['প্রোডাক্ট', '"', 'ভাঙ্গাচোরা', ',', 'এবং', 'থেতলানো', '"']
    Truncating punctuation: ['প্রোডাক্ট', 'ভাঙ্গাচোরা', 'এবং', 'থেতলানো']
    Truncating StopWords: ['প্রোডাক্ট', 'ভাঙ্গাচোরা', 'থেতলানো']
    ***************************************************************************************
    Label:  0
    Sentence:  "২৩ দিন পর প্রোডাক্ট হাতে পাইছি, আর ফ্যান দেখানো হইছে একটা আর দেওয়া হইছে আরেকটা"
    Afert Tokenizing:  ['২৩', '"', 'দিন', 'পর', 'প্রোডাক্ট', 'হাতে', 'পাইছি', ',', 'আর', 'ফ্যান', 'দেখানো', 'হইছে', 'একটা', 'আর', 'দেওয়া', 'হইছে', 'আরেকটা', '"']
    Truncating punctuation: ['২৩', 'দিন', 'পর', 'প্রোডাক্ট', 'হাতে', 'পাইছি', 'আর', 'ফ্যান', 'দেখানো', 'হইছে', 'একটা', 'আর', 'দেওয়া', 'হইছে', 'আরেকটা']
    Truncating StopWords: ['২৩', 'প্রোডাক্ট', 'হাতে', 'পাইছি', 'ফ্যান', 'দেখানো', 'হইছে', 'একটা', 'হইছে', 'আরেকটা']
    ***************************************************************************************
    Label:  1
    Sentence:  বাজারের চেয়ে কম দামে পেয়েছি
    Afert Tokenizing:  ['বাজারের', 'চেয়ে', 'কম', 'দামে', 'পেয়েছি']
    Truncating punctuation: ['বাজারের', 'চেয়ে', 'কম', 'দামে', 'পেয়েছি']
    Truncating StopWords: ['বাজারের', 'চেয়ে', 'কম', 'দামে', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  এই ভাবে প্রতারণা করা ঠিক না
    Afert Tokenizing:  ['এই', 'ভাবে', 'প্রতারণা', 'করা', 'ঠিক', 'না']
    Truncating punctuation: ['এই', 'ভাবে', 'প্রতারণা', 'করা', 'ঠিক', 'না']
    Truncating StopWords: ['প্রতারণা', 'ঠিক', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই বাজে ইলেকট্রিক শক দেয়
    Afert Tokenizing:  ['খুবই', 'বাজে', 'ইলেকট্রিক', 'শক', 'দেয়']
    Truncating punctuation: ['খুবই', 'বাজে', 'ইলেকট্রিক', 'শক', 'দেয়']
    Truncating StopWords: ['খুবই', 'বাজে', 'ইলেকট্রিক', 'শক', 'দেয়']
    ***************************************************************************************
    Label:  0
    Sentence:  "বিষয়টি দুঃখ জনক, প্যান্টি ভাঙ্গা"
    Afert Tokenizing:  ['বিষয়টি', '"', 'দুঃখ', 'জনক', ',', 'প্যান্টি', 'ভাঙ্গা', '"']
    Truncating punctuation: ['বিষয়টি', 'দুঃখ', 'জনক', 'প্যান্টি', 'ভাঙ্গা']
    Truncating StopWords: ['বিষয়টি', 'দুঃখ', 'জনক', 'প্যান্টি', 'ভাঙ্গা']
    ***************************************************************************************
    Label:  1
    Sentence:  " চমৎকার ফ্যান, ব্যবহার করে খুব ভালো লেগেছে"
    Afert Tokenizing:  ['', '"', 'চমৎকার', 'ফ্যান', ',', 'ব্যবহার', 'করে', 'খুব', 'ভালো', 'লেগেছে', '"']
    Truncating punctuation: ['', 'চমৎকার', 'ফ্যান', 'ব্যবহার', 'করে', 'খুব', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['', 'চমৎকার', 'ফ্যান', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:   বডিতে কোন সমস্যা আছে
    Afert Tokenizing:  ['বডিতে', 'কোন', 'সমস্যা', 'আছে']
    Truncating punctuation: ['বডিতে', 'কোন', 'সমস্যা', 'আছে']
    Truncating StopWords: ['বডিতে', 'সমস্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  ফ্যান টি চালানোর সময় কেমন যেন পোড়া গন্ধ করছিল
    Afert Tokenizing:  ['ফ্যান', 'টি', 'চালানোর', 'সময়', 'কেমন', 'যেন', 'পোড়া', 'গন্ধ', 'করছিল']
    Truncating punctuation: ['ফ্যান', 'টি', 'চালানোর', 'সময়', 'কেমন', 'যেন', 'পোড়া', 'গন্ধ', 'করছিল']
    Truncating StopWords: ['ফ্যান', 'চালানোর', 'সময়', 'কেমন', 'পোড়া', 'গন্ধ', 'করছিল']
    ***************************************************************************************
    Label:  0
    Sentence:  ফ্যান এক জাগায় রাখলে ঘুরে যায় আর স্পিড এ অনেক সমস্যা
    Afert Tokenizing:  ['ফ্যান', 'এক', 'জাগায়', 'রাখলে', 'ঘুরে', 'যায়', 'আর', 'স্পিড', 'এ', 'অনেক', 'সমস্যা']
    Truncating punctuation: ['ফ্যান', 'এক', 'জাগায়', 'রাখলে', 'ঘুরে', 'যায়', 'আর', 'স্পিড', 'এ', 'অনেক', 'সমস্যা']
    Truncating StopWords: ['ফ্যান', 'এক', 'জাগায়', 'রাখলে', 'ঘুরে', 'যায়', 'স্পিড', 'সমস্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  কোয়ালিটি অনেক খারাপ
    Afert Tokenizing:  ['কোয়ালিটি', 'অনেক', 'খারাপ']
    Truncating punctuation: ['কোয়ালিটি', 'অনেক', 'খারাপ']
    Truncating StopWords: ['কোয়ালিটি', 'খারাপ']
    ***************************************************************************************
    Label:  1
    Sentence:  "দাম হিসেবে আলহামদুলিল্লাহ
    Afert Tokenizing:  ['দাম', '"', 'হিসেবে', 'আলহামদুলিল্লাহ']
    Truncating punctuation: ['দাম', 'হিসেবে', 'আলহামদুলিল্লাহ']
    Truncating StopWords: ['দাম', 'হিসেবে', 'আলহামদুলিল্লাহ']
    ***************************************************************************************
    Label:  0
    Sentence:  অনেক স্পীড তবে শব্দ বেশি
    Afert Tokenizing:  ['অনেক', 'স্পীড', 'তবে', 'শব্দ', 'বেশি']
    Truncating punctuation: ['অনেক', 'স্পীড', 'তবে', 'শব্দ', 'বেশি']
    Truncating StopWords: ['স্পীড', 'শব্দ', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  অফিসে ব্যাবহার করার জন্য ভালো
    Afert Tokenizing:  ['অফিসে', 'ব্যাবহার', 'করার', 'জন্য', 'ভালো']
    Truncating punctuation: ['অফিসে', 'ব্যাবহার', 'করার', 'জন্য', 'ভালো']
    Truncating StopWords: ['অফিসে', 'ব্যাবহার', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  সবকিছুর গুণমান এবং পরিষেবার জন্য সন্তুষ্ট
    Afert Tokenizing:  ['সবকিছুর', 'গুণমান', 'এবং', 'পরিষেবার', 'জন্য', 'সন্তুষ্ট']
    Truncating punctuation: ['সবকিছুর', 'গুণমান', 'এবং', 'পরিষেবার', 'জন্য', 'সন্তুষ্ট']
    Truncating StopWords: ['সবকিছুর', 'গুণমান', 'পরিষেবার', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  "ভাংগা ফ্যান দেওয়া হয়েছে, আমার ফ্যান পরিবর্তন করে দেন"
    Afert Tokenizing:  ['ভাংগা', '"', 'ফ্যান', 'দেওয়া', 'হয়েছে', ',', 'আমার', 'ফ্যান', 'পরিবর্তন', 'করে', 'দেন', '"']
    Truncating punctuation: ['ভাংগা', 'ফ্যান', 'দেওয়া', 'হয়েছে', 'আমার', 'ফ্যান', 'পরিবর্তন', 'করে', 'দেন']
    Truncating StopWords: ['ভাংগা', 'ফ্যান', 'হয়েছে', 'ফ্যান', 'পরিবর্তন']
    ***************************************************************************************
    Label:  0
    Sentence:  কপি প্রোডাক্ট।
    Afert Tokenizing:  ['কপি', 'প্রোডাক্ট', '।']
    Truncating punctuation: ['কপি', 'প্রোডাক্ট']
    Truncating StopWords: ['কপি', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  ফ্রেমটা ঠিকমত স্টান্ডিং হয় না
    Afert Tokenizing:  ['ফ্রেমটা', 'ঠিকমত', 'স্টান্ডিং', 'হয়', 'না']
    Truncating punctuation: ['ফ্রেমটা', 'ঠিকমত', 'স্টান্ডিং', 'হয়', 'না']
    Truncating StopWords: ['ফ্রেমটা', 'ঠিকমত', 'স্টান্ডিং', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো করে এডজাস্ট করা যায় না
    Afert Tokenizing:  ['ভালো', 'করে', 'এডজাস্ট', 'করা', 'যায়', 'না']
    Truncating punctuation: ['ভালো', 'করে', 'এডজাস্ট', 'করা', 'যায়', 'না']
    Truncating StopWords: ['ভালো', 'এডজাস্ট', 'যায়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্রচুর ভাইব্রেট হয়,ফ্যান এর মধ্যেই অনেক সমস্যা"
    Afert Tokenizing:  ['প্রচুর', '"', 'ভাইব্রেট', 'হয়,ফ্যান', 'এর', 'মধ্যেই', 'অনেক', 'সমস্যা', '"']
    Truncating punctuation: ['প্রচুর', 'ভাইব্রেট', 'হয়,ফ্যান', 'এর', 'মধ্যেই', 'অনেক', 'সমস্যা']
    Truncating StopWords: ['প্রচুর', 'ভাইব্রেট', 'হয়,ফ্যান', 'সমস্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  বেশি মজবুত না
    Afert Tokenizing:  ['বেশি', 'মজবুত', 'না']
    Truncating punctuation: ['বেশি', 'মজবুত', 'না']
    Truncating StopWords: ['বেশি', 'মজবুত', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  বাজার থেকে সেইম একটা কিনেছি এটার চেয়ে অনেক অনেক স্ট্রং এবং ভাল
    Afert Tokenizing:  ['বাজার', 'থেকে', 'সেইম', 'একটা', 'কিনেছি', 'এটার', 'চেয়ে', 'অনেক', 'অনেক', 'স্ট্রং', 'এবং', 'ভাল']
    Truncating punctuation: ['বাজার', 'থেকে', 'সেইম', 'একটা', 'কিনেছি', 'এটার', 'চেয়ে', 'অনেক', 'অনেক', 'স্ট্রং', 'এবং', 'ভাল']
    Truncating StopWords: ['বাজার', 'সেইম', 'একটা', 'কিনেছি', 'এটার', 'চেয়ে', 'স্ট্রং', 'ভাল']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি খুবই লেইট!
    Afert Tokenizing:  ['ডেলিভারি', 'খুবই', 'লেইট', '!']
    Truncating punctuation: ['ডেলিভারি', 'খুবই', 'লেইট']
    Truncating StopWords: ['ডেলিভারি', 'খুবই', 'লেইট']
    ***************************************************************************************
    Label:  0
    Sentence:  ফ্যানটা স্থীর থাকে না কোন ভাবেই
    Afert Tokenizing:  ['ফ্যানটা', 'স্থীর', 'থাকে', 'না', 'কোন', 'ভাবেই']
    Truncating punctuation: ['ফ্যানটা', 'স্থীর', 'থাকে', 'না', 'কোন', 'ভাবেই']
    Truncating StopWords: ['ফ্যানটা', 'স্থীর', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ফ্যানের খাচাদুটো ফিনিশিং ভাল না।
    Afert Tokenizing:  ['ফ্যানের', 'খাচাদুটো', 'ফিনিশিং', 'ভাল', 'না', '।']
    Truncating punctuation: ['ফ্যানের', 'খাচাদুটো', 'ফিনিশিং', 'ভাল', 'না']
    Truncating StopWords: ['ফ্যানের', 'খাচাদুটো', 'ফিনিশিং', 'ভাল', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম পাতলা খাচা
    Afert Tokenizing:  ['একদম', 'পাতলা', 'খাচা']
    Truncating punctuation: ['একদম', 'পাতলা', 'খাচা']
    Truncating StopWords: ['একদম', 'পাতলা', 'খাচা']
    ***************************************************************************************
    Label:  0
    Sentence:  "পিছনে ভাঙ্গা,ফালতু একটা জিনিস পাতলা খুব প্লাস্টিক"
    Afert Tokenizing:  ['পিছনে', '"', 'ভাঙ্গা,ফালতু', 'একটা', 'জিনিস', 'পাতলা', 'খুব', 'প্লাস্টিক', '"']
    Truncating punctuation: ['পিছনে', 'ভাঙ্গা,ফালতু', 'একটা', 'জিনিস', 'পাতলা', 'খুব', 'প্লাস্টিক']
    Truncating StopWords: ['পিছনে', 'ভাঙ্গা,ফালতু', 'একটা', 'জিনিস', 'পাতলা', 'প্লাস্টিক']
    ***************************************************************************************
    Label:  0
    Sentence:   এক বছরের ওয়ারেন্টি আছে নষ্ট হয়ে গেলে কিভাবে ঠিক করে দিবেন
    Afert Tokenizing:  ['এক', 'বছরের', 'ওয়ারেন্টি', 'আছে', 'নষ্ট', 'হয়ে', 'গেলে', 'কিভাবে', 'ঠিক', 'করে', 'দিবেন']
    Truncating punctuation: ['এক', 'বছরের', 'ওয়ারেন্টি', 'আছে', 'নষ্ট', 'হয়ে', 'গেলে', 'কিভাবে', 'ঠিক', 'করে', 'দিবেন']
    Truncating StopWords: ['এক', 'বছরের', 'ওয়ারেন্টি', 'নষ্ট', 'কিভাবে', 'ঠিক', 'দিবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  "খুবই ভাল মানের ফ্যান নিতে পারেন ধন্যবাদ সেলার"
    Afert Tokenizing:  ['খুবই', '"', 'ভাল', 'মানের', 'ফ্যান', 'নিতে', 'পারেন', 'ধন্যবাদ', 'সেলার', '"']
    Truncating punctuation: ['খুবই', 'ভাল', 'মানের', 'ফ্যান', 'নিতে', 'পারেন', 'ধন্যবাদ', 'সেলার']
    Truncating StopWords: ['খুবই', 'ভাল', 'মানের', 'ফ্যান', 'ধন্যবাদ', 'সেলার']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ভালো বাতাস হয় প্রোডাকটা ও ভালো
    Afert Tokenizing:  ['অনেক', 'ভালো', 'বাতাস', 'হয়', 'প্রোডাকটা', 'ও', 'ভালো']
    Truncating punctuation: ['অনেক', 'ভালো', 'বাতাস', 'হয়', 'প্রোডাকটা', 'ও', 'ভালো']
    Truncating StopWords: ['ভালো', 'বাতাস', 'প্রোডাকটা', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  সস্তা কিন্তু কাজের না
    Afert Tokenizing:  ['সস্তা', 'কিন্তু', 'কাজের', 'না']
    Truncating punctuation: ['সস্তা', 'কিন্তু', 'কাজের', 'না']
    Truncating StopWords: ['সস্তা', 'কাজের', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে রেইনকোট এক বার পড়েছি পেন্ট এ নিছ ছিড়ে গেছে
    Afert Tokenizing:  ['বাজে', 'রেইনকোট', 'এক', 'বার', 'পড়েছি', 'পেন্ট', 'এ', 'নিছ', 'ছিড়ে', 'গেছে']
    Truncating punctuation: ['বাজে', 'রেইনকোট', 'এক', 'বার', 'পড়েছি', 'পেন্ট', 'এ', 'নিছ', 'ছিড়ে', 'গেছে']
    Truncating StopWords: ['বাজে', 'রেইনকোট', 'এক', 'পড়েছি', 'পেন্ট', 'নিছ', 'ছিড়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  100% ওয়াটারপ্রুফ বাজে কথা মিথ্যা
    Afert Tokenizing:  ['100%', 'ওয়াটারপ্রুফ', 'বাজে', 'কথা', 'মিথ্যা']
    Truncating punctuation: ['100%', 'ওয়াটারপ্রুফ', 'বাজে', 'কথা', 'মিথ্যা']
    Truncating StopWords: ['100%', 'ওয়াটারপ্রুফ', 'বাজে', 'কথা', 'মিথ্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  ছেরা ছিলো  বুজলাম না কি বাবে এটা দিলেন
    Afert Tokenizing:  ['ছেরা', 'ছিলো', 'বুজলাম', 'না', 'কি', 'বাবে', 'এটা', 'দিলেন']
    Truncating punctuation: ['ছেরা', 'ছিলো', 'বুজলাম', 'না', 'কি', 'বাবে', 'এটা', 'দিলেন']
    Truncating StopWords: ['ছেরা', 'ছিলো', 'বুজলাম', 'না', 'বাবে']
    ***************************************************************************************
    Label:  1
    Sentence:  বিল্ড কোয়ালিটি খুব ভাল
    Afert Tokenizing:  ['বিল্ড', 'কোয়ালিটি', 'খুব', 'ভাল']
    Truncating punctuation: ['বিল্ড', 'কোয়ালিটি', 'খুব', 'ভাল']
    Truncating StopWords: ['বিল্ড', 'কোয়ালিটি', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  "গায়ের সাথে ভাল ফিটেষ্ট হয়ছে, ধন্যবাদ সেলার কে"
    Afert Tokenizing:  ['গায়ের', '"', 'সাথে', 'ভাল', 'ফিটেষ্ট', 'হয়ছে', ',', 'ধন্যবাদ', 'সেলার', 'কে', '"']
    Truncating punctuation: ['গায়ের', 'সাথে', 'ভাল', 'ফিটেষ্ট', 'হয়ছে', 'ধন্যবাদ', 'সেলার', 'কে']
    Truncating StopWords: ['গায়ের', 'সাথে', 'ভাল', 'ফিটেষ্ট', 'হয়ছে', 'ধন্যবাদ', 'সেলার']
    ***************************************************************************************
    Label:  0
    Sentence:  গলার এইদিকে আরেকটু বড় হইলে ভালো হইতো
    Afert Tokenizing:  ['গলার', 'এইদিকে', 'আরেকটু', 'বড়', 'হইলে', 'ভালো', 'হইতো']
    Truncating punctuation: ['গলার', 'এইদিকে', 'আরেকটু', 'বড়', 'হইলে', 'ভালো', 'হইতো']
    Truncating StopWords: ['গলার', 'এইদিকে', 'আরেকটু', 'বড়', 'হইলে', 'ভালো', 'হইতো']
    ***************************************************************************************
    Label:  0
    Sentence:  প্যান্ট ফেটে গেছে বসার সাথে সাথে
    Afert Tokenizing:  ['প্যান্ট', 'ফেটে', 'গেছে', 'বসার', 'সাথে', 'সাথে']
    Truncating punctuation: ['প্যান্ট', 'ফেটে', 'গেছে', 'বসার', 'সাথে', 'সাথে']
    Truncating StopWords: ['প্যান্ট', 'ফেটে', 'বসার', 'সাথে', 'সাথে']
    ***************************************************************************************
    Label:  0
    Sentence:  প্যান্ট টা বাইক এ যতবার উঠছি হাইটা ততবার ছিড়ে গেছে
    Afert Tokenizing:  ['প্যান্ট', 'টা', 'বাইক', 'এ', 'যতবার', 'উঠছি', 'হাইটা', 'ততবার', 'ছিড়ে', 'গেছে']
    Truncating punctuation: ['প্যান্ট', 'টা', 'বাইক', 'এ', 'যতবার', 'উঠছি', 'হাইটা', 'ততবার', 'ছিড়ে', 'গেছে']
    Truncating StopWords: ['প্যান্ট', 'টা', 'বাইক', 'যতবার', 'উঠছি', 'হাইটা', 'ততবার', 'ছিড়ে']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো মানের রেইনকোট
    Afert Tokenizing:  ['খুব', 'ভালো', 'মানের', 'রেইনকোট']
    Truncating punctuation: ['খুব', 'ভালো', 'মানের', 'রেইনকোট']
    Truncating StopWords: ['ভালো', 'মানের', 'রেইনকোট']
    ***************************************************************************************
    Label:  0
    Sentence:  এতটাই বাজে কোয়ালিটি যে আমি আমার শত্রু কে পর্যন্ত এই জিনিসটা নিতে বলবো না
    Afert Tokenizing:  ['এতটাই', 'বাজে', 'কোয়ালিটি', 'যে', 'আমি', 'আমার', 'শত্রু', 'কে', 'পর্যন্ত', 'এই', 'জিনিসটা', 'নিতে', 'বলবো', 'না']
    Truncating punctuation: ['এতটাই', 'বাজে', 'কোয়ালিটি', 'যে', 'আমি', 'আমার', 'শত্রু', 'কে', 'পর্যন্ত', 'এই', 'জিনিসটা', 'নিতে', 'বলবো', 'না']
    Truncating StopWords: ['বাজে', 'কোয়ালিটি', 'শত্রু', 'জিনিসটা', 'বলবো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপার সেলার কথা কাজে কোনো মিল নাই
    Afert Tokenizing:  ['বাটপার', 'সেলার', 'কথা', 'কাজে', 'কোনো', 'মিল', 'নাই']
    Truncating punctuation: ['বাটপার', 'সেলার', 'কথা', 'কাজে', 'কোনো', 'মিল', 'নাই']
    Truncating StopWords: ['বাটপার', 'সেলার', 'কথা', 'মিল', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "বাজে রেইন কোর্ট পানি ঢোকে,এক সপ্তাহের মধ্যেই "
    Afert Tokenizing:  ['বাজে', '"', 'রেইন', 'কোর্ট', 'পানি', 'ঢোকে,এক', 'সপ্তাহের', 'মধ্যেই', '', '"']
    Truncating punctuation: ['বাজে', 'রেইন', 'কোর্ট', 'পানি', 'ঢোকে,এক', 'সপ্তাহের', 'মধ্যেই', '']
    Truncating StopWords: ['বাজে', 'রেইন', 'কোর্ট', 'পানি', 'ঢোকে,এক', 'সপ্তাহের', '']
    ***************************************************************************************
    Label:  0
    Sentence:  কাপড়টা বেশি সুবিধা না
    Afert Tokenizing:  ['কাপড়টা', 'বেশি', 'সুবিধা', 'না']
    Truncating punctuation: ['কাপড়টা', 'বেশি', 'সুবিধা', 'না']
    Truncating StopWords: ['কাপড়টা', 'বেশি', 'সুবিধা', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  হাতে পাওয়ার পর ভালই মনে হলো
    Afert Tokenizing:  ['হাতে', 'পাওয়ার', 'পর', 'ভালই', 'মনে', 'হলো']
    Truncating punctuation: ['হাতে', 'পাওয়ার', 'পর', 'ভালই', 'মনে', 'হলো']
    Truncating StopWords: ['হাতে', 'পাওয়ার', 'ভালই']
    ***************************************************************************************
    Label:  1
    Sentence:  যতটুকু চেয়েছিলাম তারচেয়েও ভালো
    Afert Tokenizing:  ['যতটুকু', 'চেয়েছিলাম', 'তারচেয়েও', 'ভালো']
    Truncating punctuation: ['যতটুকু', 'চেয়েছিলাম', 'তারচেয়েও', 'ভালো']
    Truncating StopWords: ['যতটুকু', 'চেয়েছিলাম', 'তারচেয়েও', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  পায়ের নিচে চেইন একটি নষ্ট
    Afert Tokenizing:  ['পায়ের', 'নিচে', 'চেইন', 'একটি', 'নষ্ট']
    Truncating punctuation: ['পায়ের', 'নিচে', 'চেইন', 'একটি', 'নষ্ট']
    Truncating StopWords: ['পায়ের', 'নিচে', 'চেইন', 'নষ্ট']
    ***************************************************************************************
    Label:  0
    Sentence:  একটার ৩টা বাল্ব জ্বলে আরেকটা পুরাই নষ্ট
    Afert Tokenizing:  ['একটার', '৩টা', 'বাল্ব', 'জ্বলে', 'আরেকটা', 'পুরাই', 'নষ্ট']
    Truncating punctuation: ['একটার', '৩টা', 'বাল্ব', 'জ্বলে', 'আরেকটা', 'পুরাই', 'নষ্ট']
    Truncating StopWords: ['একটার', '৩টা', 'বাল্ব', 'জ্বলে', 'আরেকটা', 'পুরাই', 'নষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  বাসায় নিয়ে এসে ১২ ভোল্ট এডেপটার দিয়ে চেক করে দেখলাম আলো ভালই
    Afert Tokenizing:  ['বাসায়', 'নিয়ে', 'এসে', '১২', 'ভোল্ট', 'এডেপটার', 'দিয়ে', 'চেক', 'করে', 'দেখলাম', 'আলো', 'ভালই']
    Truncating punctuation: ['বাসায়', 'নিয়ে', 'এসে', '১২', 'ভোল্ট', 'এডেপটার', 'দিয়ে', 'চেক', 'করে', 'দেখলাম', 'আলো', 'ভালই']
    Truncating StopWords: ['বাসায়', '১২', 'ভোল্ট', 'এডেপটার', 'দিয়ে', 'চেক', 'দেখলাম', 'আলো', 'ভালই']
    ***************************************************************************************
    Label:  1
    Sentence:  আলো অনেক ভাল দাম অনুযায়ী
    Afert Tokenizing:  ['আলো', 'অনেক', 'ভাল', 'দাম', 'অনুযায়ী']
    Truncating punctuation: ['আলো', 'অনেক', 'ভাল', 'দাম', 'অনুযায়ী']
    Truncating StopWords: ['আলো', 'ভাল', 'দাম', 'অনুযায়ী']
    ***************************************************************************************
    Label:  1
    Sentence:  এতো কম মূল্যে খুব ই ভালো লাইট পেয়েছি লাইট হিসেবে আলো যথেষ্ট ভালো
    Afert Tokenizing:  ['এতো', 'কম', 'মূল্যে', 'খুব', 'ই', 'ভালো', 'লাইট', 'পেয়েছি', 'লাইট', 'হিসেবে', 'আলো', 'যথেষ্ট', 'ভালো']
    Truncating punctuation: ['এতো', 'কম', 'মূল্যে', 'খুব', 'ই', 'ভালো', 'লাইট', 'পেয়েছি', 'লাইট', 'হিসেবে', 'আলো', 'যথেষ্ট', 'ভালো']
    Truncating StopWords: ['এতো', 'কম', 'মূল্যে', 'ভালো', 'লাইট', 'পেয়েছি', 'লাইট', 'হিসেবে', 'আলো', 'যথেষ্ট', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:   লাইট কত দিন টিকবে বলা যাচ্ছে না
    Afert Tokenizing:  ['লাইট', 'কত', 'দিন', 'টিকবে', 'বলা', 'যাচ্ছে', 'না']
    Truncating punctuation: ['লাইট', 'কত', 'দিন', 'টিকবে', 'বলা', 'যাচ্ছে', 'না']
    Truncating StopWords: ['লাইট', 'টিকবে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  যেমন চেয়েছি তা পাইনি ভেতরে প্লাস্টিক
    Afert Tokenizing:  ['যেমন', 'চেয়েছি', 'তা', 'পাইনি', 'ভেতরে', 'প্লাস্টিক']
    Truncating punctuation: ['যেমন', 'চেয়েছি', 'তা', 'পাইনি', 'ভেতরে', 'প্লাস্টিক']
    Truncating StopWords: ['চেয়েছি', 'পাইনি', 'ভেতরে', 'প্লাস্টিক']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের প্রোডাক্টটি হাতে পেয়েছি কিন্তু বাক্সটি ভাঙ্গা
    Afert Tokenizing:  ['আপনাদের', 'প্রোডাক্টটি', 'হাতে', 'পেয়েছি', 'কিন্তু', 'বাক্সটি', 'ভাঙ্গা']
    Truncating punctuation: ['আপনাদের', 'প্রোডাক্টটি', 'হাতে', 'পেয়েছি', 'কিন্তু', 'বাক্সটি', 'ভাঙ্গা']
    Truncating StopWords: ['আপনাদের', 'প্রোডাক্টটি', 'হাতে', 'পেয়েছি', 'বাক্সটি', 'ভাঙ্গা']
    ***************************************************************************************
    Label:  0
    Sentence:  বাক্সটি ভাঙ্গা! ফলে স্ক্রুগুলো হাড়িয়ে যাওয়ার সম্ভাবনা বেরে গেছে।
    Afert Tokenizing:  ['বাক্সটি', 'ভাঙ্গা', '!', 'ফলে', 'স্ক্রুগুলো', 'হাড়িয়ে', 'যাওয়ার', 'সম্ভাবনা', 'বেরে', 'গেছে', '।']
    Truncating punctuation: ['বাক্সটি', 'ভাঙ্গা', 'ফলে', 'স্ক্রুগুলো', 'হাড়িয়ে', 'যাওয়ার', 'সম্ভাবনা', 'বেরে', 'গেছে']
    Truncating StopWords: ['বাক্সটি', 'ভাঙ্গা', 'স্ক্রুগুলো', 'হাড়িয়ে', 'যাওয়ার', 'সম্ভাবনা', 'বেরে']
    ***************************************************************************************
    Label:  0
    Sentence:  "একবারে বাজে প্রোডাক্ট বক্সটা ফাটা দুই জায়গায়,"
    Afert Tokenizing:  ['একবারে', '"', 'বাজে', 'প্রোডাক্ট', 'বক্সটা', 'ফাটা', 'দুই', 'জায়গায়,', '"']
    Truncating punctuation: ['একবারে', 'বাজে', 'প্রোডাক্ট', 'বক্সটা', 'ফাটা', 'দুই', 'জায়গায়,']
    Truncating StopWords: ['একবারে', 'বাজে', 'প্রোডাক্ট', 'বক্সটা', 'ফাটা', 'জায়গায়,']
    ***************************************************************************************
    Label:  0
    Sentence:  "ছবির সাথে মিল আছে তবে এক্কেবাড়ে ছোট্টো,"
    Afert Tokenizing:  ['ছবির', '"', 'সাথে', 'মিল', 'আছে', 'তবে', 'এক্কেবাড়ে', 'ছোট্টো,', '"']
    Truncating punctuation: ['ছবির', 'সাথে', 'মিল', 'আছে', 'তবে', 'এক্কেবাড়ে', 'ছোট্টো,']
    Truncating StopWords: ['ছবির', 'সাথে', 'মিল', 'এক্কেবাড়ে', 'ছোট্টো,']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব ভালো প্রডাক্ট সন্তোষজনক
    Afert Tokenizing:  ['খুব', 'ভালো', 'প্রডাক্ট', 'সন্তোষজনক']
    Truncating punctuation: ['খুব', 'ভালো', 'প্রডাক্ট', 'সন্তোষজনক']
    Truncating StopWords: ['ভালো', 'প্রডাক্ট', 'সন্তোষজনক']
    ***************************************************************************************
    Label:  0
    Sentence:  "বাজে রেইন কোর্ট প্যন্ট ছিরে গেছে,কাপর ভালোনা"
    Afert Tokenizing:  ['বাজে', '"', 'রেইন', 'কোর্ট', 'প্যন্ট', 'ছিরে', 'গেছে,কাপর', 'ভালোনা', '"']
    Truncating punctuation: ['বাজে', 'রেইন', 'কোর্ট', 'প্যন্ট', 'ছিরে', 'গেছে,কাপর', 'ভালোনা']
    Truncating StopWords: ['বাজে', 'রেইন', 'কোর্ট', 'প্যন্ট', 'ছিরে', 'গেছে,কাপর', 'ভালোনা']
    ***************************************************************************************
    Label:  0
    Sentence:  লেখা আছে ৩২ কিন্তু আছে ৩০ আপনার এই প্রডাক টা না কেনার পরামর্শ দিলাম
    Afert Tokenizing:  ['লেখা', 'আছে', '৩২', 'কিন্তু', 'আছে', '৩০', 'আপনার', 'এই', 'প্রডাক', 'টা', 'না', 'কেনার', 'পরামর্শ', 'দিলাম']
    Truncating punctuation: ['লেখা', 'আছে', '৩২', 'কিন্তু', 'আছে', '৩০', 'আপনার', 'এই', 'প্রডাক', 'টা', 'না', 'কেনার', 'পরামর্শ', 'দিলাম']
    Truncating StopWords: ['লেখা', '৩২', '৩০', 'প্রডাক', 'টা', 'না', 'কেনার', 'পরামর্শ', 'দিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  "খুব ভালো মানের প্রডাক এটা,এটার দাম বাইরে,৪৫০ টাকা"
    Afert Tokenizing:  ['খুব', '"', 'ভালো', 'মানের', 'প্রডাক', 'এটা,এটার', 'দাম', 'বাইরে,৪৫০', 'টাকা', '"']
    Truncating punctuation: ['খুব', 'ভালো', 'মানের', 'প্রডাক', 'এটা,এটার', 'দাম', 'বাইরে,৪৫০', 'টাকা']
    Truncating StopWords: ['ভালো', 'মানের', 'প্রডাক', 'এটা,এটার', 'দাম', 'বাইরে,৪৫০', 'টাকা']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু একবার ব্যবহারেই ড্রাইভারের দাত বাকা হয়ে গেছে
    Afert Tokenizing:  ['ফালতু', 'একবার', 'ব্যবহারেই', 'ড্রাইভারের', 'দাত', 'বাকা', 'হয়ে', 'গেছে']
    Truncating punctuation: ['ফালতু', 'একবার', 'ব্যবহারেই', 'ড্রাইভারের', 'দাত', 'বাকা', 'হয়ে', 'গেছে']
    Truncating StopWords: ['ফালতু', 'ব্যবহারেই', 'ড্রাইভারের', 'দাত', 'বাকা', 'হয়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  যেগুলো দরকার ছিল সেই গুলো নেই
    Afert Tokenizing:  ['যেগুলো', 'দরকার', 'ছিল', 'সেই', 'গুলো', 'নেই']
    Truncating punctuation: ['যেগুলো', 'দরকার', 'ছিল', 'সেই', 'গুলো', 'নেই']
    Truncating StopWords: ['যেগুলো', 'দরকার', 'গুলো', 'নেই']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো না কিনিয়ে না লছ হবে
    Afert Tokenizing:  ['ভালো', 'না', 'কিনিয়ে', 'না', 'লছ', 'হবে']
    Truncating punctuation: ['ভালো', 'না', 'কিনিয়ে', 'না', 'লছ', 'হবে']
    Truncating StopWords: ['ভালো', 'না', 'কিনিয়ে', 'না', 'লছ']
    ***************************************************************************************
    Label:  0
    Sentence:  মাল হাতে পেয়ে হতাশ হলাম
    Afert Tokenizing:  ['মাল', 'হাতে', 'পেয়ে', 'হতাশ', 'হলাম']
    Truncating punctuation: ['মাল', 'হাতে', 'পেয়ে', 'হতাশ', 'হলাম']
    Truncating StopWords: ['মাল', 'হাতে', 'হতাশ', 'হলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  ব্যাটারি টা এখনো পর্যন্ত বেশ ভালো পাওয়ার সাপ্লাই দিচ্ছে
    Afert Tokenizing:  ['ব্যাটারি', 'টা', 'এখনো', 'পর্যন্ত', 'বেশ', 'ভালো', 'পাওয়ার', 'সাপ্লাই', 'দিচ্ছে']
    Truncating punctuation: ['ব্যাটারি', 'টা', 'এখনো', 'পর্যন্ত', 'বেশ', 'ভালো', 'পাওয়ার', 'সাপ্লাই', 'দিচ্ছে']
    Truncating StopWords: ['ব্যাটারি', 'টা', 'এখনো', 'ভালো', 'পাওয়ার', 'সাপ্লাই', 'দিচ্ছে']
    ***************************************************************************************
    Label:  1
    Sentence:  "প্রডাক্ট অনেক ভালো, আমি সেকেন্ড টাইম নিলাম"
    Afert Tokenizing:  ['প্রডাক্ট', '"', 'অনেক', 'ভালো', ',', 'আমি', 'সেকেন্ড', 'টাইম', 'নিলাম', '"']
    Truncating punctuation: ['প্রডাক্ট', 'অনেক', 'ভালো', 'আমি', 'সেকেন্ড', 'টাইম', 'নিলাম']
    Truncating StopWords: ['প্রডাক্ট', 'ভালো', 'সেকেন্ড', 'টাইম', 'নিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় চমৎকার। ধন্যবাদ সংশ্লিষ্ট সবাইকে
    Afert Tokenizing:  ['এক', 'কথায়', 'চমৎকার', '।', 'ধন্যবাদ', 'সংশ্লিষ্ট', 'সবাইকে']
    Truncating punctuation: ['এক', 'কথায়', 'চমৎকার', 'ধন্যবাদ', 'সংশ্লিষ্ট', 'সবাইকে']
    Truncating StopWords: ['এক', 'কথায়', 'চমৎকার', 'ধন্যবাদ', 'সংশ্লিষ্ট', 'সবাইকে']
    ***************************************************************************************
    Label:  1
    Sentence:  "আমি এই প্রোডাক্ট গতকালই রিসিভ করেছি, কিন্তু আজো এটেম্পট টু ডেলিভারি দেখাচ্ছে"
    Afert Tokenizing:  ['আমি', '"', 'এই', 'প্রোডাক্ট', 'গতকালই', 'রিসিভ', 'করেছি', ',', 'কিন্তু', 'আজো', 'এটেম্পট', 'টু', 'ডেলিভারি', 'দেখাচ্ছে', '"']
    Truncating punctuation: ['আমি', 'এই', 'প্রোডাক্ট', 'গতকালই', 'রিসিভ', 'করেছি', 'কিন্তু', 'আজো', 'এটেম্পট', 'টু', 'ডেলিভারি', 'দেখাচ্ছে']
    Truncating StopWords: ['প্রোডাক্ট', 'গতকালই', 'রিসিভ', 'করেছি', 'আজো', 'এটেম্পট', 'টু', 'ডেলিভারি', 'দেখাচ্ছে']
    ***************************************************************************************
    Label:  0
    Sentence:  আর কত দিন হলে ডেলিভারিটা পাবো ? আপনাদের সার্ভিস এতো বাজে কেন !
    Afert Tokenizing:  ['আর', 'কত', 'দিন', 'হলে', 'ডেলিভারিটা', 'পাবো', '', '?', 'আপনাদের', 'সার্ভিস', 'এতো', 'বাজে', 'কেন', '', '!']
    Truncating punctuation: ['আর', 'কত', 'দিন', 'হলে', 'ডেলিভারিটা', 'পাবো', '', 'আপনাদের', 'সার্ভিস', 'এতো', 'বাজে', 'কেন', '']
    Truncating StopWords: ['ডেলিভারিটা', 'পাবো', '', 'আপনাদের', 'সার্ভিস', 'এতো', 'বাজে', '']
    ***************************************************************************************
    Label:  0
    Sentence:  সকালে ডেলিভারির জন্য বের হয়েছে এখনো পর্যন্ত ডেলিভারি দেয়ার নাম নাই
    Afert Tokenizing:  ['সকালে', 'ডেলিভারির', 'জন্য', 'বের', 'হয়েছে', 'এখনো', 'পর্যন্ত', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    Truncating punctuation: ['সকালে', 'ডেলিভারির', 'জন্য', 'বের', 'হয়েছে', 'এখনো', 'পর্যন্ত', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    Truncating StopWords: ['সকালে', 'ডেলিভারির', 'বের', 'হয়েছে', 'এখনো', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "ইদের আগে অর্ডার করছি,৭ জুলাই দেওয়ার কথা ছিলো। কিন্তু আজও কোনো খোঁজ নেই। কেন? আদোও কি পণ্য পাবো?"
    Afert Tokenizing:  ['ইদের', '"', 'আগে', 'অর্ডার', 'করছি,৭', 'জুলাই', 'দেওয়ার', 'কথা', 'ছিলো', '।', 'কিন্তু', 'আজও', 'কোনো', 'খোঁজ', 'নেই', '।', 'কেন', '?', 'আদোও', 'কি', 'পণ্য', 'পাবো?', '"']
    Truncating punctuation: ['ইদের', 'আগে', 'অর্ডার', 'করছি,৭', 'জুলাই', 'দেওয়ার', 'কথা', 'ছিলো', 'কিন্তু', 'আজও', 'কোনো', 'খোঁজ', 'নেই', 'কেন', 'আদোও', 'কি', 'পণ্য', 'পাবো?']
    Truncating StopWords: ['ইদের', 'অর্ডার', 'করছি,৭', 'জুলাই', 'দেওয়ার', 'কথা', 'ছিলো', 'আজও', 'খোঁজ', 'নেই', 'আদোও', 'পণ্য', 'পাবো?']
    ***************************************************************************************
    Label:  0
    Sentence:  "অফার না দিবেন, দিবেন না,হুদাই দিয়ে মানুষ কে হয়রান করার কি আছে"
    Afert Tokenizing:  ['অফার', '"', 'না', 'দিবেন', ',', 'দিবেন', 'না,হুদাই', 'দিয়ে', 'মানুষ', 'কে', 'হয়রান', 'করার', 'কি', 'আছে', '"']
    Truncating punctuation: ['অফার', 'না', 'দিবেন', 'দিবেন', 'না,হুদাই', 'দিয়ে', 'মানুষ', 'কে', 'হয়রান', 'করার', 'কি', 'আছে']
    Truncating StopWords: ['অফার', 'না', 'দিবেন', 'দিবেন', 'না,হুদাই', 'দিয়ে', 'মানুষ', 'হয়রান']
    ***************************************************************************************
    Label:  0
    Sentence:  "কাল কোড ইউস করে রিচার্জ দিলাম ক্যান্সেল করে দিলো! রিফান্ড যে কয়দিনে দিবে।"
    Afert Tokenizing:  ['কাল', '"', 'কোড', 'ইউস', 'করে', 'রিচার্জ', 'দিলাম', 'ক্যান্সেল', 'করে', 'দিলো', '!', 'রিফান্ড', 'যে', 'কয়দিনে', 'দিবে।', '"']
    Truncating punctuation: ['কাল', 'কোড', 'ইউস', 'করে', 'রিচার্জ', 'দিলাম', 'ক্যান্সেল', 'করে', 'দিলো', 'রিফান্ড', 'যে', 'কয়দিনে', 'দিবে।']
    Truncating StopWords: ['কাল', 'কোড', 'ইউস', 'রিচার্জ', 'দিলাম', 'ক্যান্সেল', 'দিলো', 'রিফান্ড', 'কয়দিনে', 'দিবে।']
    ***************************************************************************************
    Label:  0
    Sentence:  এই অর্ডার ডেলিভারির কোন খবর নাই। নিজে থেকে ফেইলড এ্যাটেম্পট দেখাইলো। সব খেয়াল খুশি মত করতেসে।
    Afert Tokenizing:  ['এই', 'অর্ডার', 'ডেলিভারির', 'কোন', 'খবর', 'নাই', '।', 'নিজে', 'থেকে', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', '।', 'সব', 'খেয়াল', 'খুশি', 'মত', 'করতেসে', '।']
    Truncating punctuation: ['এই', 'অর্ডার', 'ডেলিভারির', 'কোন', 'খবর', 'নাই', 'নিজে', 'থেকে', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', 'সব', 'খেয়াল', 'খুশি', 'মত', 'করতেসে']
    Truncating StopWords: ['অর্ডার', 'ডেলিভারির', 'খবর', 'নাই', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', 'খেয়াল', 'খুশি', 'মত', 'করতেসে']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ ডেলিভারি চার্জ অতিরিক্ত বেশি নেওয়া হচ্ছে বর্তমানে
    Afert Tokenizing:  ['দারাজ', 'ডেলিভারি', 'চার্জ', 'অতিরিক্ত', 'বেশি', 'নেওয়া', 'হচ্ছে', 'বর্তমানে']
    Truncating punctuation: ['দারাজ', 'ডেলিভারি', 'চার্জ', 'অতিরিক্ত', 'বেশি', 'নেওয়া', 'হচ্ছে', 'বর্তমানে']
    Truncating StopWords: ['দারাজ', 'ডেলিভারি', 'চার্জ', 'অতিরিক্ত', 'বেশি', 'বর্তমানে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক বড় সু্যোগ দিলেন
    Afert Tokenizing:  ['অনেক', 'বড়', 'সু্যোগ', 'দিলেন']
    Truncating punctuation: ['অনেক', 'বড়', 'সু্যোগ', 'দিলেন']
    Truncating StopWords: ['বড়', 'সু্যোগ']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক উপকার হলো
    Afert Tokenizing:  ['অনেক', 'উপকার', 'হলো']
    Truncating punctuation: ['অনেক', 'উপকার', 'হলো']
    Truncating StopWords: ['উপকার']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই ১ তারিখে মাইক্রোফোন অর্ডার করছি। এখনও পাইনি।জামগড়া আরফান মার্কেট।
    Afert Tokenizing:  ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', '।', 'এখনও', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট', '।']
    Truncating punctuation: ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', 'এখনও', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট']
    Truncating StopWords: ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি ৪ তারিখে মোট ১১ টা জিনিস অর্ডার দিয়েছি এখনো পেলাম না ডেলিভারি
    Afert Tokenizing:  ['আমি', '৪', 'তারিখে', 'মোট', '১১', 'টা', 'জিনিস', 'অর্ডার', 'দিয়েছি', 'এখনো', 'পেলাম', 'না', 'ডেলিভারি']
    Truncating punctuation: ['আমি', '৪', 'তারিখে', 'মোট', '১১', 'টা', 'জিনিস', 'অর্ডার', 'দিয়েছি', 'এখনো', 'পেলাম', 'না', 'ডেলিভারি']
    Truncating StopWords: ['৪', 'তারিখে', '১১', 'টা', 'জিনিস', 'অর্ডার', 'দিয়েছি', 'এখনো', 'পেলাম', 'না', 'ডেলিভারি']
    ***************************************************************************************
    Label:  0
    Sentence:  সিলেট থেকে কিনতে পারছি না কেন
    Afert Tokenizing:  ['সিলেট', 'থেকে', 'কিনতে', 'পারছি', 'না', 'কেন']
    Truncating punctuation: ['সিলেট', 'থেকে', 'কিনতে', 'পারছি', 'না', 'কেন']
    Truncating StopWords: ['সিলেট', 'কিনতে', 'পারছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজের নামে প্রতারণার মামলা দিবো আমি। দুই নাম্বার মাল পাঠাই।
    Afert Tokenizing:  ['দারাজের', 'নামে', 'প্রতারণার', 'মামলা', 'দিবো', 'আমি', '।', 'দুই', 'নাম্বার', 'মাল', 'পাঠাই', '।']
    Truncating punctuation: ['দারাজের', 'নামে', 'প্রতারণার', 'মামলা', 'দিবো', 'আমি', 'দুই', 'নাম্বার', 'মাল', 'পাঠাই']
    Truncating StopWords: ['দারাজের', 'নামে', 'প্রতারণার', 'মামলা', 'দিবো', 'নাম্বার', 'মাল', 'পাঠাই']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারির সময়ের মধ্যে জিনিস না পাওয়া আমার রাগন্নিত মন
    Afert Tokenizing:  ['ডেলিভারির', 'সময়ের', 'মধ্যে', 'জিনিস', 'না', 'পাওয়া', 'আমার', 'রাগন্নিত', 'মন']
    Truncating punctuation: ['ডেলিভারির', 'সময়ের', 'মধ্যে', 'জিনিস', 'না', 'পাওয়া', 'আমার', 'রাগন্নিত', 'মন']
    Truncating StopWords: ['ডেলিভারির', 'সময়ের', 'জিনিস', 'না', 'রাগন্নিত', 'মন']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি অনেক অনেক স্লো আরেকটু ফাস্ট করলে ভালো হতো
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'অনেক', 'অনেক', 'স্লো', 'আরেকটু', 'ফাস্ট', 'করলে', 'ভালো', 'হতো']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'অনেক', 'অনেক', 'স্লো', 'আরেকটু', 'ফাস্ট', 'করলে', 'ভালো', 'হতো']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'স্লো', 'আরেকটু', 'ফাস্ট', 'ভালো', 'হতো']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমি অর্ডার করছি ৪ তারিখে আসার কথা ছিল ৭ থেকে ১০ তারিখে কিন্তু আসকে ১২ তারিখ এখো আসে নাই কেন দয়া করে জানাবেন
    Afert Tokenizing:  ['ভাই', 'আমি', 'অর্ডার', 'করছি', '৪', 'তারিখে', 'আসার', 'কথা', 'ছিল', '৭', 'থেকে', '১০', 'তারিখে', 'কিন্তু', 'আসকে', '১২', 'তারিখ', 'এখো', 'আসে', 'নাই', 'কেন', 'দয়া', 'করে', 'জানাবেন']
    Truncating punctuation: ['ভাই', 'আমি', 'অর্ডার', 'করছি', '৪', 'তারিখে', 'আসার', 'কথা', 'ছিল', '৭', 'থেকে', '১০', 'তারিখে', 'কিন্তু', 'আসকে', '১২', 'তারিখ', 'এখো', 'আসে', 'নাই', 'কেন', 'দয়া', 'করে', 'জানাবেন']
    Truncating StopWords: ['ভাই', 'অর্ডার', 'করছি', '৪', 'তারিখে', 'আসার', 'কথা', '৭', '১০', 'তারিখে', 'আসকে', '১২', 'তারিখ', 'এখো', 'আসে', 'নাই', 'দয়া', 'জানাবেন']
    ***************************************************************************************
    Label:  0
    Sentence:  কোড কাজ করে না
    Afert Tokenizing:  ['কোড', 'কাজ', 'করে', 'না']
    Truncating punctuation: ['কোড', 'কাজ', 'করে', 'না']
    Truncating StopWords: ['কোড', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ডিসকাউন্ট কোড টি কাজ করেনা কেন
    Afert Tokenizing:  ['ডিসকাউন্ট', 'কোড', 'টি', 'কাজ', 'করেনা', 'কেন']
    Truncating punctuation: ['ডিসকাউন্ট', 'কোড', 'টি', 'কাজ', 'করেনা', 'কেন']
    Truncating StopWords: ['ডিসকাউন্ট', 'কোড', 'করেনা']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাউচার ইউজ করে রিচার্জ করা যাচ্ছে নাতো
    Afert Tokenizing:  ['ভাউচার', 'ইউজ', 'করে', 'রিচার্জ', 'করা', 'যাচ্ছে', 'নাতো']
    Truncating punctuation: ['ভাউচার', 'ইউজ', 'করে', 'রিচার্জ', 'করা', 'যাচ্ছে', 'নাতো']
    Truncating StopWords: ['ভাউচার', 'ইউজ', 'রিচার্জ', 'নাতো']
    ***************************************************************************************
    Label:  1
    Sentence:  আজকে একটি ঈদের দিন । দারাজ থেকে ভালো কোনো সারপ্রাইজ অফার আশা করছি।
    Afert Tokenizing:  ['আজকে', 'একটি', 'ঈদের', 'দিন', '', '।', 'দারাজ', 'থেকে', 'ভালো', 'কোনো', 'সারপ্রাইজ', 'অফার', 'আশা', 'করছি', '।']
    Truncating punctuation: ['আজকে', 'একটি', 'ঈদের', 'দিন', '', 'দারাজ', 'থেকে', 'ভালো', 'কোনো', 'সারপ্রাইজ', 'অফার', 'আশা', 'করছি']
    Truncating StopWords: ['আজকে', 'ঈদের', '', 'দারাজ', 'ভালো', 'সারপ্রাইজ', 'অফার', 'আশা', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার অর্ডার কেনসেল হলো কেনো
    Afert Tokenizing:  ['আমার', 'অর্ডার', 'কেনসেল', 'হলো', 'কেনো']
    Truncating punctuation: ['আমার', 'অর্ডার', 'কেনসেল', 'হলো', 'কেনো']
    Truncating StopWords: ['অর্ডার', 'কেনসেল', 'কেনো']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ রেসপন্স করার জন্য
    Afert Tokenizing:  ['ধন্যবাদ', 'রেসপন্স', 'করার', 'জন্য']
    Truncating punctuation: ['ধন্যবাদ', 'রেসপন্স', 'করার', 'জন্য']
    Truncating StopWords: ['ধন্যবাদ', 'রেসপন্স']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রবলেম সলভ হলে আরো বেশী খুশী হবো। আশা করি আজকের মধ্যেই সলভ হবে
    Afert Tokenizing:  ['প্রবলেম', 'সলভ', 'হলে', 'আরো', 'বেশী', 'খুশী', 'হবো', '।', 'আশা', 'করি', 'আজকের', 'মধ্যেই', 'সলভ', 'হবে']
    Truncating punctuation: ['প্রবলেম', 'সলভ', 'হলে', 'আরো', 'বেশী', 'খুশী', 'হবো', 'আশা', 'করি', 'আজকের', 'মধ্যেই', 'সলভ', 'হবে']
    Truncating StopWords: ['প্রবলেম', 'সলভ', 'আরো', 'বেশী', 'খুশী', 'হবো', 'আশা', 'আজকের', 'সলভ']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম অনুসারে পন্য ভালো না।
    Afert Tokenizing:  ['দাম', 'অনুসারে', 'পন্য', 'ভালো', 'না', '।']
    Truncating punctuation: ['দাম', 'অনুসারে', 'পন্য', 'ভালো', 'না']
    Truncating StopWords: ['দাম', 'অনুসারে', 'পন্য', 'ভালো', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  সর্বকালের সেরা সংগ্রহ
    Afert Tokenizing:  ['সর্বকালের', 'সেরা', 'সংগ্রহ']
    Truncating punctuation: ['সর্বকালের', 'সেরা', 'সংগ্রহ']
    Truncating StopWords: ['সর্বকালের', 'সেরা', 'সংগ্রহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ অনেক ভালো
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'অনেক', 'ভালো']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'অনেক', 'ভালো']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর কালেকশন আপু।
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'কালেকশন', 'আপু', '।']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'কালেকশন', 'আপু']
    Truncating StopWords: ['সুন্দর', 'কালেকশন', 'আপু']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ এডমিন প্যানেল
    Afert Tokenizing:  ['ধন্যবাদ', 'এডমিন', 'প্যানেল']
    Truncating punctuation: ['ধন্যবাদ', 'এডমিন', 'প্যানেল']
    Truncating StopWords: ['ধন্যবাদ', 'এডমিন', 'প্যানেল']
    ***************************************************************************************
    Label:  1
    Sentence:  মাছটা পছন্দ হয়ছে
    Afert Tokenizing:  ['মাছটা', 'পছন্দ', 'হয়ছে']
    Truncating punctuation: ['মাছটা', 'পছন্দ', 'হয়ছে']
    Truncating StopWords: ['মাছটা', 'পছন্দ', 'হয়ছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও এতো বড় মাছ
    Afert Tokenizing:  ['ওয়াও', 'এতো', 'বড়', 'মাছ']
    Truncating punctuation: ['ওয়াও', 'এতো', 'বড়', 'মাছ']
    Truncating StopWords: ['ওয়াও', 'এতো', 'বড়', 'মাছ']
    ***************************************************************************************
    Label:  0
    Sentence:  হটাত করে দরাজ এর ডেলিভারি চার্জ বেড়ে জাওয়ায় অনেক সমস্যা হচ্ছে পন্য কিনতে। কারন আমি গ্রাম এ থাকি আর গ্রাম এর সব মানুষতো আর এতো টাকা ডেলিভারি চার্জ দিয়ে পন্য আনবে না।
    Afert Tokenizing:  ['হটাত', 'করে', 'দরাজ', 'এর', 'ডেলিভারি', 'চার্জ', 'বেড়ে', 'জাওয়ায়', 'অনেক', 'সমস্যা', 'হচ্ছে', 'পন্য', 'কিনতে', '।', 'কারন', 'আমি', 'গ্রাম', 'এ', 'থাকি', 'আর', 'গ্রাম', 'এর', 'সব', 'মানুষতো', 'আর', 'এতো', 'টাকা', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'পন্য', 'আনবে', 'না', '।']
    Truncating punctuation: ['হটাত', 'করে', 'দরাজ', 'এর', 'ডেলিভারি', 'চার্জ', 'বেড়ে', 'জাওয়ায়', 'অনেক', 'সমস্যা', 'হচ্ছে', 'পন্য', 'কিনতে', 'কারন', 'আমি', 'গ্রাম', 'এ', 'থাকি', 'আর', 'গ্রাম', 'এর', 'সব', 'মানুষতো', 'আর', 'এতো', 'টাকা', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'পন্য', 'আনবে', 'না']
    Truncating StopWords: ['হটাত', 'দরাজ', 'ডেলিভারি', 'চার্জ', 'বেড়ে', 'জাওয়ায়', 'সমস্যা', 'পন্য', 'কিনতে', 'কারন', 'গ্রাম', 'থাকি', 'গ্রাম', 'মানুষতো', 'এতো', 'টাকা', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'পন্য', 'আনবে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  তারিখ দেখেন সব পণ্য দিসে নিচের স্টিকার টা দেয় নাই পুনরায় অর্ডার করতে গেলাম তখন দাম আরো বেশী দেখায়
    Afert Tokenizing:  ['তারিখ', 'দেখেন', 'সব', 'পণ্য', 'দিসে', 'নিচের', 'স্টিকার', 'টা', 'দেয়', 'নাই', 'পুনরায়', 'অর্ডার', 'করতে', 'গেলাম', 'তখন', 'দাম', 'আরো', 'বেশী', 'দেখায়']
    Truncating punctuation: ['তারিখ', 'দেখেন', 'সব', 'পণ্য', 'দিসে', 'নিচের', 'স্টিকার', 'টা', 'দেয়', 'নাই', 'পুনরায়', 'অর্ডার', 'করতে', 'গেলাম', 'তখন', 'দাম', 'আরো', 'বেশী', 'দেখায়']
    Truncating StopWords: ['তারিখ', 'দেখেন', 'পণ্য', 'দিসে', 'নিচের', 'স্টিকার', 'টা', 'দেয়', 'নাই', 'পুনরায়', 'অর্ডার', 'গেলাম', 'দাম', 'আরো', 'বেশী', 'দেখায়']
    ***************************************************************************************
    Label:  0
    Sentence:  "এরা সবাই ধান্দাবাজ ঠিক ঠাক মতো পণ্য ডেলিভারি দেয় না এরা
    Afert Tokenizing:  ['এরা', '"', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating punctuation: ['এরা', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating StopWords: ['সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্রোডাক্ট এখনো হাতে পাই নাই, কিন্তু রিসিফ দেখাচ্ছে কেন? আর ১২ তারিখ অর্ডার করছি এখনো হাতে পাইনি, এতো লেট করছে কেন কেউ বলতে পারবেন?"
    Afert Tokenizing:  ['প্রোডাক্ট', '"', 'এখনো', 'হাতে', 'পাই', 'নাই', ',', 'কিন্তু', 'রিসিফ', 'দেখাচ্ছে', 'কেন', '?', 'আর', '১২', 'তারিখ', 'অর্ডার', 'করছি', 'এখনো', 'হাতে', 'পাইনি', ',', 'এতো', 'লেট', 'করছে', 'কেন', 'কেউ', 'বলতে', 'পারবেন?', '"']
    Truncating punctuation: ['প্রোডাক্ট', 'এখনো', 'হাতে', 'পাই', 'নাই', 'কিন্তু', 'রিসিফ', 'দেখাচ্ছে', 'কেন', 'আর', '১২', 'তারিখ', 'অর্ডার', 'করছি', 'এখনো', 'হাতে', 'পাইনি', 'এতো', 'লেট', 'করছে', 'কেন', 'কেউ', 'বলতে', 'পারবেন?']
    Truncating StopWords: ['প্রোডাক্ট', 'এখনো', 'হাতে', 'পাই', 'নাই', 'রিসিফ', 'দেখাচ্ছে', '১২', 'তারিখ', 'অর্ডার', 'করছি', 'এখনো', 'হাতে', 'পাইনি', 'এতো', 'লেট', 'পারবেন?']
    ***************************************************************************************
    Label:  0
    Sentence:  "ডেলিভারির ডেট একদিন পার হয়ে যাওয়ার পরও, প্রোডাক্ট হাতে পেলাম না"
    Afert Tokenizing:  ['ডেলিভারির', '"', 'ডেট', 'একদিন', 'পার', 'হয়ে', 'যাওয়ার', 'পরও', ',', 'প্রোডাক্ট', 'হাতে', 'পেলাম', 'না', '"']
    Truncating punctuation: ['ডেলিভারির', 'ডেট', 'একদিন', 'পার', 'হয়ে', 'যাওয়ার', 'পরও', 'প্রোডাক্ট', 'হাতে', 'পেলাম', 'না']
    Truncating StopWords: ['ডেলিভারির', 'ডেট', 'একদিন', 'পার', 'হয়ে', 'যাওয়ার', 'পরও', 'প্রোডাক্ট', 'হাতে', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনি যখন আপনার পণ্য অর্ডার করেন, ডেলিভারি সঠিকভাবে আসে না"
    Afert Tokenizing:  ['আপনি', '"', 'যখন', 'আপনার', 'পণ্য', 'অর্ডার', 'করেন', ',', 'ডেলিভারি', 'সঠিকভাবে', 'আসে', 'না', '"']
    Truncating punctuation: ['আপনি', 'যখন', 'আপনার', 'পণ্য', 'অর্ডার', 'করেন', 'ডেলিভারি', 'সঠিকভাবে', 'আসে', 'না']
    Truncating StopWords: ['পণ্য', 'অর্ডার', 'ডেলিভারি', 'সঠিকভাবে', 'আসে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ কমানো হোক!!!
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!', '!']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ কমানো হোক!!!
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!', '!']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক!!']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা নিয়ে পণ্য দেয় নাই
    Afert Tokenizing:  ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating punctuation: ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating StopWords: ['টাকা', 'পণ্য', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনি আমার মেসেজর রিফলাই দেন না কেন
    Afert Tokenizing:  ['আপনি', '"', 'আমার', 'মেসেজর', 'রিফলাই', 'দেন', 'না', 'কেন']
    Truncating punctuation: ['আপনি', 'আমার', 'মেসেজর', 'রিফলাই', 'দেন', 'না', 'কেন']
    Truncating StopWords: ['মেসেজর', 'রিফলাই', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার পন্য এখনো আসেনি
    Afert Tokenizing:  ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating punctuation: ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating StopWords: ['পন্য', 'এখনো', 'আসেনি']
    ***************************************************************************************
    Label:  0
    Sentence:  গ্রুপে পোস্ট করছি এপ্রুভ করে না।
    Afert Tokenizing:  ['গ্রুপে', 'পোস্ট', 'করছি', 'এপ্রুভ', 'করে', 'না', '।']
    Truncating punctuation: ['গ্রুপে', 'পোস্ট', 'করছি', 'এপ্রুভ', 'করে', 'না']
    Truncating StopWords: ['গ্রুপে', 'পোস্ট', 'করছি', 'এপ্রুভ', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমি আজকে একটা প্রডাক্ট ওর্ডার করছিলাম,,, কিছুক্ষণ পর দেখলাম আমার অর্ডার টা বাতিল করে দিয়েছে!"
    Afert Tokenizing:  ['আমি', '"', 'আজকে', 'একটা', 'প্রডাক্ট', 'ওর্ডার', 'করছিলাম,,', ',', 'কিছুক্ষণ', 'পর', 'দেখলাম', 'আমার', 'অর্ডার', 'টা', 'বাতিল', 'করে', 'দিয়েছে!', '"']
    Truncating punctuation: ['আমি', 'আজকে', 'একটা', 'প্রডাক্ট', 'ওর্ডার', 'করছিলাম,,', 'কিছুক্ষণ', 'পর', 'দেখলাম', 'আমার', 'অর্ডার', 'টা', 'বাতিল', 'করে', 'দিয়েছে!']
    Truncating StopWords: ['আজকে', 'একটা', 'প্রডাক্ট', 'ওর্ডার', 'করছিলাম,,', 'কিছুক্ষণ', 'দেখলাম', 'অর্ডার', 'টা', 'বাতিল', 'দিয়েছে!']
    ***************************************************************************************
    Label:  0
    Sentence:  "কাজ করেনা
    Afert Tokenizing:  ['কাজ', '"', 'করেনা']
    Truncating punctuation: ['কাজ', 'করেনা']
    Truncating StopWords: ['করেনা']
    ***************************************************************************************
    Label:  0
    Sentence:  "এরা সবাই ধান্দাবাজ ঠিক ঠাক মতো পণ্য ডেলিভারি দেয় না এরা
    Afert Tokenizing:  ['এরা', '"', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating punctuation: ['এরা', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating StopWords: ['সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ এর মান তেমন ভালো না... প্রোডাক্ট নরমাল
    Afert Tokenizing:  ['দারাজ', 'এর', 'মান', 'তেমন', 'ভালো', 'না..', '.', 'প্রোডাক্ট', 'নরমাল']
    Truncating punctuation: ['দারাজ', 'এর', 'মান', 'তেমন', 'ভালো', 'না..', 'প্রোডাক্ট', 'নরমাল']
    Truncating StopWords: ['দারাজ', 'মান', 'ভালো', 'না..', 'প্রোডাক্ট', 'নরমাল']
    ***************************************************************************************
    Label:  0
    Sentence:  কাজ করে না
    Afert Tokenizing:  ['কাজ', 'করে', 'না']
    Truncating punctuation: ['কাজ', 'করে', 'না']
    Truncating StopWords: ['না']
    ***************************************************************************************
    Label:  0
    Sentence:  হুদাই একটা পোস্ট করে দিলেন কাজই করে না ভাউচারটি
    Afert Tokenizing:  ['হুদাই', 'একটা', 'পোস্ট', 'করে', 'দিলেন', 'কাজই', 'করে', 'না', 'ভাউচারটি']
    Truncating punctuation: ['হুদাই', 'একটা', 'পোস্ট', 'করে', 'দিলেন', 'কাজই', 'করে', 'না', 'ভাউচারটি']
    Truncating StopWords: ['হুদাই', 'একটা', 'পোস্ট', 'কাজই', 'না', 'ভাউচারটি']
    ***************************************************************************************
    Label:  0
    Sentence:  অডার করলে রিসিভ করছে না
    Afert Tokenizing:  ['অডার', 'করলে', 'রিসিভ', 'করছে', 'না']
    Truncating punctuation: ['অডার', 'করলে', 'রিসিভ', 'করছে', 'না']
    Truncating StopWords: ['অডার', 'রিসিভ', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা নিয়ে পণ্য দেয় নাই
    Afert Tokenizing:  ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating punctuation: ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating StopWords: ['টাকা', 'পণ্য', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম লেখছেননা কেন
    Afert Tokenizing:  ['দাম', 'লেখছেননা', 'কেন']
    Truncating punctuation: ['দাম', 'লেখছেননা', 'কেন']
    Truncating StopWords: ['দাম', 'লেখছেননা']
    ***************************************************************************************
    Label:  0
    Sentence:  বলে এক দেয় আরেক ৷ চিটিং
    Afert Tokenizing:  ['বলে', 'এক', 'দেয়', 'আরেক', '৷', 'চিটিং']
    Truncating punctuation: ['বলে', 'এক', 'দেয়', 'আরেক', '৷', 'চিটিং']
    Truncating StopWords: ['এক', 'আরেক', '৷', 'চিটিং']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার পন্য এখনো আসেনি
    Afert Tokenizing:  ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating punctuation: ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating StopWords: ['পন্য', 'এখনো', 'আসেনি']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাউচার ব্যবহার করা যাচ্ছেনা কেন
    Afert Tokenizing:  ['ভাউচার', 'ব্যবহার', 'করা', 'যাচ্ছেনা', 'কেন']
    Truncating punctuation: ['ভাউচার', 'ব্যবহার', 'করা', 'যাচ্ছেনা', 'কেন']
    Truncating StopWords: ['ভাউচার', 'যাচ্ছেনা']
    ***************************************************************************************
    Label:  1
    Sentence:  ফ্রি ডেলিভারি !
    Afert Tokenizing:  ['ফ্রি', 'ডেলিভারি', '', '!']
    Truncating punctuation: ['ফ্রি', 'ডেলিভারি', '']
    Truncating StopWords: ['ফ্রি', 'ডেলিভারি', '']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি মালের অর্ডার দিয়েছি আমার তো মাল পাইলাম না ভাই মাল কয় তারিখে পাইনি
    Afert Tokenizing:  ['আমি', 'মালের', 'অর্ডার', 'দিয়েছি', 'আমার', 'তো', 'মাল', 'পাইলাম', 'না', 'ভাই', 'মাল', 'কয়', 'তারিখে', 'পাইনি']
    Truncating punctuation: ['আমি', 'মালের', 'অর্ডার', 'দিয়েছি', 'আমার', 'তো', 'মাল', 'পাইলাম', 'না', 'ভাই', 'মাল', 'কয়', 'তারিখে', 'পাইনি']
    Truncating StopWords: ['মালের', 'অর্ডার', 'দিয়েছি', 'মাল', 'পাইলাম', 'না', 'ভাই', 'মাল', 'কয়', 'তারিখে', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  11 দিন হয়ে গেছে অর্ডার দিছি পেমেন্ট ও করা কোনো খবর নাই
    Afert Tokenizing:  ['11', 'দিন', 'হয়ে', 'গেছে', 'অর্ডার', 'দিছি', 'পেমেন্ট', 'ও', 'করা', 'কোনো', 'খবর', 'নাই']
    Truncating punctuation: ['11', 'দিন', 'হয়ে', 'গেছে', 'অর্ডার', 'দিছি', 'পেমেন্ট', 'ও', 'করা', 'কোনো', 'খবর', 'নাই']
    Truncating StopWords: ['11', 'অর্ডার', 'দিছি', 'পেমেন্ট', 'খবর', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "সম্পূর্ণ মিথ্যা কথা আপনাদের
    Afert Tokenizing:  ['সম্পূর্ণ', '"', 'মিথ্যা', 'কথা', 'আপনাদের']
    Truncating punctuation: ['সম্পূর্ণ', 'মিথ্যা', 'কথা', 'আপনাদের']
    Truncating StopWords: ['সম্পূর্ণ', 'মিথ্যা', 'কথা', 'আপনাদের']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি প্রোডাক্ট অর্ডার করেছি আজকে ছয়দিনের উপরে কিন্তু ডেলিভারি পাই নাই।
    Afert Tokenizing:  ['আমি', 'প্রোডাক্ট', 'অর্ডার', 'করেছি', 'আজকে', 'ছয়দিনের', 'উপরে', 'কিন্তু', 'ডেলিভারি', 'পাই', 'নাই', '।']
    Truncating punctuation: ['আমি', 'প্রোডাক্ট', 'অর্ডার', 'করেছি', 'আজকে', 'ছয়দিনের', 'উপরে', 'কিন্তু', 'ডেলিভারি', 'পাই', 'নাই']
    Truncating StopWords: ['প্রোডাক্ট', 'অর্ডার', 'করেছি', 'আজকে', 'ছয়দিনের', 'ডেলিভারি', 'পাই', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার প্রডাক্ট টি এখোনো পাই নি আপনারা ডেলিভারি দিতে এত লেট করেন কেন???আজকে ও ডেলিভারি পাই নি?
    Afert Tokenizing:  ['আমার', 'প্রডাক্ট', 'টি', 'এখোনো', 'পাই', 'নি', 'আপনারা', 'ডেলিভারি', 'দিতে', 'এত', 'লেট', 'করেন', 'কেন???আজকে', 'ও', 'ডেলিভারি', 'পাই', 'নি', '?']
    Truncating punctuation: ['আমার', 'প্রডাক্ট', 'টি', 'এখোনো', 'পাই', 'নি', 'আপনারা', 'ডেলিভারি', 'দিতে', 'এত', 'লেট', 'করেন', 'কেন???আজকে', 'ও', 'ডেলিভারি', 'পাই', 'নি']
    Truncating StopWords: ['প্রডাক্ট', 'এখোনো', 'পাই', 'নি', 'আপনারা', 'ডেলিভারি', 'লেট', 'কেন???আজকে', 'ডেলিভারি', 'পাই', 'নি']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের কাছ কিছু জিজ্ঞেস করলে কোনো উত্তর দেন না কেনো
    Afert Tokenizing:  ['আপনাদের', 'কাছ', 'কিছু', 'জিজ্ঞেস', 'করলে', 'কোনো', 'উত্তর', 'দেন', 'না', 'কেনো']
    Truncating punctuation: ['আপনাদের', 'কাছ', 'কিছু', 'জিজ্ঞেস', 'করলে', 'কোনো', 'উত্তর', 'দেন', 'না', 'কেনো']
    Truncating StopWords: ['আপনাদের', 'জিজ্ঞেস', 'না', 'কেনো']
    ***************************************************************************************
    Label:  0
    Sentence:  "এই ওডারটা এখন ও পেলাম না।
    Afert Tokenizing:  ['এই', '"', 'ওডারটা', 'এখন', 'ও', 'পেলাম', 'না', '।']
    Truncating punctuation: ['এই', 'ওডারটা', 'এখন', 'ও', 'পেলাম', 'না']
    Truncating StopWords: ['ওডারটা', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "কেউ বিশ্বাস করবেন না এরা প্রতারক
    Afert Tokenizing:  ['কেউ', '"', 'বিশ্বাস', 'করবেন', 'না', 'এরা', 'প্রতারক']
    Truncating punctuation: ['কেউ', 'বিশ্বাস', 'করবেন', 'না', 'এরা', 'প্রতারক']
    Truncating StopWords: ['বিশ্বাস', 'না', 'প্রতারক']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা মেরে দিচে
    Afert Tokenizing:  ['আমার', 'টাকা', 'মেরে', 'দিচে']
    Truncating punctuation: ['আমার', 'টাকা', 'মেরে', 'দিচে']
    Truncating StopWords: ['টাকা', 'মেরে', 'দিচে']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ এই মাত্র দারাজ থেকে ফেবরিলাইফ এর টি শার্ট পাইলাম
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'এই', 'মাত্র', 'দারাজ', 'থেকে', 'ফেবরিলাইফ', 'এর', 'টি', 'শার্ট', 'পাইলাম']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'এই', 'মাত্র', 'দারাজ', 'থেকে', 'ফেবরিলাইফ', 'এর', 'টি', 'শার্ট', 'পাইলাম']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'দারাজ', 'ফেবরিলাইফ', 'শার্ট', 'পাইলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  লিংক কাজ করেনা
    Afert Tokenizing:  ['লিংক', 'কাজ', 'করেনা']
    Truncating punctuation: ['লিংক', 'কাজ', 'করেনা']
    Truncating StopWords: ['লিংক', 'করেনা']
    ***************************************************************************************
    Label:  0
    Sentence:  পন্যের চেয়ে ডেলিভারি চার্জ বেশি কেন সেইটা বলেন।
    Afert Tokenizing:  ['পন্যের', 'চেয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'কেন', 'সেইটা', 'বলেন', '।']
    Truncating punctuation: ['পন্যের', 'চেয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'কেন', 'সেইটা', 'বলেন']
    Truncating StopWords: ['পন্যের', 'চেয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'সেইটা']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই আমি কিছু জিনিস ওয়াডার করছি আজকে পাজ দিন হয়ে গেলো এখনও পাইনি কবে পাবো একটু জানাবেন
    Afert Tokenizing:  ['ভাই', 'আমি', 'কিছু', 'জিনিস', 'ওয়াডার', 'করছি', 'আজকে', 'পাজ', 'দিন', 'হয়ে', 'গেলো', 'এখনও', 'পাইনি', 'কবে', 'পাবো', 'একটু', 'জানাবেন']
    Truncating punctuation: ['ভাই', 'আমি', 'কিছু', 'জিনিস', 'ওয়াডার', 'করছি', 'আজকে', 'পাজ', 'দিন', 'হয়ে', 'গেলো', 'এখনও', 'পাইনি', 'কবে', 'পাবো', 'একটু', 'জানাবেন']
    Truncating StopWords: ['ভাই', 'জিনিস', 'ওয়াডার', 'করছি', 'আজকে', 'পাজ', 'হয়ে', 'গেলো', 'পাইনি', 'পাবো', 'একটু', 'জানাবেন']
    ***************************************************************************************
    Label:  1
    Sentence:  সব প্রোডাক্টের কি ডেলিভারি চার্জ ফ্রি
    Afert Tokenizing:  ['সব', 'প্রোডাক্টের', 'কি', 'ডেলিভারি', 'চার্জ', 'ফ্রি']
    Truncating punctuation: ['সব', 'প্রোডাক্টের', 'কি', 'ডেলিভারি', 'চার্জ', 'ফ্রি']
    Truncating StopWords: ['প্রোডাক্টের', 'ডেলিভারি', 'চার্জ', 'ফ্রি']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু ডেলিভারি সার্ভিস
    Afert Tokenizing:  ['ফালতু', 'ডেলিভারি', 'সার্ভিস']
    Truncating punctuation: ['ফালতু', 'ডেলিভারি', 'সার্ভিস']
    Truncating StopWords: ['ফালতু', 'ডেলিভারি', 'সার্ভিস']
    ***************************************************************************************
    Label:  0
    Sentence:  হট লাইনে কল করে কোন সাহায্য পাওয়া যায় না। বাজে সার্ভিস
    Afert Tokenizing:  ['হট', 'লাইনে', 'কল', 'করে', 'কোন', 'সাহায্য', 'পাওয়া', 'যায়', 'না', '।', 'বাজে', 'সার্ভিস']
    Truncating punctuation: ['হট', 'লাইনে', 'কল', 'করে', 'কোন', 'সাহায্য', 'পাওয়া', 'যায়', 'না', 'বাজে', 'সার্ভিস']
    Truncating StopWords: ['হট', 'লাইনে', 'কল', 'সাহায্য', 'পাওয়া', 'যায়', 'না', 'বাজে', 'সার্ভিস']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ কাস্টমার এর কোনো ধরনের সুবিধা দেখে না।
    Afert Tokenizing:  ['দারাজ', 'কাস্টমার', 'এর', 'কোনো', 'ধরনের', 'সুবিধা', 'দেখে', 'না', '।']
    Truncating punctuation: ['দারাজ', 'কাস্টমার', 'এর', 'কোনো', 'ধরনের', 'সুবিধা', 'দেখে', 'না']
    Truncating StopWords: ['দারাজ', 'কাস্টমার', 'ধরনের', 'সুবিধা', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি একটা প্রোডাক্টস অর্ডার করে ২৮ দিন অপেক্ষায় ছিলাম।
    Afert Tokenizing:  ['আমি', 'একটা', 'প্রোডাক্টস', 'অর্ডার', 'করে', '২৮', 'দিন', 'অপেক্ষায়', 'ছিলাম', '।']
    Truncating punctuation: ['আমি', 'একটা', 'প্রোডাক্টস', 'অর্ডার', 'করে', '২৮', 'দিন', 'অপেক্ষায়', 'ছিলাম']
    Truncating StopWords: ['একটা', 'প্রোডাক্টস', 'অর্ডার', '২৮', 'অপেক্ষায়', 'ছিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের ধন্যবাদ।
    Afert Tokenizing:  ['আপনাদের', 'ধন্যবাদ', '।']
    Truncating punctuation: ['আপনাদের', 'ধন্যবাদ']
    Truncating StopWords: ['আপনাদের', 'ধন্যবাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনারা বলেছেন : আমার ওর্ডার ক্যান্সেল হয়েছে। আমি অনুরোধ করবো, আমার মতো কাউকে এভাবে ঘুরাবেন না।"
    Afert Tokenizing:  ['আপনারা', '"', 'বলেছেন', '', ':', 'আমার', 'ওর্ডার', 'ক্যান্সেল', 'হয়েছে', '।', 'আমি', 'অনুরোধ', 'করবো', ',', 'আমার', 'মতো', 'কাউকে', 'এভাবে', 'ঘুরাবেন', 'না।', '"']
    Truncating punctuation: ['আপনারা', 'বলেছেন', '', 'আমার', 'ওর্ডার', 'ক্যান্সেল', 'হয়েছে', 'আমি', 'অনুরোধ', 'করবো', 'আমার', 'মতো', 'কাউকে', 'এভাবে', 'ঘুরাবেন', 'না।']
    Truncating StopWords: ['আপনারা', '', 'ওর্ডার', 'ক্যান্সেল', 'অনুরোধ', 'করবো', 'এভাবে', 'ঘুরাবেন', 'না।']
    ***************************************************************************************
    Label:  0
    Sentence:  "সব বুয়া আগে টাকা নিয়ে পরে আর কোন খবর নাই জিনিস দেবার,"
    Afert Tokenizing:  ['সব', '"', 'বুয়া', 'আগে', 'টাকা', 'নিয়ে', 'পরে', 'আর', 'কোন', 'খবর', 'নাই', 'জিনিস', 'দেবার,', '"']
    Truncating punctuation: ['সব', 'বুয়া', 'আগে', 'টাকা', 'নিয়ে', 'পরে', 'আর', 'কোন', 'খবর', 'নাই', 'জিনিস', 'দেবার,']
    Truncating StopWords: ['বুয়া', 'টাকা', 'খবর', 'নাই', 'জিনিস', 'দেবার,']
    ***************************************************************************************
    Label:  0
    Sentence:  ই অর্ডার ডেলিভারির কোন খবর নাই। নিজে থেকে ফেইলড এ্যাটেম্পট দেখাইলো। সব খেয়াল খুশি মত করতেসে।
    Afert Tokenizing:  ['ই', 'অর্ডার', 'ডেলিভারির', 'কোন', 'খবর', 'নাই', '।', 'নিজে', 'থেকে', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', '।', 'সব', 'খেয়াল', 'খুশি', 'মত', 'করতেসে', '।']
    Truncating punctuation: ['ই', 'অর্ডার', 'ডেলিভারির', 'কোন', 'খবর', 'নাই', 'নিজে', 'থেকে', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', 'সব', 'খেয়াল', 'খুশি', 'মত', 'করতেসে']
    Truncating StopWords: ['অর্ডার', 'ডেলিভারির', 'খবর', 'নাই', 'ফেইলড', 'এ্যাটেম্পট', 'দেখাইলো', 'খেয়াল', 'খুশি', 'মত', 'করতেসে']
    ***************************************************************************************
    Label:  0
    Sentence:  এইটা কি সত্যিই দিচ্ছে নাকি বাটপারি
    Afert Tokenizing:  ['এইটা', 'কি', 'সত্যিই', 'দিচ্ছে', 'নাকি', 'বাটপারি']
    Truncating punctuation: ['এইটা', 'কি', 'সত্যিই', 'দিচ্ছে', 'নাকি', 'বাটপারি']
    Truncating StopWords: ['এইটা', 'সত্যিই', 'দিচ্ছে', 'বাটপারি']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ নাকি কম দামে মিষ্টির বক্স আছেকোন প্রোডাক্ট অর্ডার করে ক্যানসেল করার পর টাকা ফেরত পাচ্ছিনা কেন
    Afert Tokenizing:  ['দারাজ', 'নাকি', 'কম', 'দামে', 'মিষ্টির', 'বক্স', 'আছেকোন', 'প্রোডাক্ট', 'অর্ডার', 'করে', 'ক্যানসেল', 'করার', 'পর', 'টাকা', 'ফেরত', 'পাচ্ছিনা', 'কেন']
    Truncating punctuation: ['দারাজ', 'নাকি', 'কম', 'দামে', 'মিষ্টির', 'বক্স', 'আছেকোন', 'প্রোডাক্ট', 'অর্ডার', 'করে', 'ক্যানসেল', 'করার', 'পর', 'টাকা', 'ফেরত', 'পাচ্ছিনা', 'কেন']
    Truncating StopWords: ['দারাজ', 'কম', 'দামে', 'মিষ্টির', 'বক্স', 'আছেকোন', 'প্রোডাক্ট', 'অর্ডার', 'ক্যানসেল', 'টাকা', 'ফেরত', 'পাচ্ছিনা']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক বড় সু্যোগ দিলেন
    Afert Tokenizing:  ['অনেক', 'বড়', 'সু্যোগ', 'দিলেন']
    Truncating punctuation: ['অনেক', 'বড়', 'সু্যোগ', 'দিলেন']
    Truncating StopWords: ['বড়', 'সু্যোগ']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক উপকার হলো
    Afert Tokenizing:  ['অনেক', 'উপকার', 'হলো']
    Truncating punctuation: ['অনেক', 'উপকার', 'হলো']
    Truncating StopWords: ['উপকার']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমার অর্ডার ডেলিভারি এটেম্পট ফেল দেখাচ্ছে, অথচ আমাকে কোনো ফোন বা মেসেজ দেয়া হয় নি।"
    Afert Tokenizing:  ['আমার', '"', 'অর্ডার', 'ডেলিভারি', 'এটেম্পট', 'ফেল', 'দেখাচ্ছে', ',', 'অথচ', 'আমাকে', 'কোনো', 'ফোন', 'বা', 'মেসেজ', 'দেয়া', 'হয়', 'নি।', '"']
    Truncating punctuation: ['আমার', 'অর্ডার', 'ডেলিভারি', 'এটেম্পট', 'ফেল', 'দেখাচ্ছে', 'অথচ', 'আমাকে', 'কোনো', 'ফোন', 'বা', 'মেসেজ', 'দেয়া', 'হয়', 'নি।']
    Truncating StopWords: ['অর্ডার', 'ডেলিভারি', 'এটেম্পট', 'ফেল', 'দেখাচ্ছে', 'ফোন', 'মেসেজ', 'দেয়া', 'নি।']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই ১ তারিখে মাইক্রোফোন অর্ডার করছি। এখনও পাইনি।জামগড়া আরফান মার্কেট।
    Afert Tokenizing:  ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', '।', 'এখনও', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট', '।']
    Truncating punctuation: ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', 'এখনও', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট']
    Truncating StopWords: ['ভাই', '১', 'তারিখে', 'মাইক্রোফোন', 'অর্ডার', 'করছি', 'পাইনি।জামগড়া', 'আরফান', 'মার্কেট']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজের মাধ্যমে বিকাশ থেকে রিচার্জ করার পরেো তো কোনো ক্যাশব্যাক পাইনি।
    Afert Tokenizing:  ['দারাজের', 'মাধ্যমে', 'বিকাশ', 'থেকে', 'রিচার্জ', 'করার', 'পরেো', 'তো', 'কোনো', 'ক্যাশব্যাক', 'পাইনি', '।']
    Truncating punctuation: ['দারাজের', 'মাধ্যমে', 'বিকাশ', 'থেকে', 'রিচার্জ', 'করার', 'পরেো', 'তো', 'কোনো', 'ক্যাশব্যাক', 'পাইনি']
    Truncating StopWords: ['দারাজের', 'বিকাশ', 'রিচার্জ', 'পরেো', 'ক্যাশব্যাক', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  "কয়েকদিন ধরে দারাজে কোন প্রডাক্ট অর্ডার দিতে পারতেছি না, কিসের প্রবলেম হচ্ছে বুঝতেছিনা"
    Afert Tokenizing:  ['কয়েকদিন', '"', 'ধরে', 'দারাজে', 'কোন', 'প্রডাক্ট', 'অর্ডার', 'দিতে', 'পারতেছি', 'না', ',', 'কিসের', 'প্রবলেম', 'হচ্ছে', 'বুঝতেছিনা', '"']
    Truncating punctuation: ['কয়েকদিন', 'ধরে', 'দারাজে', 'কোন', 'প্রডাক্ট', 'অর্ডার', 'দিতে', 'পারতেছি', 'না', 'কিসের', 'প্রবলেম', 'হচ্ছে', 'বুঝতেছিনা']
    Truncating StopWords: ['কয়েকদিন', 'দারাজে', 'প্রডাক্ট', 'অর্ডার', 'পারতেছি', 'না', 'কিসের', 'প্রবলেম', 'বুঝতেছিনা']
    ***************************************************************************************
    Label:  0
    Sentence:  "জনগণকে মূলা দেখান? কোড দিলে আর রিচার্জ হয় না। এসব ফাইজলামি এখনই বন্ধ করুন,,,"
    Afert Tokenizing:  ['জনগণকে', '"', 'মূলা', 'দেখান', '?', 'কোড', 'দিলে', 'আর', 'রিচার্জ', 'হয়', 'না', '।', 'এসব', 'ফাইজলামি', 'এখনই', 'বন্ধ', 'করুন,,,', '"']
    Truncating punctuation: ['জনগণকে', 'মূলা', 'দেখান', 'কোড', 'দিলে', 'আর', 'রিচার্জ', 'হয়', 'না', 'এসব', 'ফাইজলামি', 'এখনই', 'বন্ধ', 'করুন,,,']
    Truncating StopWords: ['জনগণকে', 'মূলা', 'দেখান', 'কোড', 'দিলে', 'রিচার্জ', 'না', 'এসব', 'ফাইজলামি', 'এখনই', 'বন্ধ', 'করুন,,,']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ খারাপ ও পুরাতন পন্য ডেলিভারি করে।আর কোন দিন দারাজ এ কিনব না
    Afert Tokenizing:  ['দারাজ', 'খারাপ', 'ও', 'পুরাতন', 'পন্য', 'ডেলিভারি', 'করে।আর', 'কোন', 'দিন', 'দারাজ', 'এ', 'কিনব', 'না']
    Truncating punctuation: ['দারাজ', 'খারাপ', 'ও', 'পুরাতন', 'পন্য', 'ডেলিভারি', 'করে।আর', 'কোন', 'দিন', 'দারাজ', 'এ', 'কিনব', 'না']
    Truncating StopWords: ['দারাজ', 'খারাপ', 'পুরাতন', 'পন্য', 'ডেলিভারি', 'করে।আর', 'দারাজ', 'কিনব', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি অর্ডার ক্রিত কিছু প্রডাক্ট পাইনি করোনিও কি
    Afert Tokenizing:  ['আমি', 'অর্ডার', 'ক্রিত', 'কিছু', 'প্রডাক্ট', 'পাইনি', 'করোনিও', 'কি']
    Truncating punctuation: ['আমি', 'অর্ডার', 'ক্রিত', 'কিছু', 'প্রডাক্ট', 'পাইনি', 'করোনিও', 'কি']
    Truncating StopWords: ['অর্ডার', 'ক্রিত', 'প্রডাক্ট', 'পাইনি', 'করোনিও']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি তো অর্ডার করতে পারছি না
    Afert Tokenizing:  ['আমি', 'তো', 'অর্ডার', 'করতে', 'পারছি', 'না']
    Truncating punctuation: ['আমি', 'তো', 'অর্ডার', 'করতে', 'পারছি', 'না']
    Truncating StopWords: ['অর্ডার', 'পারছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ফাজলামি করার জন্য আর কিছু পেলে না
    Afert Tokenizing:  ['ফাজলামি', 'করার', 'জন্য', 'আর', 'কিছু', 'পেলে', 'না']
    Truncating punctuation: ['ফাজলামি', 'করার', 'জন্য', 'আর', 'কিছু', 'পেলে', 'না']
    Truncating StopWords: ['ফাজলামি', 'পেলে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ ভাল না বিশাল ঠকবাজ-আমার পুরাতন টেবিল দিছে-এখন রিটার্ন চেঞ্জ করে দেয় না।
    Afert Tokenizing:  ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'করে', 'দেয়', 'না', '।']
    Truncating punctuation: ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'করে', 'দেয়', 'না']
    Truncating StopWords: ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'দেয়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ২৮ তারিখে দেওয়ার কথা কিন্তু প্রডাক্ট পেলাম না।
    Afert Tokenizing:  ['২৮', 'তারিখে', 'দেওয়ার', 'কথা', 'কিন্তু', 'প্রডাক্ট', 'পেলাম', 'না', '।']
    Truncating punctuation: ['২৮', 'তারিখে', 'দেওয়ার', 'কথা', 'কিন্তু', 'প্রডাক্ট', 'পেলাম', 'না']
    Truncating StopWords: ['২৮', 'তারিখে', 'দেওয়ার', 'কথা', 'প্রডাক্ট', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "মিথ্যা বিজ্ঞাপন, ৩ টা পোডাক্ট এ ফ্রি ডেলিভারি কই।"
    Afert Tokenizing:  ['মিথ্যা', '"', 'বিজ্ঞাপন', ',', '৩', 'টা', 'পোডাক্ট', 'এ', 'ফ্রি', 'ডেলিভারি', 'কই।', '"']
    Truncating punctuation: ['মিথ্যা', 'বিজ্ঞাপন', '৩', 'টা', 'পোডাক্ট', 'এ', 'ফ্রি', 'ডেলিভারি', 'কই।']
    Truncating StopWords: ['মিথ্যা', 'বিজ্ঞাপন', '৩', 'টা', 'পোডাক্ট', 'ফ্রি', 'ডেলিভারি', 'কই।']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমি এই তিনটি পন্য অর্ডার করে বাতিল করলাম, শিপিং ফি এর জন্য,"
    Afert Tokenizing:  ['আমি', '"', 'এই', 'তিনটি', 'পন্য', 'অর্ডার', 'করে', 'বাতিল', 'করলাম', ',', 'শিপিং', 'ফি', 'এর', 'জন্য,', '"']
    Truncating punctuation: ['আমি', 'এই', 'তিনটি', 'পন্য', 'অর্ডার', 'করে', 'বাতিল', 'করলাম', 'শিপিং', 'ফি', 'এর', 'জন্য,']
    Truncating StopWords: ['তিনটি', 'পন্য', 'অর্ডার', 'বাতিল', 'করলাম', 'শিপিং', 'ফি', 'জন্য,']
    ***************************************************************************************
    Label:  0
    Sentence:  ২৭ তারিখের পার্সেল এখনো আসেনি।
    Afert Tokenizing:  ['২৭', 'তারিখের', 'পার্সেল', 'এখনো', 'আসেনি', '।']
    Truncating punctuation: ['২৭', 'তারিখের', 'পার্সেল', 'এখনো', 'আসেনি']
    Truncating StopWords: ['২৭', 'তারিখের', 'পার্সেল', 'এখনো', 'আসেনি']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুন সার্ভিস ভাই আপনাদের।
    Afert Tokenizing:  ['দারুন', 'সার্ভিস', 'ভাই', 'আপনাদের', '।']
    Truncating punctuation: ['দারুন', 'সার্ভিস', 'ভাই', 'আপনাদের']
    Truncating StopWords: ['দারুন', 'সার্ভিস', 'ভাই', 'আপনাদের']
    ***************************************************************************************
    Label:  0
    Sentence:  "দারাজ থেকে কেউ কিছু কিনবেন না,,,, একদম বাটপার কোম্পানি"
    Afert Tokenizing:  ['দারাজ', '"', 'থেকে', 'কেউ', 'কিছু', 'কিনবেন', 'না,,,', ',', 'একদম', 'বাটপার', 'কোম্পানি', '"']
    Truncating punctuation: ['দারাজ', 'থেকে', 'কেউ', 'কিছু', 'কিনবেন', 'না,,,', 'একদম', 'বাটপার', 'কোম্পানি']
    Truncating StopWords: ['দারাজ', 'কিনবেন', 'না,,,', 'একদম', 'বাটপার', 'কোম্পানি']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করা যাচ্ছে না!
    Afert Tokenizing:  ['অর্ডার', 'করা', 'যাচ্ছে', 'না', '!']
    Truncating punctuation: ['অর্ডার', 'করা', 'যাচ্ছে', 'না']
    Truncating StopWords: ['অর্ডার', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  কি সমস্যা বুঝতে পারলাম না লাগাতার একই সমস্যা হচ্ছে
    Afert Tokenizing:  ['কি', 'সমস্যা', 'বুঝতে', 'পারলাম', 'না', 'লাগাতার', 'একই', 'সমস্যা', 'হচ্ছে']
    Truncating punctuation: ['কি', 'সমস্যা', 'বুঝতে', 'পারলাম', 'না', 'লাগাতার', 'একই', 'সমস্যা', 'হচ্ছে']
    Truncating StopWords: ['সমস্যা', 'বুঝতে', 'পারলাম', 'না', 'লাগাতার', 'সমস্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  "দারাজে অর্ডার করেছি ২৬ তারিখে,,,! যদি আজকে না পাই তাহলে আপনাদের পন্য আর গ্রহণ/রিসিভ করবো না"
    Afert Tokenizing:  ['দারাজে', '"', 'অর্ডার', 'করেছি', '২৬', 'তারিখে,,,', '!', 'যদি', 'আজকে', 'না', 'পাই', 'তাহলে', 'আপনাদের', 'পন্য', 'আর', 'গ্রহণ/রিসিভ', 'করবো', 'না', '"']
    Truncating punctuation: ['দারাজে', 'অর্ডার', 'করেছি', '২৬', 'তারিখে,,,', 'যদি', 'আজকে', 'না', 'পাই', 'তাহলে', 'আপনাদের', 'পন্য', 'আর', 'গ্রহণ/রিসিভ', 'করবো', 'না']
    Truncating StopWords: ['দারাজে', 'অর্ডার', 'করেছি', '২৬', 'তারিখে,,,', 'আজকে', 'না', 'পাই', 'আপনাদের', 'পন্য', 'গ্রহণ/রিসিভ', 'করবো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  কিছু অর্ডার করলে এমনি এমনি বাতিল হয়ে যাই কেন প্লিজ বলবেন একটু।
    Afert Tokenizing:  ['কিছু', 'অর্ডার', 'করলে', 'এমনি', 'এমনি', 'বাতিল', 'হয়ে', 'যাই', 'কেন', 'প্লিজ', 'বলবেন', 'একটু', '।']
    Truncating punctuation: ['কিছু', 'অর্ডার', 'করলে', 'এমনি', 'এমনি', 'বাতিল', 'হয়ে', 'যাই', 'কেন', 'প্লিজ', 'বলবেন', 'একটু']
    Truncating StopWords: ['অর্ডার', 'বাতিল', 'হয়ে', 'যাই', 'প্লিজ', 'বলবেন', 'একটু']
    ***************************************************************************************
    Label:  0
    Sentence:  শেষ তারিখ পার যাবার পরও অর্ডার পেলাম না?
    Afert Tokenizing:  ['শেষ', 'তারিখ', 'পার', 'যাবার', 'পরও', 'অর্ডার', 'পেলাম', 'না', '?']
    Truncating punctuation: ['শেষ', 'তারিখ', 'পার', 'যাবার', 'পরও', 'অর্ডার', 'পেলাম', 'না']
    Truncating StopWords: ['শেষ', 'তারিখ', 'পার', 'যাবার', 'পরও', 'অর্ডার', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাই দারাজ বড় ধান্দাবাজ
    Afert Tokenizing:  ['ভাই', 'দারাজ', 'বড়', 'ধান্দাবাজ']
    Truncating punctuation: ['ভাই', 'দারাজ', 'বড়', 'ধান্দাবাজ']
    Truncating StopWords: ['ভাই', 'দারাজ', 'বড়', 'ধান্দাবাজ']
    ***************************************************************************************
    Label:  0
    Sentence:  বয়কট দারাজ
    Afert Tokenizing:  ['বয়কট', 'দারাজ']
    Truncating punctuation: ['বয়কট', 'দারাজ']
    Truncating StopWords: ['বয়কট', 'দারাজ']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ একটা ফালতু আজ থেকে বয়কাট করলাম
    Afert Tokenizing:  ['দারাজ', 'একটা', 'ফালতু', 'আজ', 'থেকে', 'বয়কাট', 'করলাম']
    Truncating punctuation: ['দারাজ', 'একটা', 'ফালতু', 'আজ', 'থেকে', 'বয়কাট', 'করলাম']
    Truncating StopWords: ['দারাজ', 'একটা', 'ফালতু', 'বয়কাট', 'করলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  কুপন কাজ ক‌রে না
    Afert Tokenizing:  ['কুপন', 'কাজ', 'ক\u200cরে', 'না']
    Truncating punctuation: ['কুপন', 'কাজ', 'ক\u200cরে', 'না']
    Truncating StopWords: ['কুপন', 'ক\u200cরে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  এভাবে কিনা গেলে তো আমাদের সকলের জন্য অনেক সুবিধা হাটে যাওয়ার জামেলা নাই
    Afert Tokenizing:  ['এভাবে', 'কিনা', 'গেলে', 'তো', 'আমাদের', 'সকলের', 'জন্য', 'অনেক', 'সুবিধা', 'হাটে', 'যাওয়ার', 'জামেলা', 'নাই']
    Truncating punctuation: ['এভাবে', 'কিনা', 'গেলে', 'তো', 'আমাদের', 'সকলের', 'জন্য', 'অনেক', 'সুবিধা', 'হাটে', 'যাওয়ার', 'জামেলা', 'নাই']
    Truncating StopWords: ['এভাবে', 'কিনা', 'সকলের', 'সুবিধা', 'হাটে', 'যাওয়ার', 'জামেলা', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আজ ১০৬৯ টাকার একটি অর্ডার ভিসা কার্ডে পেমেন্ট করলাম কিন্তু এখনো ক্যাশব্যাক পেলাম না
    Afert Tokenizing:  ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating punctuation: ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating StopWords: ['১০৬৯', 'টাকার', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব উপকার হলো
    Afert Tokenizing:  ['খুব', 'উপকার', 'হলো']
    Truncating punctuation: ['খুব', 'উপকার', 'হলো']
    Truncating StopWords: ['উপকার']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের অর্ডার ক্যানসেলের অপশনটা কি নাই এখন?
    Afert Tokenizing:  ['আপনাদের', 'অর্ডার', 'ক্যানসেলের', 'অপশনটা', 'কি', 'নাই', 'এখন', '?']
    Truncating punctuation: ['আপনাদের', 'অর্ডার', 'ক্যানসেলের', 'অপশনটা', 'কি', 'নাই', 'এখন']
    Truncating StopWords: ['আপনাদের', 'অর্ডার', 'ক্যানসেলের', 'অপশনটা', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "রাত ৯ টার ডেলিভারি রাত ১১-২০ দেন,,,সময়টা একটু মাথায় রাইখেন
    Afert Tokenizing:  ['রাত', '"', '৯', 'টার', 'ডেলিভারি', 'রাত', '১১-২০', 'দেন,,,সময়টা', 'একটু', 'মাথায়', 'রাইখেন']
    Truncating punctuation: ['রাত', '৯', 'টার', 'ডেলিভারি', 'রাত', '১১-২০', 'দেন,,,সময়টা', 'একটু', 'মাথায়', 'রাইখেন']
    Truncating StopWords: ['রাত', '৯', 'টার', 'ডেলিভারি', 'রাত', '১১-২০', 'দেন,,,সময়টা', 'একটু', 'মাথায়', 'রাইখেন']
    ***************************************************************************************
    Label:  0
    Sentence:  "দুপুর ১২ টায় ডেলিভারি আসার কথা, এখন রাত ৮টার বেশি বাজে , এখন ও আসে নাই।"
    Afert Tokenizing:  ['দুপুর', '"', '১২', 'টায়', 'ডেলিভারি', 'আসার', 'কথা', ',', 'এখন', 'রাত', '৮টার', 'বেশি', 'বাজে', '', ',', 'এখন', 'ও', 'আসে', 'নাই।', '"']
    Truncating punctuation: ['দুপুর', '১২', 'টায়', 'ডেলিভারি', 'আসার', 'কথা', 'এখন', 'রাত', '৮টার', 'বেশি', 'বাজে', '', 'এখন', 'ও', 'আসে', 'নাই।']
    Truncating StopWords: ['দুপুর', '১২', 'টায়', 'ডেলিভারি', 'আসার', 'কথা', 'রাত', '৮টার', 'বেশি', 'বাজে', '', 'আসে', 'নাই।']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনারা ব্যাগ এর ব্যবস্থা করেন.. হাতে করে আনেন.....দেখতে মটেও ভালো লাগে নাহ..
    Afert Tokenizing:  ['আপনারা', '"', 'ব্যাগ', 'এর', 'ব্যবস্থা', 'করেন.', '.', 'হাতে', 'করে', 'আনেন.....দেখতে', 'মটেও', 'ভালো', 'লাগে', 'নাহ.', '.']
    Truncating punctuation: ['আপনারা', 'ব্যাগ', 'এর', 'ব্যবস্থা', 'করেন.', 'হাতে', 'করে', 'আনেন.....দেখতে', 'মটেও', 'ভালো', 'লাগে', 'নাহ.']
    Truncating StopWords: ['আপনারা', 'ব্যাগ', 'ব্যবস্থা', 'করেন.', 'হাতে', 'আনেন.....দেখতে', 'মটেও', 'ভালো', 'লাগে', 'নাহ.']
    ***************************************************************************************
    Label:  1
    Sentence:  খুবই ভাল সার্ভিস। আমি পর পর ৩ দিন শপিং করেছি। একদম জাস্ট টাইমের আগে ডেলিভারি দিয়েছে। ফুললি স্যাটিসফাইড
    Afert Tokenizing:  ['খুবই', 'ভাল', 'সার্ভিস', '।', 'আমি', 'পর', 'পর', '৩', 'দিন', 'শপিং', 'করেছি', '।', 'একদম', 'জাস্ট', 'টাইমের', 'আগে', 'ডেলিভারি', 'দিয়েছে', '।', 'ফুললি', 'স্যাটিসফাইড']
    Truncating punctuation: ['খুবই', 'ভাল', 'সার্ভিস', 'আমি', 'পর', 'পর', '৩', 'দিন', 'শপিং', 'করেছি', 'একদম', 'জাস্ট', 'টাইমের', 'আগে', 'ডেলিভারি', 'দিয়েছে', 'ফুললি', 'স্যাটিসফাইড']
    Truncating StopWords: ['খুবই', 'ভাল', 'সার্ভিস', '৩', 'শপিং', 'করেছি', 'একদম', 'জাস্ট', 'টাইমের', 'ডেলিভারি', 'দিয়েছে', 'ফুললি', 'স্যাটিসফাইড']
    ***************************************************************************************
    Label:  1
    Sentence:  "বিসিবি র পিয়াজ সবসময় ই আঊট অফ স্টক, কিন্তু ওনাদের সেবার মান ভালো"
    Afert Tokenizing:  ['বিসিবি', '"', 'র', 'পিয়াজ', 'সবসময়', 'ই', 'আঊট', 'অফ', 'স্টক', ',', 'কিন্তু', 'ওনাদের', 'সেবার', 'মান', 'ভালো', '"']
    Truncating punctuation: ['বিসিবি', 'র', 'পিয়াজ', 'সবসময়', 'ই', 'আঊট', 'অফ', 'স্টক', 'কিন্তু', 'ওনাদের', 'সেবার', 'মান', 'ভালো']
    Truncating StopWords: ['বিসিবি', 'পিয়াজ', 'সবসময়', 'আঊট', 'অফ', 'স্টক', 'ওনাদের', 'সেবার', 'মান', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  "সবচেয়ে ভালো লাগে Chaldal এর কর্মীদের আন্তরিক ব্যবহার !!! ডেলিভারি চার্জ সবসময় এটা থাকলে ভালো হয় !!!"
    Afert Tokenizing:  ['সবচেয়ে', '"', 'ভালো', 'লাগে', 'Chaldal', 'এর', 'কর্মীদের', 'আন্তরিক', 'ব্যবহার', '!!', '!', 'ডেলিভারি', 'চার্জ', 'সবসময়', 'এটা', 'থাকলে', 'ভালো', 'হয়', '!!!', '"']
    Truncating punctuation: ['সবচেয়ে', 'ভালো', 'লাগে', 'Chaldal', 'এর', 'কর্মীদের', 'আন্তরিক', 'ব্যবহার', '!!', 'ডেলিভারি', 'চার্জ', 'সবসময়', 'এটা', 'থাকলে', 'ভালো', 'হয়', '!!!']
    Truncating StopWords: ['সবচেয়ে', 'ভালো', 'লাগে', 'Chaldal', 'কর্মীদের', 'আন্তরিক', '!!', 'ডেলিভারি', 'চার্জ', 'সবসময়', 'থাকলে', 'ভালো', '!!!']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের ডেলিভারি ম্যান গুলো অনেক ভালো। ঠিক সময়েই চলে আসে।
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'ম্যান', 'গুলো', 'অনেক', 'ভালো', '।', 'ঠিক', 'সময়েই', 'চলে', 'আসে', '।']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'ম্যান', 'গুলো', 'অনেক', 'ভালো', 'ঠিক', 'সময়েই', 'চলে', 'আসে']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'ম্যান', 'গুলো', 'ভালো', 'ঠিক', 'সময়েই', 'আসে']
    ***************************************************************************************
    Label:  0
    Sentence:  সব প্রায় 2 গুন দাম। এগুলো সবাই কিনতে পারবে না।
    Afert Tokenizing:  ['সব', 'প্রায়', '2', 'গুন', 'দাম', '।', 'এগুলো', 'সবাই', 'কিনতে', 'পারবে', 'না', '।']
    Truncating punctuation: ['সব', 'প্রায়', '2', 'গুন', 'দাম', 'এগুলো', 'সবাই', 'কিনতে', 'পারবে', 'না']
    Truncating StopWords: ['2', 'গুন', 'দাম', 'এগুলো', 'সবাই', 'কিনতে', 'পারবে', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ চালডালকে আশা করছি সবসময় এই ভাবে সেবা দিয়ে যাবেন এবং পন্যের মান ও ঠিকটাক রাখবেন।
    Afert Tokenizing:  ['ধন্যবাদ', 'চালডালকে', 'আশা', 'করছি', 'সবসময়', 'এই', 'ভাবে', 'সেবা', 'দিয়ে', 'যাবেন', 'এবং', 'পন্যের', 'মান', 'ও', 'ঠিকটাক', 'রাখবেন', '।']
    Truncating punctuation: ['ধন্যবাদ', 'চালডালকে', 'আশা', 'করছি', 'সবসময়', 'এই', 'ভাবে', 'সেবা', 'দিয়ে', 'যাবেন', 'এবং', 'পন্যের', 'মান', 'ও', 'ঠিকটাক', 'রাখবেন']
    Truncating StopWords: ['ধন্যবাদ', 'চালডালকে', 'আশা', 'করছি', 'সবসময়', 'সেবা', 'দিয়ে', 'যাবেন', 'পন্যের', 'মান', 'ঠিকটাক', 'রাখবেন']
    ***************************************************************************************
    Label:  0
    Sentence:  গিফট কোথায়। আমি পাইনি।
    Afert Tokenizing:  ['গিফট', 'কোথায়', '।', 'আমি', 'পাইনি', '।']
    Truncating punctuation: ['গিফট', 'কোথায়', 'আমি', 'পাইনি']
    Truncating StopWords: ['গিফট', 'কোথায়', 'পাইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ, অনেক ভালো "
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ', ',', 'অনেক', 'ভালো', '', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ', 'অনেক', 'ভালো', '']
    Truncating StopWords: ['"আলহামদুলিল্লাহ', 'ভালো', '']
    ***************************************************************************************
    Label:  1
    Sentence:  এবারের গিফট গুলো ভাল ছিল ধন্যবাদ চালডাল
    Afert Tokenizing:  ['এবারের', 'গিফট', 'গুলো', 'ভাল', 'ছিল', 'ধন্যবাদ', 'চালডাল']
    Truncating punctuation: ['এবারের', 'গিফট', 'গুলো', 'ভাল', 'ছিল', 'ধন্যবাদ', 'চালডাল']
    Truncating StopWords: ['এবারের', 'গিফট', 'গুলো', 'ভাল', 'ধন্যবাদ', 'চালডাল']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ খুব ভালো গিফট পেয়েছি। ধন্যবাদ চালডালকে
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'খুব', 'ভালো', 'গিফট', 'পেয়েছি', '।', 'ধন্যবাদ', 'চালডালকে']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'খুব', 'ভালো', 'গিফট', 'পেয়েছি', 'ধন্যবাদ', 'চালডালকে']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'ভালো', 'গিফট', 'পেয়েছি', 'ধন্যবাদ', 'চালডালকে']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ চাল ডাল প্রতি বছরের মত এবার ও গিফট পেয়েছি
    Afert Tokenizing:  ['ধন্যবাদ', 'চাল', 'ডাল', 'প্রতি', 'বছরের', 'মত', 'এবার', 'ও', 'গিফট', 'পেয়েছি']
    Truncating punctuation: ['ধন্যবাদ', 'চাল', 'ডাল', 'প্রতি', 'বছরের', 'মত', 'এবার', 'ও', 'গিফট', 'পেয়েছি']
    Truncating StopWords: ['ধন্যবাদ', 'চাল', 'ডাল', 'বছরের', 'মত', 'গিফট', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  "চমৎকার অনুভূতি, ধন্যবাদ চালডাল"
    Afert Tokenizing:  ['চমৎকার', '"', 'অনুভূতি', ',', 'ধন্যবাদ', 'চালডাল', '"']
    Truncating punctuation: ['চমৎকার', 'অনুভূতি', 'ধন্যবাদ', 'চালডাল']
    Truncating StopWords: ['চমৎকার', 'অনুভূতি', 'ধন্যবাদ', 'চালডাল']
    ***************************************************************************************
    Label:  1
    Sentence:  "সকাল টা ভালো লাগলো,চালডাল এর উপহারে। ধন্যবাদ।"
    Afert Tokenizing:  ['সকাল', '"', 'টা', 'ভালো', 'লাগলো,চালডাল', 'এর', 'উপহারে', '।', 'ধন্যবাদ।', '"']
    Truncating punctuation: ['সকাল', 'টা', 'ভালো', 'লাগলো,চালডাল', 'এর', 'উপহারে', 'ধন্যবাদ।']
    Truncating StopWords: ['সকাল', 'টা', 'ভালো', 'লাগলো,চালডাল', 'উপহারে', 'ধন্যবাদ।']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ চালডাল। নিউ গিফট প্যাক পাঠানোর জন্য।অনেক ভালো গিফট দিয়েছে।
    Afert Tokenizing:  ['ধন্যবাদ', 'চালডাল', '।', 'নিউ', 'গিফট', 'প্যাক', 'পাঠানোর', 'জন্য।অনেক', 'ভালো', 'গিফট', 'দিয়েছে', '।']
    Truncating punctuation: ['ধন্যবাদ', 'চালডাল', 'নিউ', 'গিফট', 'প্যাক', 'পাঠানোর', 'জন্য।অনেক', 'ভালো', 'গিফট', 'দিয়েছে']
    Truncating StopWords: ['ধন্যবাদ', 'চালডাল', 'নিউ', 'গিফট', 'প্যাক', 'পাঠানোর', 'জন্য।অনেক', 'ভালো', 'গিফট', 'দিয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক অনেক শুভেচ্ছা ও শুভকামনা রইল।
    Afert Tokenizing:  ['অনেক', 'অনেক', 'শুভেচ্ছা', 'ও', 'শুভকামনা', 'রইল', '।']
    Truncating punctuation: ['অনেক', 'অনেক', 'শুভেচ্ছা', 'ও', 'শুভকামনা', 'রইল']
    Truncating StopWords: ['শুভেচ্ছা', 'শুভকামনা', 'রইল']
    ***************************************************************************************
    Label:  0
    Sentence:  সোয়াবিন তৈল সরকারি রেট ৫ লিঃ ৯১০/- টাকা। এ্যাড দিয়ে ৯৫০/- টাকায় বিক্রি করছেন। ভোক্তাঅধিকারের দৃষ্টি আকর্ষণ করছি।
    Afert Tokenizing:  ['সোয়াবিন', 'তৈল', 'সরকারি', 'রেট', '৫', 'লিঃ', '৯১০/-', 'টাকা', '।', 'এ্যাড', 'দিয়ে', '৯৫০/-', 'টাকায়', 'বিক্রি', 'করছেন', '।', 'ভোক্তাঅধিকারের', 'দৃষ্টি', 'আকর্ষণ', 'করছি', '।']
    Truncating punctuation: ['সোয়াবিন', 'তৈল', 'সরকারি', 'রেট', '৫', 'লিঃ', '৯১০/-', 'টাকা', 'এ্যাড', 'দিয়ে', '৯৫০/-', 'টাকায়', 'বিক্রি', 'করছেন', 'ভোক্তাঅধিকারের', 'দৃষ্টি', 'আকর্ষণ', 'করছি']
    Truncating StopWords: ['সোয়াবিন', 'তৈল', 'সরকারি', 'রেট', '৫', 'লিঃ', '৯১০/-', 'টাকা', 'এ্যাড', 'দিয়ে', '৯৫০/-', 'টাকায়', 'বিক্রি', 'ভোক্তাঅধিকারের', 'দৃষ্টি', 'আকর্ষণ', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "দাম বাড়ানোর সময় সবার আগে বাড়ান, কিন্ত কমানোর সময় যত ধান্দাবাজি।
    Afert Tokenizing:  ['দাম', '"', 'বাড়ানোর', 'সময়', 'সবার', 'আগে', 'বাড়ান', ',', 'কিন্ত', 'কমানোর', 'সময়', 'যত', 'ধান্দাবাজি', '।']
    Truncating punctuation: ['দাম', 'বাড়ানোর', 'সময়', 'সবার', 'আগে', 'বাড়ান', 'কিন্ত', 'কমানোর', 'সময়', 'যত', 'ধান্দাবাজি']
    Truncating StopWords: ['দাম', 'বাড়ানোর', 'সময়', 'বাড়ান', 'কিন্ত', 'কমানোর', 'সময়', 'ধান্দাবাজি']
    ***************************************************************************************
    Label:  0
    Sentence:  তেলের দাম আরো কমেছেই। আপনারা দাম কমান।
    Afert Tokenizing:  ['তেলের', 'দাম', 'আরো', 'কমেছেই', '।', 'আপনারা', 'দাম', 'কমান', '।']
    Truncating punctuation: ['তেলের', 'দাম', 'আরো', 'কমেছেই', 'আপনারা', 'দাম', 'কমান']
    Truncating StopWords: ['তেলের', 'দাম', 'আরো', 'কমেছেই', 'আপনারা', 'দাম', 'কমান']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্থহীন মূল্য ছাড়! আমার নিকটবর্তী কোন আউটলেটে ৫ লিটার এর তেল নেই! তাও আফার শুরুর প্রথম প্রহর থেকেই
    Afert Tokenizing:  ['অর্থহীন', 'মূল্য', 'ছাড়', '!', 'আমার', 'নিকটবর্তী', 'কোন', 'আউটলেটে', '৫', 'লিটার', 'এর', 'তেল', 'নেই', '!', 'তাও', 'আফার', 'শুরুর', 'প্রথম', 'প্রহর', 'থেকেই']
    Truncating punctuation: ['অর্থহীন', 'মূল্য', 'ছাড়', 'আমার', 'নিকটবর্তী', 'কোন', 'আউটলেটে', '৫', 'লিটার', 'এর', 'তেল', 'নেই', 'তাও', 'আফার', 'শুরুর', 'প্রথম', 'প্রহর', 'থেকেই']
    Truncating StopWords: ['অর্থহীন', 'মূল্য', 'ছাড়', 'নিকটবর্তী', 'আউটলেটে', '৫', 'লিটার', 'তেল', 'নেই', 'আফার', 'শুরুর', 'প্রহর']
    ***************************************************************************************
    Label:  0
    Sentence:  অফার তো নেই। যেগুলোতে ডিসকাউন্ট ছিলো সেগুলো এখন অরিজিনাল প্রাইস দেখাচ্ছে।
    Afert Tokenizing:  ['অফার', 'তো', 'নেই', '।', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'এখন', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে', '।']
    Truncating punctuation: ['অফার', 'তো', 'নেই', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'এখন', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে']
    Truncating StopWords: ['অফার', 'নেই', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনাদের সার্ভিস দিন দিন খারাপ হয়ে যাচ্ছে, প্রত্যেকটা ডেলিভারি রাতে করতে আসে, কমিউনিকেশন করে না, দারাজে এর কেনাকাটা করা যাবে না।"
    Afert Tokenizing:  ['আপনাদের', '"', 'সার্ভিস', 'দিন', 'দিন', 'খারাপ', 'হয়ে', 'যাচ্ছে', ',', 'প্রত্যেকটা', 'ডেলিভারি', 'রাতে', 'করতে', 'আসে', ',', 'কমিউনিকেশন', 'করে', 'না', ',', 'দারাজে', 'এর', 'কেনাকাটা', 'করা', 'যাবে', 'না।', '"']
    Truncating punctuation: ['আপনাদের', 'সার্ভিস', 'দিন', 'দিন', 'খারাপ', 'হয়ে', 'যাচ্ছে', 'প্রত্যেকটা', 'ডেলিভারি', 'রাতে', 'করতে', 'আসে', 'কমিউনিকেশন', 'করে', 'না', 'দারাজে', 'এর', 'কেনাকাটা', 'করা', 'যাবে', 'না।']
    Truncating StopWords: ['আপনাদের', 'সার্ভিস', 'খারাপ', 'হয়ে', 'প্রত্যেকটা', 'ডেলিভারি', 'রাতে', 'আসে', 'কমিউনিকেশন', 'না', 'দারাজে', 'কেনাকাটা', 'না।']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি একটা ব্লুটুথ হেডফোন অর্ডার করেছিলাম কিন্তু পাইলাম না
    Afert Tokenizing:  ['আমি', 'একটা', 'ব্লুটুথ', 'হেডফোন', 'অর্ডার', 'করেছিলাম', 'কিন্তু', 'পাইলাম', 'না']
    Truncating punctuation: ['আমি', 'একটা', 'ব্লুটুথ', 'হেডফোন', 'অর্ডার', 'করেছিলাম', 'কিন্তু', 'পাইলাম', 'না']
    Truncating StopWords: ['একটা', 'ব্লুটুথ', 'হেডফোন', 'অর্ডার', 'করেছিলাম', 'পাইলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  রিটার্ন শুধু বিজ্ঞাপনে বাস্তবে হয়না।
    Afert Tokenizing:  ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না', '।']
    Truncating punctuation: ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না']
    Truncating StopWords: ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি চার্জ কমান রেগুলার ক্রেতা ছিলাম
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  ১০ দিনে ফ্রিজ পেলাম না। জানানো হয় না দিবে নাকি। আর যদি কিছু অর্ডার করছি
    Afert Tokenizing:  ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', '।', 'জানানো', 'হয়', 'না', 'দিবে', 'নাকি', '।', 'আর', 'যদি', 'কিছু', 'অর্ডার', 'করছি']
    Truncating punctuation: ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', 'জানানো', 'হয়', 'না', 'দিবে', 'নাকি', 'আর', 'যদি', 'কিছু', 'অর্ডার', 'করছি']
    Truncating StopWords: ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', 'না', 'দিবে', 'অর্ডার', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাউচার এখনো মেসেজ এ দেয়া হয় নাই কেন
    Afert Tokenizing:  ['ভাউচার', 'এখনো', 'মেসেজ', 'এ', 'দেয়া', 'হয়', 'নাই', 'কেন']
    Truncating punctuation: ['ভাউচার', 'এখনো', 'মেসেজ', 'এ', 'দেয়া', 'হয়', 'নাই', 'কেন']
    Truncating StopWords: ['ভাউচার', 'এখনো', 'মেসেজ', 'দেয়া', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  আজ ১০৬৯ টাকার একটি অর্ডার ভিসা কার্ডে পেমেন্ট করলাম কিন্তু এখনো ক্যাশব্যাক পেলাম না
    Afert Tokenizing:  ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating punctuation: ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating StopWords: ['১০৬৯', 'টাকার', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার টাকা নিয়ে পণ্য দেয় নাই
    Afert Tokenizing:  ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating punctuation: ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating StopWords: ['টাকা', 'পণ্য', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও না চাইতেই
    Afert Tokenizing:  ['ওয়াও', 'না', 'চাইতেই']
    Truncating punctuation: ['ওয়াও', 'না', 'চাইতেই']
    Truncating StopWords: ['ওয়াও', 'না', 'চাইতেই']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমি তো পাই নাই, আমি রিচার্জ করছি কিন্তু পাই নাই"
    Afert Tokenizing:  ['আমি', '"', 'তো', 'পাই', 'নাই', ',', 'আমি', 'রিচার্জ', 'করছি', 'কিন্তু', 'পাই', 'নাই', '"']
    Truncating punctuation: ['আমি', 'তো', 'পাই', 'নাই', 'আমি', 'রিচার্জ', 'করছি', 'কিন্তু', 'পাই', 'নাই']
    Truncating StopWords: ['পাই', 'নাই', 'রিচার্জ', 'করছি', 'পাই', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "স্যার মোবাইল রিচার্জ হচ্ছে না এই লেখাটা আসতেছে। প্লিজ একটু জানাই দিবেন কি সমস্যা।"
    Afert Tokenizing:  ['স্যার', '"', 'মোবাইল', 'রিচার্জ', 'হচ্ছে', 'না', 'এই', 'লেখাটা', 'আসতেছে', '।', 'প্লিজ', 'একটু', 'জানাই', 'দিবেন', 'কি', 'সমস্যা।', '"']
    Truncating punctuation: ['স্যার', 'মোবাইল', 'রিচার্জ', 'হচ্ছে', 'না', 'এই', 'লেখাটা', 'আসতেছে', 'প্লিজ', 'একটু', 'জানাই', 'দিবেন', 'কি', 'সমস্যা।']
    Truncating StopWords: ['স্যার', 'মোবাইল', 'রিচার্জ', 'না', 'লেখাটা', 'আসতেছে', 'প্লিজ', 'একটু', 'জানাই', 'দিবেন', 'সমস্যা।']
    ***************************************************************************************
    Label:  0
    Sentence:  "দারাজে কি অর্ডার দেওয়া বন্ধ করবো, কোন ডেলিভারি ঠিক ভাবে পাচ্ছি না। ইভেলির মতো না হয়ে যায় দারাজ"
    Afert Tokenizing:  ['দারাজে', '"', 'কি', 'অর্ডার', 'দেওয়া', 'বন্ধ', 'করবো', ',', 'কোন', 'ডেলিভারি', 'ঠিক', 'ভাবে', 'পাচ্ছি', 'না', '।', 'ইভেলির', 'মতো', 'না', 'হয়ে', 'যায়', 'দারাজ', '"']
    Truncating punctuation: ['দারাজে', 'কি', 'অর্ডার', 'দেওয়া', 'বন্ধ', 'করবো', 'কোন', 'ডেলিভারি', 'ঠিক', 'ভাবে', 'পাচ্ছি', 'না', 'ইভেলির', 'মতো', 'না', 'হয়ে', 'যায়', 'দারাজ']
    Truncating StopWords: ['দারাজে', 'অর্ডার', 'বন্ধ', 'করবো', 'ডেলিভারি', 'ঠিক', 'পাচ্ছি', 'না', 'ইভেলির', 'না', 'হয়ে', 'যায়', 'দারাজ']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে একটা অনলাইন সাইট
    Afert Tokenizing:  ['বাজে', 'একটা', 'অনলাইন', 'সাইট']
    Truncating punctuation: ['বাজে', 'একটা', 'অনলাইন', 'সাইট']
    Truncating StopWords: ['বাজে', 'একটা', 'অনলাইন', 'সাইট']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ কমান।আলাদা চার্জ নেয়ায় বেশি হয়ে যাচ্ছে।
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'কমান।আলাদা', 'চার্জ', 'নেয়ায়', 'বেশি', 'হয়ে', 'যাচ্ছে', '।']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'কমান।আলাদা', 'চার্জ', 'নেয়ায়', 'বেশি', 'হয়ে', 'যাচ্ছে']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'কমান।আলাদা', 'চার্জ', 'নেয়ায়', 'বেশি', 'হয়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  "মিয়া ভাই ডেলিভারি চার্জ কমান,,,,এইরকম কইরা ডাকাতি করা ভালো না"
    Afert Tokenizing:  ['মিয়া', '"', 'ভাই', 'ডেলিভারি', 'চার্জ', 'কমান,,,,এইরকম', 'কইরা', 'ডাকাতি', 'করা', 'ভালো', 'না', '"']
    Truncating punctuation: ['মিয়া', 'ভাই', 'ডেলিভারি', 'চার্জ', 'কমান,,,,এইরকম', 'কইরা', 'ডাকাতি', 'করা', 'ভালো', 'না']
    Truncating StopWords: ['মিয়া', 'ভাই', 'ডেলিভারি', 'চার্জ', 'কমান,,,,এইরকম', 'কইরা', 'ডাকাতি', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  মেসেজ করা যায়না কেন
    Afert Tokenizing:  ['মেসেজ', 'করা', 'যায়না', 'কেন']
    Truncating punctuation: ['মেসেজ', 'করা', 'যায়না', 'কেন']
    Truncating StopWords: ['মেসেজ', 'যায়না']
    ***************************************************************************************
    Label:  0
    Sentence:  ১০ দিন হয়ে গেল এখনো প্রোডাক্ট পাই নাই
    Afert Tokenizing:  ['১০', 'দিন', 'হয়ে', 'গেল', 'এখনো', 'প্রোডাক্ট', 'পাই', 'নাই']
    Truncating punctuation: ['১০', 'দিন', 'হয়ে', 'গেল', 'এখনো', 'প্রোডাক্ট', 'পাই', 'নাই']
    Truncating StopWords: ['১০', 'এখনো', 'প্রোডাক্ট', 'পাই', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  গ্লোবাল প্রডাক্ট অর্ডার করেছিলাম ১ মাসের বেশি সময় হয়ে গেছে ডেলিভারির টাইমলাইন অনেক আগেই শেষ হয়েছে এখনো ডেলিভারি দেয়ার নাম নাই
    Afert Tokenizing:  ['গ্লোবাল', 'প্রডাক্ট', 'অর্ডার', 'করেছিলাম', '১', 'মাসের', 'বেশি', 'সময়', 'হয়ে', 'গেছে', 'ডেলিভারির', 'টাইমলাইন', 'অনেক', 'আগেই', 'শেষ', 'হয়েছে', 'এখনো', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    Truncating punctuation: ['গ্লোবাল', 'প্রডাক্ট', 'অর্ডার', 'করেছিলাম', '১', 'মাসের', 'বেশি', 'সময়', 'হয়ে', 'গেছে', 'ডেলিভারির', 'টাইমলাইন', 'অনেক', 'আগেই', 'শেষ', 'হয়েছে', 'এখনো', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    Truncating StopWords: ['গ্লোবাল', 'প্রডাক্ট', 'অর্ডার', 'করেছিলাম', '১', 'মাসের', 'বেশি', 'সময়', 'হয়ে', 'ডেলিভারির', 'টাইমলাইন', 'শেষ', 'হয়েছে', 'এখনো', 'ডেলিভারি', 'দেয়ার', 'নাম', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ ভাল না বিশাল ঠকবাজ-আমার পুরাতন টেবিল দিছে-এখন রিটার্ন চেঞ্জ করে দেয় না।
    Afert Tokenizing:  ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'করে', 'দেয়', 'না', '।']
    Truncating punctuation: ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'করে', 'দেয়', 'না']
    Truncating StopWords: ['দারাজ', 'ভাল', 'না', 'বিশাল', 'ঠকবাজ-আমার', 'পুরাতন', 'টেবিল', 'দিছে-এখন', 'রিটার্ন', 'চেঞ্জ', 'দেয়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "মিথ্যা বিজ্ঞাপন, ৩ টা পোডাক্ট এ ফ্রি ডেলিভারি কই।"
    Afert Tokenizing:  ['মিথ্যা', '"', 'বিজ্ঞাপন', ',', '৩', 'টা', 'পোডাক্ট', 'এ', 'ফ্রি', 'ডেলিভারি', 'কই।', '"']
    Truncating punctuation: ['মিথ্যা', 'বিজ্ঞাপন', '৩', 'টা', 'পোডাক্ট', 'এ', 'ফ্রি', 'ডেলিভারি', 'কই।']
    Truncating StopWords: ['মিথ্যা', 'বিজ্ঞাপন', '৩', 'টা', 'পোডাক্ট', 'ফ্রি', 'ডেলিভারি', 'কই।']
    ***************************************************************************************
    Label:  0
    Sentence:  "হ্যাঁ, অর্ডার করলাম এক কালার পেলাম আরেক কালার, আগে সেবার মান ভালো করে বিজ্ঞাপন দেন।"
    Afert Tokenizing:  ['"হ্যাঁ', ',', 'অর্ডার', 'করলাম', 'এক', 'কালার', 'পেলাম', 'আরেক', 'কালার', ',', 'আগে', 'সেবার', 'মান', 'ভালো', 'করে', 'বিজ্ঞাপন', 'দেন।', '"']
    Truncating punctuation: ['"হ্যাঁ', 'অর্ডার', 'করলাম', 'এক', 'কালার', 'পেলাম', 'আরেক', 'কালার', 'আগে', 'সেবার', 'মান', 'ভালো', 'করে', 'বিজ্ঞাপন', 'দেন।']
    Truncating StopWords: ['"হ্যাঁ', 'অর্ডার', 'করলাম', 'এক', 'কালার', 'পেলাম', 'আরেক', 'কালার', 'সেবার', 'মান', 'ভালো', 'বিজ্ঞাপন', 'দেন।']
    ***************************************************************************************
    Label:  0
    Sentence:  হুদাই চেষ্টা করলাম খালি
    Afert Tokenizing:  ['হুদাই', 'চেষ্টা', 'করলাম', 'খালি']
    Truncating punctuation: ['হুদাই', 'চেষ্টা', 'করলাম', 'খালি']
    Truncating StopWords: ['হুদাই', 'চেষ্টা', 'করলাম', 'খালি']
    ***************************************************************************************
    Label:  0
    Sentence:  "স্যার আমি গত ২১,০৪,২০২২ তারিখে মাল রিটার্ন করছি, এখনো টাকা ফেরত পাইলাম না। প্লিজ স্যার একটু দেখেন।"
    Afert Tokenizing:  ['স্যার', '"', 'আমি', 'গত', '২১,০৪,২০২২', 'তারিখে', 'মাল', 'রিটার্ন', 'করছি', ',', 'এখনো', 'টাকা', 'ফেরত', 'পাইলাম', 'না', '।', 'প্লিজ', 'স্যার', 'একটু', 'দেখেন।', '"']
    Truncating punctuation: ['স্যার', 'আমি', 'গত', '২১,০৪,২০২২', 'তারিখে', 'মাল', 'রিটার্ন', 'করছি', 'এখনো', 'টাকা', 'ফেরত', 'পাইলাম', 'না', 'প্লিজ', 'স্যার', 'একটু', 'দেখেন।']
    Truncating StopWords: ['স্যার', 'গত', '২১,০৪,২০২২', 'তারিখে', 'মাল', 'রিটার্ন', 'করছি', 'এখনো', 'টাকা', 'ফেরত', 'পাইলাম', 'না', 'প্লিজ', 'স্যার', 'একটু', 'দেখেন।']
    ***************************************************************************************
    Label:  0
    Sentence:  স্যার আমার সাথে এটা কেমন ব্যবহার করলেন?
    Afert Tokenizing:  ['স্যার', 'আমার', 'সাথে', 'এটা', 'কেমন', 'ব্যবহার', 'করলেন', '?']
    Truncating punctuation: ['স্যার', 'আমার', 'সাথে', 'এটা', 'কেমন', 'ব্যবহার', 'করলেন']
    Truncating StopWords: ['স্যার', 'সাথে', 'কেমন']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের রিটার্ন পলিসি অনেক খারাপ
    Afert Tokenizing:  ['আপনাদের', 'রিটার্ন', 'পলিসি', 'অনেক', 'খারাপ']
    Truncating punctuation: ['আপনাদের', 'রিটার্ন', 'পলিসি', 'অনেক', 'খারাপ']
    Truncating StopWords: ['আপনাদের', 'রিটার্ন', 'পলিসি', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  এই মেশিন টা ডারাজ থেকে অনলাইনে নিছিলাম কিছুদিন আগে বাট মেশিন টা ভালো না।
    Afert Tokenizing:  ['এই', 'মেশিন', 'টা', 'ডারাজ', 'থেকে', 'অনলাইনে', 'নিছিলাম', 'কিছুদিন', 'আগে', 'বাট', 'মেশিন', 'টা', 'ভালো', 'না', '।']
    Truncating punctuation: ['এই', 'মেশিন', 'টা', 'ডারাজ', 'থেকে', 'অনলাইনে', 'নিছিলাম', 'কিছুদিন', 'আগে', 'বাট', 'মেশিন', 'টা', 'ভালো', 'না']
    Truncating StopWords: ['মেশিন', 'টা', 'ডারাজ', 'অনলাইনে', 'নিছিলাম', 'কিছুদিন', 'বাট', 'মেশিন', 'টা', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  অনলাইনে অর্ডার করে লাভ কি যদি সঠিক সময়ে ডেলিভারি না পাই । ঢাকার মধ্যে ডেলিভারি দিতে সপ্তাহখানেক সময় লাগে তাহলে এরকম সার্ভিসের দরকার নেই ।
    Afert Tokenizing:  ['অনলাইনে', 'অর্ডার', 'করে', 'লাভ', 'কি', 'যদি', 'সঠিক', 'সময়ে', 'ডেলিভারি', 'না', 'পাই', '', '।', 'ঢাকার', 'মধ্যে', 'ডেলিভারি', 'দিতে', 'সপ্তাহখানেক', 'সময়', 'লাগে', 'তাহলে', 'এরকম', 'সার্ভিসের', 'দরকার', 'নেই', '', '।']
    Truncating punctuation: ['অনলাইনে', 'অর্ডার', 'করে', 'লাভ', 'কি', 'যদি', 'সঠিক', 'সময়ে', 'ডেলিভারি', 'না', 'পাই', '', 'ঢাকার', 'মধ্যে', 'ডেলিভারি', 'দিতে', 'সপ্তাহখানেক', 'সময়', 'লাগে', 'তাহলে', 'এরকম', 'সার্ভিসের', 'দরকার', 'নেই', '']
    Truncating StopWords: ['অনলাইনে', 'অর্ডার', 'লাভ', 'সঠিক', 'সময়ে', 'ডেলিভারি', 'না', 'পাই', '', 'ঢাকার', 'ডেলিভারি', 'সপ্তাহখানেক', 'সময়', 'লাগে', 'এরকম', 'সার্ভিসের', 'দরকার', 'নেই', '']
    ***************************************************************************************
    Label:  1
    Sentence:  "আমি অনেক পন্য দারাজ থেকে নিয়েছি সব পন্য সঠিক সময়ে পাইছি।
    Afert Tokenizing:  ['আমি', '"', 'অনেক', 'পন্য', 'দারাজ', 'থেকে', 'নিয়েছি', 'সব', 'পন্য', 'সঠিক', 'সময়ে', 'পাইছি', '।']
    Truncating punctuation: ['আমি', 'অনেক', 'পন্য', 'দারাজ', 'থেকে', 'নিয়েছি', 'সব', 'পন্য', 'সঠিক', 'সময়ে', 'পাইছি']
    Truncating StopWords: ['পন্য', 'দারাজ', 'নিয়েছি', 'পন্য', 'সঠিক', 'সময়ে', 'পাইছি']
    ***************************************************************************************
    Label:  1
    Sentence:  লাইটা অনেক সুন্দর …
    Afert Tokenizing:  ['লাইটা', 'অনেক', 'সুন্দর', '…']
    Truncating punctuation: ['লাইটা', 'অনেক', 'সুন্দর', '…']
    Truncating StopWords: ['লাইটা', 'সুন্দর', '…']
    ***************************************************************************************
    Label:  0
    Sentence:  আমিও আপনাদের কাছ থেকে অনেক টি শার্ট কিনেছি কিন্তু মন মত হয়নাই
    Afert Tokenizing:  ['আমিও', 'আপনাদের', 'কাছ', 'থেকে', 'অনেক', 'টি', 'শার্ট', 'কিনেছি', 'কিন্তু', 'মন', 'মত', 'হয়নাই']
    Truncating punctuation: ['আমিও', 'আপনাদের', 'কাছ', 'থেকে', 'অনেক', 'টি', 'শার্ট', 'কিনেছি', 'কিন্তু', 'মন', 'মত', 'হয়নাই']
    Truncating StopWords: ['আমিও', 'আপনাদের', 'শার্ট', 'কিনেছি', 'মন', 'মত', 'হয়নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  না কিনতাম না
    Afert Tokenizing:  ['না', 'কিনতাম', 'না']
    Truncating punctuation: ['না', 'কিনতাম', 'না']
    Truncating StopWords: ['না', 'কিনতাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "দিলো এইটা এসব কি ভাই,,ভালো না"
    Afert Tokenizing:  ['দিলো', '"', 'এইটা', 'এসব', 'কি', 'ভাই,,ভালো', 'না', '"']
    Truncating punctuation: ['দিলো', 'এইটা', 'এসব', 'কি', 'ভাই,,ভালো', 'না']
    Truncating StopWords: ['দিলো', 'এইটা', 'এসব', 'ভাই,,ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "রিটার্ন তো আর মাগনা নিবেনা,, দামটা বারতি নেন তাতে কোনো প্রবলেম হবেনা ধোঁকাবাজি গুলো না করলে গ্রাহক খুশি হবেন"
    Afert Tokenizing:  ['রিটার্ন', '"', 'তো', 'আর', 'মাগনা', 'নিবেনা,', ',', 'দামটা', 'বারতি', 'নেন', 'তাতে', 'কোনো', 'প্রবলেম', 'হবেনা', 'ধোঁকাবাজি', 'গুলো', 'না', 'করলে', 'গ্রাহক', 'খুশি', 'হবেন', '"']
    Truncating punctuation: ['রিটার্ন', 'তো', 'আর', 'মাগনা', 'নিবেনা,', 'দামটা', 'বারতি', 'নেন', 'তাতে', 'কোনো', 'প্রবলেম', 'হবেনা', 'ধোঁকাবাজি', 'গুলো', 'না', 'করলে', 'গ্রাহক', 'খুশি', 'হবেন']
    Truncating StopWords: ['রিটার্ন', 'মাগনা', 'নিবেনা,', 'দামটা', 'বারতি', 'নেন', 'প্রবলেম', 'হবেনা', 'ধোঁকাবাজি', 'গুলো', 'না', 'গ্রাহক', 'খুশি']
    ***************************************************************************************
    Label:  0
    Sentence:  "ডারাজ একটা ভুয়া, এরা সবার সাথে চিটারি করে এদের এই সফকে আইনের আওতাই আনা হোক, কারন আমি যা অর্ডার করেছি তা না দিয়ে অন্য মাল দিয়েছি এখন আমার এই টাকাটা কি পানির জলে চলে যাবে?"
    Afert Tokenizing:  ['ডারাজ', '"', 'একটা', 'ভুয়া', ',', 'এরা', 'সবার', 'সাথে', 'চিটারি', 'করে', 'এদের', 'এই', 'সফকে', 'আইনের', 'আওতাই', 'আনা', 'হোক', ',', 'কারন', 'আমি', 'যা', 'অর্ডার', 'করেছি', 'তা', 'না', 'দিয়ে', 'অন্য', 'মাল', 'দিয়েছি', 'এখন', 'আমার', 'এই', 'টাকাটা', 'কি', 'পানির', 'জলে', 'চলে', 'যাবে?', '"']
    Truncating punctuation: ['ডারাজ', 'একটা', 'ভুয়া', 'এরা', 'সবার', 'সাথে', 'চিটারি', 'করে', 'এদের', 'এই', 'সফকে', 'আইনের', 'আওতাই', 'আনা', 'হোক', 'কারন', 'আমি', 'যা', 'অর্ডার', 'করেছি', 'তা', 'না', 'দিয়ে', 'অন্য', 'মাল', 'দিয়েছি', 'এখন', 'আমার', 'এই', 'টাকাটা', 'কি', 'পানির', 'জলে', 'চলে', 'যাবে?']
    Truncating StopWords: ['ডারাজ', 'একটা', 'ভুয়া', 'সাথে', 'চিটারি', 'সফকে', 'আইনের', 'আওতাই', 'আনা', 'কারন', 'অর্ডার', 'করেছি', 'না', 'দিয়ে', 'মাল', 'দিয়েছি', 'টাকাটা', 'পানির', 'জলে', 'যাবে?']
    ***************************************************************************************
    Label:  0
    Sentence:  "ওডার করলে ডেলিভারির কোন খবর থাকেনা
    Afert Tokenizing:  ['ওডার', '"', 'করলে', 'ডেলিভারির', 'কোন', 'খবর', 'থাকেনা']
    Truncating punctuation: ['ওডার', 'করলে', 'ডেলিভারির', 'কোন', 'খবর', 'থাকেনা']
    Truncating StopWords: ['ওডার', 'ডেলিভারির', 'খবর', 'থাকেনা']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি গত পরশুদিন দারাজে কিছু পণ্যের অর্ডার করি। তার ভিতরে ৩ টা স্যাভলন সাবানের অর্ডার করি। কিন্তু পণ্য রিসিভ করার পরে প্যাকেট খুলে ২ টা স্যাভলন সাবান দেখতে পাই। যা অত্যান্ত অনাকাঙ্ক্ষিত ও দুঃক্ষজনক।
    Afert Tokenizing:  ['আমি', 'গত', 'পরশুদিন', 'দারাজে', 'কিছু', 'পণ্যের', 'অর্ডার', 'করি', '।', 'তার', 'ভিতরে', '৩', 'টা', 'স্যাভলন', 'সাবানের', 'অর্ডার', 'করি', '।', 'কিন্তু', 'পণ্য', 'রিসিভ', 'করার', 'পরে', 'প্যাকেট', 'খুলে', '২', 'টা', 'স্যাভলন', 'সাবান', 'দেখতে', 'পাই', '।', 'যা', 'অত্যান্ত', 'অনাকাঙ্ক্ষিত', 'ও', 'দুঃক্ষজনক', '।']
    Truncating punctuation: ['আমি', 'গত', 'পরশুদিন', 'দারাজে', 'কিছু', 'পণ্যের', 'অর্ডার', 'করি', 'তার', 'ভিতরে', '৩', 'টা', 'স্যাভলন', 'সাবানের', 'অর্ডার', 'করি', 'কিন্তু', 'পণ্য', 'রিসিভ', 'করার', 'পরে', 'প্যাকেট', 'খুলে', '২', 'টা', 'স্যাভলন', 'সাবান', 'দেখতে', 'পাই', 'যা', 'অত্যান্ত', 'অনাকাঙ্ক্ষিত', 'ও', 'দুঃক্ষজনক']
    Truncating StopWords: ['গত', 'পরশুদিন', 'দারাজে', 'পণ্যের', 'অর্ডার', 'ভিতরে', '৩', 'টা', 'স্যাভলন', 'সাবানের', 'অর্ডার', 'পণ্য', 'রিসিভ', 'প্যাকেট', 'খুলে', '২', 'টা', 'স্যাভলন', 'সাবান', 'পাই', 'অত্যান্ত', 'অনাকাঙ্ক্ষিত', 'দুঃক্ষজনক']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রয়োজনীয় জিনিসের বেশির ভাগই স্টকআউট হয়ে থাকে
    Afert Tokenizing:  ['প্রয়োজনীয়', 'জিনিসের', 'বেশির', 'ভাগই', 'স্টকআউট', 'হয়ে', 'থাকে']
    Truncating punctuation: ['প্রয়োজনীয়', 'জিনিসের', 'বেশির', 'ভাগই', 'স্টকআউট', 'হয়ে', 'থাকে']
    Truncating StopWords: ['প্রয়োজনীয়', 'জিনিসের', 'বেশির', 'ভাগই', 'স্টকআউট', 'হয়ে']
    ***************************************************************************************
    Label:  0
    Sentence:  ৩১ শে জুলাই এর অর্ডার এখনও পাই নাই
    Afert Tokenizing:  ['৩১', 'শে', 'জুলাই', 'এর', 'অর্ডার', 'এখনও', 'পাই', 'নাই']
    Truncating punctuation: ['৩১', 'শে', 'জুলাই', 'এর', 'অর্ডার', 'এখনও', 'পাই', 'নাই']
    Truncating StopWords: ['৩১', 'শে', 'জুলাই', 'অর্ডার', 'পাই', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "রাত ১২টাই অর্ডার করলাম,১দিনে দেওয়ার কথা,,এখনো তো পাইনি..."
    Afert Tokenizing:  ['রাত', '"', '১২টাই', 'অর্ডার', 'করলাম,১দিনে', 'দেওয়ার', 'কথা,,এখনো', 'তো', 'পাইনি...', '"']
    Truncating punctuation: ['রাত', '১২টাই', 'অর্ডার', 'করলাম,১দিনে', 'দেওয়ার', 'কথা,,এখনো', 'তো', 'পাইনি...']
    Truncating StopWords: ['রাত', '১২টাই', 'অর্ডার', 'করলাম,১দিনে', 'দেওয়ার', 'কথা,,এখনো', 'পাইনি...']
    ***************************************************************************************
    Label:  0
    Sentence:  "এইটা কখনোই হয়না। ডেলিভারী ম্যান গুলো ফাজিল আপনাদের। তারা প্রোডাক্ট নিয়ে বসে থাকে, ডেলিভারী ফেইল দেখিয়ে পরের দিন ডেলিভারী করে।"
    Afert Tokenizing:  ['এইটা', '"', 'কখনোই', 'হয়না', '।', 'ডেলিভারী', 'ম্যান', 'গুলো', 'ফাজিল', 'আপনাদের', '।', 'তারা', 'প্রোডাক্ট', 'নিয়ে', 'বসে', 'থাকে', ',', 'ডেলিভারী', 'ফেইল', 'দেখিয়ে', 'পরের', 'দিন', 'ডেলিভারী', 'করে।', '"']
    Truncating punctuation: ['এইটা', 'কখনোই', 'হয়না', 'ডেলিভারী', 'ম্যান', 'গুলো', 'ফাজিল', 'আপনাদের', 'তারা', 'প্রোডাক্ট', 'নিয়ে', 'বসে', 'থাকে', 'ডেলিভারী', 'ফেইল', 'দেখিয়ে', 'পরের', 'দিন', 'ডেলিভারী', 'করে।']
    Truncating StopWords: ['এইটা', 'কখনোই', 'হয়না', 'ডেলিভারী', 'ম্যান', 'গুলো', 'ফাজিল', 'আপনাদের', 'প্রোডাক্ট', 'ডেলিভারী', 'ফেইল', 'দেখিয়ে', 'পরের', 'ডেলিভারী', 'করে।']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ আমাদেরকে বোকা বানিয়ে ডেলিভারি চার্জ বেশি নিচ্ছে.আমি গত পরশু দিন একটি অর্ডার করেছিলাম এক্সপ্রেস ডেলিভারি চার্জ দিয়ে কিন্তু দারাজ আমাকে সেটা এখনো ডেলিভারি করতে পারে নাই।
    Afert Tokenizing:  ['দারাজ', 'আমাদেরকে', 'বোকা', 'বানিয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'নিচ্ছে.আমি', 'গত', 'পরশু', 'দিন', 'একটি', 'অর্ডার', 'করেছিলাম', 'এক্সপ্রেস', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'কিন্তু', 'দারাজ', 'আমাকে', 'সেটা', 'এখনো', 'ডেলিভারি', 'করতে', 'পারে', 'নাই', '।']
    Truncating punctuation: ['দারাজ', 'আমাদেরকে', 'বোকা', 'বানিয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'নিচ্ছে.আমি', 'গত', 'পরশু', 'দিন', 'একটি', 'অর্ডার', 'করেছিলাম', 'এক্সপ্রেস', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'কিন্তু', 'দারাজ', 'আমাকে', 'সেটা', 'এখনো', 'ডেলিভারি', 'করতে', 'পারে', 'নাই']
    Truncating StopWords: ['দারাজ', 'আমাদেরকে', 'বোকা', 'বানিয়ে', 'ডেলিভারি', 'চার্জ', 'বেশি', 'নিচ্ছে.আমি', 'গত', 'পরশু', 'অর্ডার', 'করেছিলাম', 'এক্সপ্রেস', 'ডেলিভারি', 'চার্জ', 'দিয়ে', 'দারাজ', 'এখনো', 'ডেলিভারি', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজ কি কখনো প্রতারণা করবে!!!
    Afert Tokenizing:  ['দারাজ', 'কি', 'কখনো', 'প্রতারণা', 'করবে!!', '!']
    Truncating punctuation: ['দারাজ', 'কি', 'কখনো', 'প্রতারণা', 'করবে!!']
    Truncating StopWords: ['দারাজ', 'কখনো', 'প্রতারণা', 'করবে!!']
    ***************************************************************************************
    Label:  0
    Sentence:  "প্রিয় দারাজ, খাগড়াছড়ির ডেলিভারির বয় গুলো মরছে নাকি একটু খোজ করে দেখেন গত পরশু ওদের কাছে প্রোডাক্স এসে গেছে কিন্তু ডেলিভারি দেওয়ার কোন নাম গন্ধ নাই "
    Afert Tokenizing:  ['প্রিয়', '"', 'দারাজ', ',', 'খাগড়াছড়ির', 'ডেলিভারির', 'বয়', 'গুলো', 'মরছে', 'নাকি', 'একটু', 'খোজ', 'করে', 'দেখেন', 'গত', 'পরশু', 'ওদের', 'কাছে', 'প্রোডাক্স', 'এসে', 'গেছে', 'কিন্তু', 'ডেলিভারি', 'দেওয়ার', 'কোন', 'নাম', 'গন্ধ', 'নাই', '', '"']
    Truncating punctuation: ['প্রিয়', 'দারাজ', 'খাগড়াছড়ির', 'ডেলিভারির', 'বয়', 'গুলো', 'মরছে', 'নাকি', 'একটু', 'খোজ', 'করে', 'দেখেন', 'গত', 'পরশু', 'ওদের', 'কাছে', 'প্রোডাক্স', 'এসে', 'গেছে', 'কিন্তু', 'ডেলিভারি', 'দেওয়ার', 'কোন', 'নাম', 'গন্ধ', 'নাই', '']
    Truncating StopWords: ['প্রিয়', 'দারাজ', 'খাগড়াছড়ির', 'ডেলিভারির', 'বয়', 'গুলো', 'মরছে', 'একটু', 'খোজ', 'দেখেন', 'গত', 'পরশু', 'প্রোডাক্স', 'ডেলিভারি', 'দেওয়ার', 'নাম', 'গন্ধ', 'নাই', '']
    ***************************************************************************************
    Label:  0
    Sentence:  অভিযোগ দিলে কাজ হয় না।
    Afert Tokenizing:  ['অভিযোগ', 'দিলে', 'কাজ', 'হয়', 'না', '।']
    Truncating punctuation: ['অভিযোগ', 'দিলে', 'কাজ', 'হয়', 'না']
    Truncating StopWords: ['অভিযোগ', 'দিলে', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এভাবে হয়রানি করা হলে ভোক্তা অধিকার আইনে কাষ্টমার হয়রানি করার অভিযোগ জানাতে বাধ্য হবো।
    Afert Tokenizing:  ['এভাবে', 'হয়রানি', 'করা', 'হলে', 'ভোক্তা', 'অধিকার', 'আইনে', 'কাষ্টমার', 'হয়রানি', 'করার', 'অভিযোগ', 'জানাতে', 'বাধ্য', 'হবো', '।']
    Truncating punctuation: ['এভাবে', 'হয়রানি', 'করা', 'হলে', 'ভোক্তা', 'অধিকার', 'আইনে', 'কাষ্টমার', 'হয়রানি', 'করার', 'অভিযোগ', 'জানাতে', 'বাধ্য', 'হবো']
    Truncating StopWords: ['এভাবে', 'হয়রানি', 'ভোক্তা', 'অধিকার', 'আইনে', 'কাষ্টমার', 'হয়রানি', 'অভিযোগ', 'জানাতে', 'বাধ্য', 'হবো']
    ***************************************************************************************
    Label:  0
    Sentence:  অফার তো নেই। যেগুলোতে ডিসকাউন্ট ছিলো সেগুলো এখন অরিজিনাল প্রাইস দেখাচ্ছে।
    Afert Tokenizing:  ['অফার', 'তো', 'নেই', '।', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'এখন', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে', '।']
    Truncating punctuation: ['অফার', 'তো', 'নেই', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'এখন', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে']
    Truncating StopWords: ['অফার', 'নেই', 'যেগুলোতে', 'ডিসকাউন্ট', 'ছিলো', 'সেগুলো', 'অরিজিনাল', 'প্রাইস', 'দেখাচ্ছে']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্য কমানোর দরকার। এবং যে কোন মূল্যের প্রোডাক্টে কয়েন এড করা জরুরি বলে আমি মনে করি।
    Afert Tokenizing:  ['ডেলিভারি', 'চার্য', 'কমানোর', 'দরকার', '।', 'এবং', 'যে', 'কোন', 'মূল্যের', 'প্রোডাক্টে', 'কয়েন', 'এড', 'করা', 'জরুরি', 'বলে', 'আমি', 'মনে', 'করি', '।']
    Truncating punctuation: ['ডেলিভারি', 'চার্য', 'কমানোর', 'দরকার', 'এবং', 'যে', 'কোন', 'মূল্যের', 'প্রোডাক্টে', 'কয়েন', 'এড', 'করা', 'জরুরি', 'বলে', 'আমি', 'মনে', 'করি']
    Truncating StopWords: ['ডেলিভারি', 'চার্য', 'কমানোর', 'দরকার', 'মূল্যের', 'প্রোডাক্টে', 'কয়েন', 'এড', 'জরুরি']
    ***************************************************************************************
    Label:  0
    Sentence:  রিটার্ন শুধু বিজ্ঞাপনে বাস্তবে হয়না।
    Afert Tokenizing:  ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না', '।']
    Truncating punctuation: ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না']
    Truncating StopWords: ['রিটার্ন', 'শুধু', 'বিজ্ঞাপনে', 'বাস্তবে', 'হয়না']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু একটা কোম্পানি আমাকে চুরি পাঠিয়েছে ডাট ভাঙ্গা
    Afert Tokenizing:  ['ফালতু', 'একটা', 'কোম্পানি', 'আমাকে', 'চুরি', 'পাঠিয়েছে', 'ডাট', 'ভাঙ্গা']
    Truncating punctuation: ['ফালতু', 'একটা', 'কোম্পানি', 'আমাকে', 'চুরি', 'পাঠিয়েছে', 'ডাট', 'ভাঙ্গা']
    Truncating StopWords: ['ফালতু', 'একটা', 'কোম্পানি', 'চুরি', 'পাঠিয়েছে', 'ডাট', 'ভাঙ্গা']
    ***************************************************************************************
    Label:  0
    Sentence:  একদম ফালতু সময় মতো ডেলিভারি পাওয়া জায় না 😒😤। আমার বন্ধুরা ঠিক ভলছে । যে সময় মতো পাবি না
    Afert Tokenizing:  ['একদম', 'ফালতু', 'সময়', 'মতো', 'ডেলিভারি', 'পাওয়া', 'জায়', 'না', '😒😤', '।', 'আমার', 'বন্ধুরা', 'ঠিক', 'ভলছে', '', '।', 'যে', 'সময়', 'মতো', 'পাবি', 'না']
    Truncating punctuation: ['একদম', 'ফালতু', 'সময়', 'মতো', 'ডেলিভারি', 'পাওয়া', 'জায়', 'না', '😒😤', 'আমার', 'বন্ধুরা', 'ঠিক', 'ভলছে', '', 'যে', 'সময়', 'মতো', 'পাবি', 'না']
    Truncating StopWords: ['একদম', 'ফালতু', 'সময়', 'ডেলিভারি', 'পাওয়া', 'জায়', 'না', '😒😤', 'বন্ধুরা', 'ঠিক', 'ভলছে', '', 'সময়', 'পাবি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি চার্জ কমান রেগুলার ক্রেতা ছিলাম
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'চার্জ', 'কমান', 'রেগুলার', 'ক্রেতা', 'ছিলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  ১০ দিনে ফ্রিজ পেলাম না। জানানো হয় না দিবে নাকি। আর যদি কিছু অর্ডার করছি
    Afert Tokenizing:  ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', '।', 'জানানো', 'হয়', 'না', 'দিবে', 'নাকি', '।', 'আর', 'যদি', 'কিছু', 'অর্ডার', 'করছি']
    Truncating punctuation: ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', 'জানানো', 'হয়', 'না', 'দিবে', 'নাকি', 'আর', 'যদি', 'কিছু', 'অর্ডার', 'করছি']
    Truncating StopWords: ['১০', 'দিনে', 'ফ্রিজ', 'পেলাম', 'না', 'না', 'দিবে', 'অর্ডার', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  আজ ১০৬৯ টাকার একটি অর্ডার ভিসা কার্ডে পেমেন্ট করলাম কিন্তু এখনো ক্যাশব্যাক পেলাম না
    Afert Tokenizing:  ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating punctuation: ['আজ', '১০৬৯', 'টাকার', 'একটি', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    Truncating StopWords: ['১০৬৯', 'টাকার', 'অর্ডার', 'ভিসা', 'কার্ডে', 'পেমেন্ট', 'করলাম', 'এখনো', 'ক্যাশব্যাক', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমার টাকা নিয়ে পণ্য দেয় নাই
    Afert Tokenizing:  ['আমার', '"', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating punctuation: ['আমার', 'টাকা', 'নিয়ে', 'পণ্য', 'দেয়', 'নাই']
    Truncating StopWords: ['টাকা', 'পণ্য', 'দেয়', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ কমানো হোক!
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক', '!']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'কমানো', 'হোক']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'কমানো']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার পন্য এখনো আসেনি
    Afert Tokenizing:  ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating punctuation: ['আমার', 'পন্য', 'এখনো', 'আসেনি']
    Truncating StopWords: ['পন্য', 'এখনো', 'আসেনি']
    ***************************************************************************************
    Label:  0
    Sentence:  "আগের দারাজই ভালো ছিলো,, এখন আর দারাজে অর্ডার করিনা একমাত্র ডেলিভারি চার্জ বাড়ানোর কারণে।।"
    Afert Tokenizing:  ['আগের', '"', 'দারাজই', 'ভালো', 'ছিলো,', ',', 'এখন', 'আর', 'দারাজে', 'অর্ডার', 'করিনা', 'একমাত্র', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'কারণে।।', '"']
    Truncating punctuation: ['আগের', 'দারাজই', 'ভালো', 'ছিলো,', 'এখন', 'আর', 'দারাজে', 'অর্ডার', 'করিনা', 'একমাত্র', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'কারণে।।']
    Truncating StopWords: ['আগের', 'দারাজই', 'ভালো', 'ছিলো,', 'দারাজে', 'অর্ডার', 'করিনা', 'একমাত্র', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'কারণে।।']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমি অর্ডার করছিলাম ওই শাট ,টা আর আমাকে দিছে এইটা ,, কেমন টা লাগে বলেন ?"
    Afert Tokenizing:  ['আমি', '"', 'অর্ডার', 'করছিলাম', 'ওই', 'শাট', 'টা', ',', 'আর', 'আমাকে', 'দিছে', 'এইটা', ',', ',', 'কেমন', 'টা', 'লাগে', 'বলেন', '?', '"']
    Truncating punctuation: ['আমি', 'অর্ডার', 'করছিলাম', 'ওই', 'শাট', 'টা', 'আর', 'আমাকে', 'দিছে', 'এইটা', 'কেমন', 'টা', 'লাগে', 'বলেন']
    Truncating StopWords: ['অর্ডার', 'করছিলাম', 'শাট', 'টা', 'দিছে', 'এইটা', 'কেমন', 'টা', 'লাগে']
    ***************************************************************************************
    Label:  0
    Sentence:  "ডেলিভারি চার্জটা কমান একটু!!!
    Afert Tokenizing:  ['ডেলিভারি', '"', 'চার্জটা', 'কমান', 'একটু!!', '!']
    Truncating punctuation: ['ডেলিভারি', 'চার্জটা', 'কমান', 'একটু!!']
    Truncating StopWords: ['ডেলিভারি', 'চার্জটা', 'কমান', 'একটু!!']
    ***************************************************************************************
    Label:  0
    Sentence:  "আগে মাসে ৫/৬ বার ডেলিভারি ম্যান আসতো প্রডাক্ট ডেলিভারি দিতে,,এখন এক বা এক বারও আসে না,,কারণ ডেলিভারি চার্জ বাড়ানোর জন্য অর্ডারই করি না আর"
    Afert Tokenizing:  ['আগে', '"', 'মাসে', '৫/৬', 'বার', 'ডেলিভারি', 'ম্যান', 'আসতো', 'প্রডাক্ট', 'ডেলিভারি', 'দিতে,,এখন', 'এক', 'বা', 'এক', 'বারও', 'আসে', 'না,,কারণ', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'জন্য', 'অর্ডারই', 'করি', 'না', 'আর', '"']
    Truncating punctuation: ['আগে', 'মাসে', '৫/৬', 'বার', 'ডেলিভারি', 'ম্যান', 'আসতো', 'প্রডাক্ট', 'ডেলিভারি', 'দিতে,,এখন', 'এক', 'বা', 'এক', 'বারও', 'আসে', 'না,,কারণ', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'জন্য', 'অর্ডারই', 'করি', 'না', 'আর']
    Truncating StopWords: ['মাসে', '৫/৬', 'ডেলিভারি', 'ম্যান', 'আসতো', 'প্রডাক্ট', 'ডেলিভারি', 'দিতে,,এখন', 'এক', 'এক', 'বারও', 'আসে', 'না,,কারণ', 'ডেলিভারি', 'চার্জ', 'বাড়ানোর', 'অর্ডারই', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  "প্রিয় দারাজ কোম্পানি, আমি একজন নিয়মিত ক্রেতা আপনাদের অনলাইন সপ এর।"
    Afert Tokenizing:  ['প্রিয়', '"', 'দারাজ', 'কোম্পানি', ',', 'আমি', 'একজন', 'নিয়মিত', 'ক্রেতা', 'আপনাদের', 'অনলাইন', 'সপ', 'এর।', '"']
    Truncating punctuation: ['প্রিয়', 'দারাজ', 'কোম্পানি', 'আমি', 'একজন', 'নিয়মিত', 'ক্রেতা', 'আপনাদের', 'অনলাইন', 'সপ', 'এর।']
    Truncating StopWords: ['প্রিয়', 'দারাজ', 'কোম্পানি', 'একজন', 'নিয়মিত', 'ক্রেতা', 'আপনাদের', 'অনলাইন', 'সপ', 'এর।']
    ***************************************************************************************
    Label:  0
    Sentence:  এরা সবাই ধান্দাবাজ ঠিক ঠাক মতো পণ্য ডেলিভারি দেয় না এরা
    Afert Tokenizing:  ['এরা', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating punctuation: ['এরা', 'সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'মতো', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না', 'এরা']
    Truncating StopWords: ['সবাই', 'ধান্দাবাজ', 'ঠিক', 'ঠাক', 'পণ্য', 'ডেলিভারি', 'দেয়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এই ওডারটা এখন ও পেলাম না
    Afert Tokenizing:  ['এই', 'ওডারটা', 'এখন', 'ও', 'পেলাম', 'না']
    Truncating punctuation: ['এই', 'ওডারটা', 'এখন', 'ও', 'পেলাম', 'না']
    Truncating StopWords: ['ওডারটা', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ এই মাত্র দারাজ থেকে ফেবরিলাইফ এর টি শার্ট পাইলাম
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'এই', 'মাত্র', 'দারাজ', 'থেকে', 'ফেবরিলাইফ', 'এর', 'টি', 'শার্ট', 'পাইলাম']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'এই', 'মাত্র', 'দারাজ', 'থেকে', 'ফেবরিলাইফ', 'এর', 'টি', 'শার্ট', 'পাইলাম']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'দারাজ', 'ফেবরিলাইফ', 'শার্ট', 'পাইলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  ডেলিভারি চার্জ অনেক বেশি
    Afert Tokenizing:  ['ডেলিভারি', 'চার্জ', 'অনেক', 'বেশি']
    Truncating punctuation: ['ডেলিভারি', 'চার্জ', 'অনেক', 'বেশি']
    Truncating StopWords: ['ডেলিভারি', 'চার্জ', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  আলহামদুলিল্লাহ আমি যত কেনাকাটা করছি সবি ঠিক ছিল
    Afert Tokenizing:  ['আলহামদুলিল্লাহ', 'আমি', 'যত', 'কেনাকাটা', 'করছি', 'সবি', 'ঠিক', 'ছিল']
    Truncating punctuation: ['আলহামদুলিল্লাহ', 'আমি', 'যত', 'কেনাকাটা', 'করছি', 'সবি', 'ঠিক', 'ছিল']
    Truncating StopWords: ['আলহামদুলিল্লাহ', 'কেনাকাটা', 'করছি', 'সবি', 'ঠিক']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্ডার করতে পারছিনা
    Afert Tokenizing:  ['অর্ডার', 'করতে', 'পারছিনা']
    Truncating punctuation: ['অর্ডার', 'করতে', 'পারছিনা']
    Truncating StopWords: ['অর্ডার', 'পারছিনা']
    ***************************************************************************************
    Label:  1
    Sentence:  দারাজে ট্রাস্টেড সকল ব্র্যান্ডের সেরা সব প্রোডাক্ট পাচ্ছেন ।
    Afert Tokenizing:  ['দারাজে', 'ট্রাস্টেড', 'সকল', 'ব্র্যান্ডের', 'সেরা', 'সব', 'প্রোডাক্ট', 'পাচ্ছেন', '', '।']
    Truncating punctuation: ['দারাজে', 'ট্রাস্টেড', 'সকল', 'ব্র্যান্ডের', 'সেরা', 'সব', 'প্রোডাক্ট', 'পাচ্ছেন', '']
    Truncating StopWords: ['দারাজে', 'ট্রাস্টেড', 'সকল', 'ব্র্যান্ডের', 'সেরা', 'প্রোডাক্ট', 'পাচ্ছেন', '']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি এপ্স থেকে অর্ডার করার চেস্টা করি কিন্তু হয়না
    Afert Tokenizing:  ['আমি', 'এপ্স', 'থেকে', 'অর্ডার', 'করার', 'চেস্টা', 'করি', 'কিন্তু', 'হয়না']
    Truncating punctuation: ['আমি', 'এপ্স', 'থেকে', 'অর্ডার', 'করার', 'চেস্টা', 'করি', 'কিন্তু', 'হয়না']
    Truncating StopWords: ['এপ্স', 'অর্ডার', 'চেস্টা', 'হয়না']
    ***************************************************************************************
    Label:  0
    Sentence:  প্রায় এক সপ্তাহ ধরে অর্ডার প্লেস করার চেষ্টা করছি।
    Afert Tokenizing:  ['প্রায়', 'এক', 'সপ্তাহ', 'ধরে', 'অর্ডার', 'প্লেস', 'করার', 'চেষ্টা', 'করছি', '।']
    Truncating punctuation: ['প্রায়', 'এক', 'সপ্তাহ', 'ধরে', 'অর্ডার', 'প্লেস', 'করার', 'চেষ্টা', 'করছি']
    Truncating StopWords: ['এক', 'সপ্তাহ', 'অর্ডার', 'প্লেস', 'চেষ্টা', 'করছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "ডেলিভারি অপশনে পিক আপ কালেকশন পয়েন্টে ক্লিক করলে সার্ভার সমস্যা, কানেকশন সমস্যা, এই সমস্যা, ঐ সমস্যা দেখিয়ে দিচ্ছে।"
    Afert Tokenizing:  ['ডেলিভারি', '"', 'অপশনে', 'পিক', 'আপ', 'কালেকশন', 'পয়েন্টে', 'ক্লিক', 'করলে', 'সার্ভার', 'সমস্যা', ',', 'কানেকশন', 'সমস্যা', ',', 'এই', 'সমস্যা', ',', 'ঐ', 'সমস্যা', 'দেখিয়ে', 'দিচ্ছে।', '"']
    Truncating punctuation: ['ডেলিভারি', 'অপশনে', 'পিক', 'আপ', 'কালেকশন', 'পয়েন্টে', 'ক্লিক', 'করলে', 'সার্ভার', 'সমস্যা', 'কানেকশন', 'সমস্যা', 'এই', 'সমস্যা', 'ঐ', 'সমস্যা', 'দেখিয়ে', 'দিচ্ছে।']
    Truncating StopWords: ['ডেলিভারি', 'অপশনে', 'পিক', 'আপ', 'কালেকশন', 'পয়েন্টে', 'ক্লিক', 'সার্ভার', 'সমস্যা', 'কানেকশন', 'সমস্যা', 'সমস্যা', 'সমস্যা', 'দেখিয়ে', 'দিচ্ছে।']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজে পেমেন্ট করলে সেই পণ্য আর দেয় না
    Afert Tokenizing:  ['দারাজে', 'পেমেন্ট', 'করলে', 'সেই', 'পণ্য', 'আর', 'দেয়', 'না']
    Truncating punctuation: ['দারাজে', 'পেমেন্ট', 'করলে', 'সেই', 'পণ্য', 'আর', 'দেয়', 'না']
    Truncating StopWords: ['দারাজে', 'পেমেন্ট', 'পণ্য', 'দেয়', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  দাদা আমি সব কিছু ভালো পেয়েছি।
    Afert Tokenizing:  ['দাদা', 'আমি', 'সব', 'কিছু', 'ভালো', 'পেয়েছি', '।']
    Truncating punctuation: ['দাদা', 'আমি', 'সব', 'কিছু', 'ভালো', 'পেয়েছি']
    Truncating StopWords: ['দাদা', 'ভালো', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমার সাথে প্রতারণা করা হয়ছে ২ টি পন্য দিয়ে,"
    Afert Tokenizing:  ['আমার', '"', 'সাথে', 'প্রতারণা', 'করা', 'হয়ছে', '২', 'টি', 'পন্য', 'দিয়ে,', '"']
    Truncating punctuation: ['আমার', 'সাথে', 'প্রতারণা', 'করা', 'হয়ছে', '২', 'টি', 'পন্য', 'দিয়ে,']
    Truncating StopWords: ['সাথে', 'প্রতারণা', 'হয়ছে', '২', 'পন্য', 'দিয়ে,']
    ***************************************************************************************
    Label:  0
    Sentence:  কোনো ভাবে দারাজ এপটা খুলতে পারতাছি না
    Afert Tokenizing:  ['কোনো', 'ভাবে', 'দারাজ', 'এপটা', 'খুলতে', 'পারতাছি', 'না']
    Truncating punctuation: ['কোনো', 'ভাবে', 'দারাজ', 'এপটা', 'খুলতে', 'পারতাছি', 'না']
    Truncating StopWords: ['দারাজ', 'এপটা', 'খুলতে', 'পারতাছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  অডার করার পর আমার জিনিস আসলো না কেন???
    Afert Tokenizing:  ['অডার', 'করার', 'পর', 'আমার', 'জিনিস', 'আসলো', 'না', 'কেন??', '?']
    Truncating punctuation: ['অডার', 'করার', 'পর', 'আমার', 'জিনিস', 'আসলো', 'না', 'কেন??']
    Truncating StopWords: ['অডার', 'জিনিস', 'আসলো', 'না', 'কেন??']
    ***************************************************************************************
    Label:  0
    Sentence:  দারাজে পেমেন্ট করলে সেই পণ্য আর দেয় না
    Afert Tokenizing:  ['দারাজে', 'পেমেন্ট', 'করলে', 'সেই', 'পণ্য', 'আর', 'দেয়', 'না']
    Truncating punctuation: ['দারাজে', 'পেমেন্ট', 'করলে', 'সেই', 'পণ্য', 'আর', 'দেয়', 'না']
    Truncating StopWords: ['দারাজে', 'পেমেন্ট', 'পণ্য', 'দেয়', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  দাদা আমি সব কিছু ভালো পেয়েছি।
    Afert Tokenizing:  ['দাদা', 'আমি', 'সব', 'কিছু', 'ভালো', 'পেয়েছি', '।']
    Truncating punctuation: ['দাদা', 'আমি', 'সব', 'কিছু', 'ভালো', 'পেয়েছি']
    Truncating StopWords: ['দাদা', 'ভালো', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  26 তারিখে রিটার্ন দিয়েছি আজকে 9 তারিখ এখনো কোনো খবর নাই
    Afert Tokenizing:  ['26', 'তারিখে', 'রিটার্ন', 'দিয়েছি', 'আজকে', '9', 'তারিখ', 'এখনো', 'কোনো', 'খবর', 'নাই']
    Truncating punctuation: ['26', 'তারিখে', 'রিটার্ন', 'দিয়েছি', 'আজকে', '9', 'তারিখ', 'এখনো', 'কোনো', 'খবর', 'নাই']
    Truncating StopWords: ['26', 'তারিখে', 'রিটার্ন', 'দিয়েছি', 'আজকে', '9', 'তারিখ', 'এখনো', 'খবর', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  যতসব ফালতু একটা চাইলে আরেক টা দেয়
    Afert Tokenizing:  ['যতসব', 'ফালতু', 'একটা', 'চাইলে', 'আরেক', 'টা', 'দেয়']
    Truncating punctuation: ['যতসব', 'ফালতু', 'একটা', 'চাইলে', 'আরেক', 'টা', 'দেয়']
    Truncating StopWords: ['যতসব', 'ফালতু', 'একটা', 'চাইলে', 'আরেক', 'টা', 'দেয়']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটপারেরা আমার ৪১৪টাকা কই?খারাপ মাল দিয়ে মানুষের টাকা মেরে দেওয়াকে বিজনেস বলে না এটাকে বলে বাটপারি।
    Afert Tokenizing:  ['বাটপারেরা', 'আমার', '৪১৪টাকা', 'কই?খারাপ', 'মাল', 'দিয়ে', 'মানুষের', 'টাকা', 'মেরে', 'দেওয়াকে', 'বিজনেস', 'বলে', 'না', 'এটাকে', 'বলে', 'বাটপারি', '।']
    Truncating punctuation: ['বাটপারেরা', 'আমার', '৪১৪টাকা', 'কই?খারাপ', 'মাল', 'দিয়ে', 'মানুষের', 'টাকা', 'মেরে', 'দেওয়াকে', 'বিজনেস', 'বলে', 'না', 'এটাকে', 'বলে', 'বাটপারি']
    Truncating StopWords: ['বাটপারেরা', '৪১৪টাকা', 'কই?খারাপ', 'মাল', 'দিয়ে', 'মানুষের', 'টাকা', 'মেরে', 'দেওয়াকে', 'বিজনেস', 'না', 'এটাকে', 'বাটপারি']
    ***************************************************************************************
    Label:  0
    Sentence:  এটার কোনো সমাধান পাচ্ছি না
    Afert Tokenizing:  ['এটার', 'কোনো', 'সমাধান', 'পাচ্ছি', 'না']
    Truncating punctuation: ['এটার', 'কোনো', 'সমাধান', 'পাচ্ছি', 'না']
    Truncating StopWords: ['এটার', 'সমাধান', 'পাচ্ছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি কিছু জিনিস অর্ডার করতে চাই কিন্তু পারছি না। আমাকে প্লিজ কাইন্ডলি জানাবেন এটা কি জন্য হচ্ছে?
    Afert Tokenizing:  ['আমি', 'কিছু', 'জিনিস', 'অর্ডার', 'করতে', 'চাই', 'কিন্তু', 'পারছি', 'না', '।', 'আমাকে', 'প্লিজ', 'কাইন্ডলি', 'জানাবেন', 'এটা', 'কি', 'জন্য', 'হচ্ছে', '?']
    Truncating punctuation: ['আমি', 'কিছু', 'জিনিস', 'অর্ডার', 'করতে', 'চাই', 'কিন্তু', 'পারছি', 'না', 'আমাকে', 'প্লিজ', 'কাইন্ডলি', 'জানাবেন', 'এটা', 'কি', 'জন্য', 'হচ্ছে']
    Truncating StopWords: ['জিনিস', 'অর্ডার', 'চাই', 'পারছি', 'না', 'প্লিজ', 'কাইন্ডলি', 'জানাবেন']
    ***************************************************************************************
    Label:  0
    Sentence:  "এই পন্যটি ভাঙ্গা অবস্থায় পাইছি,,রিটান করছি এখনো টাকা পাইনি,,,,ছিটার কোম্পানি"
    Afert Tokenizing:  ['এই', '"', 'পন্যটি', 'ভাঙ্গা', 'অবস্থায়', 'পাইছি,,রিটান', 'করছি', 'এখনো', 'টাকা', 'পাইনি,,,,ছিটার', 'কোম্পানি', '"']
    Truncating punctuation: ['এই', 'পন্যটি', 'ভাঙ্গা', 'অবস্থায়', 'পাইছি,,রিটান', 'করছি', 'এখনো', 'টাকা', 'পাইনি,,,,ছিটার', 'কোম্পানি']
    Truncating StopWords: ['পন্যটি', 'ভাঙ্গা', 'অবস্থায়', 'পাইছি,,রিটান', 'করছি', 'এখনো', 'টাকা', 'পাইনি,,,,ছিটার', 'কোম্পানি']
    ***************************************************************************************
    Label:  0
    Sentence:  গতকাল পণ্য অর্ডার করে বিকাশে পেমেন্ট করি।কিন্তূ ক্যাশব্যাক পাইনি।
    Afert Tokenizing:  ['গতকাল', 'পণ্য', 'অর্ডার', 'করে', 'বিকাশে', 'পেমেন্ট', 'করি।কিন্তূ', 'ক্যাশব্যাক', 'পাইনি', '।']
    Truncating punctuation: ['গতকাল', 'পণ্য', 'অর্ডার', 'করে', 'বিকাশে', 'পেমেন্ট', 'করি।কিন্তূ', 'ক্যাশব্যাক', 'পাইনি']
    Truncating StopWords: ['গতকাল', 'পণ্য', 'অর্ডার', 'বিকাশে', 'পেমেন্ট', 'করি।কিন্তূ', 'ক্যাশব্যাক', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  ৫টার আগে লিংকে ডুকেও দেখি স্টক আউট
    Afert Tokenizing:  ['৫টার', 'আগে', 'লিংকে', 'ডুকেও', 'দেখি', 'স্টক', 'আউট']
    Truncating punctuation: ['৫টার', 'আগে', 'লিংকে', 'ডুকেও', 'দেখি', 'স্টক', 'আউট']
    Truncating StopWords: ['৫টার', 'লিংকে', 'ডুকেও', 'দেখি', 'স্টক', 'আউট']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি ফোনটি গতকাল রিসিভ করেছি। কিন্তু প্যাকেটের ভিতরে কোন প্রকার বিল বা ইনভয়েজ পাইনি।
    Afert Tokenizing:  ['আমি', 'ফোনটি', 'গতকাল', 'রিসিভ', 'করেছি', '।', 'কিন্তু', 'প্যাকেটের', 'ভিতরে', 'কোন', 'প্রকার', 'বিল', 'বা', 'ইনভয়েজ', 'পাইনি', '।']
    Truncating punctuation: ['আমি', 'ফোনটি', 'গতকাল', 'রিসিভ', 'করেছি', 'কিন্তু', 'প্যাকেটের', 'ভিতরে', 'কোন', 'প্রকার', 'বিল', 'বা', 'ইনভয়েজ', 'পাইনি']
    Truncating StopWords: ['ফোনটি', 'গতকাল', 'রিসিভ', 'করেছি', 'প্যাকেটের', 'ভিতরে', 'প্রকার', 'বিল', 'ইনভয়েজ', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  চেক আউট করতে গেলে এই সমস্যা হচ্ছে কেন?
    Afert Tokenizing:  ['চেক', 'আউট', 'করতে', 'গেলে', 'এই', 'সমস্যা', 'হচ্ছে', 'কেন', '?']
    Truncating punctuation: ['চেক', 'আউট', 'করতে', 'গেলে', 'এই', 'সমস্যা', 'হচ্ছে', 'কেন']
    Truncating StopWords: ['চেক', 'আউট', 'সমস্যা']
    ***************************************************************************************
    Label:  0
    Sentence:  "ভাই রিয়েলমি সি৩১ অডার করে, ফোন হাতে পাইলাম না। প্রতারনা শিকার হয়ে দারাজ বাদ দিলাম। ফোন কিনা এত জবাবদিহিতা লজ্জা জনক।"
    Afert Tokenizing:  ['ভাই', '"', 'রিয়েলমি', 'সি৩১', 'অডার', 'করে', ',', 'ফোন', 'হাতে', 'পাইলাম', 'না', '।', 'প্রতারনা', 'শিকার', 'হয়ে', 'দারাজ', 'বাদ', 'দিলাম', '।', 'ফোন', 'কিনা', 'এত', 'জবাবদিহিতা', 'লজ্জা', 'জনক।', '"']
    Truncating punctuation: ['ভাই', 'রিয়েলমি', 'সি৩১', 'অডার', 'করে', 'ফোন', 'হাতে', 'পাইলাম', 'না', 'প্রতারনা', 'শিকার', 'হয়ে', 'দারাজ', 'বাদ', 'দিলাম', 'ফোন', 'কিনা', 'এত', 'জবাবদিহিতা', 'লজ্জা', 'জনক।']
    Truncating StopWords: ['ভাই', 'রিয়েলমি', 'সি৩১', 'অডার', 'ফোন', 'হাতে', 'পাইলাম', 'না', 'প্রতারনা', 'শিকার', 'হয়ে', 'দারাজ', 'বাদ', 'দিলাম', 'ফোন', 'কিনা', 'জবাবদিহিতা', 'লজ্জা', 'জনক।']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার পণ্য ডেলিভারি পাচ্ছি না কেন?
    Afert Tokenizing:  ['আমার', 'পণ্য', 'ডেলিভারি', 'পাচ্ছি', 'না', 'কেন', '?']
    Truncating punctuation: ['আমার', 'পণ্য', 'ডেলিভারি', 'পাচ্ছি', 'না', 'কেন']
    Truncating StopWords: ['পণ্য', 'ডেলিভারি', 'পাচ্ছি', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার প্যাকেট টা খোলা হয়েছিল এবং যত আজাইরা জিনিস দিয়ে দিছে
    Afert Tokenizing:  ['আমার', 'প্যাকেট', 'টা', 'খোলা', 'হয়েছিল', 'এবং', 'যত', 'আজাইরা', 'জিনিস', 'দিয়ে', 'দিছে']
    Truncating punctuation: ['আমার', 'প্যাকেট', 'টা', 'খোলা', 'হয়েছিল', 'এবং', 'যত', 'আজাইরা', 'জিনিস', 'দিয়ে', 'দিছে']
    Truncating StopWords: ['প্যাকেট', 'টা', 'খোলা', 'আজাইরা', 'জিনিস', 'দিছে']
    ***************************************************************************************
    Label:  0
    Sentence:  পন্য রিসিভ করার আগে চেক করার সুযোগ নেই কেন.
    Afert Tokenizing:  ['পন্য', 'রিসিভ', 'করার', 'আগে', 'চেক', 'করার', 'সুযোগ', 'নেই', 'কেন', '.']
    Truncating punctuation: ['পন্য', 'রিসিভ', 'করার', 'আগে', 'চেক', 'করার', 'সুযোগ', 'নেই', 'কেন']
    Truncating StopWords: ['পন্য', 'রিসিভ', 'চেক', 'সুযোগ', 'নেই']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রডাক্ট কোয়ালিটি খুব পছন্দ হয়েছে
    Afert Tokenizing:  ['প্রডাক্ট', 'কোয়ালিটি', 'খুব', 'পছন্দ', 'হয়েছে']
    Truncating punctuation: ['প্রডাক্ট', 'কোয়ালিটি', 'খুব', 'পছন্দ', 'হয়েছে']
    Truncating StopWords: ['প্রডাক্ট', 'কোয়ালিটি', 'পছন্দ', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  একমাত্র আপনাদের পেইজেই ভালো প্রডাক্ট পাওয়া যায়
    Afert Tokenizing:  ['একমাত্র', 'আপনাদের', 'পেইজেই', 'ভালো', 'প্রডাক্ট', 'পাওয়া', 'যায়']
    Truncating punctuation: ['একমাত্র', 'আপনাদের', 'পেইজেই', 'ভালো', 'প্রডাক্ট', 'পাওয়া', 'যায়']
    Truncating StopWords: ['একমাত্র', 'আপনাদের', 'পেইজেই', 'ভালো', 'প্রডাক্ট', 'পাওয়া', 'যায়']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি ১৯৩ টাকা ছাড় পেয়েছি।
    Afert Tokenizing:  ['আমি', '১৯৩', 'টাকা', 'ছাড়', 'পেয়েছি', '।']
    Truncating punctuation: ['আমি', '১৯৩', 'টাকা', 'ছাড়', 'পেয়েছি']
    Truncating StopWords: ['১৯৩', 'টাকা', 'ছাড়', 'পেয়েছি']
    ***************************************************************************************
    Label:  0
    Sentence:  "আমি আজকে পেমেন্ট করলাম, কিন্তু ক্যাশব্যাক পাইলাম না কেন???"
    Afert Tokenizing:  ['আমি', '"', 'আজকে', 'পেমেন্ট', 'করলাম', ',', 'কিন্তু', 'ক্যাশব্যাক', 'পাইলাম', 'না', 'কেন???', '"']
    Truncating punctuation: ['আমি', 'আজকে', 'পেমেন্ট', 'করলাম', 'কিন্তু', 'ক্যাশব্যাক', 'পাইলাম', 'না', 'কেন???']
    Truncating StopWords: ['আজকে', 'পেমেন্ট', 'করলাম', 'ক্যাশব্যাক', 'পাইলাম', 'না', 'কেন???']
    ***************************************************************************************
    Label:  0
    Sentence:  "সোনালী কালারের জুতাটা কিনছিলাম আগের দিন পরের দিনই ফিতা,হিল খুলে গেছে !
    Afert Tokenizing:  ['সোনালী', '"', 'কালারের', 'জুতাটা', 'কিনছিলাম', 'আগের', 'দিন', 'পরের', 'দিনই', 'ফিতা,হিল', 'খুলে', 'গেছে', '', '!']
    Truncating punctuation: ['সোনালী', 'কালারের', 'জুতাটা', 'কিনছিলাম', 'আগের', 'দিন', 'পরের', 'দিনই', 'ফিতা,হিল', 'খুলে', 'গেছে', '']
    Truncating StopWords: ['সোনালী', 'কালারের', 'জুতাটা', 'কিনছিলাম', 'আগের', 'পরের', 'দিনই', 'ফিতা,হিল', 'খুলে', '']
    ***************************************************************************************
    Label:  0
    Sentence:  "আজকে আমার অর্ডার,ডেলিভারি পাইছি। ডেলিভারি ম্যান চলে যাওয়ার পর দেখি জুতা অনেক ছোট হয়।"
    Afert Tokenizing:  ['আজকে', '"', 'আমার', 'অর্ডার,ডেলিভারি', 'পাইছি', '।', 'ডেলিভারি', 'ম্যান', 'চলে', 'যাওয়ার', 'পর', 'দেখি', 'জুতা', 'অনেক', 'ছোট', 'হয়।', '"']
    Truncating punctuation: ['আজকে', 'আমার', 'অর্ডার,ডেলিভারি', 'পাইছি', 'ডেলিভারি', 'ম্যান', 'চলে', 'যাওয়ার', 'পর', 'দেখি', 'জুতা', 'অনেক', 'ছোট', 'হয়।']
    Truncating StopWords: ['আজকে', 'অর্ডার,ডেলিভারি', 'পাইছি', 'ডেলিভারি', 'ম্যান', 'দেখি', 'জুতা', 'ছোট', 'হয়।']
    ***************************************************************************************
    Label:  1
    Sentence:  ড্রেস টা খুবেই সুন্দর
    Afert Tokenizing:  ['ড্রেস', 'টা', 'খুবেই', 'সুন্দর']
    Truncating punctuation: ['ড্রেস', 'টা', 'খুবেই', 'সুন্দর']
    Truncating StopWords: ['ড্রেস', 'টা', 'খুবেই', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  "চমৎকার, শুভ কামনা"
    Afert Tokenizing:  ['"চমৎকার', ',', 'শুভ', 'কামনা', '"']
    Truncating punctuation: ['"চমৎকার', 'শুভ', 'কামনা']
    Truncating StopWords: ['"চমৎকার', 'শুভ', 'কামনা']
    ***************************************************************************************
    Label:  1
    Sentence:  "বাহ সুন্দর মার্জিত রঙ এবং নকশা
    Afert Tokenizing:  ['বাহ', '"', 'সুন্দর', 'মার্জিত', 'রঙ', 'এবং', 'নকশা']
    Truncating punctuation: ['বাহ', 'সুন্দর', 'মার্জিত', 'রঙ', 'এবং', 'নকশা']
    Truncating StopWords: ['বাহ', 'সুন্দর', 'মার্জিত', 'রঙ', 'নকশা']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্যের দাম কম হলে ভালো হবে
    Afert Tokenizing:  ['পণ্যের', 'দাম', 'কম', 'হলে', 'ভালো', 'হবে']
    Truncating punctuation: ['পণ্যের', 'দাম', 'কম', 'হলে', 'ভালো', 'হবে']
    Truncating StopWords: ['পণ্যের', 'দাম', 'কম', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর কুর্তি
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'কুর্তি']
    Truncating punctuation: ['খুব', 'সুন্দর', 'কুর্তি']
    Truncating StopWords: ['সুন্দর', 'কুর্তি']
    ***************************************************************************************
    Label:  1
    Sentence:  "কালো পোষাক আরো সুন্দর
    Afert Tokenizing:  ['কালো', '"', 'পোষাক', 'আরো', 'সুন্দর']
    Truncating punctuation: ['কালো', 'পোষাক', 'আরো', 'সুন্দর']
    Truncating StopWords: ['কালো', 'পোষাক', 'আরো', 'সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনি ডেলিভারিতে দেরি করছেন
    Afert Tokenizing:  ['আপনি', 'ডেলিভারিতে', 'দেরি', 'করছেন']
    Truncating punctuation: ['আপনি', 'ডেলিভারিতে', 'দেরি', 'করছেন']
    Truncating StopWords: ['ডেলিভারিতে', 'দেরি']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনার সমস্ত পণ্য নকল
    Afert Tokenizing:  ['আপনার', 'সমস্ত', 'পণ্য', 'নকল']
    Truncating punctuation: ['আপনার', 'সমস্ত', 'পণ্য', 'নকল']
    Truncating StopWords: ['পণ্য', 'নকল']
    ***************************************************************************************
    Label:  1
    Sentence:  "ব্যাগটি আমিও নিয়েছি। চাকুরীজীবি মহিলাকে আকর্ষণীয় করে তুলে।
    Afert Tokenizing:  ['ব্যাগটি', '"', 'আমিও', 'নিয়েছি', '।', 'চাকুরীজীবি', 'মহিলাকে', 'আকর্ষণীয়', 'করে', 'তুলে', '।']
    Truncating punctuation: ['ব্যাগটি', 'আমিও', 'নিয়েছি', 'চাকুরীজীবি', 'মহিলাকে', 'আকর্ষণীয়', 'করে', 'তুলে']
    Truncating StopWords: ['ব্যাগটি', 'আমিও', 'নিয়েছি', 'চাকুরীজীবি', 'মহিলাকে', 'আকর্ষণীয়']
    ***************************************************************************************
    Label:  0
    Sentence:   আপনার পণ্যের দাম শিথিল করতে হবে
    Afert Tokenizing:  ['আপনার', 'পণ্যের', 'দাম', 'শিথিল', 'করতে', 'হবে']
    Truncating punctuation: ['আপনার', 'পণ্যের', 'দাম', 'শিথিল', 'করতে', 'হবে']
    Truncating StopWords: ['পণ্যের', 'দাম', 'শিথিল']
    ***************************************************************************************
    Label:  1
    Sentence:  এই আইটেম দীর্ঘস্থায়ী
    Afert Tokenizing:  ['এই', 'আইটেম', 'দীর্ঘস্থায়ী']
    Truncating punctuation: ['এই', 'আইটেম', 'দীর্ঘস্থায়ী']
    Truncating StopWords: ['আইটেম', 'দীর্ঘস্থায়ী']
    ***************************************************************************************
    Label:  0
    Sentence:  আমাকে বিকাশ ক্যাশব্যাক দেওয়া হয়নি কেন?
    Afert Tokenizing:  ['আমাকে', 'বিকাশ', 'ক্যাশব্যাক', 'দেওয়া', 'হয়নি', 'কেন', '?']
    Truncating punctuation: ['আমাকে', 'বিকাশ', 'ক্যাশব্যাক', 'দেওয়া', 'হয়নি', 'কেন']
    Truncating StopWords: ['বিকাশ', 'ক্যাশব্যাক', 'হয়নি']
    ***************************************************************************************
    Label:  0
    Sentence:  "৭ দিন হয়ে গেছে অর্ডার দিয়েছি, এখনো ডেলিভারি পাইনি"
    Afert Tokenizing:  ['৭', '"', 'দিন', 'হয়ে', 'গেছে', 'অর্ডার', 'দিয়েছি', ',', 'এখনো', 'ডেলিভারি', 'পাইনি', '"']
    Truncating punctuation: ['৭', 'দিন', 'হয়ে', 'গেছে', 'অর্ডার', 'দিয়েছি', 'এখনো', 'ডেলিভারি', 'পাইনি']
    Truncating StopWords: ['৭', 'হয়ে', 'অর্ডার', 'দিয়েছি', 'এখনো', 'ডেলিভারি', 'পাইনি']
    ***************************************************************************************
    Label:  0
    Sentence:  ভাউচার টিই তো ক্লেইম করা যাচ্ছে না
    Afert Tokenizing:  ['ভাউচার', 'টিই', 'তো', 'ক্লেইম', 'করা', 'যাচ্ছে', 'না']
    Truncating punctuation: ['ভাউচার', 'টিই', 'তো', 'ক্লেইম', 'করা', 'যাচ্ছে', 'না']
    Truncating StopWords: ['ভাউচার', 'টিই', 'ক্লেইম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  30 মার্চের মধ্যে দেয়ার কথা। আজ 30মার্চ এখনো কোনো মেসেজ পেলাম না
    Afert Tokenizing:  ['30', 'মার্চের', 'মধ্যে', 'দেয়ার', 'কথা', '।', 'আজ', '30মার্চ', 'এখনো', 'কোনো', 'মেসেজ', 'পেলাম', 'না']
    Truncating punctuation: ['30', 'মার্চের', 'মধ্যে', 'দেয়ার', 'কথা', 'আজ', '30মার্চ', 'এখনো', 'কোনো', 'মেসেজ', 'পেলাম', 'না']
    Truncating StopWords: ['30', 'মার্চের', 'দেয়ার', 'কথা', '30মার্চ', 'এখনো', 'মেসেজ', 'পেলাম', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  "স্পট নোটিশ ছিল ১০টায় লাইভ হবে, তাহলে আগে কেন করলো?? শুধু শুধু আমাদের মুল্যবান সময় গুলো নষ্ট করলো এত বড় প্রতিষ্ঠান থেকে এটা মোটেও কাম্য নয় "
    Afert Tokenizing:  ['স্পট', '"', 'নোটিশ', 'ছিল', '১০টায়', 'লাইভ', 'হবে', ',', 'তাহলে', 'আগে', 'কেন', 'করলো?', '?', 'শুধু', 'শুধু', 'আমাদের', 'মুল্যবান', 'সময়', 'গুলো', 'নষ্ট', 'করলো', 'এত', 'বড়', 'প্রতিষ্ঠান', 'থেকে', 'এটা', 'মোটেও', 'কাম্য', 'নয়', '', '"']
    Truncating punctuation: ['স্পট', 'নোটিশ', 'ছিল', '১০টায়', 'লাইভ', 'হবে', 'তাহলে', 'আগে', 'কেন', 'করলো?', 'শুধু', 'শুধু', 'আমাদের', 'মুল্যবান', 'সময়', 'গুলো', 'নষ্ট', 'করলো', 'এত', 'বড়', 'প্রতিষ্ঠান', 'থেকে', 'এটা', 'মোটেও', 'কাম্য', 'নয়', '']
    Truncating StopWords: ['স্পট', 'নোটিশ', '১০টায়', 'লাইভ', 'করলো?', 'শুধু', 'শুধু', 'মুল্যবান', 'সময়', 'গুলো', 'নষ্ট', 'করলো', 'বড়', 'প্রতিষ্ঠান', 'মোটেও', 'কাম্য', 'নয়', '']
    ***************************************************************************************
    Label:  1
    Sentence:  ২ দিন বেশি লাগলেও আজকে প্রডাক্টটি হাতে পেয়েছি
    Afert Tokenizing:  ['২', 'দিন', 'বেশি', 'লাগলেও', 'আজকে', 'প্রডাক্টটি', 'হাতে', 'পেয়েছি']
    Truncating punctuation: ['২', 'দিন', 'বেশি', 'লাগলেও', 'আজকে', 'প্রডাক্টটি', 'হাতে', 'পেয়েছি']
    Truncating StopWords: ['২', 'বেশি', 'লাগলেও', 'আজকে', 'প্রডাক্টটি', 'হাতে', 'পেয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  কাপড়টা অনেক ভালো লাগছে
    Afert Tokenizing:  ['কাপড়টা', 'অনেক', 'ভালো', 'লাগছে']
    Truncating punctuation: ['কাপড়টা', 'অনেক', 'ভালো', 'লাগছে']
    Truncating StopWords: ['কাপড়টা', 'ভালো', 'লাগছে']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের প্রডাক্টটি খুব ভালো
    Afert Tokenizing:  ['আপনাদের', 'প্রডাক্টটি', 'খুব', 'ভালো']
    Truncating punctuation: ['আপনাদের', 'প্রডাক্টটি', 'খুব', 'ভালো']
    Truncating StopWords: ['আপনাদের', 'প্রডাক্টটি', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনাদের ডেলিভারি সিস্টেমটা ভালো
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'সিস্টেমটা', 'ভালো']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'সিস্টেমটা', 'ভালো']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'সিস্টেমটা', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার এই বইটা আমার পড়া সেরা বই।
    Afert Tokenizing:  ['আপনার', 'এই', 'বইটা', 'আমার', 'পড়া', 'সেরা', 'বই', '।']
    Truncating punctuation: ['আপনার', 'এই', 'বইটা', 'আমার', 'পড়া', 'সেরা', 'বই']
    Truncating StopWords: ['বইটা', 'পড়া', 'সেরা', 'বই']
    ***************************************************************************************
    Label:  0
    Sentence:  ফালতু একটা জিনিস।
    Afert Tokenizing:  ['ফালতু', 'একটা', 'জিনিস', '।']
    Truncating punctuation: ['ফালতু', 'একটা', 'জিনিস']
    Truncating StopWords: ['ফালতু', 'একটা', 'জিনিস']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ এডমিন পেনেল
    Afert Tokenizing:  ['ধন্যবাদ', 'এডমিন', 'পেনেল']
    Truncating punctuation: ['ধন্যবাদ', 'এডমিন', 'পেনেল']
    Truncating StopWords: ['ধন্যবাদ', 'এডমিন', 'পেনেল']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর
    Afert Tokenizing:  ['খুব', 'সুন্দর']
    Truncating punctuation: ['খুব', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার কালেকশান
    Afert Tokenizing:  ['চমৎকার', 'কালেকশান']
    Truncating punctuation: ['চমৎকার', 'কালেকশান']
    Truncating StopWords: ['চমৎকার', 'কালেকশান']
    ***************************************************************************************
    Label:  1
    Sentence:  আসলেই সুন্দর
    Afert Tokenizing:  ['আসলেই', 'সুন্দর']
    Truncating punctuation: ['আসলেই', 'সুন্দর']
    Truncating StopWords: ['আসলেই', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশআল্লাহ্ চমৎকার
    Afert Tokenizing:  ['মাশআল্লাহ্', 'চমৎকার']
    Truncating punctuation: ['মাশআল্লাহ্', 'চমৎকার']
    Truncating StopWords: ['মাশআল্লাহ্', 'চমৎকার']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর ড্রেস টা
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'ড্রেস', 'টা']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'ড্রেস', 'টা']
    Truncating StopWords: ['সুন্দর', 'ড্রেস', 'টা']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর একটি কালার আপু
    Afert Tokenizing:  ['অনেক', 'সুন্দর', 'একটি', 'কালার', 'আপু']
    Truncating punctuation: ['অনেক', 'সুন্দর', 'একটি', 'কালার', 'আপু']
    Truncating StopWords: ['সুন্দর', 'কালার', 'আপু']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ্, দেখেই বোঝাা যাচ্ছে অনেক আরাম দায়ক, "
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ্', ',', 'দেখেই', 'বোঝাা', 'যাচ্ছে', 'অনেক', 'আরাম', 'দায়ক', ',', '', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ্', 'দেখেই', 'বোঝাা', 'যাচ্ছে', 'অনেক', 'আরাম', 'দায়ক', '']
    Truncating StopWords: ['"আলহামদুলিল্লাহ্', 'দেখেই', 'বোঝাা', 'আরাম', 'দায়ক', '']
    ***************************************************************************************
    Label:  1
    Sentence:  সব গুলো সুন্দর
    Afert Tokenizing:  ['সব', 'গুলো', 'সুন্দর']
    Truncating punctuation: ['সব', 'গুলো', 'সুন্দর']
    Truncating StopWords: ['গুলো', 'সুন্দর']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক রিজেনেবল
    Afert Tokenizing:  ['অনেক', 'রিজেনেবল']
    Truncating punctuation: ['অনেক', 'রিজেনেবল']
    Truncating StopWords: ['রিজেনেবল']
    ***************************************************************************************
    Label:  0
    Sentence:  "কোনো রকম যোগাযোগ না করেই ""ডেলিভারি ফেইলড""।"
    Afert Tokenizing:  ['কোনো', '"', 'রকম', 'যোগাযোগ', 'না', 'করেই', '"ডেলিভারি', '"', 'ফেইলড""।', '"']
    Truncating punctuation: ['কোনো', 'রকম', 'যোগাযোগ', 'না', 'করেই', '"ডেলিভারি', 'ফেইলড""।']
    Truncating StopWords: ['যোগাযোগ', 'না', '"ডেলিভারি', 'ফেইলড""।']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব খারাপ সার্ভিস বাসের টিকিট শোওজ
    Afert Tokenizing:  ['খুব', 'খারাপ', 'সার্ভিস', 'বাসের', 'টিকিট', 'শোওজ']
    Truncating punctuation: ['খুব', 'খারাপ', 'সার্ভিস', 'বাসের', 'টিকিট', 'শোওজ']
    Truncating StopWords: ['খারাপ', 'সার্ভিস', 'বাসের', 'টিকিট', 'শোওজ']
    ***************************************************************************************
    Label:  1
    Sentence:  ওয়াও দারুন
    Afert Tokenizing:  ['ওয়াও', 'দারুন']
    Truncating punctuation: ['ওয়াও', 'দারুন']
    Truncating StopWords: ['ওয়াও', 'দারুন']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই ভাল গ্রাহক সেবা
    Afert Tokenizing:  ['সত্যিই', 'ভাল', 'গ্রাহক', 'সেবা']
    Truncating punctuation: ['সত্যিই', 'ভাল', 'গ্রাহক', 'সেবা']
    Truncating StopWords: ['সত্যিই', 'ভাল', 'গ্রাহক', 'সেবা']
    ***************************************************************************************
    Label:  1
    Sentence:  ধন্যবাদ ভাইয়া..টি-শার্ট টা শুধু বাহ
    Afert Tokenizing:  ['ধন্যবাদ', 'ভাইয়া..টি-শার্ট', 'টা', 'শুধু', 'বাহ']
    Truncating punctuation: ['ধন্যবাদ', 'ভাইয়া..টি-শার্ট', 'টা', 'শুধু', 'বাহ']
    Truncating StopWords: ['ধন্যবাদ', 'ভাইয়া..টি-শার্ট', 'টা', 'শুধু', 'বাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই চমত্কার এবং চমৎকার মানের
    Afert Tokenizing:  ['সত্যিই', 'চমত্কার', 'এবং', 'চমৎকার', 'মানের']
    Truncating punctuation: ['সত্যিই', 'চমত্কার', 'এবং', 'চমৎকার', 'মানের']
    Truncating StopWords: ['সত্যিই', 'চমত্কার', 'চমৎকার', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  তাদের মহান সেবা সম্পূর্ণরূপে সন্তুষ্ট
    Afert Tokenizing:  ['তাদের', 'মহান', 'সেবা', 'সম্পূর্ণরূপে', 'সন্তুষ্ট']
    Truncating punctuation: ['তাদের', 'মহান', 'সেবা', 'সম্পূর্ণরূপে', 'সন্তুষ্ট']
    Truncating StopWords: ['মহান', 'সেবা', 'সম্পূর্ণরূপে', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্য এবং পরিষেবা দ্বারা খুব শান্তিপূর্ণভাবে সন্তুষ্ট. আশা করি আপনি আপনার গন্তব্যে পৌঁছে যাবেন (ইনশাআল্লাহ)
    Afert Tokenizing:  ['পণ্য', 'এবং', 'পরিষেবা', 'দ্বারা', 'খুব', 'শান্তিপূর্ণভাবে', 'সন্তুষ্ট', '.', 'আশা', 'করি', 'আপনি', 'আপনার', 'গন্তব্যে', 'পৌঁছে', 'যাবেন', '(ইনশাআল্লাহ', ')']
    Truncating punctuation: ['পণ্য', 'এবং', 'পরিষেবা', 'দ্বারা', 'খুব', 'শান্তিপূর্ণভাবে', 'সন্তুষ্ট', 'আশা', 'করি', 'আপনি', 'আপনার', 'গন্তব্যে', 'পৌঁছে', 'যাবেন', '(ইনশাআল্লাহ']
    Truncating StopWords: ['পণ্য', 'পরিষেবা', 'শান্তিপূর্ণভাবে', 'সন্তুষ্ট', 'আশা', 'গন্তব্যে', 'পৌঁছে', 'যাবেন', '(ইনশাআল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি তাদের পরিবেশিত পণ্যের গুণমান নিয়ে খুব সন্তুষ্ট
    Afert Tokenizing:  ['আমি', 'তাদের', 'পরিবেশিত', 'পণ্যের', 'গুণমান', 'নিয়ে', 'খুব', 'সন্তুষ্ট']
    Truncating punctuation: ['আমি', 'তাদের', 'পরিবেশিত', 'পণ্যের', 'গুণমান', 'নিয়ে', 'খুব', 'সন্তুষ্ট']
    Truncating StopWords: ['পরিবেশিত', 'পণ্যের', 'গুণমান', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব সুন্দর মানের এবং ভাল আচরণ। তাদের কাছ থেকে আবার কেনার জন্য উন্মুখ ইন শা আল্লাহ
    Afert Tokenizing:  ['খুব', 'সুন্দর', 'মানের', 'এবং', 'ভাল', 'আচরণ', '।', 'তাদের', 'কাছ', 'থেকে', 'আবার', 'কেনার', 'জন্য', 'উন্মুখ', 'ইন', 'শা', 'আল্লাহ']
    Truncating punctuation: ['খুব', 'সুন্দর', 'মানের', 'এবং', 'ভাল', 'আচরণ', 'তাদের', 'কাছ', 'থেকে', 'আবার', 'কেনার', 'জন্য', 'উন্মুখ', 'ইন', 'শা', 'আল্লাহ']
    Truncating StopWords: ['সুন্দর', 'মানের', 'ভাল', 'আচরণ', 'কেনার', 'উন্মুখ', 'ইন', 'শা', 'আল্লাহ']
    ***************************************************************************************
    Label:  1
    Sentence:  টি-শার্টের মান সত্যিই ভাল
    Afert Tokenizing:  ['টি-শার্টের', 'মান', 'সত্যিই', 'ভাল']
    Truncating punctuation: ['টি-শার্টের', 'মান', 'সত্যিই', 'ভাল']
    Truncating StopWords: ['টি-শার্টের', 'মান', 'সত্যিই', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  "ভাল মানের, কম দাম, প্রতিশ্রুতিবদ্ধ, ভাল আচরণ"
    Afert Tokenizing:  ['ভাল', '"', 'মানের', ',', 'কম', 'দাম', ',', 'প্রতিশ্রুতিবদ্ধ', ',', 'ভাল', 'আচরণ', '"']
    Truncating punctuation: ['ভাল', 'মানের', 'কম', 'দাম', 'প্রতিশ্রুতিবদ্ধ', 'ভাল', 'আচরণ']
    Truncating StopWords: ['ভাল', 'মানের', 'কম', 'দাম', 'প্রতিশ্রুতিবদ্ধ', 'ভাল', 'আচরণ']
    ***************************************************************************************
    Label:  1
    Sentence:  এই পণ্যের গুণমানও শীর্ষস্থানীয়।
    Afert Tokenizing:  ['এই', 'পণ্যের', 'গুণমানও', 'শীর্ষস্থানীয়', '।']
    Truncating punctuation: ['এই', 'পণ্যের', 'গুণমানও', 'শীর্ষস্থানীয়']
    Truncating StopWords: ['পণ্যের', 'গুণমানও', 'শীর্ষস্থানীয়']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক ধন্যবাদ। এটা চালিয়ে যান... কাস্টমাইজড টি-শার্ট সেক্টরে তাদের গুণমান এবং দাম যুক্তিসঙ্গত।
    Afert Tokenizing:  ['অনেক', 'ধন্যবাদ', '।', 'এটা', 'চালিয়ে', 'যান..', '.', 'কাস্টমাইজড', 'টি-শার্ট', 'সেক্টরে', 'তাদের', 'গুণমান', 'এবং', 'দাম', 'যুক্তিসঙ্গত', '।']
    Truncating punctuation: ['অনেক', 'ধন্যবাদ', 'এটা', 'চালিয়ে', 'যান..', 'কাস্টমাইজড', 'টি-শার্ট', 'সেক্টরে', 'তাদের', 'গুণমান', 'এবং', 'দাম', 'যুক্তিসঙ্গত']
    Truncating StopWords: ['ধন্যবাদ', 'চালিয়ে', 'যান..', 'কাস্টমাইজড', 'টি-শার্ট', 'সেক্টরে', 'গুণমান', 'দাম', 'যুক্তিসঙ্গত']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি তাদের এই দোকানের পরিষেবা নিয়ে অত্যন্ত সন্তুষ্ট
    Afert Tokenizing:  ['আমি', 'তাদের', 'এই', 'দোকানের', 'পরিষেবা', 'নিয়ে', 'অত্যন্ত', 'সন্তুষ্ট']
    Truncating punctuation: ['আমি', 'তাদের', 'এই', 'দোকানের', 'পরিষেবা', 'নিয়ে', 'অত্যন্ত', 'সন্তুষ্ট']
    Truncating StopWords: ['দোকানের', 'পরিষেবা', 'অত্যন্ত', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  "তারা খুব দক্ষ, আমার পণ্য খরপ সিলো বোলে তারা পরিবর্তন ওরে করে দিসে"
    Afert Tokenizing:  ['তারা', '"', 'খুব', 'দক্ষ', ',', 'আমার', 'পণ্য', 'খরপ', 'সিলো', 'বোলে', 'তারা', 'পরিবর্তন', 'ওরে', 'করে', 'দিসে', '"']
    Truncating punctuation: ['তারা', 'খুব', 'দক্ষ', 'আমার', 'পণ্য', 'খরপ', 'সিলো', 'বোলে', 'তারা', 'পরিবর্তন', 'ওরে', 'করে', 'দিসে']
    Truncating StopWords: ['দক্ষ', 'পণ্য', 'খরপ', 'সিলো', 'বোলে', 'পরিবর্তন', 'ওরে', 'দিসে']
    ***************************************************************************************
    Label:  1
    Sentence:  পণ্য নিয়া কোন অভিযোগ নাই।
    Afert Tokenizing:  ['পণ্য', 'নিয়া', 'কোন', 'অভিযোগ', 'নাই', '।']
    Truncating punctuation: ['পণ্য', 'নিয়া', 'কোন', 'অভিযোগ', 'নাই']
    Truncating StopWords: ['পণ্য', 'নিয়া', 'অভিযোগ', 'নাই']
    ***************************************************************************************
    Label:  0
    Sentence:  "ভীষণ বাজে একটা পেইজ,আমার পুরা টাকাটাই মাইর গেল"
    Afert Tokenizing:  ['ভীষণ', '"', 'বাজে', 'একটা', 'পেইজ,আমার', 'পুরা', 'টাকাটাই', 'মাইর', 'গেল', '"']
    Truncating punctuation: ['ভীষণ', 'বাজে', 'একটা', 'পেইজ,আমার', 'পুরা', 'টাকাটাই', 'মাইর', 'গেল']
    Truncating StopWords: ['ভীষণ', 'বাজে', 'একটা', 'পেইজ,আমার', 'পুরা', 'টাকাটাই', 'মাইর']
    ***************************************************************************************
    Label:  0
    Sentence:  "২টা নিয়েছিলাম একটা এক্স এল অন্যটা ডাবল এক্সেল,দিছে দুইটাই সেইম,বুকের মাপ ৪৮, আল্লাহ তোদের বিচার করবে"
    Afert Tokenizing:  ['২টা', '"', 'নিয়েছিলাম', 'একটা', 'এক্স', 'এল', 'অন্যটা', 'ডাবল', 'এক্সেল,দিছে', 'দুইটাই', 'সেইম,বুকের', 'মাপ', '৪৮', ',', 'আল্লাহ', 'তোদের', 'বিচার', 'করবে', '"']
    Truncating punctuation: ['২টা', 'নিয়েছিলাম', 'একটা', 'এক্স', 'এল', 'অন্যটা', 'ডাবল', 'এক্সেল,দিছে', 'দুইটাই', 'সেইম,বুকের', 'মাপ', '৪৮', 'আল্লাহ', 'তোদের', 'বিচার', 'করবে']
    Truncating StopWords: ['২টা', 'নিয়েছিলাম', 'একটা', 'এক্স', 'অন্যটা', 'ডাবল', 'এক্সেল,দিছে', 'দুইটাই', 'সেইম,বুকের', 'মাপ', '৪৮', 'আল্লাহ', 'তোদের', 'বিচার']
    ***************************************************************************************
    Label:  1
    Sentence:  সাধ্যের মধ্যে সবচেয়ে ভালো প্রোডাক্ট
    Afert Tokenizing:  ['সাধ্যের', 'মধ্যে', 'সবচেয়ে', 'ভালো', 'প্রোডাক্ট']
    Truncating punctuation: ['সাধ্যের', 'মধ্যে', 'সবচেয়ে', 'ভালো', 'প্রোডাক্ট']
    Truncating StopWords: ['সাধ্যের', 'সবচেয়ে', 'ভালো', 'প্রোডাক্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  "আলহামদুলিল্লাহ, আমার ২য় অর্ডার পেয়েছি। এটা বজায় রাখা…"
    Afert Tokenizing:  ['"আলহামদুলিল্লাহ', ',', 'আমার', '২য়', 'অর্ডার', 'পেয়েছি', '।', 'এটা', 'বজায়', 'রাখা…', '"']
    Truncating punctuation: ['"আলহামদুলিল্লাহ', 'আমার', '২য়', 'অর্ডার', 'পেয়েছি', 'এটা', 'বজায়', 'রাখা…']
    Truncating StopWords: ['"আলহামদুলিল্লাহ', '২য়', 'অর্ডার', 'পেয়েছি', 'বজায়', 'রাখা…']
    ***************************************************************************************
    Label:  0
    Sentence:  ত ৭ তারিখ আমার অর্ডার কনফার্ম করা হয়েছে। কিন্তু আজকে মেসেজ এসেছে সাইজ নেই বলে আমার অর্ডার বাতিল।
    Afert Tokenizing:  ['ত', '৭', 'তারিখ', 'আমার', 'অর্ডার', 'কনফার্ম', 'করা', 'হয়েছে', '।', 'কিন্তু', 'আজকে', 'মেসেজ', 'এসেছে', 'সাইজ', 'নেই', 'বলে', 'আমার', 'অর্ডার', 'বাতিল', '।']
    Truncating punctuation: ['ত', '৭', 'তারিখ', 'আমার', 'অর্ডার', 'কনফার্ম', 'করা', 'হয়েছে', 'কিন্তু', 'আজকে', 'মেসেজ', 'এসেছে', 'সাইজ', 'নেই', 'বলে', 'আমার', 'অর্ডার', 'বাতিল']
    Truncating StopWords: ['ত', '৭', 'তারিখ', 'অর্ডার', 'কনফার্ম', 'হয়েছে', 'আজকে', 'মেসেজ', 'এসেছে', 'সাইজ', 'নেই', 'অর্ডার', 'বাতিল']
    ***************************************************************************************
    Label:  0
    Sentence:  বাটা জুতোতে এ গ্রেড এবং বি গ্রেডের দুইধরনের জুতা আছে।বি গ্রেডের জুতো এর জুতা এ গ্রেডের জুতো থেকে একটু নিম্নমানের
    Afert Tokenizing:  ['বাটা', 'জুতোতে', 'এ', 'গ্রেড', 'এবং', 'বি', 'গ্রেডের', 'দুইধরনের', 'জুতা', 'আছে।বি', 'গ্রেডের', 'জুতো', 'এর', 'জুতা', 'এ', 'গ্রেডের', 'জুতো', 'থেকে', 'একটু', 'নিম্নমানের']
    Truncating punctuation: ['বাটা', 'জুতোতে', 'এ', 'গ্রেড', 'এবং', 'বি', 'গ্রেডের', 'দুইধরনের', 'জুতা', 'আছে।বি', 'গ্রেডের', 'জুতো', 'এর', 'জুতা', 'এ', 'গ্রেডের', 'জুতো', 'থেকে', 'একটু', 'নিম্নমানের']
    Truncating StopWords: ['বাটা', 'জুতোতে', 'গ্রেড', 'গ্রেডের', 'দুইধরনের', 'জুতা', 'আছে।বি', 'গ্রেডের', 'জুতো', 'জুতা', 'গ্রেডের', 'জুতো', 'একটু', 'নিম্নমানের']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনাদের সুজ এত বাজে ,"
    Afert Tokenizing:  ['আপনাদের', '"', 'সুজ', 'এত', 'বাজে', ',', '"']
    Truncating punctuation: ['আপনাদের', 'সুজ', 'এত', 'বাজে']
    Truncating StopWords: ['আপনাদের', 'সুজ', 'বাজে']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই দুঃখজনক এই পর্যন্ত অনলাইনে চারবার অর্ডার করলাম একবারও পাইলাম না
    Afert Tokenizing:  ['খুবই', 'দুঃখজনক', 'এই', 'পর্যন্ত', 'অনলাইনে', 'চারবার', 'অর্ডার', 'করলাম', 'একবারও', 'পাইলাম', 'না']
    Truncating punctuation: ['খুবই', 'দুঃখজনক', 'এই', 'পর্যন্ত', 'অনলাইনে', 'চারবার', 'অর্ডার', 'করলাম', 'একবারও', 'পাইলাম', 'না']
    Truncating StopWords: ['খুবই', 'দুঃখজনক', 'অনলাইনে', 'চারবার', 'অর্ডার', 'করলাম', 'একবারও', 'পাইলাম', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  আপনার একটি অনন্য সংগ্রহ আছে
    Afert Tokenizing:  ['আপনার', 'একটি', 'অনন্য', 'সংগ্রহ', 'আছে']
    Truncating punctuation: ['আপনার', 'একটি', 'অনন্য', 'সংগ্রহ', 'আছে']
    Truncating StopWords: ['অনন্য', 'সংগ্রহ']
    ***************************************************************************************
    Label:  0
    Sentence:  "আপনারা তো সময় মত ডেলিভারী করতে পারেন না, তাহলে আপনাদের অরডার দিয়ে কি লাভ"
    Afert Tokenizing:  ['আপনারা', '"', 'তো', 'সময়', 'মত', 'ডেলিভারী', 'করতে', 'পারেন', 'না', ',', 'তাহলে', 'আপনাদের', 'অরডার', 'দিয়ে', 'কি', 'লাভ', '"']
    Truncating punctuation: ['আপনারা', 'তো', 'সময়', 'মত', 'ডেলিভারী', 'করতে', 'পারেন', 'না', 'তাহলে', 'আপনাদের', 'অরডার', 'দিয়ে', 'কি', 'লাভ']
    Truncating StopWords: ['আপনারা', 'সময়', 'মত', 'ডেলিভারী', 'না', 'আপনাদের', 'অরডার', 'দিয়ে', 'লাভ']
    ***************************************************************************************
    Label:  1
    Sentence:  "খুবই ভালো একটা অফার ছিলো, আমি ও নিতে পেরেছি"
    Afert Tokenizing:  ['খুবই', '"', 'ভালো', 'একটা', 'অফার', 'ছিলো', ',', 'আমি', 'ও', 'নিতে', 'পেরেছি', '"']
    Truncating punctuation: ['খুবই', 'ভালো', 'একটা', 'অফার', 'ছিলো', 'আমি', 'ও', 'নিতে', 'পেরেছি']
    Truncating StopWords: ['খুবই', 'ভালো', 'একটা', 'অফার', 'ছিলো', 'পেরেছি']
    ***************************************************************************************
    Label:  0
    Sentence:  চিনিচম্পা কলার দাম বেশী মনে হয় আমার কাছে
    Afert Tokenizing:  ['চিনিচম্পা', 'কলার', 'দাম', 'বেশী', 'মনে', 'হয়', 'আমার', 'কাছে']
    Truncating punctuation: ['চিনিচম্পা', 'কলার', 'দাম', 'বেশী', 'মনে', 'হয়', 'আমার', 'কাছে']
    Truncating StopWords: ['চিনিচম্পা', 'কলার', 'দাম', 'বেশী']
    ***************************************************************************************
    Label:  0
    Sentence:  ডিস্কাউন্টের নামে ভাওতাবাজি।
    Afert Tokenizing:  ['ডিস্কাউন্টের', 'নামে', 'ভাওতাবাজি', '।']
    Truncating punctuation: ['ডিস্কাউন্টের', 'নামে', 'ভাওতাবাজি']
    Truncating StopWords: ['ডিস্কাউন্টের', 'নামে', 'ভাওতাবাজি']
    ***************************************************************************************
    Label:  0
    Sentence:  ১০০০ টাকার পেমেন্ট করছি কিন্তু কুপন পায়নি
    Afert Tokenizing:  ['১০০০', 'টাকার', 'পেমেন্ট', 'করছি', 'কিন্তু', 'কুপন', 'পায়নি']
    Truncating punctuation: ['১০০০', 'টাকার', 'পেমেন্ট', 'করছি', 'কিন্তু', 'কুপন', 'পায়নি']
    Truncating StopWords: ['১০০০', 'টাকার', 'পেমেন্ট', 'করছি', 'কুপন', 'পায়নি']
    ***************************************************************************************
    Label:  1
    Sentence:  গোল্ডেন রোজ ব্র্যান্ড এর প্রতিটা প্রডাক্ট অনেক ভালো। আমি নিজে ব্যবহার করি।
    Afert Tokenizing:  ['গোল্ডেন', 'রোজ', 'ব্র্যান্ড', 'এর', 'প্রতিটা', 'প্রডাক্ট', 'অনেক', 'ভালো', '।', 'আমি', 'নিজে', 'ব্যবহার', 'করি', '।']
    Truncating punctuation: ['গোল্ডেন', 'রোজ', 'ব্র্যান্ড', 'এর', 'প্রতিটা', 'প্রডাক্ট', 'অনেক', 'ভালো', 'আমি', 'নিজে', 'ব্যবহার', 'করি']
    Truncating StopWords: ['গোল্ডেন', 'রোজ', 'ব্র্যান্ড', 'প্রতিটা', 'প্রডাক্ট', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  কষ্টের টাকায় শ্রেষ্ঠ বাজার হোক স্বপ্ন তে
    Afert Tokenizing:  ['কষ্টের', 'টাকায়', 'শ্রেষ্ঠ', 'বাজার', 'হোক', 'স্বপ্ন', 'তে']
    Truncating punctuation: ['কষ্টের', 'টাকায়', 'শ্রেষ্ঠ', 'বাজার', 'হোক', 'স্বপ্ন', 'তে']
    Truncating StopWords: ['কষ্টের', 'টাকায়', 'শ্রেষ্ঠ', 'বাজার', 'স্বপ্ন', 'তে']
    ***************************************************************************************
    Label:  0
    Sentence:  বাজে টেস্ট
    Afert Tokenizing:  ['বাজে', 'টেস্ট']
    Truncating punctuation: ['বাজে', 'টেস্ট']
    Truncating StopWords: ['বাজে', 'টেস্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ স্বাদ
    Afert Tokenizing:  ['অসাধারণ', 'স্বাদ']
    Truncating punctuation: ['অসাধারণ', 'স্বাদ']
    Truncating StopWords: ['অসাধারণ', 'স্বাদ']
    ***************************************************************************************
    Label:  1
    Sentence:  অবশেষে পেয়ে গেলাম অসাধারণ এক পিজ্জা
    Afert Tokenizing:  ['অবশেষে', 'পেয়ে', 'গেলাম', 'অসাধারণ', 'এক', 'পিজ্জা']
    Truncating punctuation: ['অবশেষে', 'পেয়ে', 'গেলাম', 'অসাধারণ', 'এক', 'পিজ্জা']
    Truncating StopWords: ['অবশেষে', 'পেয়ে', 'গেলাম', 'অসাধারণ', 'এক', 'পিজ্জা']
    ***************************************************************************************
    Label:  1
    Sentence:  বিফ চিজ ব্লাস্ট আমার প্রিয়
    Afert Tokenizing:  ['বিফ', 'চিজ', 'ব্লাস্ট', 'আমার', 'প্রিয়']
    Truncating punctuation: ['বিফ', 'চিজ', 'ব্লাস্ট', 'আমার', 'প্রিয়']
    Truncating StopWords: ['বিফ', 'চিজ', 'ব্লাস্ট', 'প্রিয়']
    ***************************************************************************************
    Label:  1
    Sentence:  কালাভুনা টা আসলে অত্যান্ত ভালো ছিল
    Afert Tokenizing:  ['কালাভুনা', 'টা', 'আসলে', 'অত্যান্ত', 'ভালো', 'ছিল']
    Truncating punctuation: ['কালাভুনা', 'টা', 'আসলে', 'অত্যান্ত', 'ভালো', 'ছিল']
    Truncating StopWords: ['কালাভুনা', 'টা', 'আসলে', 'অত্যান্ত', 'ভালো']
    ***************************************************************************************
    Label:  0
    Sentence:  দামটা একটু বেশি
    Afert Tokenizing:  ['দামটা', 'একটু', 'বেশি']
    Truncating punctuation: ['দামটা', 'একটু', 'বেশি']
    Truncating StopWords: ['দামটা', 'একটু', 'বেশি']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব্বি মজার আর অনেক অনেক চিজ
    Afert Tokenizing:  ['খুব্বি', 'মজার', 'আর', 'অনেক', 'অনেক', 'চিজ']
    Truncating punctuation: ['খুব্বি', 'মজার', 'আর', 'অনেক', 'অনেক', 'চিজ']
    Truncating StopWords: ['খুব্বি', 'মজার', 'চিজ']
    ***************************************************************************************
    Label:  1
    Sentence:  মাস্ট ট্রাই আইটেম
    Afert Tokenizing:  ['মাস্ট', 'ট্রাই', 'আইটেম']
    Truncating punctuation: ['মাস্ট', 'ট্রাই', 'আইটেম']
    Truncating StopWords: ['মাস্ট', 'ট্রাই', 'আইটেম']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার জিবনে আমার খাওয়া সবচেয়ে বাজে খাবার ছিল আজকে
    Afert Tokenizing:  ['আমার', 'জিবনে', 'আমার', 'খাওয়া', 'সবচেয়ে', 'বাজে', 'খাবার', 'ছিল', 'আজকে']
    Truncating punctuation: ['আমার', 'জিবনে', 'আমার', 'খাওয়া', 'সবচেয়ে', 'বাজে', 'খাবার', 'ছিল', 'আজকে']
    Truncating StopWords: ['জিবনে', 'খাওয়া', 'সবচেয়ে', 'বাজে', 'খাবার', 'আজকে']
    ***************************************************************************************
    Label:  0
    Sentence:  বারগার র বান টা একদমি ফ্রেশ ছিলো না
    Afert Tokenizing:  ['বারগার', 'র', 'বান', 'টা', 'একদমি', 'ফ্রেশ', 'ছিলো', 'না']
    Truncating punctuation: ['বারগার', 'র', 'বান', 'টা', 'একদমি', 'ফ্রেশ', 'ছিলো', 'না']
    Truncating StopWords: ['বারগার', 'বান', 'টা', 'একদমি', 'ফ্রেশ', 'ছিলো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা স্বাদহীন ছিল
    Afert Tokenizing:  ['এটা', 'স্বাদহীন', 'ছিল']
    Truncating punctuation: ['এটা', 'স্বাদহীন', 'ছিল']
    Truncating StopWords: ['স্বাদহীন']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব খারাপ মানের খাবার
    Afert Tokenizing:  ['খুব', 'খারাপ', 'মানের', 'খাবার']
    Truncating punctuation: ['খুব', 'খারাপ', 'মানের', 'খাবার']
    Truncating StopWords: ['খারাপ', 'মানের', 'খাবার']
    ***************************************************************************************
    Label:  0
    Sentence:  বেশ খারাপ অভিজ্ঞতা
    Afert Tokenizing:  ['বেশ', 'খারাপ', 'অভিজ্ঞতা']
    Truncating punctuation: ['বেশ', 'খারাপ', 'অভিজ্ঞতা']
    Truncating StopWords: ['খারাপ', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার প্রধান সমস্যা ছিল ব্যবস্থাপনা নিয়ে
    Afert Tokenizing:  ['আমার', 'প্রধান', 'সমস্যা', 'ছিল', 'ব্যবস্থাপনা', 'নিয়ে']
    Truncating punctuation: ['আমার', 'প্রধান', 'সমস্যা', 'ছিল', 'ব্যবস্থাপনা', 'নিয়ে']
    Truncating StopWords: ['প্রধান', 'সমস্যা', 'ব্যবস্থাপনা']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার মোটেও ভালো লাগে নাই আর অনেক বেশি ঝাল ছিল
    Afert Tokenizing:  ['আমার', 'মোটেও', 'ভালো', 'লাগে', 'নাই', 'আর', 'অনেক', 'বেশি', 'ঝাল', 'ছিল']
    Truncating punctuation: ['আমার', 'মোটেও', 'ভালো', 'লাগে', 'নাই', 'আর', 'অনেক', 'বেশি', 'ঝাল', 'ছিল']
    Truncating StopWords: ['মোটেও', 'ভালো', 'লাগে', 'নাই', 'বেশি', 'ঝাল']
    ***************************************************************************************
    Label:  0
    Sentence:  তোফু পুরানো এবং বাসি স্বাদের
    Afert Tokenizing:  ['তোফু', 'পুরানো', 'এবং', 'বাসি', 'স্বাদের']
    Truncating punctuation: ['তোফু', 'পুরানো', 'এবং', 'বাসি', 'স্বাদের']
    Truncating StopWords: ['তোফু', 'পুরানো', 'বাসি', 'স্বাদের']
    ***************************************************************************************
    Label:  0
    Sentence:  সম্প্রতি তাদের মান খুব খারাপ হয়েছে
    Afert Tokenizing:  ['সম্প্রতি', 'তাদের', 'মান', 'খুব', 'খারাপ', 'হয়েছে']
    Truncating punctuation: ['সম্প্রতি', 'তাদের', 'মান', 'খুব', 'খারাপ', 'হয়েছে']
    Truncating StopWords: ['মান', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  এটি তাজা ছিল না যদিও তারা এটি বলে দাবি করে এবং এটি আক্ষরিক অর্থে আমার কাছে সবচেয়ে খারাপ কেক ছিল
    Afert Tokenizing:  ['এটি', 'তাজা', 'ছিল', 'না', 'যদিও', 'তারা', 'এটি', 'বলে', 'দাবি', 'করে', 'এবং', 'এটি', 'আক্ষরিক', 'অর্থে', 'আমার', 'কাছে', 'সবচেয়ে', 'খারাপ', 'কেক', 'ছিল']
    Truncating punctuation: ['এটি', 'তাজা', 'ছিল', 'না', 'যদিও', 'তারা', 'এটি', 'বলে', 'দাবি', 'করে', 'এবং', 'এটি', 'আক্ষরিক', 'অর্থে', 'আমার', 'কাছে', 'সবচেয়ে', 'খারাপ', 'কেক', 'ছিল']
    Truncating StopWords: ['তাজা', 'না', 'দাবি', 'আক্ষরিক', 'অর্থে', 'সবচেয়ে', 'খারাপ', 'কেক']
    ***************************************************************************************
    Label:  0
    Sentence:  তাদের আইসক্রিমগুলোও খুব খারাপ
    Afert Tokenizing:  ['তাদের', 'আইসক্রিমগুলোও', 'খুব', 'খারাপ']
    Truncating punctuation: ['তাদের', 'আইসক্রিমগুলোও', 'খুব', 'খারাপ']
    Truncating StopWords: ['আইসক্রিমগুলোও', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি তাদের খাবারে খুব হতাশ ছিলাম
    Afert Tokenizing:  ['আমি', 'তাদের', 'খাবারে', 'খুব', 'হতাশ', 'ছিলাম']
    Truncating punctuation: ['আমি', 'তাদের', 'খাবারে', 'খুব', 'হতাশ', 'ছিলাম']
    Truncating StopWords: ['খাবারে', 'হতাশ', 'ছিলাম']
    ***************************************************************************************
    Label:  0
    Sentence:  অর্থের অপচয়
    Afert Tokenizing:  ['অর্থের', 'অপচয়']
    Truncating punctuation: ['অর্থের', 'অপচয়']
    Truncating StopWords: ['অর্থের', 'অপচয়']
    ***************************************************************************************
    Label:  0
    Sentence:  স্বাদ যথেষ্ট ভাল ছিল না
    Afert Tokenizing:  ['স্বাদ', 'যথেষ্ট', 'ভাল', 'ছিল', 'না']
    Truncating punctuation: ['স্বাদ', 'যথেষ্ট', 'ভাল', 'ছিল', 'না']
    Truncating StopWords: ['স্বাদ', 'যথেষ্ট', 'ভাল', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  এটা সত্যিই খারাপ
    Afert Tokenizing:  ['এটা', 'সত্যিই', 'খারাপ']
    Truncating punctuation: ['এটা', 'সত্যিই', 'খারাপ']
    Truncating StopWords: ['সত্যিই', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  খাবার দেখতে বেশ স্বাদহীন
    Afert Tokenizing:  ['খাবার', 'দেখতে', 'বেশ', 'স্বাদহীন']
    Truncating punctuation: ['খাবার', 'দেখতে', 'বেশ', 'স্বাদহীন']
    Truncating StopWords: ['খাবার', 'স্বাদহীন']
    ***************************************************************************************
    Label:  0
    Sentence:  বান টাও খুব একটা ভালো কোয়ালিটি ছিলো না
    Afert Tokenizing:  ['বান', 'টাও', 'খুব', 'একটা', 'ভালো', 'কোয়ালিটি', 'ছিলো', 'না']
    Truncating punctuation: ['বান', 'টাও', 'খুব', 'একটা', 'ভালো', 'কোয়ালিটি', 'ছিলো', 'না']
    Truncating StopWords: ['বান', 'টাও', 'একটা', 'ভালো', 'কোয়ালিটি', 'ছিলো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  সত্যি বলতে আমি খুবই হতাশ
    Afert Tokenizing:  ['সত্যি', 'বলতে', 'আমি', 'খুবই', 'হতাশ']
    Truncating punctuation: ['সত্যি', 'বলতে', 'আমি', 'খুবই', 'হতাশ']
    Truncating StopWords: ['সত্যি', 'খুবই', 'হতাশ']
    ***************************************************************************************
    Label:  0
    Sentence:  এমন জঘন্য বার্গার সত্যি টাকাই লস
    Afert Tokenizing:  ['এমন', 'জঘন্য', 'বার্গার', 'সত্যি', 'টাকাই', 'লস']
    Truncating punctuation: ['এমন', 'জঘন্য', 'বার্গার', 'সত্যি', 'টাকাই', 'লস']
    Truncating StopWords: ['জঘন্য', 'বার্গার', 'সত্যি', 'টাকাই', 'লস']
    ***************************************************************************************
    Label:  0
    Sentence:  আমার ভয়ঙ্কর অভিজ্ঞতার পরে কয়েকজনের সাথে কথা বলেছি এবং জানানো হয়েছিল তারাও একই জিনিসের মুখোমুখি হয়েছিল
    Afert Tokenizing:  ['আমার', 'ভয়ঙ্কর', 'অভিজ্ঞতার', 'পরে', 'কয়েকজনের', 'সাথে', 'কথা', 'বলেছি', 'এবং', 'জানানো', 'হয়েছিল', 'তারাও', 'একই', 'জিনিসের', 'মুখোমুখি', 'হয়েছিল']
    Truncating punctuation: ['আমার', 'ভয়ঙ্কর', 'অভিজ্ঞতার', 'পরে', 'কয়েকজনের', 'সাথে', 'কথা', 'বলেছি', 'এবং', 'জানানো', 'হয়েছিল', 'তারাও', 'একই', 'জিনিসের', 'মুখোমুখি', 'হয়েছিল']
    Truncating StopWords: ['ভয়ঙ্কর', 'অভিজ্ঞতার', 'কয়েকজনের', 'সাথে', 'কথা', 'বলেছি', 'তারাও', 'জিনিসের', 'মুখোমুখি']
    ***************************************************************************************
    Label:  0
    Sentence:  বিশ্বাস করুন এটি মোটেও থাই স্যুপের মতো স্বাদ ছিল না
    Afert Tokenizing:  ['বিশ্বাস', 'করুন', 'এটি', 'মোটেও', 'থাই', 'স্যুপের', 'মতো', 'স্বাদ', 'ছিল', 'না']
    Truncating punctuation: ['বিশ্বাস', 'করুন', 'এটি', 'মোটেও', 'থাই', 'স্যুপের', 'মতো', 'স্বাদ', 'ছিল', 'না']
    Truncating StopWords: ['বিশ্বাস', 'করুন', 'মোটেও', 'থাই', 'স্যুপের', 'স্বাদ', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  চিকেন মাঞ্চুরিয়ানের স্বাদ খুব খারাপ
    Afert Tokenizing:  ['চিকেন', 'মাঞ্চুরিয়ানের', 'স্বাদ', 'খুব', 'খারাপ']
    Truncating punctuation: ['চিকেন', 'মাঞ্চুরিয়ানের', 'স্বাদ', 'খুব', 'খারাপ']
    Truncating StopWords: ['চিকেন', 'মাঞ্চুরিয়ানের', 'স্বাদ', 'খারাপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি এই ডেজার্ট নিয়ে সত্যিই হতাশ
    Afert Tokenizing:  ['আমি', 'এই', 'ডেজার্ট', 'নিয়ে', 'সত্যিই', 'হতাশ']
    Truncating punctuation: ['আমি', 'এই', 'ডেজার্ট', 'নিয়ে', 'সত্যিই', 'হতাশ']
    Truncating StopWords: ['ডেজার্ট', 'সত্যিই', 'হতাশ']
    ***************************************************************************************
    Label:  0
    Sentence:  স্লাইস এত ছোট ছিল এবং এটি খুব খারাপ স্বাদ ছিল
    Afert Tokenizing:  ['স্লাইস', 'এত', 'ছোট', 'ছিল', 'এবং', 'এটি', 'খুব', 'খারাপ', 'স্বাদ', 'ছিল']
    Truncating punctuation: ['স্লাইস', 'এত', 'ছোট', 'ছিল', 'এবং', 'এটি', 'খুব', 'খারাপ', 'স্বাদ', 'ছিল']
    Truncating StopWords: ['স্লাইস', 'ছোট', 'খারাপ', 'স্বাদ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি মনে করি এটা তাজা ছিল না
    Afert Tokenizing:  ['আমি', 'মনে', 'করি', 'এটা', 'তাজা', 'ছিল', 'না']
    Truncating punctuation: ['আমি', 'মনে', 'করি', 'এটা', 'তাজা', 'ছিল', 'না']
    Truncating StopWords: ['তাজা', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  স্টারটেক এর বিক্রয়োত্তর সাপোর্ট সবথেকে ভালো
    Afert Tokenizing:  ['স্টারটেক', 'এর', 'বিক্রয়োত্তর', 'সাপোর্ট', 'সবথেকে', 'ভালো']
    Truncating punctuation: ['স্টারটেক', 'এর', 'বিক্রয়োত্তর', 'সাপোর্ট', 'সবথেকে', 'ভালো']
    Truncating StopWords: ['স্টারটেক', 'বিক্রয়োত্তর', 'সাপোর্ট', 'সবথেকে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  স্টারটেক এর প্রত্যেকটা সেলসপারসনের বিহেভিয়ার খুব ভালো
    Afert Tokenizing:  ['স্টারটেক', 'এর', 'প্রত্যেকটা', 'সেলসপারসনের', 'বিহেভিয়ার', 'খুব', 'ভালো']
    Truncating punctuation: ['স্টারটেক', 'এর', 'প্রত্যেকটা', 'সেলসপারসনের', 'বিহেভিয়ার', 'খুব', 'ভালো']
    Truncating StopWords: ['স্টারটেক', 'প্রত্যেকটা', 'সেলসপারসনের', 'বিহেভিয়ার', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুন অফার
    Afert Tokenizing:  ['দারুন', 'অফার']
    Truncating punctuation: ['দারুন', 'অফার']
    Truncating StopWords: ['দারুন', 'অফার']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার সংগ্রহ
    Afert Tokenizing:  ['চমৎকার', 'সংগ্রহ']
    Truncating punctuation: ['চমৎকার', 'সংগ্রহ']
    Truncating StopWords: ['চমৎকার', 'সংগ্রহ']
    ***************************************************************************************
    Label:  1
    Sentence:  অনেক সুন্দর
    Afert Tokenizing:  ['অনেক', 'সুন্দর']
    Truncating punctuation: ['অনেক', 'সুন্দর']
    Truncating StopWords: ['সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  ওয়েবসাইট বন্ধ কেন
    Afert Tokenizing:  ['ওয়েবসাইট', 'বন্ধ', 'কেন']
    Truncating punctuation: ['ওয়েবসাইট', 'বন্ধ', 'কেন']
    Truncating StopWords: ['ওয়েবসাইট', 'বন্ধ']
    ***************************************************************************************
    Label:  0
    Sentence:  এই সব বিশ্বাস করবেন না দয়া করে
    Afert Tokenizing:  ['এই', 'সব', 'বিশ্বাস', 'করবেন', 'না', 'দয়া', 'করে']
    Truncating punctuation: ['এই', 'সব', 'বিশ্বাস', 'করবেন', 'না', 'দয়া', 'করে']
    Truncating StopWords: ['বিশ্বাস', 'না', 'দয়া']
    ***************************************************************************************
    Label:  0
    Sentence:  আপনাদের ডেলিভারি সার্ভিস খুবই বাজে
    Afert Tokenizing:  ['আপনাদের', 'ডেলিভারি', 'সার্ভিস', 'খুবই', 'বাজে']
    Truncating punctuation: ['আপনাদের', 'ডেলিভারি', 'সার্ভিস', 'খুবই', 'বাজে']
    Truncating StopWords: ['আপনাদের', 'ডেলিভারি', 'সার্ভিস', 'খুবই', 'বাজে']
    ***************************************************************************************
    Label:  1
    Sentence:  প্যাটি সঠিকভাবে রান্না করা হয়েছিল এবং এতে সঠিক মশলা রয়েছে
    Afert Tokenizing:  ['প্যাটি', 'সঠিকভাবে', 'রান্না', 'করা', 'হয়েছিল', 'এবং', 'এতে', 'সঠিক', 'মশলা', 'রয়েছে']
    Truncating punctuation: ['প্যাটি', 'সঠিকভাবে', 'রান্না', 'করা', 'হয়েছিল', 'এবং', 'এতে', 'সঠিক', 'মশলা', 'রয়েছে']
    Truncating StopWords: ['প্যাটি', 'সঠিকভাবে', 'রান্না', 'সঠিক', 'মশলা']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই এই বার্গার অনেক পছন্দ
    Afert Tokenizing:  ['সত্যিই', 'এই', 'বার্গার', 'অনেক', 'পছন্দ']
    Truncating punctuation: ['সত্যিই', 'এই', 'বার্গার', 'অনেক', 'পছন্দ']
    Truncating StopWords: ['সত্যিই', 'বার্গার', 'পছন্দ']
    ***************************************************************************************
    Label:  1
    Sentence:  এটা বেশ রিফ্রেশিং
    Afert Tokenizing:  ['এটা', 'বেশ', 'রিফ্রেশিং']
    Truncating punctuation: ['এটা', 'বেশ', 'রিফ্রেশিং']
    Truncating StopWords: ['রিফ্রেশিং']
    ***************************************************************************************
    Label:  1
    Sentence:  সব মিলিয়ে বেশ ভালো
    Afert Tokenizing:  ['সব', 'মিলিয়ে', 'বেশ', 'ভালো']
    Truncating punctuation: ['সব', 'মিলিয়ে', 'বেশ', 'ভালো']
    Truncating StopWords: ['মিলিয়ে', 'ভালো']
    ***************************************************************************************
    Label:  1
    Sentence:  হাইজিন মেইনটেইন করে বানানো যেটা সবচেয়ে বেশি ভালো লেগেছে
    Afert Tokenizing:  ['হাইজিন', 'মেইনটেইন', 'করে', 'বানানো', 'যেটা', 'সবচেয়ে', 'বেশি', 'ভালো', 'লেগেছে']
    Truncating punctuation: ['হাইজিন', 'মেইনটেইন', 'করে', 'বানানো', 'যেটা', 'সবচেয়ে', 'বেশি', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['হাইজিন', 'মেইনটেইন', 'বানানো', 'যেটা', 'সবচেয়ে', 'বেশি', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  মাটন পিস টা ওয়েল কুকড
    Afert Tokenizing:  ['মাটন', 'পিস', 'টা', 'ওয়েল', 'কুকড']
    Truncating punctuation: ['মাটন', 'পিস', 'টা', 'ওয়েল', 'কুকড']
    Truncating StopWords: ['মাটন', 'পিস', 'টা', 'ওয়েল', 'কুকড']
    ***************************************************************************************
    Label:  1
    Sentence:  পরিমাণ ও বেশ ভালো দুই পিস মাটন থাকে
    Afert Tokenizing:  ['পরিমাণ', 'ও', 'বেশ', 'ভালো', 'দুই', 'পিস', 'মাটন', 'থাকে']
    Truncating punctuation: ['পরিমাণ', 'ও', 'বেশ', 'ভালো', 'দুই', 'পিস', 'মাটন', 'থাকে']
    Truncating StopWords: ['পরিমাণ', 'ভালো', 'পিস', 'মাটন']
    ***************************************************************************************
    Label:  1
    Sentence:  আচার দিয়ে খেতে আরো ভালো লাগছিলো
    Afert Tokenizing:  ['আচার', 'দিয়ে', 'খেতে', 'আরো', 'ভালো', 'লাগছিলো']
    Truncating punctuation: ['আচার', 'দিয়ে', 'খেতে', 'আরো', 'ভালো', 'লাগছিলো']
    Truncating StopWords: ['আচার', 'দিয়ে', 'খেতে', 'আরো', 'ভালো', 'লাগছিলো']
    ***************************************************************************************
    Label:  1
    Sentence:  বাটার চিকেন টাও বেশ ভালো লেগেছে
    Afert Tokenizing:  ['বাটার', 'চিকেন', 'টাও', 'বেশ', 'ভালো', 'লেগেছে']
    Truncating punctuation: ['বাটার', 'চিকেন', 'টাও', 'বেশ', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['বাটার', 'চিকেন', 'টাও', 'ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  0
    Sentence:  তাদের সার্ভিস আরো ডেভলপ করতে হবে
    Afert Tokenizing:  ['তাদের', 'সার্ভিস', 'আরো', 'ডেভলপ', 'করতে', 'হবে']
    Truncating punctuation: ['তাদের', 'সার্ভিস', 'আরো', 'ডেভলপ', 'করতে', 'হবে']
    Truncating StopWords: ['সার্ভিস', 'আরো', 'ডেভলপ']
    ***************************************************************************************
    Label:  0
    Sentence:  আমি সম্প্রতি তাদের সুইস চকোলেট আইসক্রিম চেষ্টা করেছি এবং স্বাদে হতাশ হয়েছি
    Afert Tokenizing:  ['আমি', 'সম্প্রতি', 'তাদের', 'সুইস', 'চকোলেট', 'আইসক্রিম', 'চেষ্টা', 'করেছি', 'এবং', 'স্বাদে', 'হতাশ', 'হয়েছি']
    Truncating punctuation: ['আমি', 'সম্প্রতি', 'তাদের', 'সুইস', 'চকোলেট', 'আইসক্রিম', 'চেষ্টা', 'করেছি', 'এবং', 'স্বাদে', 'হতাশ', 'হয়েছি']
    Truncating StopWords: ['সুইস', 'চকোলেট', 'আইসক্রিম', 'চেষ্টা', 'করেছি', 'স্বাদে', 'হতাশ', 'হয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  মাশরুম সস একটি ভাল স্বাদ যোগ করেছে এবং ম্যাশড আলু একেবারে চমত্কার ছিল
    Afert Tokenizing:  ['মাশরুম', 'সস', 'একটি', 'ভাল', 'স্বাদ', 'যোগ', 'করেছে', 'এবং', 'ম্যাশড', 'আলু', 'একেবারে', 'চমত্কার', 'ছিল']
    Truncating punctuation: ['মাশরুম', 'সস', 'একটি', 'ভাল', 'স্বাদ', 'যোগ', 'করেছে', 'এবং', 'ম্যাশড', 'আলু', 'একেবারে', 'চমত্কার', 'ছিল']
    Truncating StopWords: ['মাশরুম', 'সস', 'ভাল', 'স্বাদ', 'যোগ', 'ম্যাশড', 'আলু', 'একেবারে', 'চমত্কার']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব সীমিত আইটেম
    Afert Tokenizing:  ['খুব', 'সীমিত', 'আইটেম']
    Truncating punctuation: ['খুব', 'সীমিত', 'আইটেম']
    Truncating StopWords: ['সীমিত', 'আইটেম']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি এত সুস্বাদু খাবার কখনও খাইনি
    Afert Tokenizing:  ['আমি', 'এত', 'সুস্বাদু', 'খাবার', 'কখনও', 'খাইনি']
    Truncating punctuation: ['আমি', 'এত', 'সুস্বাদু', 'খাবার', 'কখনও', 'খাইনি']
    Truncating StopWords: ['সুস্বাদু', 'খাবার', 'খাইনি']
    ***************************************************************************************
    Label:  1
    Sentence:  পিজাটা চমৎকার ছিল
    Afert Tokenizing:  ['পিজাটা', 'চমৎকার', 'ছিল']
    Truncating punctuation: ['পিজাটা', 'চমৎকার', 'ছিল']
    Truncating StopWords: ['পিজাটা', 'চমৎকার']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই খেতে দারুণ টেস্ট
    Afert Tokenizing:  ['সত্যিই', 'খেতে', 'দারুণ', 'টেস্ট']
    Truncating punctuation: ['সত্যিই', 'খেতে', 'দারুণ', 'টেস্ট']
    Truncating StopWords: ['সত্যিই', 'খেতে', 'দারুণ', 'টেস্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারটা অসাধারণ
    Afert Tokenizing:  ['খাবারটা', 'অসাধারণ']
    Truncating punctuation: ['খাবারটা', 'অসাধারণ']
    Truncating StopWords: ['খাবারটা', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  প্রতিটি খাদ্য সত্যিই  মজাদার এবং যুক্তিসংগত মূল্য
    Afert Tokenizing:  ['প্রতিটি', 'খাদ্য', 'সত্যিই', 'মজাদার', 'এবং', 'যুক্তিসংগত', 'মূল্য']
    Truncating punctuation: ['প্রতিটি', 'খাদ্য', 'সত্যিই', 'মজাদার', 'এবং', 'যুক্তিসংগত', 'মূল্য']
    Truncating StopWords: ['প্রতিটি', 'খাদ্য', 'সত্যিই', 'মজাদার', 'যুক্তিসংগত', 'মূল্য']
    ***************************************************************************************
    Label:  0
    Sentence:  পুরো টাকা টাই জলে
    Afert Tokenizing:  ['পুরো', 'টাকা', 'টাই', 'জলে']
    Truncating punctuation: ['পুরো', 'টাকা', 'টাই', 'জলে']
    Truncating StopWords: ['পুরো', 'টাকা', 'টাই', 'জলে']
    ***************************************************************************************
    Label:  0
    Sentence:  পুরো টাকা টাই জলে
    Afert Tokenizing:  ['পুরো', 'টাকা', 'টাই', 'জলে']
    Truncating punctuation: ['পুরো', 'টাকা', 'টাই', 'জলে']
    Truncating StopWords: ['পুরো', 'টাকা', 'টাই', 'জলে']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার খাবার
    Afert Tokenizing:  ['চমৎকার', 'খাবার']
    Truncating punctuation: ['চমৎকার', 'খাবার']
    Truncating StopWords: ['চমৎকার', 'খাবার']
    ***************************************************************************************
    Label:  1
    Sentence:  সার্ভিসটা খুব ভাল এবং খাবারের মানও খুব ভাল
    Afert Tokenizing:  ['সার্ভিসটা', 'খুব', 'ভাল', 'এবং', 'খাবারের', 'মানও', 'খুব', 'ভাল']
    Truncating punctuation: ['সার্ভিসটা', 'খুব', 'ভাল', 'এবং', 'খাবারের', 'মানও', 'খুব', 'ভাল']
    Truncating StopWords: ['সার্ভিসটা', 'ভাল', 'খাবারের', 'মানও', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  ইফতারের মেনু ভাল ছিল
    Afert Tokenizing:  ['ইফতারের', 'মেনু', 'ভাল', 'ছিল']
    Truncating punctuation: ['ইফতারের', 'মেনু', 'ভাল', 'ছিল']
    Truncating StopWords: ['ইফতারের', 'মেনু', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারের মান খুব ভাল ছিল
    Afert Tokenizing:  ['খাবারের', 'মান', 'খুব', 'ভাল', 'ছিল']
    Truncating punctuation: ['খাবারের', 'মান', 'খুব', 'ভাল', 'ছিল']
    Truncating StopWords: ['খাবারের', 'মান', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  হাইলি রিকমেন্ডড.
    Afert Tokenizing:  ['হাইলি', 'রিকমেন্ডড', '.']
    Truncating punctuation: ['হাইলি', 'রিকমেন্ডড']
    Truncating StopWords: ['হাইলি', 'রিকমেন্ডড']
    ***************************************************************************************
    Label:  1
    Sentence:  যুক্তিসংগত দাম সাথে খাবারটা এবং সার্ভিসটা চমৎকার মানের
    Afert Tokenizing:  ['যুক্তিসংগত', 'দাম', 'সাথে', 'খাবারটা', 'এবং', 'সার্ভিসটা', 'চমৎকার', 'মানের']
    Truncating punctuation: ['যুক্তিসংগত', 'দাম', 'সাথে', 'খাবারটা', 'এবং', 'সার্ভিসটা', 'চমৎকার', 'মানের']
    Truncating StopWords: ['যুক্তিসংগত', 'দাম', 'সাথে', 'খাবারটা', 'সার্ভিসটা', 'চমৎকার', 'মানের']
    ***************************************************************************************
    Label:  1
    Sentence:  সর্বোপরি আচরণ অনেক ভাল ছিল
    Afert Tokenizing:  ['সর্বোপরি', 'আচরণ', 'অনেক', 'ভাল', 'ছিল']
    Truncating punctuation: ['সর্বোপরি', 'আচরণ', 'অনেক', 'ভাল', 'ছিল']
    Truncating StopWords: ['সর্বোপরি', 'আচরণ', 'ভাল']
    ***************************************************************************************
    Label:  0
    Sentence:  নিশ্চয়ই এই জায়গাটায় আবার আসব
    Afert Tokenizing:  ['নিশ্চয়ই', 'এই', 'জায়গাটায়', 'আবার', 'আসব']
    Truncating punctuation: ['নিশ্চয়ই', 'এই', 'জায়গাটায়', 'আবার', 'আসব']
    Truncating StopWords: ['নিশ্চয়ই', 'জায়গাটায়', 'আসব']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি তাদের আচরণ এবং সার্ভিস নিয়ে খুবই সন্তুষ্ট
    Afert Tokenizing:  ['আমি', 'তাদের', 'আচরণ', 'এবং', 'সার্ভিস', 'নিয়ে', 'খুবই', 'সন্তুষ্ট']
    Truncating punctuation: ['আমি', 'তাদের', 'আচরণ', 'এবং', 'সার্ভিস', 'নিয়ে', 'খুবই', 'সন্তুষ্ট']
    Truncating StopWords: ['আচরণ', 'সার্ভিস', 'খুবই', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবার ছিল দারুণ  আর বার্গারটা ছিল অসাধারণ
    Afert Tokenizing:  ['খাবার', 'ছিল', 'দারুণ', 'আর', 'বার্গারটা', 'ছিল', 'অসাধারণ']
    Truncating punctuation: ['খাবার', 'ছিল', 'দারুণ', 'আর', 'বার্গারটা', 'ছিল', 'অসাধারণ']
    Truncating StopWords: ['খাবার', 'দারুণ', 'বার্গারটা', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  নম্র আচরণ এবং খাবার এত স্বাস্থ্যকর এবং সুস্বাদু ছিল
    Afert Tokenizing:  ['নম্র', 'আচরণ', 'এবং', 'খাবার', 'এত', 'স্বাস্থ্যকর', 'এবং', 'সুস্বাদু', 'ছিল']
    Truncating punctuation: ['নম্র', 'আচরণ', 'এবং', 'খাবার', 'এত', 'স্বাস্থ্যকর', 'এবং', 'সুস্বাদু', 'ছিল']
    Truncating StopWords: ['নম্র', 'আচরণ', 'খাবার', 'স্বাস্থ্যকর', 'সুস্বাদু']
    ***************************************************************************************
    Label:  1
    Sentence:  আমরা দুজনই আপনাদের খাবারের ফ্যান হয়ে গিয়েছি
    Afert Tokenizing:  ['আমরা', 'দুজনই', 'আপনাদের', 'খাবারের', 'ফ্যান', 'হয়ে', 'গিয়েছি']
    Truncating punctuation: ['আমরা', 'দুজনই', 'আপনাদের', 'খাবারের', 'ফ্যান', 'হয়ে', 'গিয়েছি']
    Truncating StopWords: ['দুজনই', 'আপনাদের', 'খাবারের', 'ফ্যান', 'হয়ে', 'গিয়েছি']
    ***************************************************************************************
    Label:  1
    Sentence:  সস্তা দামের সাথে চমৎকার খাবার
    Afert Tokenizing:  ['সস্তা', 'দামের', 'সাথে', 'চমৎকার', 'খাবার']
    Truncating punctuation: ['সস্তা', 'দামের', 'সাথে', 'চমৎকার', 'খাবার']
    Truncating StopWords: ['সস্তা', 'দামের', 'সাথে', 'চমৎকার', 'খাবার']
    ***************************************************************************************
    Label:  1
    Sentence:  পিজা এবং অন্যান্য খাবার চমৎকার ছিল এবং সার্ভিস ও ভাল
    Afert Tokenizing:  ['পিজা', 'এবং', 'অন্যান্য', 'খাবার', 'চমৎকার', 'ছিল', 'এবং', 'সার্ভিস', 'ও', 'ভাল']
    Truncating punctuation: ['পিজা', 'এবং', 'অন্যান্য', 'খাবার', 'চমৎকার', 'ছিল', 'এবং', 'সার্ভিস', 'ও', 'ভাল']
    Truncating StopWords: ['পিজা', 'অন্যান্য', 'খাবার', 'চমৎকার', 'সার্ভিস', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  "সুস্বাদু খাদ্য, তাদের সেবায় আমি সন্তুষ্ট"
    Afert Tokenizing:  ['সুস্বাদু', '"', 'খাদ্য', ',', 'তাদের', 'সেবায়', 'আমি', 'সন্তুষ্ট', '"']
    Truncating punctuation: ['সুস্বাদু', 'খাদ্য', 'তাদের', 'সেবায়', 'আমি', 'সন্তুষ্ট']
    Truncating StopWords: ['সুস্বাদু', 'খাদ্য', 'সেবায়', 'সন্তুষ্ট']
    ***************************************************************************************
    Label:  1
    Sentence:  দারুন খাবার. বিশেষ করে চিকেন মশলা
    Afert Tokenizing:  ['দারুন', 'খাবার', '.', 'বিশেষ', 'করে', 'চিকেন', 'মশলা']
    Truncating punctuation: ['দারুন', 'খাবার', 'বিশেষ', 'করে', 'চিকেন', 'মশলা']
    Truncating StopWords: ['দারুন', 'খাবার', 'বিশেষ', 'চিকেন', 'মশলা']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ মানের খাবার .....
    Afert Tokenizing:  ['অসাধারণ', 'মানের', 'খাবার', '....', '.']
    Truncating punctuation: ['অসাধারণ', 'মানের', 'খাবার', '....']
    Truncating StopWords: ['অসাধারণ', 'মানের', 'খাবার', '....']
    ***************************************************************************************
    Label:  1
    Sentence:  চমত্কার খাদ্য
    Afert Tokenizing:  ['চমত্কার', 'খাদ্য']
    Truncating punctuation: ['চমত্কার', 'খাদ্য']
    Truncating StopWords: ['চমত্কার', 'খাদ্য']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল খাদ্য এবং অসাধারণ সেবা
    Afert Tokenizing:  ['ভাল', 'খাদ্য', 'এবং', 'অসাধারণ', 'সেবা']
    Truncating punctuation: ['ভাল', 'খাদ্য', 'এবং', 'অসাধারণ', 'সেবা']
    Truncating StopWords: ['ভাল', 'খাদ্য', 'অসাধারণ', 'সেবা']
    ***************************************************************************************
    Label:  1
    Sentence:  "মানসম্মত খাদ্য, অসাধারণ পরিবেশ এবং সর্বোত্তম সেবা"
    Afert Tokenizing:  ['মানসম্মত', '"', 'খাদ্য', ',', 'অসাধারণ', 'পরিবেশ', 'এবং', 'সর্বোত্তম', 'সেবা', '"']
    Truncating punctuation: ['মানসম্মত', 'খাদ্য', 'অসাধারণ', 'পরিবেশ', 'এবং', 'সর্বোত্তম', 'সেবা']
    Truncating StopWords: ['মানসম্মত', 'খাদ্য', 'অসাধারণ', 'পরিবেশ', 'সর্বোত্তম', 'সেবা']
    ***************************************************************************************
    Label:  1
    Sentence:  আসলেই ভাল
    Afert Tokenizing:  ['আসলেই', 'ভাল']
    Truncating punctuation: ['আসলেই', 'ভাল']
    Truncating StopWords: ['আসলেই', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবার সত্যিই ভাল ছিল এবং প্রায় সব জিনিস উপভোগ করেছি
    Afert Tokenizing:  ['খাবার', 'সত্যিই', 'ভাল', 'ছিল', 'এবং', 'প্রায়', 'সব', 'জিনিস', 'উপভোগ', 'করেছি']
    Truncating punctuation: ['খাবার', 'সত্যিই', 'ভাল', 'ছিল', 'এবং', 'প্রায়', 'সব', 'জিনিস', 'উপভোগ', 'করেছি']
    Truncating StopWords: ['খাবার', 'সত্যিই', 'ভাল', 'জিনিস', 'উপভোগ', 'করেছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি ব্যক্তিগতভাবে এটি পছন্দ করেছি
    Afert Tokenizing:  ['আমি', 'ব্যক্তিগতভাবে', 'এটি', 'পছন্দ', 'করেছি']
    Truncating punctuation: ['আমি', 'ব্যক্তিগতভাবে', 'এটি', 'পছন্দ', 'করেছি']
    Truncating StopWords: ['ব্যক্তিগতভাবে', 'পছন্দ', 'করেছি']
    ***************************************************************************************
    Label:  1
    Sentence:  আমরা প্রচুর খাবার অর্ডার দিয়েছিলাম
    Afert Tokenizing:  ['আমরা', 'প্রচুর', 'খাবার', 'অর্ডার', 'দিয়েছিলাম']
    Truncating punctuation: ['আমরা', 'প্রচুর', 'খাবার', 'অর্ডার', 'দিয়েছিলাম']
    Truncating StopWords: ['প্রচুর', 'খাবার', 'অর্ডার', 'দিয়েছিলাম']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারটা জোস ছিল
    Afert Tokenizing:  ['খাবারটা', 'জোস', 'ছিল']
    Truncating punctuation: ['খাবারটা', 'জোস', 'ছিল']
    Truncating StopWords: ['খাবারটা', 'জোস']
    ***************************************************************************************
    Label:  1
    Sentence:  সত্যিই সুস্বাদু খাবার সাথে ইম্প্রেসিব সার্ভিস
    Afert Tokenizing:  ['সত্যিই', 'সুস্বাদু', 'খাবার', 'সাথে', 'ইম্প্রেসিব', 'সার্ভিস']
    Truncating punctuation: ['সত্যিই', 'সুস্বাদু', 'খাবার', 'সাথে', 'ইম্প্রেসিব', 'সার্ভিস']
    Truncating StopWords: ['সত্যিই', 'সুস্বাদু', 'খাবার', 'সাথে', 'ইম্প্রেসিব', 'সার্ভিস']
    ***************************************************************************************
    Label:  1
    Sentence:  নতুন আইটেমগুলো অনেক সুস্বাদু এবং সেরা ছিল
    Afert Tokenizing:  ['নতুন', 'আইটেমগুলো', 'অনেক', 'সুস্বাদু', 'এবং', 'সেরা', 'ছিল']
    Truncating punctuation: ['নতুন', 'আইটেমগুলো', 'অনেক', 'সুস্বাদু', 'এবং', 'সেরা', 'ছিল']
    Truncating StopWords: ['আইটেমগুলো', 'সুস্বাদু', 'সেরা']
    ***************************************************************************************
    Label:  1
    Sentence:  খুব বেশি জোস খাবার ছিল
    Afert Tokenizing:  ['খুব', 'বেশি', 'জোস', 'খাবার', 'ছিল']
    Truncating punctuation: ['খুব', 'বেশি', 'জোস', 'খাবার', 'ছিল']
    Truncating StopWords: ['বেশি', 'জোস', 'খাবার']
    ***************************************************************************************
    Label:  0
    Sentence:  কিন্তু আমি বুঝতে পারছি না কেন তারা এত বেশি দাম রাখছে
    Afert Tokenizing:  ['কিন্তু', 'আমি', 'বুঝতে', 'পারছি', 'না', 'কেন', 'তারা', 'এত', 'বেশি', 'দাম', 'রাখছে']
    Truncating punctuation: ['কিন্তু', 'আমি', 'বুঝতে', 'পারছি', 'না', 'কেন', 'তারা', 'এত', 'বেশি', 'দাম', 'রাখছে']
    Truncating StopWords: ['বুঝতে', 'পারছি', 'না', 'বেশি', 'দাম', 'রাখছে']
    ***************************************************************************************
    Label:  1
    Sentence:  অসম্ভব ভাল লেগেছে
    Afert Tokenizing:  ['অসম্ভব', 'ভাল', 'লেগেছে']
    Truncating punctuation: ['অসম্ভব', 'ভাল', 'লেগেছে']
    Truncating StopWords: ['অসম্ভব', 'ভাল', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল অভিজ্ঞতা ছিল
    Afert Tokenizing:  ['ভাল', 'অভিজ্ঞতা', 'ছিল']
    Truncating punctuation: ['ভাল', 'অভিজ্ঞতা', 'ছিল']
    Truncating StopWords: ['ভাল', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  1
    Sentence:  তাদের সেবা পছন্দনীয়
    Afert Tokenizing:  ['তাদের', 'সেবা', 'পছন্দনীয়']
    Truncating punctuation: ['তাদের', 'সেবা', 'পছন্দনীয়']
    Truncating StopWords: ['সেবা', 'পছন্দনীয়']
    ***************************************************************************************
    Label:  0
    Sentence:  দাম বেশী
    Afert Tokenizing:  ['দাম', 'বেশী']
    Truncating punctuation: ['দাম', 'বেশী']
    Truncating StopWords: ['দাম', 'বেশী']
    ***************************************************************************************
    Label:  1
    Sentence:  গ্রেট খাবার
    Afert Tokenizing:  ['গ্রেট', 'খাবার']
    Truncating punctuation: ['গ্রেট', 'খাবার']
    Truncating StopWords: ['গ্রেট', 'খাবার']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবার ভাল এবং একটু বেশি টাকা দিতে ও প্রস্তুত
    Afert Tokenizing:  ['খাবার', 'ভাল', 'এবং', 'একটু', 'বেশি', 'টাকা', 'দিতে', 'ও', 'প্রস্তুত']
    Truncating punctuation: ['খাবার', 'ভাল', 'এবং', 'একটু', 'বেশি', 'টাকা', 'দিতে', 'ও', 'প্রস্তুত']
    Truncating StopWords: ['খাবার', 'ভাল', 'একটু', 'বেশি', 'টাকা', 'প্রস্তুত']
    ***************************************************************************************
    Label:  1
    Sentence:  আমি এটা সবাইকে সুপারিশ করব
    Afert Tokenizing:  ['আমি', 'এটা', 'সবাইকে', 'সুপারিশ', 'করব']
    Truncating punctuation: ['আমি', 'এটা', 'সবাইকে', 'সুপারিশ', 'করব']
    Truncating StopWords: ['সবাইকে', 'সুপারিশ', 'করব']
    ***************************************************************************************
    Label:  0
    Sentence:  সেবা আরো ভাল হতে পারত
    Afert Tokenizing:  ['সেবা', 'আরো', 'ভাল', 'হতে', 'পারত']
    Truncating punctuation: ['সেবা', 'আরো', 'ভাল', 'হতে', 'পারত']
    Truncating StopWords: ['সেবা', 'আরো', 'ভাল', 'পারত']
    ***************************************************************************************
    Label:  1
    Sentence:  ভাল খাবার
    Afert Tokenizing:  ['ভাল', 'খাবার']
    Truncating punctuation: ['ভাল', 'খাবার']
    Truncating StopWords: ['ভাল', 'খাবার']
    ***************************************************************************************
    Label:  0
    Sentence:  কেউ টাকা খরচ করে এই বই কিনবেন না
    Afert Tokenizing:  ['কেউ', 'টাকা', 'খরচ', 'করে', 'এই', 'বই', 'কিনবেন', 'না']
    Truncating punctuation: ['কেউ', 'টাকা', 'খরচ', 'করে', 'এই', 'বই', 'কিনবেন', 'না']
    Truncating StopWords: ['টাকা', 'খরচ', 'বই', 'কিনবেন', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  কোনোভাবেই পড়া যায় না
    Afert Tokenizing:  ['কোনোভাবেই', 'পড়া', 'যায়', 'না']
    Truncating punctuation: ['কোনোভাবেই', 'পড়া', 'যায়', 'না']
    Truncating StopWords: ['কোনোভাবেই', 'পড়া', 'যায়', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  খুব বাজে এবং নিম্ন মানের একটা বই মনে হয়
    Afert Tokenizing:  ['খুব', 'বাজে', 'এবং', 'নিম্ন', 'মানের', 'একটা', 'বই', 'মনে', 'হয়']
    Truncating punctuation: ['খুব', 'বাজে', 'এবং', 'নিম্ন', 'মানের', 'একটা', 'বই', 'মনে', 'হয়']
    Truncating StopWords: ['বাজে', 'নিম্ন', 'মানের', 'একটা', 'বই']
    ***************************************************************************************
    Label:  0
    Sentence:  ভালো লাগলো না
    Afert Tokenizing:  ['ভালো', 'লাগলো', 'না']
    Truncating punctuation: ['ভালো', 'লাগলো', 'না']
    Truncating StopWords: ['ভালো', 'লাগলো', 'না']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় অসাধারণ
    Afert Tokenizing:  ['এক', 'কথায়', 'অসাধারণ']
    Truncating punctuation: ['এক', 'কথায়', 'অসাধারণ']
    Truncating StopWords: ['এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  ভালো লেগেছে
    Afert Tokenizing:  ['ভালো', 'লেগেছে']
    Truncating punctuation: ['ভালো', 'লেগেছে']
    Truncating StopWords: ['ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় অসাধারণ
    Afert Tokenizing:  ['এক', 'কথায়', 'অসাধারণ']
    Truncating punctuation: ['এক', 'কথায়', 'অসাধারণ']
    Truncating StopWords: ['এক', 'কথায়', 'অসাধারণ']
    ***************************************************************************************
    Label:  1
    Sentence:  অসাধারণ খাবার
    Afert Tokenizing:  ['অসাধারণ', 'খাবার']
    Truncating punctuation: ['অসাধারণ', 'খাবার']
    Truncating StopWords: ['অসাধারণ', 'খাবার']
    ***************************************************************************************
    Label:  1
    Sentence:  খাবারটি ভাল ছিল
    Afert Tokenizing:  ['খাবারটি', 'ভাল', 'ছিল']
    Truncating punctuation: ['খাবারটি', 'ভাল', 'ছিল']
    Truncating StopWords: ['খাবারটি', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  পরিমাণ ভাল ছিল
    Afert Tokenizing:  ['পরিমাণ', 'ভাল', 'ছিল']
    Truncating punctuation: ['পরিমাণ', 'ভাল', 'ছিল']
    Truncating StopWords: ['পরিমাণ', 'ভাল']
    ***************************************************************************************
    Label:  1
    Sentence:  কর্মীদের ব্যবহার সুন্দর
    Afert Tokenizing:  ['কর্মীদের', 'ব্যবহার', 'সুন্দর']
    Truncating punctuation: ['কর্মীদের', 'ব্যবহার', 'সুন্দর']
    Truncating StopWords: ['কর্মীদের', 'সুন্দর']
    ***************************************************************************************
    Label:  0
    Sentence:  সার্ভিস খুবই ধীরগতির
    Afert Tokenizing:  ['সার্ভিস', 'খুবই', 'ধীরগতির']
    Truncating punctuation: ['সার্ভিস', 'খুবই', 'ধীরগতির']
    Truncating StopWords: ['সার্ভিস', 'খুবই', 'ধীরগতির']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার তাদের খাবার পছন্দ হয়েছে
    Afert Tokenizing:  ['আমার', 'তাদের', 'খাবার', 'পছন্দ', 'হয়েছে']
    Truncating punctuation: ['আমার', 'তাদের', 'খাবার', 'পছন্দ', 'হয়েছে']
    Truncating StopWords: ['খাবার', 'পছন্দ', 'হয়েছে']
    ***************************************************************************************
    Label:  1
    Sentence:  সার্ভিস অনেক দ্রুত ছিল
    Afert Tokenizing:  ['সার্ভিস', 'অনেক', 'দ্রুত', 'ছিল']
    Truncating punctuation: ['সার্ভিস', 'অনেক', 'দ্রুত', 'ছিল']
    Truncating StopWords: ['সার্ভিস', 'দ্রুত']
    ***************************************************************************************
    Label:  0
    Sentence:  তাদের সার্ভিস ভাল না
    Afert Tokenizing:  ['তাদের', 'সার্ভিস', 'ভাল', 'না']
    Truncating punctuation: ['তাদের', 'সার্ভিস', 'ভাল', 'না']
    Truncating StopWords: ['সার্ভিস', 'ভাল', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  নেগেটিভ রেটিং দিতে পারলে খুশি হইতাম
    Afert Tokenizing:  ['নেগেটিভ', 'রেটিং', 'দিতে', 'পারলে', 'খুশি', 'হইতাম']
    Truncating punctuation: ['নেগেটিভ', 'রেটিং', 'দিতে', 'পারলে', 'খুশি', 'হইতাম']
    Truncating StopWords: ['নেগেটিভ', 'রেটিং', 'পারলে', 'খুশি', 'হইতাম']
    ***************************************************************************************
    Label:  0
    Sentence:  আরও এডভান্স হওয়া উচিৎ ছিল
    Afert Tokenizing:  ['আরও', 'এডভান্স', 'হওয়া', 'উচিৎ', 'ছিল']
    Truncating punctuation: ['আরও', 'এডভান্স', 'হওয়া', 'উচিৎ', 'ছিল']
    Truncating StopWords: ['এডভান্স', 'হওয়া', 'উচিৎ']
    ***************************************************************************************
    Label:  0
    Sentence:  দুঃখজনক
    Afert Tokenizing:  ['দুঃখজনক']
    Truncating punctuation: ['দুঃখজনক']
    Truncating StopWords: ['দুঃখজনক']
    ***************************************************************************************
    Label:  1
    Sentence:  আমার কাছে ভালো লেগেছে
    Afert Tokenizing:  ['আমার', 'কাছে', 'ভালো', 'লেগেছে']
    Truncating punctuation: ['আমার', 'কাছে', 'ভালো', 'লেগেছে']
    Truncating StopWords: ['ভালো', 'লেগেছে']
    ***************************************************************************************
    Label:  1
    Sentence:  এক কথায় বলা যায় অসাধারণ ।
    Afert Tokenizing:  ['এক', 'কথায়', 'বলা', 'যায়', 'অসাধারণ', '', '।']
    Truncating punctuation: ['এক', 'কথায়', 'বলা', 'যায়', 'অসাধারণ', '']
    Truncating StopWords: ['এক', 'কথায়', 'অসাধারণ', '']
    ***************************************************************************************
    Label:  0
    Sentence:  ভয়ানক অভিজ্ঞতা
    Afert Tokenizing:  ['ভয়ানক', 'অভিজ্ঞতা']
    Truncating punctuation: ['ভয়ানক', 'অভিজ্ঞতা']
    Truncating StopWords: ['ভয়ানক', 'অভিজ্ঞতা']
    ***************************************************************************************
    Label:  1
    Sentence:  খাদ্য ভাল ছিল
    Afert Tokenizing:  ['খাদ্য', 'ভাল', 'ছিল']
    Truncating punctuation: ['খাদ্য', 'ভাল', 'ছিল']
    Truncating StopWords: ['খাদ্য', 'ভাল']
    ***************************************************************************************
    Label:  0
    Sentence:  খাদ্য ছিল হতাশাজনক।
    Afert Tokenizing:  ['খাদ্য', 'ছিল', 'হতাশাজনক', '।']
    Truncating punctuation: ['খাদ্য', 'ছিল', 'হতাশাজনক']
    Truncating StopWords: ['খাদ্য', 'হতাশাজনক']
    ***************************************************************************************
    Label:  1
    Sentence:  কর্মীরা সত্যিই বন্ধুত্বপূর্ণ
    Afert Tokenizing:  ['কর্মীরা', 'সত্যিই', 'বন্ধুত্বপূর্ণ']
    Truncating punctuation: ['কর্মীরা', 'সত্যিই', 'বন্ধুত্বপূর্ণ']
    Truncating StopWords: ['কর্মীরা', 'সত্যিই', 'বন্ধুত্বপূর্ণ']
    ***************************************************************************************
    Label:  1
    Sentence:  চমৎকার খাদ্য
    Afert Tokenizing:  ['চমৎকার', 'খাদ্য']
    Truncating punctuation: ['চমৎকার', 'খাদ্য']
    Truncating StopWords: ['চমৎকার', 'খাদ্য']
    ***************************************************************************************
    Label:  0
    Sentence:  শুরু থেকে শেষ পর্যন্ত সার্ভিস ভালো ছিল না
    Afert Tokenizing:  ['শুরু', 'থেকে', 'শেষ', 'পর্যন্ত', 'সার্ভিস', 'ভালো', 'ছিল', 'না']
    Truncating punctuation: ['শুরু', 'থেকে', 'শেষ', 'পর্যন্ত', 'সার্ভিস', 'ভালো', 'ছিল', 'না']
    Truncating StopWords: ['শেষ', 'সার্ভিস', 'ভালো', 'না']
    ***************************************************************************************
    Label:  0
    Sentence:  খুবই অপেশাদার এবং অভদ্র
    Afert Tokenizing:  ['খুবই', 'অপেশাদার', 'এবং', 'অভদ্র']
    Truncating punctuation: ['খুবই', 'অপেশাদার', 'এবং', 'অভদ্র']
    Truncating StopWords: ['খুবই', 'অপেশাদার', 'অভদ্র']
    ***************************************************************************************
    Label:  0
    Sentence:  খাবারটা ভয়ংকর ছিল
    Afert Tokenizing:  ['খাবারটা', 'ভয়ংকর', 'ছিল']
    Truncating punctuation: ['খাবারটা', 'ভয়ংকর', 'ছিল']
    Truncating StopWords: ['খাবারটা', 'ভয়ংকর']
    ***************************************************************************************
    Label:  0
    Sentence:  কিছুই ফ্রেশ ছিল না
    Afert Tokenizing:  ['কিছুই', 'ফ্রেশ', 'ছিল', 'না']
    Truncating punctuation: ['কিছুই', 'ফ্রেশ', 'ছিল', 'না']
    Truncating StopWords: ['ফ্রেশ', 'না']
    ***************************************************************************************



```python
xs
```




    ['অনেকগুলা অরডার একটু দেখবেন',
     'ভালোবাসা রইল ইভ্যালির',
     'আগের প্রডাক্ট ক্লিয়ার তারাতাড়ি',
     'ভাল লাগতেছে না',
     'দয়া একটু ভাই পাবো',
     'সঠিক তারিখে দিতেন অভিযোগ দিত না',
     'কমার্সের নামে আপনারা মানুষের সাথে করতেছে একদিন হিসাব আপনাদের কড়ায় ঘন্ডায়',
     'ফাইজলামি!!',
     'দীর্ঘ হায়াত কামনা',
     'ভাই অডার মত টাকা নাই স্বপ্নের খুবই প্রয়োজনীয় বাইকটা পাব না',
     'ভাই সামান্য গ্রোসারি আইটেম পারলেন না ৩ মাসে',
     'যুবকের স্বপ্ন বেঁচে পুরণ',
     'কথা মিল রাখলে গ্রাহক বারবে না কমবে আশা দ্রুত সমাদান ধন্যবাদ',
     'ভাই পন্য গুলো পাবো,,,,,',
     'Daraz মাসেই না কিনেছি বড় ছোট অর্ডার',
     'প্রডাক্ট গুলো পাওয়ার সম্ভাবনা ভাই',
     'খবর বার্তা নাই তাড়াতাড়ি ডেলিভারি দেন..',
     'আপনারা কখনো মানুষের আস্থা অর্জন পারবেন না ১০০%!!',
     'কাষ্টমারের ভোগান্তি কমিয়ে কথা অনুযায়ী করারও পরামর্শ রইলো',
     'ওয়াদা দিবেন না যেটা রক্ষা পারবেন না ধন্যবাদ',
     'সততা ব্যবসা সফলতা আসবে ',
     'ভালো ভালো রাখবেন গ্রাহক দের',
     'বেস্ট অব লাক',
     'এগিয়ে যাক ইভেলি আগামীর পথে স্বপ্ন পূরনে',
     'বাকি অর্ডার গুলো দেরি একটু দেওয়ার অনুরোধ করতেছি',
     'অর্ডার আসতে পারছে না যথাসময়ে পণ্য ডেলিভারি পাচ্ছে না',
     'মানুষের কষ্টের টাকা ইনজয় মৃত্যু নেই',
     'সকল প্রাণীকে মৃত্যুর স্বাদ গ্রহণ',
     'ফটকাবাজী বাদ',
     'বেচে থাকুক ইভ্যালি বেচে থাক মানুষের সপ্ন',
     'ওর্ডারকৃত টিভিটি ডেলিভারির ব্যবস্থা নেওয়ার তানাহলে ভোক্তা অধিকার আইনে মামলা করব',
     'ইভ্যালির সবসময় শুভকামনা',
     'জিনিস লাগবে না টাকা যাই',
     'প্রোডাক্ট গুলো ডেলিভারি দ্রুত আশাবাদী',
     'শুভ কামনা রইল',
     'মার্চ অর্ডার করেও প্রোডাক্ট পাইনাই',
     'পাশে আছলাম পাশে আছি পাশে থাকমু',
     'প্রতারক দুরে থাকুন',
     'অরে বাটপার',
     'তোদের লজ্জা শরম নাই৷ ৷',
     'কাস্টমার কেয়ার কল রিসিভ না কেনো',
     'নাটক বন্ধ',
     'দিবেন ভাই বুড়া হয়ে গেলে???',
     'দিবেন ভাই বুড়া হয়ে গেলে???',
     'ন্যাড়া একবারই বেল তলায়',
     'না পাওয়া অর্ডার করবো না',
     'রাতে অর্ডার দিব',
     'নিব লিংক করেনা',
     'থাম রে তোরা',
     'কল রিসিভ করাও বন্ধ দিয়েছেন',
     'প্রতারককে বিশ্বাস না',
     'নিব দাম টা',
     'প্রতারণা ছাড়া না',
     'আগের টা ডেলিভারী',
     'ফ্রী দিলেও নিবে না',
     'নির্লজ্জ',
     'সালা ধান্দাবাজ',
     'ক্যাশ অন ডেলিভারিতে আপনাদের সমাস্যা কোথায় বুঝলাম না',
     'এভাবেই এগিয়ে হার মানলে না',
     'ফেসবুক পোষ্ট ডিজাইন ভালই বানিয়েছেন',
     'বিজনেস চালিয়ে যান.. ইনশাআল্লাহ বিজয় সুনিশ্চিত..',
     'লাজ লজ্জা আসে',
     'দোকানে দাম আনেক কম আপনাদের সপ',
     'আপনারা জনগনের সাথে বাটপারি করতেছেন ডেলিভারি দিচ্ছেন না',
     'ছ্যাচরামি রে ভাই',
     'মঙ্গল গ্রহের মোবাইল? জীবদ্দশায় ডেলিভারি না',
     'এখনো লোভী লোক সত্যিই অর্ডার',
     'কম দামে ভালো বাইক দিবে অর্ডার পারো',
     'চিটিং বাজ',
     'লাইক দিয়ে পাশে থাকলাম',
     'আপনাদের কাস্টমার কেয়ারে আজকে 10/12 কল দিয়েও না',
     'নিবো',
     'আচ্ছা আপনারা সুই বেচেন',
     'আপনারা ২ নাম্বার জিনিস দেন,গ্রাহকে হয়রানি',
     'ভালোবাসা অবিরাম',
     'নিব লোকেশান প্লিজ',
     'চমৎকার এগিয়ে চলুক',
     'শুভকামনা',
     'হাস্যকর ব্যাপার এখনো আপনারা অফার দিচ্ছেন',
     'ভাওতাবাজী এখনো ছাড়বা না',
     'নাহ এইবার ঠিক',
     'মামলা আমিও করবো লাখ টাকা না পাই',
     'আশা ছেড়ে দাও',
     'জীবনে কোনদিন ভাবি নাই বড় খাব',
     'একটা অর্ডার বাতিল হয়েছে',
     'চোখে না না বিশ্বাসে!',
     'একটা অর্ডার বাতিল হয়েছে',
     'চমৎকার এভাবে নিত্যনতুন অফার দিয়ে গ্রাহকদের আকৃষ্ট করুন',
     'শাড়ি টা চাই চাই',
     'ভন্ড প্রতারক',
     'ভাই ২ টা ফোন অডার দিয়ে ছিলাম ৫ মাস হয়ে খবর নাই ',
     'অনন্য শপিং এক্সপেরিয়েন্স',
     'পুরাতন অর্ডার গুলো ডেলিভারি পাবো??',
     'পেমেন্ট না',
     'ঘোরানোর ইচ্ছা',
     'ডেলিভারি চার্জ নিবে প্রোডাক্ট না দেয়ার সম্ভাবণা',
     'বাহ চমৎকার',
     'দুর্বার গতিতে এগিয়ে যাও',
     'সাথে ছিলাম আছি থাকবো',
     'ইভ্যালি চাঙা ইনশাআল্লাহ',
     'সম্পূর্ন ক্যাশ অন ডেলিভারি অর্ডার দিবো নাহলে দেয়ার ইচ্ছা নাই',
     'এগিয়ে যাও',
     'এগিয়ে যাও',
     'আজকে প্রোডাক্ট পেয়েছি,ভালো লাগছে',
     'ইনশাআল্লাহ খেলা কাল',
     'দাম লিখতে সমস্যা কোথায়',
     'দাম লিখতে সমস্যা কোথায়',
     'ভাই কিনতে আগ্রহী',
     'বেছে থাক ইভ্যালি বছর',
     'প্রোডাক্ট দিলে খুশি হইতাম',
     'যাই ম্যাসেজ রিপ্লাই দিবেন',
     'কিনতে আগ্রহী',
     'মোবাইল অফার দিলে বেশি সেল প্রফিট',
     'সাথে আছি',
     'কেমন রসিকতা',
     'বেচে থাকুক ইভালী সেবা পাক মানুষ...',
     'পাওনা বুঝিয়ে দিবেন পাশেই আছি শুভ কামনা রইলো',
     'দেশি প্রতিষ্ঠান বেচে থাকুক',
     'কিনতে চাচ্ছি পেমেন্ট যাচ্ছেনা',
     '১০০০ অডার দিব দোকানের',
     'অজও অছি কালও থাকবো বিশ্বাস অছে',
     'আজকে ১০টা বাইক অর্ডার করবো',
     'ধন্যবাদ দিব আপনাদের ভাষা পাচ্ছি না',
     'লজ্জা নাই',
     'কয়মাসে 7',
     'দয়া অর্ডার না',
     'ঘুরে দাড়াক ই-কমার্স',
     'অর্ডার পণ্য পাওয়ার নিশ্চয়তা নাই,এভাবে কয়দিন চলবে',
     'ব্যাবহার করছি মিলিয়ে দারুন একটা ফোন',
     'ডেলিভারি না!',
     'অর্ডার ইচ্ছে বার,কিন্তু কমেন্ট বক্স পড়ে ইচ্ছেটাই মাটি হয়ে যায়',
     'অর অপেক্ষা',
     'ধৈর্য্য না',
     'আড়াই মাস হয়ে এখনো পন্য পেলাম না',
     '২৪ ঘন্টার দেবে অবিশ্বাস্য',
     'আপনাদের একটা রিভিও ভাল দেখলাম না…',
     'জানুয়ারির ১৫ তারিখে অর্ডার করেছি এখনো প্রোডাক্টটি পাই নাই,ইভ্যালি টাকা মেরে দিসে',
     'আমিও প্রস্তুতি নিচ্ছি মামলা',
     'কুয়ালিটি প্রাইস ২ টাই বেস্ট',
     'তোমাদের পথচলা কন্টকমুক্ত সুন্দর',
     'দারুণ জিনিসটা',
     'আইনগত ব্যবস্থা নেয়া',
     'অপেক্ষা করবো',
     'জীবনেও evally অর্ডার করব না',
     'সুন্দর কালেকশন',
     'সুন্দর ড্রেস',
     'ওয়াও',
     'ওয়াও অসম্ভব সুন্দর',
     'দারুন',
     'সুন্দর',
     'শাড়ীর কালার টা খুবই সুন্দর',
     'সুন্দর কালেকশন',
     'বাহ চমৎকার',
     'আলহামদুলিল্লাহ',
     'দারুণ প্যাকেজিং',
     'বিউটিফুল',
     'মাশাআল্লাহ',
     'মাশাআল্লাহ',
     'ওয়াও প্রাইজ প্লিজ',
     'সুন্দর প্যাকেজ!',
     'ভালো সার্ভিস পেয়েছি ধন্যবাদ',
     'মাশাআল্লাহ এগিয়ে যাও',
     'নেট প্রাইজ আরো ১০০ টাকা বেশি চেয়েছেন',
     'আমের কোয়ালিটি ভালো ছিলো কালারো সুন্দর ছিলো',
     'দুইবার আম পাঠাইছে একবারও ভালো পরলো না',
     'এসব বাটপারদের নামে মামলা',
     'এসব বাটপারদের নামে মামলা',
     'অর্ডার কনফার্ম হওয়ার প্রোডাক্ট হাতে পাওয়া পুরোটাই ভালো একটা অভিজ্ঞতা',
     'প্রাইস হিসেবে কোয়ালিটি স্যাটিসফাইড',
     'শপিং এক্সপেরিয়েন্স পুরোপুরি সন্তোষজনক',
     'এতো কম দামে ল্যাপটপ টেবিল পাবার কথা না',
     'ধন্যবাদ আলিশা মাঠ অনেকদিন অফার',
     'আলহামদুলিল্লাহ  দিনে দিনে পাচ্ছি',
     'আলহামদুলিল্লাহ আলেশা মার্ট বাইক ভেলিভারি পাইলাম',
     'সাবাস আলেশা মাট',
     'অসংখ্য ধন্যবাদ পণ্য পাওয়ার',
     'অর্ডার টা করেছি এখনো প্রসেসিং অবস্থায় একটু তাড়াতাড়ি দিলে উপকার হতো',
     'কখনো আলেশা মার্ট কিনে প্রোতারিতো হই নাই ধন্যবাদ জানাই',
     'বাইক কই ৩ মাস হয়ে গেলো খবর নাই',
     'আরো ভালো ফোন চেষ্টা করেন।সাধু বাদ',
     'পন্য পেয়েছি উপহার হিসেবে টি-শার্ট পাইনি এইটা কথা',
     "বাংলাদেশের একমাত্র বেস্ট ই-কমার্স সাইট 'আলেশা মার্ট",
     'দেশী পন্য কিনে ধন্য',
     'বেশি খাইতে জায়েন না',
     'বছর পাব না',
     'বাটপারের দল বাইকের লিস্ট দে',
     'বাংলাদেশ চেয়ে বাজে আন লাইন একটিও নাই',
     'ভালো লাগলো',
     'ভালোই বাটপার',
     'ভাই সাইকেল এখান',
     'আপনাদের বিশ্বাস যায়',
     'দাম বেশি',
     'আপনা\u200cদের কার্যক্রম আ\u200cরো এ\u200cগি\u200cয়ে যাক প্রত\u200c্যাশা ক\u200cরি',
     'আপনাদের ডেলিভারি চার্জ একটু কমানো দরকার',
     'আলিশা মার্ট দেশ সেরা ইনশাআল্লাহ একদিন',
     'আসলে প্রোডাক্ট দেয়??',
     'পুরাই চোর সালারা',
     'ধন্যবাদ আলেশা_মার্ট গাড়ী কনফার্ম',
     'ফাঁদে পা বাড়াবেন না',
     'পেয়েছেন',
     'খুবি খারাপ অবস্থা',
     'আশা ঠিক সময়ের পাবো ',
     'স্বপ্ন পুরনের সারথি',
     'টিভিতে বিজ্ঞাপন দিয়ে টাকা নষ্ট না অফার',
     'দাম রাখছেন আপনারা ১ যুগেও উন্নতি পারবেন না!!',
     'আপনাদের অ্যাপস আরো উন্নত',
     'ইভ্যালির চাচাতো ভাই না',
     'ধান্ধাবাজির বুঝলাম না',
     'আপনাদের খারাপ দিক চাইতে না',
     'নিবো আগেও সিরাম ইউজ করেছি',
     'এবারের প্যাকেজিং টা বাজে ছিলো',
     'আলহামদুলিল্লাহ প্রডাক্ট ভালো ছিলো',
     'আপনাদের ভাল।আমি আপনাদে পন্য বলেছি সবাই প্রশংসা ছে আপনাদের',
     '৩দিন যাবত করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম',
     '4 টা নিয়েছি।খুবই ভালো প্রোডাক্ট',
     'চাই',
     'ভালো প্রোডাক্ট',
     'প্রোডাক্ট ভালো দাম বেশি',
     'ভালো কোয়ালিটি',
     'মাল হাতে পাইছি ভালো',
     'খুবই ভাল কোয়ালিটি  পড়তেও আলাদা মজা পাওয়া যাইয়া',
     'দাম গুলো বেশী',
     'যেমনটা চেয়েছিলাম তেমনটা পেয়েছি',
     'আপনাদের প্রোডাক্ট ভাল',
     'আপনাদের মাক্স গুলো যাবত করছি,খুবই ভাল কোয়ালিটি,পড়তেও আরাম',
     '১০% এন্ড ৫% দুইটাই পাইছি আলহামদুলিল্লাহ',
     'ভাই আমিও বিকাশে পেমেন্ট করছিলাম ক্যাশব্যাক পাইনি',
     'দাম জানার মনটা খারাপ হয়ে গেলো',
     'Quality অনুযায়ী দাম বেশী',
     'কোয়ালিটি দেখান না',
     'প্রতিটি প্রোডাক্টের গুনগত মান খুবিই ভালো',
     'পণ্যগুলো সুন্দর আরাম দায়ক',
     'যথেষ্ট প্রিমিয়াম প্রোডাক্ট আজকে ছেলের টি-শার্ট নিয়েছি চমৎকার ফেব্রিক্স',
     'সবই মন চায়.',
     'দাম একটু বেশি হইলেও কাপড়ের মান সৃজনশীলতা আপনাদের পন্যের বড় গুণ',
     'আপনাদের পন্যের গুণমান চমৎকার সত্যি প্রেমে গেছি',
     'প্রাইজ টা কিঞ্চিত হাই',
     'দাম টা একটু কম যায় না',
     'আপনারা উওর না  ইনবক্স ক\u200cরে\u200cছি  রেসপন্স নাই',
     'এক কথায় অসাধারণ',
     'জিনিস আপনাদের খুবই ভালো প্রাইস কমেন্টে স্ট্যাটাসে লিখে নতুবা ইনবক্সে চেক লাগত',
     'দাম কমালে আরেকটি অর্ডার করতাম',
     'আপনাদের প্রোডাক্ট গুলো অসাধারণ',
     'আপনাদে পন\u200d্য ভালো লাগছে 100% কোয়ালিটি সপ্মন\u200d্য সাতটা শার্ট আনলাম চোখবুজে বিশ্বাস যায় আশা আপনারা বিশ্বাস টুকু দরে রাখবেন',
     '৩ পিস নিলাম সুন্দর ধন্যবাদ',
     'একটা নিছি খুবই ভালো মানের প্রোডাক্ট ইনশাল্লাহ একটা নেব',
     'সুন্দর ভালো লাগছে',
     'আসলেই প্রোডাক্ট গুলা ভালো কোয়ালিটির',
     'দাম কমান',
     'পেলাম নাতো',
     'গতকাল সবুজ টি-শার্টটা অর্ডার দিয়ে আজকেই পেয়ে গেলাম',
     'অনলাইনে কেনাকাটা বড় বোকামি।পন্যটা মনের না',
     'এক টা নিয়েছি ভালো পছন্দ হয়েছে  দাম টা একটু কম মেরুন কালার টা নিতাম ',
     'চাইলাম ৩২ আপনারা দিয়ে ৩৪ কিভাবে ব্যাবহার করবো',
     'চট্টগ্রামে আজকে হাতে পেলাম ভাল যেই চেয়েছি পেয়েছি আলহামদুলিল্লাহ কথা মিল',
     'বেস্ট কোয়ালিটি',
     'জোস কালেকশন',
     'এক্সক্লুসিভ কালেকশন',
     'কোয়ালিটিফুল কালেকশন অলওয়েজ',
     'অনলাইনে ভয়টা কেটে ধন্যবাদ ',
     'ঈদের বাটার ভাউচার অর্ডার দিয়েছিলাম ৪৮ ঘন্টার ডেলিভারি দেয়ার কথা এখনো পেলাম না',
     'বাংলাদেশে ই-কমার্স সাইটের প্রসার চাই',
     'অনলাইন কেনাকাটায় দেবে একমাত্র অনলাইন নির্ভরশীল প্রতিষ্ঠান মোনার্ক মার্ঠ',
     'মিনিমাম ১০% ডিসকাউন্ট দেয়া',
     'এক টাকাও ডিস্কাউন্ট না',
     'ভালো সময়োপযোগী পদক্ষেপ',
     'মোনার্ক মার্ঠ অনলাইন নির্ভরশীল প্রতিষ্ঠান এটার অর্ডার যায়',
     'ভালো পন্য',
     'দারুণ অফার',
     '9 তারিখের অর্ডার করলাম এখনো পাচ্ছিনা',
     'অফার শিহরিত',
     'আপনাদের পন\u200d্য একটু দম কমালে ভালো',
     'গুড',
     'এভাবে ঠকানোর মানে আশাকরি রিপ্লে দিবেন অর্ডার ক্যন্সেল চাই',
     'এইসব মানে ভাই শর্তাবলী অর্ডার করলাম ক্যাসব্যাক পেলাম না তোরা শুধু প্রডাক্ট ভেলিভারি আয় বাইন্ধা রাখমু',
     'অফারের নামে এসব হয়রানি বন্ধ অফার দিয়েছেন সঠিক তথ্য না',
     'শুধু ধোঁকা বাজি',
     'নগদে পেমেন্ট লাভ ডিসকাউন্ট নাই',
     'ধান্দা ফ্রী অর্ডার নেওয়ার মানুষের টাকা আটকে রাখবে',
     'অনলাইনে অর্ডার চায়না একটাই ভয়',
     'আপনারা কম দামের প্রোডাক্ট দিয়ে বেশী দাম লিখলে জনগন খাবে না',
     'অভার প্রাইজড',
     'দাম অনুযায়ী ভালো না সাইজও ঠিক দেয়নি',
     'আপনাদের প্রোডাক্ট পেয়েছি চেয়েছিলাম ঠিক তেমনি পেয়েছি,ধন্যবাদ আপনাদের মনের প্রোডাক্ট দিবার',
     'প্রাইজ বেশি ভাই',
     'বাহ্',
     'অসাধারণ সুন্দর প্রোডাক্টগুলা. এক্কেবারে পারফেক্ট কাস্টমাইজড',
     '1st অর্ডার সন্তুষ্ট ধন্যবাদ',
     'আপনারাও',
     'আজকেই ডেলিভারি পেলাম আলহামদুলিল্লাহ কোয়ালিটি সবকিছুই ঠিকঠাক সবচেয়ে ভালো লেগেছে যেই বিষয়টা পেইজের কথা মিল যেটা পাওয়া খুবই দুষ্কর',
     'ধন্যবাদ mr fiction',
     'কোয়ালিটি ভালো প্রিন্ট চোখে পড়ার সন্তুষ্ট',
     'কিছুদিন দেখলাম আল্লাহামদুলিল্লাহ কভারের মান অত্যান্ত ভাল',
     'আল্লাহতালার রহমতে ভাল প্রিমিয়াম কলেটির কভার পেয়েছি।ধন্যবাদ',
     'মাশআল্লাহ আল্লাহামদুলিল্লাহ',
     'দাম কাভার গুলো ঠিক আসে',
     'দিনের ভিতর হাতে পেয়েছি ৷ Mr Fiction অসংখ্য ধন্যবাদ',
     'দামে কম মানে ভাল.',
     'অন্যরা না ভেবে চট অর্ডার দিয়ে দেন৷ আপনিও নিরাশ না গ্যারান্টি দিলাম কিছুর ধন্যবাদ',
     'দামি খাতায় লিখবো',
     'দাম কেও রাখে',
     'জিনিস সুন্দর দাম বেশি',
     'ব্যাবসা একটা লিমিট',
     'ভাই বিজনেস করতেছেন ভালো কথা আপনাদের ক্রেতা + কমেন্ট করতেছে মতামত গুলোকে প্রধান্য',
     'মূল্যটা দিয়েছেন বেশি',
     'আজকে প্রডাক্টা পেলাম সুন্দর হইছে ধন্যবাদ আপনাদের',
     'প্রথমটা পছন্দ হইসে বাট দামটা বেশি,,কিছু কমানো যায়না',
     'অনলাইন প্রোডাক্ট হিসেবে ভালো সুন্দর ফিটিং রেটিং দশে দশ চাইলে প্রোডাক্ট মানে ভালো',
     'দুইটা প্যান্ট অর্ডার করেছি দুইটা প্যান্টই যথেষ্ট ভালো হয়েছে,মাপ একুরেট হয়েছে ধন্যবাদ আপনাদেরকে',
     'কম মূল্যে এতো কোয়ালিটি সম্পন্ন ভালো প্রোডাক্ট সত্যিই প্রত্যাশার অধিক সেইসাথে দ্রুতগতির ডেলিভারি,প্যাকেজিং স্টাফদের ব্যাবহারে সত্যিই মুগ্ধ',
     'আপনাদের ফ্যান হয়ে গেলাম',
     'আলহামদুলিল্লাহ ভালো জিনিস হাতে পেয়েছি ধন্যবাদ..',
     'সত্যি দাম সাথে তুলনা অসাধারণ মানের শুভ কামনা রইলো',
     'আজেই আপনাদের পণ্য হাতে পেয়েছি পণ্য নের কোয়ালিটি খুবই ভালো আরামদায়ক',
     'রিভিউ ৫ ভিতর ৫ হয়তো চোখ বন্ধ অর্ডার করবো',
     'গতকাল অর্ডার করেছিলাম আজকেই হাতে পেয়েছি গুণগত মানের দিক খুবই ভালো ধন্যবাদ জানাচ্ছি শুভকামনা রইলো',
     'আশা ভবিষ্যতে পন্যের মান ঠিক আরো বহুদূর এগিয়ে যাবেন।শুভ কামনা',
     'ধন্যবাদ পণ্যগুলো দ্রুত হাতে পেয়েছি',
     'ধন্যবাদ Deen এতো সুন্দর পণ্য পৌছে দেওয়ার এগিয়ে যাক সামনের কামনা',
     'আলহামদুলিল্লাহ কিছুক্ষন অর্ডারকৃত পণ্যটি হাতে পেলাম',
     'আমিও নিলাম খুবই ভালো',
     'প্যাকেটিং ভালো দ্রুত পণ্য ডেলিভারি দেয়া হয়েছে',
     'ধন্যবাদ আপনাদের দ্রুত ডেলিভারী জন্যে',
     'Deen ভালো প্রোডাক্ট অনলাইন পাবো আশা করিনি',
     'ভালো',
     'কোয়ালিটি সার্ভিস খুব-ই ভালো আরো কালেকশনের অপেক্ষায় থাকলাম শুভকামনা',
     'দাম টাও ঠিকাছে হয়েছে প্যাকেজিং সুন্দর',
     'প্যাকিং,প্যান্ট কোয়ালিটি দ্রুত ডেলিভারী দেওয়ার ধন্যবাদ',
     'তারাতাড়ি পেয়েছি একদম বাড়িতে দিয়ে ভালো মিলিয়ে',
     'আলহামদুলিল্লাহ কাপড়ের মান ভালো.. টা দিয়ে তিনবার নিলাম ডিন থেকে... মাশাআল্লাহ ডেলিভারিও পেয়েছি খুবই তাড়াতাড়ি..',
     'আগামীতে আরো কেনাকাটা আপনাদের ইনশাআল্লাহ',
     'অসংখ্য ধন্যবাদ পণ্য পাওয়ার',
     'বর্তমান নাম্বরি ই-কমার্স মধ্য ভালো ট্রাস্টেট একটা ই-কমার্স খুঁজে দুষ্কর',
     'প্রোডাক্ট গুলো এইমাত্র হাতে পেলাম প্যাকেজিং টা ছিলো অসাধারণ',
     'আশা করবো ভবিষ্যতে আপনাদের কোয়ালিটি অক্ষুণ্ণ',
     'দোয়া আপনাদের জন্য.প্যান্ট মনের মত হয়েছে কাপরের মান ভালো',
     'ওনাদের সিস্টেম ভালো আমিও পন্য চেইঞ্জ করেছিলাম আরো অর্ডার দিবো কাজের লোকদের',
     'পণ্যের গুণগত মান খুবই ভালো অল্প টাকায় ভালো জিনিস পাবো ভাবতেও পারিনি ☺ চোখ বন্ধ ১০ ১০ মার্ক দেয়ায় যায়',
     'এইগুলা চোর বাটপার',
     'একদম ফালতু 500টাকার প্যান্ট বেচে 1200টাকায় ইসলামী নাম বিভ্রান্ত হয়েন্না প্লীজ',
     'সমস্যা লো কোয়ালিটি ম্যাটারিয়াল ইউস আপনারা',
     'মারকেটিং ভালো আপনাদের বাট কাপরের ভ্যারাইটি নাই',
     'আপনাদের কেনাকাটা এক বারো ঠকিনি ধন্যবাদ আপনাদের',
     'আপনাদের ডেলিভারি মাধ্যমটা ভালো নয়',
     'কোয়ালিটি পছন্দ হয়েছে আপনাদের শুভকামনা রইল',
     'ধন্যবাদ যেরকম চেয়েছি ঠিক রকমই পেয়েছি',
     'সবচেয়ে ভালো দিক ডেলিভারি দ্রুত প্যাকেজিং খুবই সুন্দর আল্লাহ আপনাদের মঙ্গল করুক',
     'কোয়ালিটির ধন্যবাদ',
     'পারফেক্ট',
     'ভালো মানের প্রোডাক্ট দেওয়ার ধন্যবাদ আপনাদেরকে,অনেক দূর এগিয়ে যাক',
     'প্রমান পেয়েছি আপনাদের কথায় মিল আছে।আপনাদের মঙ্গল কামনা সাথেই আছি,ধন্যবাদ',
     'সবাই সুন্দর সুন্দর কথা টাকার লোভ সামলাতে না',
     'সত্যি রিস্ক নাই আপনাদের সার্ভিস বেস্ট',
     'দাম টা খুবই বেশি',
     'প্রোডাক্ট আনেন',
     'কেনো যেনো অনলাইন প্রডাক্টে আস্থা না।কিন্তু ২য় Deen টি-শার্ট নিলাম আলহামদুলিল্লাহ নিরাশ হইনি',
     'আলহামদুলিল্লাহ কোয়ালিটি ভালো',
     'আপনাদের প্রডাক্ট ভালো মানসম্মত।ধারাবাহিক আপনারা মানসম্মত প্রডাক্ট ডেলিভারি দিলে অচিরেই আপনাদের সাফল্য পেয়ে যাবেন',
     'আলহামদুলিল্লাহ ২৪ ঘন্টারও কম সময়ের ডেলিভারি পেলাম কোয়ালিটি সন্তুষ্ট আলহামদুলিল্লাহ',
     'ভালো লাগছে পন্য হাতে পেয়ে।।',
     'অালহামদু-লিল্লাহ,,, সবগুলো প্রডাক্ট ঠিকঠাক পেয়েছি অামি সন্তুষ্ট....!',
     'আজকে সারাদিন কল করে,মেসেঞ্জারে নক করেও আপনাদের সাড়া পেলাম না,বিষয়টি দুঃখজনক',
     'পরবর্তীতে আবারো অর্ডার ইচ্ছা আছে।ধন্যবাদ',
     'দ্বিতীয়বারের মত কেনাকাটা প্যাকিং ভাল দ্রুততম সময়ে ডেলিভারি প্রোডাক্ট কোয়ালিটি ভাল পেয়ে বরাবরের সন্তুষ্ট',
     'ভাই আপনাদের সাইটে ডুকাই না প্রচুর লোডিং নিচ্ছে',
     'বিশ্বাসের মাত্রাটা বেড়ে',
     'আলহামদুলিল্লাহ ডেলিভারি ফাস্ট পেয়েছি প্রোডাক্ট ভালো',
     'মাস্ক গুলো একটা ৫০ টাকা আপনারা টাকা বেশি নিচ্ছেন না আপনাদের এইটা ধরনের প্রতারণা',
     'সবগুলা ওয়েবসাইট প্রডাক্ট সাপ্লাই দিত কাষ্টমার অনলাইনে ওর্ডার ঠকি ভ্রান্ত ধারনা দূর',
     'প্রোডাক্ট পরিমান বাড়াতে আইটেম একেবারেই কম',
     'আপনাদের ওয়েবসাইট টা একটু ইম্প্রুভ প্র-চ-ন্ড স্লো',
     'অর্ডার একটু ভয়ে ছিলাম প্রেডাক্ট হাতে পেলাম ভয় কেটে',
     'দাম একটু কম উচিৎ',
     'আলহামদুলিল্লাহ প্রোডাক্ট হাতে পেয়েছি ধন্যবাদ আপনাদেরকে কথা আপনাদের মিল প্রোডাক্টের বলব এক কথায় অসাধারণ',
     'এক কথায় প্রাইজ রেঞ্জের ভালো প্রডাক্ট আপনাদের উজ্জ্বল ভবিষ্যৎ কামনা',
     'দ্রুত ডেলিভারি পেয়েছি।ধন্যবাদ',
     'আলহামদুলিল্লাহ প্রোডাক্ট ভাল ছিলাম ঠিক তেমনই পেয়েছি',
     'ভাই আপনাদের পন্য ভালো আপনারা প্রতারক নন',
     'আপনাদে একটা অর্ডার দিছিলাম গত মাসে এখনো খবর নাই,আমি প্রোডাক্ট টা পেতে একটু জানাবেন?',
     'জুনের ১৬ তারিখে অর্ডার দিয়েছি আপনাদের খবর নাই এস\u200cএম\u200cএস রিপ্লাই না',
     'রিজেনবল প্রাইসে অরিজিনাল লেদার সত্যি আরামদায়ক আপনাদের প্রডাক্ট',
     'আপনাদের লোকেশন অনুযায়ী প্রতারিত হয়েছি',
     'সত্যি সুন্দর প্রোডাক্ট',
     'আপনাদের প্রোডাক্ট গুলো অসাধারণ',
     'সুন্দর কালেকশন',
     'প্রতিটি পন্য অসম্ভব সুন্দর',
     'দাম বেড়ে',
     'ইদে জুতা নিলাম জুতার হিল নষ্ট কেন।',
     'গিয়েছিলাম পাই নাই ভালো',
     'নিলাম ভালো মানের জুতা',
     'সত্যি সুন্দর বেশি খুশি লেগেছে কালার একদম ছবির মতই',
     'আসসালামু আলাইকুম আপু জামা অডার করেছি',
     'তোমারদে জিনিস গুলো ভালো',
     'জামা পেয়েছি আপু সুন্দর হয়েছে',
     'দামটা বেশি হয়ে যায়',
     'হাস্যকর দাম',
     'এতো দাম',
     'দাম কারন',
     'সুন্দর',
     'অসম্ভব সুন্দর আরামদায়ক',
     'সুন্দর সুন্দর প্যান্ট',
     'একটা নিয়েছি ভাল প্যান্ট আরামদায়ক',
     'অসংখ্য ধন্যবাদ একদিনের গেলাম',
     'আপনাদের পাঞ্জাবির কাপড় ভালো মানের',
     'আজকেই প্রডাক্ট টা হাতে পেলাম অসম্ভব সুন্দর',
     'অনলাইন প্রোডাক্ট হিসেবে ভালো',
     'দারুণ কালেকশন',
     'দামটা একটু বেশি',
     'অসাধারণ কম্ফোর্ট পেয়েছি খুবই আরামদায়ক',
     'দাম টা ত চড়া',
     'ভালো আরামদায়ক',
     'অর্ডার করেছি ফোন নি  অর্ডার কনফার্ম না বুঝতে পারছি না',
     'প্রডাক্ট হাতে পেলাম খুবই ভালো মানের ছিলো',
     'সুন্দর নিছি',
     'চাইলে প্রোডাক্ট মানে ভালো',
     'প্রাইজটা বেশি',
     'নিয়েছি ভালো প্রোডাক্ট',
     'টেক্সট দিয়ে রাখছি রিপ্লাই নাই',
     'দাম বেশি চাচ্ছেন',
     'অনলাইন নিব না দোকানে নিব',
     'অর্ডার প্রডাক্টস হাতে পেয়েছি কালার মান দুটোই ভাল সত্যিই ভাল লেগেছে',
     'পন্য ক্রয় দেখি সার্টের গুনগত মান খারাপ',
     'আপনাদের শোরুমের ঠিকানা সরাসরি কিনতে চাচ্ছি',
     'দামে কম মানে ভালো প্রোডাক্ট',
     'অসাধারণ সুন্দর প্রোডাক্টগুলা. এক্কেবারে পারফেক্ট কাস্টমাইজড',
     'আলহামদুলিল্লাহ আলহামদুলিল্লাহ আলহামদুলিল্লাহ',
     'দারাজের মত হবেনাতো অর্ডার দিলাম একটা দারাজ দিছে আরেকটা নাতো',
     'কাস্টমার সার্ভিস আছেন প্রফেশনাল হয়েছে',
     'আপনাদের সার্ভিস ভালো',
     'ডেলিভারি আলহামদুলিল্লাহ সুন্দর হয়েছে',
     'পেন্ডিং অর্ডারগুলি পাইলে বাঁচি..',
     'গত তিন বছর নিচ্ছি আলহামদুলিল্লাহ সার্ভিসে সন্তুষ্ট',
     'নির্ভেজাল নির্ভরযোগ্য সেবা পেয়ে যাচ্ছি',
     'উনাদের ডেলিভারিও ফাস্ট',
     'বাংলাদেশে ই-কমার্স মানেই প্রতারণা ',
     'চিটার বাটপার দের খোঁজ খবর নাই',
     'বাটপারটা জেলেই থাক ',
     'ভুক্তভুগীরা প্রোডাক্ট পাবো টাকাটাও ফেরত পাবো না',
     'ওয়েব সাইট এখনো ঢুকা যায় না কেন?',
     'বেশি দাম হয়ে যায়',
     'আপনাদের দোয়া রইল ভাই',
     'অর্ডার নিশ্চিন্তে',
     'খুবই মান সম্পন্ন আরামদায়ক জার্সি দেওয়ার ধন্যবাদ',
     'ধন্যবাদ ভাল মানের কাপড়',
     'ধন্যবাদ কম দামে সুন্দর একটা জার্সি উপহার দেওয়ার',
     'কাপড় কোয়ালিটি ১০০তে ১০০% ভালো',
     'টাকা ফেরত চাই',
     'আশাকরি জিনিশ গুলা পাবো',
     'পন্য গুলো পাবো ইনশাআল্লাহ',
     'মূল টাকাটা জাস্ট লাভ দরকার নেই',
     'মূল টাকাটা জাস্ট লাভ দরকার নেই',
     'পেজের পন্য গুলো ভালো আপনারা সবাই দুইটি নিয়েছি সেম টু সেম  ধন্যবাদ',
     'প্রাইজ বেশি',
     'খুবি ভালো প্রোডাক্টটি ধন্যবাদ',
     'আপনাদের প্রত্যেকটা প্রোডাক্ট খুবই মানসম্মত',
     'ধন্যবাদ আপনাদের।।। কথা রাখার জন্য।।।।',
     'মোটামুটি চয়েছ হয়েছে দামটা বেশি',
     'দাম বেশি',
     'আলহামদুলিল্লাহ দ্রুতই প্রোডাক্ট হাতে পেয়েছি কাপড়ের মান ভালো',
     'জিনিস পাব',
     'পচ্ছন্দ ডেলিভারি চার্জ নিরুৎসাহিত',
     'একটা চাই',
     'আপনাদের প্রোডাক্ট নিয়েছিলাম কয়মাস আগে,পুরো ফালতু',
     'ফাজিল অর্ডার মাল পাঠায় না',
     'দাম টা একটু বেশি চাইতেছে',
     'আলহামদুলিল্লাহ কাপড় টা ভালোই লাগলো',
     'আপনাদের প্রোডাক্ট পেয়ে সন্তুষ্ট আরো নিবো শীঘ্রই ইনশাআল্লাহ কোয়ালিটি রাখবেন আশা',
     'দাম অনুযায়ী প্রোডাক্ট মানসম্মত ক্রয়',
     'আলহামদুলিল্লাহ পেয়েছি পছন্দের জিনিস গুলো মিলিয়ে কম দামে অসাধারণ প্রোডাক্ট',
     'ভালো লাগছে থ্যাঙ্ক ইউ',
     'মিলিয়ে আনপ্রফেশনাল পোস্ট',
     '১ দিনে প্রোডাক্ট পেয়েছি কোয়ালিটি অসাধারণ সবাই কিনতে',
     'দাম অনুযায়ী মানসম্মত শার্ট',
     'পন্যের মান ভালো',
     'ভালো কালেকশন',
     'প্রাইজ বলবেন সরাসরি কমেন্ট বলবেন ইনবক্সে ডাকার',
     'অন্যান্য পেইজের তুলনায় দাম অনেকটাই কম সার্ভিস অনেকটাই ভালো অনেকটাই ভালো ',
     'সুন্দর প্রোডাক্ট,,দাম অন্যান্য পেজ কম, ফিটিং সুন্দর হইছে,,স্যাটিসফাইড',
     'সুন্দর',
     '২ রিভিউ দিলাম আলহামদুলিল্লাহ ভালো প্রোডাক্ট',
     'আলহামদুলিল্লাহ প্রডাক্ট কোয়ালিটি সার্ভিস খুবই ভালো',
     'প্রোডাক্ট কোয়ালিটি অসাধারন।ডেলিভারি সিষ্টেম ভালো।ব্যবহার যথেষ্ট ভালো।এককথায় খুবই ভালো লেগেছে',
     'দাম',
     'বাংলাদেশের অনলাইন শপ মানে গলাকাটা দাম',
     'অর্ডার প্রাপ্ত প্রোডাক্ট ছবির সাথে মিল না',
     'বাংলাদেশের অনলাইন পন্যের মান কেনে জীবনে কেনার ইচ্ছেও প্রকাশ না।পন্যের মান খুবই খারাপ',
     'বাংলাদেশে অনলাইন শপিং সবাই এইরকম ঠকেছেন',
     'চিটার বাটপার অনলাইনে  ভালো খুজে পাওয়া দুষ্কর',
     'পণ্যের মান ভালো লাগবে অনলাইন কেনাকাটায় দেখায় মুরগী  খাওয়ায় ডাইল',
     'বারই ক্রয় করেছি ততবারই',
     'অনলাইনে ঠকার আশংকা',
     'মাক্সিমাম মানুষ দেখবেন যাচাই বাছাই না প্রডাক্ট কিনে এরপর দোষ দেয় অনলাইনের',
     'প্রধান উদ্দেশ্যই কাষ্টমার ঠকানো',
     'কিনলেই ঠকতে',
     'লোভে অর্ধেক দামে না কিনে সঠিক দামে প্রোডাক্ট কিনুন অবশ্যই ভালো সেবা পাবেন',
     'কিভাবে কেনাকাটা করবো যতবার করেছি ততবারই ঠকেছি',
     'সঠিক পণ্যের মান নিশ্চয়তা নাই কিভাবে কিনবে',
     'দাম বেশি দিয়ে অনলাইন কিনবো',
     'সকল পঁচা নিম্নমানের বেশী দামে অনলাইনে দ্রব্য বিক্রি যতবার কিনেছি ততবার খাইছি',
     'বাংলাদেশে এখনো অধিকাংশ মানুষ স্ক্যামের স্বীকার',
     'মানুষ অনলাইনে কেনাকাটা উৎসাহ ',
     'ভালো মানের অনলাইন শপ নাই',
     'বাংলাদেশে অনলাইনে কেনাকাটা 90% মানুষ ঠকে',
     'সততার অভাব',
     'দুঃখজনক সত্যি দেশে ভালো মানের অনলাইন শপ নাই.. শুধুই গ্রাহকদের ঠকিয়ে কষ্ট দেয়',
     'বাংলাদেশের ই-কমার্স মানে কিনলেন ত ঠকলেন',
     'অনলাইনের সার্ভিস খারাপ',
     'ভালো প্রোডাক্ট',
     'সবকিছুর দাম বেড়ে',
     'দারুন অফার',
     'চমৎকার সবগুলো',
     'সুন্দর কালেকশন',
     'সুন্দর',
     'দোয়া শুভকামনা রইল',
     'টাকা ১ বছরেও রিফান্ড দিল না',
     'এধরনের ঠকবাজি কর্মকান্ডের জন্যে বাংলাদেশে ইকমার্সে ঠিকমত গ্রো পারছে না ক্রেতা বিক্রেতাকে বিশ্বাস পারছে না',
     'ম্যাক্সিমাম মানহীন পন্য বাংলাদেশের অনলাইনে বিক্রি',
     'অন্ততপক্ষে বাংলাদেশ অনলাইন কেনাকাটা না ভালো দেখায় একটা  ডেলিভারি দেয় অন্যটা..',
     'খুবই ভালো ৪ টা ফোন ডেলিভারি দিছে সময়',
     'ভাই আমিও ভুক্তভোগী রিফান্ড দিচ্ছে না',
     'বাংলাদেশে একমাত্র পিকাবু-ই ট্রাস্টেড',
     'শতভাগ ট্রাস্টেড ৪ টাকার কাছাকাছি রিলেটিভ পন্য ক্রয় ডেলিভারি ভালো',
     'এক দুইদিন দেরি ভাল প্রোডাক্টই দেয়',
     'পিকাবো গুলা ডিভাইস কেনা হইছে খারাপ পাই নাই সার্ভিস ভালো',
     'ফ্যন খায়ছি না',
     'ডেলিভারি পেতে কতদিন লাগবে যায়না।১০-১২ দিনেও পেতে পারেন,২-৩ মাসও লাগতে পারে,আবার নাও পেতে',
     'আকর্ষণীয় ছবি দেখিয়ে নকল / নিন্ম মানের পণ্য সরবরাহ',
     'সেলার পণ্য আপলোডের সময় ওজন ভুল দিয়েছেন(গ্রাম ভেবে কেজিতে দিয়েছেন)',
     'ওরে চিটার',
     'বাটপারি করেও অনায়াসে ব্যবসা করতেছে',
     'ভোক্তা অধিকার মামলা',
     'সমস্যার সম্মুখীন হয়েছি বেশি হয়রানি হয়েছি',
     'টাকা মার যাওয়ার সম্ভাবনা কিছুটা',
     'অথবাতে অর্ডার ভুক্তভুগি ৬০ ডিম অর্ডার করেছিলাম ডিমগুলো মারাত্মক তিতা স্বাদের',
     'ভাউচার দিয়ে অর্ডার করলাম,জিনিসও পেয়ে গেছি',
     'মতে ওনাদের জুতা ভালো না 3 টা টি-শার্ট নিয়েছিলাম থার্ডক্লাস কাপড়',
     'জুতা লেদার ঠিকই নিম্নমানের সস্তা লেদার۔ জায়েজ জুতা ভালোনা দাম হিসেবে',
     'সুন্দর রিভিউ কম দামে ভালোই হয়েছে ফোনটি',
     'ভালো হয়েছে দামে বাজার ৩০০+ টাকা কম পেয়েছেন',
     'E-Courier দ্রুতই ডেলিভারি পেলাম  অর্ডার ডেলিভারি সকল সার্ভিস ভালো লেগেছে',
     'দারাজে কমে পাওয়া যায় ডেলিভারি ফাস্ট',
     'ট্রাই করেছি অনেকগুলো সার্ভিস ভালো না কারোই',
     'ইভ্যালির আস্থা ছিলো',
     'অবশ্যই পিকাবো দ্রুত ট্রাস্টেড সার্ভিস',
     'নিরাপত্তা রিটার্ন রিফান্ডের মত বিষয়গুলো বিবেচনা দারাজই সেরা',
     'পিকাবু স্মার্ট সার্ভিস না ই-কমার্স ইলেকট্রনিক গ্যাজেটস পিকাবু বেস্ট',
     'সার্ভিস আসলেই ভালো',
     'ফ্রড প্রতিনিয়তই',
     'ধান্দাবাজ পার্টি',
     'বিরুদ্ধে একশন দরকার',
     'ভোক্তায় একটা অভিযোগ মেরে শিক্ষা হওয়া',
     'ব্র্যান্ড ছাড়া কাপড় অনলাইন কিনা উচিৎ না  অনলাইনে ভুক্তভুগিদের বেশির ভাগ মানুষ কাপড় কিনে খাইছে ',
     'প্রোডাক্টগুলো ভালোই',
     'মিলিয়ে পান্ডামার্টের সার্ভিসে সন্তুষ্ট পান্ডামার্ট শুভকামনা রইল',
     'দের সার্ভিস টিম ভাল তেমনি অথেনটিক প্রোডাক্ট ব্যাপারেও সচেতন',
     'ফালতু সার্ভিস',
     'মামলা ভাই সুযোগ না',
     'সকল ফ্রি প্রোডাক্ট প্যাকেজিং প্রোডাক্ট কোয়ালিটি খুবই ভালো',
     'ফালতু সার্ভিস মনেহয় ডিসপ্লের পণ্য বিক্রি প্লাস্টিক পণ্যে দাগ ধূলাবালি',
     'দাম বেশি রাখে',
     'চাল ডাল ইদানিং পুরাই ফালতু সার্ভিস দেয়',
     'নামে ভোক্তা অধিকারে মামলা করেছি কিছুদিন পোকা/পচা খেজুর দিয়েছিল',
     'ই-কমার্স সেক্টর ঘুরে দাড়াচ্ছে আলহামদুলিল্লাহ',
     'আলহামদুলিল্লাহ আমিও রিফান্ড পাবো ইনশাআল্লাহ',
     'টাকা ফেরত দেয়নি কিউকম',
     'আলহামদুলিল্লাহ ই-কমার্স সেক্টর ঘুরে দাড়াচ্ছে',
     'ই-কমার্সের সুদিন ফিরতেছে আলহামদুলিল্লাহ',
     'আলহামদুলিল্লাহ ইকমার্স সেক্টর টা ধীরে ধীরে ফিরতে',
     'সালা বাটপারের একদিন মনের মত গালি পারতাম',
     'দারাজের বাটপারি বেড়েই চলছে',
     'দারাজে বর্তমানে ভোগান্তির শেষ নেই,আমার ৩ তারিখের কমপ্লেইনের খোজখবর নেই',
     'আল্লাহর দোহায় লাগে বিশ্ব বাটপারদের বিরুদ্ধে একটা করুন সবাই প্রকৃতপক্ষেই চিটার',
     'মত ট্রাস্টেড ই-কমার্স সাইট বাংলাদেশে কমই',
     'নিশ্চিন্তে নেন অলরেডি ইউজ করছি এছাড়াও গত ৬-৭ বছরে প্রোডাক্ট নিয়েছি সবগুলো ১০০% পারফেক্ট',
     'আমিও ভুক্তভোগী আসুন সবাই মিলে প্রতিবাদ',
     'দেশের নাম্বার ওয়ান ধোকাবাজ',
     'রিভিউ আমিও অর্ডার করেছিলাম ভাল স্বাদ ধন্যবাদ ভাই',
     'জুতা হাতে পেয়ে সত্যি জিতে গেছি ভালো মানের জুতা বাকিটা ইউজ বুঝা',
     'এগিয়ে যাক দেশের ই-কমার্স যাক ই-কমার্স অভিজ্ঞতা',
     'বক্স কেটে আসল প্রোডাক্ট সরিয়ে ভুল প্রোডাক্ট ঢুকিয়েছে এটাও নষ্ট প্রোডাক্ট',
     'সারপ্রাইজ বক্স নামে কৌশল প্রতারণা করে,আমি সরাসরি ভিক্টিম',
     'আসলেই ধান্দাবাজ প্রডাক্ট কোয়ালিটিও বাজে',
     'একটাও ভালো না',
     'প্রডাক্ট ডেলিভারীতে দেরি হয়েছে',
     'প্রোডাক্টা প্যাকেজিং আরো ভালো ভালো হতো',
     'মোনার্ক মার্ট ভালো এক্সপেরিয়েন্স ছিলো',
     'অতিদ্রুত ক্যাশ অন ডেলিভারি পেয়েছি',
     'সবাইকে বলব বাটপার দূরে থাকুন ২ টাকা বেশি গেলেও বাইরে কিনুন এরকম বাটপার ইর্কমাস দূরে থাকুন',
     'ডেলিভারী পেলাম হাতে পেয়ে মুগ্ধ হয়েছি আমের সেরা প্যাকেজিং',
     'দারাজ ভুল প্রডাক্ট দিয়েছে রিটার্ন নিচ্ছে না রিফান্ড না সেলার মেসেজ দেখেও রিপ্লাই দিচ্ছে না',
     'প্যাকেজিংটা সুন্দর',
     'ধন্যবাদ এতো সুন্দর একটা প্রডাক্ট দেয়ার',
     'উনাদের অত্যন্ত ভাল শুভ কামনা রইলো আপনাদের',
     'মাত্রই পাঠাও কুরিয়ার আপনাদের স্পোর্টস জার্সি দিয়ে গিলো অবশ্যই কোয়ালিটি যথেষ্ট ভালো',
     'ভালো লাগছে ভাই দিয়েছেন আলহামদুলিল্লাহ',
     'সঠিক সময়ে হাতে পেয়েছি কাপড়ের মান ভালো লেখা দিয়েছে নির্দ্বিধায়',
     'টি-শার্ট গুলো পছন্দের মিলিয়ে কম দামে অসাধারণ প্রোডাক্ট',
     'কাজের সাথে কথায় মিল পেয়েছি আল্লাহ পাক আপনাদের বরকত দান করুক জাযাকাল্লাহ খাইরান',
     'ধন্যবাদ কথা মিল রাখার',
     'প্রোডাক্ট কোয়ালিটি আলহামদুলিল্লাহ ডেলিভারী টাইমিং পেইজের রেসপন্স ভালো পার্সোনালি ভাল্লাগসে',
     'ভালো লাগলো আপনাদের শপিং',
     'অর্ডার নিয়েছেন কখন পাবো বলছেন না ইনবক্সে রিপ্লাই না',
     'আপনাদের ইনবক্স সার্ভিস একেবারেই খারাপ',
     'আপনাদের নিয়ম মেনে তিনদিন অডার কনফার্ম করছি এখনো পণ্য পাই নি পাবো',
     'আপনাদের পণ্য গুলো ভাল ছবির সাথে পুরো মিল',
     'আপনাদের ভালো লেগেছে দুইবার নিলাম সেটিসফাইড শুভকামনা জানবেন',
     'আপনাদের বুকিং দেয়া প্রডাক্ট কোয়ালিটি খুবই ভাল লেগেছে',
     'আল্লাহামদুল্লাহ আজকে হাতে পেলাম আপনাকে ধন্যবাদ',
     'ভালো একটা প্রডাক্ট সত্যিই ভালো লেগেছে',
     'সুন্দর যেটা চেয়েছি পেয়েছি',
     'ধূর খুবই বাজে পণ্য',
     'আলহামদুলিল্লাহ মাশা-আল্লাহ সুন্দর খুবি ভালো মানের পণ্য',
     'কোয়ালিটি ভালো আগেও কিনেছিলাম রিভিউ নাই সেলারকে ধন্যবাদ অসাধারণ কোয়ালিটি',
     'মান ভালো সবাই কিনতে',
     'ধন্যবাদ ভালো প্রোডাক্ট দেওয়ার',
     'পাঞ্জাবির কোয়ালিটি ভালো সার্ভিসও ভালো ছিলো',
     'খুবি বাজে কোয়ালিটি ভাই ডিজাইন ভালো লাগছিল কিনছিলাম দাম দিয়েও বাট পুরাই বাজে',
     'খুবই বাজে সার্ভিস খেতে চাইলে অর্ডার',
     'বাজে কাপড়! ডেলিভারী সার্ভিস একটুও ভাল না অর্ডার দেই একটা,পাই আরেকটা চেঞ্জ দেওয়ার নামে ফটকামী একচুয়ালি দে ডোন্ট নো হাও টু রান আ বিজনেস!',
     'পেজ প্রোডাক্ট এক কথায় অসাধারণ দুইবার এখান প্রোডাক্ট নিয়েছি বারই তারাতারি পেয়েছি প্রোডাক্টও এক কথায় ছবিতে',
     'গ্রেট একদম পারফেক্ট সার্ভিস!',
     'পন্যের মান ভাল  প্রাইস একটু হাই পন্যের মান দেখলে সেটাকে যথাযথই ',
     'সুন্দর ভালো মানের ক্যাপ চাইলে আপনারা উনাদের ক্যাপ টেনশন ছাডা',
     'সুন্দর ক্যাপ,,,একশো তে একশো একটা নিয়েছি',
     'প্রিমিয়াম প্যাকেজিং',
     'ওনাদের প্রোডাক্ট কোয়ালিটি যথেষ্ট ভালো সময়মতো প্রোডাক্ট হাতে পৌঁছেছে ভালো কোয়ালিটির ক্যাপ খুঁজছেন প্রোডাক্ট ট্রাই',
     'ধন্যবাদ তোমাদেরকে',
     'তোমাদের কেনা-বেচার সক্ষম হয়েছি',
     'প্রতারিত হয়েছিলাম',
     'উপকার হইলো',
     'মাস্ল ৩দিন যাবত করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম',
     'কোয়ালিটি ভালো খুশি',
     'পরিতৃপ্ত',
     'সুন্দর মুখোশ জিনিস সংগ্রহ করেছি ব্যবহারে সত্যিই দীর্ঘস্থায়ী',
     'ভালো করছি',
     'দুর্দান্ত সৃজনশীল দল.....অসাধারণ',
     'সমস্যা কেন...',
     '2 টি-শার্ট অর্ডার করেছি বিকাশের পেমেন্ট করেছি নগদ ফেরত পাননি অর্ডার নগদ ফেরত পাওয়ার উপায় নেই',
     'এইটা ডিসাইন টি-শার্ট চলবে ভালো',
     'প্রিয় ব্র্যান্ড',
     'এটার ফ্রি ডেলিভারি জানলে অবশ্যই দিতাম',
     'ফেব্রিলাইভ দরকার ছিলো আপনাদের পক্ষ মাত্রই পাঠাও কুরিয়ার আপনাদের স্পোর্টস জার্সি দিয়ে গিলো অবশ্যই কোয়ালিটি যথেষ্ট ভালো',
     'অর্ডার ডেলিবারী চার্জ দেখায় নাহ',
     'কিনেছি সত্যিই সুন্দর',
     'গুণমান দুর্দান্ত',
     'আলহামদুলিল্লাহ. চাইসি ভালো. কাপড় কোয়ালিটিফুল',
     'ডেলিভারি চার্জ ফ্রি',
     'কাপড়ের মূল্য 585tk নয় আজকে হতাশ হয়েছি',
     'Fabrilife একজন বড় ভক্ত',
     'অসাধারণ ডিজাইন',
     'প্রশংসার আপনাকে ধন্যবাদ',
     'চমৎকার সুস্বাদু নকশা ভালবাসা',
     'প্রতিটি নকশা পছন্দ প্রশংসা',
     'চমৎকার সংগ্রহ',
     'পণ্য অর্ডার পেয়েছিলাম সত্যিই ডেলিভারি অন্যান্য শোরুমের তুলনায় দ্রুত গুণমান ভাল দাম যুক্তিসঙ্গত নয়',
     'ভালো দামটা একটু বেশি',
     'পণ্যের গুণমান পরিষেবা অনুসারে মূল্য একেবারে যুক্তিসঙ্গত',
     'সত্য পণ্যের গুণমান সত্যিই ভিন্ন',
     'ফ্যাশন পছন্দ চমত্কার',
     'আকর্ষণীয় ডিজাইন',
     'সত্যিই চমৎকার কাপড়',
     'সামগ্রিকভাবে সত্যিই ভাল',
     'বিকল্প এখান কেনাকাটা না প্রচুর প্রতারক বিক্রেতা পদ্ধতিগুলি চতুর',
     'নকশা সত্যিই ভাল পছন্দ ',
     'বছরের শুরুতে 2টি ভিন্ন সময়ে ছোট আইটেম অর্ডার করেছি গ্রহণ করিনি ম্যাসেজের দিচ্ছে না',
     'হতাশাজনক কোথাও কেনাকাটা করছি',
     'আলীএক্সপ্রেস সস্তা নয় আইটেম নরওয়ের বেশি দামি গুণমান সবসময় ভালো নয় কল্পনা করুন!!',
     'জিনিসপত্র ফেরত না পোকামাকড়ের আক্রমনের দায়ে সেগুলো জব্দ কাস্টমস',
     'অসাধারণ একটা জার্সি',
     'জিনিসটি বাস্তবে ভালো না',
     'চিটার বাটপার',
     'দশ পার্সেন্ট ভাটের কথা',
     'সাউন্ড বক্সের দাম বেশি',
     'হেলমেটটা নিম্নমানের',
     'হেলমেটটা ছবিতে সুন্দর বাস্তবে ভালো লাগে নাই',
     'হেলমেটে ফাইবার লেখা থাকলেও হেলমেটটি ফাইবারের না',
     'বাচ্চার হেলমেটটি সুন্দরী',
     'জিনিসটা মোটামুটি',
     'আরো সুন্দর আশা করেছিলাম',
     'বেল্ট লেদার বললেও লেদার না আর্টিফিশিয়াল লেদার',
     'আপনাদের জিনিসের দাম একটু কমানো',
     'একদমই ভালো নয়',
     'অনলাইনে দাম বেশি',
     'আপনাদের সার্ভিসে সন্তুষ্ট',
     'জুতার মান খুবই খারাপ',
     'ছবিতে দেখেছি তখনই পেয়েছি',
     'ডেলিভারি ম্যান ভাল ছিলনা',
     'গেঞ্জিটা সুন্দর',
     'এক্সক্লুসিভ কালেকশন',
     'অর্ডার করেছি তিন এখনো হাতে পায়নি',
     'ডেলিভারি দেরি',
     'দুইদিন পণ্য হাতে পেলাম',
     'এমটি হেলমেট কালেকশন আরো বাড়াতে',
     'বাইকের জিনিসপত্র অনলাইন কিনলাম',
     'জাবের ভাইয়ের অনলাইনের দোকান ভালো',
     'শুভকামনা রইলো ভাই',
     'সেরা বাংলা সেরা পন্য',
     'ম্যাম মতামত জানানোর ধন্যবাদ',
     'দামে কম মানে ভালো',
     'শার্টটি খুবই সুন্দর গুলশন পাশে শোরুম ঠাকলে আমক জানাবে',
     'এগুলো ক কালার গেরান্টি ওয়াস কালার যায়',
     'কাপড় ভালো',
     'সহজ মার্জিত',
     'ভাল্লাগছে',
     'ভাল মানের পণ্য',
     'সুপার কলেকশন',
     'এগুলো ক কালার গেরান্টি ওয়াস কালার যায়',
     'সহজ প্রিয় ব্র্যান্ড বাংলাদেশে',
     'অধিকাংশই ভালো মানের',
     'সুন্দর সংগ্রহ',
     'সুন্দর পাঞ্জাবি',
     'ভোক্তা অধিকারে মামলা করবো আপনাদের নামে-ফালতু জিনিস রং জ্বলে যায় ১ মাসে সমস্যা পরিবর্তন না দিয়ে হয়রানী পাবনা শোরুমের সকল স্টাপ আপনাদের বেয়াদব',
     'ডিজাইন গুলো বড়দের ভালো চলত',
     'সকল শুভ কামনা',
     'ভাই আপনাদের টিশার্ট গুলো একটু ভালো  আপনাদের টি-শার্টগুলো লম্বা পড়া সম্ভব না',
     'বাংলাদেশের একমাত্র ফালতু ব্যান্ড থাকলে ইজি ফ্যাশন',
     '২০১৪ সাল পযন্ত পছন্দের ব্রান্ড ইজি ফ্যাশন লিমিটেড',
     'নেক্সট প্রচুর জামাকাপড় মাতালান ২ জিন্স প্যান্ট কিনেছি দোকানগুলিতে মান পছন্দের যোগ্য শার্ট খুঁজে পাইনি',
     'সুন্দর',
     'ফুটপাত কোয়ালিটি তোদের চাইতে ভালো',
     'পাঞ্জাবি শান্তি',
     'বিউটিফুল সমাস',
     'পছন্দের ব্রান্ড ইজি।হোম ডিস্ট্রিক সিলেট',
     'আসসালামু আলাইকুম টি-শার্ট অর্ডার করেছিলাম হাতে পেলাম প্রথমে ভেবেছিলাম কাপড়ের মান ভালো কিনা টি-শার্ট হাতে মন ভরে',
     'অনলাইনে বিশ্বস্ত একটা কোম্পানি পেলাম ফ্রান্সে থাকি আরো টি-শার্ট অর্ডার করবো ইনশাআল্লাহ নিঃসন্দেহে',
     'অসাধারণ',
     'শার্ট গুলো সুন্দর ওগুলো কেনার সাধ্য নাই',
     'সুন্দর লাকছে',
     'প্রোডাক্ট ভালো না কালার শাদাশে হয়ে যায়',
     'নাইচ',
     'আপনাদের গোল গলা গেঞ্জি গুলো লক সেলাই খুলে খুলে যায় গেঞ্জি নষ্ট হইছে',
     '১০৯৪ টাকা প্রাইজ একটা সাট নিছিলাম।কিন্তু পড়তেই রং শেষ একটা চেঞ্জ দিছে টাও একি',
     'ঈদ উল ফিতর এবারের কালেকশন সত্যি ভালো',
     'মেসেঞ্জারে অর্ডার দিলে অর্ডার কনফার্ম করেনা',
     'বাহ্ অসাধারন',
     'একদম বাজে হয়ে',
     'কাপড়ের মান খুবই নিম্নমানের কয়েকদিন রং নষ্ট হয়ে যায়',
     '২০১৪ সাথে আছি।কিছু অভিযোগ ছিলো করলাম না',
     'কাপড় গুলো ফালতু দাম আকাশ চুম্বী ইজি পরি না',
     'ফালতু কালেকসন',
     'একটা প্যান্ট কিনে বাসায় দেখি কোমোরের বোতাম ভাংগা শব স্টীকার লাগানো এখনো কিন্ত মেমোটা ভুল দেয় নাই আমিও খেয়াল নাই',
     'কাপড়ের মান টা খারাপ দাম বাড়াচ্ছে',
     'ইজি ফ্যাশন সময় অসহায় মানুষের পাশে প্রত্যাশা দোয়া রইল ইজি ফ্যাশন',
     'রমজানের টি-শার্ট দাম ৪৯৫ টাকা ঈদের বেড়ে ৫৯৫ টাকা সবচেয়ে বড় কথা এক সপ্তার কালার',
     'বর্তমানে ইজি ফ্যাশন গোল গলার টি-শারট গুলোর মডেল একটাও ভালো না',
     'চেয়ে বাজে ব্রান্ড শহরে',
     'পছন্দ হয়েছে দাম বেশি',
     'একছের দাম',
     'পণ্যের দাম বেশি দেখিয়ে তারপরে ডিসকাউন্ট দেয়া একটা শুভঙ্করের ফাঁকি',
     'ধোক্কাবাজী ধাপ্পাবাজী অফার উৎসবের সময়ে নগদে নেয় না,ক্যাশ নেয় হ্যান্ডক্যাশ.নগদে নেওয়ার সময় নাই',
     'ইজির প্রোডাক্ট ভালো না সালারা ২০০৳ টি-শার্ট ৬০০ টাকায় বিক্রি',
     'শার্ট রং নষ্ট হয়ে যায়',
     'ইজি বাজে একটা ব্যন্ড কালার থাকেনা।টি সার্ট সার্ট',
     'আস্তে আস্তে ফালতু ব্র্যান্ড কাপড় গুনগত মান ভালো না',
     'আপনাদের সেলাই মান ভালো নিচের সেলাই কয় এক খুলে যায়',
     'ইজির প্রোডাক্ট ভালো না ফালতু',
     'ভাই শার্ট কালার ১ মাসের নষ্ট হয়ে যায়',
     'শার্টটা হেব্বি সুন্দর',
     'আলহামদুলিল্লাহ  ভ্যালো মোটো পেইচি... প্যাকেজিং ভ্যালো সাইলো',
     'ছবি পণ্য নিখুঁত দাম taw অমর কাচ সাশ্রয়ী মূল্যের lagche বক্সিং ভালো পাইচি রাইডার আচরণ বেশি ভালো চিলো জেটা অমর খুবি ভালো ল্যাচচে',
     'চার্জ সর্বোচ্চ এক দেড় ঘণ্টা আসলে কমদামি প্রোডাক্ট কথা ইয়ারফোন ব্যতিক্রম নয় আসলে যেটা সত্যি বললাম',
     'মানিব্যাগটা সুন্দর 🥰 গুণগতমান ভালো আসল চামড়ার তৈরি ওয়ালেট টা সফট',
     'সত্যিই প্রডাক্টটা সুন্দর পুরোটাই চামড়া',
     'প্রডাক্ট কোয়ালিটি অসাধারণ শেলাই গুলা সুন্দর',
     'চামড়া জানি একটু লোকোয়ালির',
     'কাপড় নষ্ট হয়ে যায়।যদি কিনি L সাইজ ঐইটা হয়ে যায় XL',
     'ক কথায় অসাধারন ৫ ষ্টারের বেশি দেয়া জায়না নয়ত আরো বেশি দিতাম কম দামে ভালো মধু দেয়ার সেলার দারাজকে ধন্যবাদ',
     'ভালো মধু চোখ বন্ধ',
     'আলহামদুলিল্লাহ্\u200c ঠকিনাই যেমনটা আশা করেছি চাইতে ভালো মধু',
     'পণ্য সন্তুষ্ট সবকিছু ঠিক দ্বিতীয় শেষ ছবিটি এটির ভিতরে ফোন তোলা কিনতে',
     'প্রডাক্ট টা ভালো নিরদ্বিধায় সবাই,আমি বন্ধুদের সাথে নদিতে,পুখুরে কাটানো মজার সময় গুলোর ছবি তুলার নিছি',
     'পন্যটা আজকে পৌঁছেছে..অনেকদিন লেট আসছে পন্যটা',
     'ছবির পেয়েছি পছন্দ হয়েছে',
     'তুলনামুলক দাম একটু বেশি হয়েছে দামে বাসায় প্রডাক্ট হাতে পেয়ে অত্যন্ত খুশি',
     'এককথায় দামে প্রডাক্ট পছন্দ হয়েছে আপনারা চাইলে',
     'নিবেন না চাইছি দিসে রিটার্ন টাকা টায় জলে গেলো',
     'কোয়ালিটি ভালো দেখতেও সাইজে একটু বড় অর্ডার এক সাইজ ছোট অর্ডার কিরবেন',
     '৫০০ টাকা বাজেটে সেরা ট্রিমার ৪৫ মিনিটের মত ব্যাকআপ',
     'প্যাকেজিং খুবই নরমাল বাট প্রোডাক্টটির ক্ষতি হয়নি',
     'প্রডাক্ট একদম ইনটেক পেয়েছি।আসা করছি ভালোই প্রয়োজন',
     'গেঞ্জির রং চেয়েছিলাম ছবিতে তেমনটাই পেয়েছি লাল নীল',
     'প্রোডাক্টা ভাল,কম দামে এতো ভাল প্রোডাক্ট পাব ভাবিনি কাপড়টা ভালো চাইলে',
     'টি--শার্ট একদম ঠিক আছে।দাম অনুযায়ী বেশিকিছু',
     'করলাম ২৪ ঘন্টা ভালোই করেছ + ভালোই ঠিক টাক',
     'দরকারী অ্যালার্ম ঘড়ি অ্যালার্মের শব্দটি কিছুটা বেশি ভাল',
     'পণ্যটি আশ্চর্যজনক মায়ের কিনেছি',
     'দাম হিসেবে খারাপ না ভালোই...তবে চশমার ফ্রেমের লাল ডিজাইনটা রাবাবের,কিছুদিন উঠে যাওয়ার সম্ভাবনা',
     'অল্প সময়ে পেয়েছি কাভার বক্স হয়নি জানলে ওডার টা করতাম না স্রার দিলাম',
     'ভাল প্রতিটা মাক্স ☺️ পাশাপাশি KN95 মাক্স টা সম্পূর্ণ প্যাকেজটি মূল্যে অসাধারন',
     'ঠিক ভিতরে মোটা সুতার কাপর হয়েছে দাম অনুযায়ী ঠিক',
     'আলহামদুলিল্লাহ ঠিক চাইছি পাইছি কালার সেম পাইছি',
     'জিপার সিকিউরিটির কার্ড হোল্ডারটি সত্যিই সুন্দর আউটলুক ডিজাইন চমৎকার',
     'মাশাআল্লাহ ভালো টি-সার্ট টা',
     'যেরকম দেখেছিলাম ঠিক সেরকম পেয়েছি।কাপড়ের কোয়ালিটি ভালো',
     'ঠিকঠাক পেয়েছি সুন্দর ধন্যবাদ সেলারকে 100% ওয়াটারপ্রুফ ভালো লেগেছে আপনারা চাইলে',
     'এটার সেলাই বিষয়ে সেলারকে একটু সচেতন কাপড় সেলাই মত বডার দিলে ভাল',
     'এছাড়া দাম অনুযায়ী কাপড়ের মান মোটামুটি ভাল ভাবছিলাম কাপড় একটু মোটা',
     'প্যাকেজিং ভালো প্রোডাক্ট ভালো ওয়াটার প্রুফ তাড়াতাড়িই প্রোডাক্ট হাতে পেয়েছি সবমিলিয়ে সন্তোষজনক',
     'প্রোডাক্ট টা খুবই ভালো প্যাকেটিং টা সুন্দর ঘড়িটা সুন্দর এক কথায় অসাধারণ',
     'খুবই চমৎকার  সত্যিই অবাক অনলাইন কিনলাম সম্পূর্ণভাবে স্যাটিস্ফাইড',
     'আলহামদুলিল্লাহ যেরকমটা চেয়েছি সেরকমই পেয়েছি ২ টা অর্ডার দিছিলাম ২ টাই পেয়েছি',
     'অল্প দামের ভালোই ব্যায়াম',
     'খুবই বাজে মানের পন্য এক যায় নি ভেঙে খুবই নিম্নমানের প্লাস্টিকের হয়েছে পন্য তে সবাই না নেওয়ার অনুরোধ করব',
     'অবিশ্বাস্যভাবে সুন্দর ব্যাগ আপনাকে ধন্যবাদ',
     'ব্যাগটি খুবই ভালো মানের পণ্যনিয়ে খুশি',
     'কোয়ালিটি ভালো ১৫০০ টাকা প্রইজে ভালো ব্যাগ পেয়েছি লোকাল শপে সেইম ব্যাগ দিগুন দাম',
     'প্রোডাক্ট ভালো  ঝাল',
     'প্রোডাক্ট অবস্থা খুবই খারাপ ইদুরের কাটা জিনিস দিয়ে দিছে প\u200d্যাকেট সস খারাপ অবস্থা',
     'সর্বোত্তম মূল্যে ভাল মানের',
     'প্রয়োজনীয় প্রডাক্ট...সেলার ধন্যবাদ...ভাউচার দিয়ে ২৪ টাকায় কিনেছি....আপনারা কিনতে',
     'খুবব কাজার জিনিশ কোয়ালিটিও ভালো',
     'ভালো প্রডাক্ট ব্যশ কম দামেই পেয়েছি আবারো কিনবো ইনশাআল্লাহ',
     'আলহামদুলিল্লাহ অক্ষত অবস্থায় পেয়েছি আগেও নিয়েছিলাম আবারও নিলাম',
     'পণ্যের মানও ভালো আশা করছি দাম কমে পাবো',
     'ঠিক পেয়েছি।সাথে gift পেয়েছি',
     'বেল্ট মান টা দাম হিসেবে খুবি ভালো পাঁচ দিনের ডেলিভারি পেয়েছি',
     'ভালো জিনিছ চেয়েচিলাম তএমন পেয়েছি',
     'দাম অনুযায়ী প্রডাক্ট টা ভালো সেলার ভাইয়ের যেটা চাইছি দিয়েছেন সাইজ কালার সেলাই সবকিছু ঠিক সেলার ভাইকে ধন্যবাদ',
     'ধন্যবাদ প্রোডাক্ট কোয়ালিটি ভালো ছিলো সেলাই ভালো ছিলো এক কথায় দামে প্রোডাক্ট পেয়ে পুরোপুরি সন্তুষ্ট',
     'কাপড় জার্সির কাপড় description পেয়েছি  সেলার অনেস্ট ',
     'সাইজ পারফেক্ট কাপড় টা জোস রেকোমেন্ডেড চাইলে',
     'ভালভাবেই ফিট হয়েছে।ডেলিভারিও পেয়েছি দ্রুত',
     'প্রথমে ধন্যবাদ সেলার ব্ল্যাক চাইছিলাম পাইছি এক কথায় কম দামে ভালো পণ্য ব্যাটারি চার্জ সাউন্ড কোয়ালিটি ভালো',
     'প্রডাক্ট কোয়ালিটি ভালো ছিলো আশা করছিলাম পেয়েছি  সাউন্ড কোয়ালিটি খুবই ভালো লাগছে এক কথায় 10/10 কুমতি পাইনি',
     'সেলারের খুবই ভালো লাগছে কালার চেয়েছি পেয়েছি',
     'অসাধারণ লাইট,আমি এক টানা ৪০ মিনিট জালিয়ে রেখেছি আরো যেত ইচ্ছে জালাই নি পছন্দ হয়েছে',
     'লাইটা ছোট সুন্দর দাম ভালো।আলো বেশী চাজ ভালোই',
     'চাইলে ভালো একটা পণ্য কম দামে',
     'যেমনটা করছিলাম না মিস ফায়ার',
     'বিজ্ঞাপনের',
     'ভাল পণ্য সহজ',
     'আলহামদুল্লিহা ভালো product পেয়েছি।সেলার service ভালো।তারা ভালো chat response করে।T-Shirt টা পছন্দ হয়েছে।Fabric quality যথেষ্ট ভালো',
     'অনেকদিন রিভিউ দিলাম।T-shirt সুন্দর।খুব আরামদায়ক।কাপড়ের কোয়ালিটি ভালো।Showroom তুলনায় কম দামে T-shirt পেয়েছি',
     'দ্রুত সময়ে ডেলিভারি পেলাম  মূলত লোটো kub আরামদায়ক',
     'সাবধানে ওডার দিবেন  ওডার দিছি xl তাহারা আমারে দিছে S সাইজের শার্ট',
     'নির্দিষ্ট সময়ের ভিতরেই সঠিক গুনগত মানসম্পন্ন পন্যটি দেয়ার সেলার,এবং রাইডারকে ধন্যবাদ',
     'অসাধারণ একটা ঘড়ি  পানির দামে শরবত পাইলাম সাথে ফ্রি চুলের বেন্ড  ধন্যবাদ সেলার',
     'একদম জোস,চাইলে কিনথে',
     'আলহামদুলিল্লাহ যেটা অডা্র করেছিলাম পেয়েছি চাইলে আপনারা',
     'আলহামদুলিল্লাহ টেবিলটা সুন্দর ছেলে পেয়ে অনেকটাই খুশি সবাই',
     'বাচ্চাটা লেখা পড়ার আগ্রহ পাইছে টেবিল টার ধন্যবাদ সেলারকে সময় মত ভালো পন্যটা দেয়ার',
     'স্মার্ট ওয়াচ অসাধরণ পারফম কর\u200cছে মাই\u200cক্রোফ\u200cনের সাউন্ড কোয়া\u200cলি\u200cটি ভা\u200cলো চোখ বন্ধ ক\u200cরে পা\u200cরেন',
     'পন্যটি ভালো ছিলো টা চেয়েছি টাই পেয়েছি ইস্কিনপটেক্টর টা দিয়ে দিলে ভালো',
     'আলহামদুলিল্লাহ ভালো প্রোডাক্ট  দেড় চালানোর রিভিউ দিচ্ছি  ব্যাটারি ব্যাকআপ ভালো  সবাই চাইলে',
     'খুবই উপকারী স্মার্ট ওয়াচ... সাথে সমস্যার মুখোমুখি হয়েছি',
     'আলহামদুলিল্লাহ পাঁচ টা অর্ডার করেছিলাম পাঁচ টাই হাতে পেয়েছি ভালো লাগছে বলবো সত্যিই অসাধারণ প্রডাক্ট কোথাও সমস্যা পাইনি',
     'সত্যিই দারুণ ভাবি নাই ভালো প্রচন্ড ভালো জামা নিয়েছি ভালো এসেছে প্রচন্ড খুশি আপনারা চাইলে ধন্যবাদ দারাজ',
     'জামাটা ছবিতে যেরকম দেখছি সুন্দর  কাপড়ে মান ভালো এতো ভালো মানের পোশাক দেওয়ার অসংখ্য ধন্যবাদ',
     'যেমনটা দেখেছিলাম তেমনটাই মোটামুটি ভালো',
     'দাম অনুযায়ী ঠিকঠাক এরিপ্রক্ট কালার চেঞ্জ হয়ে ভালোই!',
     'খারাপ না,আমার ভালো হয়েছে,অরিজিনাল ছবি দিলাম নিলে',
     'ইউস ডুরাবিলিটি লাগবে ভাইয়াকে ধন্যবাদ সততার',
     'বলবো বল পুরাই অস্থির একটা জিনিস প্যাকেজিং ঠিকঠাক সুন্দর চাইলে আপনারাও কিনতে',
     'প্রোডাক্ট টা ভালো শুধু খারাপ লাগছে স্টেন্ড ফোন টাচ সেকিং করে।যেটা বিরক্তিকর এছাড়া প্রবলেম ফেইস নি',
     'পণ্যটি ছবিতে দেখেছিলাম হচ্ছিল বড় হাতে পাওয়ার দেখলাম খুবই জিনিস দেখলে খুবই কম দামি',
     'চাইছি একটা পাইছি আরেকটা,পাইছি নষ্ট এগুলো নিবেন না',
     'কম প্রাইজ হিসেবে ভালোই আছে.. সার্ভিস দেয় দেখার বিষয়',
     'সুন্দর ভালো লেগেছে অসাধারণ লেগেছে',
     '2 দিনের পেয়েছি দেখায় প্রিমিয়াম মোটরটি প্রতিক্রিয়ার শক্তিশালী.ধন্যবাদ বিক্রেতা ভাল পণ্যের 2 দিনের পেয়েছে',
     'পেন্টের কাপর টা ঠিক আছে,,কিন্তু গেঞ্জির কাপর টা পাতলা,,but আরামদায়ক,',
     'সাইজের সমস্যা চেয়ে ১ সাইজ বড় দরকার সঠিক পরামর্শ পাই নাই ফেরত নাই',
     'নেওয়ার বলবো প্রাইজে এরকম জিনিস পারফেক্ট',
     'মোটামুটি ভালো পণ্য  24 টাকায় কিনেছি  আশা ভবিষ্যতে আরো কম দামে পাব',
     'খাবারটি খেতে সুস্বাদু কোয়ালিটি খুবই ভাল খেতে চাইলে অর্ডার',
     'আলহামদুলিল্লাহ সময় হাতে পেয়েছি প্রোডাক্টি সুন্দর সুস্বাদু',
     'জিনিসটা সুন্দর ভাঙ্গা ডেলিভারি ম্যান যত্নে আসতে পারেনি',
     'রাধুনি মানেই বেস্ট এতো কম দামে পাব ভাবতেই নাই কিনবো ইনশাআল্লাহ',
     'তারগুলো মোটামুটি ভালোই  কারোর লাগলে নিঃসন্দেহে',
     'ওডার ভাবছিলাম কম দামে দেব ভালো না সন্দহে ছিলাম পয়ে দেখি ভালো সাথে টেস্টার ফ্রি কম দামে ভালো দেওয়ার ধন্যবাদ',
     'কেবল ভাল সাথে টেস্টার ফ্রি দিছে ধন্যবাদ',
     'ফালতু প্রোডাক্ট ৭০ টাকার ১২ ভোল্ট মোটরের বাতাস চেয়ে বেশি',
     'ফ্যানটার স্পীড ভালই ছোট ফ্যান হিসেবে খারাপ নয় স্পীড কমানো বাড়ানোর অপশন নেই ঠিক ওয়াটের ফ্যান description লেখা নেই ধন্যবাদ সেলার ভাই',
     'এক কথায় অসাধারণ সেলার ভাই দ্রুত ডেলিভারি দিবে কল্পনাও করিনি',
     'টা চাইছি ঠিক টাই পাইছি ভাল বেল্ট টাকা ভাল হইছে',
     'জিনিস টা ভালোই বাট বড় করছিলাম একদম ছোট ভালোই',
     'কোয়ালিটি এভারেজ সেলাইগুলো দুর্বল ব্যাগটা সুন্দর বাচ্চাদের সাথে মানানসই ভারী না বহন বুদ্ধিমানের',
     'এতো অল্প দামে অসাধারণ পণ্য আগেও নিলাম বেস্ট আগের শপ',
     'Underwear অর্ডার দিলাম xxl size (41-44 ২টা..কিন্তু একটা ছোট সাইজ পিক দেখলেই বুঝবেন একটা ছোট দিছেন! একটা H&M Brand অর্ডার ছিলো ২টাই US Polo সাইজ ছোট ফালতু পুরাই!',
     'অর্ডার করেছিলাম এক্সেল পেয়েছি একদম ছোট সাইজ দুইটা সাইজের ভণ্ডামির একটা সিমা দরকার',
     'অর্ডার লার্জ সাইজের নেভি ব্লু কালার হাতে পেলাম মিডিয়াম সাইজের সাদা রং খারাপ লাগলো রিটার্নের ঝামেলা',
     'এক কথাই দারুণ  পড়লে বুঝা যায় না সেলার আন্তরিক  আগেও সেলারের পন্য কিনেছি',
     'বুঝতে পারলাম না একপ্যকেটে ২টা থাকার কথা ১টা',
     'আপাতদৃষ্টিতে ভালই কতটা মজবুত এখনো পরীক্ষা নি অর্ডার তিন দিনের মাথায় প্রোডাক্ট হাতে পেয়েছি বক্সের ভিতরে স্ক্রু ড্রাইভার হাতলের গার্ড  ভাঙ্গা',
     'দাম অনুযায়ী ভালো প্রডাক্ট ছোট খাটো চালিয়ে নেয়া প্রডাক্ট ঠিক পেয়েছি স্কু বিটের পাশে প্রতিটি বিটের নাম লেখা থাকার কথা এটার নাম নেই!',
     'মাশাআল্লাহ সুন্দর Product দাম হিসেবে ঠিক সবাই',
     'প্রোডাক্ট টিক ডেলিভারি সময় মত কালারটা পিংক',
     'জিনিসটা ভালো হইছে প্রোডাক্টটা নিছি পছন্দ',
     'পারফেক্ট সাইজ আরামদায়ক ৷ সন্তুষ্ট ৷ আপনারা নিশ্চিন্তে কিনতে',
     'ফা গুলা মোটামুটি ভালই দাম ঠিক সোফা সবাই খুবই আনন্দ!এখন',
     'মোটামুটি ভালো সোফা ফিনিশিং ভালো নি',
     'A লেভেল একটা জিনিস সেলারকে বলবো মানসম্পন্ন প্রোডাক্ট সেল',
     'প্যাকেজিং + উপরের মোড়ক এক কথায় অসাধারণ',
     'আপাতত ঠিকঠাক দেখি কেমন  প্যাকেট সুন্দর পাইছি।',
     'দাম কম ভাবছিলাম প্রোডাক্টই ভাল না আসলে দাম ভালো ',
     'খুবই ভালো মজবুত রিজনাবল',
     'আলহামদুলিল্লাহ আগের অর্ডার পরের ডেলিভারি মাস্কগুলোর তুলনা হয়না।দেখতে ফ্রেস,ব্যবহারেও খুবই আরামদায়ক',
     'ভাল অ্যান্ড্রয়েড ফোন অল্প দামের পছন্দ হয়েছে আপনারা চাইলে',
     'অনকে দ্রুত পেয়েছি রাতে অর্ডার দিয়ে সকাল ১১টায় ডেলিভারি পাইছি প্রোডাক্ট অরিজিনাল বাকিটা ইউস বুজা',
     'মোবাইল পেয়েছি ঠিক ঠাক পেকেজিং টাও ভালো ছিলো',
     'খাবারটি খেতে সুস্বাদু কোয়ালিটি খুবই ভাল খেতে চাইলে অর্ডার',
     '"আলহামদুলিল্লাহ সময় হাতে পেয়েছি প্রোডাক্টি সুন্দর সুস্বাদু।',
     'চাইছি সাদা পাইছি নীল',
     'ভালে কোয়ালিটি আঠা স্ট্রং সেলার রেস্পন্সিবল সবাই',
     'বোলবো অর্ডার কোরলাম সাদা দিলো ব্লু,এটা ঠিকনা,এছারা ঠিক প্যাকেটিং ভাল ছিল,নতুন মাল দিয়েছে,সাউন্ড ভালোই।',
     'টেপ পূরাতন এক পাশে ময়লা লাগানো  পাশ ঠিক',
     'ভালো প্রোডাক্ট ছিলো প্যেকেজিং ভালো ছিলো রিকমেন্ডড ',
     'দামের তুলনায় ভালো কোয়ালিটির একটা গ্লাস প্যাকেজিং টাও সুন্দর ২ দিনেই প্রোডাক্ট হাতে পেয়ে গেছি সেলারকে ধন্যবাদ',
     'অসাধারণ পণ্য।১৭৭ টাকায় পণ্য বাজারে পাওয়া অসম্ভব।সাথে পেয়েছি ৩ দিনে হোম ডেলিভারি',
     'কোয়ালিটি ঠিক সাইজ বড় দিয়েছে একদম হয়না পুরা মুখই ঢেকে যায় দিয়ে',
     'কম দামে ভাল চসমা পেলাম ভালো লাগতেছে।সার্ভিসম্যান ভালো ছিলো',
     'যেরকম আশা করেছিলাম চেয়েও বেশি ভালো',
     'মেশ কাপড়ের পড়েও আরাম দেখতেও খারাপ না শুধু কাপড় পাতলা ',
     'কেনার এক্সপেরিয়েন্স টা বাজে নির্দিষ্ট সময় থেকো তিন দেরি,জিনিসটা ভালই লাগলো ভালোই ভেবেছিলাম লেস এমবোটারি দেখি প্রিন্ট',
     'সত্যি মূল্য প্রোডাক্ট মাশাল্লাহ সুন্দর',
     'কোয়ালিটি সম্পন্ন প্রোডাক্ট ধন্যবাদ',
     'মানের পণ্য চমৎকার জিনিসপত্র প্যাকেজিংও দুর্দান্ত',
     'স্নোবল ফেয়ারি লাইটস টা সত্যিই অসম্ভব সুন্দর কোয়ালিটি ফুল প্রতিটি লাইট ঠিক',
     'বাহ,অনেক ভালো লাগতেছে,য়েমন দেখেছি,তার চেয়েও সুন্দর লাগতেছে অনেকটা লম্বাও',
     'দামে কম মানে ভালো',
     'জামাটি ভালো গরমের সিজন দাম অনুযায়ি আপনারা চাইলে কিনতে',
     'যেমনটা ছবিতে ঠিক তেমনই পেয়েছি',
     'দুইটা নিয়েছি সুন্দর',
     'দাম পাঞ্জাবিটা সুন্দর একটু গরম লাগে সমস্যা নাই ভালো',
     'কাপড়ের কোয়ালিটি ভাল ধাম আনুয়ায়ী ঠিক সাইজ চাইছিলাম পাইছী',
     'সত্যি ২৯৯ টাকা তে অসাধারণ মার্কেটে জিনিস ৫০০/৬০০ চাইবে ভালো হয়েছে',
     'একবারের বেশি পরতে পারিনি বাজে জিনিস',
     'দাম হিসেবে কাপড় স্টাইল ভালো',
     'মডেল ২/৩ ধোয়ার উঠে সম্ভাবনা',
     '৩০০ টাকার ভাল না সুন্দর আপনারা চাইলে',
     '"মুটামুটি পরতে বাটন খুলে গেসে সেলায় ভালো সিল না',
     'ভাল পান্জাবি টা,কম দামে ভাল জিনিস',
     'জানিনা কার কেমন লাগবে খুবই পারফেক্ট।',
     'মানান সই',
     'কোয়ালিটি দাম ঠিকই',
     'বকিছুই ঠিকঠাক সাইজটা উল্টাপাল্টা সাইজটা পাঠানো স্মল সাইজ ট্যাগ লাগানো সাইজের।',
     'অর্ডার করছি পেলাম মেজাজটাই খারাপ হয়ে',
     '"আলহামদুলিল্লাহ যেটা ওর্ডার করছি পাইছি।অল্প দামে খুবই ভালো একটা প্রোডাক্ট',
     'অর্ডার দিলাম কনভার্স দিলো লোফার কেমন ভুল অর্ডার',
     'একদম খারাপ প্রডাক্ট দিছে ছবির সাথে মিল নাই',
     'যেটা পেয়েছি ওইটার কোনু মিল নেই ৩৯ অর্ডার দিছি ৪০ পাইছি',
     'বাটপার,চাইছি এক জিনিস দিয়েছে জিনিস,আমার টাকা টাই নষ্ট',
     'জুতা হয়েছে',
     'ভাই চাইছি হলুদ দিছে কালো এইটা হইলো হলুদ জুতা লাগবে',
     'জুতা ভালই হয়েছে গুলো চাইছিলাম অইগুলো দেয় নাই',
     'চাইলাম পাইলাম খারাপ একটা অভিজ্ঞতা',
     'কালার দিয়েছি পাই নাই সাইজ টাও টিক না',
     'দিলাম সাদা সোল সাদা ফিতার আসলো পুরা কালো কথা হ্যা পুরাই পালতু',
     'একেবারে ফালতু চেয়েছি ৪২ সাইজের পাঠিয়েছে ৪৩ সাইজ',
     'একটা অর্ডার করেছি একটা দিছে মান খুবই খারাপ',
     'বাজে একটা পিরত দিয়ে দিছি এখনো টাকা দেয় নাই',
     'দিছেন চাইছি লাল দিছেন হলুদ এটাকি',
     'দানের তুলনায় ভালো জিনিস',
     'পণ্য ফেরত চাই',
     'কালার ভুল',
     'কাজটা খারাপ কিনবো নাহ ওডার দিছি দিছে',
     'আজকে প্রডাক্টটা পেয়ে খুবই খুশি বিক্রেতার প্রডাক্ট প্যাকেজিং মান ভাল',
     'যেমনটা দেখেছি তেমনটাই পেয়েছি কোনরকম প্রবলেম',
     'ভাল বোতামগুলি কিছুটা আলগা কিছুটা শক্তিশালী সুন্দর',
     'প্রোডাক্টটি অর্ডার কনফিউশনে ছিলাম কিনা আসলে যেমনটা ভেবেছিলাম সেরকম না প্রোডাক্টটা ভালো,',
     'পছন্দ কিনলে ঠকে যাই ঘড়িটি হাতে কিনে জিতলাম',
     'ছোট ভাইকে গিফট নিয়েছিলাম দিয়েছি পছন্দ ধন্যবাদ সেরার',
     'ভালো প্রোডাক্ট টা এক কথায় অসাধারন দামে ঘড়ি টা বেস্ট চয়েস',
     'প্যাকেট খোলার পরই উপরের কাচ খুলে খুবই হতাশাজনক',
     'সুন্দর।হাত নাড়া চাড়া ভিতরে ফিটিংস দূর্বলতা থাকার কারনে শব্দ',
     'ঘড়িটা ভালো উপরের গেলাসটা লাগানো ছিলোনা আসা করিনি',
     'ঘড়ি দিছেন ভাই টাইম যায়না।এখন করব',
     'ভালো মানসম্মত ঘড়ি',
     'প্রডাক্টিভ হুবুহুব একদম ছবির সাথে মিল খুশি',
     'চিটিং বাজ ২টা প্রোডাক্ট অর্ডার ছিলো শুধু ১ টা প্রোডাক্ট এসেছে নিম্নমানের',
     'দাম অনুযায়ী হাতঘড়িটা ভালো',
     'ভাই ঘড়ির মান অত্যান্ত খারাপ অবস্থা,সমস্থ পার্টস খুলা',
     'হেডফোনগুলি দুর্দান্ত দাম সাউন্ড কোয়ালিটি ভাল',
     'দেখেছিলাম তেমনটাই পেয়েছি সাউন্ড সিস্টেম ঠিক একদম পারফেক্ট',
     'বাজে জিনিস ছাউন্ড ক্লাসিক ভাল না',
     'একসেট সেট কাপড় বহনের পারফেক্ট',
     '"আলহামদুলিল্লাহ পণ্যটি ভাল বেশি স্থিতিস্থাপক নয়...।',
     'অসাধারণ কমফোর্টেবল ঠিক সাইজ পেয়েছি এছাড়া ডেলিভারিও দ্রুত পেয়েছি',
     'ভালো লেগেছে',
     'বাজে মাল',
     'টাওয়েল টা ভাল লেগেছে',
     'ঠিক যেমনটা বিজ্ঞাপন ত্রুটি নেই.. গুণমান গড়',
     'ছবি পারলাম না ফেলছি অভিযোগ কারন নাই ছবি তে ঠিক পাইছি',
     'ছেলে টা ভালো, ব্যাবহার ভালো সময় মতোন পন্য হাতে পৌছে দেয়।',
     'তাওয়াল ভালো বিশেষ গাড়ি মোছার',
     'একদম বাজে মাল খুভই পাতলা কোয়ালিটি খারাপ',
     'ভালো কালার ডেলিভারি দিছে',
     'ভালো পানি ময়লা পরিষ্কার',
     'সাইজ ছোট,মানে ভালোই',
     'ভালো না',
     'পণ্য সবসময়ের মতই খাঁটি সাবান বিনামূল্যে পেয়েছি ছাড়ের মূল্যে পেয়েছে দাম যুক্তিসঙ্গত ডেলিভারি দ্রুত ছিল।',
     'বক্স ভালো সুন্দর একটুও তেল পরেনি সাথে একটা সাবান গিফট',
     'অভিযোগ নেই বিনামূল্যে সাবান সবাই ঠিক গন্ধ ছিল।',
     '৩য় অডার করলাম সুন্দর প্যাকেট পাঠানো ধন্যবাদ',
     'সুন্দর প্যাকিং ভালো অবস্থা য় পেয়েছি',
     'দাম টা আগের এটু বেশি',
     'ভালভাবে পেলাম,ব্যবহার হয়নি',
     'ভালো নিবো',
     'প্যাকেজিং ভালো ছিলো সাথে ফ্রী ডেলিভারি',
     'লোকাল বাজার কম দামে পেয়েছি ডেলিভারি দ্রুত পেয়েছি',
     'কিনবেন না মশা মারার না সম্পূর্ণ অর্থের অপচয়।',
     ...]



Let's break down this code step by step in simple terms:

### 1. **`train_test_split`**
```python
Xtrain, Xtest, Ytrain, Ytest = train_test_split(xs, ys, test_size=0.25, random_state=0)
```

- **`train_test_split`** is a function from the **`sklearn.model_selection`** module in Python. It is used to split your dataset into two parts: one for training your machine learning model (the training set) and one for testing the model (the test set).

- **`xs`** and **`ys`** are the two lists that you want to split:
  - **`xs`** contains the input data (cleaned sentences, without stop words or punctuation).
  - **`ys`** contains the target labels (such as the category or class of each sentence).

### 2. **Arguments Passed to `train_test_split`**

- **`xs`** and **`ys`**: These are the data you're splitting into training and test sets.

- **`test_size=0.25`**: This argument tells the function to split the data so that **25%** of the data will be used for testing (`Xtest` and `Ytest`), and the remaining **75%** will be used for training (`Xtrain` and `Ytrain`).
  - In other words, if you have 100 sentences, 25 sentences will go into the test set, and 75 will go into the training set.

- **`random_state=0`**: This ensures that the split is **reproducible**. If you run the code multiple times, you will get the same train-test split every time. The number `0` is a seed value for the random number generator that `train_test_split` uses to shuffle the data before splitting it.

### 3. **Output Variables**

The result of the `train_test_split` function is **four variables**:
```python
Xtrain, Xtest, Ytrain, Ytest
```

- **`Xtrain`**: This is the **training input** data (75% of the sentences). It contains the cleaned sentences (without punctuation or stop words) that will be used to train the model.

- **`Xtest`**: This is the **test input** data (25% of the sentences). It will be used to test the performance of the trained model.

- **`Ytrain`**: This is the **training labels** (75% of the labels), corresponding to the sentences in `Xtrain`. These labels are the categories or classes associated with each sentence and are used to teach the model.

- **`Ytest`**: This is the **test labels** (25% of the labels), corresponding to the sentences in `Xtest`. These labels will be used to evaluate how well the model performs on unseen data.

### 4. **Viewing `Xtrain`**
```python
Xtrain
```
This line would display the contents of the `Xtrain` variable. It will show the **cleaned sentences** (after removing stop words and punctuation) that will be used for training the model.



```python
Xtrain, Xtest, Ytrain, Ytest = train_test_split(xs, ys, test_size=0.25, random_state=0)
Xtrain
```




    ['খুবি বাজে কোয়ালিটি ভাই ডিজাইন ভালো লাগছিল কিনছিলাম দাম দিয়েও বাট পুরাই বাজে',
     '"আলহামদুলিল্লাহ যেটা ওর্ডার করছি পাইছি।অল্প দামে খুবই ভালো একটা প্রোডাক্ট',
     'মাসে ৫/৬ ডেলিভারি ম্যান আসতো প্রডাক্ট ডেলিভারি দিতে,,এখন এক এক বারও আসে না,,কারণ ডেলিভারি চার্জ বাড়ানোর অর্ডারই না',
     'ভালো লেগেছে',
     'এক আরেক ৷ চিটিং',
     'খারাপ অভিজ্ঞতা',
     'দুইটা নিয়েছি সুন্দর',
     'হাইলি রিকমেন্ডড',
     'রমজানের টি-শার্ট দাম ৪৯৫ টাকা ঈদের বেড়ে ৫৯৫ টাকা সবচেয়ে বড় কথা এক সপ্তার কালার',
     'ধন্যবাদ ভাল মানের কাপড়',
     'স্নোবল ফেয়ারি লাইটস টা সত্যিই অসম্ভব সুন্দর কোয়ালিটি ফুল প্রতিটি লাইট ঠিক',
     'শুধু ধোঁকা বাজি',
     'পণ্যের দাম বেশি দেখিয়ে তারপরে ডিসকাউন্ট দেয়া একটা শুভঙ্করের ফাঁকি',
     'বছর পাব না',
     'কোয়ালিটি এভারেজ সেলাইগুলো দুর্বল ব্যাগটা সুন্দর বাচ্চাদের সাথে মানানসই ভারী না বহন বুদ্ধিমানের',
     'ওয়েবসাইট বন্ধ',
     'জিনিস পাব',
     'খারাপ মানের খাবার',
     'জঘন্য বার্গার সত্যি টাকাই লস',
     'দোকানের পরিষেবা অত্যন্ত সন্তুষ্ট',
     'আলহামদুলিল্লাহ প্রোডাক্ট ভাল ছিলাম ঠিক তেমনই পেয়েছি',
     'হতাশ কেনার সীমা সীমাবদ্ধ অর্ডার পারছি না',
     'টাকা ফেরত দেয়নি কিউকম',
     'প্রোডাক্ট এখনো হাতে পাই নাই রিসিফ দেখাচ্ছে ১২ তারিখ অর্ডার করছি এখনো হাতে পাইনি এতো লেট পারবেন?',
     'স্বাদহীন',
     'দিবেন ভাই বুড়া হয়ে গেলে???',
     'প্রিয় দারাজ কোম্পানি একজন নিয়মিত ক্রেতা আপনাদের অনলাইন সপ এর।',
     'জিনিস টা ভালোই বাট বড় করছিলাম একদম ছোট ভালোই',
     'কালার বিজ্ঞাপন পাইনি এগুলো এলোমেলো মাল দিয়ে কাষ্টমার বেঈমানী ঠিক আপনারা',
     'সত্য পণ্যের গুণমান সত্যিই ভিন্ন',
     'জনগণকে মূলা দেখান কোড দিলে রিচার্জ না এসব ফাইজলামি এখনই বন্ধ করুন,,,',
     'ফাঁদে পা বাড়াবেন না',
     'কম মূল্যে এতো কোয়ালিটি সম্পন্ন ভালো প্রোডাক্ট সত্যিই প্রত্যাশার অধিক সেইসাথে দ্রুতগতির ডেলিভারি,প্যাকেজিং স্টাফদের ব্যাবহারে সত্যিই মুগ্ধ',
     'সত্যিই প্রডাক্টটা সুন্দর পুরোটাই চামড়া',
     'সকল ফ্রি প্রোডাক্ট প্যাকেজিং প্রোডাক্ট কোয়ালিটি খুবই ভালো',
     'স্পীড শব্দ বেশি',
     'নিব লিংক করেনা',
     'দারাজ ডেলিভারি চার্জ অতিরিক্ত বেশি বর্তমানে',
     'ভোক্তা অধিকার মামলা',
     'তাজা না',
     'খুবই ভাল কোয়ালিটি  পড়তেও আলাদা মজা পাওয়া যাইয়া',
     'চমৎকার সুস্বাদু নকশা ভালবাসা',
     'ইভ্যালির আস্থা ছিলো',
     'এডভান্স হওয়া উচিৎ',
     'অধিকাংশই ভালো মানের',
     'পরবর্তীতে আবারো অর্ডার ইচ্ছা আছে।ধন্যবাদ',
     'সুন্দর কুর্তি',
     'অর্ডার প্রডাক্টস হাতে পেয়েছি কালার মান দুটোই ভাল সত্যিই ভাল লেগেছে',
     'পণ্য অর্ডার পেয়েছিলাম সত্যিই ডেলিভারি অন্যান্য শোরুমের তুলনায় দ্রুত গুণমান ভাল দাম যুক্তিসঙ্গত নয়',
     'দোকানের কম মূল্যে পেয়ছি',
     'বাংলাদেশের অনলাইন শপ মানে গলাকাটা দাম',
     'দারুণ জিনিসটা',
     'ফালতু ব্যবহারেই ড্রাইভারের দাত বাকা হয়ে',
     'খাদ্য হতাশাজনক',
     'অফার নেই যেগুলোতে ডিসকাউন্ট ছিলো সেগুলো অরিজিনাল প্রাইস দেখাচ্ছে',
     'শুভকামনা',
     'তোদের লজ্জা শরম নাই৷ ৷',
     'পছন্দ হয়েছে দাম বেশি',
     'অনকে দ্রুত পেয়েছি রাতে অর্ডার দিয়ে সকাল ১১টায় ডেলিভারি পাইছি প্রোডাক্ট অরিজিনাল বাকিটা ইউস বুজা',
     'অর্থহীন মূল্য ছাড় নিকটবর্তী আউটলেটে ৫ লিটার তেল নেই আফার শুরুর প্রহর',
     'এক মাস রিভিউ দিলাম দাম মোজা গুলো ভালো ছিল।',
     'ধন্যবাদ আপনাদের দ্রুত ডেলিভারী জন্যে',
     'আইটেম দীর্ঘস্থায়ী',
     'অনলাইনে কেনাকাটা বড় বোকামি।পন্যটা মনের না',
     'ডেলিভারি চার্জ ফ্রি',
     'ওডার ডেলিভারির খবর থাকেনা',
     'ঠিক পেয়েছি।সাথে gift পেয়েছি',
     'প্রোডাক্ট টা ভালো শুধু খারাপ লাগছে স্টেন্ড ফোন টাচ সেকিং করে।যেটা বিরক্তিকর এছাড়া প্রবলেম ফেইস নি',
     '২৭ তারিখের পার্সেল এখনো আসেনি',
     'প্রোডাক্ট মানসম্মত',
     'ব্লাক চেয়েছিলাম গ্রিন দিয়েছে',
     'সত্যিই ভাল গ্রাহক সেবা',
     'এভাবেই এগিয়ে হার মানলে না',
     'দের সার্ভিস টিম ভাল তেমনি অথেনটিক প্রোডাক্ট ব্যাপারেও সচেতন',
     'আপনাদের ডেলিভারি সিস্টেমটা ভালো',
     'দাম কমালে আরেকটি অর্ডার করতাম',
     'কোয়ালিটি ভালো দেখতেও সাইজে একটু বড় অর্ডার এক সাইজ ছোট অর্ডার কিরবেন',
     'আগামীতে আরো কেনাকাটা আপনাদের ইনশাআল্লাহ',
     'কালারও সেইম মানও ভালো আপনারাও পারেন।',
     'প্রডাক্টের মান খুবই বাজে পিকাবোমানুষের পকেট মারা ধরছে ভক্তা অধিদপ্তর জানাবো',
     'খাদ্য ভাল',
     'সালা ধান্দাবাজ',
     'খাবারটি খেতে সুস্বাদু কোয়ালিটি খুবই ভাল খেতে চাইলে অর্ডার',
     'অর্ডার করেছি ফোন নি  অর্ডার কনফার্ম না বুঝতে পারছি না',
     'বিরুদ্ধে একশন দরকার',
     'মাছটা পছন্দ হয়ছে',
     'মিলিয়ে ভালো',
     'মাশাআল্লাহ সুন্দর Product দাম হিসেবে ঠিক সবাই',
     'ভাল খাবার',
     'সুন্দর',
     'মত ট্রাস্টেড ই-কমার্স সাইট বাংলাদেশে কমই',
     'কালো পোষাক আরো সুন্দর',
     'মত মানুষের টাকা মেরে না খাইলেও পারতেন',
     'পন্য এখনো আসেনি',
     'চিটার বাটপার দের খোঁজ খবর নাই',
     'বেচে থাকুক ইভ্যালি বেচে থাক মানুষের সপ্ন',
     'অফার শিহরিত',
     'মার্চ অর্ডার করেও প্রোডাক্ট পাইনাই',
     'ফ্যানটা স্থীর না',
     'আপনাদের শোরুমের ঠিকানা সরাসরি কিনতে চাচ্ছি',
     'উপকার',
     'কেমন রসিকতা',
     'পরিবেশিত পণ্যের গুণমান সন্তুষ্ট',
     'এইটা কখনোই হয়না ডেলিভারী ম্যান গুলো ফাজিল আপনাদের প্রোডাক্ট ডেলিভারী ফেইল দেখিয়ে পরের ডেলিভারী করে।',
     'প্রোডাক্টটি অর্ডার কনফিউশনে ছিলাম কিনা আসলে যেমনটা ভেবেছিলাম সেরকম না প্রোডাক্টটা ভালো,',
     'প্রত্যাশা দাম গুণ অবিশ্বাস্য',
     'অর্ডার কেনসেল কেনো',
     'অনলাইনে অর্ডার লাভ সঠিক সময়ে ডেলিভারি না পাই  ঢাকার ডেলিভারি সপ্তাহখানেক সময় লাগে এরকম সার্ভিসের দরকার নেই ',
     'সিলেট কিনতে পারছি না',
     '২০১৪ সাথে আছি।কিছু অভিযোগ ছিলো করলাম না',
     'আরো সুন্দর আশা করেছিলাম',
     'আমিও আপনাদের শার্ট কিনেছি মন মত হয়নাই',
     'আকর্ষণীয় ছবি দেখিয়ে নকল / নিন্ম মানের পণ্য সরবরাহ',
     'একটাও ভালো না',
     'বাটপারি করেও অনায়াসে ব্যবসা করতেছে',
     'বাজে কাপড়! ডেলিভারী সার্ভিস একটুও ভাল না অর্ডার দেই একটা,পাই আরেকটা চেঞ্জ দেওয়ার নামে ফটকামী একচুয়ালি দে ডোন্ট নো হাও টু রান আ বিজনেস!',
     'গুণমান ভাল প্যাকেজ খারাপভাবে ক্ষতিগ্রস্ত খড় বিচ্ছিন্ন',
     'ভালো প্রডাক্ট সন্তোষজনক',
     'লোভে অর্ধেক দামে না কিনে সঠিক দামে প্রোডাক্ট কিনুন অবশ্যই ভালো সেবা পাবেন',
     'আলীএক্সপ্রেস সস্তা নয় আইটেম নরওয়ের বেশি দামি গুণমান সবসময় ভালো নয় কল্পনা করুন!!',
     'গত তিন বছর নিচ্ছি আলহামদুলিল্লাহ সার্ভিসে সন্তুষ্ট',
     'সাউন্ড বক্সের দাম বেশি',
     'লেমন ফ্লেইভর এনার্জি ড্রিংকস ভালই ধন্যবাদ',
     'মূল্য সীমার সাথে তুলনা করুন পণ্যের গুণমান ভাল ততটা',
     'ডেলিভারি অপশনে পিক আপ কালেকশন পয়েন্টে ক্লিক সার্ভার সমস্যা কানেকশন সমস্যা সমস্যা সমস্যা দেখিয়ে দিচ্ছে।',
     'যেগুলো দরকার গুলো নেই',
     'কাস্টমার কেয়ার কল রিসিভ না কেনো',
     'কম্প্লেইন নাই',
     'নিব লোকেশান প্লিজ',
     'কোয়ালিটি সম্পন্ন প্রোডাক্ট ধন্যবাদ',
     'দাম এক কথায় জঘন্য',
     'আসলেই প্রোডাক্ট গুলা ভালো কোয়ালিটির',
     'ব্যক্তিগতভাবে পছন্দ করেছি',
     'আলহামদুলিল্লাহ দারাজ ফেবরিলাইফ শার্ট পাইলাম',
     'বাংলাদেশ চেয়ে বাজে আন লাইন একটিও নাই',
     'দারাজ ভাল না বিশাল ঠকবাজ-আমার পুরাতন টেবিল দিছে-এখন রিটার্ন চেঞ্জ দেয় না',
     'সত্যি দাম সাথে তুলনা অসাধারণ মানের শুভ কামনা রইলো',
     'দীর্ঘ হায়াত কামনা',
     'অডার জিনিস আসলো না কেন??',
     'পাঞ্জাবি শান্তি',
     'আলহামদুলিল্লাহ ভালোই আরো এক সাইজ বড় অর্ডার দেয়া ছিলো',
     'ভাই ঘড়ির মান অত্যান্ত খারাপ অবস্থা,সমস্থ পার্টস খুলা',
     'প্রোডাক্ট ভালো না কালার শাদাশে হয়ে যায়',
     'পিছনে ভাঙ্গা,ফালতু একটা জিনিস পাতলা প্লাস্টিক',
     'ভালো লেগেছে',
     'মোটামুটি ভালো পণ্য  24 টাকায় কিনেছি  আশা ভবিষ্যতে আরো কম দামে পাব',
     'কাপড় কোয়ালিটি ১০০তে ১০০% ভালো',
     'সকল শুভ কামনা',
     'দাম টা একটু বেশি চাইতেছে',
     '"হ্যাঁ অর্ডার করলাম এক কালার পেলাম আরেক কালার সেবার মান ভালো বিজ্ঞাপন দেন।',
     'আসলে প্রোডাক্ট দেয়??',
     'পণ্যের গুণমান পরিষেবা অনুসারে মূল্য একেবারে যুক্তিসঙ্গত',
     'ইজি বাজে একটা ব্যন্ড কালার থাকেনা।টি সার্ট সার্ট',
     'কাপড়ের কোয়ালিটি ভাল ধাম আনুয়ায়ী ঠিক সাইজ চাইছিলাম পাইছী',
     'অনলাইনে বিশ্বস্ত একটা কোম্পানি পেলাম ফ্রান্সে থাকি আরো টি-শার্ট অর্ডার করবো ইনশাআল্লাহ নিঃসন্দেহে',
     'আপনাদে পন\u200d্য ভালো লাগছে 100% কোয়ালিটি সপ্মন\u200d্য সাতটা শার্ট আনলাম চোখবুজে বিশ্বাস যায় আশা আপনারা বিশ্বাস টুকু দরে রাখবেন',
     'একটা নিয়েছি ভাল প্যান্ট আরামদায়ক',
     'দাম অনুযায়ী ভালো প্রডাক্ট ছোট খাটো চালিয়ে নেয়া প্রডাক্ট ঠিক পেয়েছি স্কু বিটের পাশে প্রতিটি বিটের নাম লেখা থাকার কথা এটার নাম নেই!',
     'একটা প্যান্ট কিনে বাসায় দেখি কোমোরের বোতাম ভাংগা শব স্টীকার লাগানো এখনো কিন্ত মেমোটা ভুল দেয় নাই আমিও খেয়াল নাই',
     'দিলাম সাদা সোল সাদা ফিতার আসলো পুরা কালো কথা হ্যা পুরাই পালতু',
     'সুন্দর প্রোডাক্ট,,দাম অন্যান্য পেজ কম, ফিটিং সুন্দর হইছে,,স্যাটিসফাইড',
     'খাবারের মান ভাল',
     'পরিমাণ ভালো পিস মাটন',
     'ওয়াও এতো বড় মাছ',
     'বাহ্',
     '১ দিনে প্রোডাক্ট পেয়েছি কোয়ালিটি অসাধারণ সবাই কিনতে',
     'ছবিতে দেখেছি তখনই পেয়েছি',
     'জিনিসপত্র ফেরত না পোকামাকড়ের আক্রমনের দায়ে সেগুলো জব্দ কাস্টমস',
     'বক্স কেটে আসল প্রোডাক্ট সরিয়ে ভুল প্রোডাক্ট ঢুকিয়েছে এটাও নষ্ট প্রোডাক্ট',
     'ফ্রি ডেলিভারি ',
     'দাম সুন্দর ভালো প্যাকিং ভালো সবদিক ভালো আপনারা',
     'মান খারাপ',
     'দারাজের মত হবেনাতো অর্ডার দিলাম একটা দারাজ দিছে আরেকটা নাতো',
     'খাবার স্বাদহীন',
     'দাম বেশী',
     'অভিযোগ নেই বিনামূল্যে সাবান সবাই ঠিক গন্ধ ছিল।',
     'কোয়ালিটি নেই',
     'তারিখ দেখেন পণ্য দিসে নিচের স্টিকার টা দেয় নাই পুনরায় অর্ডার গেলাম দাম আরো বেশী দেখায়',
     'হেলমেটটা ছবিতে সুন্দর বাস্তবে ভালো লাগে নাই',
     'পন্যের মান ভাল  প্রাইস একটু হাই পন্যের মান দেখলে সেটাকে যথাযথই ',
     'সাইজ ছোট,মানে ভালোই',
     'নিশ্চয়ই জায়গাটায় আসব',
     'আলিশা মার্ট দেশ সেরা ইনশাআল্লাহ একদিন',
     'ভালো কালেকশন',
     '"আলহামদুলিল্লাহ্ দেখেই বোঝাা আরাম দায়ক ',
     'দারাজের বাটপারি বেড়েই চলছে',
     'বাজে রেইনকোট এক পড়েছি পেন্ট নিছ ছিড়ে',
     'কল রিসিভ করাও বন্ধ দিয়েছেন',
     'পাঞ্জাবির কোয়ালিটি ভালো সার্ভিসও ভালো ছিলো',
     'পন্যটা আজকে পৌঁছেছে..অনেকদিন লেট আসছে পন্যটা',
     'পণ্য অর্ডার ডেলিভারি সঠিকভাবে আসে না',
     'খাবার ভাল একটু বেশি টাকা প্রস্তুত',
     'দারাজে অর্ডার করেছি ২৬ তারিখে,,, আজকে না পাই আপনাদের পন্য গ্রহণ/রিসিভ করবো না',
     'শার্টটা হেব্বি সুন্দর',
     'ন্যাড়া একবারই বেল তলায়',
     'টাকা পণ্য দেয় নাই',
     'একটা নিছি খুবই ভালো মানের প্রোডাক্ট ইনশাল্লাহ একটা নেব',
     'একটা অর্ডার করেছি একটা দিছে মান খুবই খারাপ',
     'দাম কেও রাখে',
     'প্রডাক্ট ডেলিভারীতে দেরি হয়েছে',
     'চিটার বাটপার',
     '9 তারিখের অর্ডার করলাম এখনো পাচ্ছিনা',
     '২০১৪ সাল পযন্ত পছন্দের ব্রান্ড ইজি ফ্যাশন লিমিটেড',
     'আজেই আপনাদের পণ্য হাতে পেয়েছি পণ্য নের কোয়ালিটি খুবই ভালো আরামদায়ক',
     'সাত এখনো ডেলিভারি নাই টাকা ফেরত',
     'আপনাদের প্রোডাক্ট পেয়েছি চেয়েছিলাম ঠিক তেমনি পেয়েছি,ধন্যবাদ আপনাদের মনের প্রোডাক্ট দিবার',
     'এরকম খারাপ অভিজ্ঞতা প্রোডাক্ট হয়নি',
     'দারাজে কমে পাওয়া যায় ডেলিভারি ফাস্ট',
     'দারুণ কালেকশন',
     'এইগুলা চোর বাটপার',
     'আপনাদের প্রোডাক্ট পেয়ে সন্তুষ্ট আরো নিবো শীঘ্রই ইনশাআল্লাহ কোয়ালিটি রাখবেন আশা',
     'দ্বিতীয়বারের মত কেনাকাটা প্যাকিং ভাল দ্রুততম সময়ে ডেলিভারি প্রোডাক্ট কোয়ালিটি ভাল পেয়ে বরাবরের সন্তুষ্ট',
     'হতাশাজনক কোথাও কেনাকাটা করছি',
     'আমিও নিলাম খুবই ভালো',
     'ওডারটা পেলাম না',
     'স্টুডিও এক্স সাশ্রয়ী মূল্যে সেরা মানের পুরুষ পণ্য সরবরাহ',
     'আপনাদের কাস্টমার কেয়ারে আজকে 10/12 কল দিয়েও না',
     'বাজে পণ্য অর্ডার টাকা শেষ',
     'মানুষ অনলাইনে কেনাকাটা উৎসাহ ',
     'ভাই সামান্য গ্রোসারি আইটেম পারলেন না ৩ মাসে',
     'মেশ কাপড়ের পড়েও আরাম দেখতেও খারাপ না শুধু কাপড় পাতলা ',
     'বাংলাদেশের ই-কমার্স মানে কিনলেন ত ঠকলেন',
     'প্রয়োজনীয় জিনিসের বেশির ভাগই স্টকআউট হয়ে',
     'ভালভাবে পেলাম,ব্যবহার হয়নি',
     'ড্রেস টা খুবেই সুন্দর',
     'ছ্যাচরামি রে ভাই',
     'দরিদ্র মানের মোটর শক্তি পূর্ণ নয়',
     'মাটন পিস টা ওয়েল কুকড',
     'নিয়েছি ভালো প্রোডাক্ট',
     'প্যাকেজিং ভালো ছিলো সাথে ফ্রী ডেলিভারি',
     'ভালো দামটা একটু বেশি',
     'আপনাদের অ্যাপস আরো উন্নত',
     'তারাতাড়ি পেয়েছি একদম বাড়িতে দিয়ে ভালো মিলিয়ে',
     'বাংলাদেশে অনলাইন শপিং সবাই এইরকম ঠকেছেন',
     'পন্য এখনো আসেনি',
     'কর্মীরা সত্যিই বন্ধুত্বপূর্ণ',
     'ইদের অর্ডার করছি,৭ জুলাই দেওয়ার কথা ছিলো আজও খোঁজ নেই আদোও পণ্য পাবো?',
     'ওরে চিটার',
     'চমৎকার খাবার',
     'দাম বেশি চাচ্ছেন',
     'এমটি হেলমেট কালেকশন আরো বাড়াতে',
     'পণ্য পরিষেবা শান্তিপূর্ণভাবে সন্তুষ্ট আশা গন্তব্যে পৌঁছে যাবেন (ইনশাআল্লাহ',
     'ভালো নিবো',
     'প্যাকেজিংটা সুন্দর',
     'যতসব ফালতু একটা চাইলে আরেক টা দেয়',
     'সার্ভিস দ্রুত',
     'দাম অনুযায়ী মানসম্মত শার্ট',
     'এবারের গিফট গুলো ভাল ধন্যবাদ চালডাল',
     'অর অপেক্ষা',
     'প্রডাক্ট গুলো পাওয়ার সম্ভাবনা ভাই',
     'বেশি দাম হয়ে যায়',
     'স্লাইস ছোট খারাপ স্বাদ',
     'অন্যান্য পেইজের তুলনায় দাম অনেকটাই কম সার্ভিস অনেকটাই ভালো অনেকটাই ভালো ',
     'এক্সক্লুসিভ কালেকশন',
     'কিনে জয়ী',
     'এভাবে হয়রানি ভোক্তা অধিকার আইনে কাষ্টমার হয়রানি অভিযোগ জানাতে বাধ্য হবো',
     'আজকে ঈদের  দারাজ ভালো সারপ্রাইজ অফার আশা করছি',
     'দিলো এইটা এসব ভাই,,ভালো না',
     'আজকে প্রোডাক্ট পেয়েছি,ভালো লাগছে',
     'আলহামদুলিল্লাহ ভালো',
     'বিজনেস চালিয়ে যান.. ইনশাআল্লাহ বিজয় সুনিশ্চিত..',
     'প্রোডাক্ট ড্যামেজ',
     'হাস্যকর ব্যাপার এখনো আপনারা অফার দিচ্ছেন',
     'আসসালামু আলাইকুম আপু জামা অডার করেছি',
     'এসব বাটপারদের নামে মামলা',
     'প্রডাক্ট না',
     'কোয়ালিটি দাম ঠিকই',
     'একদম বাজে প্রোডাক্ট',
     'ঘড়ি দিছেন ভাই টাইম যায়না।এখন করব',
     'ভালো মানের প্রোডাক্ট দেওয়ার ধন্যবাদ আপনাদেরকে,অনেক দূর এগিয়ে যাক',
     'বাটপারের দল বাইকের লিস্ট দে',
     'দারুন সার্ভিস ভাই আপনাদের',
     'অর্ডার একটু ভয়ে ছিলাম প্রেডাক্ট হাতে পেলাম ভয় কেটে',
     'দেশি প্রতিষ্ঠান বেচে থাকুক',
     'খুবই ভালো প্রোডাক্ট অরজিনাল প্রোডাক্ট',
     'এক টা নিয়েছি ভালো পছন্দ হয়েছে  দাম টা একটু কম মেরুন কালার টা নিতাম ',
     'ভালো প্রোডাক্ট',
     'দুঃখজনক',
     'আলহামদুলিল্লাহ ভালো জিনিস হাতে পেয়েছি ধন্যবাদ..',
     'পণ্যের মানও ভালো আশা করছি দাম কমে পাবো',
     'হাস্যকর দাম',
     'পণ্যের গুণমানও শীর্ষস্থানীয়',
     'আলহামদুলিল্লাহ ২৪ ঘন্টারও কম সময়ের ডেলিভারি পেলাম কোয়ালিটি সন্তুষ্ট আলহামদুলিল্লাহ',
     'মিনিমাম ১০% ডিসকাউন্ট দেয়া',
     'ভালো লাগছে পন্য হাতে পেয়ে।।',
     'আপাতত ঠিকঠাক দেখি কেমন  প্যাকেট সুন্দর পাইছি।',
     'ফ্যান এক জাগায় রাখলে ঘুরে যায় স্পিড সমস্যা',
     'খাবার দারুণ বার্গারটা অসাধারণ',
     'প্রিয় দারাজ খাগড়াছড়ির ডেলিভারির বয় গুলো মরছে একটু খোজ দেখেন গত পরশু প্রোডাক্স ডেলিভারি দেওয়ার নাম গন্ধ নাই ',
     'মিথ্যা বিজ্ঞাপন ৩ টা পোডাক্ট ফ্রি ডেলিভারি কই।',
     'ভালো করছি',
     'পিজাটা চমৎকার',
     'খবর বার্তা নাই তাড়াতাড়ি ডেলিভারি দেন..',
     'নিব দাম টা',
     'মালের অর্ডার দিয়েছি মাল পাইলাম না ভাই মাল কয় তারিখে পাইনি',
     'কিনলেই ঠকতে',
     'অর্ডার নিয়েছেন কখন পাবো বলছেন না ইনবক্সে রিপ্লাই না',
     'শালারা ধোকাবাজ,,,খালি প্যাকেট দিয়েছ',
     'সবাই সুন্দর সুন্দর কথা টাকার লোভ সামলাতে না',
     'ডিজাইন গুলো বড়দের ভালো চলত',
     'খাবারটি ভাল',
     'বাটা জুতোতে গ্রেড গ্রেডের দুইধরনের জুতা আছে।বি গ্রেডের জুতো জুতা গ্রেডের জুতো একটু নিম্নমানের',
     'ভালো প্রোডাক্ট ছিলো প্যেকেজিং ভালো ছিলো রিকমেন্ডড ',
     'এটার সেলাই বিষয়ে সেলারকে একটু সচেতন কাপড় সেলাই মত বডার দিলে ভাল',
     'দুর্বার গতিতে এগিয়ে যাও',
     'অর্ডার ডেলিভারির খবর নাই ফেইলড এ্যাটেম্পট দেখাইলো খেয়াল খুশি মত করতেসে',
     'পণ্য ডেলিভারি পাচ্ছি না',
     'পুরাই চোর সালারা',
     'ধন্যবাদ এডমিন পেনেল',
     'বেশি খাইতে জায়েন না',
     '"চমৎকার শুভ কামনা',
     'খাবার সত্যিই ভাল জিনিস উপভোগ করেছি',
     'মান ভালো সবাই কিনতে',
     'ফ্যানটার স্পীড ভালই ছোট ফ্যান হিসেবে খারাপ নয় স্পীড কমানো বাড়ানোর অপশন নেই ঠিক ওয়াটের ফ্যান description লেখা নেই ধন্যবাদ সেলার ভাই',
     'দাম টা আগের এটু বেশি',
     'অনলাইনে দাম বেশি',
     'অথবাতে অর্ডার ভুক্তভুগি ৬০ ডিম অর্ডার করেছিলাম ডিমগুলো মারাত্মক তিতা স্বাদের',
     'ধূর খুবই বাজে পণ্য',
     'গেঞ্জিটা সুন্দর',
     'ভাল অ্যান্ড্রয়েড ফোন অল্প দামের পছন্দ হয়েছে আপনারা চাইলে',
     'মামলা আমিও করবো লাখ টাকা না পাই',
     '"আলহামদুলিল্লাহ সময় হাতে পেয়েছি প্রোডাক্টি সুন্দর সুস্বাদু।',
     'দাম কারন',
     'ভালো পন্য',
     'দারাজ ভুল প্রডাক্ট দিয়েছে রিটার্ন নিচ্ছে না রিফান্ড না সেলার মেসেজ দেখেও রিপ্লাই দিচ্ছে না',
     'দাম বেড়ে',
     'এছাড়া দাম অনুযায়ী কাপড়ের মান মোটামুটি ভাল ভাবছিলাম কাপড় একটু মোটা',
     'জিনিস অর্ডার চাই পারছি না প্লিজ কাইন্ডলি জানাবেন',
     'ধৈর্য্য না',
     'ভালো সময়োপযোগী পদক্ষেপ',
     'পছন্দ কিনলে ঠকে যাই ঘড়িটি হাতে কিনে জিতলাম',
     '১০ এখনো প্রোডাক্ট পাই নাই',
     'পারফেক্ট সাইজ আরামদায়ক ৷ সন্তুষ্ট ৷ আপনারা নিশ্চিন্তে কিনতে',
     'শাড়ি টা চাই চাই',
     'স্মার্ট ওয়াচ অসাধরণ পারফম কর\u200cছে মাই\u200cক্রোফ\u200cনের সাউন্ড কোয়া\u200cলি\u200cটি ভা\u200cলো চোখ বন্ধ ক\u200cরে পা\u200cরেন',
     'হেডফোনগুলি দুর্দান্ত দাম সাউন্ড কোয়ালিটি ভাল',
     'আলহামদুলিল্লাহ্\u200c ঠকিনাই যেমনটা আশা করেছি চাইতে ভালো মধু',
     'বাচ্চার হেলমেটটি সুন্দরী',
     'প্যাকিং জোস ছিলো ২ টাকা কমে পেয়েছি',
     'উপকার হইলো',
     'পন্যের চেয়ে ডেলিভারি চার্জ বেশি সেইটা',
     'খুবই ভালো একটা অফার ছিলো পেরেছি',
     'মাশআল্লাহ আল্লাহামদুলিল্লাহ',
     'বাংলাদেশে ই-কমার্স সাইটের প্রসার চাই',
     'দাম অনুযায়ী হাতঘড়িটা ভালো',
     'আলহামদুল্লিহা ভালো product পেয়েছি।সেলার service ভালো।তারা ভালো chat response করে।T-Shirt টা পছন্দ হয়েছে।Fabric quality যথেষ্ট ভালো',
     'মিয়া ভাই ডেলিভারি চার্জ কমান,,,,এইরকম কইরা ডাকাতি ভালো না',
     'ভাল খাদ্য অসাধারণ সেবা',
     'অনলাইন প্রোডাক্ট হিসেবে ভালো',
     'ভয়ানক অভিজ্ঞতা',
     'আলহামদুলিল্লাহ কোয়ালিটি ভালো',
     'ভালো একটা প্রডাক্ট সত্যিই ভালো লেগেছে',
     '"গুনে মানে সুগন্ধিতে ভরপুর',
     'এক কথায় অসাধারণ',
     '১২০ টাকা ছাড় মেয়াদ পর্যাপ্ত খুবই আনন্দিত হয়েছি',
     'আপনাদের সুজ বাজে',
     'খুবই ভাল মানের ফ্যান ধন্যবাদ সেলার',
     '১০০০ অডার দিব দোকানের',
     'জামাটা ছবিতে যেরকম দেখছি সুন্দর  কাপড়ে মান ভালো এতো ভালো মানের পোশাক দেওয়ার অসংখ্য ধন্যবাদ',
     'ছেলে টা ভালো, ব্যাবহার ভালো সময় মতোন পন্য হাতে পৌছে দেয়।',
     'শার্ট রং নষ্ট হয়ে যায়',
     'এক কথায় প্রাইজ রেঞ্জের ভালো প্রডাক্ট আপনাদের উজ্জ্বল ভবিষ্যৎ কামনা',
     'খুবি ভালো প্রোডাক্টটি ধন্যবাদ',
     'যাই ম্যাসেজ রিপ্লাই দিবেন',
     'একটা ব্লুটুথ হেডফোন অর্ডার করেছিলাম পাইলাম না',
     'আগের দারাজই ভালো ছিলো, দারাজে অর্ডার করিনা একমাত্র ডেলিভারি চার্জ বাড়ানোর কারণে।।',
     'বাটপার সেলার কথা মিল নাই',
     'কয়েকদিন দারাজে প্রডাক্ট অর্ডার পারতেছি না কিসের প্রবলেম বুঝতেছিনা',
     'সততার অভাব',
     'মাশাআল্লাহ ভালো টি-সার্ট টা',
     'মাল হাতে হতাশ হলাম',
     'অর্ডার দিলাম কনভার্স দিলো লোফার কেমন ভুল অর্ডার',
     'নাইচ',
     'ভালো আরামদায়ক',
     'কাপড়ের কোয়ালিটি বেশি ভালো না',
     'মাস্ক গুলো একটা ৫০ টাকা আপনারা টাকা বেশি নিচ্ছেন না আপনাদের এইটা ধরনের প্রতারণা',
     'সার্ভিস খুবই ধীরগতির',
     'সুন্দর কালেকশন',
     'খুবি খারাপ অবস্থা',
     'মঙ্গল গ্রহের মোবাইল? জীবদ্দশায় ডেলিভারি না',
     'আপনাদের রিটার্ন পলিসি খারাপ',
     'প্রধান সমস্যা ব্যবস্থাপনা',
     'ভালোই হয়।ব্যবহারে বুঝা বাকিটা',
     'গুড',
     'গতকাল পণ্য অর্ডার বিকাশে পেমেন্ট করি।কিন্তূ ক্যাশব্যাক পাইনি',
     'ছোট ভাইকে গিফট নিয়েছিলাম দিয়েছি পছন্দ ধন্যবাদ সেরার',
     'জিনিসটা মোটামুটি',
     'আপনারা উওর না  ইনবক্স ক\u200cরে\u200cছি  রেসপন্স নাই',
     '৭ দিনও নাই ফ্যানটা চালাচ্ছি হঠাৎ বন্ধ হয়ে গেলো',
     'টাওয়েল টা ভাল লেগেছে',
     'ট্রাই করেছি অনেকগুলো সার্ভিস ভালো না কারোই',
     'সুন্দর',
     'চিকেন মাঞ্চুরিয়ানের স্বাদ খারাপ',
     'বিশ্বাস না প্রতারক',
     'খারাপ সার্ভিস বাসের টিকিট শোওজ',
     'প্যান্ট ফেটে বসার সাথে সাথে',
     'চমৎকার অনুভূতি ধন্যবাদ চালডাল',
     'ভালোই বাটপার',
     'মডেল ২/৩ ধোয়ার উঠে সম্ভাবনা',
     'একদমই ভালো নয়',
     'ভাল অভিজ্ঞতা',
     'একদম বাজে একটা স্প্রে ডেট নাই',
     'দারাজে পেমেন্ট পণ্য দেয় না',
     'আশা ভবিষ্যতে পন্যের মান ঠিক আরো বহুদূর এগিয়ে যাবেন।শুভ কামনা',
     '4 টা নিয়েছি।খুবই ভালো প্রোডাক্ট',
     'চুলের জেলটা অরিজিনাল',
     'দামটা একটু বেশি',
     'প্রোডাক্ট মন টা উদাসীন',
     'সত্যিই চমৎকার কাপড়',
     'বেল্টা ভাল',
     'সমস্যার সম্মুখীন হয়েছি বেশি হয়রানি হয়েছি',
     'বেস্ট কোয়ালিটি',
     'ওর্ডারকৃত টিভিটি ডেলিভারির ব্যবস্থা নেওয়ার তানাহলে ভোক্তা অধিকার আইনে মামলা করব',
     'বিকল্প এখান কেনাকাটা না প্রচুর প্রতারক বিক্রেতা পদ্ধতিগুলি চতুর',
     'দাম লেখছেননা',
     'প্রতিটি প্রোডাক্টের গুনগত মান খুবিই ভালো',
     'মাশাআল্লাহ',
     'পাই নাই রিচার্জ করছি পাই নাই',
     'প্রতিটি পন্য অসম্ভব সুন্দর',
     'মেরিকোর প্রাডাক্ট মান সম্পর্কে আলাদাকরে বলার না',
     'রিজেনবল প্রাইসে অরিজিনাল লেদার সত্যি আরামদায়ক আপনাদের প্রডাক্ট',
     'বাক্সটি ভাঙ্গা স্ক্রুগুলো হাড়িয়ে যাওয়ার সম্ভাবনা বেরে',
     'বিউটিফুল সমাস',
     'এক টাকাও ডিস্কাউন্ট না',
     'অল্প সময়ে পেয়েছি কাভার বক্স হয়নি জানলে ওডার টা করতাম না স্রার দিলাম',
     'এগিয়ে যাও',
     'ভাই পন্য গুলো পাবো,,,,,',
     'গত পরশুদিন দারাজে পণ্যের অর্ডার ভিতরে ৩ টা স্যাভলন সাবানের অর্ডার পণ্য রিসিভ প্যাকেট খুলে ২ টা স্যাভলন সাবান পাই অত্যান্ত অনাকাঙ্ক্ষিত দুঃক্ষজনক',
     'চিটিং বাজ',
     'মোজা গুলো ভালই  ভালো লাগছে',
     'প্রোডাক্টটি ভালোভাবেই দ্রুত পেয়েছি',
     'ভয়ঙ্কর অভিজ্ঞতার কয়েকজনের সাথে কথা বলেছি তারাও জিনিসের মুখোমুখি',
     'নিবো',
     'মানসম্মত খাদ্য অসাধারণ পরিবেশ সর্বোত্তম সেবা',
     'ই-কমার্সের সুদিন ফিরতেছে আলহামদুলিল্লাহ',
     'রাত ১২টাই অর্ডার করলাম,১দিনে দেওয়ার কথা,,এখনো পাইনি...',
     'আপনাদের অনলাইন সাইট না ফ্যান নষ্ট',
     'প্রোডাক্ট টা খুবই ভালো প্যাকেটিং টা সুন্দর ঘড়িটা সুন্দর এক কথায় অসাধারণ',
     'সত্যিই দারুণ ভাবি নাই ভালো প্রচন্ড ভালো জামা নিয়েছি ভালো এসেছে প্রচন্ড খুশি আপনারা চাইলে ধন্যবাদ দারাজ',
     'পিকাবো গুলা ডিভাইস কেনা হইছে খারাপ পাই নাই সার্ভিস ভালো',
     'ধন্যবাদ তোমাদেরকে',
     'কিভাবে কেনাকাটা করবো যতবার করেছি ততবারই ঠকেছি',
     'ভাই চাইছি হলুদ দিছে কালো এইটা হইলো হলুদ জুতা লাগবে',
     'বিউটিফুল',
     'প্যাকেজিং + উপরের মোড়ক এক কথায় অসাধারণ',
     'আলহামদুলিল্লাহ  ভ্যালো মোটো পেইচি... প্যাকেজিং ভ্যালো সাইলো',
     '30 মার্চের দেয়ার কথা 30মার্চ এখনো মেসেজ পেলাম না',
     'আলহামদুলিল্লাহ ঠিক চাইছি পাইছি কালার সেম পাইছি',
     'মেসেজ যায়না',
     'ধন্যবাদ এডমিন প্যানেল',
     'আলহামদুলিল্লাহ ডেলিভারি ফাস্ট পেয়েছি প্রোডাক্ট ভালো',
     'খুবই ভাল সার্ভিস ৩ শপিং করেছি একদম জাস্ট টাইমের ডেলিভারি দিয়েছে ফুললি স্যাটিসফাইড',
     'স্টারটেক প্রত্যেকটা সেলসপারসনের বিহেভিয়ার ভালো',
     'আপনাদের ধন্যবাদ',
     'আলহামদুলিল্লাহ টেবিলটা সুন্দর ছেলে পেয়ে অনেকটাই খুশি সবাই',
     'Quality অনুযায়ী দাম বেশী',
     'টেক্সট দিয়ে রাখছি রিপ্লাই নাই',
     'ইভ্যালির চাচাতো ভাই না',
     'ইজি ফ্যাশন সময় অসহায় মানুষের পাশে প্রত্যাশা দোয়া রইল ইজি ফ্যাশন',
     'ভাওতাবাজী এখনো ছাড়বা না',
     'আপনাদের বিশ্বাস যায়',
     'প্রধান উদ্দেশ্যই কাষ্টমার ঠকানো',
     'কালার ভুল',
     'সুন্দর ড্রেস',
     'দাম কনসিডার জোড়া ৩০ টাকা পড়েছে ভালো',
     'ধন্যবাদ Deen এতো সুন্দর পণ্য পৌছে দেওয়ার এগিয়ে যাক সামনের কামনা',
     'সেলারের খুবই ভালো লাগছে কালার চেয়েছি পেয়েছি',
     'দারাজ আমাদেরকে বোকা বানিয়ে ডেলিভারি চার্জ বেশি নিচ্ছে.আমি গত পরশু অর্ডার করেছিলাম এক্সপ্রেস ডেলিভারি চার্জ দিয়ে দারাজ এখনো ডেলিভারি নাই',
     'অত স্পিড নাই',
     'বডিতে সমস্যা',
     'টি-শার্ট গুলো পছন্দের মিলিয়ে কম দামে অসাধারণ প্রোডাক্ট',
     'দাম বেশি',
     'এক কথাই দারুণ  পড়লে বুঝা যায় না সেলার আন্তরিক  আগেও সেলারের পন্য কিনেছি',
     'সুন্দর কালেকশন',
     'মেশিন টা ডারাজ অনলাইনে নিছিলাম কিছুদিন বাট মেশিন টা ভালো না',
     'ফাজলামি পেলে না',
     'দারুন খাবার বিশেষ চিকেন মশলা',
     'চামড়া জানি একটু লোকোয়ালির',
     'পন্যটি ভালো ছিলো টা চেয়েছি টাই পেয়েছি ইস্কিনপটেক্টর টা দিয়ে দিলে ভালো',
     'খাবারটা অসাধারণ',
     'দারাজ এপটা খুলতে পারতাছি না',
     'ভাল প্রোডাক্ট পেয়েছি ইন্ট্যাক্ট প্যাকেট প্যাকেজিং ভাল',
     'অর্থের অপচয়',
     'উনাদের অত্যন্ত ভাল শুভ কামনা রইলো আপনাদের',
     'মাশআল্লাহ্ চমৎকার',
     'আপনাদের একটা রিভিও ভাল দেখলাম না…',
     'জীবনে কোনদিন ভাবি নাই বড় খাব',
     '১০০০ টাকার পেমেন্ট করছি কুপন পায়নি',
     'দারুণ প্যাকেজিং',
     'বয়কট দারাজ',
     'উপকার',
     'তুলনামুলক দাম একটু বেশি হয়েছে দামে বাসায় প্রডাক্ট হাতে পেয়ে অত্যন্ত খুশি',
     'তাওয়াল ভালো বিশেষ গাড়ি মোছার',
     'অর্ডার পণ্য পাওয়ার নিশ্চয়তা নাই,এভাবে কয়দিন চলবে',
     'কাপড়টা বেশি সুবিধা না',
     'কোড না',
     'সত্যি ২৯৯ টাকা তে অসাধারণ মার্কেটে জিনিস ৫০০/৬০০ চাইবে ভালো হয়েছে',
     'আপনাদের পাঞ্জাবির কাপড় ভালো মানের',
     'শাড়ীর কালার টা খুবই সুন্দর',
     'সবচেয়ে ভালো দিক ডেলিভারি দ্রুত প্যাকেজিং খুবই সুন্দর আল্লাহ আপনাদের মঙ্গল করুক',
     'সুন্দর নিছি',
     'শেষ তারিখ পার যাবার পরও অর্ডার পেলাম না',
     'কম দামে পেয়েছি চালিয়ে দেখলা ঠিকই',
     'প্রোডাক্ট বাইরে ধূলার আস্তরণ',
     'দামটা বেশি হয়ে যায়',
     'চাইলাম পাইলাম খারাপ একটা অভিজ্ঞতা',
     'একদম তাজা পণ্য পেইছি',
     'আপনাদের খারাপ দিক চাইতে না',
     'সঠিক পণ্যের মান নিশ্চয়তা নাই কিভাবে কিনবে',
     'সত্যি রিস্ক নাই আপনাদের সার্ভিস বেস্ট',
     'ডেলিভারি চার্জ নিবে প্রোডাক্ট না দেয়ার সম্ভাবণা',
     'ওনাদের প্রোডাক্ট কোয়ালিটি যথেষ্ট ভালো সময়মতো প্রোডাক্ট হাতে পৌঁছেছে ভালো কোয়ালিটির ক্যাপ খুঁজছেন প্রোডাক্ট ট্রাই',
     'আল্লাহর দোহায় লাগে বিশ্ব বাটপারদের বিরুদ্ধে একটা করুন সবাই প্রকৃতপক্ষেই চিটার',
     'প্রতারক দুরে থাকুন',
     'বাংলাদেশের অনলাইন পন্যের মান কেনে জীবনে কেনার ইচ্ছেও প্রকাশ না।পন্যের মান খুবই খারাপ',
     'জীবনেও evally অর্ডার করব না',
     'ভাই দারাজ বড় ধান্দাবাজ',
     'ওয়াও প্রাইজ প্লিজ',
     'মুজাগুলা সুন্দর মাশাল্লাহ ধন্যবাদ সেলার ভাইয়াকে পিকাবো ধন্যবাদ',
     'প্রোডাক্টগুলো ভালোই',
     'ডেজার্ট সত্যিই হতাশ',
     'টাকা ফেরত দাও নষ্ট ফেরত দাও',
     'খুবই ভালো ৪ টা ফোন ডেলিভারি দিছে সময়',
     'পায়ের নিচে চেইন নষ্ট',
     'ছবি পারলাম না ফেলছি অভিযোগ কারন নাই ছবি তে ঠিক পাইছি',
     'প্রাইজ টা কিঞ্চিত হাই',
     'দোকানের ড্যামেজ যেগুলো রির্টান দিছে সেগুলো অনলাইন কাস্টমারদের দিচ্ছি',
     'যেমনটা চেয়েছিলাম তেমনটা পেয়েছি',
     'আজকে অর্ডার,ডেলিভারি পাইছি ডেলিভারি ম্যান দেখি জুতা ছোট হয়।',
     'ধন্যবাদ যেরকম চেয়েছি ঠিক রকমই পেয়েছি',
     'ফালতু ডেলিভারি সার্ভিস',
     'অজও অছি কালও থাকবো বিশ্বাস অছে',
     'সামগ্রিকভাবে সত্যিই ভাল',
     'ডেলিভারী পেলাম হাতে পেয়ে মুগ্ধ হয়েছি আমের সেরা প্যাকেজিং',
     'খাবারটা ভয়ংকর',
     'আজকে প্রডাক্টটা পেয়ে খুবই খুশি বিক্রেতার প্রডাক্ট প্যাকেজিং মান ভাল',
     'দোকানে দাম আনেক কম আপনাদের সপ',
     'কষ্টের টাকায় শ্রেষ্ঠ বাজার স্বপ্ন তে',
     'ভালো প্রডাক্ট ব্যশ কম দামেই পেয়েছি আবারো কিনবো ইনশাআল্লাহ',
     'আপনাদের ডেলিভারি চার্জ একটু কমানো দরকার',
     'আপনাদের পন্যের গুণমান চমৎকার সত্যি প্রেমে গেছি',
     "বাংলাদেশের একমাত্র বেস্ট ই-কমার্স সাইট 'আলেশা মার্ট",
     'ধন্যবাদ চালডালকে আশা করছি সবসময় সেবা দিয়ে যাবেন পন্যের মান ঠিকটাক রাখবেন',
     'নিলাম ভালো মানের জুতা',
     'সুস্বাদু খাবার খাইনি',
     'শতভাগ ট্রাস্টেড ৪ টাকার কাছাকাছি রিলেটিভ পন্য ক্রয় ডেলিভারি ভালো',
     'জিনিসটি বাস্তবে ভালো না',
     'বর্তমান নাম্বরি ই-কমার্স মধ্য ভালো ট্রাস্টেট একটা ই-কমার্স খুঁজে দুষ্কর',
     'কোয়ালিটিফুল কালেকশন অলওয়েজ',
     'বুয়া টাকা খবর নাই জিনিস দেবার,',
     'একটার ৩টা বাল্ব জ্বলে আরেকটা পুরাই নষ্ট',
     'প্যাকেট খোলার পরই উপরের কাচ খুলে খুবই হতাশাজনক',
     'সত্যি সুন্দর প্রোডাক্ট',
     'ফালতু একটা কোম্পানি চুরি পাঠিয়েছে ডাট ভাঙ্গা',
     'তোফু পুরানো বাসি স্বাদের',
     'অসংখ্য ধন্যবাদ পণ্য পাওয়ার',
     'বকিছুই ঠিকঠাক সাইজটা উল্টাপাল্টা সাইজটা পাঠানো স্মল সাইজ ট্যাগ লাগানো সাইজের।',
     'ধন্যবাদ ভালো প্রোডাক্ট দেওয়ার',
     'অনলাইন কেনাকাটায় দেবে একমাত্র অনলাইন নির্ভরশীল প্রতিষ্ঠান মোনার্ক মার্ঠ',
     'দাম কম ভাবছিলাম প্রোডাক্টই ভাল না আসলে দাম ভালো ',
     'কমার্সের নামে আপনারা মানুষের সাথে করতেছে একদিন হিসাব আপনাদের কড়ায় ঘন্ডায়',
     'মোনার্ক মার্ঠ অনলাইন নির্ভরশীল প্রতিষ্ঠান এটার অর্ডার যায়',
     'পণ্য সন্তুষ্ট সবকিছু ঠিক দ্বিতীয় শেষ ছবিটি এটির ভিতরে ফোন তোলা কিনতে',
     'অরে বাটপার',
     'চাইছি সাদা পাইছি নীল',
     'কাল কোড ইউস রিচার্জ দিলাম ক্যান্সেল দিলো রিফান্ড কয়দিনে দিবে।',
     'আসলেই ভাল',
     'পিকাবু স্মার্ট সার্ভিস না ই-কমার্স ইলেকট্রনিক গ্যাজেটস পিকাবু বেস্ট',
     'কিনতে চাচ্ছি পেমেন্ট যাচ্ছেনা',
     'করলাম ২৪ ঘন্টা ভালোই করেছ + ভালোই ঠিক টাক',
     'প্রতিটি নকশা পছন্দ প্রশংসা',
     'আজকে একটা প্রডাক্ট ওর্ডার করছিলাম,, কিছুক্ষণ দেখলাম অর্ডার টা বাতিল দিয়েছে!',
     'এগুলো ক কালার গেরান্টি ওয়াস কালার যায়',
     'দাম লিখতে সমস্যা কোথায়',
     'একবারে বাজে প্রোডাক্ট বক্সটা ফাটা জায়গায়,',
     'অর্ডার প্রাপ্ত প্রোডাক্ট ছবির সাথে মিল না',
     'দুজনই আপনাদের খাবারের ফ্যান হয়ে গিয়েছি',
     'মাস্ল ৩দিন যাবত করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম',
     'আপনাদের সার্ভিসে সন্তুষ্ট',
     'সততা ব্যবসা সফলতা আসবে ',
     'ফালতু প্রোডাক্ট ৭০ টাকার ১২ ভোল্ট মোটরের বাতাস চেয়ে বেশি',
     'আপনাদের প্রোডাক্ট গুলো অসাধারণ',
     'প্রতারককে বিশ্বাস না',
     'নেট প্রাইজ আরো ১০০ টাকা বেশি চেয়েছেন',
     'মানের পণ্য চমৎকার জিনিসপত্র প্যাকেজিংও দুর্দান্ত',
     'ভালো মানসম্মত ঘড়ি',
     'অর্ডার নিশ্চিন্তে',
     'তারগুলো মোটামুটি ভালোই  কারোর লাগলে নিঃসন্দেহে',
     '১০৯৪ টাকা প্রাইজ একটা সাট নিছিলাম।কিন্তু পড়তেই রং শেষ একটা চেঞ্জ দিছে টাও একি',
     'স্মেল ভালো লেগেছে,ডেটও',
     'দামে কম মানে ভালো',
     'না',
     'সোনালী কালারের জুতাটা কিনছিলাম আগের পরের দিনই ফিতা,হিল খুলে ',
     'প্রথমে ধন্যবাদ সেলার ব্ল্যাক চাইছিলাম পাইছি এক কথায় কম দামে ভালো পণ্য ব্যাটারি চার্জ সাউন্ড কোয়ালিটি ভালো',
     'জিনিস সুন্দর দাম বেশি',
     'আসসালামু আলাইকুম টি-শার্ট অর্ডার করেছিলাম হাতে পেলাম প্রথমে ভেবেছিলাম কাপড়ের মান ভালো কিনা টি-শার্ট হাতে মন ভরে',
     'আলহামদুলিল্লাহ  দিনে দিনে পাচ্ছি',
     'সহজ প্রিয় ব্র্যান্ড বাংলাদেশে',
     'ফ্যাশন পছন্দ চমত্কার',
     'ডেলিভারি চার্য কমানোর দরকার মূল্যের প্রোডাক্টে কয়েন এড জরুরি',
     'প্রাইজ বেশি ভাই',
     '100% ওয়াটারপ্রুফ বাজে কথা মিথ্যা',
     '26 তারিখে রিটার্ন দিয়েছি আজকে 9 তারিখ এখনো খবর নাই',
     'খাবার পছন্দ হয়েছে',
     'যেমনটা ছবিতে ঠিক তেমনই পেয়েছি',
     'সত্যি মতে স্বাদ ভালো লাগে না।',
     'আপনাদের ডেলিভারি চার্জ কমান রেগুলার ক্রেতা ছিলাম',
     '১০৬৯ টাকার অর্ডার ভিসা কার্ডে পেমেন্ট করলাম এখনো ক্যাশব্যাক পেলাম না',
     'অসাধারণ একটা জার্সি',
     'খুব্বি মজার চিজ',
     'ভাল প্রতিটা মাক্স ☺️ পাশাপাশি KN95 মাক্স টা সম্পূর্ণ প্যাকেজটি মূল্যে অসাধারন',
     'এবারের প্যাকেজিং টা বাজে ছিলো',
     'কোয়ালিটি পছন্দ হয়েছে আপনাদের শুভকামনা রইল',
     'প্রথমটা পছন্দ হইসে বাট দামটা বেশি,,কিছু কমানো যায়না',
     'প্রোডাক্ট ভালো দাম বেশি',
     'সেবা আরো ভাল পারত',
     'দাম পাঞ্জাবিটা সুন্দর একটু গরম লাগে সমস্যা নাই ভালো',
     'ডেলিভারিতে দেরি',
     'অর্ডার কনফার্ম হওয়ার প্রোডাক্ট হাতে পাওয়া পুরোটাই ভালো একটা অভিজ্ঞতা',
     'পন্য ক্রয় দেখি সার্টের গুনগত মান খারাপ',
     '১০% এন্ড ৫% দুইটাই পাইছি আলহামদুলিল্লাহ',
     'কথা মিল রাখলে গ্রাহক বারবে না কমবে আশা দ্রুত সমাদান ধন্যবাদ',
     'টাকা পণ্য দেয় নাই',
     'ভাই ২ টা ফোন অডার দিয়ে ছিলাম ৫ মাস হয়ে খবর নাই ',
     'গ্রুপে পোস্ট করছি এপ্রুভ না',
     'বাহ সুন্দর মার্জিত রঙ নকশা',
     'সুন্দর ভালো লেগেছে অসাধারণ লেগেছে',
     'টাকা পণ্য দেয় নাই',
     '১০৬৯ টাকার অর্ডার ভিসা কার্ডে পেমেন্ট করলাম এখনো ক্যাশব্যাক পেলাম না',
     'মূল্যটা দিয়েছেন বেশি',
     'ভাই জিনিস ওয়াডার করছি আজকে পাজ হয়ে গেলো পাইনি পাবো একটু জানাবেন',
     'সুস্বাদু খাদ্য সেবায় সন্তুষ্ট',
     'অনলাইনে ঠকার আশংকা',
     'ভালোবাসা অবিরাম',
     'এধরনের ঠকবাজি কর্মকান্ডের জন্যে বাংলাদেশে ইকমার্সে ঠিকমত গ্রো পারছে না ক্রেতা বিক্রেতাকে বিশ্বাস পারছে না',
     'নামে ভোক্তা অধিকারে মামলা করেছি কিছুদিন পোকা/পচা খেজুর দিয়েছিল',
     'আরো ভালো ফোন চেষ্টা করেন।সাধু বাদ',
     'প্রোডাক্ট দিলে খুশি হইতাম',
     'পাওনা বুঝিয়ে দিবেন পাশেই আছি শুভ কামনা রইলো',
     'আল্লাহতালার রহমতে ভাল প্রিমিয়াম কলেটির কভার পেয়েছি।ধন্যবাদ',
     'কিনেছি সত্যিই সুন্দর',
     'ছবির পেয়েছি পছন্দ হয়েছে',
     'মিথ্যা বিজ্ঞাপন ৩ টা পোডাক্ট ফ্রি ডেলিভারি কই।',
     'ফ্যান চালানোর সময় কেমন পোড়া গন্ধ করছিল',
     'সুন্দর ভালো মানের ক্যাপ চাইলে আপনারা উনাদের ক্যাপ টেনশন ছাডা',
     'শুভেচ্ছা শুভকামনা রইল',
     'দাম টা ত চড়া',
     'সমস্যা কেন...',
     'এক কথায় অসাধারণ',
     'আপনারাও',
     'পণ্যের দাম শিথিল',
     'পণ্যের গুণগত মান খুবই ভালো অল্প টাকায় ভালো জিনিস পাবো ভাবতেও পারিনি ☺ চোখ বন্ধ ১০ ১০ মার্ক দেয়ায় যায়',
     'আপনাদের প্রোডাক্টটি হাতে পেয়েছি বাক্সটি ভাঙ্গা',
     'সত্যিই বার্গার পছন্দ',
     'সুন্দর জিনিস  চাইলে সবাই',
     'রাতে অর্ডার দিব',
     'একছের দাম',
     'আস্তে আস্তে ফালতু ব্র্যান্ড কাপড় গুনগত মান ভালো না',
     'আপনাদের প্রোডাক্ট নিয়েছিলাম কয়মাস আগে,পুরো ফালতু',
     'পণ্য নকল',
     'কালার ঠিক ছিলো দাম প্রডাক্টটি ভালো',
     '"আলহামদুলিল্লাহ ভালো ',
     'অসাধারণ সুন্দর প্রোডাক্টগুলা. এক্কেবারে পারফেক্ট কাস্টমাইজড',
     'কোয়ালিটি সার্ভিস খুব-ই ভালো আরো কালেকশনের অপেক্ষায় থাকলাম শুভকামনা',
     'বছরের শুরুতে 2টি ভিন্ন সময়ে ছোট আইটেম অর্ডার করেছি গ্রহণ করিনি ম্যাসেজের দিচ্ছে না',
     'প্রমান পেয়েছি আপনাদের কথায় মিল আছে।আপনাদের মঙ্গল কামনা সাথেই আছি,ধন্যবাদ',
     'আশাকরি জিনিশ গুলা পাবো',
     'বুঝতে পারছি না বেশি দাম রাখছে',
     'দাম পন্য ভালো সেলাই ভালো দেখলাম সাইজ একদম পারফেক্ট আরামদায়ক',
     'এক বছরের ওয়ারেন্টি নষ্ট কিভাবে ঠিক দিবেন',
     'স্বাদ যথেষ্ট ভাল না',
     'আপনাদের সেলাই মান ভালো নিচের সেলাই কয় এক খুলে যায়',
     'কুয়ালিটি প্রাইস ২ টাই বেস্ট',
     'পেজের পন্য গুলো ভালো আপনারা সবাই দুইটি নিয়েছি সেম টু সেম  ধন্যবাদ',
     'শার্টটি খুবই সুন্দর গুলশন পাশে শোরুম ঠাকলে আমক জানাবে',
     'আপনাদের ডেলিভারি চার্জ কমান রেগুলার ক্রেতা ছিলাম',
     'ভাল পণ্য সহজ',
     'দিছেন চাইছি লাল দিছেন হলুদ এটাকি',
     'আপনারা চোখ বন্ধ',
     'দারাজ কম দামে মিষ্টির বক্স আছেকোন প্রোডাক্ট অর্ডার ক্যানসেল টাকা ফেরত পাচ্ছিনা',
     'এতো দাম',
     'সুন্দর',
     'হুদাই একটা পোস্ট কাজই না ভাউচারটি',
     '১০ দিনে ফ্রিজ পেলাম না না দিবে অর্ডার করছি',
     'ভালো ভালো রাখবেন গ্রাহক দের',
     'ক কথায় অসাধারন ৫ ষ্টারের বেশি দেয়া জায়না নয়ত আরো বেশি দিতাম কম দামে ভালো মধু দেয়ার সেলার দারাজকে ধন্যবাদ',
     'তোমারদে জিনিস গুলো ভালো',
     'গতকাল সবুজ টি-শার্টটা অর্ডার দিয়ে আজকেই পেয়ে গেলাম',
     'আপনাদের ডেলিভারি ম্যান গুলো ভালো ঠিক সময়েই আসে',
     'বাংলাদেশের সেরা পানীয়',
     'দশ পার্সেন্ট ভাটের কথা',
     'অসাধারণ খাবার',
     'দাম অনুসারে পন্য ভালো না',
     'ভালো লাগে নাই না হুদাই কিনলাম',
     'কেবল ভাল সাথে টেস্টার ফ্রি দিছে ধন্যবাদ',
     'আপনাদের পণ্য গুলো ভাল ছবির সাথে পুরো মিল',
     'প্রোডাক্ট অর্ডার করেছি আজকে ছয়দিনের ডেলিভারি পাই নাই',
     'আপনাদের সার্ভিস খারাপ হয়ে প্রত্যেকটা ডেলিভারি রাতে আসে কমিউনিকেশন না দারাজে কেনাকাটা না।',
     'তোমাদের কেনা-বেচার সক্ষম হয়েছি',
     'ভালে কোয়ালিটি আঠা স্ট্রং সেলার রেস্পন্সিবল সবাই',
     'কম প্রাইজ হিসেবে ভালোই আছে.. সার্ভিস দেয় দেখার বিষয়',
     'এগুলো ক কালার গেরান্টি ওয়াস কালার যায়',
     'সবাইকে সুপারিশ করব',
     'সুন্দর সংগ্রহ',
     'টাকা খরচ বই কিনবেন না',
     'প্যাকেটিং ভালো দ্রুত পণ্য ডেলিভারি দেয়া হয়েছে',
     '২ বেশি লাগলেও আজকে প্রডাক্টটি হাতে পেয়েছি',
     'ডেলিভারি চার্জ কমানো',
     'প্রিমিয়াম প্যাকেজিং',
     'দয়া অর্ডার না',
     'বাটপার  সময় এক হয়/ ছবিতে দেখায় একরকম আরেক',
     'ঠিকঠাক মেয়াদ',
     'যেমনটা দেখেছিলাম তেমনটাই মোটামুটি ভালো',
     'গেঞ্জির রং চেয়েছিলাম ছবিতে তেমনটাই পেয়েছি লাল নীল',
     'আপনাদের ভালো লেগেছে দুইবার নিলাম সেটিসফাইড শুভকামনা জানবেন',
     'কর্মীদের সুন্দর',
     'গ্লোবাল প্রডাক্ট অর্ডার করেছিলাম ১ মাসের বেশি সময় হয়ে ডেলিভারির টাইমলাইন শেষ হয়েছে এখনো ডেলিভারি দেয়ার নাম নাই',
     'প্রাইজটা বেশি',
     'সম্পূর্ন ক্যাশ অন ডেলিভারি অর্ডার দিবো নাহলে দেয়ার ইচ্ছা নাই',
     'কাপড়ের মান টা খারাপ দাম বাড়াচ্ছে',
     'আশা করবো ভবিষ্যতে আপনাদের কোয়ালিটি অক্ষুণ্ণ',
     'রিটার্ন শুধু বিজ্ঞাপনে বাস্তবে হয়না',
     'সাবধানে ওডার দিবেন  ওডার দিছি xl তাহারা আমারে দিছে S সাইজের শার্ট',
     'দ্রুত সময়ে ডেলিভারি পেলাম  মূলত লোটো kub আরামদায়ক',
     'কালাভুনা টা আসলে অত্যান্ত ভালো',
     'যেটা পেয়েছি ওইটার কোনু মিল নেই ৩৯ অর্ডার দিছি ৪০ পাইছি',
     'প্যাকেট টা খোলা আজাইরা জিনিস দিছে',
     'বিশ্বাস করুন মোটেও থাই স্যুপের স্বাদ না',
     'লাইট টিকবে না',
     'হাতে পাওয়ার ভালই',
     'এটার ফ্রি ডেলিভারি জানলে অবশ্যই দিতাম',
     'ইনশাআল্লাহ খেলা কাল',
     'বেল্ট মান টা দাম হিসেবে খুবি ভালো পাঁচ দিনের ডেলিভারি পেয়েছি',
     'লোকাল বাজার কম দামে পেয়েছি ডেলিভারি দ্রুত পেয়েছি',
     'সেলার ছেরা মাল দেয়',
     'স্টারটেক বিক্রয়োত্তর সাপোর্ট সবথেকে ভালো',
     'পচ্ছন্দ ডেলিভারি চার্জ নিরুৎসাহিত',
     'বর্তমানে অনলাইন শপিং বিশ্বাস যাইনা একটু দ্বিধা নিয়েই অর্ডার টা করেছিলাম',
     'মোনার্ক মার্ট ভালো এক্সপেরিয়েন্স ছিলো',
     'প্যাটি সঠিকভাবে রান্না সঠিক মশলা',
     'সেবা পছন্দনীয়',
     'দাম বেশি দিয়ে অনলাইন কিনবো',
     'সহজ মার্জিত',
     'দ্রুত ডেলিভারি পেয়েছি।ধন্যবাদ',
     '"আলহামদুলিল্লাহ পণ্যটি ভাল বেশি স্থিতিস্থাপক নয়...।',
     'প্রোডাক্টের ডেলিভারি চার্জ ফ্রি',
     'আজকেই প্রডাক্ট টা হাতে পেলাম অসম্ভব সুন্দর',
     'অনলাইন নিব না দোকানে নিব',
     'তেলের দাম আরো কমেছেই আপনারা দাম কমান',
     'ভালো এডজাস্ট যায় না',
     'আলহামদুলিল্লাহ আলেশা মার্ট বাইক ভেলিভারি পাইলাম',
     'ডেলিভারি দেরি',
     'চাইলে ভালো একটা পণ্য কম দামে',
     'আপনাদের সার্ভিস ভালো',
     'খুবই অপেশাদার অভদ্র',
     '২৩ প্রোডাক্ট হাতে পাইছি ফ্যান দেখানো হইছে একটা হইছে আরেকটা',
     'দাদা ভালো পেয়েছি',
     'সুন্দর',
     'দাম অনুযায়ী ভালো না সাইজও ঠিক দেয়নি',
     'প্রবলেম সলভ আরো বেশী খুশী হবো আশা আজকের সলভ',
     '৯০ টাকায় জোস প্রোডাক্ট',
     'টাকা পণ্য দেয় নাই',
     'দুর্দান্ত সৃজনশীল দল.....অসাধারণ',
     'কাপড় গুলো ফালতু দাম আকাশ চুম্বী ইজি পরি না',
     'বাংলাদেশে এখনো অধিকাংশ মানুষ স্ক্যামের স্বীকার',
     'ক্যান চেপ্টা খাওয়া প্যাকেট সিলড ছিলনা ব্যবহৃত প্রোডক্ট দিয়েছে',
     'প্রডাক্টি ভালো কফি মিক্সড',
     'দুইদিন পণ্য হাতে পেলাম',
     'অনেকগুলা অরডার একটু দেখবেন',
     'গায়ের সাথে ভাল ফিটেষ্ট হয়ছে ধন্যবাদ সেলার',
     'দাম হিসেবে কাপড় স্টাইল ভালো',
     'ট্রাই করেছি না',
     'অর্ডার ক্রিত প্রডাক্ট পাইনি করোনিও',
     '২৮ তারিখে দেওয়ার কথা প্রডাক্ট পেলাম না',
     'প্যাকিং,প্যান্ট কোয়ালিটি দ্রুত ডেলিভারী দেওয়ার ধন্যবাদ',
     'এগিয়ে যাও',
     '৩দিন যাবত করছি।।খুবই ভাল কোয়ালিটি।।পড়তেও আরাম',
     'বাজার সেইম একটা কিনেছি এটার চেয়ে স্ট্রং ভাল',
     'বলবো বল পুরাই অস্থির একটা জিনিস প্যাকেজিং ঠিকঠাক সুন্দর চাইলে আপনারাও কিনতে',
     'সুন্দর ভালো লাগছে',
     'এক কথায় অসাধারণ',
     'গিফট কোথায় পাইনি',
     'মহান সেবা সম্পূর্ণরূপে সন্তুষ্ট',
     'সবাই ধান্দাবাজ ঠিক ঠাক পণ্য ডেলিভারি দেয় না',
     '৩১ শে জুলাই অর্ডার পাই নাই',
     'বিশ্বাসের মাত্রাটা বেড়ে',
     'দাম কমান',
     'কাপড়ের মান খুবই নিম্নমানের কয়েকদিন রং নষ্ট হয়ে যায়',
     'মানুষের কষ্টের টাকা ইনজয় মৃত্যু নেই',
     'অসাধারণ একটা ঘড়ি  পানির দামে শরবত পাইলাম সাথে ফ্রি চুলের বেন্ড  ধন্যবাদ সেলার',
     'টি-শার্টের মান সত্যিই ভাল',
     'আপনাদের জিনিসের দাম একটু কমানো',
     '11 অর্ডার দিছি পেমেন্ট খবর নাই',
     'ভাল লাগতেছে না',
     '২৪ ঘন্টার দেবে অবিশ্বাস্য',
     'একদম জোস,চাইলে কিনথে',
     'কেনো যেনো অনলাইন প্রডাক্টে আস্থা না।কিন্তু ২য় Deen টি-শার্ট নিলাম আলহামদুলিল্লাহ নিরাশ হইনি',
     'কিছুদিন দেখলাম আল্লাহামদুলিল্লাহ কভারের মান অত্যান্ত ভাল',
     'দামে কম মানে ভালো',
     'একটার সাইজ চেঞ্জ জনু সেলারের সাথে কন্ট্রাক্ট করেও রিপ্লাই পায়নি',
     'ডেলিভারি চার্জটা কমান একটু!!',
     'পণ্য নিয়া অভিযোগ নাই',
     'প্রোডাক্ট পরিমান বাড়াতে আইটেম একেবারেই কম',
     'যোগাযোগ না "ডেলিভারি ফেইলড""।',
     'ভাই আমিও বিকাশে পেমেন্ট করছিলাম ক্যাশব্যাক পাইনি',
     'একদম ফালতু 500টাকার প্যান্ট বেচে 1200টাকায় ইসলামী নাম বিভ্রান্ত হয়েন্না প্লীজ',
     'এপ্স অর্ডার চেস্টা হয়না',
     'আপনাদের নিয়ম মেনে তিনদিন অডার কনফার্ম করছি এখনো পণ্য পাই নি পাবো',
     'আপনাদের প্রত্যেকটা প্রোডাক্ট খুবই মানসম্মত',
     'আপনারা জনগনের সাথে বাটপারি করতেছেন ডেলিভারি দিচ্ছেন না',
     'বক্স ভালো সুন্দর একটুও তেল পরেনি সাথে একটা সাবান গিফট',
     'দাম টা একটু কম যায় না',
     'প্রশংসার আপনাকে ধন্যবাদ',
     'মোবাইল অফার দিলে বেশি সেল প্রফিট',
     'ভালো লাগলো না',
     'পন্য দারাজ নিয়েছি পন্য সঠিক সময়ে পাইছি',
     'জুতা হাতে পেয়ে সত্যি জিতে গেছি ভালো মানের জুতা বাকিটা ইউজ বুঝা',
     'ঘুরে দাড়াক ই-কমার্স',
     'সেম্পল পন্যের সাথে রিসিভ করলাম মিল আছে? মর্মাহত হয়েছি এতো নিম্ন মানের পন্য পেয়ে',
     'সবাই ধান্দাবাজ ঠিক ঠাক পণ্য ডেলিভারি দেয় না',
     'সমস্যা বুঝতে পারলাম না লাগাতার সমস্যা',
     'ইউস ডুরাবিলিটি লাগবে ভাইয়াকে ধন্যবাদ সততার',
     'জাবের ভাইয়ের অনলাইনের দোকান ভালো',
     'আমিও প্রস্তুতি নিচ্ছি মামলা',
     'টা ছবিতে দেখানো মুজা গুলো না যেটা এগুলো ভালো না',
     'ধন্যবাদ চাল ডাল বছরের মত গিফট পেয়েছি',
     'ফাইজলামি!!',
     'বাংলাদেশে অনলাইনে কেনাকাটা 90% মানুষ ঠকে',
     'পণ্যটি ছবিতে দেখেছিলাম হচ্ছিল বড় হাতে পাওয়ার দেখলাম খুবই জিনিস দেখলে খুবই কম দামি',
     'একেবারে ফালতু চেয়েছি ৪২ সাইজের পাঠিয়েছে ৪৩ সাইজ',
     'কোয়ালিটি ভালো খুশি',
     '"মুটামুটি পরতে বাটন খুলে গেসে সেলায় ভালো সিল না',
     'হুদাই চেষ্টা করলাম খালি',
     'দামটা একটু বেশি',
     'আইটেমগুলো সুস্বাদু সেরা',
     'আপনাদের প্রডাক্ট ভালো মানসম্মত।ধারাবাহিক আপনারা মানসম্মত প্রডাক্ট ডেলিভারি দিলে অচিরেই আপনাদের সাফল্য পেয়ে যাবেন',
     'আলহামদুলিল্লাহ',
     'আপনাদের ডেলিভারি মাধ্যমটা ভালো নয়',
     'আজকে প্রডাক্টা পেলাম সুন্দর হইছে ধন্যবাদ আপনাদের',
     'চমৎকার কালেকশান',
     'কোয়ালিটি ভালো আগেও কিনেছিলাম রিভিউ নাই সেলারকে ধন্যবাদ অসাধারণ কোয়ালিটি',
     'বেশি জোস খাবার',
     'আপনারা কম দামের প্রোডাক্ট দিয়ে বেশী দাম লিখলে জনগন খাবে না',
     'পন্যটি ভাঙ্গা অবস্থায় পাইছি,,রিটান করছি এখনো টাকা পাইনি,,,,ছিটার কোম্পানি',
     'অসাধারণ মানের খাবার ....',
     'অসম্ভব ভালো ফেস ওয়াস ',
     'সুন্দর সুন্দর প্যান্ট',
     'আলহামদুলিল্লাহ পেয়েছি পছন্দের জিনিস গুলো মিলিয়ে কম দামে অসাধারণ প্রোডাক্ট',
     'সুন্দর কালেকশন আপু',
     'দুঃখজনক সত্যি দেশে ভালো মানের অনলাইন শপ নাই.. শুধুই গ্রাহকদের ঠকিয়ে কষ্ট দেয়',
     'ভাউচার দিয়ে অর্ডার করলাম,জিনিসও পেয়ে গেছি',
     'প্রাপ্ত পণ্য নিম্নমানের ত্রুটিপূর্ণ 5 পিস সেটের 2 জোড়া',
     'বোলবো অর্ডার কোরলাম সাদা দিলো ব্লু,এটা ঠিকনা,এছারা ঠিক প্যাকেটিং ভাল ছিল,নতুন মাল দিয়েছে,সাউন্ড ভালোই।',
     'দারাজ একটা ফালতু বয়কাট করলাম',
     'ভোক্তায় একটা অভিযোগ মেরে শিক্ষা হওয়া',
     'আপনাদের প্রডাক্টটি ভালো',
     'সোয়াবিন তৈল সরকারি রেট ৫ লিঃ ৯১০/- টাকা এ্যাড দিয়ে ৯৫০/- টাকায় বিক্রি ভোক্তাঅধিকারের দৃষ্টি আকর্ষণ করছি',
     '৫ মিনিট চলার না',
     'কালার দিয়েছি পাই নাই সাইজ টাও টিক না',
     'আমের কোয়ালিটি ভালো ছিলো কালারো সুন্দর ছিলো',
     'ওনাদের সিস্টেম ভালো আমিও পন্য চেইঞ্জ করেছিলাম আরো অর্ডার দিবো কাজের লোকদের',
     'ফেসবুক পোষ্ট ডিজাইন ভালই বানিয়েছেন',
     'দাম পণ্য ঠিক কফি ফোম না',
     'সত্যিই খারাপ',
     'নম্র আচরণ খাবার স্বাস্থ্যকর সুস্বাদু',
     'মূল টাকাটা জাস্ট লাভ দরকার নেই',
     'চাইছি একটা পাইছি আরেকটা,পাইছি নষ্ট এগুলো নিবেন না',
     'বড় সু্যোগ',
     'একটা প্রোডাক্টস অর্ডার ২৮ অপেক্ষায় ছিলাম',
     'বইটা পড়া সেরা বই',
     'ভুক্তভুগীরা প্রোডাক্ট পাবো টাকাটাও ফেরত পাবো না',
     'দাম হিসেবে খারাপ না ভালোই...তবে চশমার ফ্রেমের লাল ডিজাইনটা রাবাবের,কিছুদিন উঠে যাওয়ার সম্ভাবনা',
     'টা চাইছি ঠিক টাই পাইছি ভাল বেল্ট টাকা ভাল হইছে',
     'বিফ চিজ ব্লাস্ট প্রিয়',
     'কাপড় জার্সির কাপড় description পেয়েছি  সেলার অনেস্ট ',
     'ডেলিভারি চার্জ কমানো হোক!!',
     'দামের তুলনায় বেশি ভালো',
     'অতিদ্রুত ক্যাশ অন ডেলিভারি পেয়েছি',
     'জুতা হয়েছে',
     'পারফেক্ট',
     'ওয়াও না চাইতেই',
     'ভালো,,কিন্তু পায়ে লুজ',
     'একদম পাতলা খাচা',
     'ফাজিল অর্ডার মাল পাঠায় না',
     'সর্বকালের সেরা সংগ্রহ',
     'ডেলিভারি খুবই লেইট',
     'বান টাও একটা ভালো কোয়ালিটি ছিলো না',
     'চমৎকার সংগ্রহ',
     'নিশ্চিন্তে নেন অলরেডি ইউজ করছি এছাড়াও গত ৬-৭ বছরে প্রোডাক্ট নিয়েছি সবগুলো ১০০% পারফেক্ট',
     'কাপড়ের মূল্য 585tk নয় আজকে হতাশ হয়েছি',
     'সঠিক সময়ে হাতে পেয়েছি কাপড়ের মান ভালো লেখা দিয়েছে নির্দ্বিধায়',
     'A লেভেল একটা জিনিস সেলারকে বলবো মানসম্পন্ন প্রোডাক্ট সেল',
     'চমৎকার সংগ্রহ',
     'তিনটি পন্য অর্ডার বাতিল করলাম শিপিং ফি জন্য,',
     'মিলিয়ে আনপ্রফেশনাল পোস্ট',
     'আলহামদুলিল্লাহ যেরকমটা চেয়েছি সেরকমই পেয়েছি ২ টা অর্ডার দিছিলাম ২ টাই পেয়েছি',
     'অভার প্রাইজড',
     'মানান সই',
     'ঈদ উল ফিতর এবারের কালেকশন সত্যি ভালো',
     'বাংলাদেশের একমাত্র ফালতু ব্যান্ড থাকলে ইজি ফ্যাশন',
     'অনেকদিন রিভিউ দিলাম।T-shirt সুন্দর।খুব আরামদায়ক।কাপড়ের কোয়ালিটি ভালো।Showroom তুলনায় কম দামে T-shirt পেয়েছি',
     'দানের তুলনায় ভালো জিনিস',
     ' চমৎকার ফ্যান ভালো লেগেছে',
     'কাপড়ের মান খুবই নরম পরতে আরামদায়ক',
     'রাত ৯ টার ডেলিভারি রাত ১১-২০ দেন,,,সময়টা একটু মাথায় রাইখেন',
     'ধন্যবাদ আপনাদের।।। কথা রাখার জন্য।।।।',
     'প্রোডাক্টা প্যাকেজিং আরো ভালো ভালো হতো',
     'ওয়াও অসম্ভব সুন্দর',
     'ভালো সার্ভিস পেয়েছি ধন্যবাদ',
     'ভাই ছবি তে একটা দিছে আরেক',
     'ইজির প্রোডাক্ট ভালো না সালারা ২০০৳ টি-শার্ট ৬০০ টাকায় বিক্রি',
     'ভালো কফি মিক্সার এইদামে',
     'পেন্টের কাপর টা ঠিক আছে,,কিন্তু গেঞ্জির কাপর টা পাতলা,,but আরামদায়ক,',
     'পণ্য ফেরত চাই',
     'ভাই রিয়েলমি সি৩১ অডার ফোন হাতে পাইলাম না প্রতারনা শিকার হয়ে দারাজ বাদ দিলাম ফোন কিনা জবাবদিহিতা লজ্জা জনক।',
     'দারাজের নামে প্রতারণার মামলা দিবো নাম্বার মাল পাঠাই',
     'উপকার',
     'প্রতারিত হয়েছিলাম',
     'আপনাদের বুকিং দেয়া প্রডাক্ট কোয়ালিটি খুবই ভাল লেগেছে',
     'দুইবার আম পাঠাইছে একবারও ভালো পরলো না',
     'দাম লিখতে সমস্যা কোথায়',
     'বাজে প্রডাক্ট অর্ডার চেয়ে গাজিপুরে ফুটপাতে ভাল পাওয়া যায় প্রডাক্টে ময়লা মহিলাদের চুল এমনটা আশা নাই',
     'খুবই বাজে সার্ভিস খেতে চাইলে অর্ডার',
     'না কিনতাম না',
     'আলহামদুলিল্লাহ কাপড়ের মান ভালো.. টা দিয়ে তিনবার নিলাম ডিন থেকে... মাশাআল্লাহ ডেলিভারিও পেয়েছি খুবই তাড়াতাড়ি..',
     'আলহামদুলিল্লাহ প্রডাক্ট কোয়ালিটি সার্ভিস খুবই ভালো',
     'সকল পঁচা নিম্নমানের বেশী দামে অনলাইনে দ্রব্য বিক্রি যতবার কিনেছি ততবার খাইছি',
     'তাজা না দাবি আক্ষরিক অর্থে সবচেয়ে খারাপ কেক',
     'ওডারটা পেলাম না',
     'পেমেন্ট না',
     'প্রয়োজনীয় প্রডাক্ট...সেলার ধন্যবাদ...ভাউচার দিয়ে ২৪ টাকায় কিনেছি....আপনারা কিনতে',
     'উনাদের ডেলিভারিও ফাস্ট',
     'প্রোডাক্ট গুলো তে ধুলাবালু ক্র্যাচ',
     'দিবেন ভাই বুড়া হয়ে গেলে???',
     'সাইজের সমস্যা চেয়ে ১ সাইজ বড় দরকার সঠিক পরামর্শ পাই নাই ফেরত নাই',
     'একমাত্র আপনাদের পেইজেই ভালো প্রডাক্ট পাওয়া যায়',
     'প্রডাক্টিভ হুবুহুব একদম ছবির সাথে মিল খুশি',
     'সুন্দর পাঞ্জাবি',
     'অসম্ভব ভাল লেগেছে',
     'সীমিত আইটেম',
     '১০৬৯ টাকার অর্ডার ভিসা কার্ডে পেমেন্ট করলাম এখনো ক্যাশব্যাক পেলাম না',
     'কাপড় নষ্ট হয়ে যায়।যদি কিনি L সাইজ ঐইটা হয়ে যায় XL',
     'প্রোডাক্ট টিক ডেলিভারি সময় মত কালারটা পিংক',
     'মামলা ভাই সুযোগ না',
     'সার্ভিস আরো ডেভলপ',
     'লাইক দিয়ে পাশে থাকলাম',
     'টাকা ১ বছরেও রিফান্ড দিল না',
     'দারুন অফার',
     'সমস্যা লো কোয়ালিটি ম্যাটারিয়াল ইউস আপনারা',
     'দাম গুলো বেশী',
     'দারাজ ভাল না বিশাল ঠকবাজ-আমার পুরাতন টেবিল দিছে-এখন রিটার্ন চেঞ্জ দেয় না',
     'ওয়াও',
     'এভাবে কিনা সকলের সুবিধা হাটে যাওয়ার জামেলা নাই',
     'এইটা জুস খেতে ভালোই লাগে স্টকে চাই',
     'প্রোডাক্ট ভাঙ্গাচোরা থেতলানো',
     'অবিশ্বাস্যভাবে সুন্দর ব্যাগ আপনাকে ধন্যবাদ',
     'সবকিছুর দাম বেড়ে',
     'আসলেই সুন্দর',
     'অর্ডার টা করেছি এখনো প্রসেসিং অবস্থায় একটু তাড়াতাড়ি দিলে উপকার হতো',
     'ব্যাগটি খুবই ভালো মানের পণ্যনিয়ে খুশি',
     'প্রাইস হিসেবে কোয়ালিটি স্যাটিসফাইড',
     'দাদা ভালো পেয়েছি',
     'অসাধারণ স্বাদ',
     'ডেলিভারি না!',
     'ভাই সাইকেল এখান',
     'জানুয়ারির ১৫ তারিখে অর্ডার করেছি এখনো প্রোডাক্টটি পাই নাই,ইভ্যালি টাকা মেরে দিসে',
     'সাধ্যের সবচেয়ে ভালো প্রোডাক্ট',
     'দারাজ কাস্টমার ধরনের সুবিধা না',
     'খুবই ভালো মজবুত রিজনাবল',
     'ধন্যবাদ আলেশা_মার্ট গাড়ী কনফার্ম',
     'ধন্যবাদ mr fiction',
     'পাশে আছলাম পাশে আছি পাশে থাকমু',
     'ভালো মধু চোখ বন্ধ',
     'যথেষ্ট প্রিমিয়াম প্রোডাক্ট আজকে ছেলের টি-শার্ট নিয়েছি চমৎকার ফেব্রিক্স',
     'আলহামদুলিল্লাহ. চাইসি ভালো. কাপড় কোয়ালিটিফুল',
     'প্রোডাক্ট গুলো ডেলিভারি দ্রুত আশাবাদী',
     'টাকা অনুযায়ী ছাতাটা ভালো',
     'পিজা অন্যান্য খাবার চমৎকার সার্ভিস ভাল',
     '১৫ টা অর্ডার করেছিলাম ১১ টা দিয়েছে বাকিগুলা না টাকা ফেরত দিয়ে',
     'জিনিসটা সুন্দর ভাঙ্গা ডেলিভারি ম্যান যত্নে আসতে পারেনি',
     'মারকেটিং ভালো আপনাদের বাট কাপরের ভ্যারাইটি নাই',
     'পান্ডামার্ট বিশ্বাসটা উঠে গেলো',
     'পেয়েছেন',
     'ডেলিভারি আলহামদুলিল্লাহ সুন্দর হয়েছে',
     'ভালো লেগেছে',
     'এতো অল্প দামে অসাধারণ পণ্য আগেও নিলাম বেস্ট আগের শপ',
     'আপনাদের গোল গলা গেঞ্জি গুলো লক সেলাই খুলে খুলে যায় গেঞ্জি নষ্ট হইছে',
     'জামাটি ভালো গরমের সিজন দাম অনুযায়ি আপনারা চাইলে কিনতে',
     'দামে কম মানে ভাল.',
     'আপনারা ২ নাম্বার জিনিস দেন,গ্রাহকে হয়রানি',
     'প্রোডাক্ট অবস্থা খুবই খারাপ ইদুরের কাটা জিনিস দিয়ে দিছে প\u200d্যাকেট সস খারাপ অবস্থা',
     'ফ্রড প্রতিনিয়তই',
     'বাজে একটা পিরত দিয়ে দিছি এখনো টাকা দেয় নাই',
     'অবশেষে পেয়ে গেলাম অসাধারণ এক পিজ্জা',
     'চমত্কার খাদ্য',
     '"আলহামদুলিল্লাহ ২য় অর্ডার পেয়েছি বজায় রাখা…',
     'ভালভাবেই ফিট হয়েছে।ডেলিভারিও পেয়েছি দ্রুত',
     'নকশা সত্যিই ভাল পছন্দ ',
     'ডেলিভারির ডেট একদিন পার হয়ে যাওয়ার পরও প্রোডাক্ট হাতে পেলাম না',
     'সত্যিই চমত্কার চমৎকার মানের',
     ...]



Let's break down this code step by step in simple terms:

### 1. **`TfidfVectorizer`**
```python
tfidf = TfidfVectorizer(ngram_range=(1, 3), use_idf=True, tokenizer=lambda x: x.split())
```

- **`TfidfVectorizer`** is a tool from the **`sklearn.feature_extraction.text`** module. It is used to convert a collection of text documents into numerical data that a machine learning model can understand. Specifically, it transforms text into **TF-IDF** (Term Frequency-Inverse Document Frequency) features, which are often used in text processing to represent the importance of words in a document.

  Here's what each argument does:
  
  - **`ngram_range=(1, 3)`**: This tells the vectorizer to consider **unigrams, bigrams, and trigrams** (i.e., 1-word, 2-word, and 3-word sequences) from the text. For example:
    - A unigram would be a single word, like **"apple"**.
    - A bigram would be two consecutive words, like **"green apple"**.
    - A trigram would be three consecutive words, like **"fresh green apple"**.
  
  - **`use_idf=True`**: This means that the vectorizer will use **Inverse Document Frequency (IDF)** to weight the words. IDF helps to down-weight words that are common across many documents (like "the" or "is"), and gives more importance to rare words that are specific to a document or a group of documents.
  
  - **`tokenizer=lambda x: x.split()`**: This defines how to break the text into words (tokens). The `lambda x: x.split()` is a simple function that splits each sentence (`x`) into a list of words by spaces. This is the same as the default tokenizer in many text processing tasks.

### 2. **`fit_transform`**
```python
Xtrain_tf = tfidf.fit_transform(Xtrain)
```

- **`Xtrain`**: This is the input data for training the model, and it contains the sentences (cleaned and preprocessed).
  
- **`fit_transform(Xtrain)`**: This method performs two operations:
  1. **`fit`**: It **learns** the vocabulary (the words and n-grams) from the `Xtrain` data and calculates the necessary statistics (like term frequency and inverse document frequency).
  2. **`transform`**: It then converts each sentence in `Xtrain` into a vector of numbers based on the TF-IDF score of each word (or n-gram). Each sentence is represented as a vector, and the vector contains weights for the words (or n-grams) based on their importance in the sentence.
  
  The result is stored in `Xtrain_tf`, which is a sparse matrix representing the transformed sentences.

### 3. **Understanding `Xtrain_tf.shape`**
```python
print("n_samples: %d, n_features: %d" % Xtrain_tf.shape)
```

- **`Xtrain_tf.shape`** gives the dimensions of the transformed matrix `Xtrain_tf`:
  - **`n_samples`**: The number of sentences (or documents) in the training set (`Xtrain`).
  - **`n_features`**: The number of unique words or n-grams (from unigrams, bigrams, and trigrams) in the entire training set. This represents how many features (columns) are used to describe each sentence.

- **`print("n_samples: %d, n_features: %d" % Xtrain_tf.shape)`**: This prints the shape of the transformed matrix, showing how many samples (sentences) and features (words or n-grams) it contains.

### Example:
If `Xtrain` has 100 sentences and the vocabulary (considering unigrams, bigrams, and trigrams) has 500 unique words/n-grams, `Xtrain_tf.shape` would be `(100, 500)`, meaning there are 100 sentences and 500 unique features (words or n-grams) in the dataset.


```python
tfidf = TfidfVectorizer(ngram_range=(1,3),use_idf=True,tokenizer=lambda x: x.split())
Xtrain_tf = tfidf.fit_transform(Xtrain)
print("n_samples: %d, n_features: %d" % Xtrain_tf.shape)

```

    n_samples: 1223, n_features: 12988


Let’s break down the code step by step to understand it clearly:

### 1. **Transforming the Test Data into TF-IDF Matrix**
```python
Xtest_tf = tfidf.transform(Xtest)
```

- **`Xtest`**: This is the test data, which contains the sentences (or documents) that you want to evaluate the model on. These sentences should be in the same format as the training data (`Xtrain`), meaning they should already be preprocessed (cleaned, tokenized, etc.).

- **`tfidf.transform(Xtest)`**:
  - The `transform` method is used to **apply the same transformation** (i.e., convert the sentences into numerical vectors using the TF-IDF method) to the test data (`Xtest`).
  - Importantly, **`transform` does not re-learn the vocabulary or the IDF values**. Instead, it uses the vocabulary and IDF values that were **learned from the training data** (`Xtrain`) when `fit_transform` was applied. This ensures that the model sees the same set of words and n-grams during both training and testing, maintaining consistency.
  - The result, `Xtest_tf`, is the **TF-IDF matrix** for the test data. It contains the same number of features (words and n-grams) as `Xtrain_tf`, but for the test sentences.

### 2. **Understanding `Xtest_tf.shape`**
```python
print("n_samples: %d, n_features: %d" % Xtest_tf.shape)
```

- **`Xtest_tf.shape`**: This returns the **shape** of the TF-IDF matrix for the test data.
  - **`n_samples`**: The number of test samples (or sentences) in `Xtest`. It represents how many test sentences are there.
  - **`n_features`**: The number of unique features (words and n-grams) in the test data, which corresponds to the vocabulary learned from the training data (`Xtrain`). This is the same as the number of features in `Xtrain_tf`, since the vectorizer uses the same vocabulary for both training and testing.

### Example:
- If you have **100 test sentences** in `Xtest` and the TF-IDF vectorizer has learned **500 unique words or n-grams** from the training data, the shape of `Xtest_tf` will be `(100, 500)`.
  - **100** represents the **100 test sentences**.
  - **500** represents the **500 unique features** (unigrams, bigrams, and trigrams) from the training data that the test sentences are now represented with.

### Why Use `transform` and Not `fit_transform` on Test Data?
- **`fit_transform`** learns a vocabulary and IDF values from the input data, which is done only once on the training data (`Xtrain`).
- **`transform`** is used for any new data (like the test set). It uses the **existing vocabulary and IDF values** learned from the training set and applies the same transformation to the test data.
- You should **never fit the vectorizer on the test data** because that would result in a model that has "seen" the test data, which leads to data leakage (using information from the test data during training).


```python
#transforming test data into tf-idf matrix
Xtest_tf = tfidf.transform(Xtest)
print("n_samples: %d, n_features: %d" % Xtest_tf.shape)


```

    n_samples: 408, n_features: 12988


# Model Building

## Naive Bayes

Let's break down the code step by step in simple terms:

### 1. **Importing the Naive Bayes Classifier**
```python
from sklearn.naive_bayes import MultinomialNB
```

- **`from sklearn.naive_bayes import MultinomialNB`**: This line imports the **Multinomial Naive Bayes** classifier from the `sklearn.naive_bayes` module.
  
  - **Naive Bayes** is a family of probabilistic classifiers based on **Bayes' theorem**. It assumes that the features (in this case, words or n-grams) are independent of each other, which simplifies the computation.
  
  - **MultinomialNB** is a specific type of Naive Bayes classifier that is especially useful for **classification tasks with discrete features**, like text classification (spam detection, sentiment analysis, etc.). It works well when the features are counts or frequencies of events, like word counts or TF-IDF values.

### 2. **Creating the Classifier Object**
```python
naive_bayes_classifier = MultinomialNB()
```

- **`naive_bayes_classifier = MultinomialNB()`**: This line creates an instance of the **MultinomialNB** classifier and stores it in the variable `naive_bayes_classifier`.

  - At this point, the classifier is ready to be trained on the data but hasn't learned anything yet.

### 3. **Training the Classifier**
```python
naive_bayes_classifier.fit(Xtrain_tf, Ytrain)
```

- **`naive_bayes_classifier.fit(Xtrain_tf, Ytrain)`**: This line trains (fits) the Naive Bayes classifier on the **training data** (`Xtrain_tf`) and the **training labels** (`Ytrain`).

  - **`Xtrain_tf`**: This is the **TF-IDF matrix** of the training sentences, which represents each sentence as a vector of features (word or n-gram weights).
  
  - **`Ytrain`**: These are the **labels** (or target values) for the training sentences. For example, these could represent categories like "positive" or "negative" for sentiment analysis, or different topics in text classification.
  
  - The **`fit`** method will learn the patterns in the data and try to understand how the features (the words or n-grams) in `Xtrain_tf` relate to the target labels in `Ytrain`. Essentially, it "trains" the model to recognize the patterns in the input data that correspond to different labels.

### 4. **Making Predictions on the Test Data**
```python
y_pred = naive_bayes_classifier.predict(Xtest_tf)
```

- **`y_pred = naive_bayes_classifier.predict(Xtest_tf)`**: After the classifier is trained, we can use it to make predictions on new, unseen data (the test set).
  
  - **`Xtest_tf`**: This is the **TF-IDF matrix** of the test sentences, just like `Xtrain_tf`, but this data has not been seen by the model during training.
  
  - **`predict`**: This method uses the trained Naive Bayes classifier to predict the labels for the test sentences based on the patterns it learned during training. The output will be the predicted labels for the test set.

- The result of the `predict` method is stored in `y_pred`, which will contain the predicted labels for each of the test sentences.


```python
#naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(Xtrain_tf, Ytrain)
#predicted y
y_pred = naive_bayes_classifier.predict(Xtest_tf)
```

Let's break down this code step by step to explain what each part does in simple terms.

### 1. **Confusion Matrix**
```python
print(confusion_matrix(Ytest, y_pred))
```

- **`confusion_matrix(Ytest, y_pred)`**: This function from the **`sklearn.metrics`** module computes the **confusion matrix** for the predicted labels (`y_pred`) and the actual labels (`Ytest`).
  
  - A **confusion matrix** is a table that is often used to evaluate the performance of a classification model. It compares the predicted labels against the actual labels and shows how many times the model got each prediction correct and how many times it made mistakes.
  
  - For a binary classification (e.g., "positive" vs. "negative"), the confusion matrix will look like this:

    |                 | Predicted Positive | Predicted Negative |
    |-----------------|--------------------|--------------------|
    | **Actual Positive** | True Positive (TP)  | False Negative (FN) |
    | **Actual Negative** | False Positive (FP) | True Negative (TN)  |

  - The confusion matrix helps you understand things like:
    - How many times did the model predict "positive" when the actual label was "positive" (True Positive)?
    - How many times did it predict "negative" when the actual label was "positive" (False Negative)?
    - And so on...

### 2. **Accuracy Score**
```python
print(accuracy_score(Ytest, y_pred))
```
![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApAAAADNCAIAAACAWNcHAAAgAElEQVR4Ae19/3NU13n+5x+40ljUy9ARHvAIa5aMsMZgxohYCY69zMYjOzFmImxAFpNcM2AcOygUxm0YC8KkxV0Ta5pEzRLHv8TmywQzCRdbJKorCi5j7MUEiJuU2hB5FmRRIKxEWejeD9XTPH1z7hettFpJK179AOeee8775Xnfc557zv2y/8/VP0VAEVAEFAFFQBEY9wj8v3FvoRqoCCgCioAioAgoAq4StiaBIqAIKAKKgCJQAggoYZdAkNRERUARUAQUAUWg6IR97dq1S5cu9eqfIjBGCFy6dOnatWs61BUBRUARKHUEikvY165dG6NZWtUqAn+GgHJ2qU9Var8ioAgUl7B1bf1npKEHY4fApUuXdLQrAoqAIlDSCBSXsMduflbNioCJgO9ATaVSkUjE+vO/aDSaTqcdx0HB6JhOp6PRqG3bRn3QYSKRsCzLV1RQlyHVw4VEIjGkXrdm40QiEY/HM5nMrel+kNccBY7jBLVxXXdcoec4TngoM5lMPB4vZFxgpPtiYtt2yIgO6SjhTSQSkUgklUrJyvCyErY5revxREUgfCSkUqm6urp0Os1mQYTNBvkURkSIoQgzke88YrQs6cNEIpH/VVGQp8a0Pq4oJ8jmUa4PYZfxjJ5hmxe0ohK2V52sCYFUNhtGWQm7BOjpnXfemTt37qlTp0rA1lE3sampafPmzVDb1NS0Y8eOIBPCh0fxCDt8HRBule9ZJWxfWHwrjWldCduLUjqdrqur813njWf0DNu8filhezEZpCZo6tT6/BE4depUdXU1eEiWIaGpqenBBx88O/D34IMPhtDVoBp37NgBUYO2HFcNJGGfOnVq7ty577zzjq+FGMBBi7YgwnYcB3vm3AGTEwHK2FD3EjM2w3HWtu1UKhWNRjkzyokSRLJz505fUbZtoz4Sibzxxhvcw8d+mnE5z+1Ny7JoEm2mqJCtQraR+3WstCyLi3vsHySTSfooAWEzwzsi6bquPfDHeQSrainEsiyYKisNv3xBg3CcsiwLcTcsoRzXdX3l0zDXdT/77LM1a9ZMnjzZsqzp06fv3r07l8uhwe9///tFixaVl5dblrVw4cILFy64ruutNLZA5SHKW7ZsmT59ejwev3Tp0o9//OOZM2dallVeXr5u3bq+vj7o6uvr+853vlNZWWlZVmVl5f79++8d+Ovt7UWD7du3l5WVHThwQBrvui7yBIAwBDJbWImOMuKDoifl+KZWnqkSZCdMchyH9ieTSRk+WksvEFBfY+ggz8J+Zqxt24lEAojt3LkzHo9DLxsYV35ypDuOE9JRBkVuv9E1Ob5kY5QDV9gnT56MxWKWZcVisZMnT3p75lPjO29q5ZAQkITU29u7Y8eO6upqrLblyvvs2bNK2MAn6LLDSw8yh30Jm5yHwY9pS04Ekm+SyaT35qhcB4QTNkkFo51TiW3bnJi6urpSqRQM4NyB9jjE5QWvCdgXXTgXGM0kDuziuq5UxwsdObVhlsEpTtmwJJFIcOrEdEYJ8pQEEHdJZTOWJeYyjrL7oPhDvi/OQfIlMseOHVu2bNmBAweOHz/+pS99qbKyEhPj2bNna2tr//Iv//Jv//Zvd+3atXbt2vPnz/tWSoaGMbwkAkQLFizo7u7G1YNt29u3b//3f//3F154wbKs9vZ213Wz2eyaNWssy/rGN76xa9euF198sbOzc8uWLbfddtvhw4dd171+/frSpUvr6+svXrwojZdRg2pGR144yi6u68rsDUFPJrbMRiktz1QJsVMmLbRwXMg0IJUaMZXGoCy9g3kYdOhI3pUxImjUYoCZSqW6urpgnm9HaQYJWwIICbKZLAcSNtgalxWxWEz2yb9sMBMWiJBJ1unt7QXZoF7Otk1NTUZjyV4s79ixo2ngD43lYtTo7qtr8+bNVApLuMVq2I9DWnX77bdzMefrwo4dO2DAzWeOuPZtamr6/ve/X11dbVmWBIFiaUxvb6+kZBoDZ3t7e4nA5s2bqciyLGmYV6w0FQbIuEBOU1MT1eVTeOedd6qrq4mGhNTbffPmzRs2bHjwwQelqWfPnn3ssce+//3v33777ZZlEQRpm4wL/bIsS9bDO6IttYfnrS9hc8hh8sJwlRMBrsRDJMtJQQ5LrCS4FSk5DFMAWdC4s861oJewpWEwiXMxThn8Rwm03wuCdJzNeHeZMw5OyXmTqo0ZjUsoaJddpONGWcLoui7tHBL+Xks47QbJp8tGYe/evbz6aWlpmTRp0qFDh2Qb38pBCXvfvn1SCMrd3d3V1dWIXWdnZ1lZ2aZNm7i4BxqRSGTLli2u63788cdVVVUoS1EGzuQkIw9lF1/CJmMBTFCmEQWmh5SWZ6oE2enNbYaMyQB1TDxvF2kPcOOOF/TCHQoE7/LSWV6LMHOoTgoP6SibERNqlGd9y4GELQnAsizfzoNWyunSKG8e+CODehmCm72yIylK0hV4EXO0775uuC65j+pLkNKAd95557HHHjt79qysBEkYLhhLYfJZU1MTeZru0MLe3l5ZxrWI1NXb2wuDwXm0xJeopCiWfWWGr00NA3wP6YuvJbILLi8Qr82bNwM39AJPUwJYHNcBssxLFpkGVEGZrEEhPF29Y4bDCR15KCcCXJjLWczQwmnFmCCMiZLjH915yIIUK2dbyX9yQkF7tpQ2k/I5E1G4rzpvJZ0iJpAgJ245kRkSaJVcK0OCnOuNsjEj4VpqSPhLjqE6zNFySxOK5LUaGvf39+/cubOhoWHGjBlo4zgOfInFYleuXEEzwmtUQrsUK/lbll3XzeVy77333tNPP11TU1NRUcFdATR7//33qct13f7+/q985StwZNeuXVOmTOEWC5oZ0UclgyUjJcX6Eja0oBnC2tPTwx1jxkg2Q+N8UiXETm9uyySkXhYYGm+S00fmYTqdjsfjx44di8fjeD0El0eGUnnIlDb8gnDZUo5QqjYwgSW8+2M0k4eBhF2MFTYmYmKKmdqXIyWJymmXrCBnal+Szl+X5EiymlQqy1jwySVs0DpYiiIDSbMp1jDVsixyvxTC9jBYrtp53SNXlkFi33nnndtvv50XDRTrCyPPDlrgpQwLQV0kobKxhIgdYSoTBi4buSFTAh2Drkhk3nvLwyNsyDG2fKVwTivDJmwuiymWEw1qODt4Z162RIGTl3FIyZIjQyrplDFbkQPCL0do1ZAI28sB0kJSGitRoKk45DxrHBr1hhAw6Lp168rLyzdu3Hj06NGf/exnWGHDF8M238ohEXZnZ2d5efmiRYs6OjpSqdRdd92FNDB4nXZu3769srLyN7/5zcqVK7/yla/09/fzFC8gjA0VBsubNuybD3ogbKYW+xqFfFJFJga7w06vkbTNkMyOkBZuGBLecRzbtqldIhONRokbB5q88qMZ1OtlaNlRNjMsRzN5SScboxxI2MW4hy13SjlrF4mw89dFxj169Ohjjz3GfV3ShrdALgRB+roguVaykZdg5NrR0CWFyFNejVIFWoaIxTK9urpaXnkUSNg0IMhm2s/QA3nsWLA7m8mzsnIcErbBT3KYyfFsTDpyh9wgDB4aQxqSObPgkNOBd4aiRuOUcUiDfdV5K8nrxinOdAYgdIcGy3sB8opE7ohSi++2PG32ypSnJP5ynkUbGmY4IiWg3Nvbe++995KY29vbQdi5XG7lypXGJOtbCe282ZzNZpcsWcKOBhO3tLTw1MmTJysrK4HSvn37fNdhn3zySXV1dWtr68yZM7dv3+61XwJrUDiTxNsrT/QM4V453ggGpYohiqnuzVjGTo4jqdrbRZ5FGQ62traClROJRGtra2NjI7YoOLLQWB6Ga5ctvfxNM3yzTqY9W7IQSNhsUUhBzrNYXGLtiHWq3AuVtyHRS257Ug4rsQ2OXr5Mw5b56KJtXNpSY0iBzASyMVyQW+Ky7CVsaOddW6kxaLHoJWzftbvvbQUpX74H5SvTdzUvJcgyAvHggw+Gv4EmCZto+BI2YmcAK9HG7rrRgHGRtuEZWmM6kLk9jBV2JpNpbW3Fg2ZBs4ac8jCDYOZFmZMyxz9M4qHsYjwFxqWDnB3kgzly/WpMXsYhcQhRR1qVnhozTtAsjB0IGiyjIA2WzyUZ5GoYlk6nN23alCf+8oYFgQ3HGfIJCxmusrLypz/96fbt26dPn8572EePHq2srKyqqvrhD3/Ih858K7u6usrLy+fPn3/zQv+ZZ56ZOnWqTACW4btlWevXr//FL36xcOFC7h9cvny5oaGhvLx87dq1e/bswUNneNasubm5oqIiGo1+/PHH0nKUvQ9z8eIjnLDzQc8Q7gz8GTbkmSqGKBkv+ZwHmtEF+aQks0ImeRALYuwQeUOsHFkG70rDpHb50Jnv0lzCQkwkYnJ0yMYojyphc4fz9ttv37BhA9kR8zJ2PslbmJdRyf1bSnjwwQc3bNgQQthsmY8u3ME1NpmN6R6H8jkymsoFq+ECnwWTC1lSlJQvnZVmBJGob710mfd98WwXDONtYxzKvXcYwye5GBoiY5CiNJ5leW3ESm+BsEgDfAkbi2w8hiaf0aOnAFPaFiSnGIQNRgSYHPPGGJOEzeeDLMuKRCKO48iFJmcfX66CFs6e4DYoNaYVnEJ7sqycvEg/ZFBpM1oa6mSl9JQzDiSEEHY8Hm9tbYVY6anEEN1pM/zigpKHQA9rIL7MI63ydQdi5Txr4OwrX4r68MMPZ8+ejRe3uMJGg3ffffcLX/gCXsFavnz5pYHv4Hors9ns1q1bKyoqygfe1Nq0aRPNNlbYly9fXrFiBd4f++lPfyq/rHfu3LlVq1bhFbJ77rmHt6sPHDhQVlbW3Nx8/fp1aTbLYCNvCEIIm3EfFD2ZdUZ8YUCeqSLHCN/RoAt81CAej+NtK1wu0054B6ZEJZJcZialoSDpFjnAcYFDX941EsnIw5COUjsxkaFh/suWLI8qYXvn7vFTIxfB48cq33XzKJsHGpZ3x4MMyLOlXGEHiRp2ve92C6Qx6bUwyggYs9soa79F1L3xxhtlZWW+z5nfIggEuZnJZLjLHdSmVOqVsP9nMseyTC7Uhk0YI94xTxYccb0QiDV3PmyNzXO56xBkUvEI27i9bRhQKmNy4tmphF28mGYymba2tldeeSUSiTQ0NFy+fLl4ukpUcjqdxjNlJWq/NLu4hF0Sv9YFTpKbwFjXYndF/jtWjM5HqQ0GGj+HuFPAOxcwTG59E0ZsYhtoj5Qj8pa8IRO7lDL1tTxqCChhFw/qvr6+hoYGy7KWL19+7ty54ilSyeMBgeIStv4etkEbejhWCOjvYY+H6UZtUAQUgUIQKC5hu6577dq1klhnjxWRqN5iI3Dp0iVl60LmCO2rCCgC4wSBohP2OPFTzVAEFAFFQBFQBEoaASXskg6fGq8IKAKKgCJwqyCghH2rRFr9VAQUAUVAEShpBJSwSzp8arwioAgoAorArYKAEvatEmn1UxFQBBQBRaCkEVDCLunwqfGKgCKgCCgCtwoCSti3SqTVT0VAEVAEFIGSRkAJu6TDp8YrAoqAIqAI3CoIKGHfKpFWPxUBRUARUARKGgEl7JIOnxqvCCgCioAicKsgoIR9q0Ra/VQEFAFFQBEoaQRGlbDlL4qXNGpjbjx+yD0ajabT6TE3ZpQNQBbhV+UTiUQkEkmlUnnaYNv2hAHNtm3LsuLxeCaTGdR9/l7WqI3BVCpVV1eH/BxSmNLpdDQatW17UKfGQwPbthOJxHiwRG24FRAoOmE7jsM5ZdQmi2JHLpFIjOGE4jjOOGSdUcNEEvaggU6n03V1dfkz+qACR7yBHCD5CycB59mF7UdtDErCHtTI4YEwqNhRaKCEPQogqwoioIRNKIZQGDVy8rVpfM5uo4aJErbrukNFWwnbdygVXqmEXTiGKiF/BIpL2Ni1swb+bNvm1T3r5W5SKpWKRCJoLOvpDFaWjuOgmbEX6jgO+lqWhf1SzGvxeLy1tdWyLMjEhhtacukf0nfnzp2yMVygIsiU3aXl9CgSiSSTSe4Quq4ru9BaeooCu8udT2yGwwDfVT6xlfiwUoIDPJPJpGVZtm0DmWQyGR34w2YmOxpretZHIpHDhw/H43EDE/oCsTt37mQbQpS/Rgl7a2trPB4HaMZmg0TMtm0JMuwnb8E8eiFh8RrMAOFUCPiygUTM0MtLLmmAbzRluImb7MVKAi695lkawDHI9ih4vWZfI09c15Xh4CByXZf4GwlvhEk65TiOdEfOErRQNmAsIJOzgURbIsD2lCbdQeZLd9heVkrhruvSHnvgj0DJLhIWqtaCIlAgAsUlbDATc5cJjVGBwYbtylQqFY1GUcbcwZFDDzEOKU12lzOCFIWpgSMKknmYTCYzmUx4X0yjRkdjfbNp0ybQm1SNyQtewHEO+yCN9BS4GYwrHWdZdsFUwlNdXV2pVAqqSQbSKuDJU/CRRkIaz3LG99USsuaDWPoikcxTo+ECYsoUosHStUwmk0wmXdc1tsTphSFT9jUMTiQSUIEuSB7Kl/hLIQCEtlEv2pOwjQEipRk4S9xC0HZd1zcbaYD0QqozvJbqjDwxJIC0yNaDJjzxRJeuri4DBCkfZSahRFjOBrKZMQYhX3oq3YFAXq4ZtiGFjIFg2zaHGEQxJeLxOOcWwiJVa1kRKBCBMSBsDj85Mo2dJYMR4aRkaF7mJxIJyJEET2lyBPpOc8Pr62seTYIlxojlLb0QjYylRAaVknjkdM8umP7kIp6gkTZQQ+MNPOUc7ZVGA+iIVO2LLRoYYtESU55xypBMjfL6xgBZnmLQpWEUQt+hWnbkKWSmYRUleAMnFRnTumEn+RJdZARlWQqUxOPtxSDKLkZZGkwDvKmFXobXMkxGnhgGM2pBCQ8+RhISTMNUKVNaGBKmIKtoj6GCh4ZMafag5nmDwqyTLniHD7VrQREoBIExIGxehHJkosA9VRR4GUv3jJGGKTKRSGCuMbpj8uU8xQmU2iE2z75y/vKSk2G/4zh0jcZzHgnRyMZoIy9BIBA1xtTAXtLZkEp2N/A0lGL1YKDqOI6vFi8mNMAQK5dTxqk8NRpQkAmi0ahEDAYYUzCNZ4F2EhbDKnmIxb03M4Oyi7O5oY66JBq0BAUjOgYHhBC2Nxtl9nozk0AZANJIwxK5oY304G0RObiY8JKwDVF0mboMJA3cJFyGKB7SfWkMFUljUMkYefdjZEIGXeexuy8s4/lpR4mJlksFgXFE2EEDjFByTLIGo0VOqTyFghzwcqJnszz7yinPICfsqsF4qmCBijh/hWhkY4NmOIsNStjcvaAo78zOydHA0zDMOBsiEKe8ilBviJVzrnEqT40SW3bxIkbt8ilx5oPXWsJiWGUc4jKRj0QQFmkVK+VsLmmeuiQa7IWCbIMappCRgbKjbzbK7IWd3rHmdZMGEGQoIoZSr9d9aS0lUKbsa4AgLQwJE2VClHEId3gjRqozWjJGBmHLNwBpNgsUyO6+sLCZFhSBEUFgXBC2dzvR1zdjpHGOkCPc6GiMIrn9hZb595Wi5DwiyzTJ6xGND9FI471tJCF5Zw10pArK8a4n5FxvtDembO/uX4gWKVZqxyRoLN2I2PA0yl50wYsYzJC4Sd5iR1obZJVUx8aSjVhpZJdMBiNkMpeMU5TmxV+2pLVsj4KslwZQYwhQQWEysDIOaYDhvmzGstcpdJeuSQvZkVrooHHKOPSiQQlGSzKuJGwjbWieYX+4qdSoBUVgpBAYDcLmPVSZ31w14mJfPk4CmvFub2KHiotIzkFoL6+mk8kkNqNkG2wqRiIRSuZDZ/n0laJkWY5/43koisWkTxyM22+0VgbVaCNnQ04fsj3xJD5BD53x4T5puS+zyudrMplMa2trZuAvHo8bWiQXGlYZvsspz8uFvhrRjItCPKPLzQZfVPlQmKGCgUMq0osQqyghnU5v2rQJ3hnQodLIYeoyEg/NuOD2FQWBEg2agVPkLRzyXylNZiONMcYgO0I+wZSASJneNCMsMmMNaVKCdCqVSvGhM6qWFoaEScqUF6bOwB/8kqOGnhodfQlb2gBfEC/DHiAsd9eYUYRFIkkbtKAIDA+BohM2Uly+PsGZV44KDDneMeV0Jr3CSMNrSJZlcYSjjbyHxGHDeYpyMF1CkWxG1bJSmiFFYQxzX5Svecg3jqRHWLvIJ8J8raWRKMhbaLQKYqVhshfRlvjISl5DyDkOEuAUr2Y4NRMZnpICGQUDE1qFerwtBlGUk79GCrcsC2+IeQlbAs7Q4EqCaMggSi8kLIZVPJTt6TXdREFmlxEjRjwej8MFfKGMYmWIKZapxSeZcSqIsLljb1mWzEY6DnUcg1QEN33DZDAcr+0QTQkdM9ZIeEMCnWJfCYJhIU9ZlsX23uylChkCX0jZEr77EjavsZA5yWSS0ZT2IAoEU2YpTVXCZo5poXAEik7YhZtICcZIY31JFIKWxSVhfCFGkvAKEaJ9i42AhqnYCKt8RaBwBJSwC8dwcAmYDXklPniHCdRCmaAkgqlhKokwqZG3OAJK2MVKAG4PGvvAxdI3XuUqE4zXyPyZXRqmP4NDDxSBcYlAKRH2uARQjVIEFAFFQBFQBEYDASXs0UBZdSgCioAioAgoAgUioIRdIIDaXRFQBBQBRUARGA0ElLBHA2XVoQgoAoqAIqAIFIiAEnaBAGp3RUARUAQUAUVgNBBQwh4NlFWHIqAIKAKKgCJQIAJK2AUCqN0VAUVAEVAEFIHRQEAJezRQVh2KgCKgCCgCikCBCChhFwigdlcEFAFFQBFQBEYDASXs0UBZdSgCioAioAgoAgUioIRdIIDaXRFQBBQBRUARGA0ElLBHA2XVoQgoAoqAIqAIFIiAEnaBAGp3RUARUAQUAUVgNBBQwh4NlFWHIqAIKAKKgCJQIAJK2AUCqN0VAUVAEVAEFIHRQEAJezRQVh2KgCKgCCgCikCBCChhFwigdlcEFAFFQBFQBEYDASXs0UBZdSgCioAioAgoAgUioIRdIIDaXRFQBBQBRUARGA0ElLBHA2XVoQgoAoqAIqAIFIhAqRK2bduW5y+RSKTT6Wg06jhOgbiUVvdEIhGJRFKp1Aianclk4vF4IpEYhsxEIhGPxzOZzDD6hnRxHKcYYkM06ilFQBFQBMYPAqVK2ETQtm1JKrcmYRONESwUlbAhfKjXVUrYIxhfFaUIKAIlh4ASdsmFbJQMVsIeJaBVjSKgCCgC+SEwMQl7586d8XgcW+ZyGec4DvfRZT2xAkuhTTQaTafTPJVIJLx9uTOPHelUKhWNRrk1nU6n6+rqcIhd4tbWVsuysCVg9KWiVCoViUSgy/7TnzxbV1cnDXNd13EcWjuoj9iEgHzuMHuN8RK20WZQZ7ElLu2B49JB7uTLSrllwu7RaDSZTNJgAuK6blDU2NeyLMdx0My2bdlXy4qAIqAIlAoCY0DYJ0+ejMVilmXFYrGTJ08WiJTvljhpIJFISCZj2SAb2pBKpZLJJA7BlShLOalUqqury3Vd27bJH11dXamBvxDCJlW7rptOpzdt2kThlAPewsVEJpNJJpOSjF3XTSQSXsphG+kX7aR30BuNRsmIyWQyk8n4GmMQ9lCdpUebNm3C5YW0DcJ5zSRPyZsajuMwlKinWOmUb9SkTEIhYyolaFkRUAQUgfGPwBgQNtgaK7xYLFYgRr6ETULi7G8wBOiWzXxt4B1TuVBmy1Qq5V3pSpIAO8oVNi8XKAQFKcpwhxRLCo/H4+Q5ypGE7bWKzYL4XjagMZKwWWm0DLk68TKrDIEse2OBixJpAPQyItIMo8w2vjYbjfVQEVAEFIESQmAMCJsbyygUCJbBcGRoiOUhCoZq71IV28tsBtYhHUpTfR+EDidsg8PkJjC4nNZKReAzmEo2MhrQQpCcXMrLll4K5FmvMbJxIc7SJKDqvfIwGqBZPB4/ffq08cB/kPu+UaPY8MsyIqAFRUARUATGOQJjQNijsMLmGpQUyEJ4PGzb5jqY9MCC7Ou7NZ0/YcsXsbgW9F3Ku66bSqUaGxszmYxxdUJ7SNiogbPcTGYzcBjBYb2vMWgMthu2s7gOgBCp3Vv20qoXDd9A4ILGG7VwKOi7FhQBRUARKBUExoCwR+EeNjmJPC3pJyg2BkOQHgwaRneDI1FpSJAd5SJV0hX4GPvYQUZmMpnGxkbHcbjBbrjga4wvy3pv4uZjjK/8fJyVNkhFsiy3EKRfXjQkhmxpmMGosUE+NwJkYy0rAoqAIjA+ERgDwh5ZIIxFJxkaWuShfILJdd1kMsnHudFYMgQ6chNbPnKFJ5jQmJvqeOhMVqLMZa5BNiRONJMLRHbBQ2ewLZFIzJs3j/YYGJJQnYE/nKUK2Vg+1AYQsHCHI9IYlOXieBjO0jCwJp7W5nPdXFUbVtEL41m/SCTiRUDaKaNGIfKCwBcTiY+WFQFFQBEYtwjcQoRNzsBdUtKPjA1v5XpfIjJeaiLrQBoZlxIikYhcExuEDWqxLCsSiSSTSfmkmHwZKYjSpM3ytS5qtyzL10Es6OVrY3yuzTBGEmEhzhK31tZW+cQc3OTVifRasjLfpovH43hbz/sBNXoto8ZKCYUStpE5eqgIKAIlhEDJE3YJYV2IqbzPXYgQ7asIKAKKgCJQuggoYZdA7LDYDVoxl4ADaqIioAgoAopAwQgoYRcMYZEFYEtZ2brIMKt4RUARUATGOwJK2OM9QmqfIqAIKAKKgCLguq4StqaBIqAIKAKKgCJQAggoYZdAkNRERUARUAQUAUVACVtzQBFQBBQBRUARKAEElLBLIEhqoiKgCCgCioAioIStOaAIKAKKgCKgCJQAAkrYJRAkNVERUAQUAUVAEVDC1hxQBBQBRUARUARKAAEl7BIIkpqoCCgCioAioAgoYWsOKAKKgCKgCCgCJYCAEnYJBElNVAQUAUVAEVAElLA1BxQBRUARUAQUgRJAQAm7BIKkJioCioAioGr8qGAAACAASURBVAgoAkrYmgOKgCKgCCgCikAJIKCEXQJBUhMVAUVAEVAEFAElbM0BRUARUAQUAUWgBBBQwi6BIKmJioAioAgoAoqAErbmgCKgCCgCioAiUAIIKGGXQJDUREVAEVAEFAFFQAlbc0ARGD0Estns+fPn0wN/ly5domJZn06nr169ylOFF06fPr1nz55f/vKX//mf/1m4tCFJuHHjxltvvfX8889v3rw5SHtfX9/Pf/7z559/fs2aNT/4wQ/++Mc/5nK5t99+W+IzJKWj0DiTySCIxr+ZTGZEtPf393d2du7Zs+e9994bEYEqZGIgoIQ9MeKoXpQGAmfOnPn2t7/9yCOPWJY1bdq0U6dOwe4zZ86sXbs2Go1WVFQsW7bsww8/HEF/XnvttQceeCASiaRSqREUO6io69evP/fcc08//XRHR0d5eXlbW5vRJZfL7d69u7Ky8mtf+9q//uu/ptPpVCq1ePHi73znO7W1tel02mg/fg47OjpWrVpVVVVVVla2ZMmSNQN/y5Yti0QiLS0tfX19BZp65syZZ599NhKJ2LYNUf39/U8++eRdd931+9//fqjCC+k7VF3avqgIKGEXFV4Vrgj4IOA4zlNPPVVeXr548eL+/n62cBxn69atPBzBwvvvv19dXT3KhJ1Kpe68887Dhw9/9tlnW7Zs+c1vfmN49Oabb952220vv/xyLpfjqcuXLy9cuDAajY4tYafTadu2w1fMtm0bl0HvvfdeJBJZs2ZNNpulR8Mr5HK5FStWSMJevHjxnXfe+bvf/W5QgWvXrpWx7u/vz7/voMK1wRgioIQ9huCr6lsUAcdx9u7d+9xzz1mW9corrxAFx3ESiQQPR7CQSqWi0aicxEdQeJAox3EMPpMtT506NW3atK985SvykgUNDhw4cPfdd48tYadSqcbGxqES9sWLF+vr66dMmXL8+HHp7PDK9sAf+2az2WvXrvEwqJDJZBobG41Y59k3SKbWjxMElLDHSSDUjFsIAWfgr7u7e/bs2ZWVlceOHYPztxRht7S0WJa1Y8cOb+A//fTThx9+eAwJO5vNrlmzJh6PD5WwM5lMPB4PuUzxOhtSYxB2SEt56s0335w2bZpB2LKBlksXASXs0o2dWl6qCICwXdfdt29feXl5Q0PD5cuXXdeVhL1+/fpoNFpVVYWZF4dcJfPs/v37V65cOWvWrKqqqtdff/2TTz75xje+MWvgb/fu3dxqxgq7ra2tsbGxtrb2nnvuaW9v57ZtX1/funXr6uvrH3/88fvuu+/gwYOu61LFSy+9NGfOnHvvvdf3znpfX9+GDRseeugh27br6uqoNJlMTp061bKs6dOn33fffSdOnJDRunLlSiwWi0Qi77//vqxH+erVq6+99hoeOstms6+88soDDzxg2/b999//yiuvwGxf8zo7O6PR6NSpU1taWhobG6dNm/bjH/84N/C3e/fuOXPmNDU1zZw5k0Jc1z1z5kxDQ0N9ff2yZctqamo6OjrOnDnz6KOPWpZVXl5eXV3tNZ4Ge7fEP/7446qqqrlz5/b09Jw4ceK+++6bPn36k08++fzzz1dWVm7cuBHGHzx4sL6+/oknnqitrV23bh3veXd3dzc3N0ej0c9//vNbtmxZsmQJt8TpL5n45s7Exo0bP/e5z61YsaK2tnbjxo1//OMfW1tbKyoqAHs0Gk0mkzKU7BuEajKZjEajkydP3rVr13e+85177rnn5nMVhCubzba1tS1atGjNmjVf/vKXFyxYMIYXVYzCLVVQwr6lwq3OjgsESNi5XG7dunU3uQH3cSVh37hx4yc/+QnXatlsNplMGoeWZT366KMg+/b29oqKiqVLl/Kwurr6k08+gcOpVEreWz179uzs2bO3bNmSy+Wwmly+fDm45I033qitrf3DH/5w48aNN954o6ysbPny5YcPH45EIt6nxvA008aNG2/cuOG67rlz5+6///6f/exnUBqyJZ5Op6PRKN0Jikoul9uwYcPy5cuxbZ7JZBYvXvzd7343l8sFmffRRx/V1NTMmDHj2LFjDz30UENDQ19f35tvvllTU3P69GnXdU+fPj1z5sy9e/e6rgscAP7x48enTJnS2Nh47do1LJTzXGH/6le/wrPix48fX7Ro0axZs3hl09PTE4vFJk2a9M///M9f//rXa2pqzp8/f/To0Wg0eujQIdd1L1y48IUvfAH3QWDMSy+9BDBPnToVjUZJ2N58WLNmDS71+vr6Ghoapk2bhufREomEAazRNxzVt99++7bbbps7d+7Zs2dd1z1w4MCUKVOOHDnium5nZ+fDDz985coVWL58+XIl7KDULVK9EnaRgFWxikAgAiRs13V7enrmzp1bWVl59OhRSdhYcMuZ1+A/x3Esy9q3bx/U3OxbVlZ24MABHsq+IGzHcWjT1q1b77jjjo8++ujIkSOTJk0CgbmuizUiDtFr7969uVzu4sWLIBJKcF13165dkydPlqvkrVu38kLBMFh2zJOwDdtc1927d++kSZPAH77mgWtXrFiRy+UymczVq1cvX768YMEC1Liue/369aVLl+KwpaWlqqrq448/dl03m82+/fbbeKQrf8KWT4m3tLT8+te/5r4F/LVtOxaLXbly5erVq5lM5vr1683NzahBgxdffBGHGzZsqK6u7u7uRj1sIGEb+dDZ2VlWVsYbCh9++GFnZycC5CVso28+qPJZCpk5juNUVla+9dZb8PHmcxjj+dU7mW8TpqyEPWFCqY6UDAKO42zfvp3mHjp0aNKkSQsXLnz99dc5URqTbIGHctqF3r1791qW5ThOW1ubZVlz5syJD/zFYrGamhpspaZSqerqasnHtBmF1atXy8sC13UTiURZWVlnZ6fXYNn36tWrixYtuu222w4fPizrWQYZwDZ5nYHLFKDnax54Tj5sj6Xz1KlT4WA8Hp8zZ85TTz312WefxeNxyZ3UDvPyXGFzn5ndZcG2bV4ruK57/vz5mpqaSCQSi8VgT11dXSwW++STT3DIu+bhhA1W9g3N9u3bjaAYsRgU1UgkQsxl5nR3d8+bN88a+Js/fz5unUhntVxsBJSwi42wylcETASMlXQul9u0aZNlWfPmzRs1wgbzkbA5QUtbB3223HsTN5FIWJbV0dFhkIQUi3J7e7tlWe3t7d5TV65caWlpuXjxIqRJ22A2Nud9zQPPSRhB2HKpCo1oGcTKtm3jVDab5T1mw1Sv+0YD13WNB8dA2F6lXmPyIWzfawVubORyuStXruA5BlbiogrXarTWQDWIsF3X7e/v37Nnz4oVKyoH/vi8JEVpoagIKGEXFV4Vrgj4IGAQtuu6ePnYsizJNHKSxW6wXDkZZ8MP5ToJBrW1tUEaNki5ueq6bm9vLzZmfRlR+oPFnFzncad9UMLGvQA8nyVluq575MiR7373u7hpWlZWxu16gMAVvK95XsLGlvjSpUuvX78ORdls9re//e1///d/r1y5klviruvmcrnf/va32O8lYadSKew3GEaCjGVEvA28hI0t8fr6+osXL6I9lPb39zc3N+MmN+rDCXvfvn3GM/bd3d29vb0S9kwm09raiiW7TA9sp4egGkTYb7311q5du2CevPvu67hWFgMBJexioKoyFYFABHK53Guvvfa9732Pj3CjKb65IQm7s7OzsrISdJjNZleuXCkfq5ZTsJymIc04C8J+6aWXoBTXB/i+Bx46W7hwIZ5Wy+VyL730Em4SD/q5FZAuxWIS37Rpk3dV5wvHBx98cPNRqSeffFJ+tfTcuXOrV6/GE0/44sfKlStBotlsdvnAHw59zcPz5xJG13XxphM/83no0KGXX37Zdd2jR49WVlbSfnyHDg+4rV69Grvl+/bte/XVV33txwpbXq8YzYyPn+Ds0aNH77zzzt27d+Pw9OnT69evv379+qFDh6ZMmcKnEM6cOVNTUyOvM2RML1++3NDQwKhls9m//uu/xofzOjo68GDBp59++q1vfevqwGduZd9wVI1rO3noOA4fTrx+/fry5cv5CIXhuB4WCQEl7CIBe+uKzWQyTz311DvvvHPrQhDs+YkTJ2pqanAXcOrUqbjXi+a5XO7ll1/etm0be2cymebm5kceeeS11177+te/vnbt2psfWqmoqFg/8Ie3dyZPnozDyZMnW5bFQ57l3ejm5ua2trann366ra1t/vz5q1ev5k5vX19fS0tLTU3NqlWrmpqatm/fnsvl1q9fT5mxWKynp4eGycK5c+cee+yxhx9+eNWqVXV1ddu2bQObyu5PPfUUdcm+rut2d3c/8cQTkydP/uY3v7lr166/+Zu/+epXvyq/vtnX1/fMM8/U19evWbOmvr6eL0FJ+TSvs7Nz+vTpfCOL8OIbqNXV1cuWLVu5cuVzzz2HqxPXdQ8ePHj33XfflLBq1arm5mY+8/XBBx9UV1cvXbq0ubnZ63symZwxYwbiWFFRQQOkdydOnJgzZw7azJgxQy7TDx48eM899zz++ONSaS6X279//8yZM1944YWNGzc2NTXNmjXLsqza2toTJ07Q36lTp0LUuXPnnnjiiZkzZ65ataqxsZFMf/ny5a9+9asPPPDA0qVLMQy9fYNQTSaTCHpFRcVTTz21bds2vJuHwz179syePfvxxx9vb29vamqSKSQd13LxEFDCLh62t6hkXJKvXbvWWEHeonAU7PalS5fOnz+fzWavXr2KQiEir169mk6n+WSTFIVTWJDJ+nzKmUymENsymUxXV9cvfvGLM2fO+KZNIbbR/hs3bvT09HgfbM7lchcG/gzV2Wy2p6cH1x8UMlKFEKXnz5+/dOkSGnitNQzwRR59faMsuw8V1Zvr+Bs3buCHagYVLhVpeaQQUMIeKSRVzv8igAeFZs+eff78eQVFEVAEFAFFYKQQUMIeKSRVzv8gkMlkVqxY0dDQIN8JVmgUAUVAEVAECkdACbtwDFXC/yGQSqW++c1vHjhwoKysTHfF/w8XLSkCioAiUDACStgFQ6gCBALbtm3btWtXb2/vvffeO3PmzE8//VSc/J+i9wPIuEfoW298PxmH3u9pGx+77ujo+OIXv/jMM888/PDDCxculE8w3bhxo729/XOf+1xTU9P8+fNXrFhx+fJlfIA6OvC3fv16fnuZn2I2XNBDRUARUATGBAEl7DGBfWIqzWQyX//61/H9avwWk/HWB94g8n4AOaje+Aay8T1t369J4+1V/JoyPkjCN33xGPbs2bPxytDq1avLy8u7urpc1/3ss8/q6+sXL178X//1X3h0ecGCBR0dHd6PcabT6Q0bNqwJ/fvhD3/IV34nZqTVK0VAERgLBJSwxwL1CarzyJEjq1evBlfh4wwrV66UT94GfQA5qD7P14uNj12/++67+/fvh175+ulHH310xx13vPjii4D/P/7jP/bv34+Xbl3XbW9vr6ysPHnypOu6J0+e3Lx5s7ScEcPzt/ixB++/w3vEmsK1oAgoAopACAJK2CHg6KmhIbBly5b7778fi88VK1ZMmjTJ2BUP+gByUH0+hO392HUul/v973//4osvfvGLX5w9eza/RYXvL8oPPEn3QOf44Mbf//3fB33jWnYZRhlv5eq/isCEQWAYo0C7DBsBJexhQ6cd/wyBy5cvL1269P333+e685lnnpE/J4WPGJM+ZWcQdviHkdFerphd1/V+nDKbzT733HPTp08/cODAjRs3ZHt+PVuqZhnfjFywYMHvfve7lStX8sMabIBC+Aobvg/67qwhUw8VAUVAEcgHASXsfFDSNoMjIPfD0dq7Kx70AeSgeu8Ke+/evZLyvYRt7K6DsN999929e/ceOXIkEolwS1x+NBsGHzhwoKKiYtWqVf/4j/8Y5LDeww5CRusVAUWg2AgoYRcb4VtC/s2HszZu3PiDH/xAeut9VjzoA8hB9fj5h5DvaXu/Jt3R0WFZFva98VnsSCTyT//0T62trRcvXlyzZk1NTc2ZM2fwSw9tbW38eiV+gWPBggX8OWfpi5YVAUVAERhzBJSwxzwEJW8Avz9sWdYTTzyBr0bzq874wDW/Jh30AeSg+vDvaXs/dp3NZteuXXvHHXds2bLl2Wef3bZtW0NDw9SpU//hH/4hl8vho9l33HGHbdv8aLYMQCKRaG5uHpNnvC9dusS7Cb6FkX2i7caNG4cPH96zZ09XV5fXX3x+0rvDL+vT6fTImnT69Ok9e/b88pe/lL8FIqNTvPKNGzfeeuut559/fvPmzVJ7JpPxjcVIfZizv7+/s7Nzz549/GGS4vmokicAAkrYEyCIpeeC7weQ8aE0309SD/V72phnQSf4+rHEKOQTyn/3d3/HH1GQXYpdxs86WZYVj8fXrFmzZMmSsrKyadOmrVy50rbtu+66y/gB48LtuXTp0gsvvFBVVeX9bWbXdfG7VY888ohlWdOmTcPPQKF+7dq10Wi0oqJi2bJlH374YeGWUMJrr732wAMPyFsePFXUwvXr15977rmnn366o6OjvLwcP7YNjR0dHatWraqqqiorK1uyZAkeqFy2bFkkEmlpaQn6RZP8rT1z5syzzz4biUT4c90fffRRVVXVsmXLhnEx1N/f/+STT951113y2wP5G6Mtxz8CStjjP0ZqYXEROHHiRCwWe/fdd3t6elavXh30uFlRjchkMosWLeJPnOEHVEiluONg/GSkrz3pdNq27fzXf/zVZ19pjuM89dRT5eXlixcv5vtveLBg69atvl0KrPTe4yhQYD7dU6nUnXfeefjw4c8++2zLli2/+c1vjF74GU35UCR+CxW/T2o0Huqh8ROcp06dmjZt2pIlS/Ih7LVr10qr8LuZd9555+9+97uhmqHtSwIBJeySCJMaWUQEOjs7/+Iv/uLnP//597//fcdxiqgpWHQmk/mrv/qrixcvoolB2K7rHj58+Ec/+lGwgP89k0qlGhsbR5Cw9+7d+9xzz938Wc9XXnmF2h3Hyefqge3zL3ifIsy/77BbylcJfIV4CfvixYv19fVTpkw5fvy4b5chVdoDf+zS39/v/WIPz7KQyWQaGxslYbuum81mr127xjZamGAIKGFPsICqO0NGIJvNvvrqq88///y//Mu/+H4sZcgSh94hnU4///zzXFR5Cfv48ePf/va3wwXjg3Fcl4c3xtlBV9iO43R3d8+ePbuysvLYsWPopYSND+qN1O69Qdj5BM513TfffHPatGkGYefZV5uVKAJK2CUaODV7QiFw48aNK1eu0CUvYcsGBw8ejMVizc3NDQ0Nzc3N586dw93lRx991LKs8vLy6urq++6778SJE3j03bbtJUuW2LZdW1v7yiuvyB94zoewXdfdt29feXk5vimLLXGusPP83vv+/ftXrlw5a9asqqqq119//ZNPPvnGN74xa+Bv9+7dvE7CCrutra2xsbG2tvaee+5pb2+nwX19fevWrauvr3/88cfvu+++gwcP8sPvVVVVxiflCSYKfX19GzZseOihh2zbrquro9JkMjl16lTLsqZPn07QjL7eFfbHH39cVVWFr97iW/RTp05taWlpbGycNm3aj3/8Y3h08ODB+vr6J554ora2dt26dbzn3d3d3dzcHI1GP//5z2/ZsgXRcV23p6cnFovNmDGjoaGB2yRnzpxpaGior69ftmxZTU1NR0fH1atXW1tbKyoqYDY/em/EwnXdIK/ZknGZNWsWMclms21tbYsWLVqzZs2Xv/zlBQsWpNNpAxM9HBMElLDHBHZVqgiEIeAlbLbu7Oysra39t3/7N7yZ1t7eHovFLly4gEf24gN/nOtBrpZlgV9Pnz5dXV2dTCYpLU/CzuVy69atu3k18PLLL+dyObnCDv/eOz7/blnWo48+iocD2tvbKyoqli5dykP5Hh0c573hs2fPzp49e8uWLblcDvsHy5cvB3+/8cYbtbW1f/jDH3w/KU8HUcDTWBs3bsRW87lz5+6///6f/exnOJvnlvivfvUrPDF+/PjxRYsWzZo1i8/cffTRRzU1NTNmzDh27NhDDz3U0NDQ19d39OjRaDR66NAh13UvXLjwhS98AVGAUy+99BKMOXXqVDQa5UNnfX19ixcv5jYJGgP248ePT5kypbGxEZveiUTCWOIbsQjx2jcuDERnZ+fDDz+MK8gLFy4sX75cCdvIqLE6VMIeK+RVryIQiEAQYV++fHnBggUrVqzgkrS7u7u6unrLli1BhN3X1/eTn/wEq3Bs5JIbXNfNk7Cx+Js7d25lZeXRo0clYXs/bmPwHz4wx5+BudlX/la60RiOyycJtm7descdd3z00UdHjhyZNGkSvyyLNS4O0cv4pLwEd9euXZMnT37//fdZuXXrVvKTYQPbsGDbtnxKvKWl5de//jXX/UQecbn5OMLVq1fx4bxYLMaNkxdffBGHGzZsqK6u7u7uhvzwoLS0tFRVVX388ce4P/3222/zgTIvYRuxGNRr+SFCCYLjOJWVlW+99RZ8vPkcg368j8kwtgUl7LHFX7UrAj4IBBE21liScdPpdDQaXbRo0dWBv0WLFnFxRrl9fX0///nPly5dOn/+fPkGUT6EvX37dso5dOjQpEmTFi5c+Prrr3NL3CCJAg+9hL1371680tbW1mZZ1pw5c7CLEIvFampqsFuQSqW8n5Sn2Td/jW316tXGYjSRSJSVleGzOZKrZC+WvVviPIUCSFc+OX/+/PmamppIJBKLxWBwXV1dLBb75JNPjF2QEMLGKcn6Uu/27dsNpwzwh+S1BKG7u3vevHn42vn8+fNx60Gq1vJYIaCEPVbIq15FIBCBIMJGvZewsQcrCTibzeKO6QcffDB9+nTbti9cuBDCDb6mGCtp/FypZVnz5s0bNcLmF+BB2HLxTZsHfbbcy7iJRMKyrI6ODoPkKFMWvN3lWa6wJSYgbO/FE0Ig60OC4m0s9ZJic7nclStXsOnCSiSDweghXsuO+N36PXv2rFixonLgj88bSgO0PPoIKGGPPuaqUREYBIEgwsbXXr1b4i0tLZDILe5UKpVMJq9evbpo0aL6+nq8MEZu+HDgTxK8r0EGYeMRtoULF/KmOHoZc73xvXfjbPihd4Xd1tYG1sGW+I4dO2hqb28vNpYHJWwsRo0tcey0F4mwsSVO5PHAwW9/+9v+/v7m5uaamprz58/DEQaFfjGIuVxu5cqV3BKnEOxUE8lMJtPa2ooHF1jpuu6QvJYd33rrrV27dsEeefedFmphrBBQwh4r5FWvIhCIQBBhu66bTCbvvvtufA7ddd033nijurr69OnTkLV69WrsoO7bt+/VV1/t6+traGjgnuqZM2dqampWrFixb98+LFXJDV5Tcrnca6+99r3vfY/3y9EG3wyRq8nOzs6Q771LJvCyo3EWjr/00ktQevny5YULF+IZNDx0tnDhQjythg/FHzlyxHXdQT+30tPTM3fuXIoFCW3atMm7KvXiwKWq5Huj2ZUrV2KxmMTEdd2jR4/eeeedu3fvRuPTp0+vX7/++vXrhw4dmjJlCj+oh6AsXbqUH4iVQTl69GhlZSUtxxfo8BGbjo4O3Jj/9NNPv/Wtb+GdQInnkLyWHR3H4cN9169fX758OR9BMBzXw1FGQAl7lAFXdYpAGAJ43wbfSOdLO/IXSnK53Ouvv15bW2vb9te+9rWFCxfK71B+8MEH1dXVS5cubW5u7unpcV0XNY888sjLL7/c3Nz8ox/9KBKJPPTQQ8eOHfvSl75UXl5uWVZtbS3eAaNlJ06cqKmpwV3MqVOnGga8/PLL27ZtY+Pw773j7aPJkyevH/jj599xyLO8G93c3NzW1vb000+3tbXNnz9/9erVfBsKn4KvqalZtWoVPwW/fv16yozFYvCatrFw7ty5xx577OGHH161alVdXd22bduwTpXd+cV79komkzNmzAAOFRUVvvL52Xy8UCexOnjw4D333PP444+vWrWqubkZ+wG5XG7//v0zZ8584YUXNm7c2NTUNGvWLETh4MGDDMqcOXMQlIMHD9599903VUsh2O346le/+sADDyxduhTfyKMvU6dOBZ4hXhP59QN/xHD9+vWO48yePfvxxx9vb29vamqSISAyWhgTBJSwxwR2VaoIFITAjRs3enp6fJ/dzWazPT098hlmNL5w4QIWlNeuXTMWzQWZ8qfOQ/3e+5/6+f+P773L99PYLuRT8GwTVAj6iH1Q+8Lr8QPqBJ8C8TMqly5dQgPfULJxkBDU+6LEvrjF7vuJftlGlrPZLL7Af/78+UGFy45aLjYCStjFRljlKwKKgCKgCCgCI4CAEvYIgKgiFAFFQBFQBBSBYiOghF1shFW+IqAIKAKKgCIwAggoYY8AiCpCEVAEFAFFQBEoNgJK2MVGWOUrAoqAIqAIKAIjgIAS9giAqCIUAUVAEVAEFIFiI6CEXWyEVb4ioAgoAoqAIjACCChhjwCIKkIRUAQUAUVAESg2AkrYxUZY5SsCioAioAgoAiOAgBL2CICoIhQBRUARUAQUgWIjoIRdbIRVviKgCCgCioAiMAIIKGGPAIgqQhFQBBQBRUARKDYCStjFRljlKwKKgCKgCCgCI4CAEvYIgKgiFAFFQBFQBBSBYiOghF1shFW+IqAIKAKKgCIwAgiUMGHbtp1IJIaNQTqdjkajtm0PW8LIdrRtOxqNptPpwsXCNcdxChdVohKAgGVZRoYkEol4PD4Of+KXhuUZO8dxRipbCg9xJpOJx+Ph+VZUgxOJhGVZYwhIKpWqq6sbkcEbEg7gbKR0SPvRP5VPJoyIVYlEIhKJpFKpIGkcUEaDVCoViUTCMRzPOE8QwnYcZ3xOxEa6jM5hnpP+6Bgz+lpCxlvQMB59Iw2NNCwodsZUWFT+M2wb9NCwzbd98QwunmRfR3wri0fYiUSCi4qQxPa1yqhMp9N1dXUhJGe0l4fSDFlvlPPJhELMgLpMJtPY2BiNRkN4lwPKsDCfwwJxzkfFjFbn6QAAFLJJREFUsNsoYQ8buvHbMWjSH78Wj6hlIbNGIcN4RG00hdGwoNgZTo0HlqIPhm2sl4XiGTweLtaVsBHrfDKhcMJOpVKNjY07d+4MWaRxQMkkzLMML0KuBvKUU4xmE4Gwbdu2/vTHq1EJluM4OB+NRpPJJMLMqLDALgw2TqGvTA5swVmW5bstI3sFbdPRJMuysJdoKG1tbY3H41TB9jQDMzuyFhYyw4xJn32pi56iANWtra2UAwnUzva+orDLhL7EX4IAmw2r0AC+YzZPJpOWZUGCtzttkAUZeoii5V7jXdeFpzt37vTG1HVdrzRjIrZtm/gbp1zXNWqCGvtiyOgbKMFZiTBSDog5jhOJRLy7wXTEN/2A7aAJBtVMdeP+ggwQRHFLXFrLnAwibCMihsFeR4BPMpmMDvwxaZk5vnF0XdfIMSmHANJZprHrutJTRh/hBviRSCSZTAZticuElN4ZjkvJQF7qBfiowWKXQ5XDwRd2npVZRzN8u0iDbdv2mkGZXjuNTJBKkQmyhmYwyr7TqaEOoziRSMBOZp3RLAheoxft8aUGGsY0NrSM/uEYEPbJkydjsZhlWbFY7OTJk8P2Wd7DDrnKxqSGjSBEC2OD2c95HLc2Ue84jmyAWQDDWE49XV1d3i2mVCqVTCbhlz3wZ/iYSqWi0Sg6plKprq4uaQP0MneRNLAZ9iN7UGYz7ynJhbi7JvVKkzBPyRFFsZJ1pONSVDKZlPBCr3Q8mUxmMhlYyAFGnDGZygk3CHlpM9pwYsXUA+FSsuwCkKlIIhYkTYpCmbOMN+XyaRyEYThhkzaIHiYamcmEQiJPsRIH2ClDbFmWN8GQ86h3XTcELuQPbJOJISMuHZfGoC+NTyQSRNjXEchkGyQPjYRrlCazAojxlJSDXkwM2QunOF/TJG8baRIdlM2QfmwmHZfYsi8Kci+adgJqObMFwS6lGUtb3y7S30wmw3lMmiFloj1RlZnguu6mTZu8M49hRjqd3rRpE51lKKUWWZbdGQ7ZgKIYUAkvyl4AUQ/tITh7FY1yzRgQNtgaV4ixWGzYDudD2DL/oIjzrDwlc5frJLZEx6D6cPsNIYYo2ZdzqzQMF/Jkd4x5DA+ZhZBDCUxKiOIsjymYsw+1syOJgW3odZ6iGBQWqIVWoUYKlFOPMQV7V67o7iUAzilSMrWjIPlAXiSFSOOkkEql4vF4XV0d8PQ6KK/qfBt7DaMQhsBAifYbfb2IgQkYL3SUs5shSobYN8HkoEBfJrMBl7SNHqELg2J0oTFGROh+kCNoQOONbPFqkQbwGoXXHxwaRkd6QZdhMK1iVhj19AsFo5kEynCcdhoSZD26kx1xCChoMLrLXhRoJINvF2khO8ppR1YCfF6CcPYgqmwsxRpmsE3QSJcNBg03G3vhBRkzwSR66MVY45QvzpQ/VoUxIGxQNf8dtucy4Yi1IY3hYT1byoDJMnMdV4u0k1uskBm0vQxF3Gnh2oUGMK2NPUZO2dIYbxLTvBDXeIqmSi+YiDSJqmkbhxxnqHBR3DuiU0BADmZaBb1wE4qM6TIIeRosuZaVRmTpAht4e9FxFtiY0lhIiD889uLdXAlvHIIhDTBQoj0SLu9ESQBl4jHoBhR5Jhhl0gYmA63FKdqGAvWigInSKw19g0QFOeLFh5h74yvnd8MAQ44UIq9rfVPx8OHD8XhcXjQQGWLFoSSbGZLlatLAgXI43r0CGccQ2CkH1yh86CykC1yWtoUQtmE2xDLfDC2o9xI29iGQLXLGkMazLKd9I4hs480E2skuLLAXcwBmM3DGIduPSWEMCHs0V9je5AiKCgaGTDjG2DcwSDLf9JIvaFGdVwgyhlf9VGfkhzEXcACHJBxPseDVLmuompMChxy1B4lCPS8C5HDCsDE2pihZQm1MptIeaacsEwdWEmopmWdRMCTzMEQaUqirq6uxsTGVSkFLV1eX7z3L8MZBGMr5JaiN4ZSBGA9ZMByXh3kmGPFkXyaDARdtMySzo/cKg6cYAtRQVJAjXnyknYZhI0LYBnV5x4j3qtrwhc4WlbBJMFKdLMvJMCRS6IJLcMr0AotmRj3DB0z4ApWsl2Yg8zkHMsGk2bKM6BtXhJx8ZEsjr3jI/DHMkKligGMcSi2jXx4Dwh7Ne9herBk54xRyxXGcxsZG3MwOmjJkkAx+Mi5jZRLIXrLMjA83jK94sj0zj9K8pwwf2dIoULV3MuIQChIlp0s5H1EFBwYkkLCl/QbUxiFFyYK3Dd03FMle0lNJk4NKSyaTeAAnnU7H43EcSskoQ3VQ4yAMpSUSFinfcMowmIfefWwpRBrJuZghxlnC6BXFWFMdukibjU1gaje6sN6ICLPFq92rCzW0yveygO4YBkibveOU49roRbMNN/NsJoNoOG4cUhHt59hk4GQ6GfawuywQW1QO2kUmhjRDyjQcl6jKLtJxaYasD7ruMdQZ109BeWLgyUNaKNGDCrYxThmH0p7RL48BYY+UkxxUvgOVWuTNDCyLEXJvGGzbnjdvnjEeePnGhyP4jJVXgjGokBxGhsFa8haHTVC6yGGDmR0mQTiX+DJxmZTQxQtY13VpPPGRbEH7aZ7Ubtw3hSipF9uYiUQik8m0trbiukc2kI+w4RIeioxhD2C9yEubjTZSC07RBdmLIKOShyHSgA/3CXBREnJDBDuKNF66GRIOWiJjJy2HhUxOAzF5KEGWgaA0Q5QMsUwweMrslYahTGOkj8ZjVs7AX8gIBVxSlISO2umINAMeScIOiaOEyPceNnXJ605DICcBORZgEkciceYqk6nIKBuDzntIIbKLETh5GAQ75Xhd9u1CB42QSTO8MmX4ODQk4IgycDAiyAkQ7gBGow00Sn9pg2+lF0/aLyXnSQ1BKmjDaBYmCGEDUzmrShCRLridzLf3vGGQgxDdEV3swJD25N01ZqpUh5GAd0X4FplvA2kwU8owLGg+hW14vwUWcl6QSYnc5SYSZ0NpD1WHE3aQKMKLsQdMMInzxr8XTwQCNsux7W1J5KXNNNWIDuuJhuwlPTVGNVPIsNl32vWdnaHImAflpIAGhMs3+kbspPFIPKBhICYPpSOcPaWcPBMMXRhHQ5QcGjKUmOiZb2RBaaE0BhHh21lszzhSlO90D3VBvWTmGAYYOEvWl4RNnvOmGeeBaDTqOI7vLRImD7pLO0NSUeJDnHEdLO+dG3GkPb7PzUAmco/Z6+0ik4fNJAjeGY8WWpZlZAKTx3jdS5rB7vLtOCM6MF5elEuIDCTpqS/ahmQOxng8HkQNBs5S9eiXS5iwhweW3KgZnoRx0svIvHFilZqhCAwJAd/ZdkgStLEiMCIIlAQ13FqEbax+RiTMYyVECXuskFe9I4iAEvYIgqmiho1AqVDDxCdsbnp4tzqHHd3x0FEJezxEQW0oEAEl7AIB1O7DRqAUqWHiE/aww6kdFQFFQBFQBBSB8YOAEvb4iYVaoggoAoqAIqAIBCKghB0IjZ5QBBQBRUARUATGDwJK2OMnFmqJIqAIKAKKgCIQiIASdiA0ekIRUAQUAUVAERg/CChhj59YqCWKgCKgCCgCikAgAkrYgdDoCUVAEVAEFAFFYPwgoIQ9fmKhligCioAioAgoAoEIKGEHQqMnFAFFQBFQBBSB8YOAEvb4iYVaoggoAoqAIqAIBCKghB0IjZ5QBBQBRUARUATGDwJK2OMnFmqJIqAIKAKKgCIQiIASdiA0t+AJ4zeDxxYB/Ayt789a07Agg23blr/my/YjWwj6/ZXR0T6yvoytNPlz1ENCDz+y5P2R5rF1p0ja4azxw+SGrpH98eZbCl4DyfF5qIQ9PuMySlYZpBjEf6NkzZ+rMWz785P/ezS2BgcRtq+pJVeZTqfr6upSqdQoWC4Je1B1JfG7xSFeDA/YkGST6BVI2PkMuhDX9FSxEVDCLjbC41q+MT7Hlv8MpAzbjLM4HFuDQ+ZQX2tLq3J4vDI8HyXlDCrhliXsoOsniZ4S9qD5U9INSpiwuUFkWZZt2zIMtm1bA3+RSARLBMytqIzH45lMxvghXiY9Mr61tTUej6M7atDX2GiVig4fPhyPx+XuHGXSNimcv8/tOI40jI0pXCr19Rq85ThOJBKxLEu2p3DvTpoUBU9D5Liu62sPrR2Sa/KXaCViEmqEgFvi0lp2CSJsGVzqYjLQZtd10bK1tRUhSCQSTBXZXhpGeCVho2ykFnqBYCif2hmaaDSaTCbRl2dZkI4zz2UlO8oQwHjpHUCTHQmj67qy3rZt2makE6wy8kQClb9GqjDcl7GTWReJRDDEgKRlWQwWk4SxM8yGzJ07d3oH2oikBxJJGsbwGfLptWEh2/u6IKPD9EMXjkpMg+EpJ+XI6EOUPIuYygyXkZXgcx6jC9JHhoZntVAIAiVM2MlkUpIxM8O2bU5hXV1dqVQKaccETSaTgxK2nINSqVQymQTK9sAfy4Yi71xDpeiC4UThGGwQYhgpFUmxvl5jhEAOVGBmT6VS0WgUKKVSqa6uLiNX0JjQBcnBvEm2kPZQ4JBcI27Sa2k5Z0DYJh2Rk8ighC0bIBloMAqYTxEmuC+jQzt9c4CWeC2XseClEq6oEA5ZhhzqkhZiDgUImUwGeSj7IjRSHe0nhkzCIBh9tYSssGWeuK4r7ZF44jqAGUi4jC6G+zK7vGMZTjEVpUzpBZqR22AVeqELI842haSHtHNQ+SHAhrgQ0ksuDJCKvikXFH2Zb+jOCUHCKyMbMlLkiJMapRYtDxuBEiZs6bNt2xiBqVSqrq4unU7Ls757aHJewAwiZz3OCFKObOarSCao7wDDePCdQOVMZAj3FYWZmvOOnKY5gA05hi+u6xrjU868cBbTmSHH1578XeMMDntorRzqhm2ML7owoEYXOsjgUjhPGQW2pEYZHW8uyRzgdCbna8TRN5eItoGVlGmYZzjuNdJ1XYYDYmXqJhIJEpJMGGghjF4tUqxhEqyV+SbdyUejbA/hMkyMiJF1NINmw8hoNAqCkde4BAqnDKsoQeqlfFmgMRToTQ856g13fOUzXlIRyiEuhPSSWowckFAbUSYI0gy0DyJsXlNKsbgsw0gxuntTTurS8jAQKG3ClttBGEhygAEOI7eIkdGSSe/bHleUcj/N6O7VFTIevGMe3dlFquM+G0eR12uDt3gIX7B/RcdlwRhg7Ig2PAy3B40N3IzZVromKUSOdgNS2kYvCIVlWZg7aKF0SlImOJULDqOZbMkZmTgbLkgQoB3C582bx4kM8umIgQkPyfS0h+nHGoONWO/tawDF7PL1TmIIGE+fPk3Oo5ZBCdsIIsmAvhPPfDRK9ymBBWkVnOJFCdEgtrKxr1USFnQvMD28ScjM8ZUfRL3hLgT1Mq72DCE8RMEbi0wmIxFDMw4BwitBY2TZzOuvVMRgSUVaHh4CpUrYyCSmghyZrAQiRgoSJmM64JTBFGdL+Z4Jm5GB2AwFx3Fs2w5SaghnoqMvZXqnADQI8tpobxyil1wS0WbDTqMjD1lgR28hf9cMeiMIdB/CaZshWaoOMswILnYaDY6BHNmSGnGKhmGhwO7MAQA7b948A1vKNCznoXfypUzpnbeZL4/SbMqnEFrCSVbSOZr5avFVRLFe2OUAZHy99gRplO7TZiMfqF3WIwSO4xAENpPLO8rEWeOwwPSQxkO+zBw+H8D8CQI83IWgXkMibG/0JVxMEjIx4R0SYbO7IVwPC0egVAnbGCScL7xTCcatweJGlst0NGYZY5xQr68iTHPxeNxxHN/dVEO4MbA5E3k32RBpasdhkNe+tlG4TBpjjjA68jDIHq8ozgj5u0anqA5i5WRhbBVSr9GF9caMjHrCxWYy7t7Zii4E5QAt9O64grSMcPOQBVria7C3GY0kzpJZve0Nsb4wenvBKsNrmoqxQ/qhSZimh6dR9mI5JLgczgyBd5jDL1+rqEI6Nez08I4OprSv/BBgjQBJF0J6SXVGNOWhIVzaxrLUiNTi7osEzWjGkSLVUaYWRhCBUiVsOUiwV4kpDBnD8YwHSYxHOfDQmaxEOXySZfrKZoYiBMa27Xnz5vGUjJaR0Ex0tJGcKm+LZjKZ1tbWTCYT5LUxtfHQGfijVV6TDHvYEV3koa89hbsmJ1yUSUXY+ceEK4MFwkC9tFAaw8mFz+gZnrIxWxrEI/fqZV8YiRyQxsttGMqUHSkfDkqOl+lHw1CQTxWEPHSGyBrqjMsRLvUAnQEjNwmoRXrntUq+mkF/89cY4j6lwR0mLR8KYwMOSd8kkc1kWRo5IumBawWkhGGSr/wQYI08l2aHEzavn4wckIeGcDk5ML6yveGLNAbNmEhyHpMZ67ouEAhxmaq1kA8CpUrYGHW4U4IrR070SCacYh4jWdke0OC5R9zJ27lzp2RiSuM0h9cw5Os3voowD3L6M2JgjAeZ6PCI05MULu+x0WbptcFbPPT12jAJlzswmB3RRh4G2UNp+buGCY53uTjsOUHgFCLCs95byN6lHo3h5CJ7yZh6W5JQqVFGh0jKd5DkNAT3ASO1G5gYhwxlPB5n+tEwFnxdkJVGzkg3aYmvNHIMkGREKAEWchBJIcABXWSD/DUGuS8lADFDC2CXbxbJkOHNRj7lAJulTEnYEkZ6TTdly/D0CErpIPlBwMrZxnAhhLCJEm/G0Rcj5aQ9MvrSZbRBJssMlxhCrIRdbigysryqk3KkLi0PFYESJuyhujpq7eUO1agpVUUljYDcXBn/jsgrufFvrVqoCEwYBJSwRziUuJbkFe4IS1dxExEBY69y/LuohD3+Y6QWTkgElLBHLKzcmFK2HjFMJ64guW0YdANl3HqvhD1uQ6OGTWwElLAndnzVO0VAEVAEFIEJgoAS9gQJpLqhCCgCioAiMLERUMKe2PFV7xQBRUARUAQmCAJK2BMkkOqGIqAIKAKKwMRGQAl7YsdXvVMEFAFFQBGYIAgoYU+QQKobioAioAgoAhMbASXsiR1f9U4RUAQUAUVggiCghD1BAqluKAKKgCKgCExsBJSwJ3Z81TtFQBFQBBSBCYKAEvYECaS6oQgoAoqAIjCxEVDCntjxVe8UAUVAEVAEJggC/x/SlMVgH87XnQAAAABJRU5ErkJggg==)


### 3. **Classification Report**
```python
print(classification_report(Ytest, y_pred))
```

- **`classification_report(Ytest, y_pred)`**: This function generates a **detailed classification report**, which includes various evaluation metrics like:
  - **Precision**: The proportion of positive predictions that were actually correct.
  - **Recall**: The proportion of actual positives that were correctly identified by the model.
  - **F1-Score**: The harmonic mean of precision and recall, which balances the two metrics.
  - **Support**: The number of true instances for each class.

  ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApUAAADDCAIAAABce30zAAAgAElEQVR4Ae2d/ZMcxXnH8w8MVxfqtiRBLCPLUu3FiWwpEOswNrJSUBtCEoIdnyqKbVVgbJCrYlnGDsGKhV4oClKDDU4AcYK4BJYLlQtjmxGWEOCjdAU65BUSIsj3YnS+rY1KxC/lTXhLMUndE3/90DO719O3sze7+90fqJ6Z7qef/ny7++numUO/E/FHAiRAAiRAAiTQbgR+p90cpr8kQAIkQAIkQAIR4zc7AQmQAAmQAAm0HwHG7/bTjB6TAAmQAAmQAOM3+wAJkAAJkAAJtB8Bxu/204wekwAJkAAJkADjN/sACZAACZAACbQfAcbv9tOMHpMACZAACZCAVfz2fd9TP9/35xFctVotFothGDr4EARBqVSq1WoOZRsX8X0floMg8DyvWCxWq9XGpXL7tFarlUqlIAjSelitVgcGBsrlcjDzS1u8Qf4gCAqFQrlcjufxfd+BtrQxnPll0aV9348DzLrSOJz4nXK5XCgU4r7Fcxp3wjAslUrVanVwcDAuRAOBDDvze+nWW+bXZ6N2t+HZgoaHYRgfiXpuNBpS79KtgfWsdfB92/jtMNozopbD+C3zmiwLEntwRiiyM+s8frKL37qxqEXfTJvOOpTmKn4HQTD3NUrj+J2Wf8vy6+HZmkqzrtF5eFo2H0PDMj+yJc5+ltZ0F826gXB47ommzEXObjB+O6P7bUHf93EekPXQ/W2tWaacxw96c9P337q5qEXfTJvGtBKG4dxjW7x2xu84k3m50/ohmXWNzsPTkj+GhmV+ZEuM31EU2QBh/AZG+8Sc4rccxMnJOk6PRftt27aVSqXEA09dCtt6KSWmjBMYOY6WR2EYyv774YcfLpVKuJnYYMmp3dPn59oNPX2jOjgfhqEY8TwPcRo1VqtVOVSMoghlPc+DTf32AcWlow8NDemcsKlpAKw+hhLnxRrsw2HxpFQqbdu2TTwPggA0kE1qkWEj2eAzHolLmhUkg7eNE9LSMAwLhYLneahdSsEr442Dvi9eYWrQckhXgaz+zA/+oIjMIA1ERBGd0K0GGaM5Rl+FFuKJDSugRlldSvug7wNCsVgcGhpCJ9H5xWfdlzzPQ08Iw7BcLheLRZyEw5MoirQdXa/mY6RBG3YSW6RLoRXimH4kae082ph4s54u8AEDDb1F+tjQ0FCxWET3wxCG4lEUzbFG3S5tCp1HbuqRqJmjCRg7yI8EqkCk1GxlokDDGw8HLT1qjPuQaAR9AE+lIY0X3JqJ9AS0C/VqINpDfR8QpKUy+0mGxCJGn0FjxY4uYvQ9RLdvfOMbmFWgJtxoQcI9fsuMjMGP6CLoDRZoiZ4yZPxI3yqXy0NDQ5JNz8JBEIBLuVweHh6WUrCvM6CWKIokG9QdGhqq1Wq6Bw8NDYnz2g3d/4aHh8szP8xx4oCuRbop1DUuBQUmAukQ0l4ZXXikbaLvGjS0n9jbVavV7du3S07dOpmGpPlSF4gZSmFRor3VPtSTTPvcIC21A5HuNhqIrDlEa117rVaTjqGlMaYDNFzniaIIlPR93ZwGbkdRVK+HeJ4nzdHEpDrdzHoxyahUjEAFgw/6nlZf55H7qDfRZ2GLzgZTmrPE7IGBgWq1qhEhs+F2/BKQG7TIKLV9+3b5RkTXqPPoqUCGsOEzMuhuJnnQXoQ0sYzeIk3TXQ4xXvfMudeoW5Q40TUgljjAtUtojtQifV7zxKyFnIlPtZNiX2YqmUvjk0yiEfQBDRALIBjUdSEd33+nGhSwg30UJn/tqu7Pus/I1I1JUg8xPbSFDLIJHPniRzvQsrRt/MYqQ5jqDiS+YkqVRxg8Rkswpcp9rRlyYrzBJh7FA7PWQ2dLtIwerHMaEz2mQslTLpdlUjOK4NI4fYXz0iGw+JD88MroH7BmrAD0rIpHw8PDiS5pV3VLDbGQLa4Uhp8uYiOZ9t9IGy01LOt+Io/CMERCm4Jv8TGDxurOgM4Tt2a0SNdSL40iRnMgt54jxAiK1LMp98U9cNDeGhak80gGzE3oFfGvMnVxdDyMIJlMgU7HeF1Q32/cEAjUoEX1LOhW6zyGJ/HGojPX0yVeBE3WvUWy6dGKqiGxOOZQo26RTsOyJTFUrfuA7njo88ipq0PDE5/qnPXk0NNRohHpA8eOHSsWi7qL6jlWV6TTuosmAhGD0EXK6lKwZmzq6hUx+gyoIgGDAGs4Fp+LUKQ1Cdv4bYhhdH29woo3Hi2RR3opgK2MjB88kiCKGQEWjNknfik56/mAHizZcDiDrZK0C+s+tAsZtCeSNjoQxqRMfMZqAE8TmwaD4CAJLPcAUK9kZakrOTEB6ZZKKRTBwItTij9Cjdolo1FxJvpOvKUynOK160Eu5we6Im0HwwnEkFNvyOQmNNVNQLzUrsbT8R6i3dCTvnFftyVuVt8xOOCyHvmJiQnj7y/QqcRs3GcjBuvBi9lfqsPiSbPSg1R7bqRBAE2QDMalLmW0EV0UeWSHhF4tDTF8k9GB2qWsvjT4YGhoDvEwj0kfJ+qoN22NaA58gynpogYi4zI+wHUGncZcJDeNWQsNT3yqnZQMWo56PhhVSESMB2+b4QDnMesi6KCN8BwAEzsnWgpTOj+K6E4izRfRjY4BIxgdcKyN47dxaADJwVp3CEk3eKT/tgHjDQltyoBrXOqKdP+T+9BVSmESx1iVbNJZ9awh+RFHtT8N9t+6R0oRtCjedWATTuKOTsjsjKbpP9pB9DWWDpBG7CBbXI74o3ge7YxNOt5SHb/REDFlCCGNldGi7TSI3yAMU4k9ZFbP6/UQ7YYRv7GGSGxLvRoNwrhEwihotF0Hnno+N4jfUou8C5dDnXr1Gm7EL0HGsGBcoqCMMhEXbuCpTkgEldFab3SgdimoL9El5BEsGB3DyIb+g/zaJS39rDXqgokTnYFIXyYOcJ1BPPF9P85QGohZy2iI8VQ7aZhK9EHyG0YQv42x0Nz4rcOndhtp3VKDFfLEFYST8SEGIHFr8cy6iqzTjvvvBs2IP9JtwPZI3zQQYCBhc2Bk1vsPYxAiZ2JF0BVVSH6MVRSHlvpOPB7r2VNyast6EpGnsBB/hIoaPyqVSvLtXm3mVyqVEAIRfZ3jN5zXIiaShLezJozmiGXx2bCsH8EsGqXtGB0GsmI5LA2Rd6u6LTA7awIoJCd6iHZDTwFGX7Wv1MipLw0+4onOIHfQ/Ho+N4jfeKT/XiCxXhtisuQ1PDQuYQdjQe9v8NRIQHGDP7IZ9/WlgQW4jKnDyFZPcYcaUQStkDuo0UCES0nEBzgyiB35hDYMw8TXauCMhsMfqK/vGHLU80EXQRWCfXx8vFQqYXdkGNQFdRpGkB9xWrfXpnMaLa1XRHcSVNp4k609EecNTXWLWpB2jN8yc2FlJ9FOBIu3UDdDFt3okfJ/z9BFZFBh+YaPreTVC75fgwVjEKIuoyLj+zU928oZnfQVfP4Dl8RDMZvYD/T350Y4FyPox0alen8Pt9GNUAofsKCl8C2OHTZ1D5b8IIaIKPchIuzDB2FikAQQPd60/0Za8KI52jHDMh6hyTpA6sGmXTUWK3K5evVq1Bjvq1DZcFVfGmLhnFC7od0ztNbfD2qz8bRW0548hBaGMl7q+WwgMujJBLR69Wp8i2roAsXjzus7INOgRYn5xT390kqy1Wq1bdu2yXt9NM3gjK6C2qWsvtRpjcLggGgqFhC/514jWq3JSO0inL5v9AHMOZIncYUk84Du81oyWMD4SnwKJw0H6k0yiUaAWlqHMaiDXL15A+7FHdB8bDqnNiWBo1AoYAKE5w2mJuPVOBhqT4SY0Ys0xhak3eO3TF54rwCp4i00miHUpCDitKgif8Wh/x5Geo9klkhj8DIudV2wic9Kta54rSXaSKzSvunoJQ6gjboW8VB3DjQKHVH7L2XR0Q1TcimN0qWEKhxAJ0bOQqEwNDSEBbhuqZSFh0b8lr+FkLqkyXAblxoLWocJLrEJuCktlb+Uw1/p4KnWCJbFYXEJgcogJvLJU93Y+HCVuiA3OgN8qJdAEaOHwCUdvwENJC35GOPFuEwkj4Anb/JwHqPva5/xmYisQuJDRq+ShUa9euux0iiMJhiX2gJe1UsnRBdFHmTAKlO3Rf85otE99KU4AN3RWwwO9eL33GtEc9A5jYnOQKQvEwe4ziDGjXijhxUmDTQ88al2EnO7nnKFNiaZRCMau2SQQa3Z1hsXaGkQBEYDjctZOydaikYlFhFv601NuggYGp6IfT0XocbWJKzid2tcad9adO9so1Yk9kVL/2u1WuL/RDNeXA/p+FPemTuBehuauVumhXYhkOcpSOYZWZzZzxstIN8BUxPjd3P6SXwH0xy7WVqZS/yuVqvyycysDnbAIJm1jfOYQXY58Z3rPLrEqltMQHauOC1rce2zVqfnRvt5Y1azc8/QAVMT4/fcu0G7WphL/LZvcwcMEvvGtiYnDvb1AXJrqmYtuSIgQxgfZ+TKt/w70wFTE+N3/rsZPSQBEiABEiABkwDjt0mE1yRAAiRAAiSQfwKM3/nXiB6SAAmQAAmQgEmA8dskwmsSIAESIAESyD8Bxu/8a0QPSYAESIAESMAkwPhtEuE1CZAACZAACeSfAON3/jWihyRAAiRAAiRgEmD8NonwmgRIgARIgATyT4DxO/8a0UMSIAESIAESMAkwfptEeE0CJEACJEAC+SfA+J1/jeghCZAACZAACZgEGL9NIrwmARIgARIggfwTcIzf8X9gtcVNbc2/vZG2UR3wP8RP22TmJwESIAESmBcC7RS/9b9xy/g9L92FlZIACZAACeSEAON3M4Xg/ruZNGmLBEiABEigPgGr+B2GofebXxiGURTJ+fnDDz8st0ulUq1WQy2+7/8muyf5Zbss6SiKwjAsFArlcjmKIjySRKLBKIq0Td/3sf/Gff3P15fL5UKhIKZwX/6Ve7np+75427hS1IKm4cWBrgLWdPz2Z366ILIlVhq/qSnBDhMkQAIkQAIkEEXR7PG7XC4Xi0WJteVyeXh4WOK353kSkCQuSpiUIIRAJUEOIR/3gyDwvP8P7dVqdWBgoFwu64A3NDSkFwQiVfz8HEZ0qNMOi29hGOpIXKvVhoaGsHRAgNcOoHMgYCO/NGdoaEiYoApZlxSLxWq1KgsOtFeIyaX2RGfTtUvzdUPgDxMkQAIkQAIkYBu/BwYGJCYBWRAECFTYjtdqNb0BlcxBEEjcwqNarTY488N92b77vo9Qiop0Ih6/ESB1UDTsiAOSAQcAYlYbjKKoXC7HW6qDaGIGicHiOdqoA3Ocgz6ugE3Dbd1wpkmABEiABEjAIDD7/lsin+d5OrjqXamO38Z92ZJKuMI+u1wuDw4ODg8PDw4O1mq1YOYnOT3P08sCw1cdbnXAxs44CAJ4iwN8z/PEAdn069gpd3ROnOqjal0R1iLyFEf3gGMTv+tVKi8pGjQfLjFBAiRAAiRAArPHb2Ekp8QIb0acxqUR4XT8xj5VAraE8+Hh4VKppLfFEt6wsdYK2cdvvdTQFsQHhFu4beQxLuM7eKEBJ7F1tozfeg0RrwsvJoxHvCQBEiABEiABELCN31IA4dmIfLjUAcwognfAmzdvljfHvu8PDQ2VSiXjcB47dXgpCZv4HT+4Nozoc/K4t/HMyB+GoRwYGIsSrEvkPjbQYCU28Xp71krrNT/RN94kARIgARLoTgKzx+9w5id0EIQQsOU+LuW0GRtT/fJYAmGhUECEkxNjyVyr1bZt2ybfrBmlIIyOfPpYW5+foxbs6cX/arW6fft2MQU7hrc6DyqVhO/7q1evxrZeeyitiL//1p/UGS0tlUpAJJUmNl/XYvjDSxIgARIggS4nMHv8TvxDKQRswacvJSjKS2Wct0s2I17qr9Nxsu15nlEKCsGy/vsxbRnxVeKl+CCH1ShrvGKXk/BEb1GvbKwNr/AaW9Y08fitWyQH7DpmF4tFo1K8TUdFjN9aAqZJgARIgAQ0gdnjt87NNAmQAAmQAAmQQB4IMH7nQQX6QAIkQAIkQALpCDB+p+PF3CRAAiRAAiSQBwKM33lQgT6QAAmQAAmQQDoCjN/peDE3CZAACZAACeSBAON3HlSgDyRAAiRAAiSQjgDjdzpezE0CJEACJEACeSDA+J0HFegDCZAACZAACaQjwPidjhdzkwAJkAAJkEAeCDB+50EF+kACJEACJEAC6QgwfqfjxdwkQAIkQAIkkAcCjN95UIE+kAAJkAAJkEA6Aozf6XgxNwmQAAmQAAnkgQDjdx5UoA8kQAIkQAIkkI4A43c6XsxNAiRAAiRAAnkgYBu/X3/99fHx8SNHjjzDHwmQAAmQAAmQQFMJHDlyZHx8/PXXX4+sf1bx++c///nIyMiJEydeeeWVCn+5IfDMM8/kxhc68g4ClOYdOOwuCM2OU+a5KETmiJMqeOWVV1588cWRkZGf//znlhF89vj9+uuvj4yMjI2NJdXIe/NJgMNsPuk3rJvSNMST/JDQkrm0/C6FaDny31Y4MTExMjJiuQufPX6Pj4+fOHHit+aZyg0BDrPcSGE6QmlMIhbXhGYBqRVZKEQrKNev48SJE+Pj4zZb8Nnj95EjR3hsXh/1fD7hMJtP+g3rpjQN8SQ/JLRkLi2/SyFajvwdFU5MTBw5cqQ58ZtavgNtni4oTZ7UeIcvlOYdOOwuCM2OU+a5KETmiBtW8LOf/eyZZ55h/G4Iqf0fcpjlVkNK4yANoTlAy6IIhciCaiqbjN+pcLVlZg6z3MpGaRykITQHaFkUoRBZUE1lk/E7Fa62zMxhllvZKI2DNITmAC2LIhQiC6qpbDJ+p8LVlpk5zHIrG6VxkIbQHKBlUYRCZEE1lU3G71S42jIzh1luZaM0DtIQmgO0LIpQiCyoprKZl/i9devWZcuWHTt2LJX3zGxDoPXDbP/+/V/5yldOnToVd6/Bo3jmjr/TemkE6YEDB/r6+h588MFKpbJ+/fq1a9e20f95ab6gdXxvTNtACpGWWNPzz0P8PnXq1KZNmxYtWuR5Xk9Pz7XXXjs5Ocn43XRpYdBmmI2Nja1du9ab+fX09Pz1X//1XNZSX//615cuXXr48GH4gESDR8jTPYm00niet3LlykceeWSOiLohfm/dulX6s/xXFivT09N79uz5oz/6o8TuPT4+/oUvfAFT0+233z5Hzp1d3Kb3ViqV73//+x/60IdEhYsuuuj48eMtxjI5Ofn5z3/+2muvbXG9Laiu1fF7dHR05cqVCxcu/MpXvnL//fd//etfv/7668fGxhi/sxPbZphJ/P7Yxz527NixBx54YNmyZVdeeeXExER2XtFypVJJK82hQ4cuueSSFStWvPDCC3MB2CXx+z3vec+hQ4eOzfwmJycPHz58+eWXe56XeNQ3PT29cePGvr6+O+64Y3R0dNeuXffdd99cIHd8WZve+/TTT7/rXe+64oorDh069NRTT910002tj98yua1fv77zFGlp/J6env7sZz+7ZMmSuPCI32NjY5/5zGd6e3s9z1uxYsUTTzxRqVROnTp1zTXX9PT0eJ73N3/zN5OTkw8//HB/f7/neYsWLXr00Uc7T5gmtihOO27c6OJBEJx//vnDw8Nbt25dsmTJ+vXrPc978MEHJyYmNm3a1DPz27hx4/j4eKVSOX369I4dO2TXcskll5w8eRJqnj59+h//8R/7+vo8z1uzZo1+VKlUjhw5cuWVV8oxDHb8Elr+4R/+YcWKFZ7nfeITn0g8h483oR3vOEjzwAMPLFiw4NChQ5VK5fHHHxdKK1aswKb88OHDa9asEar/8i//cvr06S1btog6ixcv3rt3b6VS6ZL4bcTpW2655dOf/vR1111n3JeeI0Ng3bp109PTRl8ykFYqlYMHDw4MDHie19vb+7nPfU4GglC95pprFi5cuH79+sTBYlhu60ub3vvggw/29fU9/vjjuqXGbIPpQgBu3bpV2K5Zs+bo0aPorvH79YTQs9bu3btxsiiTmPak3dMtjd8vvPBCf3//DTfcEKcGCU+ePPm1r33t6NGjTz31VH9//xVXXCGn64sXL37ssceef/75HTt2HD58ePny5ddff/2JEyfuv/9+zFxxs7yTapOHJeo999yD+O153j/90z9NTU3JBqW/v/+Jmd+yZctuvPHG6enpm2++ube399Zbbx0dHQ2C4IUXXoCae/bs6evru//++0+cOLFjxw796Pjx4ytXrlyzZs3w8PD+/fv7+/tlxy9jWO7ffvvt55xzzj333NOpOtrMgDLZydHIo48+etFFF1199dUTExNPP/30u9/97i996UsvvvjiDTfcsHTp0ueee250dPR973vfJZdccujQof379+/atWt8fPyuu+569tlnn3/++UsvvfQDH/jAiy++2J3xW3oROme8U+3YseOcc8757Gc/e/LkSTyNI5U95bp160ZHRx966KFFixZt3LhxenpaqP7Jn/zJSy+9lDhYYLMzEja99+jRoytWrOjv79+3b9/U1JQ0vHH8XrFixf6ZX39//+Dg4OnTpwVs/H49IeTVCWat559//kMf+pCMoMnJyc6AL61oafw+dOjQggULHnjggTjBxEGFL2tuvPHGRYsW7d27V3rAc889t3Tp0nXr1ulhFrfJO9B4VhR6RD333HMXX3xxqVSamJiQleyzzz5bqVRk+bV161axtn79+ssuu+z555//wAc+sGnTJr1rgZq7du3q7e294447Tp8+LaXwaO/evb/7u7/7ve99T+5jxy9jVTrJsWPHli1btmXLlln9b9MMNjOgSINXuTfddJO819i5c+fy5ct//OMfywZlwYIFe/fuDYJg8eLFIyMjiUAAXyB39vdr+v13X1/fgQMHhAkgyLd7AlZWrlNTU7t27Vq4cGFPT8+mTZuEcxzpzp07lyxZIoOiUqlcd911elUky83EwSLb9ERp2vGmTe+tVCpHjx79q7/6KzlPlY24nm0qlQoU0WO/Uqls2bJFTkrq3a8nhJ61KpWKUV07oq7nc0vj99GjR5cuXZo4HUPCiYmJ22+//cILLzzvvPM8z5MvY0+dOrVu3TrP83AAuHfv3sWLF3uet27dug4+X60nW6r7NsPMCBIrVqwYHh7WQwunWAgkos5jjz0WX5NBzdOnT2/evLmnp6evr++f//mfp6en8Wjr1q1Lly6V87FKpSLnbAdmfvg0WuI3TgVStbotMttLI+exGzZseN/73jc6OqpjDxR58MEHN2zYcOmll/7kJz9B86empu6///6LL75YxoueEDs+fuP99wsvvBBfQQKRkfi/f9Ppq1/9am9vr+yq40jXr1+vIaNL61WRpCENpjKjrra+tOm90sDp6eknn3xyYGDgXe9619NPP20E1ESAevLRYPX9ekLAoNRuVNfWzA3nWxq/Jycnr7jiCkxA2hUQv/HGG5ctWxaG4eTkJPbfkvPkyZMf//jH+/v75eOdqampb33rW319fTt37tSmmDYI2Awz6eJyxHTy5ElspqFLpVKRaHrbbbdp+3LTeCeiS/1fpBkfH9+8ebO8tcWj+P5b9jR6rDJ+G7sHeenw8Y9//PTp01u3bo1/yGbsPCqVyq5duxYsWLBnzx79laiGbIwyLW4+0zb9Wc/yRivQA437xuXf//3fy1QTRxrf9q1evfrf//3fNdXEwWJU0e6XlkKgmSMjI4sXL965c6fMNvjUAF8kCEAc0N5www0iQb379YQwJGb8nv3fD7XU8vHHH1+0aNGSJUtuvfXWxO/PN2zY0N/ff/DgQXknKvvvLVu2/PCHP3zppZd83+/v7//Od76zc+fOl1566Qc/+MGiRYsYvzFCEhM20tTr4noknD59esOGDcuWLfv2t7997NixoaGhH/zgB/+3rbn22mv7+vr+9V//dXR09NZbb9Uvue+88869e/f+5Cc/2b59uxG/4++/r732Wrzrkq0h47cRvyuVyt13393b27tnz54DBw4sWLDA9/3R0dGRkZHbbrttfHz84MGDixYt+ou/+Ivh4eFHHnlk165dO3fu7Ovr27dv3/Dw8Jo1a7pq/534nZru0nq8jI2NXX/99Xfffffo6Kj8CcZVV101OTkZR2q8dj3vvPN27NiBAyrpuomDRVfXAWmbiWXfvn3XXXfd448/PjIy8nd/93e9vb2PPPLI9PT0Jz/5yUWLFn3nO9956KGHFi9erLulfPsS/7Agfr+eEIbEsm9cvXr1s88+i3fwHcBfvm2KLH5Ni9+VSuXw4cNXXHGFfExeKBS+/OUvy2ZCJHziiSfw4fGf/dmfSfyW4yzP8/r7+x9++OFnn332j//4j+UL22uuuYbn5437os0ws4nflUrlpZdeWrdunWi3cuXKp556SrbXGzdulJsf+9jHXn75ZYyf++67D18+33nnnfr8XHoCvpTG1+zxTQzPz9euXQsIotTAwMCLL7740EMPvec975GvoGUcVSqVRx55REbQ4sWLv/nNbx49elQgX3LJJb7v64my48/PU8Vv+SvhQqEgPAcHB/E34gZSgbxy5UrP8/r6+nbs2CGH87rr1hssjYdqez21mVgOHDggoDzPW758+X333SdneyMjI6tWrfI870//9E8/97nP6W55ww03SK/Gu1EBG79fTwjMP+D5zW9+U/4KRv7+AvfbPdHS8/N2h9Wm/tsMszZtWru7TWkcFCQ0B2hZFGm6EMYCCD7Xu48MXZtg/O586Zs+zDofWataSGkcSBOaA7QsijRdiHpxut79LBrVXjYZv9tLLxdvmz7MXJxgmSQClCaJyiz3CG0WQK163HQh6sXpevdb1dD81sP4nV9tmuVZ04dZsxyjHUrj0AcIzQFaFkUoRBZUU9lk/E6Fqy0zc5jlVjZK4yANoTlAy6IIhciCaiqbjN+pcLVlZg6z3MpGaRykITQHaFkUoRBZUE1lk/E7Fa62zMxhllvZKI2DNITmAC2LIhQiC6qpbDJ+p8LVlpk5zAFhaLMAABg0SURBVHIrG6VxkIbQHKBlUYRCZEE1lU3G71S42jIzh1luZaM0DtIQmgO0LIpQiCyoprLJ+J0KV1tm5jDLrWyUxkEaQnOAlkURCpEF1VQ2mxm/n+GPBEiABEiABEigVQQii9/s///zU6dOWdhhlnkgQGnmAbpdlZTGjtM7chHaO3DM3wWFmD/2/1+zpQSM3/OulLsDlhq7V8CSrgQojQM5QnOAlkURCpEF1VQ2LSVg/E5FNV+ZLTXOl9Pd4Q2lcdCZ0BygZVGEQmRBNZVNSwkYv1NRzVdmS43z5XR3eENpHHQmNAdoWRShEFlQTWXTUgLG71RU85XZUuN8Od0d3lAaB50JzQFaFkUoRBZUU9m0lIDxOxXVfGW21DhfTneHN5TGQWdCc4CWRREKkQXVVDYtJWD8TkU1X5ktNc6X093hDaVx0JnQHKBlUYRCZEE1lU1LCRi/U1HNV2ZLjfPldHd4Q2kcdCY0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUgPE7FdV8ZbbUOF9Od4c3lMZBZ0JzgJZFEQqRBdVUNi0lyEX89md+RvOq1WqxWAzD0LjPSxCw1Bj5mWgZAUrjgJrQHKBlUYRCZEE1lU1LCZoWv4Mg8H7zKxaL1Wo1lbuSuVwuDwwMuJV1qK7di1hqLCshz/MKhUK5XDZaHYah6KZV02qWSqVarWaU4mVjAtlJ01jNxl7l/KkltCiKfN+XThsEgdGoWq1WKpX0UxD7zfzksUsb0IxLeyGEre/7sADaxmzTQDKUZQIELCVoZvzGWAqCwG2EMH5DP5uEjcYynckxRhiGOkhHUVStVgcGBiSoB0GAcRjM/Gx8YJ5EAnOXplwuF4tFQ5rGaiZ60kY3baBFUYSOKqHCOKLTT9G3AUEDxE0mDAL2QhSLxc2bN2Pe0Hj1bKNF4amqQTvx0lKCTOJ3tVotlUrVmV+xWJRlLzSOL5993w+CABtBz/OCIEBoCcNQrwZ835cRiz2itpzIolNv2mhcLpcHBwdlA61HlzDRT3UakDsVXdbtmrs0egmFAaU1iquZdaOytm8DrVarDQ4O4hgJgQG+6a6r05IhDMOunS6AaNaEjRAwoiVI7J+zSgZTTICApQQZxu+JiYlSqYRNuQRpvXyGr3ik99+I30jIflFWBjqoozgMdknCRmNjwjJYabZ6HOKwy9ivdwnYuTdz7tLo+I0ZsLGac3d7fi3YQMNSRlzV84DcQTfWfVseAeP8NjP/tdsIgVYAeBRFif1zVslgigkQsJQgk/gt36MZQwtLM+O+bMclzCfGb8mAE2Df942dh9FpgKDjEzYa69EliyesqIQP3lcl7kucX4V0PPzGDZy7NPr4MQxDeZs4q5qNvcr5UxtoeoqIogizim6anMwZ718luuiTPF2EaU3ARgjk131SpzHb2EgGa0wIAUsJmhm/8XmIRAIjrOpVmAwwjCVsCrXSevksprB8lviN6jyvSz9IsdHYUAGopZcYQSK+2wZzjqtUBOYujUx/0skHZ37lcrmxmqk8zGFmG2h6GomHZJkZZIWq09LY+HF6DiHkwSUbIeCnjtmJ/bOxZLDDhCZgKUEz47exsTP22fGVMoRHUKkXv6UHhGEor3IZVERpG421Csa5hT75iKIo/tT4wE13L6YbE5i7NNo+xk5jNXWRdkzbQNPL+vjLOD2BGNHdiCLtyKdlPtsIAWcwjRvAMZ80lgx2mNAELCXIMH7LwSyCOoI0vMRkhEd6+Bmqy5k8rPFcN4oiG40xivToAmdIIEeR8sFztVrdvn27yETO6K6pEnOXBtVpBXVaa4fMbZ2wgaZjtp4iZA7RfGR5irdCxtawrUFl7bylEOKGjt+av+6fyKMly7oVbW3fUoIM47fs3vD9uQ69ciqI01rEbxlyxvfnIgNeAUIVfGPleZ7xNyTI09kJS43L5XKhUPA8D8ARv/UhLd4X4o14176YmHu3mbs0WgXdveNqzt3bnFiwhCZBQuYQkMEcorkheBtHTTlpb27dsBRC/EdslsvE/pkoWW6bnwfHLCVoWvzOQ5u7zQdLjbsNSx7aS2kcVCA0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUgPE7FdV8ZbbUOF9Od4c3lMZBZ0JzgJZFEQqRBdVUNi0lYPxORTVfmS01zpfT3eENpXHQmdAcoGVRhEJkQTWVTUsJGL9TUc1XZkuN8+V0d3hDaRx0JjQHaFkUoRBZUE1l01ICxu9UVPOV2VLjfDndHd5QGgedCc0BWhZFKEQWVFPZtJSA8TsV1XxlttQ4X053hzeUxkFnQnOAlkURCpEF1VQ2LSVg/E5FNV+ZLTXOl9Pd4Q2lcdCZ0BygZVGEQmRBNZVNSwkYv1NRzVdmS43z5XR3eENpHHQmNAdoWRShEFlQTWXTUgLG71RU85XZUuN8Od0d3lAaB50JzQFaFkUoRBZUU9m0lMAqfp/ijwRIgARIgARIoFUEbOK9Vfy2McQ8rSdguUZrvWOskdI49AFCc4CWRREKkQXVVDYtJWD8TkU1X5ktNc6X093hDaVx0JnQHKBlUYRCZEE1lU1LCRi/U1HNV2ZLjfPldHd4Q2kcdCY0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUgPE7FdV8ZbbUOF9Od4c3lMZBZ0JzgJZFEQqRBdVUNi0lYPxORTVfmS01zpfT3eENpXHQmdAcoGVRhEJkQTWVTUsJGL9TUc1XZkuN8+V0d3hDaRx0JjQHaFkUoRBZUE1l01ICxu9UVPOV2VLjfDndHd5QGgedCc0BWhZFKEQWVFPZtJSA8TsV1XxlttQ4X053hzeUxkFnQnOAlkURCpEF1VQ2LSVg/E5FNV+ZLTXOl9Pd4Q2lcdCZ0BygZVGEQmRBNZVNSwnaL36Xy+VisVgul1Ph6MjMlhp3ZNtz3ihK4yAQoTlAy6IIhciCaiqblhI0LX4HQVAoFHRYLZfLg4ODtVotld+JmcMwLJVKTTGVaL9Nb1pqXK1Wi8Wi53mGQNLqMAy9mV+xWKxWq3KzcZE2xdVKt+cuTRAEogv+G4ZhFEUdLI0ltCiKfN8XLEEQaFkBB9Bk3iiXy4VCQW4mjgJthGl7IQS47/uABgkMzvUkQ0EmNAFLCZoZv1evXq2jLOO31iOLtI3GtVqtVCrJ1B+GoQ7SEgwGBgZk1RUEgYzDxkWyaEjn2Zy7NJoJ1q+dLY0NtCiK0FElVEjf1rgkrVk1cS6KV9R5d+yFKBaLmzdvRvzWzPVsYylZ55F0bpGlBM2M3yISVsR6zCQuynCzWCwODQ2hE2ClJqsBvREJw7BcLg8MDFSrVd/3UVe1Wi2VSrJ9RHE8dYaY84I2GmsV9OiSpumnSCMRRVG8SM6Z5MS9uUuDhtRqtcHBQVljdbY0NtA0DR3LgQuJMAwxpeg0MjBRj4CNECiL2BxFUWL/tJcMNpmwlKDJ8btarWI/By11DED01TclkMtgq1ar27dvR+SQGIz9h3QRid/6JsYnOpO236m9wUZjkBEIetFTb//duEinwmxuu+YuDfzRcui0HCN30iLVBppeqUdRpCcBEJPZA4seyYYT9Xr7dV28y9M2QgARplzhjDUT+qelZDDIRBRFlhI0OX7rEYX4rccY1mL6Zlx4kTCY+WmbOn7rbuH7fhiGevUga/NOmt3i3dpGYz26EpngFAQDb9YicU94xyDQFGmwikXI6WxpbKBhAyDAMckY/I3pBU/59StQNEjYCIHiuk/qNGYbS8lgkIn5jN9YdmFo4QspvQQ2dhL6Uh+YN9h/S0Vyoi4fyiEUoSLEpI7sEzbDTIOFNKCh31Eh3bgIyjLRgMDcpRHjGERy2dnS2EDTq3ZjZa/lkAW9voM0dgW4w4RBwEYIFNExO7F/WkoGg0zMc/yWffDQ0JCEVUNUkcdYIKMTIIHlmzFK9WpOLGNAGh2l4/uBzTDTnOPvFPRxOp42LtLxVJvSwLlLI26gb8tlZ0tjAy1+xhZfozeeB3Sfb4rWnWfERgi0Ws/Yif3TRjJYY0IIWErQ/PNzqV723PIBmmyLcQYoGfRNScs4xOiScNJ4/y0DVb/o8md+XdIJbDRGVNbLIKyB9HjD0WJikS5B2qxmzl2a+Etc4zhda9cst+fXjg00WdbLXKEDA+YN6edGUN++fbt83IpOPr8tzXntlkJIK3T8rjd1II+WLOcQ5tc9Swmyit9yVIs/J9N/fxm/qb8/xxl4oVAYHByU+C3dwvM8/f258PV9HwYxwXXJH3paagz4+OMxxG+ZDeO44kXmtze3Xe1NkUbLBAIdLI0lNMwGMiFgHsDHLjqW42m8kwMpEwYBSyGkFGKzXCb2z0TJjEp5qQlYStC0+K3rdkgb54QOFrqwiKXGXUhm3ptMaRwkIDQHaFkUoRBZUE1l01KCXMRvHmqlkhaZLTVGfiZaRoDSOKAmNAdoWRShEFlQTWXTUoJ5i984J5dzLePteKqmdm1mS427ls88NpzSOMAnNAdoWRShEFlQTWXTUoJ5i9+pGsPMiQQsNU4sy5uZEqA0DngJzQFaFkUoRBZUU9m0lIDxOxXVfGW21DhfTneHN5TGQWdCc4CWRREKkQXVVDYtJWD8TkU1X5ktNc6X093hDaVx0JnQHKBlUYRCZEE1lU1LCRi/U1HNV2ZLjfPldHd4Q2kcdCY0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUgPE7FdV8ZbbUOF9Od4c3lMZBZ0JzgJZFEQqRBdVUNi0lYPxORTVfmS01zpfT3eENpXHQmdAcoGVRhEJkQTWVTUsJrOL3Kf5IgARIgARIgARaRcAm3lvFbxtDzNN6ApZrtNY7xhopjUMfIDQHaFkUoRBZUE1l01ICxu9UVPOV2VLjfDndHd5QGgedCc0BWhZFKEQWVFPZtJSA8TsV1XxlttQ4X053hzeUxkFnQnOAlkURCpEF1VQ2LSVg/E5FNV+ZLTXOl9Pd4Q2lcdCZ0BygZVGEQmRBNZVNSwkYv1NRzVdmS43z5XR3eENpHHQmNAdoWRShEFlQTWXTUgLG71RU85XZUuN8Od0d3lAaB50JzQFaFkUoRBZUU9m0lIDxOxXVfGW21DhfTneHN5TGQWdCc4CWRREKkQXVVDYtJWD8TkU1X5ktNc6X093hDaVx0JnQHKBlUYRCZEE1lU1LCRi/U1HNV2ZLjfPldHd4Q2kcdCY0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUgPE7FdV8ZbbUOF9Od4c3lMZBZ0JzgJZFEQqRBdVUNi0laHL8Pnjw4DnnnHP11Ve/9tprqdxlZgcClho7WGaRORKgNA4A7aH5vu/N/IIgMCqq1WqlUinxqe/7xWKxWq0aRXhpELAUIggC4aypVqvVYrHoeV6hUCiXy7DcQDLkYQIELCVoZvx+6623NmzYcOmllxaLxZMnT8IVJjIiYKlxRrXTbAMClKYBnHqPLKEFQeD7fhRFEirCMNQG9dOBgQEJIeVyuVAobNu2bWBggPFb40pM2whRrVa3b98uxYMgKJVKtZlfqVQSRcIwRFzXohSLRUOyRB+6/KaNBFEUNTN+//SnP12xYsWBAwfWrFlzyy23QID/+I//+NSnPtXT0+N53s033xxF0ZtvvnnnnXeef/75nuetXbv2P//zP33flx4QRVEYhrJ2k/H5qU996g//8A9LpdIvf/nL2267TUpdcMEFBw4ckCrGxsYuv/xyz/N6enr+7d/+7eqrr4apSqXS399/7733wplOSlhq3ElNbpe2UBoHpWyg1Wq1wcFBbOwQGFCd7/sIDzodRVG5XGb8BqgGCRshdHGALZfLg4ODtVotiiI5CAnDcFbJtCmmhYClBM2M3/fee++aNWt+9atfIRFF0a9+9asrr7zyD/7gD4aHh19++eVdu3a9/fbbd9xxR29v79133z01NbV79+4zZ840iN/vf//7JyYmoij6r//6rz179kxOTv7sZz+77LLLLrzwwldffXVqaur973//2rVrjx8/Pjo6um/mt3DhQhnhjz322Hvf+96f/vSnHdktLDXuyLbnvFGUxkEgG2jVarVUKmEPHYYhFutSIyJ6tVrF/lseIcw4+NZVRWyE0EDCMJQTESTkqe/7QRDMKpk2xbQQsJSgafFbVluy0z158uSSJUuefPLJKIqefPLJQqHwox/9CMK8+uqrF1544ZYtW95++23cbBC/ZcuOnJIIgkAOZ3bv3n3BBReMj48jwyuvvLJ8+fLdu3e//fbbmzdv3rBhw1tvvYWnnZSw1LiTmtwubaE0DkrZQDNisN7woUZ5L2u8f+X+G3xmTdgIASPlcrlYLMp+CYsneRrM/GwkgzUmhIClBE2L388999y5554rnzPIfzdv3vz222/fddddy5cvn56ehjDHjx9fuHDho48+ijtRFDWI3/hE5X/+53+++93vrlmz5oILLvA8T+L3xo0bL7vssl//+tewJmF7/fr11Wr1oosu2rdvHx51WMJS4w5rdVs0h9I4yGQDrfFmTnYRMmPotDhjBBIHD7ukiI0QggJvvuWS++9m9RBLCZoWv7/4xS/++Z//+dTUVHXm97WvfW358uWvvPJKGIa9vb0jIyNomLzVNnbVvu8jDO/evVu//0b83rdv38KFCx977LFarYb9dxAES5cuNU7IDx48WCwWh4aGVq9efebMGVTdYQlLjTus1W3RHErjIJMNNONU3NjwGRHaOF03njp42CVFbISIoki215qJBo73340l08WZBgFLCZoTv8+cObNq1SoEWjmqWrhw4b59+6anp1etWnXxxReXy+Xjx4/fc889b7311uc///lCofCtb31ramrq7rvvPnPmTBAEPT093/72t0dHR1etWpUYv++6665CofDkk0++/PLLl19+uey/jx07dv7553/iE594+eWXh4eHZbf9i1/84iMf+cjv//7vyxkAoHRYwlLjDmt1WzSH0jjIZAkNMVsHBnnVipghtfszP3jC+A0UjRM2QiS+udD8dSxPlKyxD13+1EaCpn1/vm/fvnPPPVdvsn/xi198+MMf/su//Mv//u//Hhsb++hHPyp/EfiNb3xDvkT70pe+JF+k/+3f/u0vf/nLs2fPXnXVVZ7nffCDH7zjjjsS4/f09LR8Z7527dpNmzbhjxOeeeaZVatWeZ53wQUXfP/73xfhb7nllnPOOUfewXdqV7DUuFObn+d2URoHdSyhSZCQl3T6U3PZP8jxnjyVj6rgCeM3UDRO2AgRhqF+W+p5nmghf6qH95tSUaJkjX3o8qc2EjQtfueQ9S233CIfw+fQt2a5ZKlxs6qjHXsClMaeFXISGlDMb4JCzC//KIosJWjO+fm8t1Y78Nprr+3fv//3fu/3hoaG9P3OS1tq3HkNz3+LKI2DRoTmAC2LIhQiC6qpbFpK0IHx2/f9np6er371q2+++WYqZG2X2VLjtmtXBzhMaRxEJDQHaFkUoRBZUE1l01KCDozfqTC1dWZLjdu6jW3qPKVxEI7QHKBlUYRCZEE1lU1LCRi/U1HNV2ZLjfPldHd4Q2kcdCY0B2hZFKEQWVBNZdNSAsbvVFTzldlS43w53R3eUBoHnQnNAVoWRShEFlRT2bSUYPb4PTk5+cYbb6Sqm5lbQ8BS49Y4w1o0AUqjaVimCc0SVNbZKETWhBvbf+ONNyYnJxvnkaezx+8zZ86cPXvWxhbztJgAh1mLgdtXR2nsWSEnoQHF/CYoxPzyP3v2rOX/NnT2+P3mm2+OjY3JPwk3v61i7QYBDjMDSH4uKY2DFoTmAC2LIhQiC6qWNmu12tjYmOUfT80ev+Vfch0bGzt79iwP0i01aE02DrPWcHaohdIQmgOBnBRh750XId54442zZ8+m2i1bxe8oit58880zZ85MTk6e4o8ESIAESIAESKCpBCYnJ8+cOWO585YVhm38npf1CCslARIgARIgARJIJMD4nYiFN0mABEiABEgg1wQYv3MtD50jARIgARIggUQCjN+JWHiTBEiABEiABHJNgPE71/LQORIgARIgARJIJMD4nYiFN0mABEiABEgg1wQYv3MtD50jARIgARIggUQCjN+JWHiTBEiABEiABHJNgPE71/LQORIgARIgARJIJMD4nYiFN0mABEiABEgg1wT+F8pt05N6fgwpAAAAAElFTkSuQmCC)

- This report gives a more comprehensive evaluation of your model's performance than just accuracy, especially when dealing with imbalanced datasets.

### 4. **Plotting the Confusion Matrix**
```python
matrix = plot_confusion_matrix(naive_bayes_classifier, Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Naive Bayes Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10, 6)
plt.show()
```

- **`plot_confusion_matrix(naive_bayes_classifier, Xtest_tf, Ytest, cmap=plt.cm.Reds)`**:
  - This function plots a graphical representation of the confusion matrix.
  - **`naive_bayes_classifier`**: The trained Naive Bayes classifier.
  - **`Xtest_tf`**: The TF-IDF matrix of the test sentences.
  - **`Ytest`**: The actual labels for the test data.
  - **`cmap=plt.cm.Reds`**: This sets the color map for the plot to a range of red colors, where higher values are shown in darker red.
  
  The plot will visually represent how many times each class was predicted correctly or incorrectly. The matrix will show a grid where:
  - The rows represent the **true labels**.
  - The columns represent the **predicted labels**.
  - The colors of the cells in the matrix will indicate the number of correct and incorrect predictions.

- **`matrix.ax_.set_title('Confusion Matrix Plot for Naive Bayes Classifier', color='black')`**: This sets the title of the plot, with the title color as **black**.

- **`plt.xlabel('Predicted Label', color='black')`** and **`plt.ylabel('True Label', color='black')`**: These lines label the x-axis and y-axis of the plot. The x-axis corresponds to the **predicted labels**, and the y-axis corresponds to the **true labels**.

- **`plt.gcf().axes[0].tick_params(colors='black')`** and **`plt.gcf().axes[1].tick_params(colors='black')`**: These lines change the color of the tick marks (the numbers or labels on the axes) to **black**.

- **`plt.gcf().set_size_inches(10, 6)`**: This sets the size of the plot to **10 inches by 6 inches**.

- **`plt.show()`**: This displays the plot, showing the confusion matrix on the screen.


```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest,y_pred))
print(classification_report(Ytest,y_pred))
matrix = plot_confusion_matrix(naive_bayes_classifier , Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Naive Bayes Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[167  37]
     [ 13 191]]
    0.8774509803921569
                  precision    recall  f1-score   support
    
               0       0.93      0.82      0.87       204
               1       0.84      0.94      0.88       204
    
        accuracy                           0.88       408
       macro avg       0.88      0.88      0.88       408
    weighted avg       0.88      0.88      0.88       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_24_2.png)
    


Let's break down this line of code and explain it in a simple and understandable way:

### Code:
```python
fprNB, tprNB, thresholdsNB = metrics.roc_curve(Ytest, y_pred)
```

### 1. **`metrics.roc_curve(Ytest, y_pred)`**
- **`metrics.roc_curve`** is a function from the **`sklearn.metrics`** module, which is used to compute the **Receiver Operating Characteristic (ROC) curve**.
  
  - **ROC Curve** is a graphical representation of a model's performance at different classification thresholds. It helps to evaluate how well a binary classification model (like Naive Bayes) distinguishes between the two classes (e.g., "positive" vs. "negative").
  
  - The function **`roc_curve`** returns three values:
    - **FPR (False Positive Rate)**: This is the proportion of negative instances that were incorrectly classified as positive by the model.
    - **TPR (True Positive Rate)**: Also known as recall or sensitivity, this is the proportion of actual positive instances that were correctly identified by the model.
    - **Thresholds**: These are the classification thresholds used to compute the TPR and FPR. A threshold determines the point at which the predicted probability is classified as positive or negative. For example, if the predicted probability is greater than 0.5, the sample might be classified as "positive."

### 2. **Parameters:**

- **`Ytest`**: This is the actual **true labels** of the test data (the ground truth, such as "positive" or "negative").
  
- **`y_pred`**: This is the predicted **labels** for the test data, which are generated by the classifier (in this case, Naive Bayes). These labels are the model's best guesses at whether a sample is "positive" or "negative."
  
  - Note: The `roc_curve` function typically expects the **predicted probabilities** (i.e., how likely the model thinks a sample belongs to the positive class) rather than the final predicted labels. However, in this case, since `y_pred` contains the predicted labels (either 0 or 1), the function will still work, but it is better practice to use the predicted probabilities.

### 3. **Outputs:**

The `roc_curve` function returns **three values**:
```python
fprNB, tprNB, thresholdsNB
```

- **`fprNB`**: This is the **False Positive Rate** (FPR) for each threshold value.
  - FPR = (False Positives) / (False Positives + True Negatives).
  - It tells you how often the model incorrectly classifies a negative instance as positive.

- **`tprNB`**: This is the **True Positive Rate** (TPR) for each threshold value.
  - TPR = (True Positives) / (True Positives + False Negatives).
  - It tells you how often the model correctly classifies a positive instance as positive (i.e., how sensitive the model is).

- **`thresholdsNB`**: These are the different threshold values that were used to calculate the TPR and FPR.
  - The model calculates the TPR and FPR for different cutoff points (thresholds) to classify an instance as positive or negative.
  - For example, if the model predicts a probability of 0.8 for a sample, it might be classified as positive if the threshold is 0.5, or negative if the threshold is 0.9.

### 4. **What Does This Mean?**

The ROC curve is generated by plotting the **True Positive Rate (TPR)** on the y-axis against the **False Positive Rate (FPR)** on the x-axis. Here's what each component represents:

- **FPR (x-axis)**: False positives, which occur when the model incorrectly classifies a negative instance as positive.
- **TPR (y-axis)**: True positives, which occur when the model correctly classifies a positive instance as positive.
- **Thresholds**: Different decision thresholds for classifying a sample as positive or negative. The higher the threshold, the fewer predictions will be classified as positive, which will change both the FPR and TPR.


```python
fprNB, tprNB, thresholdsNB = metrics.roc_curve(Ytest,y_pred)
```

# Logistic Regression

Let's break down the code step by step in simple and understandable terms.

### 1. **Importing Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
```

- **`from sklearn.linear_model import LogisticRegression`**: This line imports the **Logistic Regression** classifier from the **`sklearn.linear_model`** module.

  - **Logistic Regression** is a widely used machine learning algorithm, particularly for **binary classification** (classifying into two categories, like "positive" vs. "negative" or "spam" vs. "not spam").
  - Despite its name, Logistic Regression is actually a **classification algorithm** (not a regression algorithm). It predicts the probability that a given input belongs to a particular class (usually, "0" or "1").
  - In simple terms, it finds a linear relationship between the input features (like words in a sentence) and the target labels (like "spam" or "not spam").

### 2. **Creating the Logistic Regression Classifier Object**
```python
LRClassification = LogisticRegression()
```

- **`LRClassification = LogisticRegression()`**: This line creates an instance (or object) of the **LogisticRegression** class and assigns it to the variable `LRClassification`.

  - At this point, the model is **initialized** but not yet trained. You still need to provide it with data so that it can learn how to classify based on the input features and target labels.

### 3. **Training the Logistic Regression Classifier**
```python
LRClassification.fit(Xtrain_tf, Ytrain)
```

- **`LRClassification.fit(Xtrain_tf, Ytrain)`**: This line trains (or **fits**) the Logistic Regression model on the **training data** (`Xtrain_tf`) and **training labels** (`Ytrain`).
  
  - **`Xtrain_tf`**: This is the **TF-IDF matrix** of the training sentences. Each sentence is represented as a vector of numerical features (words or n-grams with their corresponding TF-IDF scores).
  
  - **`Ytrain`**: These are the **actual labels** for the training sentences. For example, in a sentiment analysis task, these could be "positive" (1) or "negative" (0).
  
  - The **`fit`** method adjusts the model's internal parameters to find the best relationship between the features in `Xtrain_tf` and the labels in `Ytrain`. In other words, it trains the model to predict the labels based on the words (or n-grams) in each sentence.

### 4. **Making Predictions with the Trained Model**
```python
y_pred = LRClassification.predict(Xtest_tf)
```

- **`y_pred = LRClassification.predict(Xtest_tf)`**: Once the model is trained, we can use it to **predict** the labels for new, unseen data (the test data).

  - **`Xtest_tf`**: This is the **TF-IDF matrix** of the test sentences, similar to `Xtrain_tf` but for the test data.
  
  - The **`predict`** method uses the trained model to classify each test sample (sentence) by predicting the most likely label for each sentence based on the patterns it learned during training.

  - **`y_pred`**: This is the list of predicted labels for each test sentence. These are the model's best guesses about whether each test sentence is "positive" or "negative" (or whatever the labels are for your task).


```python
from sklearn.linear_model import LogisticRegression
LRClassification=LogisticRegression()
LRClassification.fit(Xtrain_tf, Ytrain)
y_pred=LRClassification.predict(Xtest_tf)

```


```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest, y_pred))
print(classification_report(Ytest, y_pred))
matrix = plot_confusion_matrix(LRClassification , Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Logistic Regression Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[165  39]
     [ 13 191]]
    0.8725490196078431
                  precision    recall  f1-score   support
    
               0       0.93      0.81      0.86       204
               1       0.83      0.94      0.88       204
    
        accuracy                           0.87       408
       macro avg       0.88      0.87      0.87       408
    weighted avg       0.88      0.87      0.87       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_30_2.png)
    



```python
fprLR, tprLR, thresholdsLR = metrics.roc_curve(Ytest,y_pred)
```

# Decision Tree

Let's break down the code step by step in simple terms to understand what each part does:

### 1. **Importing the Decision Tree Classifier**
```python
from sklearn.tree import DecisionTreeClassifier
```

- **`from sklearn.tree import DecisionTreeClassifier`**: This line imports the **DecisionTreeClassifier** class from the **`sklearn.tree`** module.
  
  - **Decision Tree** is a **supervised machine learning algorithm** used for both **classification** and **regression tasks**.
  - A **decision tree** splits the data at each step (node) based on the most informative feature, creating branches that lead to decisions or predictions.
  - In classification, the tree continues to split the data until it reaches a decision (leaf), which is the predicted class label.

### 2. **Creating the Decision Tree Classifier Object**
```python
DTClassification = DecisionTreeClassifier(criterion='entropy', random_state=0)
```

- **`DTClassification = DecisionTreeClassifier(criterion='entropy', random_state=0)`**: This line creates an instance (or object) of the **DecisionTreeClassifier** and stores it in the variable `DTClassification`.

  - **`criterion='entropy'`**: The **criterion** parameter specifies the function used to measure the quality of a split at each node in the tree. In this case, `'entropy'` is chosen, which refers to **information gain**. Information gain measures how much uncertainty (or "entropy") is reduced after splitting the data on a particular feature. A split that reduces entropy the most is considered the best choice.
    - Alternatively, you could use **'gini'** (another measure of impurity), but **'entropy'** is chosen here.
  
  - **`random_state=0`**: This is used to set the **seed for random number generation**, ensuring that the results are reproducible. By setting the `random_state`, you can ensure that every time you run the code, the model splits the data the same way, making the process deterministic.

### 3. **Training the Decision Tree Classifier**
```python
DTClassification.fit(Xtrain_tf, Ytrain)
```

- **`DTClassification.fit(Xtrain_tf, Ytrain)`**: This line trains the **Decision Tree** model using the **training data** (`Xtrain_tf`) and **training labels** (`Ytrain`).
  
  - **`Xtrain_tf`**: This is the **TF-IDF matrix** of the training sentences. Each sentence is represented as a vector of features, where each feature corresponds to a word or n-gram's TF-IDF value.
  
  - **`Ytrain`**: These are the **true labels** for the training data. For example, these could be "positive" or "negative" labels for a sentiment analysis task.
  
  - The **`fit`** method is used to **train** the Decision Tree model. The model looks at the features (words/n-grams) in `Xtrain_tf` and the corresponding labels in `Ytrain` to learn how to split the data in a way that maximizes the correct classification of new, unseen data.

### 4. **Making Predictions with the Trained Model**
```python
y_pred = DTClassification.predict(Xtest_tf)
```

- **`y_pred = DTClassification.predict(Xtest_tf)`**: After the model is trained, we can use it to **predict** the labels for the **test data** (`Xtest_tf`).
  
  - **`Xtest_tf`**: This is the **TF-IDF matrix** for the test sentences. Just like `Xtrain_tf`, it contains the TF-IDF values for the words in each sentence in the test set.
  
  - **`predict`**: This method uses the trained **Decision Tree** model to predict the labels (such as "positive" or "negative") for the test sentences. The model will make a prediction for each test sample based on the learned splits and decisions it made during training.
  
  - **`y_pred`**: This variable will store the predicted labels for each of the test sentences. These are the model's best guesses for the labels based on the patterns it learned from the training data.


```python
from sklearn.tree import DecisionTreeClassifier
DTClassification=DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTClassification.fit(Xtrain_tf, Ytrain)
y_pred=DTClassification.predict(Xtest_tf)
```


```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest,y_pred))
print(classification_report(Ytest,y_pred))
matrix = plot_confusion_matrix(DTClassification , Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Decision Tree Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[187  17]
     [ 47 157]]
    0.8431372549019608
                  precision    recall  f1-score   support
    
               0       0.80      0.92      0.85       204
               1       0.90      0.77      0.83       204
    
        accuracy                           0.84       408
       macro avg       0.85      0.84      0.84       408
    weighted avg       0.85      0.84      0.84       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_35_2.png)
    



```python
fprDT, tprDT, thresholdsDT = metrics.roc_curve(Ytest,y_pred)
```

# SVM

Let's break down this code step by step to understand what each part does in simple terms.

### 1. **Importing the Support Vector Machine (SVM) Classifier**
```python
from sklearn import svm
```

- **`from sklearn import svm`**: This line imports the **Support Vector Machine (SVM)** model from the **`sklearn`** (Scikit-learn) library.

  - **SVM** is a powerful machine learning algorithm used for **classification** and **regression** tasks.
  - The goal of SVM is to find a hyperplane (a line or a plane) that best separates the data into different classes. In classification, it tries to find the boundary that maximizes the margin between the two classes.
  - SVM can work with different types of **kernels**. A **kernel** is a function used to transform the data into a higher-dimensional space, allowing SVM to handle non-linear classification problems.
  - In this code, you're using a **linear kernel**, which is suited for data that can be separated by a straight line (or hyperplane).

### 2. **Creating the SVM Classifier Object**
```python
clf = svm.SVC(kernel='linear')  # Linear Kernel
```

- **`clf = svm.SVC(kernel='linear')`**: This line creates an instance of the **Support Vector Classifier (SVC)** class and stores it in the variable `clf`.

  - **`kernel='linear'`**: This specifies that you're using a **linear kernel**, meaning the model will try to find a straight line (or hyperplane) to separate the data into two classes.
    - For example, in a two-class classification problem, it would try to find a line that separates the positive and negative instances in the best possible way.
    - If the data cannot be separated by a straight line, you could choose other types of kernels (e.g., **'rbf'** for a non-linear transformation), but in this case, we're using a **linear kernel**.

  - **`clf`**: This is just a variable name, and it's short for "classifier". It will hold the model after it's trained.

### 3. **Training the Model**
```python
clf.fit(Xtrain_tf, Ytrain)
```

- **`clf.fit(Xtrain_tf, Ytrain)`**: This line trains the **SVM classifier** using the **training data** (`Xtrain_tf`) and the **training labels** (`Ytrain`).

  - **`Xtrain_tf`**: This is the **TF-IDF matrix** of the training data, where each sentence is represented as a vector of features (words or n-grams) with corresponding **TF-IDF** values.
  
  - **`Ytrain`**: These are the **true labels** for the training data. For example, if this is a sentiment analysis task, these could be labels like "positive" (1) or "negative" (0).

  - The **`fit`** method is used to train the SVM model. During this step, the model learns to classify the data by finding the **best hyperplane** that separates the data into the different classes. It essentially learns from the patterns in the training data to make future predictions.

### 4. **Making Predictions with the Trained Model**
```python
y_pred = clf.predict(Xtest_tf)
```

- **`y_pred = clf.predict(Xtest_tf)`**: After training, the model is used to **predict** the labels for the **test data** (`Xtest_tf`).

  - **`Xtest_tf`**: This is the **TF-IDF matrix** of the test data, which contains the feature vectors for the test sentences. These vectors represent the words or n-grams in each sentence, just like in the training data.
  
  - **`predict`**: This method uses the trained SVM model to classify each test sample based on the learned decision boundary (hyperplane). It predicts whether each test sentence is "positive" or "negative" (or whatever the classes are for your specific task).
  
  - **`y_pred`**: This is the array or list of predicted labels for each of the test sentences. These are the model's best guesses about the labels for the test data, based on what it learned from the training data.


```python
from sklearn import svm
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(Xtrain_tf, Ytrain)
#Predict the response for test dataset
y_pred = clf.predict(Xtest_tf)
```


```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest,y_pred))
print(classification_report(Ytest,y_pred))
matrix = plot_confusion_matrix(clf , Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for SVM Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[179  25]
     [ 14 190]]
    0.9044117647058824
                  precision    recall  f1-score   support
    
               0       0.93      0.88      0.90       204
               1       0.88      0.93      0.91       204
    
        accuracy                           0.90       408
       macro avg       0.91      0.90      0.90       408
    weighted avg       0.91      0.90      0.90       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_40_2.png)
    



```python
fprSVM, tprSVM, thresholdsSVM = metrics.roc_curve(Ytest,y_pred)
```

# Random Forest

Let's break down the code step by step to understand what each part does in simple terms.

### 1. **Importing the Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier
```

- **`from sklearn.ensemble import RandomForestClassifier`**: This line imports the **RandomForestClassifier** from the **`sklearn.ensemble`** module.
  
  - **Random Forest** is an **ensemble machine learning algorithm** that combines many **decision trees** to improve classification performance.
  - A **decision tree** makes predictions based on feature values and splits the data into subsets at each node. However, individual decision trees can overfit the data (i.e., become too specific to the training data and not generalize well to new data).
  - **Random Forest** creates many decision trees (hence the "forest") and then makes predictions by averaging (or "voting") the predictions of all the trees in the forest. This tends to reduce overfitting and improve accuracy.
  - The Random Forest model is robust, can handle large datasets, and is less prone to overfitting than a single decision tree.

### 2. **Creating the Random Forest Classifier Object**
```python
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0).fit(Xtrain_tf, Ytrain)
```

- **`rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)`**: This creates an instance of the **RandomForestClassifier** class and assigns it to the variable `rf_classifier`.

  - **`n_estimators=100`**: This parameter sets the number of decision trees in the random forest. In this case, the forest will consist of **100 trees**.
    - More trees typically result in better performance, but they also increase the computation time and complexity. Here, 100 trees are a good balance between performance and computation.

  - **`criterion='entropy'`**: This parameter specifies the method for measuring the quality of a split when building each decision tree. You have two options:
    - **'entropy'** (Information Gain) measures how well the data is split by considering the reduction in uncertainty (entropy).
    - The alternative is **'gini'** (Gini Impurity), which is another way to measure the "impurity" of the splits.
    - **'entropy'** is chosen here, which means that each tree in the forest will try to split the data in a way that maximizes information gain (reduces entropy).

  - **`random_state=0`**: This sets the random seed for the random number generator used in the training process. By fixing the random state, you ensure that the results are reproducible each time you run the code. Without setting this, the trees might be built differently each time you run the model.

- **`.fit(Xtrain_tf, Ytrain)`**: This method trains the Random Forest model using the **training data** (`Xtrain_tf`) and the **training labels** (`Ytrain`).

  - **`Xtrain_tf`**: These are the features of the training data, represented as a **TF-IDF matrix**. Each sentence is transformed into a numerical vector, where each feature corresponds to the TF-IDF score of a word or n-gram in the sentence.
  
  - **`Ytrain`**: These are the **true labels** of the training data, which represent the correct class for each sentence. For example, this could be "positive" or "negative" for sentiment analysis.

  - The **`fit`** method builds the **Random Forest model** by training 100 decision trees, each of which learns to classify the data based on the patterns in `Xtrain_tf` and `Ytrain`.

### 3. **Making Predictions with the Trained Model**
```python
y_pred = rf_classifier.predict(Xtest_tf)
```

- **`y_pred = rf_classifier.predict(Xtest_tf)`**: After the model is trained, it is used to **predict** the labels for the **test data** (`Xtest_tf`).

  - **`Xtest_tf`**: This is the **TF-IDF matrix** for the test sentences, which contains the features (words or n-grams) of each sentence in the test data, similar to the training data.
  
  - **`predict`**: This method uses the **trained Random Forest model** to make predictions for each sentence in the test data. The model aggregates the predictions of the 100 decision trees to produce a final prediction for each sentence.
  
  - **`y_pred`**: This variable contains the predicted labels for the test sentences. These predictions are the Random Forest model's best guesses about the classes (e.g., "positive" or "negative") of the test data.


```python
from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0).fit(Xtrain_tf, Ytrain)
y_pred=rf_classifier.predict(Xtest_tf)
```


```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest,y_pred))
print(classification_report(Ytest,y_pred))
matrix = plot_confusion_matrix(rf_classifier, Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Random Forest Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[149  55]
     [ 13 191]]
    0.8333333333333334
                  precision    recall  f1-score   support
    
               0       0.92      0.73      0.81       204
               1       0.78      0.94      0.85       204
    
        accuracy                           0.83       408
       macro avg       0.85      0.83      0.83       408
    weighted avg       0.85      0.83      0.83       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_45_2.png)
    



```python
fprRF, tprRF, thresholdsRF = metrics.roc_curve(Ytest,y_pred)
```

# SGD

Let's break down the code step by step to understand each part in simple terms:

### 1. **Importing the SGDClassifier**
```python
from sklearn.linear_model import SGDClassifier
```

- **`from sklearn.linear_model import SGDClassifier`**: This line imports the **SGDClassifier** from the **`sklearn.linear_model`** module.

  - **SGD** stands for **Stochastic Gradient Descent**, which is an optimization method used for training machine learning models.
  - The **SGDClassifier** is a linear model that uses stochastic gradient descent to find the best-fitting model parameters.
  - The classifier can be used for both **binary classification** (e.g., "yes" or "no") and **multiclass classification** problems.
  - In this case, you are using it with a **logistic loss function**, which makes it suitable for **logistic regression** tasks like classification (e.g., predicting whether an email is "spam" or "not spam").

### 2. **Creating the SGDClassifier Object**
```python
sgd_classifier = SGDClassifier(loss='log', penalty='l2', max_iter=10, random_state=0).fit(Xtrain_tf, Ytrain)
```

- **`sgd_classifier = SGDClassifier(loss='log', penalty='l2', max_iter=10, random_state=0)`**: This line creates an instance of the **SGDClassifier** and stores it in the variable `sgd_classifier`.

  - **`loss='log'`**: This specifies that you are using **logistic regression** for the classification task, as logistic regression uses a **log loss** function (also called **logistic loss**).
    - The **log loss** is a measure of how well the model's predicted probabilities match the actual class labels. It's commonly used for binary or multiclass classification tasks.

  - **`penalty='l2'`**: This defines the regularization technique used by the model. Regularization is a method to prevent the model from overfitting to the training data.
    - **'l2' penalty** means **Ridge regularization** (also known as **L2 regularization**), which adds a penalty to the cost function based on the sum of the squares of the model's coefficients. This helps to shrink the coefficients and reduce overfitting.

  - **`max_iter=10`**: This specifies the maximum number of **iterations** the algorithm should run during the training process.
    - Here, **10** means that the model will perform **up to 10 iterations** to update the weights and converge to an optimal solution. If the model converges earlier, it will stop training before reaching 10 iterations.

  - **`random_state=0`**: This is used to set the **random seed** for the random number generator. It ensures that the training process is reproducible, meaning that every time you run the code, the results will be the same. Without setting this, the randomness in the algorithm may lead to slightly different results each time you run it.

- **`.fit(Xtrain_tf, Ytrain)`**: This method **trains** the SGDClassifier model using the **training data** (`Xtrain_tf`) and **training labels** (`Ytrain`).

  - **`Xtrain_tf`**: This is the **TF-IDF matrix** of the training data. Each sentence is represented as a vector of features (e.g., words or n-grams) with corresponding **TF-IDF** values.
  
  - **`Ytrain`**: These are the **true labels** of the training data, which represent the actual class for each sentence. For example, these could be "positive" or "negative" in a sentiment analysis task.

  - The **`fit`** method trains the model using **stochastic gradient descent (SGD)**. The model tries to find the best set of weights for the features by minimizing the **log loss** using the SGD algorithm. This helps the model learn the patterns in the training data that best separate the classes.

### 3. **Making Predictions with the Trained Model**
```python
y_pred = sgd_classifier.predict(Xtest_tf)
```

- **`y_pred = sgd_classifier.predict(Xtest_tf)`**: After training the model, this line uses the trained **SGDClassifier** to **predict** the labels for the **test data** (`Xtest_tf`).

  - **`Xtest_tf`**: This is the **TF-IDF matrix** for the test data, just like the training data, containing the features (words or n-grams) of the test sentences.

  - **`predict`**: This method uses the **trained SGDClassifier model** to make predictions about the class labels of the test data. The model takes the test features and classifies them into the predicted classes based on the patterns it learned during training.

  - **`y_pred`**: This is the list of predicted labels for the test data. These are the model's best guesses about the classes (e.g., "positive" or "negative") of each sentence in the test set.


```python
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(loss = 'log',penalty='l2', max_iter=10,random_state=0).fit(Xtrain_tf, Ytrain)
y_pred=sgd_classifier.predict(Xtest_tf)
```

    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\linear_model\_stochastic_gradient.py:173: FutureWarning: The loss 'log' was deprecated in v1.1 and will be removed in version 1.3. Use `loss='log_loss'` which is equivalent.
      warnings.warn(



```python
print(confusion_matrix(Ytest,y_pred))
print(accuracy_score(Ytest,y_pred))
print(classification_report(Ytest,y_pred))
matrix = plot_confusion_matrix(sgd_classifier, Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for SGD Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[177  27]
     [ 16 188]]
    0.8946078431372549
                  precision    recall  f1-score   support
    
               0       0.92      0.87      0.89       204
               1       0.87      0.92      0.90       204
    
        accuracy                           0.89       408
       macro avg       0.90      0.89      0.89       408
    weighted avg       0.90      0.89      0.89       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_50_2.png)
    



```python
fprSGD, tprSGD, thresholdsSGD = metrics.roc_curve(Ytest,y_pred)
```

# RandomizerSearchCV

Let's break down the code into smaller steps and explain each part in simple terms.

### 1. **Setting up the SVM Classifier**
```python
svm_clf = svm.SVC(probability=True, random_state=1)
```

- **`svm.SVC`**: This is the **Support Vector Classifier (SVC)** from the **`sklearn.svm`** module. It is used for classification tasks, where the goal is to separate data into different categories using a hyperplane (line, plane, or higher-dimensional hyperplane).
  
- **`probability=True`**: This parameter enables **probability estimates** for the classifier. Normally, SVM provides a hard decision (either class 0 or class 1), but when `probability=True`, the classifier also outputs the probability of each sample belonging to each class. This is useful when you need to evaluate model performance using metrics like **AUC** (Area Under the Curve).
  
- **`random_state=1`**: This ensures that the random processes in the algorithm (like the random initialization of the model) are reproducible. Setting the same `random_state` value will give the same result each time you run the code.

### 2. **Setting Up the AUC Scorer**
```python
auc = make_scorer(roc_auc_score)
```

- **`make_scorer(roc_auc_score)`**: This creates a custom scoring function using **AUC** (Area Under the ROC Curve) as the metric to evaluate the model during the search. **AUC** is used to measure how well the model distinguishes between the classes. A higher AUC score indicates a better performing model.
  
- **`roc_auc_score`**: This is a metric from the **`sklearn.metrics`** module that computes the **AUC score**, which evaluates the classifier’s ability to correctly classify both classes (e.g., class 0 and class 1) based on the predicted probabilities.

### 3. **Defining the Parameter Search Space**
```python
rand_list = {
    "C": stats.uniform(2, 10),
    "gamma": stats.uniform(0.1, 1),
    'kernel': ['rbf']
}
```

- **`rand_list`**: This is a dictionary that defines the parameter space over which we will perform the random search.

  - **`"C": stats.uniform(2, 10)`**: This defines the range for the **`C`** parameter of the SVM. The **`C`** parameter controls the trade-off between achieving a low error on the training data and minimizing the model complexity. Higher values of `C` can lead to better fit on the training data, but can also cause overfitting. This line tells the random search to sample values for `C` from a uniform distribution between **2 and 12** (the value starts at 2 and ranges up to 10).
  
  - **`"gamma": stats.uniform(0.1, 1)`**: This defines the range for the **`gamma`** parameter, which controls the influence of each training sample. Lower values make the model simpler, and higher values make the model more flexible but can lead to overfitting. This line tells the random search to sample values for `gamma` from a uniform distribution between **0.1 and 1**.

  - **`'kernel': ['rbf']`**: This specifies the type of kernel to use in the SVM. The **Radial Basis Function (RBF)** kernel is commonly used for classification tasks and can handle non-linear relationships between the features. We are only considering the **RBF kernel** in this case.

### 4. **Performing the Random Search for Hyperparameter Tuning**
```python
rand_search = RandomizedSearchCV(
    svm_clf,
    param_distributions=rand_list,
    n_iter=20,
    n_jobs=4,
    cv=3,
    random_state=2017,
    scoring=auc
)
```

- **`RandomizedSearchCV`**: This is a function that performs **randomized hyperparameter search** to find the best parameters for the model. It randomly samples parameter combinations from the specified parameter distributions and evaluates them using cross-validation.

  - **`svm_clf`**: This is the SVM classifier defined earlier. It is the model we want to tune.
  
  - **`param_distributions=rand_list`**: This is the dictionary that defines the search space for the hyperparameters (the range of values for **`C`**, **`gamma`**, and **`kernel`**). The random search will sample values from this distribution.
  
  - **`n_iter=20`**: This specifies that the random search should try **20 different combinations** of hyperparameters. So it will randomly pick 20 different values from the search space and evaluate the performance of the model using those values.
  
  - **`n_jobs=4`**: This specifies that the random search will use **4 CPU cores** in parallel to perform the search more efficiently. It helps speed up the process when you have multiple CPU cores available.

  - **`cv=3`**: This specifies that **3-fold cross-validation** will be used. This means the data will be split into 3 parts, and the model will be trained and validated 3 times (with different splits), and the average performance will be reported.

  - **`random_state=2017`**: This ensures that the random sampling of parameters is reproducible (for consistent results).

  - **`scoring=auc`**: This specifies that the **AUC score** will be used as the evaluation metric during the search. The random search will select the combination of hyperparameters that yields the highest AUC score.

### 5. **Fitting the Random Search Model**
```python
rand_search.fit(Xtrain_tf, Ytrain)
```

- **`rand_search.fit(Xtrain_tf, Ytrain)`**: This trains the **randomized search** using the training data (`Xtrain_tf`) and the corresponding labels (`Ytrain`).
  
  - The search will run for 20 iterations, trying different combinations of hyperparameters. Each combination will be evaluated using **3-fold cross-validation** and the AUC score as the evaluation metric.
  - After the search is complete, `rand_search` will contain the best combination of hyperparameters for the SVM classifier.

### 6. **Viewing the Results of the Random Search**
```python
rand_search.cv_results_
```

- **`rand_search.cv_results_`**: This contains the results of the **randomized search** for each parameter combination tried. It includes details like:
  - The set of hyperparameters that were tested.
  - The mean and standard deviation of the cross-validation scores (AUC) for each set of parameters.
  - The best combination of hyperparameters based on the AUC score.

### 7. **Making Predictions Using the Best Model**
```python
rand_predictions = rand_search.predict(Xtest_tf)
```

- **`rand_predictions = rand_search.predict(Xtest_tf)`**: After the random search completes and finds the best set of hyperparameters, it uses the **best model** to predict the labels of the test data (`Xtest_tf`).

  - The predicted labels (`rand_predictions`) are based on the best SVM model, which was trained with the optimal hyperparameters found during the random search.


```python
svm_clf = svm.SVC(probability = True, random_state = 1)
auc = make_scorer(roc_auc_score)

# RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
rand_list = {"C": stats.uniform(2, 10),
             "gamma": stats.uniform(0.1, 1),
             'kernel': ['rbf']}

rand_search = RandomizedSearchCV(svm_clf , param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = 3, random_state = 2017, scoring = auc)
rand_search.fit(Xtrain_tf, Ytrain)
rand_search.cv_results_
rand_predictions = rand_search.predict(Xtest_tf)

```


```python
print(confusion_matrix(Ytest,rand_predictions))
print(accuracy_score(Ytest,rand_predictions))
print(classification_report(Ytest,rand_predictions))
matrix = plot_confusion_matrix(rand_search, Xtest_tf, Ytest, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix Plot for Random Forest Classifier', color='black')
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10,6)
plt.show()
```

    [[180  24]
     [ 14 190]]
    0.9068627450980392
                  precision    recall  f1-score   support
    
               0       0.93      0.88      0.90       204
               1       0.89      0.93      0.91       204
    
        accuracy                           0.91       408
       macro avg       0.91      0.91      0.91       408
    weighted avg       0.91      0.91      0.91       408
    


    C:\Users\USER\AppData\Roaming\Python\Python310\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)



    
![png](output_55_2.png)
    



```python
fprSVM_optimized, tprSVM_optimized, thresholds_optimized = metrics.roc_curve(Ytest,y_pred)
```

# ROC


```python
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc

plt.figure(1)
plt.figure(figsize=(8, 6), dpi=300)
plt.plot([0, 1], [0, 1], 'k--')

auc_NB = auc(fprNB, tprNB)
auc_LR = auc(fprLR, tprLR)
auc_DT = auc(fprDT , tprDT)
auc_SVM = auc(fprSVM, tprSVM)
auc_RF = auc(fprRF, tprRF)
auc_SGD = auc(fprSGD, tprSGD)
auc_SVM_optimized = auc(fprSVM_optimized, tprSVM_optimized)

plt.plot(fprNB, tprNB, label='Naive Bayes (area = {:.3f})'.format(auc_NB))
plt.plot(fprLR, tprLR, label='Logistic Regression (area = {:.3f})'.format(auc_LR))
plt.plot(fprDT, tprDT, label='Decision Tree (area = {:.3f})'.format(auc_DT))
plt.plot(fprSVM, tprSVM, label='SVM (area = {:.3f})'.format(auc_SVM))
plt.plot(fprRF, tprRF, label='Random Forest (area = {:.3f})'.format(auc_RF))
plt.plot(fprSGD, tprSGD, label='SGD (area = {:.3f})'.format(auc_SGD))
plt.plot(fprSVM_optimized, tprSVM_optimized, label='SVM after RandomizedSearchCV (area = {:.3f})'.format(auc_SVM_optimized))


plt.legend(loc='lower right')
plt.show()
```


    <Figure size 640x480 with 0 Axes>



    
![png](output_58_1.png)
    



```python

```
