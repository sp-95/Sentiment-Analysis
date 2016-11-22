from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")
print(stop_words)

example_sentence = "This is an example showing off stop word filtration."
words = word_tokenize(example_sentence)

# filtered_sentence = [w for w in words if w not in stop_words]
filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)