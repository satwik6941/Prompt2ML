import nltk
corpora = ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]
for c in corpora:
    nltk.download(c, quiet=True)
print("NLTK corpora downloaded")