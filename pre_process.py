import re
import string
import nltk
import unicodedata
from bs4 import BeautifulSoup
from word2number import w2n
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pycontractions import Contractions

class TweetTextCleaner(object):
    def __init__(self):
        self.contractions_expander = Contractions('models/GoogleNews-vectors-negative300.bin')
        self.contractions_expander.load_models()

    # Remove html tags in text
    def remove_html_tags(self, text):
        cleaned_text = BeautifulSoup(text, 'lxml')
        return cleaned_text.get_text()
    
     # Remove 'RT' or 'rt' in text
    def remove_retweets(self, text):
        cleaned_text = re.sub(r'\bRT\b', '', text)
        return cleaned_text

    # Remove URLs in text
    def remove_urls(self, text):
        cleaned_text = re.sub('(?:http?\:\/\/|http?\:\/|http?\:|https?\:\/\/|https?\:\/|https?\:|www)\S+', '', text)
        return cleaned_text

    # Remove username with '@' in text
    def remove_mentions(self, text):
        cleaned_text = re.sub('@[^\s]+', '', text)
        return cleaned_text

    # Remove hashtags '#' in text
    def remove_hashtags(self, text):
        cleaned_text = re.sub('#\w+\s*', '', text)
        return cleaned_text

    # Normalized the accented character in text
    def remove_non_ascii(self, text):
        cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        # Remove turncate symbol
        cleaned_text = re.sub('[\.]{3,}', '', cleaned_text)
        return cleaned_text

    # Change the written number in word to actual number in text
    def change_word_to_number(self, text):
        changed_text = []
        for word in text.split():
            try:
                changed_text += [str(w2n.word_to_num(word))]
            except ValueError:
                changed_text += [word] 
        changed_text = ' '.join(changed_text)
        return changed_text

    # Remove numbers in text
    def remove_numbers(self, text):
        cleaned_text = ''.join([word for word in text if not word.isdigit()])  
        return cleaned_text
    
    # Lower case all words in text
    def case_folding(self, text):
        return text.lower()

    # Expand the existing contractions in text
    def expand_contractions(self, text):
        expanded_text = list(self.contractions_expander.expand_texts([text], precise=True))
        expanded_text = ' '.join(expanded_text)
        return expanded_text

    # Remove punctuations in text
    def remove_punctuations(self, text):
        english_punctuations = string.punctuation
        translator = str.maketrans('', '', english_punctuations)
        cleaned_text = text.translate(translator)
        return cleaned_text

    # Get the correspond antonym to the given word
    def get_antonym(self, word):
        word_antonyms = set()
        # Get the synsets to the given word
        for syn in wordnet.synsets(word):
            # Get the correspond lemmas to the given word
            for lemma in syn.lemmas():
                # Get all the antonyms to the given word
                for antonym in lemma.antonyms():
                    word_antonyms.add(antonym.name())

        # Get the relevant antonym
        if len(word_antonyms) != 0:
            return word_antonyms.pop()
        else:
            return None

    # Replace the negation words with the antonym in text
    def replace_negation(self, text):
        replaced_text = []
        index = 0
        tokenized_text = self.tokenize(text)
        token_length = len(tokenized_text)
        while index < token_length:
            current_word = tokenized_text[index]
            # Check if word is negation
            if (current_word == 'not' or current_word == 'no') and index+1 < token_length:
                current_antonym = self.get_antonym(tokenized_text[index+1])
                
                # Replace if negation word
                if current_antonym:
                    # Store the replaced negation word with the antonym
                    replaced_text.append(current_antonym)
                    index += 2
                    continue
            # Store the non-negation word
            replaced_text.append(current_word)
            index += 1

        replaced_text = ' '.join(replaced_text)

        return replaced_text

    # Tokenize sentence into word
    def tokenize(self, text):
        tokenized_text = word_tokenize(text) 
        return tokenized_text

    # Remove stopwords in text
    def remove_stopwords(self, text):
        stop_words = stopwords.words('english')
        cleaned_text =  [word for word in self.tokenize(text) if word not in stop_words]
        cleaned_text = ' '.join(cleaned_text)
        return cleaned_text

    # Get the correspond word tag
    def get_pos_tag(self,word):
        current_word_tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_map = {'J': wordnet.ADJ,
                   'N': wordnet.NOUN,
                   'V': wordnet.VERB,
                   'R': wordnet.ADV}

        return tag_map.get(current_word_tag, wordnet.NOUN)

    # Lemmatize each word in text
    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        lematized_text = [lemmatizer.lemmatize(word, self.get_pos_tag(word)) for word in self.tokenize(text)]
        lematized_text = [word for word in lematized_text if len(word) > 2]
        lematized_text = ' '.join(lematized_text)
        return lematized_text