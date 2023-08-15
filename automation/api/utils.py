from nltk.corpus import stopwords
import re

# Download of stopwords is required for Docker image
import nltk
nltk.download("stopwords")

def text_preprocessing(text, language, minWordSize):
    
    # remove html
    # text_no_html = BeautifulSoup(str(text),"html.parser" ).get_text()
    
    # remove non-letters
    # text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(text_no_html)) 
    # text_alpha_chars = re.sub("[^a-zA-Z']", " ", str(text))
    # text_alpha_chars = re.sub("[^а-яА-ЯЁ]", " ", str(text))
    if language == 'russian' :
        text_alpha_chars = re.sub("[^a-zA-Zа-яА-ЯЁ]", " ", str(text))
    else:
        text_alpha_chars = re.sub("[^a-zA-Z'æøåÆØÅ]", " ", str(text))
        
    # convert to lower-case
    text_lower = text_alpha_chars.lower()
    
    # remove stop words
    stops = set(stopwords.words(language)) 
    text_no_stop_words = ' '
    
    for w in text_lower.split():
        if w not in stops:  
            text_no_stop_words = text_no_stop_words + w + ' '
      
    # do stemming
    # not sure this is usefull for short product descriptions
    #
    #text_stemmer = ' '
    #stemmer = SnowballStemmer(language)
    #for w in text_no_stop_words.split():
    #    text_stemmer = text_stemmer + stemmer.stem(w) + ' '
         
    # remove short words
    text_no_short_words = ' '
    # for w in text_stemmer.split():
    for w in text_no_stop_words.split(): 
        if len(w) >=minWordSize:
            text_no_short_words = text_no_short_words + w + ' '
 
    return text_no_short_words