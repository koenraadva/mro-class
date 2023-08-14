import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from dotenv import load_dotenv
from nltk.corpus import stopwords
from utils import connectWithAzure
from azureml.core import Dataset
from azureml.data.datapath import DataPath

# When you work locally, you can use a .env file to store all your environment variables.
# This line reads those variables.
load_dotenv()

MRO_LANG = os.environ.get('MRO_LANG').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))
# Environment variables are always strings, so we need to convert them to booleans.
PROCESS_DATA = os.environ.get('PROCESS_DATA').lower() in ('true', '1', 't')
SPLIT_DATA = os.environ.get('SPLIT_DATA').lower() in ('true', '1', 't')

def processAndUploadMroData(ws, mro_data, mro_lang):

    # Read the dataset into a pandas dataframe
    df_mro = ws.datasets[mro_data].to_pandas_dataframe(on_error='null', out_of_range_datetime='null')
    print('Reading the MRO dataset')
    # print(mro_df['product_descr'].head(50))

    # Fill Nan with ' '
    df_mro['product_descr'] = df_mro['product_descr']. fillna(' ')
    # mro_df['manufacturer'] = df['manufacturer']. fillna(' ')

    # Concatenate (or not) manufacturer to description
    # mro_df['product_descr_proc'] = df['product_descr'].astype(str)+' '+df['manufacturer'].astype(str)
    df_mro['product_descr_proc'] = df_mro['product_descr']

    # Preprocessing of our descriptions
    print('Text preprocessing')
    minWordLength = 2
    df_mro['product_descr_proc'] = df_mro.apply(lambda row: text_preprocessing(row.product_descr_proc, mro_lang, minWordLength), axis=1)

    # Remove rows with TBD category
    df_mro = df_mro[df_mro['category_lbl1']!='TBD']

    # Remove rows with empty descriptions
    df_mro = df_mro[df_mro['product_descr_proc']!='']
    
    # Register the processed dataframe as a new dataset
    print(f'Register processed dataset {mro_data}')
    new_dataset = Dataset.Tabular.register_pandas_dataframe(
                            dataframe=df_mro,
                            target=(ws.get_default_datastore(), f'processed/{mro_data}'),
                            name=f'processed_{mro_data}',
                            description=f'{mro_data} dataset with processed descriptions',
                            tags={'language': mro_lang, 'AI-Model': 'NB/LR', 'GIT-SHA': os.environ.get('GIT_SHA')}) # Optional tags, can always be interesting to keep track of these!

    print(f"Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'Processing completed')

# Text Preprocessor
# to be improved for short texts
#
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

def prepareDatasets(ws):

    # One dataset per language
    for mro_lang in MRO_LANG: 
        # mro_lang = MRO_LANG[0]
        # Language code is OK for current languages
        # todo ... not for Dutch
        mro_data = 'spare_parts_'+mro_lang[:2]
        processAndUploadMroData(ws, mro_data, mro_lang)

def splitDatasets(ws):

    for mro_lang in MRO_LANG:
        mro_data = 'spare_parts_'+mro_lang[:2]
        trainTestSplitData(ws, mro_data, mro_lang)

    print(f'Finished splitting train & test all datasets.')

def trainTestSplitData(ws, mro_data, mro_lang):

    print(f'Starting to split train & test {mro_data} datasets.')

    # Read the processed dataset into a pandas dataframe
    mro_df = ws.datasets[f'processed_{mro_data}'].to_pandas_dataframe(on_error='null', out_of_range_datetime='null')
    
    X = mro_df['product_descr_proc']
    y = mro_df['category_lbl1']+'/'+mro_df['category_lbl2']

    # Split the data into train and test sets
    # We might use the TabularDataset.random_split() method instead
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_FACTOR, random_state=42)

    # Create a new dataframe with the train data
    df_train = pd.DataFrame({'product_descr': X_train, 'class': y_train})

    # Create a new dataframe with the test data
    df_test = pd.DataFrame({'product_descr': X_test, 'class': y_test})

    # Register the train dataframe as a new dataset
    print(f'Register train dataset {mro_data}')
    train_dataset = Dataset.Tabular.register_pandas_dataframe(
                            dataframe=df_train,
                            target=(ws.get_default_datastore(), f'train/{mro_data}'),
                            name=f'train_{mro_data}',
                            description=f'{mro_data} Train dataset',
                            tags={'language': mro_lang, 'AI-Model': 'NB/LR', 'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')}) # Optional tags, can always be interesting to keep track of these!
    print(f"Training dataset registered: {train_dataset.id} -- {train_dataset.version}")

    print(f'Register test dataset {mro_data}')
    test_dataset = Dataset.Tabular.register_pandas_dataframe(
                            dataframe=df_test,
                            target=(ws.get_default_datastore(), f'test/{mro_data}'),
                            name=f'test_{mro_data}',
                            description=f'{mro_data} Test dataset',
                            tags={'language': mro_lang, 'AI-Model': 'NB/LR', 'Split size': str(TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')}) # Optional tags, can always be interesting to keep track of these!
    print(f"Testing dataset registered: {test_dataset.id} -- {test_dataset.version}")
    

def main():
    ws = connectWithAzure()

    if PROCESS_DATA:
        print('Processing the MRO data')
        prepareDatasets(ws)
    
    if SPLIT_DATA:
        print('Splitting the MRO data')
        splitDatasets(ws)

if __name__ == '__main__':
    main()