import re
import string
from zhon.hanzi import punctuation as hanzi_punctuation
import emoji
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
import datetime
import logging

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

stop_words = stopwords.words('english')
english_words = set(nltk.corpus.words.words())
# stemmer = PorterStemmer()
stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()

"""
Preprocessing
(1) 去除 RT @XXX:
(2) 去除@XXX
(3) 去除http://XXX, https://XXX
(4) 替代任何标点符号为空格
(5) 去除非英文内容, 包括Emoji,"...", 不包括#,-,',
    注意: 为#XXX前预留空格, ...后预留空格
NOTE: json文件解析出来的text包含...表示没有结束
"""
# def preprocess_text(text: str) -> str:
#     text = re.sub(r'RT @(\w+): ', ' ', text)
#     text = re.sub(r'@[\w_]+', ' ', text)
#     text = re.sub(r'http(s?)://[a-zA-Z0-9.?/&=:]+', ' ', text)
#     text = re.sub(r'[…\\\'\’\‘]', ' ', text)
#     text = re.sub(r'\d+', ' _NUMBER_ ', text)
#     # text = re.sub(r'[\U00010000-\U0010ffff\uD800-\uDBFF\uDC00-\uDFFF]+', '', text)
#     text = re.sub(r'[{}]+'.format(string.punctuation), ' ', text)
#     text = re.sub(r'[{}]+'.format(hanzi_punctuation), ' ', text)
#     text = emoji.demojize(text, delimiters=(' _','_ '))

#     # processed_words = []
#     # for word in nltk.wordpunct_tokenize(text):
#     #     word = str(TextBlob(word).correct())
#     #     word = stemmer.stem(word)
#     #     word = lemmatizer.lemmatize(word)
#     #     if word in english_words and word not in stop_words:
#     #         processed_words.append(word)
#     # text = " ".join(processed_words)
#     text = " ".join(stemmer.stem(str(TextBlob(word).correct())) for word in nltk.wordpunct_tokenize(text) if word in english_words and word not in stop_words)
#     # text = " ".join(word for word in nltk.wordpunct_tokenize(text) if word not in stop_words)
#     text = re.sub(r' +', ' ', text) # 去除多余的空格
#     return text

def preprocess_text(text: str) -> str:
    text = re.sub(r'RT @(\w+): ', ' ', text)
    text = re.sub(r'@[\w_]+', ' ', text)
    text = re.sub(r'http(s?)://[a-zA-Z0-9.?/&=:]+', ' ', text)
    text = re.sub(r'[\\\'\’\‘]', ' ', text)
    # 用TextBlob.correct()补全带有...的字符单词
    text = re.sub(r'\w+…', lambda match: str(TextBlob(match.group()[:-1]).correct()), text)
    text = re.sub(r'\d+', ' _NUMBER_ ', text)
    text = re.sub(r'[{}]+'.format(string.punctuation), ' ', text)
    text = re.sub(r'[{}]+'.format(hanzi_punctuation), ' ', text)
    text = emoji.demojize(text, delimiters=(' _','_ '))
    text = " ".join(word for word in nltk.wordpunct_tokenize(text) if word not in stop_words)
    text = re.sub(r' +', ' ', text) # 去除多余的空格
    return text

user_texts = load_pickle("/root/Lab_Related/data/Heter-GAT/Classic/text/User-Text.p")

# Time spent on RTX-2080-Ti: 3min 35.2s
# NOTE: 采用思路(2)
part = 5 # start from 0
usertext_docs = []
docs_filename = f"/root/data/HeterGAT/lda-model/texts-processed/ProcessedTexts_{part}.p"
for idx, (user, texts) in enumerate(user_texts.items()):
    if idx < part*20000:
        continue
    if idx % 100 == 0:
        logger.info(f"user={user}, len(docs)={len(usertext_docs)}")
        save_pickle(usertext_docs, docs_filename)
    if idx == (part+1)*20000:
        logger.info(f"Break At idx={idx}, user={user}")
        break
    docs = " _END_ ".join([preprocess_text(elem["text"]) for elem in texts.values()])
    usertext_docs.append(docs)
logger.info(f"Total Users={len(usertext_docs)}")
save_pickle(usertext_docs, docs_filename)
