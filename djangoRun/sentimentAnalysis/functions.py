import os,requests,twitter,re,emoji,nltk
import run_classifier,json
import tensorflow as tf
from twitter_creds import *
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from  nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()


def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def tagger(sublist):
  for i in sublist:
    if(type(i)!=tuple): return tagger(i)
    else:
      for j in i:
        if "NNP"==j:
          return i[0]
  return "No Entity Found"

def cleanData(text):
  text = text.lower()
  text = re.compile(r'https?://\S+|www\.\S+').sub(r'',text)
  text = re.compile(r'http?://\S+|www\.\S+').sub(r'',text)
  text = BeautifulSoup(text,'html.parser').text
  hashtags = [ [ tag[1: ] for tag in i.split() if tag.startswith('#') ] for i in [text] ]
  tags = [ [ ent[1: ] for ent in i.split() if ent.startswith("@") ] for i in [text] ]
  text = decontracted(text)
  text = [ " ".join([ sent for sent in text.split(" ") if sent.startswith("@") == False  ]) for text in [text]][0]
  text = re.sub('_',' ',emoji.demojize(text))
  text = text.replace('[^\w\s]','')
  text =  [" ".join(([word for word in word_tokenize(i) if not word in list((stopwords.words('english')))]))  for i in [text] ][0]
  text = lemmatizer.lemmatize(text)
  return text

class instantiateModel():
  def __init__(self,path):
    self.init_checkpoint = path+"/anshaj.ckpt"
    self.tokenization = run_classifier.tokenization
    processor = run_classifier.ColaProcessor()
    BATCH_SIZE = 32
    self.MAX_SEQ_LENGTH = 50
    self.tokenization.validate_case_matches_checkpoint(False, self.init_checkpoint)
    bert_config = run_classifier.modeling.BertConfig.from_json_file(path+"/bert_config.json")
    self.tokenizer = self.tokenization.FullTokenizer(vocab_file=path+"/vocab.txt", do_lower_case=False)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(model_dir=path,cluster=None,master=None,save_checkpoints_steps=500,tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1000,num_shards=8,per_host_input_for_training=is_per_host))
    model_fn = run_classifier.model_fn_builder(bert_config=bert_config,num_labels=3,init_checkpoint=self.init_checkpoint,learning_rate=1e-05,num_train_steps=None,num_warmup_steps=None,use_tpu=False,use_one_hot_embeddings=False)
    self.estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False,model_fn=model_fn,config=run_config,train_batch_size=BATCH_SIZE,eval_batch_size=BATCH_SIZE,predict_batch_size=BATCH_SIZE)
  
  def scorePredict(self):
    sentences = self.sentences
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = "0") for x in sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, ['0', '1', '-1'], self.MAX_SEQ_LENGTH, self.tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=self.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = self.estimator.predict(input_fn=predict_input_fn)
    self.pred = list(predictions)
    return self.pred

  def set_sentence(self,sentence=""):
    if not sentence: return "Please set correct sentence in array format [sentence1,sentence2]"
    self.sentences = sentence
    self.clean_sentence = [cleanData(i) for i in self.sentences]
    return

  def returnSentiment(self):
    if not self.sentences: return {"data":["Invalid Data"]}
    sentences = self.sentences
    prediction = self.scorePredict()
    score = [i['probabilities'].tolist() for i in prediction]
    labels = ['neutral','positive','negative']
    fullList = []
    pred = self.pred
    for i in range(len(sentences)):
      tag = ["neutral" if list(pred[i].values())[0].tolist().index(max(list(pred[i].values())[0].tolist())) == 0 else "positive" if \
      (list(pred[i].values())[0].tolist().index(max(list(pred[i].values())[0].tolist())) == 1) else "negative"  ][0]
      fi = {"text":sentences[i],"classifications":[]}
      fi['classifications'] = [{"confidence":max(score[i])}]
      fi['classifications'][0]['tag_name'] = tag
      fullList.append(fi)
    tag_name = [i['classifications'][0]['tag_name'] for i in fullList]
    name = [i['text'] for i in fullList]
    confidence = [i['classifications'][0]['confidence'] for i in fullList]
    return {"sentiment":[self.sentences,tag_name,confidence]}

  def monkey(self):
    if not self.sentences: return {"data":["Invalid Data"]}
    try:
      url = 'https://api.monkeylearn.com/v3/classifiers/cl_qkjxv9Ly/classify/'
      headers = {'content-type': 'application/json','Authorization': 'Token ca179abfed07484e5cea99e723fee37d1e7fd167'}
      data = """{}""".format({"data":self.sentences}).replace("'",'"')
      print(data)
      r = requests.post(url, headers=headers, data=data)
      print(r.text)
      tag_name = [i['classifications'][0]['tag_name'] for i in r.json()]
      name = [i['text'] for i in r.json()]
      confidence = [i['classifications'][0]['confidence'] for i in r.json()]
      return {"sentiment":[self.clean_sentence,tag_name,confidence]}
    except:
      try:
        url = 'https://api.monkeylearn.com/v3/classifiers/cl_qkjxv9Ly/classify/'
        headers = {'content-type': 'application/json','Authorization': 'Token ca179abfed07484e5cea99e723fee37d1e7fd167'}
        data = """{}""".format({"data":self.clean_sentence}).replace("'",'"')
        print(data)
        r = requests.post(url, headers=headers, data=data)
        print(r.text)
        tag_name = [i['classifications'][0]['tag_name'] for i in r.json()]
        name = [i['text'] for i in r.json()]
        confidence = [i['classifications'][0]['confidence'] for i in r.json()]
        return {"sentiment":[self.clean_sentence,tag_name,confidence]}
      except Exception as e:
        print(e)
        o = len(self.clean_sentence)
        k = ["none"]*o
        return {"sentiment":[k,k,k]}
      
  def findHashTags(self):
    sentences = self.sentences
    Hashtags = [re.findall(r'#[0-9A-Za-z]+',i) for i in sentences]
    return {"tags":Hashtags}

  def findatTheRates(self):
    sentences = self.sentences
    entities = [re.findall(r'@[0-9A-Za-z]+',i) for i in sentences]
    return {"@":entities}
  def twitName(self,p=[]):
    api = twitter.Api()
    try: api = twitter.Api(consumer_key=consumer_key,consumer_secret=consumer_secret,access_token_key=access_token_key,access_token_secret=access_token_secret)
    except Exception as e: return {"twitname":p+([e]*(len(self.sentences)-len(p)))}
    for i in self.sentences:
      a = re.findall(r'@[0-9A-Za-z]+',i)
      b = [j[1:] for j in a]
      for c in range(len(b)):
        try:
          b[c] = api.GetUser(screen_name = b[c]).name
          p.append(b)
        except Exception as e: p.append(str(e))
    return {"twitName":p}

  def nltkEntity(self):
    sentences = self.sentences
    p = []
    for i in range(len(sentences)):
      tokens = nltk.word_tokenize(sentences[i])
      tagged = nltk.pos_tag(tokens)
      entities = nltk.chunk.ne_chunk(tagged)
      entities = tagger(entities)
      p.append(entities)
    return {"nltk":p}
