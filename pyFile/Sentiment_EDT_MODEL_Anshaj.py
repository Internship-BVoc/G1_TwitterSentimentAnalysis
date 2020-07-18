# -*- coding: utf-8 -*-
"""
Title - Twitter Sentiment Analysis

Author 1 : Anshaj Goyal
Author 2 : Yash Satwani
Author 3 : Shreya Sharma

Github Repo - https://github.com/Internship-BVoc/G1_TwitterSentimentAnalysis
Colab Link - https://colab.research.google.com/drive/124dpNho9i4-0b6aVoVoifI8eolutfdkD
"""



import platform

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

def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 612, height = 612, background_color='white', stopwords=STOPWORDS, mask=mask).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='black')
    print()
    plt.imshow(word_cloud),plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = numpy.zeros(len(words))
    for t in count_data: total_counts+=t.toarray()[0]
    count_dict = sorted((zip(words, total_counts)), key=lambda x:x[1], reverse=True)[:50]
    words,counts = [w[0] for w in count_dict],[w[1] for w in count_dict]
    x_pos = numpy.arange(len(words)) 
    plt.figure(None, figsize = (15, 10))
    plt.subplot(title = '50 Most Common Words After cleaning entities')
    sns.set_context("notebook", font_scale = 0.8, rc={"lines.linewidth": 1})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.show()

def save_dataframe():
  try: df_data.to_csv('/content/drive/My Drive/'+str(int(time()))+'_Update_TS.csv')
  except: df_data.to_csv(str(int(time()))+'_Update_TS.csv')

def tagger(sublist,entit=[]):
  for i in sublist:
    if(type(i)!=tuple):
      return tagger(i)
    else:
      for j in i:
        if "NNP"==j: return i[0]

def NLTK_Entity_Set(i):
  global scount,fcount
  # sentence = " ".join(i)
  tokens = nltk.word_tokenize(i)
  tagged = nltk.pos_tag(tokens)
  entities = nltk.chunk.ne_chunk(tagged)
  if( "organization" in str(entities).lower()): scount+=1
  else: fcount+=1
  return entities

def buildWordVector(tokenText,vec=numpy.zeros(300).reshape((1, 300)),count=0.0):
  for word in tokenText:
    try: vec, count = vec+wv[word].reshape((1, 300)),count+1
    except KeyError: pass
  if(count): vec /= count
  return vec

def cleanSentence(text):
  text = text.lower()
  text = re.compile(r'https?://\S+|www\.\S+').sub(r'',text)
  text = re.compile(r'http?://\S+|www\.\S+').sub(r'',text)
  text = bs4.BeautifulSoup(text,'html.parser').text
  hashtags = [ [ tag[1: ] for tag in i.split() if tag.startswith('#') ] for i in [text] ]
  tags = [ [ ent[1: ] for ent in i.split() if ent.startswith("@") ] for i in [text] ]
  text = decontracted(text)
  text = [ " ".join([ sent for sent in text.split(" ") if sent.startswith("@") == False  ]) for text in [text]][0]
  text = re.sub('_',' ',emoji.demojize(text))
  text = text.replace('[^\w\s]','')
  text =  [" ".join(([word for word in word_tokenize(i) if not word in list((stopwords.words('english')))]))  for i in [text] ][0]
  text = lemmatizer.lemmatize(text)
  return text

def clean_df_data(p=0):
	global df_data
	df_data = df.copy()
	df_data.drop(columns=['tweet_id','tweet_coord'],inplace=True)
	df_data.text = df_data.text.apply(lambda text:text.lower() if type(text) == str else text)
	df_data.text = df_data.text.apply(lambda text: re.compile(r'https?://\S+|www\.\S+').sub(r'',text))
	df_data.text = df_data.text.apply(lambda text: re.compile(r'http?://\S+|www\.\S+').sub(r'',text))
	df_data.text = df_data.text.apply(lambda text: bs4.BeautifulSoup(text,'html.parser').text)
	df_data['hashtags'] = [ [ tag[1: ] for tag in i.split() if tag.startswith('#') ] for i in df_data.text ]
	df_data['tags'] = [ [ ent[1: ] for ent in i.split() if ent.startswith("@") ] for i in df_data.text ]
	df_data.text = df_data.text.apply(lambda x :decontracted(x))
	df_data.text = [ " ".join([ sent for sent in text.split(" ") if sent.startswith("@") == False  ]) for text in df_data.text]
	df_data.text = df_data.text.apply(lambda text : re.sub('_',' ',emoji.demojize(text)))
	df_data.text = df_data.text.str.replace('[^\w\s]','')
	df_data.text = [ " ".join(([word for word in word_tokenize(i) if not word in list((stopwords.words('english')))]))  for i in df_data.text ]
	df_data.text = df_data.text.apply(lambda text: lemmatizer.lemmatize(text))
	df_data['tokenised_text'] = [[word for word in word_tokenize(i)] for i in df_data.text]
	df_data.tweet_location,df_data.user_timezone = [df_data[i].fillna(df_data[i].value_counts().nlargest(n=1).index[0]) for i in ['tweet_location','user_timezone']]
	if(p): print("No Missing Data" if (all(df_data.isna().any()==False )) else False)
	df_data.tweet_created = df.tweet_created.apply(lambda string : "/".join(string.split()[0].split("/")[:2])+"/20"+string.split()[0].split("/")[2] )
	df_data['Numberless_text'] =df_data['text'].apply(lambda x: re.sub('[0-9]', '', x))

def normalClean_df():
	global df1
	df1 = df.copy()
	df1.drop(columns=['tweet_id','tweet_coord'],inplace=True)
	df1.text = df1.text.apply(lambda text:text.lower() if type(text) == str else text)
	df1.text = df1.text.apply(lambda text: re.compile(r'https?://\S+|www\.\S+').sub(r'',text))
	df1.text = df1.text.apply(lambda text: re.compile(r'http?://\S+|www\.\S+').sub(r'',text))
	df1.text = df1.text.apply(lambda text: bs4.BeautifulSoup(text,'html.parser').text)
	df1.text = df1.text.apply(lambda x :decontracted(x))
	df1.text = df1.text.apply(lambda text : re.sub('_',' ',emoji.demojize(text)))
	df1.text = df1.text.str.replace('[^\w\s]','')
	df1.text = df1.text.apply(lambda text: lemmatizer.lemmatize(text))
	df1['tokenised_text'] = [[word for word in word_tokenize(i)] for i in df1.text]

class downloadResources():

	def __init__(self):
		self.nerGit = "https://github.com/channel960608/pynerer.git"

	def gitClone(self,link=""):
		if not link:
			link = self.nerGit
		get_ipython().system(f'git clone {link}')

	def packages(self,nltkDownload=[],polyglotDownload=[]):
		for _ in nltkDownload:
			nltk.download(_)
		for _ in polyglotDownload:
			polyglot.downloader.downloader.download(_)


class installModules():

	def __init__(self,):
		self.system = platform.platform()
		print("Your system is {}".format(self.system))

	def install(self,moduleNames = []):
		if  type(moduleNames) != list or not moduleNames:
			return "Send modules in [moduleName(s)] format"
		for module in moduleNames:
			get_ipython().system(f'pip install {module}')
		return True

	def gitInstall(self):
		get_ipython().system('cd pynerer; python3 setup.py install')

	def mountDrive(self):
		from google.colab import drive
		drive.mount('/content/drive')

	def d2c(self,tf="",tt=""): #drive2colab
		if not tf or not tt: return "0"
		get_ipython().system(f'cp {tf} {tt}')

	def restartNotebook(self):
		chk = input("Please only perform this operation if you are running it on colab, to continue please press 1 else 0")
		if int(chk):
			print("Please perform all other operation again except installing modules")
			exit()

class importModules(installModules):

	def __init__(self):
		super().__init__()
		print("Super Class Initiated")

	def importAndInstallModules(self,finalImport=""):
		moduleFile = open("moduleFile.txt").read().split(";")
		for module in moduleFile:
			try: exec(module)
			except ModuleNotFoundError as e: 
				errorModule = e.name
				self.install([errorModule])
			exec(module)
			finalImport += module+"\n"
		return finalImport

	def ignoreWarnings(self):
		simplefilter("ignore", category=ConvergenceWarning)
		simplefilter("ignore", category=DeprecationWarning)



moduleNames = ['emoji','pyLDAvis','pycld2','polyglot','pyicu','Morfessor','python-twitter','ktrain','BertLibrary']
install = installModules()
install.install(moduleNames)
download = downloadResources()
download.gitClone()
install.gitInstall()
install.mountDrive()
install.d2c("/content/drive/My\ Drive/twitter_creds.py", "/content")
install.restartNotebook()

polyglotDownload = ["embeddings2.en","er2.en"]
nltkDownload = ['sentiwordnet','punkt','stopwords','wordnet','words','averaged_perceptron_tagger','maxent_ne_chunker','maxent_ne_chunker']
moduleImport = importModules()
finalImport = moduleImport.importAndInstallModules()
exec(finalImport)
moduleImport.ignoreWarnings()
download = downloadResources()
download.packages(nltkDownload,polyglotDownload)
lemmatizer = nltk.stem.WordNetLemmatizer()
sns.set_style('darkgrid')

try: df = pd.read_csv('/content/drive/My Drive/internship/Tweets-A.csv')
except : df = pd.read_csv('Tweets-A.csv')
df['tokenised_text'] = [[word for word in word_tokenize(i)] for i in df.text]
df = df.drop('Unnamed: 0',axis=1)
df['tags'] = [ [ ent[1: ] for ent in i.split() if ent.startswith("@") ] for i in df.text ]
print(df.head())

class visualisation():
	def __init__(self):
		"""Class for data visualisation"""
		return
	def entityTweetsArea(self):
		self.split_data = df_data.copy()
		print("Maximum Tweets Mentioned [UnitedAirways] & [VirginAmerica]")
		self.split_data,self.split_data.tags = df_data.iloc[[i for i,j in enumerate(df_data['tags']) for value in j] , :],[value for i in df_data['tags'] for value in i]
		self.split_data.tags.value_counts()[:20].plot.area(figsize=[10,10],color="g").set_xlabel('entities mentioned')
		return

	def labelEnityDistribution(self):
		print("virgin america airways have almost equal number of positive negative and neutral\nouthwestair have neutral higher means it is ok for airline to continue\nmost negative are of american air which shd be a pressing issue\npositive sentiment is maximum for usairways")
		split_data = self.split_data
		self.raw_df = split_data[(split_data.tags.isin(split_data.tags.value_counts()[:10].index.to_list()))]
		a = sns.FacetGrid(self.raw_df,col="tags",col_wrap=5, height=5,aspect =1)
		a = a.map(plt.hist,'airline_sentiment',color='g')
		return

	def tweetLabelEntity(self):
		a = sns.FacetGrid(self.raw_df,col="airline_sentiment",col_wrap=3, height=5,aspect =2)
		a = a.map(plt.hist,'tags',color='g')
		return

	def entityTweetsValueCounts(self):
		print("Tweet Value Counts in respect with entities")
		self.split_data.tags.value_counts()[1:15].plot.bar()
		return

	def topTimezone(self,n=10):
		df_data['user_timezone'].value_counts().sort_values(ascending=False)[:n].plot.bar().set_xlabel('')
		return

	def labelVStimezone(self,n=3):
		a=sns.FacetGrid(df_data[df_data['user_timezone'].isin(df_data.user_timezone.value_counts()[:n].index.to_list())],col='airline_sentiment',col_wrap=2,height=6,)
		a=a.map(plt.hist,'user_timezone',color='0')
		return

	def sentiMentDivision(self):
		print("# Out of 14500 Tweets it is visible that more than 70% of tweets have negative sentiment\n# It can also be a problem which can be more clearly told after topic or entities are separated from text with respect to sentiment")
		df_data['airline_sentiment'].value_counts().plot.bar().set_xlabel('Sentiments_Division')
		return

	def lda_pyldavis(self,n=50):
		self.n = n
		self.dictionary_LDA = gensim.corpora.Dictionary(df_data.tokenised_text)
		self.dictionary_LDA.filter_extremes(no_below=3)
		self.corpus = [self.dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in df_data.tokenised_text]
		%time self.lda_model = models.LdaModel(self.corpus, num_topics=n,id2word=self.dictionary_LDA,passes=4,alpha=[0.01]*n,eta=[0.01]*len(self.dictionary_LDA.keys()))
		return

	def showTopics(self,n=10):
		if(n>self.n): n = self.n
		print("CommonTopics With Thier Occurent With Chars")
		for i,topic in self.lda_model.show_topics(formatted=True, num_topics= n):
		    print(topic)
		print(self.lda_model[self.corpus[1]])
		return

	def wc1(self):	
		print("Most common Words")
		allWords = ' '.join([twts for twts in df_data.text])
		wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
		plt.imshow(wordCloud, interpolation="bilinear"),plt.axis('off')
		plt.show()
		return

	def wc2(self):
		print("Most Frequent Words In Whole Text")
		self.mask = numpy.array(PIL.Image.open(requests.get('http://www.clker.com/cliparts/F/O/V/V/E/w/tweeter-bird-hi.png', stream=True).raw))
		generate_wordcloud(' '.join([twts for twts in df_data.text]), self.mask)
		return

	def wc3(self):
		print("# Most Frequent Words In Negative Labeled Sentiment")
		generate_wordcloud(' '.join([twts for twts in df_data.loc[df_data["airline_sentiment"] == "negative", 'text']]), self.mask)
		return

	def wc4(self):
		print("# Most Frequent Word In Positive Labeled Sentiment")
		generate_wordcloud(' '.join([twts for twts in df_data.loc[df_data["airline_sentiment"] == "positive", 'text']]), self.mask)
		return

	def visPlot(self):
		print("# Visualise the 50 most common words after cleaning entities and text")
		count_vectorizer = CountVectorizer(stop_words='english')
		count_data = count_vectorizer.fit_transform(df_data['text'])
		common_words(count_data, count_vectorizer)
		return

	def tweetVStime(self):
		print("# Tweet Created Frequency v/s Time Distribution")
		df_data.tweet_created.value_counts().sort_values(ascending=True).plot.bar()
		df_data['tweet_created']= pd.to_datetime(df['tweet_created'])
		df_data['tweet_time'] = df_data.tweet_created.apply(lambda d : "day" if datetime.datetime.time(d).hour >=6 and datetime.datetime.time(d).hour<18 else "evening or night" )

		return

	def ptVStime(self):
		print("# positive tweets vs timing")
		f = df_data.tweet_time[df_data['airline_sentiment'] == 'positive'].value_counts().plot.bar()
		f.set_title('positive tweets vs timing')
		f.set_ylabel('count of positive tweets')
		f.set_xlabel('timing of tweet')
		return

	def neuVStime(self):
		print("# neutral tweets vs timing")
		f = df_data.tweet_time[df_data['airline_sentiment'] == 'neutral'].value_counts().plot.bar()
		f.set_title('neutral tweets vs timing')
		f.set_ylabel('count of neutral tweets')
		f.set_xlabel('timing of tweet')
		return

	def ntVStime(self):
		print("# negative tweets vs timing")
		f = df_data.tweet_time[df_data['airline_sentiment'] == 'negative'].value_counts().plot.bar()
		f.set_title('negative tweets vs timing')
		f.set_ylabel('count of negative tweets')
		f.set_xlabel('timing of tweet')

	def fetchLatLong(self):
		print("# Latitude Longitude Map Visualisation")
		print("# One Time Run Code as it'll take approx 1 hour for 1000 rows aka 15 hours for given dataset so we've updated it in out dataset")
		print("Instead Run { object.mapPlot() }")
		if(int(input("Run? 0/1\n{0-False, 1-True\n"))):
		  geolocator = Nominatim(user_agent="Mozilla/5.0 (Macintosh; U; PPC Mac OS X; fi-fi) AppleWebKit/420+ (KHTML, like Gecko) Safari/419.3")
		  geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
		  df_data['coordinate'] = df_data['tweet_location'].apply(geocode)
		  df_data['latitude'] = df_data.coordinate.apply(lambda x : x.latitude if x!= None else None)
		  df_data['longitude'] = df_data.coordinate.apply(lambda x : x.longitude if x!= None else None)

		else:
			return self.mapPlot()
	def mapPlot(self):
		print("# Geographicial Representation of tweets origin location")
		BBox = (df_data.longitude.min(),df_data.longitude.max(),df_data.latitude.min(),df_data.latitude.max())
		style.use('ggplot')
		ruh_m = plt.imread('/content/drive/My Drive/internship/map.jpeg')
		fig, sc = plt.subplots(figsize = (15,5))
		sc.scatter(df_data[df_data['airline_sentiment']== 'positive'].longitude, df_data[df_data['airline_sentiment']== 'positive'].latitude, zorder=1, alpha= 0.2, c='g', s=50 ,marker= "^")
		sc.scatter(df_data[df_data['airline_sentiment']== 'neutral'].longitude, df_data[df_data['airline_sentiment']== 'neutral'].latitude, zorder=1, alpha= 0.2, c='y', s=25 , marker="s")
		sc.scatter(df_data[df_data['airline_sentiment']== 'negative'].longitude, df_data[df_data['airline_sentiment']== 'negative'].latitude, zorder=1, alpha= 0.2, c='r', s=5)
		sc.set_title('Plotting sentiment red-negative , green- postive , yellow- neutral')
		sc.set_xlim(BBox[0],BBox[1])
		sc.set_ylim(BBox[2],BBox[3])
		sc.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
		return


class misc():
	def __init__(self):
		print("Misceallanious Class")

	def USairlineFetch(self):
		self.entities = [ i['node']['name'] for i in requests.get('https://cache-api.ranker.com/lists/386932/items?limit=200&offset=0').json()['listItems']]
		self.tags = list(dict.fromkeys([ j for i in df_data.tags for j in i]))
		return

	def posNevNeutTokenise(self):
		print("#Positive Negative Neutral Labeled Text Tokenised")
		self.positiveTokens = word_tokenize(' '.join([i for i in df_data.loc[df_data['airline_sentiment'] == 'positive']['text']]))
		self.negativeTokens = word_tokenize(' '.join([i for i in df_data.loc[df_data['airline_sentiment'] == 'negative']['text']]))
		self.neutralTokens  = word_tokenize(' '.join([i for i in df_data.loc[df_data['airline_sentiment'] == 'neutral']['text']]))
		print("Positive Token - {}\nNegative Tokens - {}\nNeutral Tokens - {}".format(len(self.positiveTokens),len(self.negativeTokens),len(self.neutralTokens)))
		return

	def nerApproach(self):
		print("#Stanford NER Approach - python distro is obsolete")
		tag = nerer.HttpNER(host='127.0.0.1', port=8080)
		checker = [tag.get_entities(i) for i in df.text]
		print(checker.count({}))
		print("# Non Functional { Library Abandoned }")
		return

	def twitterEntityRecog(self):
		print("# Twitter API Rates Limit Error So Not Usable At Current Scenario")
		api = twitter.Api()
		api = twitter.Api(consumer_key=consumer_key,consumer_secret=consumer_secret,access_token_key=access_token_key,access_token_secret=access_token_secret)
		try: print(api.GetUser(screen_name = 'virginamerica').name)
		except Exception as e: print(e.message[0]['message'])

class polyGlotEntityApproach():

	def singleSentence(self,sentence = ""):
		for i in word_tokenize(sentence):
			print(Text(" ".join(i),hint_language_code='en').entities)


	def rawText(self):
		print("#For Raw Text")
		entities = [Text(" ".join(i),hint_language_code='en').entities if Text(" ".join(i),hint_language_code='en').entities!=[] \
            else 1 for i in df[df['tokenised_text'].str.len()>0].tokenised_text ]
		entitiesArray = [i for i in entities if i!=1]
		emptyEntities = [i for i in entities if i==1]
		df['PolyGlot_Entities'] = entities
		print("Total %s" % len(entitiesArray+emptyEntities))
		print("Failed %s" % len(emptyEntities))
		print("Success but can be inaccurate %s" % len(entitiesArray))
		print("Call df.head(20)[['text','PolyGlot_Entities']] to view data")
		return

	def preProcessText(self):
		print("#For PRE-Processed Text {HREF LINKS REMOVED, HTML TEXT REMOVED, WORDS LEMMATIZED , PUNCTUATIONS ARE REMOVED}")
		entities = [Text(" ".join(i),hint_language_code='en').entities if Text(" ".join(i),hint_language_code='en').entities!=[] \
		            else 1 for i in df1[df1['tokenised_text'].str.len()>0].tokenised_text ]
		entitiesArray = [i for i in entities if i!=1]
		emptyEntities = [i for i in entities if i==1]
		df1['PolyGlot Entities'] = entities
		print("Total %s" % len(entitiesArray+emptyEntities))
		print("Failed %s" % len(emptyEntities))
		print("Success but can be inaccurate %s" % len(entitiesArray))
		print("Call df1.head(20)[['text','PolyGlot Entities']] to view data")
		return

	def processedText(self):
		print("#For Processed Text")
		entities = [Text(" ".join(i),hint_language_code='en').entities if Text(" ".join(i),hint_language_code='en').entities!=[] \
		            else 1 for i in df_data[df_data['tokenised_text'].str.len()>0].tokenised_text ]
		entitiesArray = [i for i in entities if i!=1]
		emptyEntities = [i for i in entities if i==1]
		df_data['PolyGlot Entities'] = entities+[1]*(len(df_data)-len(entities))
		print("Total %s" % len(entitiesArray+emptyEntities))
		print("Failed %s" % len(emptyEntities))
		print("Success but can be inaccurate %s" % len(entitiesArray))
		print("Call df_data.head(20)[['text','PolyGlot Entities']] to view data")
		return

	def essence(self):
		print("PolyGlot Entity Approach came out to be very bad\nin our use case of fetching airlines")


class nltkEntity():
	# print 

	def singleSentence(self, sentence = ""):
		if not sentence: return "No Sentence Passed"
		tokens = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(tokens)
		entities = nltk.chunk.ne_chunk(tagged)
		print(entities)

	def rawText(self):
		scount=fcount=0
		print("#For Raw Text")
		df['NLTK_Entity_Set'] = df.text.apply(lambda x: NLTK_Entity_Set(x))
		print('Raw{Orignial} Text',scount,fcount,sep=" : ")
		print("Call df.head(20)[['text','NLTK_Entity_Set']] to view data")

	def preProcessText(self):
		scount=fcount=0
		print("#For Raw Text")
		df['NLTK_Entity_Set'] = df1.text.apply(lambda x: NLTK_Entity_Set(x))
		print('Raw{Orignial} Text',scount,fcount,sep=" : ")
		print("Call df1.head(20)[['text','NLTK_Entity_Set']] to view data")

	def processedText(self):
		scount=fcount=0
		print("#For Raw Text")
		df['NLTK_Entity_Set'] = df_data.text.apply(lambda x: NLTK_Entity_Set(x))
		print('Raw{Orignial} Text',scount,fcount,sep=" : ")
		print("Call df_data.head(20)[['text','NLTK_Entity_Set']] to view data")

	def essence(self):
		print("# It was very clear that raw/original text fetched more natural entities\n# than the formatted or processed text and the subset original text gave is quite promising despite of it's accuracy/inaccuracy")


	def apply(self):
		print(" Applying Extracted Entities in columns and replacing others to 1")
		df['Entities_Extracted'] = df.NLTK_Entity_Set.apply(tagger)
		indexes = df.Entities_Extracted.value_counts()[:5].index
		df.Entities_Extracted = df.Entities_Extracted.apply(lambda x: x if x in indexes  else 1 )
		print("Correctly Filled Entities in {} columns".format(df['Entities_Extracted'].value_counts()[1:6].sum()))
		print("Success Percentage {} columns".format(df['Entities_Extracted'].value_counts()[1:6].sum()/df.shape[0]*100))
		print("Fail Percentage {} columns".format(df['Entities_Extracted'].value_counts()[:1].sum()/df.shape[0]*100))
		print("Call df.head(20)[['text','Entities_Extracted']] to view data")



Models
df_target = df_data.airline_sentiment.replace({'positive':1,'neutral':0,"negative":-1})
df_target.value_counts()
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
corpus  = tfidfconverter.fit_transform(df_data['text']).toarray()
#Spliiting the data into train and test set.
X_train, X_test, Y_train, Y_test = train_test_split(corpus, df_target, test_size=0.20, stratify=df_target,random_state=30)


# Logistic Regression
if(int(input("Logistic Regression?? 0/1"))):
	lr = LogisticRegression()
	lr.fit(X_train,Y_train)
	y_pred = lr.predict(X_test)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("\nAccuracy of Logisticregression Model: {}\n".format(accuracy_score(Y_test, y_pred)))
	print(classification_report(Y_test,y_pred))


# Random Forest Classifier
if(int(input("Random_Forest?? 0/1"))):
	RFC =RandomForestClassifier()
	RFC.fit(X_train,Y_train)
	y_pred =RFC.predict(X_test)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("\nAccuracy of RandomForestClassifier Model: {}\n".format(accuracy_score(Y_test, y_pred)))
	print(classification_report(Y_test,y_pred))


# Support Vector Machine
if(int(input("SVM?? 0/1"))):
	sup =svm.SVC()
	sup.fit(X_train,Y_train)
	y_pred = sup.predict(X_test)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("\nAccuracy of SupportVectorMachine Model: {}\n".format(accuracy_score(Y_test, y_pred)))
	print(classification_report(Y_test,y_pred))



# Decision Tree Classifer
if(int(input("Decision Tree?? 0/1"))):
	DTC =DecisionTreeClassifier()
	DTC.fit(X_train,Y_train)
	y_pred =DTC.predict(X_test)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("\nAccuracy of DecisionTreeClassifier Model: {}\n".format(accuracy_score(Y_test, y_pred)))
	print(classification_report(Y_test,y_pred))

#Using Gridsearch parameter tuning
if(int(input("Gridsearch LR?? 0/1"))):
	param_grid = {'random_state': [ 10, 25,30, 50 ],'class_weight' : ['balanced'],'multi_class':["auto"],"solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
	CV_lr = GridSearchCV(estimator = lr, param_grid=param_grid, cv= 3)
	CV_lr.fit(X_train, Y_train)
	print(CV_lr.best_params_)
	print(CV_lr.best_estimator_)
	y_pred_lr = CV_lr.predict(X_test)
	print("Confusion Matrix \n {}".format(confusion_matrix(Y_test,y_pred_lr)))
	print("Accuracy of : {}".format(accuracy_score(Y_test, y_pred_lr)))
	print(classification_report(Y_test,y_pred_lr))


if(int(input("Gridsearch RFC?? 0/1"))):
	max_depth = [30, 20, 75, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10] 
	hyperF = dict( max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf)
	CV_RFC = GridSearchCV(RFC, hyperF, cv = 3)
	CV_RFC.fit(X_train, Y_train)
	print(CV_RFC.best_params_)  
	print(CV_RFC.best_estimator_)
	y_pred_rf = CV_RFC.predict(X_test)
	print("Confusion Matrix \n {}".format(confusion_matrix(Y_test,y_pred_rf)))
	print("Accuracy of : {}".format(accuracy_score(Y_test, y_pred_rf)))
	print(classification_report(Y_test,y_pred_rf))

if(int(input("Gridsearch SVM?? 0/1"))):
	param_grid = {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['poly']}  
	CV_sup = GridSearchCV(sup, param_grid, refit = True, verbose = 3) 
	CV_sup.fit(X_train, Y_train) 
	print(CV_sup.best_params_)  
	print(CV_sup.best_estimator_)
	y_pred_sup = CV_sup.predict(X_test)
	print("Confusion Matrix \n {}".format(confusion_matrix(Y_test,y_pred_sup)))
	print("Accuracy of : {}".format(accuracy_score(Y_test, y_pred_sup)))
	print(classification_report(Y_test,y_pred_sup))

if(int(input("Gridsearch DT?? 0/1"))):
	sample_split_range = list(range(1, 50))
	param_grid = dict(min_samples_split=sample_split_range)
	CV_DTC = GridSearchCV(DTC, param_grid, cv=10, scoring='accuracy')
	CV_DTC.fit(X_train, Y_train) 
	print(CV_DTC.best_params_) 
	print(CV_DTC.best_estimator_)
	y_pred_DTC = CV_DTC.predict(X_test)
	print("Confusion Matrix \n {}".format(confusion_matrix(Y_test,y_pred_DTC)))
	print("Accuracy of : {}".format(accuracy_score(Y_test, y_pred_DTC)))
	print(classification_report(Y_test,y_pred_DTC))


# Word2Vec Approach
# GoogleNews-vectors-negative300.bin.gz
if(int(input("Word2Vec Lib download?? 0/1"))):
	!gdown --id 0B7XkCwpI5KDYNlNUTTlSS21pQmM 
	wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
	corpus = [word_tokenize(i) for i in df_data.Numberless_text ]
	X_train, X_test, Y_train, Y_test = train_test_split(corpus, df_target, test_size=0.25, stratify=df_target,random_state=30)

# Logistic Regression Word2Vec Approach
if(int(input("LR Word2Vec?? 0/1"))):
	lr =LogisticRegression()
	lr.fit(train_vecs,Y_train)
	y_pred = lr.predict(test_vecs)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("Accuracy of Logisticregression Model:")
	print(accuracy_score(Y_test, y_pred))
	print(classification_report(Y_test,y_pred))  
	print(accuracy_score(Y_test, y_pred))

# SVM Word2Vec Approach
if(int(input("SVM Word2Vec?? 0/1"))):
	sup =svm.SVC(kernel='linear',decision_function_shape='ovr')
	sup.fit(train_vecs,Y_train)
	y_pred = sup.predict(test_vecs)
	print("Confusion Matrix")
	print(confusion_matrix(Y_test,y_pred))
	print("Accuracy of Logisticregression Model:")
	print(accuracy_score(Y_test, y_pred))
	print(classification_report(Y_test,y_pred))  
	print(accuracy_score(Y_test, y_pred))

# SentiWordNet Approach
# Found inaccurate in case of non-binary sentiment
def score_calculate(tokenised_text,totScore = 0.0):
	if not tokenised_text: return 0
	for token in tokenised_text:
		scoreList = swn.senti_synsets(token)
		if not scoreList: continue
		for score in scoreList: totScore = score.pos_score() - score.neg_score()
	if totScore ==0 : return 0
	if totScore > 0 : return 1
	if totScore < 0 : return -1
# return [0 if totScore==0 else 1 if totScore>0 else -1][0] #Incase to write in 1 line

def applySentiScore():
	for i in [df,df1,df_data]:
		i['sentiScore'] = i.tokenised_text.apply(score_calculate)

	print("Raw Data")
	X_train, X_test, y_train, y_test = train_test_split(df['text'], df_target, train_size=0.2, random_state=30)
	pred_y = df.iloc[y_test.index].sentiScore
	score1 = accuracy_score(y_test, pred_y)
	print("Confusion Matrix")
	print(confusion_matrix(y_test, pred_y))
	print("\nAccuracy of SentiWordNet Approach: {}\n".format(accuracy_score(y_test, pred_y)))
	print(classification_report(y_test, pred_y))

	print("Semi Processed Data")
	X_train, X_test, y_train, y_test = train_test_split(df1['text'], df_target, train_size=0.2, random_state=30)
	pred_y = df1.iloc[y_test.index].sentiScore
	score2 = accuracy_score(y_test, pred_y)
	print("Confusion Matrix")
	print(confusion_matrix(y_test, pred_y))
	print("\nAccuracy of SentiWordNet Approach: {}\n".format(accuracy_score(y_test, pred_y)))
	print(classification_report(y_test, pred_y))

	print("Processed Data")
	X_train, X_test, y_train, y_test = train_test_split(df_data['text'], df_target, train_size=0.2, random_state=30)
	pred_y = df_data.iloc[y_test.index].sentiScore
	score3 = accuracy_score(y_test, pred_y)
	print("Confusion Matrix")
	print(confusion_matrix(y_test, pred_y))
	print("\nAccuracy of SentiWordNet Approach: {}\n".format(accuracy_score(y_test, pred_y)))
	print(classification_report(y_test, pred_y))
	return pd.DataFrame({'Score':[score1,score2,score3]},index=['Raw_Data','Semi_Processed_Data','Processed_Data'])

# Bucket List Usage to extract positive text from negative labeled text or vice versa
# !gdown --id nLeY_hx8DxOI8E31Ngynu4xQLHeq #incase you don't have the list just download them on notebook using these 2 lines
# !gdown --id 13KKH4eZJedMjs4l0o_KQySMOb9olDIUs
def calculateScoreFromBucketList(sentence):
  sentence = word_tokenize(sentence.lower())
  negArray=[]
  posArray=[]
  for word in sentence:
    if word in negWords:
      negArray.append(word)
    if word in posWords:
      posArray.append(word)
  return [posArray,negArray]

def applyBucketList():
	global negWords,posWords
	negWords = open("negative-words.txt", encoding = "ISO-8859-1").read().lower().split("\n")
	posWords = open("positive-words.txt", encoding = "ISO-8859-1").read().lower().split("\n")
	df_data['posNegScore'] = df_data.text.apply(lambda x: calculateScoreFromBucketList(x))
	return df_data[['text','airline_sentiment','posNegScore']].head(20)

"""SentiWordNet Approach to find positive and negative word either of the text and their +,- score"""

def synsetPosNegScore(text):
  if not text: return [0,0]
  posScore=negScore=0
  tokenised_text = word_tokenize(text.lower())
  for token in tokenised_text:
    scoreList = swn.senti_synsets(token)
    if not scoreList: continue
    for score in scoreList: 
      posScore+= score.pos_score()
      negScore+= score.neg_score()
    return [posScore,negScore]

def applySentiForScoreOfOppLabel():
	df_data['sentiWordNetPosNegScore'] = df_data.text.apply(synsetPosNegScore)
	df_data[['text','airline_sentiment','sentiWordNetPosNegScore']].head(20)
	return 

# Deep Learning - #1
# Bidirectional Encoder Representations from Transformers (BERT) Using Ktrain

# pip install ktrain
def bertKtrain():
	global predictor
	import ktrain,random
	from ktrain import text
	import tensorflow as tf
	arr = ["the service is good", "The cost is expensive and customer service sucked","the flight was late but prices are ok","service is fine and cost is also fine"]
	arr1 = [cleanSentence(text) for text in arr]
	predictor.predict(arr)

	indexList = list(df_data.index)
	random.shuffle(indexList)
	eightList = [indexList[i] for i in range(0,len(indexList)*80//100)]
	data_train = df_data.iloc[eightList]
	twentyList = [indexList[i] for i in range(len(indexList)*80//100,len(indexList))]
	data_test = df_data.iloc[twentyList]
	print(data_train.shape[0]+data_test.shape[0],df_data.shape)
	(X_train,y_train), (X_text,y_test), preprocess = text.texts_from_df(data_train,'text','airline_sentiment',data_test,maxlen=100,preprocess_mode='bert')
	model = text.text_classifier('bert',(X_train,y_train), preproc= preprocess,multilabel=False)
	learner = ktrain.get_learner(model,(X_train,y_train),val_data=(X_text,y_test),batch_size=6)
	learner.lr_find()
	learner.lr_plot()
	learner.fit_onecycle(lr=1e-3,epochs=1) #learning rate 1e-3/1e-6
	predictor = ktrain.get_predictor(learner.model,preprocess)
	predictor.predict(arr)
	return "Use predictor.predict([]) to predict in future"


# Deep Learning #2 Bert using Ktrain with Data Balancing
def bertKtrainDataBalancing():
	posDataFrame = df_data[df_data.airline_sentiment=="positive"].airline_sentiment
	negDataFrame = df_data[df_data.airline_sentiment=="negative"].airline_sentiment
	neutralDataFrame = df_data[df_data.airline_sentiment=="neutral"].airline_sentiment
	posArray,negArray,neutArray = list(posDataFrame.index),list(negDataFrame.index),list(neutralDataFrame.index)
	random.shuffle(negArray)#,random.shuffle(neutArray),random.shuffle(posArray)
	finalDf = pd.concat([df_data.iloc[posArray[:2000]],df_data.iloc[negArray[:2000]],df_data.iloc[neutArray[:2000]]])
	print(finalDf.airline_sentiment.value_counts())
	indexList_2 = list(finalDf.index)
	random.shuffle(indexList_2)
	eightList_2 = [indexList_2[i] for i in range(0,len(indexList_2)*80//100)]
	data_train_2 = df_data.iloc[eightList_2]
	twentyList_2 = [indexList_2[i] for i in range(len(indexList_2)*80//100,len(indexList_2))]
	data_test_2 = df_data.iloc[twentyList_2]
	print(data_train_2.shape[0]+data_test_2.shape[0],finalDf.shape)
	print(finalDf.airline_sentiment.value_counts())
	(X_train_2,y_train_2), (X_text_2,y_test_2), preprocess2 = text.texts_from_df(data_train_2,'text','airline_sentiment',data_test_2,maxlen=50,preprocess_mode='bert')
	model2 = text.text_classifier('bert',(X_train_2,y_train_2), preproc= preprocess2,multilabel=True)
	learner2 = ktrain.get_learner(model2,(X_train_2,y_train_2),val_data=(X_text_2,y_test_2),batch_size=6)
	learner2.lr_find()
	learner2.lr_plot() #1e-6/1e-3
	learner2.fit_onecycle(lr=1e-6,epochs=1)
	predictor2 = ktrain.get_predictor(learner2.model,preprocess2)
	print("Normal Data : ",predictor2.predict(arr))
	print("Clean Data : ",predictor2.predict(arr1))

# RNN implementation with LSTM
def rnnLstm():
	onehot_repr=[one_hot(words,5000)for words in df_data['Numberless_text'].tolist()] 
	embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=20)
	model1=Sequential()
	model1.add(Embedding(5000,40,input_length=20))
	model1.add(Bidirectional(LSTM(100)))
	model1.add(Dropout(0.3))
	model1.add(Dense(1,activation='sigmoid'))
	model1.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	print(model1.summary())
	X_final,y_final=numpy.array(embedded_docs),numpy.array(df_target)
	X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=30,stratify =y_final)
	model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=64)
	model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)
	y_pred=model1.predict_classes(X_test)
	print(confusion_matrix(y_test,y_pred))
	print(accuracy_score(y_test,y_pred))
	return

# Deep Learning #3 Bert using BertLibrary
# pip install BertLibrary tensorflow-gpu==1.15.0
def bertWithBertLibrary():
	print("Please download https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip")
	print("Please Unzip it")
	if not input("Downloaded?? 1/0"): return
	global ft_model,ft_trainer,ft_evaluator,predictor
	from BertLibrary import BertFTModel
	TRAIN_SIZE = 0.80
	VAL_SIZE = 0.05
	label_text = df_data[['airline_sentiment','text']]
	label_text['airline_sentiment'] = label_text.airline_sentiment.replace({'positive':1,'neutral':0,"negative":-1})
	df_train_val, df_test = train_test_split(label_text, test_size=1-TRAIN_SIZE-VAL_SIZE, random_state=30)
	df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE / (VAL_SIZE + TRAIN_SIZE), random_state=30)
	print("TRAIN size:", len(df_train))
	print("VAL size:", len(df_val))
	print("TEST size:", len(df_test))
	ft_model = BertFTModel( model_dir='uncased_L-12_H-768_A-12',ckpt_name="bert_model.ckpt",labels=['0','1','-1'],lr=1e-05,num_train_steps=30000,\
		num_warmup_steps=1000,ckpt_output_dir='output',save_check_steps=1000,do_lower_case=False,max_seq_len=50,batch_size=32,)
	ft_trainer =  ft_model.get_trainer()
	ft_evaluator = ft_model.get_evaluator()
	if not os.path.exists("dataset"): os.mkdir("dataset")
	df_train.sample(frac=1.0).reset_index(drop=True).to_csv('dataset/train.tsv', sep='\t', index=None, header=None)
	df_val.to_csv('dataset/dev.tsv', sep='\t', index=None, header=None)
	df_test.to_csv('dataset/test.tsv', sep='\t', index=None, header=None)
	ft_trainer.train_from_file('dataset',steps=14000)
	ft_evaluator.evaluate_from_file('dataset') 
	predictor =  ft_model.get_predictor()
	prediction = list(predictor(arr))
	print("# Regular Text without features extracted")
	for i in range(len(prediction)):
  		print(arr[i],["neutral" if list(prediction[i]).index(max(list(prediction[i]))) == 0 else "positive" if (list(prediction[i]).index(max(list(prediction[i]))) == 1) else "negative"  ][0],sep=" : ")
	prediction = list(predictor(arr1))
	print("Processed and clean text")
	for i in range(len(prediction)):
  		print(arr1[i],["neutral" if list(prediction[i]).index(max(list(prediction[i]))) == 0 else "positive" if (list(prediction[i]).index(max(list(prediction[i]))) == 1) else "negative"  ][0],sep=" : ")




