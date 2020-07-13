from django.shortcuts import render, redirect
import requests
from django.views import View
from functions import *

class work():
    def __init__(self):
        self.counter = 0
        self.model = instantiateModel()

    def index(self,request):
        text = request.POST.get('text')
        if not text: return render(request,'index.html')
        text = text.split("\n")
        text = [i for i in text if i not in ["","\r"]]
        self.model.set_sentence(text)
        bert_sentiments = self.model.returnSentiment()
        monkey = self.model.monkey()
        findHashTags = self.model.findHashTags()['tags']
        findatTheRates = self.model.findatTheRates()['@']
        twitName = self.model.twitName()['twitName']
        nltkEntity = self.model.nltkEntity()['nltk']
        sentence_bert,labelBert,confidenceBert = bert_sentiments['sentiment']
        sentence_monkey,labelMonkey,confidenceMonkey = monkey['sentiment']
        # print(sentence_bert,findatTheRates,twitName,nltkEntity,findHashTags,labelMonkey,confidenceMonkey,labelBert,confidenceBert)
        result = zip(sentence_bert,sentence_monkey,findatTheRates,twitName,nltkEntity,findHashTags,labelMonkey,confidenceMonkey,labelBert,confidenceBert)
        return render(request, 'index.html', {"result":result, "result1":result})