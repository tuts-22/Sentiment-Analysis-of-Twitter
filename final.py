from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import PorterStemmer
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import sys
import csv
import random

##Cliticization

def clitic(sent):
	new_sent="";
	check="";
	i=0;
	while (i < len(sent)-1):
		if (sent[i+1]=="'"):
			if (sent[i]=="n"):
				if (i+2<len(sent) and sent[i+2]=="t"):
					new_sent+=" not";
					i+=2;
			else:
				new_sent+=sent[i];
				i+=1;
				if (i < len(sent)-1):
					i =i+1;
					check = sent[i];
					if (check == "s"):
						new_sent = new_sent +"s";
					else:
						if (check == 'm'):
							new_sent = new_sent +" am";
						else:
							if (i+1 < len(sent) and sent[i+1]!=' '):
								i =i+1;
								check = check + sent[i];
							if (check == "re"):
								new_sent = new_sent +" are";
							else:
								if (check=='ve'):
									new_sent+=" have";
								else:
									new_sent = new_sent + check; 
				
		else:
			new_sent = new_sent+sent[i];
		i=i+1;
	return new_sent;


def remove_tag(sent):
	new_sent="";
	check="";
	i=0;
	while (i < len(sent)):
		if (sent[i] =='@'):
			while (i<len(sent)-1 and ((sent[i+1]>='a' and sent[i+1]<='z') or (sent[i+1]>='A' and sent[i+1]<='Z') or(sent[i+1]>='0' and sent[i+1]<='9') or sent[i+1]=='_')):
				i = i+1;
		else:
			new_sent = new_sent+sent[i];
		i=i+1;
	return new_sent;


def remove_rep(sent):
	i=0;
	new_sent="";
	while (i<len(sent)-1):
		while (i<len(sent)-1 and sent[i]==sent[i+1]):
			i+=1;
		new_sent = new_sent + sent[i];
		i+=1;
	if (i==(len(sent)-1)):
		new_sent = new_sent + sent[i];

	return new_sent;


def remove_punc(sent):
	i=0;
	new_sent="";
	while (i<len(sent)):
		if ((sent[i]>="!" and sent[i]<=".") or (sent[i]>=":" and sent[i]<="@")):
			new_sent+=" ";
		else:
			new_sent = new_sent + sent[i];
		i+=1;

	return new_sent;

#def rem_rt(sent):
#	return re.sub(r'rt ', r' ', sent);

def not_follow(sent):
	i=0;
	new_sent="";
	n=0;
	for word in sent.split():
		if (word=="not" or word=="no" or word=="nt"):
			n=3
		else:
			if (n>0):
				new_sent += "not" + word + " ";
				n = n -1;
			else:
				new_sent+=word+" ";
		#print word

	return new_sent;

def preprocess(l):
	new_l = l;
	new_sent="";
	k=0;
	for sent in l:
		new_sent = clitic(sent);
		new_sent = remove_tag(new_sent);
		new_sent = remove_punc(new_sent);
		new_sent = remove_rep(new_sent);
		new_sent = not_follow(new_sent);
		#new_sent = stemmer(new_sent);
		new_l[k] = new_sent;
		k=k+1;
	return new_l;

###Input creation
test_file = open(sys.argv[1],'r');

data = [];

for row in test_file:
	data.append(row);
test_file.close();

##Feature extraction
test_input = preprocess(data);
vectorizer = joblib.load('vectorizer_12.joblib');

clf = joblib.load('model_12.pkl') 

##Testing
Xn =  vectorizer.transform(test_input); 
pred = clf.predict(Xn);

output_file = open(sys.argv[2],'w+');

for out in pred:
	output_file.write(out + '\n')
output_file.close(); 
