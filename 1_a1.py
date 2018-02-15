from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
csvfile = open('training.csv')
train_file = csv.reader(csvfile, delimiter=',');

ratio = 0.8;

data = [];

for row in train_file:
	data.append(row);

random.shuffle(data)

train_input =[];
train_label =[];

for i in range(0,int(ratio*len(data))):
	train_label.append(data[i][0]);
	train_input.append(data[i][1]);

dev_input = [];
dev_label = [];

for i in range(int(ratio*len(data))+1,int((ratio+0.2)*len(data))):
	dev_label.append(data[i][0]);
	dev_input.append(data[i][1]);

##Feature extractionr
train_input = preprocess(train_input);
vectorizer = CountVectorizer( analyzer = 'word',ngram_range = (1,1), decode_error='ignore',min_df=0.0, max_df=1.0)
##You can also see tfidf based vectorizer

X = vectorizer.fit_transform(train_input)


##Model training
clf = MultinomialNB();
#clf = LogisticRegression(penalty = 'l2', C=10.0,);
clf.fit(X, train_label);

##Testing

Xn = vectorizer.transform(dev_input); 

pred = clf.predict(Xn);
accuracy=accuracy_score(pred,dev_label)
#print(accuracy(pred,dev_label));
print(accuracy)

