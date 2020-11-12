from random import shuffle
from gensim.models import Word2Vec
from gensim.models import Word2Vec
from keras.datasets import mnist
import numpy as np
import os
import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from numpy import array
from keras.utils import to_categorical
from random import shuffle
from sklearn.metrics import roc_curve
from cnn import CNN


class review_w2v:
	def __init__(self):
		pass

	def gen_images(self,sentences,train_test_objs,y):
		TOPN = 20
		model = Word2Vec(sentences, size=TOPN,window=1,min_count=0)
		for i in range(len(y)-1,-1,-1):
			obj = train_test_objs[i]
			if obj not in model.wv.vocab:
				train_test_objs.pop(i)
				y.pop(i)
				continue
			image = []
			for sim in model.most_similar(obj,topn = TOPN):
				image.append(model.wv[sim[0]])
			train_test_objs[i] = image


		y = [1 if x == '-1' else 0 for x in y]
		bal_reviews,bal_y = [],[]
		c = sum(y)
		for i in range(len(y)):
			pos_revs = len(bal_y) - sum(bal_y)
			if y[i] == 0 and  pos_revs > c:
				continue

			bal_reviews.append(train_test_objs[i])
			bal_y.append(y[i])

		img_train, img_test, y_train, y_test = train_test_split(bal_reviews, bal_y, test_size=0.2, random_state=42)
		il = [img_train, img_test, y_train, y_test]
		cnn = CNN()
		roc = cnn.run(il)
		print('neighbor : ',roc)
		score = cnn.run_NN(il)
		print('NN : ', score)


class review_sentence_generator:

	def __init__(self,pdm,pred_atoms,test_size,query_pred):
		sentences = []
		train_test_objs,y = [],[]
		for pname in pred_atoms:
			for objs in pred_atoms[pname]:
				s = []
				t = (pname,)
				for i,obj in enumerate(objs):
					s += [str(pdm[pname][i]) + '_' + str(obj)]
				s += [pname]
				t += tuple(objs)
				sentences += [s]
				
		print("sentences",len(sentences))
		self.sentences = sentences
		self.train_test_objs = train_test_objs
		self.y = y
		#rvw = review_w2v()
		#rvw.gen_images(sentences,train_test_objs,y)


	def gen_train_test_atoms(self,pred_atoms,test_size):
		atoms = {}
		train_atoms = {}
		test_atoms = {}
		for p in pred_atoms:
			for atom in pred_atoms[p]:
				t = (p,)
				t += tuple(atom)
				atoms[t] = 1
		atoms = list(atoms.keys())
		shuffle(atoms)

		l = len(atoms)
		test_lim = int(l*test_size)
		for atom in atoms[:test_lim]:
			test_atoms[atom] = 1

		for atom in atoms[test_lim:]:
			train_atoms[atom] = 1

		self.test_atoms = test_atoms
		self.train_atoms = train_atoms


