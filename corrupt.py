
from random import randint
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from NTN import NeuralTensorLayer
from DBManager import dbconfig
from pkl import pkl
import os
import pickle

class corrupt(dbconfig):
	def __init__(self,dsname,pred_atoms,pdm,dom_obj_map,sentences):
		dbconfig.__init__(self,dsname)
		self.in_size = 50
		self.out_size = 10
		self.pred_atoms = pred_atoms
		self.pdm = pdm
		self.sentences = sentences
		self.dom_obj_map = dom_obj_map
		print('generating embeddings')
		model = self.load('cor_embed.p',self.gen_embeddings)
		self.wv = model.wv
		self.vocab = model.wv.vocab
		
		print('generating corrupted atoms')
		p_atoms = self.load('cor_atoms.p',self.gen_corrupt_atoms)
		self.p_atoms = p_atoms

		print('running ntn')
		self.run_ntn()

	def load(self,fname,func):
		pfile = self.pickle_location + fname
		if os.path.exists(pfile):
			return pickle.load(open( pfile, "rb" ))
		obj = func()
		#pickle.dump(obj,open(pfile,"wb"))
		return obj


	def run_ntn(self):
		p_atoms = self.p_atoms
		for p in p_atoms:
			print('******************',p)
			#continue
			X_train,y_train,X_test,y_val = p_atoms[p]
			x_train1,x_train2 = np.array([self.wv[x[0]] for x in X_train]),np.array([self.wv[x[1]] for x in X_train])
			x_val1,x_val2 = np.array([self.wv[x[0]] for x in X_test]),np.array([self.wv[x[1]] for x in X_test])
			ntn = NeuralTensorLayer(self.out_size,self.in_size)
			#print(y_train,y_val)

			ntn.fit(x_train1,x_train2,y_train,x_val1,x_val2,y_val)



	def gen_embeddings(self):
		sentences = self.sentences
		model = Word2Vec(sentences, size=self.in_size,window=10,min_count=0)
		return model


	def gen_corrupt_atoms(self):
		dom_obj_map = self.dom_obj_map
		pred_atoms,pdm = self.pred_atoms,self.pdm
		pred_atom_map = {}
		for p in pred_atoms:
			if len(pred_atoms[p]) == 0:
				continue

			l = len(pred_atoms[p][0])
			if l < 2:
				continue

			l1 = len(dom_obj_map[pdm[p][0]])
			l2 = len(dom_obj_map[pdm[p][1]])
			poss_atoms,avail_atoms = l1*l2,len(pred_atoms[p])
			print(p,poss_atoms,avail_atoms)
			if poss_atoms < avail_atoms*2:
				continue

			pred_atom_map[p] = {}
			d1,d2 = str(pdm[p][0]),str(pdm[p][1])
			for [v1,v2] in pred_atoms[p]:
				dobj1,dobj2 = d1 + '_' + str(v1),d2 + '_' + str(v2)
				if dobj1 in self.vocab and dobj2 in self.vocab:
					pred_atom_map[p][(dobj1,dobj2)] = True


			#for x in self.vocab:
				#if x.startswith('2'):
					#print(x)
			#return
			c = 0
			k = 0
			while c < avail_atoms:
				k += 1
				if k > 1000 and c == 0:
					del pred_atom_map[p]
					break
				v1,v2 = randint(0,l1-1),randint(0,l2-1)
				dobj1,dobj2 = d1 + '_' + str(v1),d2 + '_' + str(v2)
				#print((dobj1,dobj2) not in pred_atom_map[p] , dobj1 in self.vocab , dobj2 in self.vocab,d1,d2)
				if (dobj1,dobj2) not in pred_atom_map[p] and (dobj1 in self.vocab and dobj2 in self.vocab):
					c += 1
					pred_atom_map[p][dobj1,dobj2] = False
					#print(c,avail_atoms)
			#return
		p_atoms = {}
		x,y = [],[]
		for p in pred_atom_map:
			x,y = x+list(pred_atom_map[p].keys()),y+list(pred_atom_map[p].values())

		x,y = np.array(x),np.array(y)
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,shuffle=True)
		p_atoms['all'] = (X_train,y_train,X_test,y_test)

		return p_atoms

				