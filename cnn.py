from gensim.models import Word2Vec
from keras.datasets import mnist
import numpy as np
import os
import pickle


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from numpy import array
from keras.utils import to_categorical
from random import shuffle
from sklearn.metrics import roc_curve

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.input_layer import InputLayer


class CNN:
	def __init__(self):
		pass


	def NNWithSum(self,il):
		X_train,X_test, y_train,y_test = il[0],il[1],il[2],il[3]
		X_train = np.asarray(X_train)
		X_test = np.asarray(X_test)
		y_train = np.asarray(y_train)
		y_test = np.asarray(y_test)
		clf = MLPClassifier(hidden_layer_sizes=(50,10), max_iter=10000)
		clf.fit(X_train, y_train)
		pred = y_pred = clf.predict(X_test)
		print('sum: ',pred.shape,y_test.shape)
		#self.plot_roc(y_test,pred)
		tp,fp,fn = 0,0,0
		for i in range(len(pred)):
			if y_test[i][0] == 1:
				if pred[i][0] == y_test[i][0]:
					tp += 1
				else:
					fn += 1
			else:
				if pred[i][0] != y_test[i][0]:
					fp += 1
		p = (tp/(tp+fp))
		r = (tp/(tp+fn))
		f1 = 2*p*r/(p+r)
		print(f1)

	def run_NN(self,il):
		X_train,X_test, y_train,y_test = il[0],il[1],il[2],il[3]
		X_train = np.asarray([img[0] for img in X_train])
		X_test = np.asarray([img[0] for img in X_test])
		y_train = np.asarray(y_train)
		y_test = np.asarray(y_test)
		clf = MLPClassifier(hidden_layer_sizes=(50,50,10), max_iter=100000)
		clf.fit(X_train, y_train)
		pred = y_pred = clf.predict(X_test)
		print(y_test.shape,pred.shape,X_test.shape)
		return self.get_fscore(y_test,pred)


	def get_fscore(self,pred,y_test):
		tp,fp,fn,tn = 0,0,0,0
		for i in range(len(y_test)):
			if pred[i] == y_test[i] and pred[i] == 1:
				tp += 1
			elif pred[i] == y_test[i] and pred[i] == 0:
				tn += 1
			elif pred[i] == 1:
				fp += 1
			else:
				fn += 1
		p = tp/(tp+fp)
		r = tp/(tp+fn)
		fscore = 2*p*r/(p+r)
		return fscore



	def plot_roc(self,y_test,pred):
		pred = pred[:,1]
		auc = roc_auc_score(y_test, pred)
		print('AUC: %.3f' % auc)
		return auc

		fpr, tpr, thresholds = roc_curve(y_test, pred,)
		pyplot.plot([0, 1], [0, 1], linestyle='--',label='ROC curve (area = %0.2f)' % auc)
		pyplot.plot(fpr, tpr, marker='.',label='ROC curve (area = %0.2f)' % auc)
		pyplot.show()

	def run_sum(self,model,X_train):
		inp = model.input                                           # input placeholder
		outputs = [layer.output for layer in model.layers]          # all layer outputs
		functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

		layer_outs = [func([X_train]) for func in functors]
		ll = layer_outs[2][0]
		#print(len(ll))
		ndata = []
		for i in range(len(ll)):
			ndata.append([])
			l = [0]*len(ll[i])
			for j in range(len(ll[i])):
				tot = 0
				for k in range(len(ll[i][j])):
					l[k] += ll[i][j][k][0]
			ndata[i] = l
		ndata = np.array(ndata)
		return ndata

	def run(self,il):
		img_train,img_test, y_train,y_test = il[0],il[1],il[2],il[3]
		y1 = np.array(y_test[:])
		c = sum([1 for y in y_train if y == 1])
		print('ratio',c,len(y_train))
		m = len(img_train[0])

		y_train = np.array(y_train)
		
		y_test = np.array(y_test)
		#print(len(img_train),len(img_train[0]))
		for i,img in enumerate(img_train):
			#print(i)
			img_train[i] = array(img)
		img_train = array(img_train)

		for i,img in enumerate(img_test):
			img_test[i] = array(img)
		img_test = array(img_test)



		img_train = img_train.reshape((len(img_train),m,m,1))
		img_test = img_test.reshape((len(img_test),m,m,1))
		print(img_train.shape)

		y_train = to_categorical(y_train)
		y_test = to_categorical(y_test)
		#print('tot',len(y_train) + len(y_test))
		#print(y_train[0])
		c = sum([1 for y in y_test if y[0] == 1])
		#print('ratio',c,len(y_test))
		CVN = 4
		MPN = 2
		model = Sequential()
		model.add(Conv2D(m*2, kernel_size=CVN, activation='relu', input_shape=(m,m,1),padding='same'))
		model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding='same'))
		model.add(Conv2D(1, kernel_size=CVN, activation='relu',padding='same'))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(img_train, y_train, epochs=3)
		print(model.summary())
		pred = model.predict(img_test)
		nx_train = self.run_sum(model,img_train)
		nx_test = self.run_sum(model,img_test)
		nil = [nx_train,nx_test,y_train,y_test]

		#pred = np.array([p[0] for p in pred])
		print('cnn :',y1.shape,pred.shape)
		self.plot_roc(y1,pred)
		print("with sum:")
		self.NNWithSum(nil)	
		for th in range(5,1000,5):
			tp,fp,fn,tn = 0,0,0,0
			nth = th*1.0/1000.0
			#print(nth)
			for i in range(len(pred)):
				if pred[i][1] > nth and y1[i] == 1:
					tp += 1
				elif pred[i][1] > nth and y1[i] == 0:
					fp += 1
				elif y1[i] == 0:
					tn += 1
				else:
					fn += 1
			if tp == 0:
				continue
			#print(tp,fp,fn,tn)
			p = tp/(tp+fp)
			r = tp/(tp+fn)
			#print('Threashold : ',nth)
			#print('Precision : ',p)
			#print('Recall : ',r)
			print('F score : ',2*p* r/(p+r))


class word2vec_cnn:
	def __init__(self):
		pass

	def generate_train_test_db(self,ds,train_atoms,test_atoms):
		ofile = open(ds + '/db/train.txt','w')
		for atom in train_atoms:
			pname = atom[0]
			objs = atom[1]
			if len(atom) > 2:
				objs += ',' + atom[2]
			atom = pname + '(' + objs + ')\n'
			ofile.write(atom)
		ofile.close()

		ofile = open(ds + '/db/test.txt','w')
		for atom in test_atoms:
			pname = atom[0]
			objs = atom[1]
			if len(atom) > 2:
				objs += ',' + atom[2]
			atom = pname + '(' + objs + ')\n'
			ofile.write(atom)
		ofile.close()

	def make_images(self,sentences,pdm,pred_atoms,dom_sizes_map,ds,train_atoms,test_atoms,test_size):
		self.generate_train_test_db(ds,train_atoms,test_atoms)
		self.false_atoms = self.get_false_atoms(ds)
		pfile = ds + '/pickle/image_labels.p'
		if os.path.exists(pfile):
			il = pickle.load(open( pfile, "rb" ) )
			cnn = CNN()
			return cnn.run(il)


		c = sum([len(pred_atoms[p]) for p in pred_atoms])
		print('total atoms : ',c)
		images,random_images,labels = self.get_images_labels(sentences,pdm,pred_atoms,dom_sizes_map,train_atoms,test_atoms)
		
		#il = self.get_train_test_images(random_images[:],labels,test_size,train_atoms,test_atoms)
		#cnn = CNN()
		#roc_score = cnn.run_NN(il)
		#print('random : ',roc_score)

		cnn = CNN()
		il = self.get_train_test_images(images[:],labels,test_size,train_atoms,test_atoms)
		roc_score = cnn.run_NN(il)
		print('neighbor : ',roc_score)


		fscore = cnn.run_NN(il)
		print('NN : ',fscore)
		#pickle.dump(il,open(pfile,"wb"))
		fscore = cnn.run(il)


	def get_train_test_images(self,images,labels,test_size,train_atoms,test_atoms):
		img_train,img_test,y_train,y_test = [],[],[],[]
		for i in range(len(labels) -1,-1,-1):
			l = labels[i][0]
			atom = l[0] + '('
			if len(l) == 2:
				atom = l[0] + str((int(l[1]))) 
			else:
				atom = l[0] + str((int(l[1]),int(l[2])))


			atom = atom.replace(' ','')
			if atom in self.false_atoms:
				img_train.append(images[i])
				y_train.append(0)
				images.pop(i)
				labels.pop(i)
				continue

			if labels[i][0] in train_atoms:
				img_train.append(images[i])
				y_train.append(1)
				images.pop(i)
				labels.pop(i)
				continue

			if labels[i][0] in test_atoms:
				img_test.append(images[i])
				y_test.append(1)
				images.pop(i)
				labels.pop(i)
				continue
			labels[i] = 0

		shuffle(images)
		shuffle(labels)

		print('after filter:',len(y_train) + len(y_test))
		lim = int(len(labels)*test_size)
		img_test += images[:len(y_test)]
		y_test += labels[:len(y_test)]
		img_train += images[lim:lim + len(y_train)]
		y_train += labels[lim:lim + len(y_train)]
		c = sum([1 for y in y_train if y == 1])
		print('nratio',c,len(y_train))
		x1, x2, y1, y2 = train_test_split(img_train, y_train, test_size=0.33, random_state=42,shuffle=True)
		img_train = x1 + x2
		y_train = y1 + y2
		il = [img_train,img_test, y_train,y_test]
		return il


	def get_images_labels(self,sentences,pdm,pred_atoms,dom_sizes_map,train_atoms,test_atoms):
		TOPN = 30
		model = Word2Vec(sentences, size=TOPN,window=1,min_count=0)
		images,random_images,labels = [],[],[]
		pos_atoms = {}
		evids = self.get_evid_objs(pdm,pred_atoms)
		vocab = list(model.wv.vocab)
		l = len(model.wv.vocab)
		for p in pdm:
			print(p)
			if p not in model.wv.vocab:
				continue
			pvec = model.wv[p]
			key = (pdm[p][0])
			if len(pdm[p]) == 2:
				key = (pdm[p][0],pdm[p][0])

			if key not in pos_atoms:
				pos_atoms[key] = self.possible_atoms(pdm[p],dom_sizes_map)
			atom_objs = pos_atoms[key]
			for atom_obj in atom_objs:
				t = (p,)
				o1 = atom_obj[0]
				if o1 not in model.wv.vocab:
					continue
				if len(atom_obj) == 1:
					image = []
					rand_image = []
					vecs = model.most_similar(o1,topn = TOPN)
					for obj in vecs:
						dp = model.wv[obj[0]] + pvec
						ri = random.randint(0,l-1)
						w = vocab[ri]
						rp = model.wv[w] + pvec
						rand_image.append(rp)
						image.append(dp)
					images.append(image)
					random_images.append(rand_image)

					t += (o1.split('_')[1],)

					if o1 in evids[p]:
						labels.append([t,1])
					else:
						labels.append([t,0])
				else:
					if atom_obj[1] not in model.wv.vocab:
						continue
					o2 = atom_obj[1]
					image = []
					rand_image = []
					vecs1 = model.most_similar(o1,topn = TOPN)
					vecs2 = model.most_similar(o2,topn = TOPN)
					for i in range(len(vecs1)):
						dt_prod = model.wv[vecs1[i][0]]+pvec
						dt_prod = model.wv[vecs2[i][0]]+dt_prod

						ri = random.randint(0,l-1)
						w1 = vocab[ri]
						ri = random.randint(0,l-1)
						w2 = vocab[ri]
						rp = model.wv[w1] + pvec
						rp = model.wv[w2] + rp
						rand_image.append(rp)
						image.append(dt_prod)

					images.append(image)
					random_images.append(rand_image)

					t += (atom_obj[0].split('_')[1],atom_obj[1].split('_')[1],)
					if (atom_obj[0],atom_obj[1]) in evids[p]:
						labels.append([t,1])
					else:
						labels.append([t,0])
		

		return images,random_images,labels

	def get_false_atoms(self,ds):
		ifile = open(ds + '/db/orig_db.txt')
		atoms = {}

		for l in ifile:
			l = l.strip()
			if len(l) == 0:
				continue
			if l[0] == '!':
				l = l[1:]
				atoms[l] = True
		return atoms

	def get_evid_objs(self,pdm,pred_atoms):
		evids = {}
		for p in pred_atoms:
			evids[p] = {}
			atoms = pred_atoms[p]
			if len(atoms) == 0:
				continue
			if len(atoms[0]) == 1:
				for atom in atoms:
					atom[0] = int(atom[0])
					a = p + str(tuple(atom))
					a = a.replace(' ','')
					if a not in self.false_atoms: 
						obj = str(pdm[p][0]) + '_' + str(atom[0])
						evids[p][obj] = 1
			else:
				for atom in atoms:
					atom[0] = int(atom[0])
					atom[1] = int(atom[1])
					a = p + str(tuple(atom))
					a = a.replace(' ','')
					#print(a,self.false_atoms)
					if a not in self.false_atoms: 
						obj1 = str(pdm[p][0]) + '_' + str(atom[0])
						obj2 = str(pdm[p][1]) + '_' + str(atom[1])
						evids[p][obj1,obj2] = 1
		c = sum([len(pred_atoms[p]) for p in pred_atoms])
		e = sum([len(evids[p]) for p in evids])
		print('-------',c,e)
		return evids

	def possible_atoms(self,pred_doms,dom_sizes_map):
		if len(pred_doms) == 1:
			d = pred_doms[0]
			objs = []
			s = int(dom_sizes_map[d])
			for i in range(s):
				objs += [[str(d) + '_' +str(i)]]
			return objs

		d1,d2 = pred_doms[0],pred_doms[1]
		s1,s2 = int(dom_sizes_map[d1]),int(dom_sizes_map[d2])
		objs = []
		d1,d2 = str(d1),str(d2)
		for i in range(s1):
			for j in range(s2):
				objs.append([d1 + '_' + str(i),d2 + '_' + str(j)])
		return objs



