from gensim.models import Word2Vec
from keras.datasets import mnist
import numpy as np
import os
import pickle
import klepto

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
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
from keras import layers
import random
from sklearn.neural_network import MLPClassifier
import random

class CNN:
	def __init__(self):
		pass

	def run_NN(self,il):
		X_train,X_test, y_train,y_test = il[0],il[1],il[2],il[3]
		X_train = np.asarray([img[0] for img in X_train])
		X_test = np.asarray([img[0] for img in X_test])
		y_train = [y[2] for y in y_train]
		y_test = [y[2] for y in y_test]
		y_train = np.asarray(y_train)
		y_test = np.asarray(y_test)
		#print(y_train)
		clf = MLPClassifier(hidden_layer_sizes=(50,50,10), max_iter=100000)
		clf.fit(X_train, y_train)
		pred = y_pred = clf.predict(X_test)
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
		print('nn ', fscore)
		return fscore


	def plot_roc(self,y_test,pred):
		pred = pred[:,1]
		auc = roc_auc_score(y_test, pred)
		print('AUC: %.3f' % auc)
		return
		fpr, tpr, thresholds = roc_curve(y_test, pred,)
		pyplot.plot([0, 1], [0, 1], linestyle='--',label='ROC curve (area = %0.2f)' % auc)
		pyplot.plot(fpr, tpr, marker='.',label='ROC curve (area = %0.2f)' % auc)
		pyplot.show()

	def balance(self,imgs,y):
		n,mcount = len(y),sum(y)
		c = 0
		nimgs,ny = [],[]
		for i in range(n):
			if y[i] == 0 and c >= mcount:
				continue
			if y[i] == 0:
				c += 1

			nimgs.append(imgs[i])
			ny.append(y[i])
		return nimgs,ny

	def run(self,il):
		pfile = 'pickle/pred.p'
		if os.path.exists(pfile):
			print('sdfs')
			pil = pickle.load(open( pfile, "rb" ) )
			self.show_performance(pil)
			return

		pfile = 'pickle/il.p'
		if os.path.exists(pfile):
			il = pickle.load(open( pfile, "rb" ) )

		#self.run_NN(il)

		img_train,img_test, y_train,y_test = il[0],il[1],il[2],il[3]
		orig_y_train,orig_y_test = y_train[:], y_test[:]
		y_train = [v[2] for v in y_train]
		y_test = [v[2] for v in y_test]

		img_train,y_train = self.balance(img_train,y_train)
		img_test,y_test = self.balance(img_test,y_test)

		y1 = np.array(y_test[:])
		c = sum([1 for y in y_train if y == 1])
		print('ratio',c,len(y_train))
		m = len(img_train[0])

		y_train = np.array(y_train)
		
		y_test = np.array(y_test)
		for i,img in enumerate(img_train):
			img_train[i] = array(img)
		img_train = array(img_train)

		for i,img in enumerate(img_test):
			img_test[i] = array(img)
		img_test = array(img_test)



		img_train = img_train.reshape((len(img_train),m,m,1))
		img_test = img_test.reshape((len(img_test),m,m,1))

		y_train = to_categorical(y_train)
		y_test = to_categorical(y_test)
		#print(y_train[0])
		c = sum([1 for y in y_test if y[0] == 1])
		#print('ratio',c,len(y_test))
		CVN = 35
		MPN = 3
		print('------------------',m)
		model = Sequential()
		model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding= 'same' ,input_shape=(m,m,1)))
		model.add(layers.MaxPooling2D((MPN, MPN),2))
		model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding= 'same'))
		model.add(layers.MaxPooling2D((MPN, MPN),2))
		model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding= 'same'))
		model.add(layers.MaxPooling2D((MPN, MPN)))		
		#model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding= 'same'))
		#model.add(layers.MaxPooling2D((MPN, MPN)))	
		model.add(Flatten())
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(img_train, y_train, epochs=100)
		pred = model.predict(img_test)
		self.plot_roc(y1,pred)
		return
		#pred = np.array([p[0] for p in pred])
		best_th,max_fscore = None,0
		for th in range(5,1000,5):
			tp,fp,fn,tn = 0,0,0,0
			nth = th*1.0/1000.0
			print(nth)
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

			p = tp/(tp+fp)
			r = tp/(tp+fn)
			fscore = 2*p* r/(p+r)
			if fscore > max_fscore:
				max_fscore = fscore
				best_th = nth
		print(max_fscore)

		pfile = 'pickle/pred.p'
		il = [y1,pred,orig_y_train,orig_y_test,best_th]
		#pickle.dump(il,open(pfile,"wb"))
		self.show_performance(il)


	def show_performance(self,il):
		print('fsf')
		y1,pred,orig_y_train,orig_y_test,best_th = il[0],il[1],il[2],il[3],il[4]
		self.plot_roc(y1,pred)
		return
		print('best th :',best_th)
		pred_img = [[0 for j in range(368)] for i in range(272)]
		for pixel in orig_y_train:
			i,j = pixel[0],pixel[1]
			#pred_img[i][j] = 0 if pixel[2] == 0 else 255 

		c = 0
		for i in range(len(pred)):
			pixel = orig_y_test[i]
			i,j = pixel[0],pixel[1]
			if pred[i][1] >= 0.6:
				pred_img[i][j] = 255
				c += 1
		print('cccccccc : ',c)



		for i in range(200):
			l = [y[2] for y in orig_y_test if y[0] == i]
			#print(i,l,)
		pred_img = np.asarray(pred_img)
		cv2.imwrite('nimg.jpg',pred_img)


		best_th,max_fscore = None,0
		for th in range(5,1000,10):
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
			#print(tp,fp,tp+fn,fn,tn,fp+tn)
			p = tp/(tp+fp)
			r = tp/(tp+fn)
			fscore = 2*p* r/(p+r)
			if fscore > max_fscore:
				max_fscore = fscore
				best_th = nth
			#print('Threashold : ',nth)
			#print('Precision : ',p)
			#print('Recall : ',r)
			#print('F score : ',fscore)
		print(max_fscore)

class word2vec_cnn:
	def __init__(self):
		pass

	def make_images(self,sentences,pdm,pred_atoms,dom_sizes_map,ds,train_atoms,test_atoms,test_size):
		self.false_atoms = self.get_false_atoms(ds)
		pfile = ds + '/pickle/image_labels.p'
		if os.path.exists(pfile):
			il = pickle.load(open( pfile, "rb" ) )
			cnn = CNN()
			return cnn.run(il)
			return

		c = sum([len(pred_atoms[p]) for p in pred_atoms])
		print('total atoms : ',c)
		images,labels = self.get_images_labels(sentences,pdm,pred_atoms,dom_sizes_map,train_atoms,test_atoms)

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
		pickle.dump(il,open(pfile,"wb"))





	def get_images_labels(self,sentences,pdm,pred_atoms,dom_sizes_map,train_atoms,test_atoms):
		TOPN = 25
		model = Word2Vec(sentences, size=TOPN,window=1,min_count=0)
		images,labels = [],[]
		pos_atoms = {}
		evids = self.get_evid_objs(pdm,pred_atoms)
		vocab = list(model.wv.vocab)
		l = len(model.wv.vocab) - 1
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
					vecs = model.most_similar(o1,topn = TOPN)
					for obj in vecs:
						ri = random.randint(0,l)
						dp = model.wv[vocab[ri]]+pvec
						image.append(dp)
					images.append(image)

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
					vecs1 = model.most_similar(o1,topn = TOPN)
					vecs2 = model.most_similar(o2,topn = TOPN)
					for i in range(len(vecs1)):
						ri = random.randint(0,l)
						w1 = vocab[ri]
						ri = random.randint(0,l)
						w2 = vocab[ri]
						dt_prod = model.wv[w1]+pvec
						dt_prod = model.wv[w2]+dt_prod
						image.append(dt_prod)

					images.append(image)
					t += (atom_obj[0].split('_')[1],atom_obj[1].split('_')[1],)
					if (atom_obj[0],atom_obj[1]) in evids[p]:
						labels.append([t,1])
					else:
						labels.append([t,0])
		

		return images,labels

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

#CNN().run()
