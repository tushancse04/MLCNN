from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2

class nn:
	def __init__(self):
		pass

	def run(self,il):
		pfile = 'pickle/il.p'
		if os.path.exists(pfile):
			il = pickle.load(open( pfile, "rb" ) )


		img_train,img_test, y_train,y_test = il[0],il[1],il[2],il[3]
		orig_y_train,orig_y_test = y_train[:], y_test[:]
		y_train = [v[2] for v in y_train]
		y_test = [v[2] for v in y_test]
		y_train = np.array(y_train)
		y_test = np.array(y_test)

		for i,x in enumerate(img_train):
			img_train[i] = x[0]
		for i,x in enumerate(img_test):
			img_test[i] = x[0]

		img_train = np.asarray(img_train)
		print(img_train[0])
		img_test = np.asarray(img_test)

		clf = MLPClassifier(hidden_layer_sizes=(50, 50,50,10), max_iter=100000)
		clf.fit(img_train,y_train)
		y_pred = clf.predict(img_test)
		tp,fp,fn,tn = 0,0,0,0
		for i,y in enumerate(y_pred):
			if y_test[i] == 1 and y == 1:
				tp += 1
			elif y_test[i] == 1 and y == 0:
				fn += 1
			elif y_test[i] == 0 and y == 1:
				fp += 1
			else:
				tn += 1
		if tp == 0:
			return
		p = tp/(tp+fp)
		r = tp/(tp+fn)
		fscore = 2*p*r/(p+r)
		print(tp,fp,fn,tn)
		print('fscore : ',fscore)

		pred_img = [[0 for j in range(368)] for i in range(272)]
		for pixel in orig_y_train:
			i,j = pixel[0],pixel[1]
			#pred_img[i][j] = 0 if pixel[2] == 0 else 255 

		c = 0
		for i in range(len(y_pred)):
			pixel = orig_y_test[i]
			i,j = pixel[0],pixel[1]
			if y_pred[i] == 1 and y_test[i] == 0:
				pred_img[i][j] = 255
				c += 1
		print('cccccccc : ',c)
		pred_img = np.asarray(pred_img)
		cv2.imwrite('nimg.jpg',pred_img)

#nn().run(None)