import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import random
from PIL import Image
import pickle


def cal_fscore():
	pred,gt = pickle.load(open('result.p', "rb" ))
	mn = np.min(pred)
	pred = np.array(pred) - mn
	mx = np.max(pred)
	m,n = len(pred),len(pred[0])
	pred = [[0 if pred[i][j]/mx < .5 else 1 for j in range(n)] for i in range(m)]
	gt = [[0 if gt[i][j]/mx < .5 else 1 for j in range(n)] for i in range(m)]
	tp,fp,fn = 0,0,0
	#mcount = np.sum(gt)
	nc = 0
	for i in range(m):
		for j in range(n):
			p,g = pred[i][j],gt[i][j]
			#if nc > mcount and g == 0:
				#continue
			if g == 0:
				nc += 1

			if p == g and p == 0:
				tp += 1
			elif p == g:
				fn += 1
			elif g == 1:
				fp += 1
	print(tp,fp,fn)
	p = tp/(tp+fp)
	r = tp/(tp+fn)
	f = 2*p*r/(p+r)
	print(f)




def get_data():
	gt_path = 'GTMasks/sideviews-cows2/'
	inp_path = 'PNGImages/sideviews-cows2/'
	gts = os.listdir(gt_path)
	lmin,wmin = None,None
	inp_imgs,gt_imgs = [],[]
	for f in gts:
		fname = gt_path + f
		img = mpimg.imread(fname)
		m1,m2 = len(img),len(img[0])
		if not lmin:
			lmin,wmin = m1,m2
		if lmin > m1:
			lmin = m1
		if wmin > m2:
			wmin = m2
		gt_imgs.append(img.tolist())

		fname = inp_path + f
		img = mpimg.imread(fname)
		inp_imgs.append(img.tolist())


	print(wmin,lmin)
	lmin,wmin = 128,128
	for i in range(len(inp_imgs)):
		img = inp_imgs[i][:lmin]
		img = [img[j][:wmin] for j in range(lmin)]
		inp_imgs[i] = img

		img = gt_imgs[i][:lmin]
		img = [img[j][:wmin] for j in range(lmin)]
		gt_imgs[i] = img
		img = [[int(img[j][k]*255) for k in range(len(img[j]))] for j in range(len(img))]
		img = Image.fromarray(np.array(img)).convert('L')
		img.save('out/' + str(i) + '.jpg')

	n = len(gt_imgs)
	r =random.randint(0,n-1)
	x_test,y_test = [inp_imgs[r]],[gt_imgs[r]]
	x_train,y_train = inp_imgs[:r] + inp_imgs[r+1:],gt_imgs[:r] + gt_imgs[r+1:]
	x_train,y_train,x_test,y_test = np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
	y_train = np.reshape(y_train,(len(y_train),len(y_train[0]),len(y_train[0][0]),1))
	y_test = np.reshape(y_test,(len(y_test),len(y_test[0]),len(y_test[0][0]),1))
	print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
	return x_train,y_train,x_test,y_test


cal_fscore()