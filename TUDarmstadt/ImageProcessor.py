from os import listdir
from os.path import isfile, join
from scipy import misc
import math
from gensim.models import Word2Vec
from cnn import CNN
from nn import nn
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pickle

X_REG = 10
Y_REG = 10
TOT_CAT = 2


class w2v_tud:
	def __init__(self):
		pass

	def get_images(self,sentences):
		TOPN = 20
		model = Word2Vec(sentences, size=TOPN,window=2,min_count=0)
		images = []
		for s in sentences:
			image = []
			for obj in s:
				obj_prob_pairs = model.most_similar(obj,topn = TOPN)
				for i,op in enumerate(obj_prob_pairs):
					if i < len(image):
						pvec = image[i]
						image[i] = model.wv[op[0]]+pvec
					else:
						image.append(model.wv[op[0]])
			images.append(image)
		return images


class ImageProcessor:
	def __init__(self):
		in_out_fmap = self.get_fmap()
		in_images,out_images = self.get_image_map(in_out_fmap)
		self.get_atoms(in_images,out_images)


	def get_atoms(self,in_images,out_images):
		ofile = open('orig_db.txt','w')
		for f in in_images:
			sentences,y = self.get_objs_by_image(in_images[f],out_images[f])
			wv = w2v_tud()
			images = wv.get_images(sentences)

			img_train, img_test, y_train, y_test = train_test_split(images, y, test_size=0.8, random_state=42,shuffle=True)

			il = [img_train, img_test, y_train, y_test]
			pfile = 'pickle/il.p'
			pickle.dump(il,open(pfile,"wb"))
			cn = CNN()
			cn.run(il)
			return

	def get_objs_by_image(self,image,out_img):
		sentences = []
		lr = len(image)
		lc = len(image[0])
		y = []
		for i,r in enumerate(image):
			for j,c in enumerate(r):
				reg_x = 'regx_' + str(math.floor(i*X_REG/lr))
				reg_y = 'regy_' + str(math.floor(i*Y_REG/lc))
				s = [reg_x,reg_y]
				for k in range(3):
					s.append(self.get_obj(image,i-4,j,k,'top_'))
					s.append(self.get_obj(image,i+4,j,k,'bot_'))
					s.append(self.get_obj(image,i,j-4,k,'left_'))
					s.append(self.get_obj(image,i,j+4,k,'right'))
					s.append(self.get_obj(image,i-4,j-4,k,'lt_'))
					s.append(self.get_obj(image,i-4,j+4,k,'rt_'))
					s.append(self.get_obj(image,i+4,j-4,k,'lb_'))
					s.append(self.get_obj(image,i+4,j+4,k,'rb_'))
				
				sentences.append(s)
				if out_img[i][j] < (255/2):
					y.append([i,j,0])
				else:
					y.append([i,j,1])
		print(len(sentences)*30)
		return sentences,y



	def get_obj(self,image,i,j,k,domname):
		in_top_k = domname + str(k) + '_'
		if i >= 0 and j >= 0 and i < len(image) and j < len(image[0]):
			in_top_k += str(int(image[i][j][k]*TOT_CAT/255))
		else:
			in_top_k += '-1'
		return in_top_k




	def get_image_map(self,in_out_fmap):
		in_images = {}
		out_images = {}
		for f in in_out_fmap:
			in_images[f] = cv2.imread(f,1)
			out_images[f] = cv2.imread(in_out_fmap[f],0)

		return in_images,out_images



	def get_fmap(self):
		inpath = 'PNGImages/sideviews-cows2/'
		infiles = [f for f in listdir(inpath) if isfile(join(inpath, f))]
		opath = 'GTMasks/sideviews-cows2/'
		outfiles = [f for f in listdir(opath) if isfile(join(opath, f))]
		in_out_fmap = {}
		for fname in infiles:
			ifile = inpath + fname
			for ofname in outfiles:
				if ofname == fname:
					in_out_fmap[ifile] = opath + ofname
		return in_out_fmap

ImageProcessor()