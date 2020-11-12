import sys
#import lift_approx as la
from DBManager import dbconfig
from common_f import common_f
import pickle
import os
import numpy as np
import random
#import nimfa

class bmf_cluster(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)

	def cluster(self,db,cr,pdm,dom_obj_map):
		cf = common_f()
		atoms = db.pred_atoms
		newatoms = ''
		orig_meta_map = {}
		r = random.randint(3,30)
		ifile = open('time.txt','a')
		ifile.write('BMFFFFF' + str(r))
		ifile.close()
		for p in atoms:
			if len(pdm[p]) < 2:
				for a in atoms[p]:
					d_name = 'd' + str(pdm[p][0]) + '_' + str(a[0])
					orig_meta_map[d_name] = d_name 
					newatoms += p + '(' + str(a[0]) + ')\n'
				continue
			dom1 = pdm[p][0]
			dom2 = pdm[p][1]

			compress_dom1 = int(len(dom_obj_map[dom1])*cr)
			compress_dom2 = int(len(dom_obj_map[dom2])*cr)
			bmf_matrix = [[0 for j in range(compress_dom2)] for i in range(compress_dom1)]
			for i,atom in enumerate(atoms[p]):
				obj1 = int(atom[0])
				obj2 = int(atom[1])
				if (obj1 < compress_dom1) and (obj2 < compress_dom2):
					bmf_matrix[obj1][obj2] = 1
			bmf_matrix = np.array(bmf_matrix)
			bmf = nimfa.Bmf(bmf_matrix, seed="nndsvd", rank=r, max_iter=100, lambda_w=1.1, lambda_h=1.1)
			bmf_fit = None
			try:
				bmf_fit = bmf()
			except:
				print('error',r)
				self.cluster(db,cr,pdm,dom_obj_map)
				return
			W = bmf_fit.basis()
			H = bmf_fit.coef()
			T = np.dot(W,H)
			T = T.tolist()
			for i,x in enumerate(T):
				for j,y in enumerate(x):
					if T[i][j] > .5:
						T[i][j] = 1
					else:
						T[i][j] = 0
			bmf_matrix = bmf_matrix.tolist()
			for i,row in enumerate(T):
				for j,c in enumerate(row):
					d1_obj = 'd' + str(dom1) + '_' + str(i)
					d2_obj = 'd' + str(dom2) + '_' + str(j)
					orig_meta_map[d1_obj] = d1_obj
					orig_meta_map[d2_obj] = d2_obj
					if row[j] == 1:
						newatoms += p + '(' + str(i) + ',' + str(j) + ')\n'
		ofile_name = self.bmf__cluster_db_file
		ofile = open(ofile_name,'w')
		ofile.write(newatoms)
		ofile.close()
		self.bmf_orig_meta_map = orig_meta_map
