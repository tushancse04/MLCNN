from DBManager import dbconfig
from mln import MLN
import pickle
import os
from gensim.models import Word2Vec
from collections import Counter
from common_f import common_f
from random import randint

class word2vec_mln(MLN):
	def __init__(self,dsname):
		MLN.__init__(self,dsname)
		self.set_pred_pairs()
		
		
	def set_pred_pairs(self):
		ifile = open(self.mln_file_full_name)
		ifile.readline()
		pred_pairs = []
		for l in ifile:
			l = l.strip()
			if len(l) == 0:
				continue
			l = l.split(':')[1]

			atom_vars = l.split('v')
			if len(atom_vars) <= 1:
				continue
			for i,av1 in enumerate(atom_vars):
				for j,av2 in enumerate(atom_vars[i+1:]):
					av1,av2 = av1.strip(),av2.strip()
					pred_name1 = av1.split('(')[0].replace('!','')
					vars1 = av1.split('(')[1].split(')')[0].split(',')
					pred_name2 = av2.split('(')[0].replace('!','')
					vars2 = av2.split('(')[1].split(')')[0].split(',')
					for a,v1 in enumerate(vars1):
						for b,v2 in enumerate(vars2):
							if v1 == v2:
								pred_pairs += [[[pred_name1,a],[pred_name2,b]]]

		self.pred_pairs = pred_pairs





class word2vec(word2vec_mln):
	def __init__(self,dsname,db,CLUSTER_MIN_SIZE,embedding_size):
		word2vec_mln.__init__(self,dsname)
		self.db = db
		self.embedding_size = embedding_size
		self.CLUSTER_MIN_SIZE = CLUSTER_MIN_SIZE
		self.set_word2vec_sentences()

		common_ws_file = self.pickle_location + 'common_ws.p'
		if os.path.exists(common_ws_file):
			self.common_ws = pickle.load(open( common_ws_file, "rb" ) )
		else:
			self.run_word2vec()

		self.replace_orig_atoms()



	def get_model(self):
		model_file = self.pickle_location + 'model.p'
		if os.path.exists(model_file):
			self.model = pickle.load(open( model_file, "rb" ) )
		else:
			model = Word2Vec(self.sentences, size=500,window=2,min_count=1)
			self.model = model
		return self.model

	def run_word2vec(self):
		self.MIN_FREQ = 15
		model_file = self.pickle_location + 'model.p'
		if os.path.exists(model_file):
			self.model = pickle.load(open( model_file, "rb" ) )
		else:
			print('no')
			model = Word2Vec(self.sentences, size=self.embedding_size,window=2,min_count=1)
			self.model = model
			pickle.dump(self.model,open(model_file,"wb"))
		print('starting eval')
		common_ws = []
		TOPN = randint(10,40)
		for w in self.model.wv.vocab:
			simwords_p = self.model.most_similar(w,topn=TOPN)
			simw = [d for (d,p) in simwords_p] + [w]
			#print(simw)
			c = 0
			for (d,p) in simwords_p:
				c += 1
				simwords_p = self.model.most_similar(d,topn=TOPN)
				simd = [d1 for (d1,p) in simwords_p] + [d]
				simw += simd

			common_w = []
			for x in simw:
				c = simw.count(x)
				if c >= self.MIN_FREQ and x not in common_w:
					common_w += [x]
			if len(common_w) == 0:
				continue
			doms = [x.split('_')[0] for x in common_w]
			most_dom = max(set(doms), key=doms.count)
			common_w = [x for x in common_w if x.split('_')[0] == most_dom]

			if len(common_w) == 1:
				continue
			if len(common_w) < self.CLUSTER_MIN_SIZE:
				continue
			common_ws += [common_w]
		pickle.dump(common_ws,open(self.pickle_location + 'common_ws.p',"wb"))
		self.common_ws = common_ws

	def replace_orig_atoms(self):
		w2v_orig_meta_map = {}
		w2v_meta_orig_map = {}
		w2v_meta_atoms = {}

		for p in self.db.pred_atoms:
			if p not in w2v_meta_atoms:
				w2v_meta_atoms[p] = []
			for atom in self.db.pred_atoms[p]:
				w2v_meta_atoms[p] += [atom[0:]]


		print('total common objs cluster : ' + str(len(self.common_ws)))



		for cw in self.common_ws:
			dom = int(cw[0].split('_')[0].replace('d',''))
			meta_obj = cw[0].split('_')[1]
			for i,w in enumerate(cw):
				ow = w.split('_')[1]
				if w in w2v_orig_meta_map:
					continue
				for pred_idx in self.dom_pred_map[dom]:
					predname = pred_idx[0]
					obj_idx = pred_idx[1]
					if predname not in w2v_meta_atoms:
						continue
					for atom in w2v_meta_atoms[predname]:
						if atom[obj_idx] == ow:
							atom[obj_idx] = meta_obj
							w2v_orig_meta_map[w] = cw[0]



		for p in w2v_meta_atoms:
			for atom in w2v_meta_atoms[p]:
				for i,obj in enumerate(atom):
					obj = 'd' + str(self.pdm[p][i]) + '_' + obj
					if obj not in w2v_orig_meta_map:
						w2v_orig_meta_map[obj] = obj

		print('w2v mapping done')
		self.w2v_orig_meta_map = w2v_orig_meta_map
		self.w2v_meta_atoms = {}
		for p in w2v_meta_atoms:
			self.w2v_meta_atoms[p] = []
			for atom in w2v_meta_atoms[p]:
				if atom not in self.w2v_meta_atoms[p]:
					self.w2v_meta_atoms[p] += [atom]

		print('distinct w2v mapping done')
		d = {}
		for p in self.w2v_meta_atoms:
			d[p] = len(self.w2v_meta_atoms[p])
		self.pred_atoms_reduced_numbers = [(k, d[k]) for k in sorted(d, key=d.get, reverse=False)]
		cluster_evid_file = self.w2v__cluster_db_file
		cf = common_f()
		cf.write_atoms(cluster_evid_file,self.w2v_meta_atoms)


			




	def set_word2vec_sentences(self):
		sentences_file = self.pickle_location + 'sentences.p'
		if os.path.exists(sentences_file):
			self.sentences = pickle.load(open( sentences_file, "rb" ) )
			print('Loaded w2v sentences.')
			return
		p_obj_dic = {}
		sentences = []
		for pred_pair in self.pred_pairs:
			pred1,com1 = pred_pair[0][0],pred_pair[0][1]
			pred2,com2 = pred_pair[1][0],pred_pair[1][1]
			if pred1 not in self.db.pred_atoms:
				continue
			if pred2 not in self.db.pred_atoms:
				continue
			objs1 = self.db.pred_atoms[pred1]
			objs2 = self.db.pred_atoms[pred2]
			if (pred1,com1) not in p_obj_dic:
				p_obj_dic[pred1,com1] = {}
				for obj in objs1:
					p_obj = p_obj_dic[pred1,com1]
					if obj[com1] not in p_obj:
						p_obj[obj[com1]] = []
					p_obj[obj[com1]] += [obj]

			if (pred2,com2) not in p_obj_dic:
				p_obj_dic[pred2,com2] = {}
				for obj in objs2:
					p_obj = p_obj_dic[pred2,com2]
					if obj[com2] not in p_obj:
						p_obj[obj[com2]] = []
					p_obj[obj[com2]] += [obj]

			for obj1 in p_obj_dic[pred1,com1]:
				atoms1 = p_obj_dic[pred1,com1][obj1]
				if obj1 not in p_obj_dic[pred2,com2]:
					continue
				atoms2 = p_obj_dic[pred2,com2][obj1]
				for a1 in atoms1:
					a = a1[0:]
					for i,obj in enumerate(a1):
						a[i] = 'd' + str(self.pdm[pred1][i]) + '_' + obj
					sentences += [a]
				for a2 in atoms2:
					a = a2[0:]
					for i,obj in enumerate(a2):
						a[i] = 'd' + str(self.pdm[pred2][i]) + '_' + obj
					sentences += [a]

		pickle.dump(sentences,open(sentences_file,"wb"))
		self.sentences = sentences
