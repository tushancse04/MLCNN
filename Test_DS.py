from DBManager import DBManager,dbconfig
from mln import MLN
from word2vec import word2vec
from random_cluster import random_cluster
from tuffy import tuffy
from magician import magician
from performance import performance
import sys
from common_f import common_f
from kmeans_cluster import kmeans_cluster
from bmf_cluster import bmf_cluster
import time
import random
from sentence_generator import sentence_generator
from cnn import word2vec_cnn
import pickle
from numpy import array
from review_sentence_generator import review_sentence_generator
import time
from corrupt import corrupt
from pkl import pkl
from balancer import balancer

class Test_DS:
	def __init__(self):
		pass

	def merge(self):
		for dsname in ['protein']:
			mln = MLN(dsname)
			db = DBManager(dsname,mln)
			db.merge()

	def compress(self):
		for dsname in ['er','protein','webkb']:
			mln = MLN(dsname)
			db = DBManager(dsname,mln)
			db.compress(mln, .3)

	def embed(self,pred_atoms,pdm,dom_sizes_map,is_cnn):
		b = balancer()
		b.embed(pdm,pred_atoms)
		p_atom_map = b.get_bal_atoms(pred_atoms,pdm,dom_sizes_map)
		pred_atoms = b.run_cnn_ntn(p_atom_map,pdm,is_cnn)
		return pred_atoms

	def test(self):
		#self.merge()
		#self.compress()
		#return
		embedding_size = 100
		for CLUSTER_MIN_SIZE in range(4,19,2):
			for dsname in ['webkb']:
				mln = MLN(dsname)
				db = DBManager(dsname,mln)
				print('merge db dom sizes:')
				db.set_doms_atoms(mln,db.merge_db_file)

				cf = common_f()
				#cf.delete_files(mln.pickle_location)
				if dsname == 'er':
					cf.remove_irrelevant_atoms()
				
				embedding_size = 300
				print('generating sentences')
				start = time.time()
				cnn_atoms,ntn_atoms = db.pred_atoms,db.pred_atoms
				while True:
					#cnn_atoms = self.embed(cnn_atoms,mln.pdm,mln.dom_sizes_map,True)
					ntn_atoms = self.embed(ntn_atoms,mln.pdm,mln.dom_sizes_map,False)

				sg = None
				if dsname == 'review':

					return
					#end = time.time()
					#print('Time : ',end-start)
				else:
					sg = sentence_generator(mln.pdm,db.pred_atoms,db.TEST_SIZE,db)
					#print('calling w2v')
					#wv = word2vec_cnn()
					#print('making images')
					#wv.make_images(sg.sentences,mln.pdm,db.pred_atoms,mln.dom_sizes_map,dsname,sg.train_atoms,sg.test_atoms,db.TEST_SIZE)

			
				cor = corrupt(dsname,db.pred_atoms,mln.pdm,db.dom_objs_map,sg.sentences)
				return
				bmf = bmf_cluster(dsname)
				bmf.cluster(db,1,mln.pdm,dom_obj_map)

				print('original db dom sizes(after compression):')
				orig_dom_objs_map = db.get_dom_objs_map(mln,mln.orig_db_file)
				CLUSTER_MIN_SIZE = 10
				w2v = word2vec(dsname,db,CLUSTER_MIN_SIZE,embedding_size)
				print('w2v cluster dom sizes:')
				w2v_dom_objs_map = db.get_dom_objs_map(mln,w2v.w2v__cluster_db_file)
				cr = cf.calculate_cr(orig_dom_objs_map,w2v_dom_objs_map)


				print('cr : ' + str(cr))
				rc = random_cluster(dsname)
				rc.generate_random_db(db,w2v.pred_atoms_reduced_numbers,mln,w2v_dom_objs_map)
				print('random cluster dom sizes')
				db.get_dom_objs_map(mln,mln.random__cluster_db_file)




				kmc = kmeans_cluster(dsname)
				kmc.cluster(db,str(cr),mln.pdm,w2v_dom_objs_map,mln.dom_pred_map)
				print('kmeans cluster dom sizes:')
				kmeans_dom_objs_map = db.get_dom_objs_map(mln,kmc.kmeans__cluster_db_file)
				mln.create_magician_mln()
				magician(dsname,mln)
				#tuffy(dsname)
				orig_meta_map = {}

				orig_meta_map['bmf'] = bmf.bmf_orig_meta_map
				orig_meta_map['w2v'] = w2v.w2v_orig_meta_map
				orig_meta_map['random'] = rc.rand_orig_meta_map
				orig_meta_map['kmeans'] = kmc.kmeans_orig_meta_map
				print('Dataset : ' + dsname +  '; CR : ' + str(cr))
				p = performance(dsname,embedding_size)
				p.compare_marginal(mln,orig_meta_map,cr)
				#p.compare_map(mln,orig_meta_map,cr)


t = Test_DS()
t.test()