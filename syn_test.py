from DBManager import DBManager,dbconfig
from mln import MLN
from word2vec import word2vec
from random_cluster import random_cluster
from tuffy import tuffy
from syn_performance import syn_performance
import sys
from common_f import common_f
from kmeans_cluster import kmeans_cluster
from shutil import copyfile


for CLUSTER_MIN_SIZE in range(4,8,2):
	for dsname in ['synthetic']:
		mln = MLN(dsname)
		db = DBManager(dsname,mln)
		cf = common_f()
		cf.delete_files(mln.pickle_location)
		#cf.remove_irrelevant_atoms()

		#db.merge()
		#sys.exit()
		#db.compress(mln,.1)
		db.set_atoms()

		print('original db dom sizes(after compression):')
		orig_dom_objs_map = db.get_dom_objs_map(mln,mln.orig_db_file)

		w2v = word2vec(dsname,db,CLUSTER_MIN_SIZE)
		print('w2v cluster dom sizes:')
		w2v_dom_objs_map = db.get_dom_objs_map(mln,w2v.w2v__cluster_db_file)
		cr = cf.calculate_cr(orig_dom_objs_map,w2v_dom_objs_map)
		dest_file = db.dblocation + 'w2v_' + str(cr) + '.txt'
		copyfile(db.w2v__cluster_db_file,dest_file)
		print('cr : ' + str(cr))




		kmc = kmeans_cluster(dsname)
		kmc.cluster(db,str(cr),mln.pdm,w2v_dom_objs_map,mln.dom_pred_map)
		print('kmeans cluster dom sizes:')
		kmeans_dom_objs_map = db.get_dom_objs_map(mln,kmc.kmeans__cluster_db_file)
		dest_file = db.dblocation + 'kmeans_' + str(cr) + '.txt'
		copyfile(db.kmeans__cluster_db_file,dest_file)


