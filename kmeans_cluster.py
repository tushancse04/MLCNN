import sys
sys.path.append('kmeans')
#import lift_approx as la
from DBManager import dbconfig
from common_f import common_f
import pickle
import os

class kmeans_cluster(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)

	def cluster(self,db,cr,pdm,w2v_dom_objs_map,dom_pred_map):
		cf = common_f()
		meta_file = self.pickle_location + 'kmeans_meta_atoms.p'
		map_file = self.pickle_location + 'kmeans_orig_meta_map.p'
		reduce_file = self.pickle_location + 'kmeans_pred_atoms_reduced_numbers.p'
		if os.path.exists(meta_file):
			self.kmeans_meta_atoms = pickle.load(open( meta_file, "rb" ) )
			self.kmeans_orig_meta_map = pickle.load(open( map_file, "rb" ) )
			self.pred_atoms_reduced_numbers = pickle.load(open( reduce_file, "rb" ) )
			print('Loaded meta sentences.')
			return
		meta_atoms,orig_meta_map,reduced_dom = la.do_clustering(db.pred_atoms,'-p', self.new_mln_file_full_name, self.orig_db_file,cr,pdm,w2v_dom_objs_map,dom_pred_map)
		self.kmeans_meta_atoms = meta_atoms
		self.kmeans_orig_meta_map = orig_meta_map


		self.pred_atoms_reduced_numbers = reduced_dom
		cluster_evid_file = self.kmeans__cluster_db_file

		pickle.dump(self.kmeans_meta_atoms,open(meta_file,"wb"))
		pickle.dump(self.kmeans_orig_meta_map,open(map_file,"wb"))
		pickle.dump(self.pred_atoms_reduced_numbers,open(reduce_file,"wb"))

		cf.write_atoms(cluster_evid_file,self.kmeans_meta_atoms)