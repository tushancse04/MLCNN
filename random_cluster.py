from DBManager import dbconfig
import random
from common_f import common_f

class random_cluster(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)
	
	def generate_random_db(self,db,pred_atoms_reduced_numbers,mln,dom_objs_map):
		rand_pred_atoms = {}
		dom_size = 0
		dom_objs_taken = {}


		print('Generating random atoms!')
		for (pname,n) in pred_atoms_reduced_numbers:
			if pname not in rand_pred_atoms:
				rand_pred_atoms[pname] = []
			while True:
				nobj = len(rand_pred_atoms[pname])
				if nobj >= n:
					break
				if nobj >= len(db.pred_atoms[pname]):
					break
				atom = random.choice(db.pred_atoms[pname])
				if atom in rand_pred_atoms[pname]:
					continue
				exceeded = False
				for i,obj in enumerate(atom):
					dom = mln.pdm[pname][i]
					if dom not in dom_objs_taken:
						dom_objs_taken[dom] = []

					if len(dom_objs_map[dom]) <= len(dom_objs_taken[dom]):
						exceeded = True
						break

				for i,obj in enumerate(atom):
					dom = mln.pdm[pname][i]	
					if dom not in dom_objs_taken:
						dom_objs_taken[dom] = []
					dom_objs_taken[dom] += [obj]
				#print(len(dom_objs_map[dom]),len(dom_objs_taken[dom]))
				rand_pred_atoms[pname] += [atom]

		cluster_file = self.random__cluster_db_file
		cf = common_f()
		cf.write_atoms(cluster_file,rand_pred_atoms)

		print('randomly selecting clusters')
		orig_dom_obj_map = cf.get_dom_obj_map(mln,db.pred_atoms)
		random_dom_obj_map = cf.get_dom_obj_map(mln,rand_pred_atoms)
		rand_orig_meta_map = {}
		for dom in orig_dom_obj_map:
			for obj in orig_dom_obj_map[dom]:
				meta = random.choice(random_dom_obj_map[dom])
				dom_init = 'd' + str(dom) + '_' 
				rand_orig_meta_map[dom_init + obj] = dom_init + meta
		
		print('random clustering complete')
		self.rand_orig_meta_map = rand_orig_meta_map



		