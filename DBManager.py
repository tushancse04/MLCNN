import os
import pickle
import os.path



class dbconfig:
	def __init__(self,dsname):
		self.MIN_DOM_SIZE = 100
		self.dsname = dsname
		self.dblocation = dsname + '/db/'
		self.db_init_name = 'db'
		self.merge_db_file = self.dblocation + 'orig_' + self.db_init_name + '.txt'
		self.orig_db_file = self.dblocation + 'orig_' + self.db_init_name + '.txt'
		self.w2v__cluster_db_file = self.dblocation  + 'w2v_db.txt'
		self.random__cluster_db_file = self.dblocation  + 'random_db.txt'
		self.kmeans__cluster_db_file = self.dblocation  + 'kmeans_db.txt'
		self.bmf__cluster_db_file = self.dblocation  + 'bmf_db.txt'
		self.mlnlocation = dsname + '/mln/'
		self.qryfile = dsname + '/query/qry.txt'
		self.output_location = dsname + '/out/'
		self.mln_file_name ='mln.txt'
		self.mln_tuffy_full_name = self.mlnlocation + 'mln_tuffy.txt'
		self.mln_file_full_name = self.mlnlocation + self.mln_file_name 
		self.new_mln_file_full_name = self.mlnlocation + 'mln.txt' 
		self.pickle_location = dsname + '/pickle/'
		self.pdom_obj_map = self.pickle_location + 'dom_obj_map.txt'
		self.performance_file = dsname + '/out/perform.txt'
		self.TOPN = 20
		self.MIN_FREQ = 14
		self.MAGICIAN_ITER = 100000
		self.TEST_SIZE = .2

		self.dbtypes = ['orig','bmf','w2v','random','kmeans']
		self.marginal_inf_location = dsname + '/marginal/'
		self.map_inf_location = dsname + '/map/'
		self.mar_magician_inf_location = dsname + '/mar_magician/'
		self.map_magician_inf_location = dsname + '/map_magician/'
		self.MINF = 0

		self.syn_marginal_file = self.marginal_inf_location + 'mar_out.txt'



class DBManager(dbconfig):
	def __init__(self,dsname,mln):
		dbconfig.__init__(self,dsname)
		self.mln = mln

	def set_atoms(self):
		pred_atoms_file = self.pickle_location + 'pred_atoms.p'
		if os.path.exists(pred_atoms_file):
			self.pred_atoms = pickle.load(open( pred_atoms_file, "rb" ) )
			return

		pred_atoms = {}
		ifile = open(self.orig_db_file)
		for l in ifile:
			pred_name = l.split('(')[0].replace('!','').strip()
			if pred_name not in pred_atoms:
				pred_atoms[pred_name] = []
			objs = l.split('(')[1].split(')')[0].split(',')
			pred_atoms[pred_name] += [objs]
		self.pred_atoms = pred_atoms
		pickle.dump(pred_atoms,open(self.pickle_location + "pred_atoms.p","wb"))




	def merge(self):
		files = os.listdir(self.dblocation)
		ofile = open(self.merge_db_file,'w')
		atoms = []
		atoms_dic = {}
		atom_lines = ''
		for f in files:
			if f.startswith(self.db_init_name):
				f = self.dblocation + f
				ifile = open(f)
				for l in ifile:
					if l not in atoms_dic:
						atoms += [l]
						atoms_dic[l] = 1
						ofile.write(l)
				ifile.close()
		ofile.close()

	def set_doms_atoms(self,mln,dbfile_name):
		if os.path.exists(self.pdom_obj_map):
			self.pred_atoms,self.dom_objs_map = pickle.load(open( self.pdom_obj_map, "rb" ) )
		ifile = open(dbfile_name)
		dom_objs_map = {}
		dom_counter = {}
		pred_atoms = {}
		old_new_map = {}
		for l in ifile:
			l = l.strip()
			p = l.split('(')[0]
			if p not in mln.pdm:
				continue
			if p not in pred_atoms:
				pred_atoms[p] = []

			objs = l.split('(')[1].split(')')[0].split(',')
			for i,obj in enumerate(objs):
				d = mln.pdm[p][i]
				if d not in dom_objs_map:
					dom_objs_map[d] = []
				if d not in dom_counter:
					dom_counter[d] = 0
				dobj = str(d) + '_' + obj
				if dobj not in old_new_map:
					dom_counter[d] += 1
					old_new_map[dobj] = str(dom_counter[d])
					dom_objs_map[d].append(old_new_map[dobj])
				objs[i] = old_new_map[dobj]

			pred_atoms[p].append(objs)
		self.pred_atoms = pred_atoms
		self.dom_objs_map = dom_objs_map
		pickle.dump((pred_atoms,dom_objs_map),open(self.pdom_obj_map,"wb"))


	def compress(self,mln,cr):
		ifile = open(self.merge_db_file)
		ofile_name = self.dblocation + 'orig_db.txt'
		ofile = open(ofile_name,'w')
		dom_objs_map = {}
		for l in ifile:
			l = l.strip()
			pred_name = l.split('(')[0]
			pname = pred_name[1:] if pred_name[0] == '!' else pred_name
			if pname not in mln.pdm:
				continue

			objs = l.split('(')[1].split(')')[0].split(',')
			
			size_exceeded = False

			for i,obj in enumerate(objs):
				dom = mln.pdm[pname][i]
				dom_size = float(mln.dom_sizes_map[dom])
				if dom not in dom_objs_map:
					dom_objs_map[dom] = []
				cl = len(dom_objs_map[dom])
				if cl < (cr*dom_size):
					continue
				if cl < self.MIN_DOM_SIZE:
					continue
				if obj in dom_objs_map[dom]:
					continue
				size_exceeded = True

			if size_exceeded:
				continue

			for i,obj in enumerate(objs):
				dom = mln.pdm[pname][i]
				if obj not in dom_objs_map[dom]:
					dom_objs_map[dom] += [obj]

			ofile.write(l + '\n')
		ifile.close()
		ofile.close()


