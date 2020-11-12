import os, shutil
class common_f():
	def __init__(self):
		pass
	def write_atoms(self,file_name,pred_atoms):
		ofile = open(file_name,'w')
		for predname in pred_atoms:
			for atom in pred_atoms[predname]:
				atomstr = predname + '('
				for i,obj in enumerate(atom):
					if i != 0:
						atomstr += ','
					atomstr += obj
				atomstr += ')\n'
				ofile.write(atomstr)
		ofile.close()

	def print_pred_atoms_size(self,Atoms):
		ats = []
		c = 0
		for p in Atoms:
		    for atom in Atoms[p]:
		        atom = [p] + atom
		        if atom not in ats:
		            c += 1
		            ats += [atom]
		        else:
		            print(atom)
		print(c)


	def get_dom_obj_map(self,mln,pred_atoms):
		dom_obj_map = {}
		for dom in mln.dom_pred_map:
			if dom not in dom_obj_map:
				dom_obj_map[dom] = []
			for pred_idx in mln.dom_pred_map[dom]:
				predname = pred_idx[0]
				obj_idx = pred_idx[1]
				if predname not in pred_atoms:
					continue
				for atom in pred_atoms[predname]:
					if atom[obj_idx] not in dom_obj_map:
						dom_obj_map[dom] += [atom[obj_idx]]
		return dom_obj_map

	def calculate_cr(self,orig_dom_obj_map,meta_dom_obj_map):
		ot = 0
		for dom in orig_dom_obj_map:
			ot += len(orig_dom_obj_map[dom])

		mt = 0
		for dom in meta_dom_obj_map:
			mt += len(meta_dom_obj_map[dom])
		print('printing mt,ot & cr : ')
		print(mt,ot)
		return round(mt*1.0/ot,2)
			
	def get_meta_orig_map(self,orig_meta_map):
		meta_orig_map= {}
		for orig_obj in orig_meta_map:
			meta_obj = orig_meta_map[orig_obj]
			if meta_obj not in meta_orig_map:
				meta_orig_map[meta_obj] = []
			meta_orig_map[meta_obj] += [orig_obj]
		return meta_orig_map

	def remove_irrelevant_atoms(self):
		for i in range(10):
			fname = 'er/db/db-'+str(i) + '.txt'
			ifile = open(fname)
			lines = ''
			for l in ifile:
				if l.startswith('Has'):
					continue
				lines += l
			ofile = open(fname,'w')
			ofile.write(lines)
			ofile.close()
			ifile.close()

	def delete_files(self,path):
		for the_file in os.listdir(path):
		    file_path = os.path.join(path, the_file)
		    try:
		        if os.path.isfile(file_path):
		            os.unlink(file_path)
		        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
		    except Exception as e:
		        print(e)