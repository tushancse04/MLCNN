from random import shuffle

class sentence_generator:

	def __init__(self,pdm,pred_atoms,test_size,db):
		self.gen_train_test_atoms(pred_atoms,test_size,db.qryfile)
		sentences = []
		for pname in pred_atoms:
			for objs in pred_atoms[pname]:
				s = []
				t = (pname,)
				for i,obj in enumerate(objs):
					 s += [str(pdm[pname][i]) + '_' + str(obj)]
				s += [pname]
				t += tuple(objs)
				#print(t)
				#if t in self.train_atoms:
					#sentences += [s]
				sentences += [s]
		print("sentences",len(sentences))
		self.sentences = sentences

	def gen_train_test_atoms(self,pred_atoms,test_size,qryfile):
		print('filtering by query preds')
		atoms = {}
		train_atoms = {}
		test_atoms = {}
		for p in pred_atoms:
			for atom in pred_atoms[p]:
				t = (p,)
				t += tuple(atom)
				atoms[t] = 1
		atoms = list(atoms.keys())
		qrypreds = []
		ifile = open(qryfile)
		for l in ifile:
			l = l.strip()
			qrypreds.append(l)
		natoms = []
		for atom in atoms:
			p = atom[0]
			if p in qrypreds:
				natoms.append(atom)
		atoms = natoms
		print('shuffling atoms..')
		shuffle(atoms)

		l = len(atoms)
		test_lim = int(l*test_size)
		for atom in atoms[:test_lim]:
			test_atoms[atom] = 1

		for atom in atoms[test_lim:]:
			train_atoms[atom] = 1

		self.test_atoms = test_atoms
		self.train_atoms = train_atoms


