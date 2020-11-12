from DBManager import dbconfig
import os
import pickle

class pkl(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)


	def load(self,fname):
		pfile = self.pickle_location + fname
		if os.path.exists(pfile):
			return pickle.load(open( pfile, "rb" ))
		return None

	def store(self,obj,fname):
		pfile = self.pickle_location + fname
		pickle.dump(obj,open(pfile,"wb"))