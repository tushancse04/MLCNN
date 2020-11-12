from subprocess import call
import time
from DBManager import dbconfig
import os.path

class tuffy(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)
		self.run_tuffy()

	def run_tuffy(self):
		ifile = open('time.txt','a')
		for i in [0,1]:
			for t in self.dbtypes:
				start_time = time.time()
				out = self.map_inf_location + 'tuffy_' + t + "_out.txt"
				inf_type = 'map'
				if i == 1:
					inf_type = 'marginal'
					out = self.marginal_inf_location + 'tuffy_' + t + "_out.txt"
				command = ['java','-jar', 'tuffy.jar','-i',self.dsname + "/mln/mln_tuffy.txt"]
				if i == 1:
					command = ['java','-jar', 'tuffy.jar','-marginal','-i',self.dsname + "/mln/mln_tuffy.txt"]
				command += ['-e', self.dsname + "/db/" + t + "_db.txt",'-queryFile',self.dsname + "/query/qry.txt"]
				command += ['-r',out]
				call(command)
				while(True):
					if os.path.exists(out):
						break
					time.sleep(5)
				end_time = time.time()
				time_diff = (end_time - start_time)
				ifile.write(inf_type + ' ' + t + ' ' + str(time_diff) + '\n')
		ifile.close()