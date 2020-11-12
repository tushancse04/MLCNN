from subprocess import call
import time
from DBManager import dbconfig
import os.path

class magician(dbconfig):
	def __init__(self,dsname,mln):
		dbconfig.__init__(self,dsname)
		self.mln = mln
		self.run_magician()

	def run_magician(self):
		pdm = self.mln.pdm
		mag_orig_map = self.mln.mag_orig_map
		ifile = open('magician_time.txt','a')
		for i in [0,1]:
			for t in self.dbtypes:
				start_time = time.time()
				out = self.marginal_inf_location + 'magician_' + t + "_out.txt"
				inf_type = 'mar'

				command = ['magician/Release/cdlearn',self.mlnlocation + 'mag_' + t + '_mln.txt']
				command += [self.dblocation + 'mag_' + t + '_db.txt',self.qryfile,str(self.MAGICIAN_ITER)]
				command += ['5','MAR',out,'the']
				ctext = ''
				for c in command:
					ctext += c + ' '
				print(ctext)
				call(command)
				while(True):
					if os.path.exists(out):
						break
					time.sleep(5)
				end_time = time.time()
				time_diff = (end_time - start_time)
				ifile.write(inf_type + ' ' + t + ' ' + str(time_diff) + '\n')
				mag_ofile = open(out)
				orig_out = ''
				for l in mag_ofile:
					l = l.strip()
					l = l.split(':')
					pname = l[0]
					objs = l[1:]
					prob = objs[-1]
					objs = objs[0:len(objs) -1]
					orig_out += prob + '\t'
					orig_out += pname + '('
					for i,obj in enumerate(objs):
						if i != 0:
							orig_out += ','
						dname = pdm[pname][i]
						orig_out += mag_orig_map[t][dname][obj]
					orig_out += ')\n'
				mag_ofile.close()
				ofile = open(out,'w')
				ofile.write(orig_out)
				ofile.close()

		ifile.close()
