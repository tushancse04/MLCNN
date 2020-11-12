import os
from DBManager import dbconfig
from common_f import common_f

class MLN(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)
		self.initialize()


	def create_magician_mln(self):
		mag_orig_map = {}
		pdm = self.pdm
		for t in self.dbtypes:
			domobjs = {}
			mag_db = ''
			mlnstr = ''
			ifile = open(self.dblocation + t + '_db.txt')
			for l in ifile:
				if len(l) < 2:
					continue
				l = l.split('(')
				pname = l[0].strip()
				mag_db += pname + '('
				objs = l[1].split(')')[0].split(',')
				for i,obj in enumerate(objs):
					domname = pdm[pname][i]
					if domname not in domobjs:
						domobjs[domname] = []
					if obj not in domobjs[domname]:
						domobjs[domname].append(obj)
					if i != 0:
						mag_db += ','
					idx = str(domobjs[domname].index(obj))
					if t not in mag_orig_map:
						mag_orig_map[t] = {}
					if domname not in mag_orig_map[t]:
						mag_orig_map[t][domname] = {}						
					mag_orig_map[t][domname][idx] = obj
					mag_db += idx
				mag_db += ')\n'
			ofilename = self.dblocation + 'mag_' + t + '_db.txt'
			ofile = open(ofilename,'w')
			ofile.write(mag_db)
			ofile.close()
			for pname in pdm:
				mlnstr += pname
				for dname in pdm[pname]:
					#print(domobjs)
					mlnstr += ':' + str(len(domobjs[dname]))
				mlnstr += ' '
			mlnstr += '\n'
			mlnstr += self.mln_body
			ofile = open(self.mlnlocation + 'mag_' + t + '_mln.txt','w')
			ofile.write(mlnstr)
			ofile.close()		
			ifile.close()
		self.mag_orig_map = mag_orig_map






	def initialize(self):
		ifile = open(self.mln_file_full_name)
		l = ifile.readline()
		l=l.strip()
		parts = l.split(' ')
		pdm = {}
		dom_sizes_map = {}
		dom_pred_map = {}
		size_dom_map ={}
		for p in parts:
			pred_domains = p.split(':')
			predname = pred_domains[0]
			ds = pred_domains[1:]
			if predname not in pdm:
				pdm[predname] = []
			for i,s in enumerate(ds):
				if s not in size_dom_map:
					size_dom_map[s] = len(size_dom_map)
				d = size_dom_map[s]
				if d not in dom_sizes_map:
					dom_sizes_map[d] = s					
				pdm[predname] += [d]
				if d not in dom_pred_map:
					dom_pred_map[d] = []
				dom_pred_map[d] += [[predname,i]]
		#print(size_dom_map)
		self.dom_pred_map = dom_pred_map
		self.pdm = pdm
		self.dom_sizes_map = dom_sizes_map
		self.size_dom_map = size_dom_map
		mln_body = ''
		for l in ifile:
			mln_body += l
		self.mln_body = mln_body
		ifile.close()
		return pdm
