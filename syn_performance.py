from DBManager import dbconfig
from common_f import common_f
class syn_performance:
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)


	def compare_marginal(self,mln,orig_meta_map,cr):
		orig_db_meta_map = {}
		for orig_obj in orig_meta_map['w2v']:
			orig_db_meta_map[orig_obj] = orig_obj
		orig_meta_map['orig'] = orig_db_meta_map

		ofile = open(self.performance_file,'a')
		ofile.write('marginal result => cr : {} \n'.format(cr))
		result = {}
		self.dbtypes = ['orig','w2v','kmeans','mar']
		for inf_type in ['marginal']:
			for t in self.dbtypes:
				if (inf_type,t) not in result:
					result[inf_type,t] = {}
				ifilename = self.dsname + '/'+ inf_type +'/' + t + '_out.txt'
				ifile = open(ifilename)
				for l in ifile:
					prob = float(l.split('\t')[0])
					atom = l.split('\t')[1].strip()
					pname = atom.split('(')[0]
					objs = atom.split('(')[1].split(')')[0].split(',')
					if pname not in mln.pdm:
						continue
					obj1 = 'd' + str(mln.pdm[pname][0]) + '_' + objs[0].strip()
					obj2 = -1
					if len(objs) == 2:
						obj2 = 'd' + str(mln.pdm[pname][1]) + '_' + objs[1].strip()
					result[inf_type,t][pname,obj1,obj2] = prob

		o = 'mar'
		for inf_type in ['marginal']:		
			for w in ['orig','w2v','kmeans']:
				t,c,diff,avg_prob,n = 0,0,0.0,0.0,0
				for (pname,obj1,obj2) in result[inf_type,o]:
					c += 1
					if obj1 not in orig_meta_map[w]:
						diff += result[inf_type,o][pname,obj1,obj2]
						avg_prob += result[inf_type,o][pname,obj1,obj2]
						continue
					w_obj1 = orig_meta_map[w][obj1]
					w_obj2 = -1
					if len(mln.pdm[pname]) == 2:
						if obj2 not in orig_meta_map[w]:
							diff += result[inf_type,o][pname,obj1,obj2]
							avg_prob += result[inf_type,o][pname,obj1,obj2]
							continue
						w_obj2 = orig_meta_map[w][obj2]
					orig_prob = result[inf_type,o][pname,obj1,obj2]
					if (pname,w_obj1,w_obj2) in result[inf_type,w]:
						w2v_prob = result[inf_type,w][pname,w_obj1,w_obj2]
						diff += abs(orig_prob - w2v_prob)
						avg_prob += orig_prob
						n += 1
					t += 1
				#error = diff*100/avg_prob
				#print(error,w,c)
				#ofile.write(w + ' error : ' + str(error) + '\n')
		ofile.close()