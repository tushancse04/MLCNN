from DBManager import dbconfig
from common_f import common_f
import math
class performance:
	def __init__(self,dsname,embedding_size):
		dbconfig.__init__(self,dsname)
		self.embedding_size = embedding_size

	def compare_map(self,mln,orig_meta_map,cr):
		ofile = open(self.performance_file,'a')
		ofile.write('map result => cr : {} \n'.format(cr))
		ofile.write('embedding size: ' + str(self.embedding_size))
		for w in orig_meta_map:
			orig_meta_map[w][-1] = -1
		result = {}
		for t in self.dbtypes:
			if t not in result:
				result[t] = {}
			ifilename = self.dsname + '/map/tuffy_' + t + '_out.txt'
			ifile = open(ifilename)
			for l in ifile:
				atom = l.strip()
				l = l.split('(')
				pname = l[0]
				d1 = 'd' + str(mln.pdm[pname][0]) + '_'
				objs = l[1].split(')')[0].split(',')
				obj1,obj2 = d1 + objs[0].strip(),-1
				if len(objs) > 1:
					obj2 = 'd' + str(mln.pdm[pname][0]) + '_' + objs[1].strip()
				result[t][pname,obj1,obj2] = 1

		o = 'orig'
		for w in self.dbtypes:
			if w == 'orig':
				continue
			ofile.write('database : {} \n'.format(w))
			tp1,fp,fn = 0,0,0
			r = result[o]
			m = orig_meta_map[w]
			for (pname,obj1,obj2) in r:
				if (obj1 not in m) and (obj2 == -1):
					fn += 1
					continue
				if (obj1 not in m) or (obj2 not in m):
					fn += 1
					continue
				mobj1 = m[obj1]
				mobj2 = m[obj2]
				if (pname,mobj1,mobj2) in result[w]:
					tp1 += 1
				else:
					fn += 1
			recall = round(tp1/(tp1 + fn),2)
			ofile.write('Recall : {} \n'.format(recall))
			rw = result[w]
			ro = result[o]
			tp,fp,fn = 0,0,0

			cf = common_f()
			meta_orig_map = cf.get_meta_orig_map(orig_meta_map[w])

			#print(meta_orig_map)
			for (pname,obj1,obj2) in rw:
				if obj1 not in meta_orig_map:
					fp += 1
					continue
				#print(meta_orig_map[obj1])
				map_count = len(meta_orig_map[obj1])
				if obj2 != -1:
					if obj2 not in meta_orig_map:
						fp += map_count
						continue
					map_count = map_count*len(meta_orig_map[obj2])

				found = False
				c = 0
				for p1,o1,o2 in ro:
					if o2 == -1:
						if o1 in orig_meta_map[w]:
							if obj1 == orig_meta_map[w][o1]:
								c += 1
					else:
						if o1 in orig_meta_map[w] and  o2 in orig_meta_map[w]:
							if obj1== orig_meta_map[w][o1] and obj2 == orig_meta_map[w][o2]:
								c += 1
				if c >= map_count:
					tp += 1
				else:
					fp += 1
			if (tp1 + fp) > 0:
				precision1 = round(tp1/(tp1 + fp),2)
				if (precision1 + recall) > 0:
					f_score1 = 2*precision1*recall/(precision1 + recall)
					ofile.write('f_score1 : {} \n'.format(f_score1))
			if (tp + fp) > 0:
				precision2 = round(tp/(tp + fp),2)
				if (precision2 + recall) > 0:
					f_score2 = 2*precision2*recall/(precision2 + recall)
					ofile.write('Precision : {} \n'.format(precision2))
					ofile.write('f_score2 : {} \n'.format(f_score2))
		ofile.close()


	def compare_marginal(self,mln,orig_meta_map,cr):
		ofile = open(self.performance_file,'a')
		ofile.write('marginal result => cr : {} \n'.format(cr))
		for inf_type in ['magician']:
			ofile.write('Calculating performance for ' + inf_type)
			result = {}
			for t in self.dbtypes:
				if (inf_type,t) not in result:
					result[inf_type,t] = {}
				ifilename = self.dsname + '/marginal/'+ inf_type +'_' + t + '_out.txt'
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

			r,w,o = 'random','w2v','orig'		
			for w in self.dbtypes:
				if w == 'orig':
					continue
				KLD,x,y = 0,0,0
				t,c,diff,avg_prob,n = 0,0,0.0,0.0,0
				for (pname,obj1,obj2) in result[inf_type,o]:
					c += 1
					if obj1 not in orig_meta_map[w]:
						x = result[inf_type,o][pname,obj1,obj2]
						diff += x
						continue

					w_obj1 = orig_meta_map[w][obj1]
					w_obj2 = -1
					if len(mln.pdm[pname]) == 2:
						if obj2 not in orig_meta_map[w]:
							x = result[inf_type,o][pname,obj1,obj2]
							diff += x
							continue
						w_obj2 = orig_meta_map[w][obj2]
					orig_prob = result[inf_type,o][pname,obj1,obj2]
					x = orig_prob
					if (pname,w_obj1,w_obj2) in result[inf_type,w]:
						y = result[inf_type,w][pname,w_obj1,w_obj2]
						if x > 0:
							KLD += y*math.log(y/x)
						w2v_prob = y
						diff += abs(orig_prob - w2v_prob)
						avg_prob += orig_prob
						n += 1
					t += 1
				if avg_prob <= 0:
					continue
				error = diff*100/avg_prob
				print(error,w)
				ofile.write(w + ' error : ' + str(error) + '\n')
				if n > 0:
					ofile.write(w + ' KLD : ' + str(KLD/n) + '\n')
		ofile.close()
