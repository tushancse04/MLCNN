import pickle
import sys


entity_dict = pickle.load(open("pickle/entity_dict.p",'rb'))
entity_norm = pickle.load(open("pickle/entity_norm.p",'rb'))
relation_dict = pickle.load(open("pickle/relation_dict.p",'rb'))
relation_norm = pickle.load(open("pickle/relation_norm.p",'rb'))
test_triples = pickle.load(open("pickle/test_triples.p",'rb'))
training_triples = pickle.load(open("pickle/training_triples.p",'rb'))
validation_triples = pickle.load(open("pickle/validation_triples.p",'rb'))
triples_dict = pickle.load(open("pickle/triples_dict.p",'rb'))

entities = entity_dict.values()
en_idx_dic = {}
for i,v in enumerate(entities):
	en_idx_dic[v] = i

rels = relation_dict.values()
rel_idx_dic = {}
for i,v in enumerate(rels):
	rel_idx_dic[v] = i

triples = []
for t in training_triples:
	triples.append(t)
for t in test_triples:
	triples.append(t)
for t in validation_triples:
	triples.append(t)

ofile = open('db/orig_db.txt','w')
print(len(triples))
c = 0
th,tr = 14600,100
obj_idx_1,obj_idx_2 = {},{}
idx_obj_1,idx_obj_2 = {},{}
for (h,t,r) in triples:

	h,t,r = int(h),int(t),int(r)
	if h > th or t > th or r > tr:
		continue

	c += 1
	#print(c)		




	h,t,r =   str(h),str(t),str(r)
	print(h,r,t)
	o1 =   (h,r)
	o2 =   (t,r)
	if o1 not in obj_idx_1:
		n = len(obj_idx_1)
		obj_idx_1[o1] = n
		idx_obj_1[n] = o1

	if o2 not in obj_idx_2:
		n = len(obj_idx_2)
		obj_idx_2[o2] = n
		idx_obj_2[n] = o2

	o1,o2 = str(obj_idx_1[o1]),str(obj_idx_2[o2])

	ofile.write('JHJT(' + o1 + ',' + o2 + ')\n')

	ofile.write('JHH(' + o1 + ',' + h + ')\n')
	ofile.write('JHR(' + o1 + ',' + r + ')\n')
	ofile.write('HR(' + h + ',' + r + ')\n')

	ofile.write('JTT(' + o2 + ',' + t + ')\n')
	ofile.write('JTR(' + o2 + ',' + r + ')\n')
	ofile.write('TR(' + t + ',' + r + ')\n')

ofile.close()
d = [obj_idx_1,idx_obj_1,obj_idx_2,idx_obj_2]
pickle.dump(d,open('pickle/d.p',"wb"))


dsize = ''
l1,l2 = str(len(obj_idx_1)),str(len(obj_idx_2)+1)
dsize += 'JHJT' + ':' + l1 + ':' + l2
dsize += ' JHH' + ':' + l1 + ':' + str(th)
dsize += ' JHR' + ':' + l1 + ':' + str(tr)
dsize += ' HR' + ':' + str(th) + ':' + str(tr)

dsize += ' JTT' + ':' + l2 + ':' + str(th)
dsize += ' JTR' + ':' + l2 + ':' + str(tr)
dsize += ' TR' + ':' + str(th) + ':' + str(tr)
dsize += '\n'

dpreds = ''
dpreds += '.2:JHJT(x,y)\n'
dpreds += '.2:JHH(x,y)\n'
dpreds += '.2:JHR(x,y)\n'
dpreds += '.2:HR(x,y)\n'
dpreds += '.2:JTT(x,y)\n'
dpreds += '.2:JTR(x,y)\n'
dpreds += '.2:TR(x,y)'

ofile = open('mln/mln.txt','w')
ofile.write(dsize)
ofile.write(dpreds)
ofile.close()