import pickle
import sys


il = pickle.load(open("pickle/il.p",'rb'))
triples_dict = pickle.load(open("pickle/triples_dict.p",'rb'))
d = pickle.load(open('pickle/d.p','rb'))
idx_obj_1,idx_obj_2 = d[1],d[3]

pred = il[-1]
y_test = il[3]
orig_atoms = il[4]

d = {}
for i,p in enumerate(pred):
	if y_test[i] == 0:
		continue

	o1,o2 = orig_atoms[i][0],orig_atoms[i][1]
	o1,o2 = int(o1.split('_')[1]),int(o2.split('_')[1])

	if o1 not in idx_obj_1 or o2 not in idx_obj_2:
		continue

	o1 = idx_obj_1[o1]
	o2 = idx_obj_2[o2]
	if o1[1] != o2[1]:
		continue

	h,r = o1[0],o1[1]
	if (h,r) not in d:
		d[h,r] = []

	d[h,r].append([o2[0],pred[i][1]])

d2 = {}
for (h,t,r) in triples_dict:
	if (h,r) not in d2:
		d2[h,r] = []
	d2[h,r] += [t]

#print(d2)
for k in d:
	d[k].sort(key = lambda x: x[1])
	v = d[k]
	k = (int(k[0]),int(k[1]))
	if k in d2:
		f = False
		for p in v:
			if v[0] in d2[k]:
				print('True')
				f = True
		if not f:
			print('False')

