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



l = []
th = .999999968218116
h,t = None,None
for (h,t,r) in triples:

	idx = en_idx_dic[h]
	hn = entity_norm[idx]

	idx = en_idx_dic[t]
	tn = entity_norm[idx]

	if r not in rel_idx_dic:
		continue

	idx = rel_idx_dic[r]
	rn = relation_norm[idx]
	dif = abs(hn+rn-tn)
	#if dif > th:

s1,s2 = [],[]


for r in rel_idx_dic:
	id1 = en_idx_dic[h]
	id2 = rel_idx_dic[r]
	id3 = en_idx_dic[t]
	diff = abs(entity_norm[id1] +relation_norm[id2] - entity_norm[id3])
	if (h,t,r) in triples:
		s1 += [diff]
	else:
		s2 += [diff]

print(len(s1),sum(s1)/len(s1),len(s2),sum(s2)/len(s2))
sys.exit()

triples_dict = {}
d1 = pickle.load(open("d1.p",'rb'))
d2 = pickle.load(open("d2.p",'rb'))

th1 = 1.23
th2 = 10
for (h,t,r) in triples:
	if (h,r) not in d1:
		continue

	if (t,r) not in d2:
		continue

	if (d1[h,r] <= th1) and (d2[t,r] <= th2):
		triples_dict[h,r,t] = 1
	elif (d1[h,r] <= th1) and (d2[t,r] > th2):
		triples_dict[h,r,t] = 2
	elif (d1[h,r] > th1) and (d2[t,r] <= th2):
		triples_dict[h,r,t] = 3
	else:
		triples_dict[h,r,t] = 4

pickle.dump(triples_dict, open( "triples_dict.p", "wb" ) )
print('printing type...')
n = len(triples_dict)
t1 = sum([1 for x in triples_dict if triples_dict[x] == 1])
t2 = sum([1 for x in triples_dict if triples_dict[x] == 2])
t3 = sum([1 for x in triples_dict if triples_dict[x] == 3])
t4 = sum([1 for x in triples_dict if triples_dict[x] == 4])
print(t1/n,t2/n,t3/n,t4/n)