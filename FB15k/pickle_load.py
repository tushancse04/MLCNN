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

triples = []
for t in training_triples:
	triples.append(t)
for t in test_triples:
	triples.append(t)
for t in validation_triples:
	triples.append(t)

d1,d2 = {},{}
for t in triples:
	if (t[0],t[1]) not in d1:
		d1[t[0],t[1]] = 0
	d1[t[0],t[1]] += 1

	if (t[2],t[1]) not in d2:
		d2[t[2],t[1]] = 0
	d2[t[2],t[1]] += 1


print('cal avg numbers..')

print(len(d1))
c = 0
d3,d4 = {},{}
for (e,r) in d1:
	c += 1
	print(c)


	if r not in d3:
		l = [d1[e1,r1] for (e1,r1) in d1 if r1 == r]
		d3[r] = sum(l)/len(l)

	d1[e,r] = d3[r]

	if r not in d4:
		l = [d2[e1,r1] for (e1,r1) in d2 if r1 == r]
		d4[r] = sum(l)/len(l)

	d2[e,r] = d4[r]

print('assigning relations')
pickle.dump(d1, open( "d1.p", "wb" ) )
pickle.dump(d2, open( "d2.p", "wb" ) )
triples_dict = {}
th1 = 1.7
th2 = 1.2
for (h,r,t) in triples:
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