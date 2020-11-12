import pickle
entity_dict = pickle.load(open("pickle/entity_dict.p",'rb'))
relation_dict = pickle.load(open("pickle/relation_dict.p",'rb'))

files = ['raw/freebase_mtr100_mte100-test.txt','raw/freebase_mtr100_mte100-train.txt','raw/freebase_mtr100_mte100-valid.txt']
for f in files:
	ifile = open(f)
	for l in ifile:
		l = l.strip().split('	')
		if l[0] not in entity_dict or l[1] not in relation_dict or l[2] not in entity_dict:
			print(l)
			continue