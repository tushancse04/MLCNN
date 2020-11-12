
files = ['raw/freebase_mtr100_mte100-test.txt','raw/freebase_mtr100_mte100-train.txt','raw/freebase_mtr100_mte100-valid.txt']

entities = {}
labels = {}

count_ent = 0
count_labels = 0
rels = {}

for f in files:
	ifile = open(f)
	for l in ifile:
		l = l.strip().split('	')
		if len(l) < 3:
			continue
		if l[1].strip().startswith('!'):
			continue
		l[0] = l[0].split('/')[-1]
		l[2] = l[2].split('/')[-1]
		h = l[0]
		l = l[1]
		t = l[2]


		if h not in entities:
			count_ent += 1
			entities[h] = count_ent

		if t not in entities:
			count_ent += 1
			entities[t] = count_ent

		if l not in labels:
			count_labels += 1
			labels[l] = count_labels

		h = entities[h]
		l = labels[l]
		t = entities[t]
		if (h,t) not in rels:
			rels[h,t] = []
		rels[h,t].append(l)



	ifile.close()

print(len(entities),len(labels))