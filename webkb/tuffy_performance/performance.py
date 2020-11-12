ifile = open('test.txt')
test_atoms = {}
for l in ifile:
	l = l.strip().split('(')
	pname = l[0]
	objs = l[1].strip().split(')')[0].split(',')
	atom = ()
	atom += (pname,)
	atom += tuple(objs)
	#print(atom)
	test_atoms[atom] = 1

ifile.close()

ifile = open('test_out_map.txt')
out_atoms = {}
for l in ifile:
	l = l.strip().split('(')
	pname = l[0]
	objs = l[1].strip().split(')')[0].split(',')
	atom = ()
	atom += (pname,)
	atom += tuple(objs)
	#print(atom)
	out_atoms[atom] = 1

tp,fp,fn,tn = 0,0,0,0

print(len(test_atoms))
for atom in out_atoms:
	if atom not in test_atoms:
		fp += 1
	else:
		tp += 1

for atom in test_atoms:
	if atom not in out_atoms:
		tn += 1
fn = 55078 - tp - fp - tn
print(tp,fp,fn,tn)


ifile.close()

	
