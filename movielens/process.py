ifile = open('raw/genome-scores.csv')
ifile.readline()
mtags = []
d1,d2,d3 = {},{},{}
MAX_ID = 200
USER_ID = 500

for l in ifile:
	l = l.strip()
	l = l.split(',')
	rel = int(float(l[2])*5) + 1

	if int(l[0]) > MAX_ID:
		continue

	#print(l[0])
	if l[0] not in d1:
		d1[l[0]] = len(d1)
	if l[1] not in d2:
		d2[l[1]] = len(d2)
	for i in range(rel):
		mtags.append((d1[l[0]],d2[l[1]]))
ifile.close()
print('tag',len(d2))

ifile = open('raw/links.csv')
ifile.readline()
mimdb = []
imdbtmb = []
mtmb = []
d2,d3 = {},{}
for l in ifile:
	l = l.strip()
	l = l.split(',')
	if int(l[0]) > MAX_ID:
		continue
	if l[0] not in d1:
		d1[l[0]] = len(d1)
	if l[1] not in d2:
		d2[l[1]] = len(d2)

	if l[2] not in d3:
		d3[l[2]] = len(d3)

	mimdb.append((d1[l[0]],d2[l[1]]))
	imdbtmb.append((d2[l[1]],d3[l[2]]))
	mtmb.append((d1[l[0]],d3[l[2]]))

ifile.close()
print('imdb',len(d2) )
print('tmb',len(d3))

ifile = open('raw/links.csv')
ifile.readline()
mgen = []
gidx = {}
c = 0
for l in ifile:
	l = l.strip()
	l = l.split(',')
	l[2] = l[2].split('|')
	if int(l[0]) > MAX_ID:
		continue
	for g in l[2]:
		if g not in gidx:
			gidx[g] = c
			c += 1
		g = gidx[g]
		mgen.append((l[0],g))


ifile.close()
print('gen',len(gidx))

ifile = open('raw/ratings.csv')
ifile.readline()
muser_rate = []
d2,d3 = {},{}
for l in ifile:
	l = l.strip()
	l = l.split(',')
	l[2] = int(float(l[2]))


	if int(l[1]) > MAX_ID:
		continue
	if int(l[0]) > USER_ID:
		continue

	if l[1] not in d1:
		d1[l[1]] = len(d1)
	if l[0] not in d2:
		d2[l[0]] = len(d2)
	if l[2] not in d3:
		d3[l[2]] = len(d3)

	#print(l)
	muser_rate.append((l[1],l[0],l[2])) 

ifile.close()

print('user',len(d2))
print('rate',len(d3))

ifile = open('raw/tags.csv',encoding='utf-8')
ifile.readline()
muser = []
mcat = []
ucat = []
d2,d3 = {},{}
for l in ifile:
	l = l.strip()
	l = l.split(',')

	if int(l[1]) > MAX_ID:
		continue
	if int(l[0]) > USER_ID:
		continue

	if l[1] not in d1:
		d1[l[1]] = len(d1)
	if l[0] not in d2:
		d2[l[0]] = len(d2)
	if l[2] not in d3:
		d3[l[2]] = len(d3)
	muser.append((d1[l[1]],d2[l[0]]))
	mcat.append((d1[l[1]],d3[l[2]]))
	ucat.append((d2[l[0]],d3[l[2]]))

ifile.close()

print('movie',len(d1))
print('user',len(d2))
print('cat',len(d3))


ofile = open('db/orig_db.txt','w')
c = 0
for x in mtags:
	x = str(x).replace(' ','')
	ofile.write('movie_tag' + x  + '\n')
	c += 1


for x in mimdb:
	x = str(x).replace(' ','')
	ofile.write('movie_imdb' + x+ '\n')
	c += 1


for x in imdbtmb:
	x = str(x).replace(' ','')
	ofile.write('imdb_tmb' + x+ '\n')
	c += 1

for x in mtmb:
	x = str(x).replace(' ','')
	ofile.write('movie_tmb' + x+ '\n')
	c += 1

for x in mgen:
	x = str(x).replace(' ','')
	ofile.write('movie_gen' + x+ '\n')
	c += 1

for x in muser_rate:
	if x[2] == 1:
		ofile.write('movie_user_rate1(' + x[0] + ',' + x[1] + ')\n')
	elif x[2] == 2:
		ofile.write('movie_user_rate2(' + x[0] + ',' + x[1] + ')\n')
	elif x[2] == 3:
		ofile.write('movie_user_rate3(' + x[0] + ',' + x[1] + ')\n')
	elif x[2] == 4:
		ofile.write('movie_user_rate4(' + x[0] + ',' + x[1] + ')\n')
	elif x[2] == 5:
		ofile.write('movie_user_rate5(' + x[0] + ',' + x[1] + ')\n')
	c += 1



for x in ucat:
	x = str(x).replace(' ','')
	ofile.write('user_cat' + x+ '\n')
	c += 1

for x in mcat:
	x = str(x).replace(' ','')
	ofile.write('movie_cat' + x+ '\n')
	c += 1
print(c)
import shutil
shutil.copyfile('db/orig_db.txt','db/w2v_db.txt')