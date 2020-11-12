from DBManager import dbconfig,DBManager
from word2vec import word2vec
from common_f import common_f
from mln import MLN
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(model,clusters):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for meta_obj in clusters:
    	for orig_obj in clusters[meta_obj]:
    		tag = meta_obj.split('_')[1] + '_' + orig_obj.split('_')[1]
    		tokens.append(model[orig_obj])
    		labels.append(tag)
    print(tokens)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=len(clusters))
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
    	x.append(value[0])
    	y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
    	plt.scatter(x[i],y[i])
    	plt.annotate(labels[i],xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig("foo.pdf")


dsname = 'review'
mln = MLN(dsname)
db = DBManager(dsname,mln)
cf = common_f()
#cf.delete_files(mln.pickle_location)
db.set_atoms()

print('original db dom sizes(after compression):')
orig_dom_objs_map = db.get_dom_objs_map(mln,mln.orig_db_file)
CLUSTER_MIN_SIZE = 3
w2v = word2vec(dsname,db,CLUSTER_MIN_SIZE)
print('w2v cluster dom sizes:')
w2v_dom_objs_map = db.get_dom_objs_map(mln,w2v.w2v__cluster_db_file)

clusters = {}
for orig_obj in w2v.w2v_orig_meta_map:
	meta_obj = w2v.w2v_orig_meta_map[orig_obj]
	if not meta_obj.startswith('d1'):
		continue
	if meta_obj not in clusters:
		clusters[meta_obj] = []
	if orig_obj not in clusters[meta_obj]:
		clusters[meta_obj] += [orig_obj]

for meta_obj in list(clusters):
	if len(clusters[meta_obj]) < 2:
		del clusters[meta_obj]
model = w2v.get_model()
tsne_plot(model,clusters)