import os
import sys
#import matplotlibs
import npclustering as npc
import pclustering as pc
import numpy as np
from collections import defaultdict
import math
def do_clustering(atoms,p_np, mln_file, evid_file,cr,pdm,w2v_dom_objs_map,dom_pred_map):	
	if p_np == "-p":
		return pc.doclustering(atoms,mln_file,evid_file,cr,"kmeans/cmln.txt","kmeans/cevid.txt","kmeans/clog.dat",pdm,w2v_dom_objs_map,dom_pred_map)


