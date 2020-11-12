import os
import sys
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import time
import random

def calc_dist(p1,p2):
    res = 0.0
    for i in range(0,len(p1),1):
        res = res + ((p2[0] - p1[0]) ** 2)
    return math.sqrt(res ** 2)
    
def intersect(a,b):
    return list(set(a) & set(b))

def FindPredObjPos(pred_domain_dict,domain_idx):
    pairs = []
    for p in pred_domain_dict:
        for i,dom_idx in enumerate(pred_domain_dict[p]):
            if dom_idx == domain_idx:
                pairs += [[p,i]]
    return pairs

def generate_new_mln(pdm,dom_obj_map,old_mln_path,new_mln_path):
    ifile = open(old_mln_path)
    ifile.readline()
    ofile = open(new_mln_path,'w')

    l = ''
    for p in pdm:
        l += p
        for i,d in enumerate(pdm[p]):
            l += ':'
            l += str(len(dom_obj_map[d]) + 1)
        l += ' '
    ofile.write(l + '\n')
    for l in ifile:
        ofile.write(l)
    ofile.close()
    ifile.close()

def generate_new_mln_evid_file(old_evid_file,new_evid_file,old_mln,new_mln,pdm):
    ifile = open(old_evid_file)
    ofile = open(new_evid_file,'w')
    dom_map = {}
    for l in ifile:
        l = l.strip()
        l.replace('!','')
        l = l.split('(')
        pname = l[0]
        new_atom_str = pname + '('
        objs = l[1].split(')')[0].split(',')
        for i,obj in enumerate(objs):
            obj = obj.strip()
            dom = pdm[pname][i]
            if dom not in dom_map:
                dom_map[dom] = []
            if obj not in dom_map[dom]:
                dom_map[dom] += [obj]
            if i != 0:
                new_atom_str += ','
            new_atom_str += str(dom_map[dom].index(obj))
        new_atom_str += ')n'
        ofile.write(new_atom_str)
    ofile.close()
    ifile.close()
    generate_new_mln(pdm,dom_map,old_mln,new_mln)


def doclustering(atoms,mlnfile,evidfile,cratio,outmfile,outefile,outclog,pdm,w2v_dom_objs_map,dom_pred_map):

    new_evid_file = 'kmeans/new_evid.txt'
    new_mln_file = 'kmeans/new_mln.txt'
    generate_new_mln_evid_file(evidfile,new_evid_file,mlnfile,new_mln_file,pdm)
    evidfile = new_evid_file
    mlnfile = new_mln_file

    #exepath = "./Release/cdlearn"
    Atoms = {}
    for p in atoms:
        Atoms[p] = atoms[p][0:]







    basefile = os.path.basename(mlnfile)
    os.system('python kmeans/genfeatures.py '+mlnfile+" "+cratio+"\n")
    tmpspecfile  = "kmeans/tmp-"+basefile[:basefile.rfind(".")]+".spec"

    ifile=open(tmpspecfile)
    firstline = ifile.readline()
    numdoms = int(ifile.readline().strip())
    reducedsizes = []
    tables = []
    tableids = []
    for i in range(0,numdoms,1):
        domline = ifile.readline()
        domvals = domline.strip().split(":")
        reducedsizes.append(int(domvals[0]))
        featureids = domvals[1].split(",")
        namevec = []
        valvec = []
        for d in featureids:
            d1 = d.split("-")
            namevec.append(d1[0])
            valvec.append(int(d1[1][2:]))
        tables.append(namevec)
        tableids.append(valvec)
    #print(tables)
    #print(tableids)        
    #print(reducedsizes)

    parts = firstline.strip().split()
    prednames = []
    predtables = []
    for p in parts:
        p1 = p.split(":")
        prednames.append(p1[0])
        tmp1 = []
        for j in range(1,len(p1),1):
            sz = int(p1[j])
            tmp = []
            for k in range(0,sz,1):
                tmp.append(0)
            tmp1.append(tmp)
        predtables.append(tmp1)

    evidtable = []
    sevidtable = []
    for i,p in enumerate(parts):
        p1 = p.split(":")
        tmp2 = []
        stmp1 = []
        for j in range(1,len(p1),1):
            sz = int(p1[j])
            for k in range(0,sz,1):
                if len(p1)==2:
                    stmp1.append(-1)
                if j==1:
                    tmp3 = []
                    tmp2.append(tmp3)                
        evidtable.append(tmp2)
        sevidtable.append(stmp1)
    parts = ifile.readline().strip().split()
    pvec = []
    pindvec = []
    sizevec = []
    for p in parts:
        parts1 = p.split(":")
        predname = parts1[0]
        predindexes = []
        for p1 in parts1[1:]:
            p2 = p1.split(",")
            predindexes.append(int(p2[0]))
        pvec.append(predname)
        pindvec.append(predindexes)
        ix = prednames.index(predname)
        sizes = []
        for i in range(0,len(predtables[ix]),1):
            sz = 1
            for j in range(0,len(predtables[ix]),1):
                if i==j:
                    continue
                sz = sz*len(predtables[ix][j])
            sizes.append(sz)
        sizevec.append(sizes)
    #print(pvec)
    #print(pindvec)
    #print(sizevec)

    ifile.readline()
    otherlines=ifile.readlines()
    ifile.close()

    ifile = open(evidfile)
    for l in ifile:
        parts = l.strip().split("(")
        if parts[0][0]=='!':
            continue
        arglist = parts[1][:parts[1].find(")")].split(",")
        for i,a in enumerate(arglist):
            idx = int(a.strip())
            if idx==-1:
                continue
            tidx = -1
            try:
                tidx = prednames.index(parts[0])
            except ValueError:
                continue
            predtables[tidx][i][idx] = predtables[tidx][i][idx]+5
    ifile.close()
    #print(predtables)


    ifile = open(evidfile)
    evidlines = ifile.readlines()
    totalevidences = len(evidlines)
    for l in evidlines:
        parts = l.strip().split("(")
        ix = -1
        try:
            ix = prednames.index(parts[0])
        except ValueError:
            continue
        arglist = parts[1][:parts[1].find(")")].split(",")
        argsint = []
        for i,a in enumerate(arglist):
            argsint.append(int(a.strip()))
        if argsint[0]==-1:
            continue
        if len(argsint)==1:
            sevidtable[ix][argsint[0]]=argsint[0]   
        for a in argsint[1:]:
            evidtable[ix][argsint[0]].append(a)
    #print(evidtable)        
    ifile.close()
    #sys.exit(0)
    allclusters = []
    numclusters = []
    for r in reducedsizes:
        numclusters.append(r)


    
    pred_domain_dict = {}
    for i,t in enumerate(tables):
        for j,predname in enumerate(t):
            if predname not in pred_domain_dict:
                pred_domain_dict[predname] = []

            l = len(pred_domain_dict[predname])
            obj_pos = tableids[i][j]
            if l < (obj_pos + 1):
                for a in range(l,obj_pos +1):
                    pred_domain_dict[predname] += [0]

            pred_domain_dict[predname][tableids[i][j]] = i
    pickle.dump(pred_domain_dict,open("kmeans/pred_domain_dict_kmeans.p","wb"))

    converted_pdm = {}
    for pname in pred_domain_dict:
        for i,dom in enumerate(pred_domain_dict[pname]):
            converted_pdm[dom] = pdm[pname][i]



    for i,t in enumerate(tables):
        idx = prednames.index(t[0])
        sz = len(predtables[idx][tableids[i][0]])
        features = []
        for j in range(0,sz,1):
            v_features = []
            for k in range(0,len(t),1):
                ix = prednames.index(t[k])
                v_features.append(predtables[ix][tableids[i][k]][j])
                ix1 = pvec.index(t[k])
                #print(str(float(predtables[ix][tableids[i][k]][j])/sizevec[ix1][tableids[i][k]]))
            
            features.append(v_features)
        clustervec = []
        #print("numclusters="+str(numclusters[i]))
        for j in range(0,numclusters[i],1):
            tmp = []
            clustervec.append(tmp)
        #if os.name=="nt":
        #    os.system("java -cp .;./weka.jar JCLift "+"d-"+str(i)+".csv "+sys.argv[5]+" "+str(numclusters[i])+" 102923810 > d-"+str(i)+".dat")
        #else:
        #    os.system("java -cp .:./weka.jar JCLift "+"d-"+str(i)+".csv "+sys.argv[5]+" "+str(numclusters[i])+" 102923810 > d-"+str(i)+".dat")
        




        #print(Z[i])
        print("Clustering using KMeans")
        KM = KMeans(n_clusters=numclusters[i])
        KM.fit(features)
            
        '''
        ifile = open("d-"+str(i)+".dat")
        lines = ifile.readlines()
        ifile.close()
        intvals = []
        for j,l in enumerate(lines):
            clustervec[int(l.strip())].append(j)
        '''
        for j,l in enumerate(KM.labels_):
            clustervec[l].append(j)

        allclusters.append(clustervec)


    for i,d in enumerate(allclusters):
        for j in range(len(d)-1,-1,-1):
            c = allclusters[i][j]
            if len(c) == 0:
                del allclusters[i][j]


    for i,d in enumerate(allclusters):
        cdom = converted_pdm[i]
        while len(w2v_dom_objs_map[cdom]) > len(allclusters[i]):

            max_size = -1
            max_index = -1
            for j,c in enumerate(allclusters[i]):
                l = len(c)
                if l > max_size:
                    max_size = l
                    max_index = j
            c = allclusters[i][max_index][0:]
            del allclusters[i][max_index]
            allclusters[i] += [c[0:max_size//2],c[max_size//2:]]

    for i,d in enumerate(allclusters):
        for j,c in enumerate(d):
            for k,obj in enumerate(c):
                allclusters[i][j][k] = 'd' + str(converted_pdm[i]) + '_' + str(allclusters[i][j][k])






    common_ws = []
    for d in allclusters:
        for c in d:
            common_ws += [c]


    doms = {}
    for cw in common_ws:
        for w in cw:
            dom = w.split('_')[0]
            if dom not in doms:
                doms[dom] = []
            if w not in doms[dom]:
                doms[dom] += [w]


    kmeans_orig_meta_map = {}
    kmeans_meta_orig_map = {}
    kmeans_meta_atoms = {}

    for p in Atoms:
        if p not in kmeans_meta_atoms:
            kmeans_meta_atoms[p] = []
        for atom in Atoms[p]:
            kmeans_meta_atoms[p] += [atom[0:]]






    for cw in common_ws:
        dom = int(cw[0].split('_')[0].replace('d',''))
        meta_obj = cw[0].split('_')[1]
        for i,w in enumerate(cw):
            ow = w.split('_')[1]
            if w in kmeans_orig_meta_map:
                continue
            for pred_idx in dom_pred_map[dom]:
                predname = pred_idx[0]
                obj_idx = pred_idx[1]
                if predname not in kmeans_meta_atoms:
                    continue
                for atom in kmeans_meta_atoms[predname]:
                    if atom[obj_idx] == ow:
                        atom[obj_idx] = meta_obj
                        kmeans_orig_meta_map[w] = cw[0]



    for p in kmeans_meta_atoms:
        for atom in kmeans_meta_atoms[p]:
            for i,obj in enumerate(atom):
                obj = 'd' + str(pdm[p][i]) + '_' + obj
                if obj not in kmeans_orig_meta_map:
                    kmeans_orig_meta_map[obj] = obj




    print('kmeans mapping done')
    self_kmeans_meta_atoms = {}
    for p in kmeans_meta_atoms:
        self_kmeans_meta_atoms[p] = []
        for atom in kmeans_meta_atoms[p]:
            if atom not in self_kmeans_meta_atoms[p]:
                self_kmeans_meta_atoms[p] += [atom]

    print('---------------------')




    print('distinct kmeans mapping done')
    d = {}
    for p in self_kmeans_meta_atoms:
        d[p] = len(self_kmeans_meta_atoms[p])
    self_pred_atoms_reduced_numbers = [(k, d[k]) for k in sorted(d, key=d.get, reverse=False)]

    return self_kmeans_meta_atoms,kmeans_orig_meta_map,self_pred_atoms_reduced_numbers


