import os
import math
import numpy as np
import random

from collections import defaultdict

def add_log_weights(x,y):
    """
        Add two probabilities in log space.
        This is a more numerically stable version of log(e^x + e^y).
    """
    # addition is symmetric, so make x the bigger number => exp(y-x) < 1
    if y > x:
        y,x = x,y
    # if one number is way smaller than the other, don't bother
    # a difference much greater than e^300 over/undeflows 64-bit floats
    if x - y > 40:
        return x
    else:
        return x + np.log1p(np.exp(y-x))
    
def nCr(n,r):
    f = math.factorial
    return math.log1p(f(n)) - (math.log1p(f(r)) + math.log1p(f(n-r)))

db_dir = 'synthetic/db/'
exact_mar_file = 'synthetic/marginal/mar_out.txt'

exact_marginals = {}
ifile = open(exact_mar_file)
for l in ifile:
    parts = l.split('\t')
    idx = int(parts[1].split('(')[1].split(')')[0])
    prob = float(parts[0].strip())
    exact_marginals[idx] = prob
ifile.close()
# coding: utf-8

# In[1]:
K = 15
D1 = 50
D2 = 100
D3 = 100

import random
w = []
for k in range(0,K,1):
    tmp = []
    w1 = random.uniform(-0.001,0)
    w.append(w1)

w = [-0.0002970596982730298, -0.0006734531050690709, -0.0001620596460277842, -2.9390005094202852e-05, -0.00015471233947247313, -0.0006234923737992741, -0.0009158949338865249, -0.0001778573958377092, -0.0002115525145984713, -0.0005780059604759512, -0.00022799326694984462, -0.00040561938807269276, -0.00037283415926450366, -0.0005407277937738422, -3.093510086443329e-05]
# In[ ]:
meta_marginals = {}
files = os.listdir(db_dir)
for fname in files:
    if not (fname.startswith('kmeans') or fname.startswith('w2v')):
        continue

    f_full = db_dir + fname
    ifile = open(f_full)
    Evids = []
    Evids_1 = []

    for k in range(0,K,1):
        tmp = []
        for j in range(0,D1,1):
            tmp.append(0)
        Evids.append(tmp)
        tmp = []
        for j in range(0,D1,1):
            tmp.append(0)
        Evids_1.append(tmp)
    Ex = {}
    Ex_1 = {}
    for ln in ifile:
        parts = ln.strip().split("(")
        parts1 = parts[1][:len(parts[1])-1].split(",")

        ix = int(parts[0][1:len(parts[0])])
        if parts[0][0]=="S":
            if str(ix)+"-"+str(parts1[0]) not in Ex:
                Ex[str(ix)+"-"+str(parts1[0])] = 1
            else:
                Ex[str(ix)+"-"+str(parts1[0])] = Ex[str(ix)+"-"+str(parts1[0])] + 1
        else:
            if str(ix)+"-"+str(parts1[0]) not in Ex_1:
                Ex_1[str(ix)+"-"+str(parts1[0])] = 1
            else:
                Ex_1[str(ix)+"-"+str(parts1[0])] = Ex_1[str(ix)+"-"+str(parts1[0])] + 1

    for k in Ex.keys():
        parts = k.split("-")
        ix = int(parts[0])
        ix1 = int(parts[1])
        Evids[ix][ix1] = Ex[k]
    for k in Ex_1.keys():
        parts = k.split("-")
        ix = int(parts[0])
        ix1 = int(parts[1])
        Evids_1[ix][ix1] = Ex_1[k]
    ifile.close()
    #print(Evids)


    # In[ ]:





    S_T = []
    S_F = []
    for i in range(0,D1,1):
        S_T.append(0)
        S_F.append(0)
    for j in range(0,K,1):
        E = Evids[j]
        for i in range(0,D1,1):
            s = 0
            dc = D2+D3-(E[i]+Evids_1[j][i])
            dc1 = D2 - E[i]
            dc2 = D3 - Evids_1[j][i]
            for k1 in range(1,dc1+1,1):
                for k2 in range(1,dc2+1,1):
                    #l = np.log1p(nCr(dc,k))
                    #l_1 = nCr(dc1,k1)
                    #l_2 = nCr(dc2,k2)
                    #L = add_log_weights(l_1,l_2)
                    #n_t = dc1*dc2 - (dc1 - k1)*(dc2-k2)
                    #s1 = add_log_weights(L,math.log1p(n_t))
                    #s = add_log_weights(s1,s)
                    n_g1 = nCr(dc1,k1)
                    n_g2 = nCr(dc2,k2)
                    n_t = (dc1*dc2 - (dc1 - k1)*(dc2-k2))*w[j]
                    s = add_log_weights(s,n_g1 + n_g2 + n_t)
            #print(math.exp(s))
            #print(2**dc)
            #print(math.log1p(2**dc))
            #x =w[i]+s - add_log_weights(w[i]+s,math.log1p(2**dc))
            S_F[i] = add_log_weights(S_F[i],s)
            V = add_log_weights(math.log1p(2**dc),(D2+D3)*w[j])
            S_T[i] = add_log_weights(S_T[i],V)
            #print(w[i]*s)
            #print(2**dc)
            #print(x)
            #print(math.log1p(2**dc))
            #print(s)
            #print(math.exp(x))

    meta_marginals[fname] = {}
    ofile = open("synthetic/out/error_marginals.txt",'w')
    for i in range(0,len(S_T),1):
        x = S_T[i] - add_log_weights(S_T[i],S_F[i])
        x = math.exp(x)
       # print(x)
        ofile.write(str(x)+" R("+str(i)+")\n")
        meta_marginals[fname][i] = x
    ofile.close()
    print(fname)

    diff = 0
    orig_prob = 0
    c = 0
    ofile = open('synthetic/out/error.txt','a')
    for idx in meta_marginals[fname]:
        if idx in exact_marginals:
            c += 1
            diff += abs(meta_marginals[fname][idx] - exact_marginals[idx])
            orig_prob += exact_marginals[idx]
    if c > 0:
        ofile.write(fname + ' ' + str(diff/orig_prob) + ' ' + str(c) + '\n')
    ofile.close()



