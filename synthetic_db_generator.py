from DBManager import dbconfig

import math
import numpy as np
import random

class sdg(dbconfig):
	def __init__(self,dsname):
		dbconfig.__init__(self,dsname)
		self.generate_db()


	def add_log_weights(self,x,y):
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
    
	def nCr(self,n,r):
	    f = math.factorial
	    return math.log1p(f(n)) - (math.log1p(f(r)) + math.log1p(f(n-r)))

	def generate_db(self):

		#domain-size of R
		D1 = 50
		#domain-size of each of S_0, S_1,...
		D2 = 60

		D3 = 70
		#number of formulas
		K = 15

		w = []
		Evids = []
		Evids_1 = []
		ofile = open(self.orig_db_file,'w')
		for k in range(0,K,1):
		    tmp = []
		    tmp1 = []
		    #w1 = random.randint(0.001,0.1)
		    #w1 = random.uniform(0.0000001,0.000001)
		    w1 = random.uniform(-0.001,0)
		    w.append(w1)
		    for i in range(0,D1,1):
		        r = random.randint(10,60)
		        tmp.append(r)
		        S = np.random.randint(0,D2-1,size=r)
		        S = np.unique(S)
		        for s in S:
		            ofile.write("S"+str(k)+"("+str(i)+","+str(s)+")\n")
		        r = random.randint(10,30)
		        tmp1.append(r)
		        S = np.random.randint(0,D3-1,size=r)
		        S = np.unique(S)
		        for s in S:
		            ofile.write("T"+str(k)+"("+str(i)+","+str(s)+")\n")


		    Evids.append(tmp)
		    Evids_1.append(tmp1)
		ofile.close()

		#print(w)
		#print(len(Evids))

		S_T = []
		S_F = []
		for i in range(0,D1,1):
		    S_T.append(0)
		    S_F.append(0)
		for j,E in enumerate(Evids):
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
		                n_g1 = self.nCr(dc1,k1)
		                n_g2 = self.nCr(dc2,k2)
		                n_t = (dc1*dc2 - (dc1 - k1)*(dc2-k2))*w[j]
		                s = self.add_log_weights(s,n_g1 + n_g2 + n_t)
		        #print(math.exp(s))
		        #print(2**dc)
		        #print(math.log1p(2**dc))
		        #x =w[i]+s - add_log_weights(w[i]+s,math.log1p(2**dc))
		        S_F[i] = self.add_log_weights(S_F[i],s)
		        V = self.add_log_weights(math.log1p(2**dc),(D2+D3)*w[j])
		        S_T[i] = self.add_log_weights(S_T[i],V)
		        #print(w[i]*s)
		        #print(2**dc)
		        #print(x)
		        #print(math.log1p(2**dc))
		        #print(s)
		        #print(math.exp(x))

		ofile = open(self.syn_marginal_file,'w')
		for i in range(0,len(S_T),1):
		    x = S_T[i] - self.add_log_weights(S_T[i],S_F[i])
		    x = math.exp(x)
		    print(x)
		    ofile.write(str(x)+"\tR("+str(i)+")\n")
		ofile.close()

		ofile = open(self.mln_tuffy_full_name,'w')
		for k in range(0,K,1):
		    ofile.write("*S"+str(k)+"(x,y)\n")
		    ofile.write("*T"+str(k)+"(x,z)\n")
		ofile.write("R(x)\n\n")

		for k in range(0,K,1):
		    ofile.write(str(math.exp(w[k]))+" R(x) v S"+str(k)+"(x,y) v T"+str(k)+"(x,z)\n")
		ofile.close()

		ofile = open(self.mln_file_full_name,'w')
		first_line = 'R:' + str(D1) + ' '
		formulas = ''
		for k in range(0,K,1):
			first_line += 'S' + str(k) + ':' + str(D1) + ':' + str(D2) + ' '
			first_line += 'T' + str(k) + ':' + str(D1) + ':' + str(D3) + ' '
			formulas += str(math.exp(w[k]))+":R(x) v S"+str(k)+"(x,y) v T"+str(k)+"(x,z)\n"
		ofile.write(first_line + '\n')
		ofile.write(formulas)
		ofile.close()


sdg('synthetic')