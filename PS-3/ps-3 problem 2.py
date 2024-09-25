#10.2 in Newman


from random import random
import numpy as np
import matplotlib.pyplot as plt

dt=1
tmax=20000

nbi = 10000
ntl = 0
npb = 0
nbi209 = 0

bipoints=[]
tlpoints=[]
pbpoints=[]
bi209points=[]

TBi = 60*46
TTl = 60*2.2
TPb = 60*3.3
ppbdecay = 1-2**(-dt/TPb)
ptldecay = 1-2**(-dt/TTl)
pbidecay = 1-2**(-dt/TBi)
    
   
tpoints=np.arange(0,tmax,dt)
for t in tpoints:
    decayed_pb = 0
    decayed_tl = 0
    decayed_bi=0
    bi_decayto_pb = 0
    bi_decayto_tl = 0
    
    #how many pb atoms have decayed at time t
    for i in range (npb):
        if random() < ppbdecay:
            decayed_pb += 1
    #pb atoms remaining 
    npb -= decayed_pb
    #where pb atoms went
    nbi209 += decayed_pb
            
    for i in range (ntl):
        if random() < ptldecay:
            decayed_tl += 1
    ntl -= decayed_tl
    npb += decayed_tl
    
    for i in range (nbi):
        if random() < pbidecay:
            decayed_bi += 1
    nbi -= decayed_bi
    
    #where do the bi atoms go
    for i in range (decayed_bi):
        if random() < 0.0209:
            bi_decayto_tl += 1
        else:
            bi_decayto_pb +=1
            
    ntl += bi_decayto_tl
    npb += bi_decayto_pb
    
    bipoints.append(nbi)
    tlpoints.append(ntl)
    pbpoints.append(npb)
    bi209points.append(nbi209)
plt.figure(figsize=[9,7])    
plt.plot(tpoints,bipoints,label='No. of Bi213 Atoms')
plt.plot(tpoints,tlpoints,label='No. of Tl Atoms')
plt.plot(tpoints,pbpoints,label='No. of Pb Atoms')
plt.plot(tpoints,bi209points,label='No. of Bi209 Atoms')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('No. of Atoms')
plt.title('Radioactive Decay of Bi 213')
#plt.savefig("decay.png")
plt.show()
    
