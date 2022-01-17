import numpy as np

def float_endf(string):
    """
    convert endf-float format to scientific notation
    """
    if "Inf" in string:
        string = "0.0"
    if "." in string:
        string = string[0] + string[1:].replace("+", "e+", 1)
        string = string[0] + string[1:].replace("-", "e-", 1)
    return float(string)

def parser(buffer, mode="num"):
    """
    parsing endf-format textline
    if mode="num", 6-float src will be return
    if mode="str", 66-string src will be return
    """
    scr = []
    line = buffer.readline()
    mat = int(line[66:70])
    mf  = int(line[70:72])
    mt  = int(line[72:75])
    nsh = int(line[75:80])
    if mode == "num":
        for i in range(6):
            scr += line[11*i:11*(i+1)].split()
        scr = list(map(float_endf, scr))
        return np.array(scr), mat, mf, mt, nsh
    elif mode == "str":
        scr = line[:66]
        return scr, mat, mf, mt ,nsh
    else:
        raise ValueError("illegal mode!")

#### particle type ####
particle_type = {6  : "n"           , 21 : "p"           , 22 : "d"           ,
                 23 : "t"           , 24 : "he-3"        , 25 : "a"           ,
                 26 : "res"         , 27 : "depo"}
    
#### reaction type ####
reaction_type = {1  : "(z,total)"   , 2  : "(elastic)"   , 3  : "(nonelastic)",
                 4  : "(z,n')"      , 5  : "(z,any)"     , 10 : "(continuum)" ,
                 11 : "(z,2nd)"     , 16 : "(z,2n)"      , 17 : "(z,3n)"      ,
                 18 : "(z,fission)" , 19 : "(z,f)"       , 20 : "(z,nf)"      ,
                 21 : "(z,2nf)"     , 22 : "(z,na)"      , 23 : "(z,n3a)"     ,
                 24 : "(z,2na)"     , 25 : "(z,3na)"     , 27 : "(z,absorp)"  ,
                 28 : "(z,np)"      , 29 : "(z,n2a)"     , 30 : "(z,2n2a)"    ,
                 32 : "(z,nd)"      , 33 : "(z,nt)"      , 34 : "(z,nHe-3)"   ,
                 35 : "(z,nd2a)"    , 36 : "(z,nt2a)"    , 37 : "(z,4n)"      ,
                 38 : "(z,3nf)"     , 41 : "(z,2np)"     , 42 : "(z,3np)"     ,
                 44 : "(z,n2p)"     , 45 : "(z,npa)"     , 91 : "(z,nc)"      ,
                 101: "(disapp)"    , 102: "(z,gamma)"   , 103: "(z,p)"       ,
                 104: "(z,d)"       , 105: "(z,t)"       , 106: "(z,He-3)"    ,
                 107: "(z,a)"       , 108: "(z,2a)"      , 109: "(z,3a)"      ,
                 111: "(z,2p)"      , 112: "(z,pa)"      , 113: "(z,t2a)"     ,
                 114: "(z,d2a)"     , 115: "(z,pd)"      , 116: "(z,pt)"      ,
                 117: "(z,da)"      , 221: "(thermal)"   , 649: "(z,pc)"      ,
                 699: "(z,dc)"      , 749: "(z,tc)"      , 799: "(z,3-Hec)"   ,
                 849: "(z,ac)"}

for i in range(41): # (z,n') reactions
    reaction_type[i+50] = "(z,n" + str(i) + ")"

for i in range(49): # (z,p') reactions
    reaction_type[i+600] = "(z,p" + str(i) + ")"

for i in range(49): # (z,d') reactions
    reaction_type[i+650] = "(z,d" + str(i) + ")"

for i in range(49): # (z,t') reactions
    reaction_type[i+700] = "(z,t" + str(i) + ")"

for i in range(49): # (z,He-3') reactions
    reaction_type[i+750] = "(z,3-He" + str(i) + ")"

for i in range(49): # (z,a') reactions
    reaction_type[i+800] = "(z,a" + str(i) + ")"

#### list of reactions that generate secondary particle ####
reaction_secondary = [11 , 16 , 17 , 22 , 23 , 24 , 25 , 28 ,
                      29 , 30 , 32 , 33 , 34 , 35 , 36 , 37 , 
                      41 , 42 , 44 , 45 , 111, 115, 116]

reaction_secondary_neutron = []
for i in range(50, 92): # (z,n') reactions
    reaction_secondary_neutron += [i]

reaction_secondary_proton = []
for i in range(600, 650): # (z,p') reactions
    reaction_secondary_proton += [i]

#### list of nonelastic reactions ####
reaction_nonelastic = [4  , 5  , 11 , 16 , 17 , 18 , 22 , 23 , 
                       24 , 25 , 26 , 28 , 29 , 30 , 31 , 32 , 
                       33 , 34 , 35 , 36 , 37 , 41 , 42 , 44 , 
                       45 , 103, 104, 105, 106, 107, 108, 109, 
                       111, 112, 113, 114, 115, 116, 117]
                      
#### list of reactions that don't generate secondary particle ####
reaction_absorption = [108, 109, 113, 114, 117]

for i in range(650,850): # heavy ion producting reactions
    reaction_absorption += [i]

#### hadron multiplicity (particle=26 is always 1) ####
reaction_multiplicity = {2  : {6 : 1},
                         4  : {6 : 1},
                         11 : {6 : 2, 22: 1},
                         16 : {6 : 2},
                         17 : {6 : 3},
                         22 : {6 : 1, 25: 1},
                         23 : {6 : 1, 25: 3},
                         24 : {6 : 2, 25: 1},
                         25 : {6 : 3, 25: 1},
                         28 : {6 : 1, 21: 1},
                         29 : {6 : 1, 25: 2},
                         30 : {6 : 2, 25: 2},
                         32 : {6 : 1, 22: 1},
                         33 : {6 : 1, 23: 1},
                         34 : {6 : 1, 24: 1},
                         35 : {6 : 1, 22: 1, 25: 2},
                         36 : {6 : 1, 23: 1, 25: 2},
                         37 : {6 : 4},
                         41 : {6 : 2, 21: 1},
                         42 : {6 : 3, 21: 1},
                         44 : {6 : 1, 21: 2},
                         45 : {6 : 1, 21: 1, 25: 1},
                         103: {21: 1},
                         111: {21: 2},
                         112: {21: 1, 25: 1},
                         115: {21: 1, 22: 1},
                         116: {21: 1, 23: 1}}

for i in range(50, 92): # (z,n') reactions
    reaction_multiplicity[i] = {6 : 1}

for i in range(600, 650): # (z,p') reactions
    reaction_multiplicity[i] = {21: 1}

#### list of cutoff particles ####
particle_cutoff_target = [22, 23, 24, 25]

def getCutoffParticleNumber(mt):
    n = 0
    secondary_list = reaction_multiplicity[mt]
    for particle in particle_cutoff_target:
        if particle in secondary_list.keys():
            n += secondary_list[particle]
    return n
