import numpy as np
import math
import os
import random
import copy
from string import digits

onlysolvatebox=False
# solvate box file
solvateboxmol2 = os.path.dirname(os.path.abspath(__file__)) + "/tip3p_30x30.mol2"

# minimal distance to solute (distance between atom centers - vdw radii)
mindistance = 0.001

class Water(object):
    def __init__(self,Oatom,H1atom,H2atom):
        #~ self.resid = resid
        self.Oatom = Oatom
        self.H1atom = H1atom
        self.H2atom = H2atom
       
    def setmindistance(self):
        self.mindistance=min(self.Oatom.distance,self.H1atom.distance,self.H2atom.distance)


class Atom(object):
    def __init__(self,atomnumber,symbol,x,y,z,atomtype,resid,resname,charge=None,radius=None):
        self.atomnumber = atomnumber
        self.symbol = symbol.translate(None, digits)
        self.resname = resname
        self.xcor = x
        self.ycor = y
        self.zcor = z
        self.atomtype = atomtype
        self.charge = charge
        self.radius = radius
        self.distance = None
        
    def setdistance(self,distance):
        self.distance = distance
        
def symbol_to_vdw(symbol):
    """ 
    extract from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    vdw radii values in A
    """
    vdwdic = {"H":  1.2,
    "He":   1.4,
    "Li":   1.82,
    "Be":   1.53,
    "B":    1.92,
    "C":    1.7,
    "N":    1.55,
    "O":    1.52,
    "F":    1.47,
    "Ne":   1.54,
    "Na":   2.27,
    "Mg":   1.73,
    "Al":   1.84,
    "Si":   2.1,
    "P":    1.8,
    "S":    1.8,
    "Cl":   1.75,
    "Ar":   1.88,
    "K":    2.75,
    "Ca":   2.31,
    "Sc":   2.11,
    "Ti":   None ,
    "V":    None ,
    "Cr":   None ,
    "Mn":   None ,
    "Fe":   None ,
    "Co":   None ,
    "Ni":   1.63,
    "Cu":   1.4,
    "Zn":   1.39,
    "Ga":   1.87,
    "Ge":   2.11,
    "As":   1.85,
    "Se":   1.9,
    "Br":   1.85,
    "Kr":   2.02,
    "Rb":   3.03,
    "Sr":   2.49,
    "Y":    None ,
    "Zr":   None ,
    "Nb":   None ,
    "Mo":   None ,
    "Tc":   None ,
    "Ru":   None ,
    "Rh":   None ,
    "Pd":   1.63,
    "Ag":   1.72,
    "Cd":   1.58,
    "In":   1.93,
    "Sn":   2.17,
    "Sb":   2.06,
    "Te":   2.06,
    "I":    1.98,
    "Xe":   2.16,
    "Cs":   3.43,
    "Ba":   2.68,
    "La":   None ,
    "Ce":   None ,
    "Pr":   None ,
    "Nd":   None ,
    "Pm":   None ,
    "Sm":   None ,
    "Eu":   None ,
    "Gd":   None ,
    "Tb":   None ,
    "Dy":   None ,
    "Ho":   None ,
    "Er":   None ,
    "Tm":   None ,
    "Yb":   None ,
    "Lu":   None ,
    "Hf":   None ,
    "Ta":   None ,
    "W":    None ,
    "Re":   None ,
    "Os":   None ,
    "Ir":   None ,
    "Pt":   1.75,
    "Au":   1.66,
    "Hg":   1.55,
    "Tl":   1.96,
    "Pb":   2.02,
    "Bi":   2.07,
    "Po":   1.97,
    "At":   2.02,
    "Rn":   2.2,
    "Fr":   3.48,
    "Ra":   2.83,
    "Ac":   None ,
    "Th":   None ,
    "Pa":   None ,
    "U":    1.86,
    "Np":   None ,
    "Pu":   None ,
    "Am":   None ,
    "Cm":   None ,
    "Bk":   None ,
    "Cf":   None ,
    "Es":   None ,
    "Fm":   None ,
    "Md":   None ,
    "No":   None ,
    "Lr":   None ,
    "Rf":   None ,
    "Db":   None ,
    "Sg":   None ,
    "Bh":   None ,
    "Hs":   None ,
    "Mt":   None ,
    "Ds":   None ,
    "Rg":   None ,
    "Cn":   None ,
    "Nh":   None ,
    "Fl":   None ,
    "Mc":   None ,
    "Lv":   None ,
    "Ts":   None ,
    "Og":   None ,
    "D":    0.01  , #dummy atom
    }
    try:
        vdwradius = vdwdic[symbol]
        if vdwradius == None:
            print "vdw radius of %s is not defined. Please check this value."%(symbol)
            return 0
        else:
            return vdwradius
    except:
        return "There is an error in the atom symbol %s. Please check this."%(symbol)



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def measure_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-
    p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))


def read_mol2(filename):
    """
    inputmolecule mol2 file
    """
    mol2file = open(filename,"r")
    mol2lines = mol2file.readlines()
    mol2file.close()
        
    for iline in range(0,len(mol2lines)):
        if "@<TRIPOS>ATOM" in mol2lines[iline]:
            minline = iline
        elif "@<TRIPOS>BOND" in mol2lines[iline]:
            maxline = iline
   
    atomlst = []
    
    # to calculate pseudo centroid
    minxcor=None
    maxxcor=None
    minycor=None
    maxycor=None
    minzcor=None
    maxzcor=None
    for iline in range(minline+1,maxline): 
        if iline>minline and iline<maxline:
            mol2linesdiv = mol2lines[iline].split()
            atomnumber = int(mol2linesdiv[0])
            symbol = mol2linesdiv[1]
            desc = mol2linesdiv[0]
                        
            xcor =  float(mol2linesdiv[2]) 
            if xcor < minxcor or minxcor==None:
                minxcor = xcor
            if xcor > maxxcor or maxxcor==None:
                maxxcor = xcor
            
            ycor =  float(mol2linesdiv[3])
            if ycor < minycor or minycor==None:
                minycor = ycor
            if ycor > maxycor or maxycor==None:
                maxycor = ycor
                
            zcor =  float(mol2linesdiv[4])
            if zcor < minzcor or minzcor==None:
                minzcor = zcor
            if zcor > maxzcor or maxzcor==None:
                maxzcor = zcor
                
            atomtype = mol2linesdiv[5]
            resid =  int(mol2linesdiv[6])
            resname = mol2linesdiv[7]
            charge =  mol2linesdiv[8]
            
            atom = Atom(atomnumber,symbol,xcor,ycor,zcor,atomtype,resid,resname,charge)             
            atomlst.append(atom)
    
    centroid = [minxcor + (maxxcor-minxcor)/2, minycor + (maxycor-minycor)/2, minzcor + (maxzcor-minzcor)/2]
    return [atomlst,centroid]


def solvateshell(inputmolecule, conformernumber=10, waternumber=50):
    filename = inputmolecule.split(".")[0]
    
    print "Building %i complex with %i waters"%(conformernumber,waternumber)

    solvatomlst,solvcentroid = read_mol2(solvateboxmol2)
    soluteatomlst,solutecentroid = read_mol2(inputmolecule)
    
    trasvector = [solutecentroid[0]-solvcentroid[0],solutecentroid[1]-solvcentroid[1],solutecentroid[2]-solvcentroid[2]]
    
    # solvent box is move to solute centroid
    centeredsolvatomlst = []
    for solvatom in solvatomlst:
        atom = copy.deepcopy(solvatom)
        atom.xcor = solvatom.xcor + trasvector[0]
        atom.ycor = solvatom.ycor + trasvector[1]
        atom.zcor = solvatom.zcor + trasvector[2]
        centeredsolvatomlst.append(atom)
                   
    # build n complex (solvent + solute)
    for conformerindex in range(0,conformernumber):
        #random values to rotate solute
        #~ axis = [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]
        axis = solutecentroid
        theta = random.uniform(-10, 10) 
        
        # solvate box is rotated
        rotatedsolvatomlst = []
        for solvatom in centeredsolvatomlst: 
            rotatedsolvatom = copy.copy(solvatom)
            rotatedsolvatom.xcor, rotatedsolvatom.ycor, rotatedsolvatom.zcor = np.dot(rotation_matrix(axis,theta), [solvatom.xcor, solvatom.ycor,solvatom.zcor]) 
            rotatedsolvatomlst.append(rotatedsolvatom)
        
        # measure distance
        for solvatom in rotatedsolvatomlst:
            solvdistance = []
            for soluteatom in soluteatomlst:
                p1 = [soluteatom.xcor, soluteatom.ycor, soluteatom.zcor]
                p2 = [solvatom.xcor, solvatom.ycor, solvatom.zcor]
                # distance - ligand atom vdw radius * 0.7 - solvent atom vdw radius * 0.7
                solvdistance.append(measure_distance(p1,p2)-symbol_to_vdw(soluteatom.symbol)*0.7-symbol_to_vdw(solvatom.symbol)*0.7)
            solvatom.distance = min(solvdistance)

        # create water molecules
        waterlst = []           
        for sindex in range(0,len(rotatedsolvatomlst),3):
            water = Water(rotatedsolvatomlst[sindex],rotatedsolvatomlst[sindex+1],rotatedsolvatomlst[sindex+2])
            water.setmindistance()
            waterlst.append(water)

        sortwaterlst = sorted(waterlst, key=lambda k: k.mindistance) 

        outputfile = open("%s_%iWAT-%i.xyz"%(filename,waternumber,conformerindex),"w")
        if onlysolvatebox==True:
			outputfile.write("%i\n\n"%(waternumber*3))
        else:
			outputfile.write("%i\n\n"%(len(soluteatomlst)+waternumber*3))
			
        if onlysolvatebox!=True:
			for atom in soluteatomlst:
				outputfile.write("%s \t %7.4f \t %7.4f \t %7.4f \n"%(atom.symbol, atom.xcor, atom.ycor, atom.zcor))
        
        # select nearest water and exclude overlaping waters
        wateradded = 0
        waterindex = 0
        while wateradded < waternumber and waterindex<=len(sortwaterlst):
            water=sortwaterlst[waterindex]
            if water.mindistance > mindistance or onlysolvatebox==True:
                outputfile.write("%s \t %7.4f \t %7.4f \t %7.4f \n"%(water.Oatom.symbol, water.Oatom.xcor, water.Oatom.ycor, water.Oatom.zcor))
                outputfile.write("%s \t %7.4f \t %7.4f \t %7.4f \n"%(water.H1atom.symbol, water.H1atom.xcor, water.H1atom.ycor, water.H1atom.zcor))
                outputfile.write("%s \t %7.4f \t %7.4f \t %7.4f \n"%(water.H2atom.symbol, water.H2atom.xcor, water.H2atom.ycor, water.H2atom.zcor))
                wateradded += 1
            waterindex += 1
        if wateradded < waternumber-1:
            print "Warning: %s_%iWAT-%i.xyz has %i waters!"%(filename,waternumber,conformerindex,wateradded)
        outputfile.close()
        

