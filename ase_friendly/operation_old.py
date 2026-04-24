########
import re
import math
import copy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from contextlib import redirect_stdout
from .src import ReadFile as RF
from .src import MathBx as MB
import random
import warnings

#the below makes comparison of different objects give False without warming
warnings.simplefilter(action='ignore', category=FutureWarning)


########
class Poscar:
    #prerequisite as statics method... the initiation will come later
            
####
    @staticmethod
    def Print2dList_2dlistNsapce( llist , nspace ):
        for line in llist:
            print(" " * nspace , end = '')
            print(*line  )

    @staticmethod
    def angle(vector_1,vector_2):
      unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
      unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
      return np.arccos(np.dot(unit_vector_1, unit_vector_2)) * 180/np.pi
            
    @staticmethod
    def DetectCart_File( filepath  ):
        linesToBeSearched=RF.FileToList_FileStartEnd(filepath , 7 , 9)
        for line in linesToBeSearched:
            if re.search( '^[Cc]' , line ):
                return True
        return False
    
    @staticmethod
    def DetectSeleDyna_File( filepath ):
        linesToBeSearched=RF.FileToList_FileStartEnd(filepath , 7 , 9)
        for line in linesToBeSearched:
            if re.search( '^[Ss]' , line ):
                return True
        return False
    
    @staticmethod
    def EmptyLine_File( filepath ):
        F=open(filepath, "r")
        for n, line in enumerate(F):
            if not line.strip():
                F.close()
                return n
        F.close()
        return False
    
    @staticmethod
    def DetectVelo_File( filepath ):
        nempty=Poscar.EmptyLine_File( filepath )
        if nempty is False:
            return False
        F=open(filepath, "r")
        for n, line in enumerate(F):
            if re.search( r'\s*[0-9].*' , line ) and n==nempty+1 :
                return True
        return False
        
        
    @staticmethod    
    def FileToPoscarArgs( filepath ):
        out=[]
        df=pd.read_csv( filepath , delimiter=r'\s+', skipinitialspace=True, engine='python', names=['c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9']) #if more element, then this needs to be longer
        twolines = [ [ entry  for entry in row if not pd.isna(entry) ] for row in df[0:2].to_numpy() ]
        out.append( twolines )
        cell = np.array( [ [ entry  for entry in row if not pd.isna(entry) ] for row in df[2:5].to_numpy() ] , dtype=float )
        out.append( cell )
        elem = np.array( [ [ entry  for entry in row if not pd.isna(entry) ] for row in df[5:7].to_numpy() ] )
        elem=elem.tolist()
        elem[1]=[ int(i) for i in elem[1]]
        out.append( elem )
        out.append( Poscar.DetectCart_File( filepath ) ) #cartboolean
        atomsstart=8+int( Poscar.DetectSeleDyna_File(filepath) )
        natoms=sum(  list(map(int, elem[1] ) ) )
        atomsAndTF = np.array( [ [ entry  for entry in row if not pd.isna(entry) ] for row in df[atomsstart:atomsstart+natoms].to_numpy() ] )
        atoms=atomsAndTF[:,:3]
        atoms=atoms.astype(float)
        out.append( atoms )
        if Poscar.DetectVelo_File( filepath ):
            velostart=atomsstart+natoms+0 
            # this is +0 without skipping the empty line because pandas dataframe skip the empty line
            F=open(filepath, "r")
            for n, line in enumerate(F):
                if re.search( r'\s*Lattice velocities.*' , line ) and n!=0 :
                    velostart=atomsstart+natoms+4      #ignore lattice velocity
            F.close()
            velo=np.array( [ [ entry  for entry in row if not pd.isna(entry) ] for row in df[velostart:velostart+natoms].to_numpy() ] , dtype=float )
            out.append(velo)
        else:
            out.append(False)
        
        if Poscar.DetectSeleDyna_File( filepath ):
            SeleDynaTF=atomsAndTF[:,3:6]
            out.append(SeleDynaTF)
        else:
            out.append(False)
        
        return out
                
####
    #initiation
    def __init__(self, twolines, cell, elem, cartboolean, atoms, velo, SeleDynaTF ):
        # use the static method FileToPoscarArgs( filepath ) to figure out the format of each attribute
        self.twolines = twolines
        self.cell = cell
        self.elem  =elem #elem has to be a list of lists, while other stuff can be np array or list of lists
        self.cartboolean  =cartboolean
        self.atoms  =atoms
        self.velo  =velo
        self.SeleDynaTF  =SeleDynaTF

####
    def FromFile( filepath ):
        return Poscar( *Poscar.FileToPoscarArgs(filepath) )
        
####
    def CpPoscar(self):
        if self.velo is not False:
            newvelo=np.copy(self.velo)
        else:
            newvelo=copy.deepcopy(self.velo)
        
        if self.SeleDynaTF is not False:
            newSeleDynaTF=np.copy(self.SeleDynaTF)
        else:
            newSeleDynaTF=copy.deepcopy(self.SeleDynaTF)        
        return Poscar( copy.deepcopy(self.twolines) , np.copy(self.cell) , copy.deepcopy(self.elem) , copy.deepcopy(self.cartboolean) , np.copy(self.atoms) , newvelo , newSeleDynaTF )
####
    def pr(self):
        Poscar.Print2dList_2dlistNsapce(self.twolines , 0)
        Poscar.Print2dList_2dlistNsapce(self.cell , 1)
        Poscar.Print2dList_2dlistNsapce(self.elem , 2)
        
        if self.SeleDynaTF is False :
            pass
        else:
            print('Select Dynamics')
            
        if self.cartboolean :
            print('Cartesian')
        else:
            print('Direct')
        
        if self.SeleDynaTF is False :
            Poscar.Print2dList_2dlistNsapce(self.atoms , 1)
        else:
            Poscar.Print2dList_2dlistNsapce(\
            np.concatenate((  np.array(self.atoms)  ,   np.array(self.SeleDynaTF)   ), axis=1) \
            , 1)
            
        if self.velo is False :
            print('')
        else:
            print('')
            Poscar.Print2dList_2dlistNsapce( self.velo    , 1)
            
####
    def pr_outputpath(self, outputpath):
        with open(str(outputpath), 'w') as f:
            with redirect_stdout(f):
                self.pr()
        return 0
            
####
    def Mod0To1(self):
        def mmod(x):
            return x % 1
        myfunc_vec = np.vectorize(mmod)
        self.atoms = myfunc_vec(self.atoms)
        return 0
        
    def Mod0p5To0p5(self):
        def mmod(x):
            return  ( (x+0.5) % 1 ) - 0.5
        myfunc_vec = np.vectorize(mmod)
        self.atoms = myfunc_vec(self.atoms)
        return 0
        
####
    def ToCart(self):
        if self.cartboolean==True :
            return 0
        elif self.cartboolean==False :
            self.cartboolean=True
            self.atoms=self.atoms @ self.cell
            if self.velo is False :
                pass
            else:
                self.velo=self.velo @ self.cell
            return 0
        else:
            print( 'cannot detect Direct or Cartesian' )
            return 0
    
        
    def ToDire(self):
        if self.cartboolean==False :
            return 0
        elif self.cartboolean==True :
            self.cartboolean=False
            self.atoms=self.atoms @ np.linalg.inv( self.cell )
            if self.velo is False :
                pass
            else:
                self.velo=self.velo @ np.linalg.inv( self.cell )
            return 0
        else:
            print( 'cannot detect Direct or Cartesian' )
            return 0
        
####
    def ChangeCellInCartMod0p5_Cell(self,cell):
        self.ToDire()
        self.Mod0p5To0p5()
        self.ToCart()
        self.cell=cell
        self.ToDire()
        return 0
        
    def ChangeCellInCartMod0To1_Cell(self,cell):
        self.ToDire()
        self.Mod0To1()
        self.ToCart()
        self.cell=cell
        self.ToDire()
        return 0
        
####
    def StandarizedCell(self):
        para=MB.CellMatToCellParameters_2dlist(self.cell)
        self.cell=MB.CellParamtersToCellMat_List(para)
        return 0
        
####
    def Translate_ABC(self, a, b, c):
        TBAdded=np.empty( (len(self.atoms),3) , dtype=float )
        TBAdded[:,0]=a
        TBAdded[:,1]=b
        TBAdded[:,2]=c
        self.atoms=np.add( self.atoms , TBAdded )
        return 0
        
####
    def RotAtomsBrute_XYZ(self, x, y, z):
        #self.Mod0p5To0p5()
        r=R.from_rotvec( np.array([x,y,z], dtype=float) *np.pi/180 ) 
        self.ToCart()
        self.atoms= self.atoms @ np.transpose( r.as_matrix() )
        if self.velo is not False:
            self.velo= self.velo @ np.transpose( r.as_matrix() )
        else:
            pass
        self.ToDire()
        return 0
        
    def RotAtomsMod0p5_XYZ(self, x, y, z):
        self.Mod0p5To0p5()
        r=R.from_rotvec( np.array([x,y,z], dtype=float) *np.pi/180 ) 
        self.ToCart()
        self.atoms= self.atoms @ np.transpose( r.as_matrix() )
        if self.velo is not False:
            self.velo= self.velo @ np.transpose( r.as_matrix() )
        else:
            pass
        self.ToDire()
        return 0   
####
    def ReflectInCart_Xyz(self, Xyz):
        self.ToCart()
        xyz=int(Xyz)-1
        mat=np.identity(3, dtype=float)
        mat[xyz,xyz]=-1.0
        self.atoms=self.atoms @ mat
        if self.velo is not False:
            self.velo=self.velo @ mat
        else:
            pass
        self.ToDire()
        return 0
        
####
    def TransformCell_Matrix(self, mat ):
        newcell= mat @ self.cell
        self.ToCart()
        self.cell=newcell
        self.ToDire()
        return 0
        
    def TransformToNear90Cell(self):
        self.StandarizedCell()
        def trans( cell, m , n , i , j ):
            out=np.identity( 3, dtype=float)
            out[m,n]=-np.rint( cell[m,n]/cell[i,j] )
            return out
        mat= trans( self.cell, 2, 1, 1, 1 ) @ trans( self.cell, 1, 0, 0, 0 ) @ trans( self.cell, 2, 0, 0, 0 )
        self.TransformCell_Matrix( mat )
        return 0
        
    def ToNear90ByChangeOneAxis_Xyz(self, a1):
        a0=int(a1)-1 #to 0 index start. a1/a0 is the only vector that is changing.
        b0=(a0+1)%3 # 0 1 2  ;  1 2 0 ; 2 0 1
        c0=(a0+2)%3
        va=self.cell[a0,:] # vector to be changed
        vb=self.cell[b0,:] # vector to be fixed
        vc=self.cell[c0,:] # vector to be fixed
        angleMinTemp=180.0 # this is a dummy value for now
        for ib in [0,1,2,-1,-2]:
            for ic in [0,1,2,-1,-2]:
                vaTemp=va+vb*ib+vc*ic # the only vector that is changing, is changed in this line
                angleABFrom90=np.absolute( np.absolute(Poscar.angle(vaTemp,vb))-90.0 )
                angleACFrom90=np.absolute( np.absolute(Poscar.angle(vaTemp,vc))-90.0 )
                angleNet=angleABFrom90+angleACFrom90
                if angleNet<angleMinTemp:
                    angleMinTemp=angleNet
                    ibmin=ib
                    icmin=ic
        mat=np.identity( 3, dtype=float)
        mat[a0,b0]=ibmin
        mat[a0,c0]=icmin
        self.TransformCell_Matrix(mat)
        return 0
####
    def Combine_Theother(self, p):
        if self.elem[0] != p.elem[0] :
            print( 'elements of the two poscars might not match. the output might be nonsensical' )
        
        #create a function that give the start and end of an element's atoms
        def atomsstartend(p , ielem):
            start=sum(list(map(int, self.elem[1][:ielem])))
            return ( start , start+int(self.elem[1][ielem]) )
            
        nelem=len(self.elem[0])
        #create the new atoms coordinates
        ListOfAtomsArray=[]
        for i in range(nelem) :
            startend1=atomsstartend(self, i)
            startend2=atomsstartend(p, i)
            ListOfAtomsArray.append( self.atoms[startend1[0]:startend1[1]] )
            ListOfAtomsArray.append( p.atoms[startend2[0]:startend2[1]] )
        self.atoms=np.concatenate(ListOfAtomsArray, axis=0)
        
        #create the new atoms coordinates
        if self.velo is not False:
            ListOfVeloArray=[]
            for i in range(nelem) :
                startend1=atomsstartend(self, i)
                startend2=atomsstartend(p, i)
                ListOfVeloArray.append( self.velo[startend1[0]:startend1[1]] )
                ListOfVeloArray.append( p.velo[startend2[0]:startend2[1]] )
            self.velo=np.concatenate(ListOfVeloArray, axis=0)
        else:
            pass
        
        #create the new element list
        #self.elem[1]=np.add( self.elem[1] , p.elem[1] )
        for i in range(nelem) :
            self.elem[1][i]=int(self.elem[1][i])+int(p.elem[1][i])
            

        return 0
        
####
    def SuperCellPartial_AbcMultSlot(self, Abc, multiplier, slot):
        self.ToDire()
        abc=int(Abc)-1
        self.cell[abc]=self.cell[abc]*float(multiplier)
        shrink=np.identity(3, dtype=float)
        shrink[abc,abc]=1/float(multiplier)
        
        self.atoms=self.atoms @ shrink
        if self.velo is not False:
            self.velo=self.velo @ shrink
        else:
            pass
        
        translate=np.zeros( (len(self.atoms),3) , dtype=float )
        translate[:,abc]= (float(slot)-1)/float(multiplier)
        self.atoms=np.add( self.atoms , translate )
        
        return 0
        
    def SuperCellDouble_Abc(self, Abc):
        p2=self.CpPoscar()
        self.SuperCellPartial_AbcMultSlot( Abc, 2, 1)
        p2.SuperCellPartial_AbcMultSlot( Abc, 2, 2)
        self.Combine_Theother(p2)
        return 0

    def SuperCell_AbcN(self, abc, n):
        n=int(n)
        pini=self.CpPoscar()
        self.SuperCellPartial_AbcMultSlot(abc,n,1)
        for i in range(2,n+1):
            ptemp=pini.CpPoscar()  #check elem type
            ptemp.SuperCellPartial_AbcMultSlot(abc,n,i)
            self.Combine_Theother(ptemp)
        return 0
####
 
    def ChangeCellEntry_RowColXfunc(self, row, col, xexp):
        x=self.cell[int(row),int(col)]
        exec( "self.cell[int(row),int(col)]=" + xexp )
        return 0

    def ChangeCellEntryMult_RowColXfuncEtc(self, *arg):
        for a in range(int(len(arg)/3)):
            x=self.cell[int(arg[3*a]),int(arg[3*a+1])]
            exec( "self.cell[int(arg[3*a]),int(arg[3*a+1])]=" + arg[3*a+2] )
        return 0
        
    def ChangeCellEntryMultMod0p5_RowColXfuncEtc(self, *arg):
        self.ToDire()
        self.Mod0p5To0p5()
        self.ToCart()
        for a in range(int(len(arg)/3)):
            x=self.cell[int(arg[3*a]),int(arg[3*a+1])]
            exec( "self.cell[int(arg[3*a]),int(arg[3*a+1])]=" + arg[3*a+2] )
        self.ToDire()
        return 0
        
    def ChangeCellEntryMultMod0To1_RowColXfuncEtc(self, *arg):
        self.ToDire()
        self.Mod0To1()
        self.ToCart()
        for a in range(int(len(arg)/3)):
            x=self.cell[int(arg[3*a]),int(arg[3*a+1])]
            exec( "self.cell[int(arg[3*a]),int(arg[3*a+1])]=" + arg[3*a+2] )
        self.ToDire()
        return 0
        
    def ChangeCellEntryMultMod0To1Recenter_RowColXfuncEtc(self, *arg):
        self.Translate_ABC(-0.5, -0.5, -0.5)
        self.ToDire()
        self.Mod0p5To0p5()
        self.ToCart()
        for a in range(int(len(arg)/3)):
            x=self.cell[int(arg[3*a]),int(arg[3*a+1])]
            exec( "self.cell[int(arg[3*a]),int(arg[3*a+1])]=" + arg[3*a+2] )
        self.ToDire()
        self.Translate_ABC(0.5, 0.5, 0.5)
        return 0
####
    def ChangeCell1D_XyzL1L2(self, Xyz, l1, l2):
        xyz=int(Xyz)-1
        
        v0=self.cell[xyz]
        n0=np.linalg.norm(v0)
        a0=np.array([0.0,0.0,0.0])
        a0[xyz]=n0
        
        m=np.array( [[float(l1),0.0],[0.0,float(l2)]] )
        m1=np.insert( m , xyz , [ 0.0, 0.0] , 0)
        m2=np.insert( m1 , xyz , [ 0.0, 0.0, 0.0] , 1)
        m2[xyz]=a0
        
        self.StandarizedCell()
        self.ChangeCellInCartMod0p5_Cell(m2)
        return 0
####
    def AddBiGaus_MeanStdDegRxRyRz(self, mean, std, deg, rx, ry, rz):
        # generating rotation matrix
        rv = R.from_rotvec( float(deg) * np.pi/180 * np.array([ float(rx) , float(ry) , float(rz) ]))
        rm = rv.as_matrix()
        
        # generate gaussian along [1,0,0]. it detects i=0+1 , and then print gaussian. otherwise 0. there is also a random +1 -1
        g = [ [ np.prod([  np.add(i,1)==1 , [-1,1][random.randrange(2)] , np.random.normal( float(mean) , float(std) )  ])  for i in range( 3 )]    for j in range( len(self.atoms) ) ]
        g = np.array(g)
        
        rotatedGuassian= np.transpose( np.matmul( rm  , np.transpose(g) ) )
        self.atoms=np.add(self.atoms , rotatedGuassian)
        return 0
        
        
    def AddGaus_MeanStdDegRxRyRz(self, mean, std, deg, rx, ry, rz):
        # generating rotation matrix
        rv = R.from_rotvec( float(deg) * np.pi/180 * np.array([ float(rx) , float(ry) , float(rz) ]))
        rm = rv.as_matrix()
        
        # generate gaussian along [1,0,0]. it detects i=0+1 , and then print gaussian. otherwise 0.
        g = [ [ np.prod([  np.add(i,1)==1 , np.random.normal( float(mean) , float(std) )  ])  for i in range( 3 )]    for j in range( len(self.atoms) ) ]
        g = np.array(g)
        
        rotatedGuassian= np.transpose( np.matmul( rm  , np.transpose(g) ) )
        self.atoms=np.add(self.atoms , rotatedGuassian)
        return 0
####
    def Rearrange_Dict(self, d):
        self.atoms=MB.Reorder_ListDict( self.atoms , d )

        #create the new atoms coordinates
        if self.velo is not False:
            self.velo=MB.Reorder_ListDict( self.velo , d )
        else:
            pass
        
            

        return 0
        
####
    def AddVeloByDiff_TheotherSign(self, p, sign ):
        #np.array(self.velo)
        velodiff=np.subtract( np.array(self.atoms , dtype=float) , np.array(p.atoms , dtype=float) ) * float(sign)
        self.velo=velodiff
        return 0
        
####
    def RemoveAtoms_NelemSequ(self, nelem, *atomlist ):
        atomlist=[int(i)-1 for i in atomlist]
        nelem=int(nelem)-1
        #element number and element list to total atom list
        startingindex=sum(self.elem[1][:nelem])
        self.elem[1][nelem]=self.elem[1][nelem]-len(atomlist)
        self.atoms=np.delete(self.atoms, [atom+startingindex for atom in atomlist], axis=0)
        if self.velo is not False:
            self.velo=np.delete(self.velo, [atom+startingindex for atom in atomlist], axis=0)
        return 0
        
    def RemoveChunk_XminXmaxYYZZ(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.Mod0To1()
        mask = ((self.atoms[:, 0] >= float(xmin)) & (self.atoms[:, 0] <= float(xmax)) & (self.atoms[:, 1] >= float(ymin)) & (self.atoms[:, 1] <= float(ymax)) & (self.atoms[:, 2] >= float(zmin)) & (self.atoms[:, 2] <= float(zmax)))
        filtered_rows = np.where(mask)[0]
        #print(filtered_rows)
        #print(self.elem)
        #total atom list to element number and elment list
        # from nelem to cutoff partition
        startindexlist=[]
        for ielem in range(len(self.elem[0])):
            istartingindex=sum(self.elem[1][:ielem])
            startindexlist.append(istartingindex)
        #print(startindexlist)
        # separate a 2d numpy list by a given partition # not actully used here... can be deleted
        def separate_by_partition(array, partitions):
            separated_arrays = []
            start_index = 0
            for partition in partitions:
                separated_arrays.append(array[start_index:partition])
                start_index = partition
            separated_arrays.append(array[start_index:])
            return separated_arrays[1:] #drop first element if starting index is 0
        #print(separate_by_partition(self.atoms,cutofflist))
        # the below is used
        def sort_with_partition(partition, numbers):
            sorted_lists = []
            numbers = sorted(numbers)
            index = 0
            for cutoff in partition:
                sublist = []
                while index < len(numbers) and numbers[index] < cutoff:
                    sublist.append(numbers[index])
                    index += 1
                sorted_lists.append(sublist)
            sorted_lists.append([])  # Add an empty sublist for numbers exceeding the last cutoff
            while index < len(numbers):
                sorted_lists[-1].append(numbers[index])
                index += 1
            return sorted_lists
        NAtomByElement=sort_with_partition(startindexlist[1:],filtered_rows)
        #print()
        #print(NAtomByElement,startindexlist)
        # substract atom number by startingindex for each element
        NAtomByElementStartEach=[]
        for i in range(len(NAtomByElement)):
            itemp=[]
            if not NAtomByElement[i]:
                pass
            else:
                for j in NAtomByElement[i]:
                    #print(j)
                    itemp.append( j-startindexlist[i]+1 )
            NAtomByElementStartEach.append(itemp)
        #print(self.elem)
        #print(NAtomByElementStartEach)
        for ielem2 in range(len(self.elem[0])):
            self.RemoveAtoms_NelemSequ( ielem2+1 , *NAtomByElementStartEach[ielem2] )
        return 0
####
