########
import numpy
import copy

########
def CellMatToCellParameters_2dlist( mat ):
    Out=[]
    v1=numpy.array(mat[0], dtype=float)
    v2=numpy.array(mat[1], dtype=float)
    v3=numpy.array(mat[2], dtype=float)
    n1=numpy.linalg.norm(v1)
    n2=numpy.linalg.norm(v2)
    n3=numpy.linalg.norm(v3)
    Out.append( n1 )
    Out.append( n2 )
    Out.append( n3 )
    Out.append(  numpy.arccos( numpy.dot(v2/n2,v3/n3) )*180/numpy.pi  )
    Out.append(  numpy.arccos( numpy.dot(v3/n3,v1/n1) )*180/numpy.pi  )
    Out.append(  numpy.arccos( numpy.dot(v1/n1,v2/n2) )*180/numpy.pi  )
    return Out
########
def CellParamtersToCellMat_List( v ):
    Out=[]
    a , b , c  = v[0] , v[1] , v[2] 
    alphaM90 , betaM90 , gammaM90 = (v[3]-90)*numpy.pi/180 , (v[4]-90)*numpy.pi/180 , (v[5]-90)*numpy.pi/180
    Out.append([a,0.0,0.0])
    Out.append([-b*numpy.sin(gammaM90) , b*numpy.cos(gammaM90),0.0])
    v32=c*( numpy.cos(alphaM90+numpy.pi/2)-numpy.sin(gammaM90)*numpy.sin(betaM90) )/numpy.cos(gammaM90)    
    Out.append([-c*numpy.sin(betaM90), v32 , numpy.sqrt( numpy.square(c*numpy.cos(betaM90)) - numpy.square(v32) ) ])
    return numpy.array(Out)
########
def Deg2FitToXminYminA2A1A0_XlistYlist(xlist,ylist):
    x=numpy.array(xlist)
    y=numpy.array(ylist)
    z=numpy.polyfit(x,y,2)
    out=[]
    out.append( -z[1]/(2*z[0]) )
    out.append( z[2]-(z[1]*z[1])/(4*z[0]) )
    out=numpy.concatenate([out,z])
    return out
########
def Reorder_ListDict( l , d ):
    out = copy.deepcopy(l)
    for newindex, oldindex in d.items():
        out[newindex]=l[oldindex]
    return out
########
def RescaleShiftColumn_2dlistColScaleShift( mat , col, scale , shift ):
    out = [row0[:] for row0 in mat]
    for row in range(len(out)):
        out[row][col]= scale*out[row][col] + shift
    return out
########
