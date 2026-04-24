########
import re
import numpy

########
def DeleteByStr_ListString( filelist , string ):
    fout=[]
    for line in filelist:
        if not re.search(string, line ):
            fout.append(line)
    return fout
########
def FileTo2dListByPartition_FileList( filepath, partition ):
    FIn=open(filepath, "r")
    Out=[]
    nPart=0
    listInsidePart=[]
    for nline, line in enumerate(FIn):
        if partition[nPart] <= nline < partition[nPart+1]:
            listInsidePart.append(line.strip())
        if nline==partition[nPart+1]-1:
            Out.append( listInsidePart )
            listInsidePart=[]
            nPart=nPart+1
        if nPart==len(partition)-1:
            FIn.close()
            return Out
########
def FileToListBySearch_File2String2Shift( filepath , startstr , endstr , startshift , endshift ):
    NStart=NLineFindGive0IfNotFound_FileString( filepath , startstr )
    NEnd=NLineFind_FileString( filepath , endstr )
    NStartShift=NStart+startshift
    NEndShift=NEnd+endshift+1
    return FileToList_FileStartEnd( filepath , NStartShift , NEndShift )
########
def FileToList_FileStartEnd( filepath, start , end ):
    FIn=open(filepath, "r")
    Out=[]
    for n, line in enumerate(FIn):
        if start <= n:
            Out.append(line.strip())
        if n==end-1:
            FIn.close()
            return Out
########
def GetColumns_MatStartEnd( mat, start , end ):
    return numpy.matrix( mat )[ : , start:end  ].tolist()
########
def GrepFile_FileString( filepath , string ):
    fin=open(filepath, "r")
    fout=[]
    for line in fin:
        if re.search(string, line ):
            fout.append(line.strip())
    fin.close()
    return fout
########
def GrepList_ListString( llist , string ):
    out=[]
    for line in llist:
        if re.search(string, line ):
            out.append(line.strip())
    return out
########
def ListTo2dList_List( llist ):
    Out=[]
    for line in llist:
        Out.append( line.split() )
    return Out
########
def ListToDictGiveIndex_List(l):
    d={}
    for i in range(len(l)):
        d[l[i]]=i
    return d
########
def MatStringsToFloats_Mat( mat ):
    return numpy.matrix(mat).astype( numpy.float )
########
def NLineFind_FileString( filepath , string ):
    FIn=open(filepath, "r")
    for n, line in enumerate(FIn):
        if re.search(string, line ):
            FIn.close()
            return n
    FIn=open(filepath, "r")
    n=len(FIn.readlines())-1
    FIn.close()
    return n
    
########
def NLineFindGive0IfNotFound_FileString( filepath , string ):
    FIn=open(filepath, "r")
    for n, line in enumerate(FIn):
        if re.search(string, line ):
            FIn.close()
            return n
    return 0
    
########
def Print2dList_2dlist( llist ):
    for line in llist:
        print(*line  )    
    #for line in llist:
    #    for field in line:
    #        print( field , end=' ', file=fileout )
    #    print('' , file=fileout)
########
