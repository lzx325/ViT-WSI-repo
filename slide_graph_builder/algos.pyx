# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3
import cython
from cython.parallel cimport prange, parallel
cimport numpy as cnp
import numpy as np

cdef extern from "limits.h":
    cdef long LONG_MAX
cdef extern from "float.h":
    cdef float FLT_EPSILON 
    cdef double DBL_EPSILON
cdef extern from "math.h":
    cdef float INFINITY

cdef class FloydWarshallPred(object):
    cdef:
        readonly float[:,::1] adj
        readonly size_t n
        readonly float[:,::1] M
        readonly long[:,::1] Pred
    def __init__(self,float[:,::1] adj):
        self.adj=adj
        assert adj.shape[0]==adj.shape[1]
        self.n=adj.shape[0]
    cdef initialize_M_Pred(self):
        cdef size_t i=0,j=0
        cdef float[:,::1] M =self.adj.copy()
        cdef long[:,::1] Pred = np.full((self.n,self.n),-1,dtype=np.int64)
    
        for i in range(self.n):
            for j in range(self.n):
                if M[i,j]==0:
                    M[i,j]=INFINITY
                if i==j:
                    M[i,j]=0

        for i in range(self.n):
            for j in range(self.n):
                if M[i,j]<INFINITY:
                    Pred[i,j]=i

        return M,Pred
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void floyd_warshall(self):
        cdef size_t i=0,j=0,k=0
        cdef:
            float[:,::1] M
            long[:,::1] Pred

        M,Pred=self.initialize_M_Pred()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if M[i,k]+M[k,j]<M[i,j]:
                        M[i,j]=M[i,k]+M[k,j]
                        Pred[i,j]=Pred[k,j]

        self.M=M
        self.Pred=Pred
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void floyd_warshall_parallel(self):
        cdef size_t i=0,j=0,k=0
        cdef:
            float[:,::1] M
            long[:,::1] Pred
        M,Pred=self.initialize_M_Pred()
        for k in range(self.n):
            for i in prange(self.n,nogil=True,schedule="dynamic"):
                for j in range(self.n):
                    if M[i,k]+M[k,j]<M[i,j]:
                        M[i,j]=M[i,k]+M[k,j]
                        Pred[i,j]=Pred[k,j]

        self.M=M
        self.Pred=Pred
    
    cpdef void floyd_warshall_eager_update(self):
        cdef size_t i=0,j=0,k=0
        cdef:
            float[:,::1] M
            long[:,::1] Pred
        M,Pred=self.initialize_M_Pred()

        for k in range(self.n):
            
            for i in range(self.n):
                for j in range(self.n):
                    if M[i,k]+M[k,j]<=M[i,j] and M[i,k]<INFINITY and M[k,j]<INFINITY: # eager update
                        if i==j: # prevent un-necessary updates
                            continue
                        elif k==i or k==j: # prevent un-necessary updates
                            continue
                        else:
                            M[i,j]=M[i,k]+M[k,j]
                            Pred[i,j]=Pred[k,j]       
        self.M=M
        self.Pred=Pred
    
    cpdef list get_path(self,long i, long j):
        cdef long curr=j, prev=self.Pred[i,j]
        cdef list path=[j]
        while prev!=curr:
            curr=prev
            prev=self.Pred[i,curr]
            path.append(curr)
        return list(reversed(path))


cdef class FloydWarshallSucc(object):
    cdef:
        readonly float[:,::1] adj
        readonly size_t n
        readonly float[:,::1] M
        readonly long[:,::1] Succ
    def __init__(self,float[:,::1] adj):
        self.adj=adj
        assert adj.shape[0]==adj.shape[1]
        self.n=adj.shape[0]
    cdef initialize_M_Succ(self):
        cdef size_t i=0,j=0
        cdef float[:,::1] M =self.adj.copy()
        cdef long[:,::1] Succ = np.full((self.n,self.n),-1,dtype=np.int64)
    
        for i in range(self.n):
            for j in range(self.n):
                if M[i,j]==0:
                    M[i,j]=INFINITY
                if i==j:
                    M[i,j]=0

        for i in range(self.n):
            for j in range(self.n):
                if M[i,j]<INFINITY:
                    Succ[i,j]=j

        return M,Succ

    cpdef void floyd_warshall(self):
        cdef size_t i=0,j=0,k=0
        M,Succ=self.initialize_M_Succ()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if M[i,k]+M[k,j]<M[i,j]:
                        M[i,j]=M[i,k]+M[k,j]
                        Succ[i,j]=Succ[i,k]

        self.M=M
        self.Succ=Succ

    cpdef list get_path(self,long i, long j):
        cdef long curr=i, succ=self.Succ[i,j]
        cdef list path=[i]
        while succ!=curr:
            curr=succ
            succ=self.Succ[curr,j]
            path.append(curr)
        return path

cdef class FloydWarshallMid(object):
    cdef:
        readonly float[:,::1] adj
        readonly size_t n
        readonly float[:,::1] M
        readonly long[:,::1] Mid
    def __init__(self,float[:,::1] adj):
        self.adj=adj
        assert adj.shape[0]==adj.shape[1]
        self.n=adj.shape[0]
    cdef initialize_M_Mid(self):
        cdef size_t i=0,j=0
        cdef float[:,::1] M =self.adj.copy()
        cdef long[:,::1] Mid = np.full((self.n,self.n),-1,dtype=np.int64) # -1 means 'no nodes in the middle'
    
        for i in range(self.n):
            for j in range(self.n):
                if M[i,j]==0:
                    M[i,j]=INFINITY
                if i==j:
                    M[i,j]=0

        return M,Mid

    cpdef void floyd_warshall(self):
        cdef size_t i=0,j=0,k=0
        M,Mid=self.initialize_M_Mid()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if M[i,k]+M[k,j]<M[i,j]:
                        M[i,j]=M[i,k]+M[k,j]
                        Mid[i,j]=k # difference here

        self.M=M
        self.Mid=Mid

    cpdef list get_path(self,long i, long j):
        cdef long mid=self.Mid[i,j]
        if mid==-1:
            if i==j:
                return [i]
            else:
                return [i,j]
        else:
            return self.get_path(i,mid)[:-1]+self.get_path(mid,j)


def floyd_warshall_original(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    # cdef cnp.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    # cdef cnp.ndarray[long, ndim=2, mode='c'] path = np.zeros([n, n], dtype=np.int64)
    cdef long[:,:] M = adj_mat_copy
    cdef long[:,:] path = np.zeros([n, n], dtype=np.int64)
    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    print("begin floyd algorithm")
    for k in prange(n,nogil=True,num_threads=10):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path