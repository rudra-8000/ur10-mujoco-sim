import numpy as np
from scipy.spatial import cKDTree
from build_pincopen import load_stl
from pathlib import Path
S=str(Path(__file__).resolve().parent.parent/'cad'/'PincOpen_Gripper'/'solidworks')+'/'
def sample(t,n=4000,seed=0):
    rng=np.random.default_rng(seed)
    a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    i=rng.choice(len(t),n,p=a/a.sum()); u=rng.random((n,2)); m=u.sum(1)>1; u[m]=1-u[m]
    return t[i,0]+u[:,:1]*(t[i,1]-t[i,0])+u[:,1:]*(t[i,2]-t[i,0])
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(P-pc).T@(Q-qc); U,_,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,d])@U.T; return R,qc-R@pc
def register(src,dst):
    P,Q=sample(src),sample(dst); tq=cKDTree(Q); best=None
    def pca(X):
        c=X.mean(0); w,v=np.linalg.eigh(np.cov((X-c).T)); return c,v
    cp,vp=pca(P); cq,vq=pca(Q)
    import itertools
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1,-1],repeat=3):
            M=vq[:,list(perm)]*np.array(sg); R=M@vp.T
            if np.linalg.det(R)<0: continue
            t=cq-R@cp; X=P@R.T+t
            for _ in range(12):
                d,i=tq.query(X); R2,t2=kabsch(P,Q[i]); R,t=R2,t2; X=P@R.T+t
            d,_=tq.query(X); e=np.sqrt((d**2).mean())
            if best is None or e<best[0]: best=(e,R,t)
    return best
if __name__=='__main__':
    print_v2=load_stl(S+'try3/print/CAMERA MOUNT_V2.STL'); asm=load_stl(S+'try2/print_3/TRY2_ASS - CAMERA MOUNT_V2-2.STL')
    e,R,t=register(print_v2,asm); print('rms',e); print(R.round(4)); print(t.round(3))
