import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn.manifold as m

scurve_dat = np.genfromtxt('/home/max/Desktop/AMSC/HW4/ScurveData.csv', delimiter = ',')
(a,b) = scurve_dat.shape
colors = plt.cm.rainbow(np.linspace(0,1,a))
faceDat = np.genfromtxt('/home/max/Desktop/AMSC/HW4/FaceDat.csv', delimiter = ',')
colors2 = np.genfromtxt('/home/max/Desktop/AMSC/HW4/Colors.csv', delimiter = ',')
(a2,b2) = faceDat.shape
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(scurve_dat[:,0],scurve_dat[:,1],scurve_dat[:,2], color=colors)

doPCA = 1
doIsomap = 0
doLLE = 0
doTSNE = 0
doDiffusion = 0


def PCA(dat, k):
    (U,S,Vt) = np.linalg.svd(dat)
    return U[:,:k]@np.diag(S[:k])
    
def testPCA():   
    projScurve = PCA(scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projScurve[:,0],projScurve[:,1],color=colors)
    
    sig = 0.3
    noisy_scurve_dat = scurve_dat + np.random.normal(scale=sig, size=(a,b))
    projNScurve = PCA(noisy_scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projNScurve[:,0],projNScurve[:,1],color=colors)
    
    projFaces = PCA(faceDat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projFaces[:,0],projFaces[:,1],color=colors2)
    
def Isomap(dat, k):
    embedding = m.Isomap(n_components = k, path_method='D', n_neighbors=20)
    return embedding.fit_transform(dat)
    
def testIsomap():
    projScurve = Isomap(scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projScurve[:,0],projScurve[:,1],color=colors)
        
    sig = 0.5
    noisy_scurve_dat = scurve_dat + np.random.normal(scale=sig, size=(a,b))
    projNScurve = Isomap(noisy_scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projNScurve[:,0],projNScurve[:,1],color=colors)
    
    projFaces = Isomap(faceDat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projFaces[:,0],projFaces[:,1],color=colors2)
    
def LLE(dat, k):
    embedding = m.LocallyLinearEmbedding(n_components = k, reg=0.001, n_neighbors = 20)
    return embedding.fit_transform(dat)
    
def testLLE():
    projScurve = LLE(scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projScurve[:,0],projScurve[:,1],color=colors)
        
    sig = 0.1
    noisy_scurve_dat = scurve_dat + np.random.normal(scale=sig, size=(a,b))
    projNScurve = LLE(noisy_scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projNScurve[:,0],projNScurve[:,1],color=colors)
    
    projFaces = LLE(faceDat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projFaces[:,0],projFaces[:,1],color=colors2)
    
def TSNE(dat, k):
    embedding = m.TSNE(n_components = k, perplexity=30)
    return embedding.fit_transform(dat)
    
def testTSNE():
    projScurve = TSNE(scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projScurve[:,0],projScurve[:,1],color=colors)
        
    sig = 0.6
    noisy_scurve_dat = scurve_dat + np.random.normal(scale=sig, size=(a,b))
    projNScurve = TSNE(noisy_scurve_dat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projNScurve[:,0],projNScurve[:,1],color=colors)
    
    projFaces = TSNE(faceDat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projFaces[:,0],projFaces[:,1],color=colors2)
    
def DiffusionMap(dat, dim, const):
    (A,B) = dat.shape
    delta = np.array([[np.sum((dat[i]-dat[j])**2) for j in range(A)] for i in range(A)])
    rowmins = np.array([np.min(np.append(delta[i,:i],delta[i,i+1:])) for i in range(A)])
    eps = const*np.mean(rowmins)
    k = np.exp(-delta/eps)
    q = np.sum(k, axis=1)
    pi = q/np.sum(q)
    Pi = np.diag(pi)
    Q = np.diag(q)
    P = np.linalg.solve(Q,k)
    lam,r = np.linalg.eig(P)
    lam = np.real(lam)
    r = np.real(r)
    s = np.transpose(r)@Pi@r
    R = np.array([r[:,i]/np.sqrt(s[i,i]) for i in range(A)])
    t = np.ceil(np.log(0.1)/np.log(np.abs(lam[2]/lam[1])))
    psi = np.zeros((A,dim))
    for j in range(dim):
        psi[:,j] = (lam[j+1]**t)*R[j+1]
    return psi
    
def testDiffusion():
    projScurve = DiffusionMap(scurve_dat,2,2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projScurve[:,0],projScurve[:,1],color=colors)
        
    sig = 0.2
    noisy_scurve_dat = scurve_dat + np.random.normal(scale=sig, size=(a,b))
    projNScurve = DiffusionMap(noisy_scurve_dat,2,2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projNScurve[:,0],projNScurve[:,1],color=colors)  
    
    projFaces = DiffusionMap(faceDat,2,2)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(projFaces[:,0],projFaces[:,1],projFaces[:,2],color=colors2)

    

if doPCA: testPCA()
if doIsomap: testIsomap()
if doLLE: testLLE()
if doTSNE: testTSNE()
if doDiffusion: testDiffusion()