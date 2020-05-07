# convex hull (Graham scan by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002
#from list of points, calculate typical parameter : size, radius....

#note myself: see johan jarnested illustration imaging across iological length scales.
#note myself: website  : support image sc ? 

# Just a compilaton of function to calculate parameters of clusters.


from __future__ import generators
import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

#list of points : [[cx[i],cy[i]] for i in range(0,len(cx))]  , return :  #minx miny maxx maxy
        #basically return Feret points, or points further apart, but seems faster than doing it yourself?
def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return pair


def diameter_min(Points):
    '''Given a list of 2d points, returns the pair that's least apart.'''
    diam,pair = min([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return pair


#Adrien Mau
#return angle (degree) that dominates x,y data. #note: dtheta is approx decimal precision ( dtheta = 0.01 or 0.02 gives same result.)
#note: return angle in negative  (if data has 20° angle: return -20° , so that rotate(getangle) will correct data.)
def get_angle( x, y , dtheta = 0.1):
    x = x - np.median(x)
    y = y - np.median(y)
    v1 = np.sum((x**2-y**2)) #may over- for too many x ? maybe use mean
    v2 = np.sum(x*y)
    nmax = int(round( -np.log(dtheta)/np.log(10)+0.5) )
    center = -45
    dc = 44
    for dt in 1/10**(np.arange(0,nmax)):
        T = np.arange(center-dc,center+dc, dt)
        c = np.cos(T* np.pi/180)
        s = np.sin(T* np.pi/180)
        theta_sol = T[ np.argmin( np.abs( v1*c*s + v2*(c**2-s**2) )) ]
        center = theta_sol
        dc = dt
    return theta_sol #in degree.


#return data of angle theta[radian] around (0,0).
def rotate(x,y, theta, mx=0 , my=0):
    if mx!=0:
        x = x - mx
    if my!=0:
        y = y - my
    c = np.cos(theta)
    s = np.sin(theta)
    newx = x*c-y*s
    newy = x*s + y*c
    if mx!=0:
        newx = newx + mx
    if my!=0:
        newy = newy + my
    return newx, newy

#return data of angle theta[degree] around (mx,my)
def rotated(x,y, theta_deg, mx=0 , my=0):
    if mx!=0:
        x = x - mx
    if my!=0:
        y = y - my
    c = np.cos(theta_deg*np.pi/180)
    s = np.sin(theta_deg*np.pi/180)
    newx = x*c-y*s
    newy = x*s + y*c
    if mx!=0:
        newx = newx + mx
    if my!=0:
        newy = newy + my
    return newx, newy


#get mean point distance to center, and std.
def get_radius( x, y , correct_circularity = False):
    x = x - np.median(x)
    y = y - np.median(y)
    if correct_circularity: #change elliptic shape to circular, but will modify radius measurement result .( if elliptic: two r should be taken in account..)
        x, y = rotate( x, y, get_angle(x,y) )
        y = y*np.std(x)/np.std(y)
    dr = np.hypot(x,y)
    return np.mean(dr), np.std(dr)


# Estimate center for a radial distributed point list with fit. (minimize dispersion of r)
# this consider that x and y somehow form a circle.
def center_fit(x,y):
    p0 = [ np.median(x), np.median(y) ]
    
    def diff(p,x,y):
        r = np.hypot(x-p[0],y-p[1]) # do square ?
        return r - np.mean(r)
    
    res = least_squares(diff, p0, args=(x,y), method='lm')
    pfit = res['x']
    return pfit[0], pfit[1]

#return center (minimizing radius dispersion), mean radius, and mean radius std.
def get_annularparams(x,y):
    cx, cy = center_fit(x,y)
    rs = np.hypot(x-cx,y-cy)
    rmean = np.mean(rs)
    sigma = np.std(rs)
    return cx, cy, rmean, sigma
    


#fit points (x,y,z) along a sphere with parameter: mx,my,mz; minimizing radius dispersion.
#good to find center on a SPHERE or unrotated ellipse (axis = x or y or z)
# generally better than median.
def center_fit3D(x,y,z, p0=None):
    if p0 is None:
        p0 = [ np.median(x), np.median(y), np.median(z) ]
    
    def diff3(p,x,y,z):
        r = (x-p[0])**2 + (y-p[1])**2 + (z-p[2])**2
#        r = np.sqrt(r) #good with or without it ...
        return r - np.mean(r)
    
    res = least_squares(diff3, p0, args=(x,y,z), method='lm')
    pfit = res['x']
    return pfit[0], pfit[1], pfit[2]



#fit points (x,y,z) along a sphere/3Dellipse  with parameter: rx,ry,rz
#xyz points should already be roughly centered in 0.
#Fit radius.
    # p is rx,ry,rz
def radial_fit3D(x,y,z, p0=None):
    if p0 is None:
        rmean = np.sqrt( np.median(x**2 + y**2 + z**2) ) #could be better: we take global mean r for rx ry rz initial.
        p0 = [ rmean, rmean, rmean]
    
    def diffr3(p,x,y,z):
        minimizer = 1 -  ( (x/p[0])**2 + (y/p[1])**2 + (z/p[2])**2 )
        return minimizer
    
    res = least_squares(diffr3, p0, args=(x,y,z), method='lm')
    pfit = res['x']
    return pfit[0], pfit[1], pfit[2]


def radialstd_fit3D(x,y,z, p0=None):
    r = np.sqrt(x**2 + y**2 + z**2)
    if p0 is None:
        rstd =  np.std( r ) #could be better: we take global mean r for rx ry rz initial.
        p0 = [ rstd, rstd, rstd ]    
        print('p0 is', p0)
        #for now p0 is not change during fit...
        
    theta = np.angle(x+y*1j)
    rxy = np.hypot(x,y)
    phi = np.angle(rxy+z*1j)   
    ct = np.cos(theta)
    st = np.sin(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
        
    def diff_std3(p,dr,theta,phi):
        sx,sy,sz = p0
        
        #note: this may not be conventional theta,phi spherical coordinate convention.
        #but sigma_eff has been checked to be okay this way:
        sigma_eff =  np.sqrt( (sx *st*cp)**2 +  (sy *ct*cp)**2 +  (sz *sp)**2 )
        
#        proba = np.exp( -0.5*dr**2/sigma_eff**2)/np.sqrt(2*np.pi) / sigma_eff
        
        proba = np.exp( -0.5*dr**2/sigma_eff**2)/np.sqrt(2*np.pi) / np.sqrt(sigma_eff)
        minimizer = 1 - proba
        
        return minimizer
    
    r_eff = np.sqrt( (rx *st*cp)**2 +  (ry *ct*cp)**2 +  (rz *sp)**2 )

    dr = r - r_eff
    res = least_squares(diff_std3, p0, args=(dr,theta,phi), method='lm')
    pfit = res['x']
    return pfit[0], pfit[1], pfit[2]
    

#from points defining a radius fit the radius std (basically it's like fitting a normal dist of points...)
#here it should be better to just do std. I use this functon to test the 1-prob fit.
def radialstd_fit1D(r, p0=None):
    if p0 is None:
        rstd =  np.std( r ) # actually best estimate i believe.
        p0 = [ rstd ]    
    
    dr = r - np.mean(r)
        
    def diff_std1(p,dr,fun):
        sr = p[0]
        
        proba = np.exp( -0.5*dr**2/sr**2) / np.sqrt(2*np.pi) / sr
        minimizer = 1 - np.log(proba)
        
#        proba = np.exp( -0.5*dr**2/sr**2) / np.sqrt(2*np.pi) / np.sqrt(sr)
#        minimizer = 1 - proba

        return minimizer

    res = least_squares(diff_std1, p0, args=(dr,1), method='lm')
    pfit = res['x']
    return pfit[0]




#generation random de points uniforme  en 3D :
def spherexyz(r=1,npoints=1000, zmax=None):
    theta = np.random.uniform(0,2*np.pi,npoints)
    if zmax is None:
        phi = np.arccos( 1 - 2*np.random.uniform(0,1,npoints))
    else:
        phi = np.arccos( 1 - 2*np.random.uniform( 0.5-zmax/2/r, 1, npoints))

    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return x,y,z



    
#generate x,y,z sphre (or 3Delliptic) +rotated data and try to find all parameters.
def test_xyzdata( nloc = 200, doplot=False):
    global x,y,z,xr,yr,zr,rx,ry,rz, mx,my,mz, xc,yc,zc, sx,sy,sz, r
    #GENERATING SPHERICAL DATA , eventually elliptic.
    mx0,my0,mz0 = 10,5,-3
    sx,sy,sz = 7,3,5
    rx = 20
    ry = 35
    rz = 80
    theta = np.pi/5
    phi = np.pi/6

    # x²/rx² + y²/ry² + z²/rz² = 1
    x,y,z = spherexyz(r=1, npoints=nloc)

    xr = x*rx + np.random.normal( 0,sx, nloc)
    yr = y*ry + np.random.normal( 0,sy, nloc)
    zr = z*rz + np.random.normal( 0,sz, nloc) 
    
    
#    plt.figure()
#    plt.hist( (xr/rx)**2 + (yr/ry)**2 + (zr/rz)**2 , bins=200) #centered along 1..
#    plt.hist( x**2 + y**2 + z**2 , bins=200)  #exactly 1
    
    xr,yr = rotate(xr,yr,theta)
    xr,zr = rotate(xr,zr,phi)
    xr = xr + mx0
    yr = yr + my0
    zr = zr + mz0

    
    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xr, yr, zr, s=0.5,depthshade=1,c=1+zr,cmap='summer')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('base data')
        dxyz = 50
        plt.scatter( [ mx0-dxyz,mx0+dxyz] , [ mz0-dxyz,mz0+dxyz] , [ mz0-dxyz,mz0+dxyz] , alpha=0)
        
    mx, my, mz = center_fit3D(xr, yr, zr)
    print('mx found:', round(mx,3),'median',round(np.median(xr),3), 'true:',mx0) #find somehow 90% of the correct value
    print('my found:', round(my,3),'median',round(np.median(yr),3), 'true:',my0)
    print('mz found:', round(mz,3),'median',round(np.median(zr),3), 'true:',mz0)


    mphi = get_angle( xr, zr,  0.01 )
    print('mtheta found:', round(mphi*np.pi/180,3),'true phi:',phi)
    xc,zc = rotated(xr,zr,mphi,  mx,mz)
#    xc,zc = rotated(xr,zr,mphi)
    mtheta = get_angle( xc, yr,  0.01 ) #with phi=0 works directly. Else you have to rerotate for phi.
    print('mphi found:', round(mtheta*np.pi/180,3),'true theta:',theta)
    xc, yc = rotated(xc,yr,mtheta,  mx,my)
#    xc, yc = rotated(xc,yr,mtheta)
    xc,yc,zc = xc-mx, yc-my, zc-mz

    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xc, yc, zc, s=0.5,depthshade=1,c=1+zc,cmap='summer')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('mean and angle corrected data')
        dxyz = 50
        plt.scatter( [ -dxyz,+dxyz] , [ -dxyz,+dxyz] , [ -dxyz,+dxyz] , alpha=0)
    
    #now we have elliptic (or spheric) data, aligned with axis x,y,z.
    #we know what was the angle and the mean, normally now they are 0 (for xc yc zc)
    #we still have to get eccentricity (rx,ry,rz) and dispersion.
    global mrx,mry,mrz
    mrx, mry, mrz = np.abs( radial_fit3D( xc,yc,zc ) ) #works well.
    global msx,msy,msz
    msx,msy,msz = np.abs( radialstd_fit3D(xc,yc,zc) )
    

    global thetas, rxy, phis, sigma_eff, proba, ct,st,cp,sp
    x,y,z = xc,yc,zc
    r = np.sqrt(x**2 + y**2 + z**2)
    rmean = np.mean(r)
    thetas = np.angle(x+y*1j) #between -pi and pi.
    rxy = np.hypot(x,y)
    phis = np.angle(rxy+z*1j)   #between -pi/2 and pi/2
    
    ct = np.cos(thetas)
    st = np.sin(thetas)
    sp = np.sin(phis)
    cp = np.cos(phis)
#    sigma_eff =  np.sqrt( (sx *st*cp)**2 +  (sy *ct*cp)**2 +  (sz *sp)**2 )
#    r_eff = np.sqrt( (mrx *st*cp)**2 +  (mry *ct*cp)**2 +  (mrz *sp)**2 )
#    
#    proba = np.exp( -0.5*(r-r_eff)**2/sigma_eff**2)/np.sqrt(2*np.pi) / sigma_eff
#    minimizer = 1 - proba
#    
#    if doplot:
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
##        ax.scatter(x,y,z, s=0.5,depthshade=1,c=phis,cmap='summer')
##        p = ax.scatter(x,y,z, s=0.5,depthshade=1,c= proba ,cmap='summer')
#        v=z>0; p = ax.scatter(x[v],y[v],z[v], s=0.5,depthshade=1,c= proba[v] ,cmap='summer')
#        
#        plt.xlabel('x')
#        plt.ylabel('y')
#        plt.title('angle measure data')
#        dxyz = 50
#        fig.colorbar(p)
#        ax.scatter( [ -dxyz,+dxyz] , [ -dxyz,+dxyz] , [ -dxyz,+dxyz] , alpha=0)
        
    #std is :  np.sqrt( np.mean( (x-x.mean())**2) )
    global sumz
#    sumz = np.sum( sp**2*(r)**2 ) /np.sum(np.abs(sp))  # r or zc...
    sp = sp
    sumz = np.sum( np.abs(sp)*(r)**2 ) /np.sum(np.abs(sp))  # r or zc...

    msz = np.sqrt( sumz)
    print('msz test',msz)
        
        
        
        


#test of r data  : finding std with proba MLE...
        # for some reason sigma is always slightly higher than correct value...
def test_rdata(nloc=200, doplot=False):
    global xr,sr,rmean,R,S,value
    rmean = 50
    sr = 15
    xr = np.random.normal( rmean, sr, nloc)
    
    if doplot:
        R = np.arange(rmean-8*sr, rmean+8*sr ) - rmean
        plt.figure()
        plt.hist(xr - rmean,bins=R)
        
        R = np.arange(0, rmean+8*sr ) - rmean
        form = nloc *  np.exp( -0.5*R**2/sr**2) / np.sqrt(2*np.pi) / sr
        plt.plot(R,form )
    
    my_sr = radialstd_fit1D(xr, p0=None)
    print('my sr is ',my_sr, 'and true sr was', sr ,' r std gives' , np.std(xr) )

    print('test: 1 minus formulas mean... \n')
    print('\t true sum:' , np.mean( 1 - np.exp( -0.5*R**2/sr**2) / np.sqrt(2*np.pi) / sr ))

    S = np.arange(0,30)
    value = []
    for asr in S:
        value.append( np.mean( 1 - np.exp( -0.5*(xr-rmean)**2/asr**2) / np.sqrt(2*np.pi) / asr ) )
    plt.figure()
    plt.plot(S,value)
    plt.title('Test of probability for different std...')
    
    
    
#test of smart coefficient for std... bof.
    #best is to take the part at axis.
    #OR : for each dtheta get std dev. Then plot along sx and sy... maybe this can worK.
def test_xydata(nloc=500, doplot=False):
    global theta,x,y, mystd, mr, tcount, tbins, tcountcum, valuestd, theta_v
    rx = 40
    ry = 40
    sx = 2
    sy = 4
    
    theta = np.random.uniform(0,2*np.pi, nloc)
    x = rx*np.cos(theta) + np.random.normal(0,sx,nloc) + 0
    y = ry*np.sin(theta) + np.random.normal(0,sy,nloc) + 0
    
#    if doplot:
#        plt.figure()
#        plt.title('xy distribution')
#        plt.scatter(x,y)
    
    mr = np.hypot(x-0,y-0)
    print('classical std r is', np.std(mr) )
    #same as :
    mystd = np.sqrt( np.sum( (mr - np.mean(mr))**2) / len(mr) )
    
#    xcoef = np.abs( np.cos(theta) )**8
    xcoef = ( np.abs(np.cos(theta)) > 0.95 ) #GOOD FOR FIRST ESTIMATION OF PARAMETER. ON DATA MAY REALLY DEPEND ON THE FEW POINTS THAT ARE AT THIS ANGLE...
    mystd_x = np.sqrt( np.sum( xcoef*(mr - np.mean(mr))**2) / np.sum(xcoef**2) )
    mystd_x = np.sqrt( np.sum( xcoef*(mr - rx)**2) / np.sum(xcoef**2) )
    print('my std rx is', mystd_x )
    
    if doplot:
        fig = plt.figure()
        plt.title('xy distribution')
        plt.scatter(x,y,c = xcoef)
        plt.colorbar()
        plt.title('coef for different std...')
        
#        plt.figure()
#        plt.scatter( theta,  np.sqrt( (mr - np.mean(mr))**2) )
#        plt.xlabel('theta')
#        plt.ylabel('some var')
    
    
    #Essai de fit sx et sy selon des intervalles d'angle dtheta. 
    #std = hypot( sx*cos(theta), sy*sin(theta) )
    #creation données:
    ts = np.argsort(theta)
    x,y,theta,mr = x[ts],y[ts],theta[ts],mr[ts]
    dtheta = np.pi/20 #note: si intervalle trop petit std diminue...
    thetacount = theta//dtheta
    tcount , tbins = np.histogram( thetacount , bins=np.arange(thetacount.min() , thetacount.max() ))
    tcountcum = np.append(0, np.cumsum(tcount))
    valuestd = []
    theta_v = tbins[:-1]*dtheta
    for tpos, theta_area in enumerate(theta_v):
        r_here = mr[ tcountcum[tpos]:tcountcum[tpos+1]]
        valuestd.append( np.std( r_here ) )
    
        
    if doplot:
        plt.figure()
        plt.title('along dtheta intervals')
        plt.plot( theta_v, valuestd, label='intervals')
#        plt.plot( theta, np., label='intervals')
        plt.plot( theta_v, np.hypot( sx*np.cos(theta_v),sy*np.sin(theta_v)) , label='wiht interv fnc')
        
        
        
if __name__ == "__main__":
    print('testing xyzdata.')
#    test_xyzdata( 3000 , doplot=1)
#    test_rdata( 500 , doplot=1)
    test_xydata( 300 , doplot=1)