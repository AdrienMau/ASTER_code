# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:35:58 2019
Last modified - 9 September 2019.
    FR//
        -Ajouts : densité, nphoton, evolution frame/temporelle, conditions sur le process.
        -Ajout: chargement npz
    ENG::
        Added: Density, nphoton, evolution of frame/temporal. Conditions on processing.
        Added: loading of npz files.
    
    Ajout 20 Avril 2020: res et res2 sauvegardés.
    
- Opens X Y ... data of NanoRulers (in data0, drift). Made to open csv files.
- Find cluster of close points (dmin, dmax)
 + Throw away unvalid clusters (targets, noise, bad nanorulers ...)
- Find orientation of points (3 points here)
- Fit localisations to get a feeling of the resolution. - todo
- Save Results and Related Parameters (if CROP => save in a specific folder )

+ You may also open already analyzed data results (doload==2)

Change "control board" values to do different processes.
For ex. Change "manip" value to process different data.
    
@author: Adrien MAU / ISMO & Abbelight

Note: pour le sub cluster on pourrait utiliser autre chose que dbscan (type Kmean ?) vu qu'on sait qu'on doit trouver 3 echantillons.
            Une Gaussian Mixture serait ptet optimale. => YES c'est chouette.

Note: mean shift pourrait etre cool au lieu de dbscan dans la segmentation des nanorulers. => probablement balec.

Note; il faudrait filtrer les points aberrants. (seuls) => C'est fait!
Note; pour GMM les sigma x et y sont confondus pour l'instant (on les moyenne). GMM est pratique.
        GMM trouve sigma entre 6 et 13 (mode full sur la covariance)
Note: il serait bien de simuler des nanorulers et voir comment ce code filtre et quantifie. On a probablement avec nos filtres de distance un passe bande (ou passe bas en tout cas) sur la résolution.


Version 1.2


- ajout pour le sub clustering de method_sub , on peut choisir entre dbscan, Kmean et GMM
    - KMean et GMM trouvera toujours des candidats valide (bon nombre de cluster)
    - GMM : calcul de l'id (sub cluster 1  2 ou 3 et obtention mean, var...)
- Ajout de filtres (distance entre cluster, excentré ou non ... )



"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys, os, time
import pandas, scipy
from scipy.interpolate import griddata


import sklearn
from sklearn import cluster, mixture
import imageio

""" Global parameters """

npzfolder = "'EPISpy_res/"  #my notation for Python localized data.
npzfile = 'save.npz'    #my notation for Python localized data.

colorsList = ['c','r','y']
CustomCmap = matplotlib.colors.ListedColormap(colorsList)
                
FCOL = 1 #for csv file reading
XCOL = 2 #for csv file reading
YCOL = 3 #for csv file reading
NPHCOL = 5
SCOL = 4 #for csv file : sigma colum number  (in my 2D data it is same sigma for x and y direction.)

npzFCOL = 0
npzXCOL = 1 #for npz file reading
npzYCOL = 2 #for npz file reading
npzSCOL = [3,4] #for npz file - is 3 and 4

#str_data_list = [x,y,]

PIX = 108 #pixel size (to convert drift to x,y,sx,sy same unit (nm) )
PIXCOR = 60/64   #applied to x and y at Opening  (pour Pierre mettre à 1)

NANOR_DIST = 50   #typical distance in which we have three spots of nanoruler. - typ 30-50
NANOR_SUB_DIST = 15   #typical sub distance between nanorulers. - typ 15
N_SUB_SPOT = 3   #expected number of clusters for nanoruler. - Three spot nanoruler.

#Last filters :
MAX_EXC = 18  #max excentricity (here in nm) from center to nanoruler edges. - typ 18
MIN_DSUB = 20 # min distance between two sub cluster - typ 25
MAX_DSUB = 60 # max distance between two sub cluster - typ 65
MIN_DSUBTOT = 60 #min ditance between edge clusters - typ 66
MAX_DSUBTOT = 105 #max ditance between edge clusters - typ 110


#creation of IMGS:
imgpixel = 5
# imgsshape =( np.array([140,140]) / imgpixel ).astype('int')
imgsshape =( np.array([140,140]) ).astype('int')



def load_npz( datafile ):
    a = np.load( datafile )
    ad = a['data']
    frame = ad[:, npzFCOL]
    x = ad[:,npzXCOL]
    y = ad[:,npzYCOL]
    sigma = np.mean( ad[:,npzSCOL], axis=1)
    nphoton = a['Ioff']
    frame = a['refi']
    #also keys:  rmse, I , bgvar
    return frame, x, y, sigma, nphoton

    
# Load csv data and eventually apply drift ( for abbelight files drift is already applied)
def load_csv( datafile, driftfile=None, applydrift=0 ):
    print('Loading data ... ')
    df = pandas.read_csv(datafile, sep=',')
    data = df.values
    if not (driftfile is None):
        drift = pandas.read_csv( driftfile, sep=',' )
        datadrift = drift.values
        if applydrift:
            print('\t Applying drift to loaded data')
            cumcount = 0
            fcount = (datadrift[:,0]).astype('int')
            for pos, nmolf in enumerate( fcount ):
                fdriftxy = datadrift[pos,[1,2]]*PIX
                data[cumcount:cumcount+nmolf,[XCOL,YCOL]] = data[cumcount:cumcount+nmolf,[XCOL,YCOL]] + fdriftxy 
                cumcount = cumcount + nmolf
        return data, datadrift
    return data

def crop_array(data, C):
    x = data[:,XCOL]
    y = data[:,YCOL]
    return (x>C[0])*(x<C[0]+C[2])*(y>C[1])*(y<C[1]+C[3])

#Return mean sigma from an input covariance matrix, with number of rows equal to N_SUB_SPOT
def get_sigmas(cov):
    if cov.ndim==2:
        return np.tile( np.sqrt(np.trace(cov/3)) , 3 )
    else:
        return [ np.sqrt(  np.trace(cov[i])/N_SUB_SPOT) for i in range(0,N_SUB_SPOT) ]

        

#return x,y of angle theta, around center ox,oy
def rotate_points(x,y,theta,ox=0,oy=0):
    dx = (x - ox)
    dy = (y - oy)
    fx = ox + np.cos(theta) * dx - np.sin(theta) * dy
    fy = oy + np.sin(theta) * dx + np.cos(theta) * dy
    return fx, fy

#return image/histogram of the mean or median spatial value
    #input is x y values: positions x y and values 
    #nanzero : if 1 => nan are changed for zeros
    #note: x y may be reversed :o
def vhist2D(x,y,values, method='mean', nb=(50,50), rng=None, nanzero = 0):
    img = scipy.stats.binned_statistic_2d(x, y, values, statistic=method, bins=nb, range=rng, expand_binnumbers=False)[0]
    if nanzero:
        img[ np.isnan(img)] = 0 
    return img


#Return histogram count image.
def loc2imgC(x,y,shape_=None,binning=1,binningy=None):
    if binningy is None:
        binningy = binning
    if shape_ is None:
        sx=int(0.5 + x.max())+1
        sy=int(0.5 + y.max())+1
    else:
        sx = shape_[1]
        sy = shape_[0]
    spx=int((sx/binning - 0.5)); 
    spy=int((sy/binningy - 0.5));
    IMG = np.histogram2d(y,x,bins=(spy+1,spx+1),range=((0,(spy+1)*binningy),(0,(spx+1)*binning)))
    return IMG[0].astype('uint16')

#2D interpolation of an image from scatter data. (x y value)
# Input: value-x value-y and value at positions. Binning of image.
    # method of griddata: linear, nearest, cubic.
def interpolate_img( vx, vy, value, binning=1, method='nearest'):
    vx, vy = vx//binning, vy//binning
    nx, ny = vx.max(), vy.max()
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    return griddata((vy, vx), value, (Y, X), method=method)
"""
# =============================================================================
#  CONTROL BOARD
# =============================================================================
"""
DO_NOT_PROCESS = False  # Process. (note: if doload==2 this is put to False and load results)

#DO_NOT_PROCESS = True    #  Just plot. - comment this if needed.
                          #will set to False all processes.- only final plot is left.

"""
# =============================================================================
#  CONTROL BOARD - LOADING and SAVING
# =============================================================================
"""

plt.close('all')

doload=1   #2 : load result data. 1:load coordinate SMLM data.
if doload==2:
    DO_NOT_PROCESS = True    #  just plot. - comment this if needed.


#Data path: Change manip value to process different data.
manip = 2 #0 for gaussian, 1 for ASTER small, 2 for ASTER wide.

M = ""
files = ["gaussian/","ASTER-small/", "ASTER-wide/"]
main = M + files[ manip ] + "180/"
print('Chosing file: '  + files[ manip%10 ] + ' - \n ' + main + '\n' )


file = "CoordTable2D.csv"
drift_file = "DriftTable.csv"


save_resultdata = 1
save_result_folder = main + "results/"
if not os.path.isdir( save_result_folder ):
    os.mkdir( save_result_folder )
save_measurements = 1  #save median resolution and its deviation, same for angle, excentricity, mX, mY ...


docrop=0 #crop loaded data0 with CROP -and- CROP MX MY result before plot.
CROP = None    #x0 y0 dx dy

shift_nearzero = 1  #shift loaded data after crop so that the minimum is at (PIX,PIX)
plot_load = 0  # Scatter plot of data after load, crop, and shift (Heavy !)


"""
# =============================================================================
#  CONTROL BOARD - PROCESSES
# =============================================================================
"""

""" FIRST FILTER - lonesome points """
filter_lonesome_guys_with_dbscan_before_anything = 1
kill_everyone_with_less_than_this_number_of_friends = 4 #we will only keep data with this number of friend (-1?) in around area
area_for_valid_friends = 15   # define area of friendliness. People farther away don't like you.
plt_filterlonesome = 0  #HEAVY: scatter plot of filter lonesome before and after. (use CROP to not f*ck up your memory)

""" Separate different nanorulers """
do_association = 1
method=12    #1 or 12 for sklearn dbscan   (( 2 for sklearn meanshift.   3 ELKI 4 self dbscan ))
dmax = NANOR_DIST  #maximum distance to be associated. (radius or diameter )  - 50 typ (= NANOR_DIST)
minloc = 10  #minimum number of localization to be in a cluster (chosen as a valid fluorophore) - typ 7-15
show_idassoc = 0  #scatter plot with id number annotation. (heavy+.)

""" Sort and keep nanorulers only """
checkid = 1  #sort along id order, and throw away id=-1   (Needed for correct sub association as it expects sorted data.)
method_id = 1

""" Associate nano sites/spots. (for each nanoruler) """
do_sub_association = 1
method_subid = 3  #1 for dbscan, 2 for k-mean  #3 for Gaussian Mixture Model
plotsub = 0   #scatter plot of sub cluster id association. ~     # if set to 2: shift data by minimum.
plotsub2 = 0  #scatter color plot of sub cluster and Center of each subcluster + line of approximate direction
plotsub3 = 0  #scatter plot of rotated and validated data, and histogram profile.     # if set to 2: shift data by minimum.
printsub = 0   #print sub cluster treatment advancement. - angle, excentrism, number, and filter_gmm_prob effect.s
print_sub_invalidity = 0  #when sub cluster is rejected, print causes (usually it is too few loc.)

""" PLOT """
resmin = 6 #200200: typ 6-12? else typ 5-11
resmax = 14
nph_vmax = 30000  #for map
nph_vmaxhist = 30000 
map_ticks = True  #add µm ticks to res and photon map.

plotfinal = 1   #plot final data in general. --- detail below
plot_hist = 1   #histogram of results.
plot_map = 0  #map of resolution and scatter map
plot_graph = 0 #plot res on scatter along axis
plot_angle = 0
plot_density = 0 #plot Global density map (of localisation) and of validated Nanorulers. (note: global also takes in account nanoruler that were incorrects)
plot_nph = 0
plot_dataframe = 0 #plot evolution of different datas along frame (global)

plot_nanor = 1  #plot all found nano rulers (or 100 bests...)



save_nanor = 0 #save all found nano rulers in separate imgs.

gmm_covtype = 'tied' #full tied ... of covariance for GMM clustering. (si les gaussiennes des sub cluster sont liées, ou non)
plot_ggm = 0   #plot contour area probability ( needs method_subid==3)
#filter_ggm_prob = 12  #if 0 : does nothing, else filter out points that have a log prob > typ. 12
filter_ggm_prob = 100


minsubloc = 4          #for dbscan=> min number of points in a cloud/spot to be a valid cluster. (7-10 is good) 
                        # for kmean or GMM: min number of points in all 3 cluster to be valid. (filter-like) , depends on your data but 7-12 is good

N = None #if none, take all cluster for sub associations. Or chose specific cluster number yourself.
#N = np.array([12])
#N = np.array([10,11,12,13,14])
#N = np.arange(0,28)

nanomax = None  #Limit final plot to this number of nanoruler (plot_nanor = 1)
#nanomax = 977  #Limit final plot to this number of nanoruler 
     # To use if for ex. you want to compare with another experiment with a same number of nanoruler.


    
    
plotcrop=0 #in not processing ,is set to docrop value
if DO_NOT_PROCESS:
    print('NOT PROCESSING ANYTHING !')
    if doload==2:
        newdoload = 2
    else:
        newdoload = 0
    doload, filter_lonesome_guys_with_dbscan_before_anything, do_association, checkid, do_sub_association = newdoload,0,0,0,0
    plotcrop = docrop
    docrop = -1  # => not renewing data with data0.
    
else:
    print('PROCESSING')
"""
# =============================================================================
#  PROCESSES
# =============================================================================
"""

#Load
t00 =  time.perf_counter()
if doload==1:
    t0 = time.perf_counter()   
    
    if main.endswith( npzfolder ):
        print('Loading npz data, and drift from csv')
        frame,x,y,sigma, nph = load_npz( main + npzfile )
        
        main0 = main[:-len(npzfolder)+1]
        (datafun, drift) = load_csv( main+file, main+drift_file )
        del(datafun)
        fcount = drift[:,0]
        print('TODO : APPLY DRIFT..')
#        dx = [ np.repeat(drift[i,1])]
#        x = x - drift[:,1]*PIX
#        y = y - drift[:,2]*PIX
        
        arr = np.arange(0,len(frame))
        data0 = np.vstack(( arr,frame,x,y,sigma,nph )).T  #according to FCOL XCOL.. etc
    else:
        print('Loading in' + main + file)
        (data0, drift) = load_csv( main+file, main+drift_file )
        #fcount = drift[:,0]   #nmol per frame, (note: is not modified while docrop)
        driftx = drift[:,1]*PIX #drift in nm
        drifty = drift[:,2]*PIX #drift in nm
        
    fstart = int( np.min( data0[:,FCOL]) )  #first frame number
    uframe = np.arange(fstart, fstart+len(driftx) ,dtype='uint')
    dt_load = time.perf_counter() - t0 

#crop
if doload==1:
    if docrop==1 and not(CROP is None):
        print('Cropping with',CROP)
        valid = crop_array(data0,CROP)
        data = data0[valid,:]
        if not(any(valid == True)):
            print(' SELECTED CROPPING THROW AWAY ALL DATA ! ')
            plt.figure('Cropping is inadequate :' + str(CROP) )
            plt.title('Cropping is inadequate :' + str(CROP))
            plt.hist( data0[:,XCOL], bins=200, alpha=0.6, label='x')
            plt.hist( data0[:,YCOL], bins=200, alpha=0.6, label='y')
            sys.exit( 'Cropping throw away all data : ' + str(CROP) )
        
        main = main + 'CROP-' + str(CROP) + '/'
        if not os.path.isdir( main ):
            print('Creating CROP res folder :' + main )
            os.mkdir( main )
        
    elif docrop!=-1:
        data=data0.copy()

#Shift et correction de taille pixel.
if doload==1:
    x = data[:,XCOL]
    y = data[:,YCOL]
    nph = data[:,NPHCOL]
    sigma = data[:,SCOL]
    if PIXCOR!=1:
        print('Applying pixel correction of ratio ' + str(PIXCOR) )
        x = x*PIXCOR
        y = y*PIXCOR
        print('\t Now pixel of x y data is :' + str(PIX*PIXCOR) )
        #note that drift is still on PIX. As it is already applied i don't have to care.
    
    if shift_nearzero:
        print('Shifting Near zero')
        x = x - x.min() + PIX
        y = y - y.min() + PIX
    frame = data[:,FCOL]
    if plot_load:
        plt.scatter(x,y,s=20)

if filter_lonesome_guys_with_dbscan_before_anything:
    t0 = time.perf_counter() 
    print('\nFiltering lonesome guys (Area='+ str(area_for_valid_friends) + ' Nelem_min= '+ str(kill_everyone_with_less_than_this_number_of_friends) + ')')
    C = cluster.DBSCAN(eps=area_for_valid_friends, min_samples=kill_everyone_with_less_than_this_number_of_friends, n_jobs=1)
    C.fit( np.transpose( np.vstack((x,y)) ) )
    res0 = C.labels_
    
    valid = (res0!=-1)
    if plt_filterlonesome:
        plt.figure('Filter lonesome effect')
#        plt.scatter(x,y,s=20, marker='o', label='before filter')
        plt.scatter(x[~valid], y[~valid], s=10, marker='o', label='filtered out')
        plt.scatter(x[valid], y[valid], s=10, marker='+', label='Kept')
        plt.title('Lonesome filter, Area='+ str(area_for_valid_friends) + ' NelemMin= '+ str(kill_everyone_with_less_than_this_number_of_friends))
        plt.legend()
        
    x=x[valid]
    y=y[valid]
    nph=nph[valid]
    sigma=sigma[valid]
    frame = frame[valid]
    print('\t ' + str(np.sum(valid)) + ' / ' + str(len(valid)) + ' points kept')
    dt_filter_lonesome = time.perf_counter() - t0 
    

if do_association:
    t0 = time.perf_counter() 
    print('\nAssociating ID')
    t0 = time.perf_counter()
    if method==1:
        print('\t Using sklearn Dbscan with dist=',dmax,' and min loc=',minloc)
        C = cluster.DBSCAN(eps=dmax, min_samples=minloc, n_jobs=1)
        C.fit( np.transpose( np.vstack((x,y)) ) )
        res = C.labels_
    if method==12:
        print('\t Using sklearn Dbscan-ball_tree, dist=',dmax,' and min loc=',minloc)
        C = cluster.DBSCAN(eps=dmax, min_samples=minloc, algorithm='ball_tree')
        C.fit( np.transpose( np.vstack((x,y)) ) )
        res = C.labels_

    print( '\t ' + str(res.max()+1) + ' clusters found' )
    dt_association = time.perf_counter() - t0 
    
    if show_idassoc:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Scatter- cluster ID association')
        ax.scatter(x,y, s=0.2, c=res,cmap='prism')
        for i, txt in enumerate(res):
            ax.annotate(str(txt), (x[i], y[i]))
    
        
        
if checkid:
    t0 = time.perf_counter() 
    print('\nSorting in id number, and throwing -1s')
    test = res.copy()
    rsort = res.argsort()
    
    resint = res[rsort]
    not_alone = (resint!=-1)
    rsort = rsort[not_alone]
    print('\t ' + str(np.sum(not_alone)) + ' / ' + str(len(not_alone)) + ' points kept')
    x=x[rsort]
    y=y[rsort]
    nph=nph[rsort]
    sigma=sigma[rsort]
    frame=frame[rsort]
    res=res[rsort]
    del(resint)
    
    idcount , uid = np.histogram(res,bins=np.arange(0,res.max()+ 2)) #ok
    cumcount = np.cumsum(np.concatenate(([0],idcount)))      #note : idcount[N] = A   <=>  np.sum(res==N) = A
    if idcount.min()<minloc:
        invalid = idcount<minloc
        nincorrect = str(np.sum( invalid ))
        print('\t Cluster fucked up at some point, ' + nincorrect+' clusters have nelem<' + str(minloc) )
        print('\t Corresponding to ids ' +  str(uid[:-1][invalid]) )
    dt_checkid = time.perf_counter() - t0 

if 'uid' in locals():
    if N is None:
        N = uid[:-1]


if do_sub_association: 
    print('\nAssociation for each nanoruler')
    t0 = time.perf_counter() #we will measure how long some sub processes take.
    tgmm = 0
    tgmm_tcluster = 0
    tgmm_tdist = 0
    
    tgmmfilter = 0
    tgmmcount = 0
    tgmm_dataline = 0
    tgmm_psub3 = 0
    
    SIGMAS0 = [] #Global data - append for each sub clusters that are validated. 
    EXC0 = []
    INTERD0 = []
    ANGLE0 = []   #Angle of nanoruler
    MX0 = [] #x and y positions.
    MY0 = []
    IMGS0 = []
    NPH0 = []  #number of photon
    n_toofewloc = 0 #we'll keep track on how many nanorulers has spot with nloc<minloc, and thus were thrown away.
    
    print('\n')
    for idn in N:
        if printsub:
            print('\r - Handling Cluster- ' + str(idn) , end='')
        
        cx = x[cumcount[idn]:cumcount[idn+1]]
        cy = y[cumcount[idn]:cumcount[idn+1]]
        nphlocal = nph[cumcount[idn]:cumcount[idn+1]]
        if len(cx)< minloc:
            print('\tCluster with id' + str(idn)+ ' is incorrect from dbscan, skipped')
            continue
        
        #Association via chosen method :
        
        if method_subid==3:  #GMM
            tgmm0 = time.perf_counter()
            C2 = mixture.GaussianMixture(n_components=N_SUB_SPOT , covariance_type=gmm_covtype, weights_init= np.repeat(1/N_SUB_SPOT, N_SUB_SPOT) )
            #note on covariance type: full - independant covariance (and sigmas) , tied , diag, spherical
            subdata = np.transpose( np.vstack((cx,cy)) )
            C2.fit( subdata )
            tgmm_tcluster = tgmm_tcluster + (time.perf_counter() - tgmm0)
            
            gmm_sub_centers = C2.means_
#            gmm_sub_sigmas = C2.weights_
            cov = C2.covariances_
            gmm_sub_sigmas = get_sigmas( cov )
            
            tgmm_tdist0 = time.perf_counter()
            xdiff= (np.transpose(np.repeat([cx],N_SUB_SPOT,axis=0)) - gmm_sub_centers[:,0]) / gmm_sub_sigmas
            ydiff= (np.transpose(np.repeat([cy],N_SUB_SPOT,axis=0)) - gmm_sub_centers[:,1]) / gmm_sub_sigmas
#            distances = np.hypot(xdiff,ydiff)  #not the effective distance for gaussian distributions...
            distances = 0.5*np.hypot(xdiff,ydiff) + np.log(gmm_sub_sigmas)  # I believe this is a good estimate of closeness to a gaussian distribution
            
            res2 = np.argmin( distances , axis=1)  #Association à la gaussienne de centre (et dist/sigma) la plus proche
            
            tgmm_tdist =tgmm_tdist + (time.perf_counter() - tgmm_tdist0 )
            tgmm = tgmm + (time.perf_counter() - tgmm0)
            
            if  filter_ggm_prob: #filter points with log prob of being chosen >= filter_ggm_prob
                tgmmfilter0 = time.perf_counter() 
                validprob = ( -C2.score_samples( subdata ) < filter_ggm_prob)
                cx = cx[validprob]
                cy = cy[validprob]
                nphlocal = nphlocal[validprob]
                res2 = res2[validprob]
                if printsub:
                    print(' Prob Filtering ' + str(np.sum(validprob)) + '/' + str (len(validprob)) )
                tgmmfilter = tgmmfilter + (time.perf_counter() -tgmmfilter0)
                
            if plot_ggm:
                X,Y = np.meshgrid( np.arange(cx.min(),cx.max(),5) , np.arange(cy.min(),cy.max(),5) )
                XX = np.array([X.ravel(), Y.ravel()]).T
#                prob = -C2.score_samples( np.transpose( np.vstack((cx,cy)) ) )
                prob = -C2.score_samples( XX )
                prob = prob.reshape(X.shape)          
                
                plt.figure()
                plt.scatter( gmm_sub_centers[:,0], gmm_sub_centers[:,1] , marker='o')
                plt.scatter(cx,cy,c=res2,marker='*')
                CS = plt.contour(X, Y, prob, norm=LogNorm(vmin=1.0, vmax=50.0),
                 levels=np.logspace(0, 2, 50))
                CB = plt.colorbar()
                
        else:
            if method_subid==1:  #dbscan
                C2 = cluster.DBSCAN(eps=NANOR_SUB_DIST, min_samples=minsubloc, n_jobs=1)
                
            elif method_subid==2:  #Kmean
                C2 = cluster.KMeans(n_clusters=N_SUB_SPOT, n_jobs=None)
                
            C2.fit( np.transpose( np.vstack((cx,cy)) ) ) #note: possibility for cx cy to be void [] ... because of gmm filter
            res2 = C2.labels_
            
        tgmmcount0 = time.perf_counter()
        sub_idcount , sub_uid = np.histogram(res2,bins=np.arange(0,res2.max()+2))
        tgmmcount = tgmmcount + ( time.perf_counter() - tgmmcount0 )
        
                #NOTE/ with dbscan, we could check that each sub clusters have close number of localisations or so...
        #       if Nsubcluster = 4 we could delete the part with  the least loc in a subcluster..
        
        state='valid candidate'
        if res2.max() != N_SUB_SPOT-1: #we mark as unvalid a cluster with more or less than N_SUB_SPOT subcluster (this is a bit harsh)
            if printsub or print_sub_invalidity:
                print('\nCluster ' + str(idn) + 'has ' + str(res2.max()+1) + ' sub-clusters instead of ' + str(N_SUB_SPOT))
            state='unvalid'
            
        if plotsub:
            fig, ax = plt.subplots()
            fig.canvas.set_window_title( 'SubCluster - '+str(idn) )
            plt.title(state)
            plt.title( state + str ( gmm_sub_sigmas ) )
            if plotsub==2:
                cxshift = cx-cx.min()
                cyshift = cy-cy.min()
            else:
                cxshift= cx.copy()
                cyshift = cy.copy()
            ax.scatter(cxshift,cyshift, s=2, c=res2+1, cmap='prism')
            for i, txt in enumerate(res2):
                ax.annotate(str(txt), (cxshift[i], cyshift[i]))
            
        if state is 'valid candidate':
            mxs, mys, stdxs, stdys  = ( [] ,[] ,[] ,[] )
            
            #Collecting data , to get mean , variance, excentricity ... 
            if method_subid!=3:  #(method 3 = GMM => we already have center of clusters. kinda. Maybe we could redo it without extreme points..)
                for nsub in sub_uid: #Loop sur chaque Spot
    #                print(nsub)
                    if nsub!=-1:
                        v = (res2==nsub) #bit slow to do this but sorting small array x and y seems like overdoing it.
                        subx = cx[v]  
                        suby = cy[v]
                        msubx= np.mean(subx)
                        msuby= np.mean(suby)
                        mxs.append(msubx)
                        mys.append(msuby)
                        stdxs.append( np.std(subx) )
                        stdys.append( np.std(suby) )
                        if nsub==0:
                            nphlocal0 = np.median( nphlocal[v] )  #note : to check if works.
                        elif nsub==1:
                            nphlocal1 = np.median( nphlocal[v] )
                        elif nsub==2:
                            nphlocal2 = np.median( nphlocal[v] )
                        
            else: #pour la méthode GMM on a déjà accès au centre et à la variance sans boucler sur chaque spot:
                mxs = gmm_sub_centers[:,0]
                mys = gmm_sub_centers[:,1]
                stdxs = gmm_sub_sigmas
                stdys = gmm_sub_sigmas
                
                #En revanche pour le nombre de photon on est obligé de faire:
                nphlocal0 = np.median( nphlocal[ res2==0] )
                nphlocal1 = np.median( nphlocal[ res2==1] )
                nphlocal2 = np.median( nphlocal[ res2==2] )
            
            tgmm_dataline0 = time.perf_counter()
            #note: pour calcul de la pente On trie pour que les cluster 0 et 2 soient bien aux extrémités.
            mx = np.mean(mxs)
            my = np.mean(mys) # coord du centre, on va détecter le cluster 1 ( le plus proche du centre)
            middle = np.argmin( np.hypot( mxs-mx, mys-my) ) #position du cluster le plus proche du centre.
            
            shift = 1-middle  #Adapted to N SUB SPOT = 3 only. for more a shift will not be enough.
            mxs=np.roll(mxs,shift)    #Adapted to N SUB SPOT = 3 only.
            mys=np.roll(mys,shift)    #Adapted to N SUB SPOT = 3 only.
            stdxs=np.roll(stdxs,shift)  #Adapted to N SUB SPOT = 3 only.
            stdys=np.roll(stdys,shift)  #Adapted to N SUB SPOT = 3 only.
            
            a= (mys[-1]-mys[0])/(mxs[-1]-mxs[0]) # ax+b , droite passant par le premier et dernier sub cluster.
            b = -a*mxs[0] + mys[0]
            d2_sub = ( mys[1]-a*mxs[1]-b)/np.sqrt(1+a*a) #distance du 2eme point à la droite 1-3
            
            angle = np.arctan(a) #Note : here we get angle with only two points but best would be a linear regression on the 3 
#            angle = angle*180/np.pi  #degree
            
            (d20,d01,d12) = np.hypot( mxs - np.roll(mxs,1), mys - np.roll(mys,1) ) #distance between each cluster.
            if printsub:
                print('\t Excentrism: ' + str(d2_sub) + '\n \t ' + 'angle:' + str(angle) )
            tgmm_dataline = tgmm_dataline + (time.perf_counter()- tgmm_dataline0)
            


            valid_exc = ( np.abs(d2_sub) < MAX_EXC)  #max excentricity for middle subcluster to others

            valid_nc = True  #always true with method=1 because condition of dbscan is minsubloc
            if method_subid!=1:  # Checking if we have enough points in each sub cluster:
                if any(sub_idcount<minsubloc):
                    valid_nc = False
                
             # Checking distance between subsequent clusters,  then edges clusters :
            valid_dsubs = (d01>MIN_DSUB)*(d01<MAX_DSUB)*(d12>MIN_DSUB)*(d12<MAX_DSUB)
            valid_dsubtot = (d20>MIN_DSUBTOT)*(d20<MAX_DSUBTOT)
            
            #Creation of rotated points (for further historam plot along one direction)
            rcx, rcy = rotate_points(cx,cy,-angle, mx, my)
            rmcx , rmcy = rotate_points(mxs,mys,-angle, mx, my)
            subvalid = (res2!=-1)
            rcx, rcy = rcx[subvalid], rcy[subvalid]
            
            
            valid_tot = valid_exc and valid_nc and valid_dsubs and valid_dsubtot
            
            if valid_nc is False:
                n_toofewloc = n_toofewloc + 1
                
            if print_sub_invalidity:
                if valid_exc is False:
                    print('\nCluster ' + str(idn) + ' has excentricity ' + str(d2_sub) + ' above limit of ' + str(MAX_EXC))
                if valid_nc is False:
                    print('\nCluster ' + str(idn) + ' has too few localisation on at least one spot: '+str(minsubloc) + ' > '+ str(sub_idcount))
                if valid_dsubs is False:
                    print('\nCluster ' + str(idn) + ' inter-spot distance is out of limits')
                if  valid_dsubtot is False:
                    print('\nCluster ' + str(idn) + ' total distance is out of limits')
                    
                
                
            if plotsub2:
                plt.figure( 'Plotsub2: C-'+str(idn) + ' exc' + str(round(d2_sub,2)) + 'distances' + str(round(d01,1)) + ' ' +  str(round(d12,1)) + ' ' +  str(round(d20,1)) )
                
                plt.scatter(cx,cy, c=res2, cmap=CustomCmap)
                plt.scatter(mxs, mys, s=100, color='k', marker='+')
                plt.scatter(mx,my,s=200, color='pink')
                plt.title('valid='+str(valid_tot))
                xx = np.arange(mx-2*NANOR_SUB_DIST,mx+2*NANOR_SUB_DIST,1)
                plt.plot(xx, a*xx+b , linestyle='dashed')
                
                
            if ( valid_tot ):
                if printsub:
                    print('\t Sigmas: ' , gmm_sub_sigmas )
                
                SIGMAS0.append( gmm_sub_sigmas )
                EXC0.append( d2_sub )
                INTERD0.append( np.array([ d01,d12,d20 ]) )
                ANGLE0.append( angle )
                MX0.append( mx )
                MY0.append( my )
                IMGS0.append( loc2imgC( cx-cx.min()+20, cy-cy.min()+20, imgsshape, imgpixel, imgpixel) )
                NPH0.append( np.array([ nphlocal0,nphlocal1,nphlocal2 ]) )
                if plotsub3:  #note: As for now, we could move rotation calculus here because its only used for this.
                    tgmm_psub3_0 = time.perf_counter()

                    if plotsub3==2:  #shift data by min
                        rcx = rcx - rcx.min() + 0.5 #+0.5 may impact hist x plot ?
                        rcy = rcy - rcy.min() + 0.5
                        
                    
                    fig,ax = plt.subplots(3,1)
                    fig.canvas.set_window_title( 'Plotsub3: C-'+str(idn) + ' exc' + str(round(d2_sub,2)) + 'distances' + str(round(d01,1)) + ' ' +  str(round(d12,1)) + ' ' +  str(round(d20,1)) )
                    plt.subplot(311)
                    plt.scatter(rcx,rcy,c=res2[subvalid] ,cmap=CustomCmap )
                    ax[0].axis('equal')
                    
                    plt.title('Validated Cluster-'+str(idn) )
                    
                    plt.subplot(312)
                    H = plt.hist(rcx,bins=20)
                    
            
                    plt.subplot(313)
                    plt.hist(rcx[res2==0],bins=H[1], color=colorsList[0],alpha=0.6)
                    plt.hist(rcx[res2==1],bins=H[1], color=colorsList[1],alpha=0.6)
                    plt.hist(rcx[res2==2],bins=H[1], color=colorsList[2],alpha=0.6)
                    
                    tgmm_psub3 = tgmm_psub3 + (time.perf_counter() - tgmm_psub3_0)
    
    dt_sub_process = time.perf_counter() - t0 
    print('\t Kept '+str(len(MX0))+' Nanorulers on ' + str(len(N)))
    print('\t'+ str(n_toofewloc) + ' Spots had too few localisation (<' +str(minsubloc) +')'  )
    #End of loop on clusters --
    
"""
# =============================================================================
#   End of processes - Now Data handling and Plots :
# =============================================================================
"""
if doload==2:
    print('Loading previously saved data')
    MX0 = np.load( save_result_folder + 'MX0.npy')  #nanorulers data
    MY0 = np.load( save_result_folder + 'MY0.npy')
    SIGMAS0 = np.load( save_result_folder + 'SIGMAS0.npy')
    EXC0 = np.load( save_result_folder + 'EXC0.npy')
    IMGS0 = np.load( save_result_folder + 'IMGS0.npy')
    INTERD0 = np.load( save_result_folder + 'INTERD0.npy')
    ANGLE0 = np.load( save_result_folder + 'ANGLE0.npy')
    if os.path.isfile( save_result_folder +'NPH0.npy'): 
        NPH0 = np.load( save_result_folder + 'NPH0.npy')
    if os.path.isfile( save_result_folder +'frame.npy'): 
        frame = np.load( save_result_folder + 'frame.npy')
        x = np.load( save_result_folder + 'x.npy')
        y = np.load( save_result_folder + 'y.npy')
        nph = np.load( save_result_folder + 'nph.npy')
        
#    if os.path.isfile( save_result_folder +'frame.npy'):  #global loc data
#        frame = np.load( save_result_folder + 'NPH0.npy')
#        x = np.load( save_result_folder + 'NPH0.npy')
#         = np.load( save_result_folder + 'NPH0.npy')
    
    
if 'SIGMAS0' in locals():
    if len(SIGMAS0):
        #Data for plot : (any modification to it will not affect raw results_0) :
        SIGMAS = np.array( SIGMAS0 )  #Global data 
        mSIGMAS = np.mean(SIGMAS0,axis=1)
        EXC = np.array( EXC0 )
        INTERD = np.array( INTERD0 )
        ANGLE = np.array(ANGLE0)*180/np.pi
        MY = np.array(MY0)
        MX = np.array(MX0)
        IMGS = IMGS0.copy()
        NPH = NPH0.copy()
    else:
        print('\nNO CORRECT NANORULER FOUND - Check your parameters, plot stuff...')
        print('\t Aborting plot and save')
        plotcrop, plotfinal, plot_nanor, save_nanor, save_resultdata, save_measurements = 0, 0, 0, 0, 0, 0



if plotcrop and not(CROP is None):
    
    cvalid = (MX>CROP[0])*(MX<CROP[0]+CROP[2])*(MY>CROP[1])*(MY<CROP[1]+CROP[3])
    if np.sum(cvalid)!=len(cvalid): #modified.
        print('Cropping nanoruler''s results with',CROP)
        SIGMAS = SIGMAS[cvalid]
        mSIGMAS = mSIGMAS[cvalid]
        EXC = EXC[cvalid]
        INTERD = INTERD[cvalid]
        ANGLE = ANGLE[cvalid]
        MY = MY[cvalid]
        MX = MX[cvalid]
        IMGS = [IMGS[i] for i in range(0,len(IMGS)) if cvalid[i] ]
        NPH = NPH[cvalid]
    else:
        print('Crop already applied for plot')
        
    if not os.path.isdir( main ):
        print('Creating CROP res folder :' + main )
        os.mkdir( main )
        
        
if plotfinal:
    submethods = ['dbScan_','MeanS_','GMM_'+gmm_covtype+'_']
    str_submethod = submethods[method_subid-1]
    
    if not(( nanomax is None) or (nanomax==0)):
        print('SELECTING ONLY FIRST ' + str(nanomax) + ' Nanorulers for plot')
        main = main + "Nanomax=" + str(nanomax) + "_"
        
        SIGMAS = SIGMAS[:nanomax]
        mSIGMAS = mSIGMAS[:nanomax]
        EXC = EXC[:nanomax]
        INTERD = INTERD[:nanomax]
        ANGLE = ANGLE[:nanomax]
        MY = MY[:nanomax]
        MX = MX[:nanomax]
        IMGS = IMGS[:nanomax]
        NPH = NPH[:nanomax]
    
    print('Converting MX and MY in µm unit')
        
    if docrop==1 and not(CROP is None):
        main = main + '_crop_'
    hist_bin = 40
    
    if plot_hist:
        print('\t Plotting Histogram Result images')
        figf, axf = plt.subplots(2,2,figsize=(10,10))
        figf.canvas.set_window_title(' Some final Datas ')
        plt.subplot(221)
        plt.hist(mSIGMAS,bins=hist_bin, range=(4,18) )
        plt.title('Histogram of localization precisions ')
        plt.xlabel('Localization precision - nm')
        plt.ylabel('Occurence')
        
        
        plt.subplot(222)
        if method_subid==3 and (gmm_covtype is 'tied'): #same sigma for every sub cluster... so we replace this histogram for another
            plt.title('Histogram of excentrism ')
#            plt.hist(EXC,bins=hist_bin, range=(0,18) )
            plt.hist(EXC,bins=hist_bin, range=(-MAX_EXC,MAX_EXC) )
            plt.xlabel('Excentrism - nm')
            plt.ylabel('Occurence')
        
        
        else:
            plt.title('Histogram of each localization precisions ')
            plt.hist(SIGMAS[:,0],bins=hist_bin,alpha=0.6,color='r')
            plt.hist(SIGMAS[:,1],bins=hist_bin,alpha=0.6,color='g')
            plt.hist(SIGMAS[:,2],bins=hist_bin,alpha=0.6,color='cyan')
            plt.xlabel('Localization precision - nm')
            plt.ylabel('Occurence')
        
        #note: ideally hist ranges are set on your filtering.
        
        plt.subplot(223)
        plt.hist(INTERD[:,0],bins=hist_bin,alpha=0.5,color='g', range=(20,60) )
        plt.hist(INTERD[:,1],bins=hist_bin,alpha=0.5,color='r', range=(20,60) )
    #    plt.hist( np.ravel(INTERD[ :,[0,1] ]),bins=hist_bin,alpha=0.5,color='r') # distance 12 and 01 indinstinctely
        plt.title('Histogram of short distances ')
        plt.xlabel('Distance between cluster - nm')
        plt.ylabel('Occurence')
        
        plt.subplot(224)
        plt.hist(INTERD[:,2],bins=hist_bin, range=(60,100) )
        plt.title('Histogram of nanoruler size ')
        plt.xlabel('Size - nm')
        plt.ylabel('Occurence')
    
        plt.savefig( main + str_submethod + 'results.png')
        plt.savefig( main + str_submethod + 'results.eps')
    
    if plot_map:
        print('\t Plotting Map/Scatter Result images')
        plt.figure('Res image')
        map_bin = 15
        img_sigma = vhist2D(MX, MY, mSIGMAS, method='median', nb=(map_bin,map_bin), rng=None, nanzero = 0)
        plt.imshow(img_sigma, cmap ='nipy_spectral',vmin=resmin, vmax=resmax)
        plt.title(' Map of Precision ')
        plt.colorbar()
        #adding xticks would be cool..
        if map_ticks:
            tickvaluey = (MY.max()//map_bin)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
            tickvaluex = (MX.max()//map_bin)/1000
            tickX = np.arange(0, map_bin-1, 10/tickvaluex )
            tickY = np.arange(0, map_bin-1, 10/tickvaluey )
            plt.gca().set_xticks( tickX )
            plt.gca().set_yticks( tickY )
            plt.gca().set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
            plt.gca().set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
            plt.gca().set_xlabel('field - µm')
        else:
            plt.xticks([])
            plt.yticks([])
        plt.savefig( main + 'Map_' + str_submethod + 'results.png')
        plt.savefig( main + 'Map_' + str_submethod + 'results.eps')
        
        plt.figure('Res image Scatter')
        plt.scatter(MX/1000,MY/1000, s=10, c= mSIGMAS, vmin=resmin, vmax=resmax)
        plt.title(' Scatter Map of Resolution ')
        plt.colorbar()
        plt.xlabel('field - µm')
        plt.ylabel('field - µm')
        plt.savefig( main + 'rm='+str(resmax) + 'MapScat_' + str_submethod + 'results.png')
        plt.savefig( main + 'rm='+str(resmax) + 'MapScats_' + str_submethod + 'results.eps')
        
        plt.figure('Res image Interpolation 1')
        Iinterp = interpolate_img( MX, MY, mSIGMAS, binning=100, method='linear')
        plt.imshow( Iinterp , vmin=resmin, vmax=resmax)
        plt.title(' Linear Interpolated Map of Resolution ')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.savefig( main + 'rm='+str(resmax) + 'MapInterpL_' + str_submethod + 'results.png')
        plt.savefig( main + 'rm='+str(resmax) + 'MapInterpL_' + str_submethod + 'results.eps')
        
        plt.figure('Res image Interpolation 2')
        Iinterp = interpolate_img( MX, MY, mSIGMAS, binning=100, method='nearest')
        plt.imshow( Iinterp , vmin=resmin, vmax=resmax)
        plt.title(' Nearest Interpolated Map of Resolution ')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.savefig( main + 'rm='+str(resmax) + 'MapInterpN_' + str_submethod + 'results.png')
        plt.savefig( main + 'rm='+str(resmax) + 'MapInterpN_' + str_submethod + 'results.eps')
    

    
    if plot_graph:
        print('\t Plotting Res Scatter Graph')
        plt.figure('Scatter - field res graph',figsize=(10,5))
        plt.scatter(MX/1000, mSIGMAS,alpha=0.8,s=4, label='x axis ')
        plt.scatter(MY/1000, mSIGMAS,alpha=0.8,s=4, label='y axis ')
        plt.ylabel(' resolution - nm')
        plt.xlabel('axial position - µm')
        plt.legend()
        plt.ylim(resmin,resmax)
        plt.title('Scatter plot of resolution')
        plt.savefig( main + 'GraphResolScat_XY' + str_submethod + 'results.png')
        plt.savefig( main + 'GraphResolScat_XY' + str_submethod + 'results.eps')
        
        #Scatter but not on all image
        dcut = 20  #in µm - taken on both side of the center.
        dcut = dcut*1000
        validx = (MX> np.mean(MX)-dcut )*(MX< np.mean(MX)+dcut )
        validy = (MY> np.mean(MY)-dcut )*(MY< np.mean(MY)+dcut )
        
        plt.figure('ScatterCut - field res graph',figsize=(10,5))
        plt.scatter(MX[validy]/1000, mSIGMAS[validy],alpha=0.8,s=4, label='x axis ')
        plt.scatter(MY[validx]/1000, mSIGMAS[validx],alpha=0.8,s=4, label='y axis ')
        plt.ylabel(' resolution - nm')
        plt.xlabel('axial position - µm')
        plt.legend()
        plt.ylim(resmin,resmax)
        plt.title('Scatter plot of resolution')
        plt.savefig( main + 'GraphResolScat_XY_cut' + str(dcut) + '_' + str_submethod + 'results.png')
        plt.savefig( main + 'GraphResolScat_XY_cut' + str(dcut) + '_' + str_submethod + 'results.eps')
        
        
        
        
        plt.figure('field resolution graph - Y')
        dy = 2 #in µm
        
        ysort = np.argsort( MY )
        MYsort = MY[ysort] / 1000  #in µm
        SIGMAsort = mSIGMAS[ysort]
        Ycoord, Nnanoruler_in_dy = np.unique( MYsort//dy, return_counts=True)
        Ycoord = Ycoord*dy
        count = 0
        Yvalue = []
        for n_nano in Nnanoruler_in_dy:
            Yvalue.append( np.median( SIGMAsort[count:count+n_nano] ) )
            count = count + n_nano
        Ycoord = np.array( Ycoord )
        Yvalue = np.array( Yvalue )
        plt.plot( Ycoord, Yvalue)
        plt.title('Field dependance of resolution')
        plt.xlabel('field - µm')
        plt.ylabel('resolution - nm ')
    
        plt.savefig( main + 'GraphResol_Y_dy=' + str(dy) + str_submethod + 'results.png')
        plt.savefig( main + 'GraphResol_Y_dy=' + str(dy) + str_submethod + 'results.eps')
        
    
        plt.figure('field resolution graph - Y on Xcut')
        dy = 2 #in µm
        
        dx = 2  #µM - taken on each side.
        valid = np.abs(MX-np.median(MX)) < dx*1000 #note: mx unit is nm
        cutMY = MY[valid]
        cutmSIGMAS = mSIGMAS[valid]
        
        ysort = np.argsort( cutMY )
        MYsort = cutMY[ysort] / 1000  #in µm
        SIGMAsort = cutmSIGMAS[ysort]
        Ycoord, Nnanoruler_in_dy = np.unique( MYsort//dy, return_counts=True)
        Ycoord = Ycoord*dy
        count = 0
        Yvalue = []
        for n_nano in Nnanoruler_in_dy:
            Yvalue.append( np.median( SIGMAsort[count:count+n_nano] ) )
            count = count + n_nano
        Ycoord = np.array( Ycoord )
        Yvalue = np.array( Yvalue )
        plt.plot( Ycoord, Yvalue)
        plt.title('Field dependance of resolution')
        plt.xlabel('field - µm')
        plt.ylabel('resolution - nm ')
    
        plt.savefig( main + 'GraphResol_Y_dy=' + str(dy) + 'dx='+str(dx)+ str_submethod + 'results.png')
        plt.savefig( main + 'GraphResol_Y_dy=' + str(dy) + 'dx='+str(dx)+ str_submethod + 'results.eps')
        
        
        plt.figure('field-Radial resolution graph - Y')
        
        dr = 2 #in µm
        R = np.sqrt( (MX-np.median(MX))**2 + (MY - np.median(MY))**2 )
        
        rsort = np.argsort( R )
        Rsort = R[rsort] / 1000
        SIGMAsort = mSIGMAS[rsort]
        Rcoord, Nnanoruler_in_dr = np.unique( Rsort//dr, return_counts=True)
        Rcoord = Rcoord*dr
        count = 0
        Rvalue = []
        for n_nano in Nnanoruler_in_dr:
            Rvalue.append( np.median( SIGMAsort[count:count+n_nano] ) )
            count = count + n_nano
        Rcoord = np.array( Rcoord )
        Rvalue = np.array( Rvalue )
        plt.plot( Rcoord, Rvalue)
        plt.title('Field Radial dependance of resolution')
        plt.xlabel('radius - µm')
        plt.ylabel('resolution - nm ')
    
        plt.savefig( main + 'GraphResol_R_dr=' + str(dr) + str_submethod + 'results.png')
        plt.savefig( main + 'GraphResol_R_dr=' + str(dr) + str_submethod + 'results.eps')
        
        
    if plot_angle:
        print('\tPlotting Angle map and histograms')
        
        ANGSHIFT = ANGLE-ANGLE.min()
        sx=int(0.5 + SIGMAS.max())+1
        sy=int(0.5 + ANGSHIFT.max())+1
        bx = 1 #lié à l'angle.. heu... les ticks sont ok qu'avec bin=1. On va faire avec
        by= 1
        spx=int((sx/bx - 0.5)); 
        spy=int((sy/by - 0.5));
        rng=((0,(spy+1)*by),(0,(spx+1)*bx))
        imgangle = loc2imgC(ANGSHIFT, mSIGMAS ,shape_=(sx,sy),binning=bx, binningy = by)
      
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(' Resolution dependance to nanoruler angle ')
        plt.subplot(211)
        plt.imshow(imgangle)
        plt.ylabel('Resolution - nm')
        plt.xlabel('Angle - °')
        plt.yticks( np.arange( rng[1][0],rng[1][1], 4 ) )
        
        
        plt.subplot(212)
        dang = 30
        ang_bins = np.arange(-90,90,dang)
        
        for ag in ang_bins:
            sig_data_in_this_bin = mSIGMAS[ (ANGLE > ag)*(ANGLE <= ag+dang)]
            plt.hist(sig_data_in_this_bin,bins=50, density=1, alpha=0.5, label = str(ag) + ' to ' + str(ag+dang) )
        plt.legend()
        plt.ylabel('Resolution - nm')
        plt.xlabel('Occurence')
        plt.savefig( main + 'Angle_' + str_submethod + 'results.png')
        plt.savefig( main + 'Angle_' + str_submethod + 'results.eps')
        
        plt.show()
        
    if plot_density:
        dbin = 3 
        if 'x' in locals():
            print("\tPlotting General and Nanoruler density maps")
            gdens = loc2imgC( x/1000, y/1000, binning=dbin)
            plt.figure('Localisation Density Map')
            plt.title('Localisation Density Map')
            plt.imshow(gdens, cmap='nipy_spectral')
            plt.colorbar()
            plt.savefig( main + 'DensityGlobalMap_' + str_submethod + 'results.png')
            plt.savefig( main + 'DensityGlobalMap_' + str_submethod + 'results.eps')
        
        else:
            print('No general density map because x,y data is not loaded')
        
        dbin = 3
        ndens = loc2imgC(MX/1000,MY/1000, binning=dbin)
        plt.figure('Nanoruler Density Map')
        plt.title('Nanoruler Density Map')
        plt.imshow(ndens, cmap='nipy_spectral')
        plt.colorbar()
        plt.savefig( main + 'DensityNanoRMap_' + str_submethod + 'results.png')
        plt.savefig( main + 'DensityNanoRMap_' + str_submethod + 'results.eps')
        
    if plot_nph:
        if 'NPH' in locals():
            NPH = np.array(NPH)
            nph_nanor = np.mean(NPH,axis=1)
            nph_edge = 0.5*(NPH[:,0] + NPH[:,-1]) #getting data for histogram
            nph_center = NPH[:,1]
            dbin = 100
            map_bin = 15
            
            print('Plotting photon stuff')
            plt.figure('Median Photon count image')
            img_sigma = vhist2D(MX, MY, nph_nanor, method='median', nb=(map_bin,map_bin), rng=None, nanzero = 0)
#            nphvm = np.nanmedian(img_sigma) + 2*np.nanstd(img_sigma)
            nphvm = nph_vmax
            plt.imshow(img_sigma, cmap ='nipy_spectral',vmin=0, vmax=nphvm)
            plt.title(' Map of Photon ')
            plt.colorbar()
            if map_ticks:
                tickvaluey = (MY.max()//map_bin)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
                tickvaluex = (MX.max()//map_bin)/1000
                tickX = np.arange(0, map_bin-1, 10/tickvaluex )
                tickY = np.arange(0, map_bin-1, 10/tickvaluey )
                plt.gca().set_xticks( tickX )
                plt.gca().set_yticks( tickY )
                plt.gca().set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
                plt.gca().set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
                plt.gca().set_xlabel('field - µm')
            else:
                plt.gca().set_xticks( [] )
                plt.gca().set_yticks( [] )
            plt.savefig( main + 'rm='+str(nphvm) + 'NphNanoRMap_' + str_submethod + 'results.png')
            plt.savefig( main + 'rm='+str(nphvm) + 'NphNanoRMap_' + str_submethod + 'results.eps')
            
            plt.figure('Nphoton histogram',figsize=(12,4))
            plt.subplot(121)
            plt.hist( nph_nanor, bins=dbin, range=(0,nph_vmaxhist)  )
            plt.title('Photon per nanoruler')
            plt.subplot(122)
            plt.hist( nph_edge, bins=dbin, range=(0,nph_vmaxhist), alpha=0.6, label='edge' )
            plt.hist( nph_center, bins=dbin, range=(0,nph_vmaxhist) , alpha=0.6 , label='center' )
            plt.title('Photon per spot edge or center')
            plt.legend()
            
    if plot_dataframe:
        print('Plotting evolution of global data along frames...') #Useful for PFS/focus check along time.
        
        fsort = frame.argsort()
        fframe=frame[fsort]
        fsigma=sigma[fsort]
        fnph = nph[fsort]
        ufr, fcount = np.unique( fframe, return_counts=True)
        fcumcount = np.append( [0], np.cumsum(fcount) )
        meansigma_frame = []
        meannph_frame = []
        for pos, fvalue in enumerate(ufr): #getting median of data for each frame :
            pos0 = fcumcount[pos]
            posf = fcumcount[pos+1]
            nphdata = fnph[ pos0:posf ]
            sigmadata = fsigma[ pos0:posf ]
            meannph_frame.append( np.median( nphdata ))
            meansigma_frame.append( np.median( sigmadata ))
            
        plt.figure('Fluorophore count evolution along frame')
        plt.title('Temporal evolution of localisation count')
        plt.plot( ufr, fcount )
        plt.xlabel('frame number')
        plt.ylabel('Number of localization')
        plt.savefig( main + 'FrameEvol_Count_' + str_submethod + 'results.png')
        plt.savefig( main + 'FrameEvol_Count_' + str_submethod + 'results.eps')
            
        plt.figure('Median sigma evolution along frame')
        plt.title('Temporal evolution of PSF size')
        plt.plot( ufr, meansigma_frame )
        plt.xlabel('frame number')
        plt.ylabel('Median sigma - nm')
        p = PIX*PIXCOR
        plt.ylim(1.5*p,2.2*p)
        plt.savefig( main + 'FrameEvol_Sigma_' + str_submethod + 'results.png')
        plt.savefig( main + 'FrameEvol_Sigma_' + str_submethod + 'results.eps')
        
        plt.figure('Median nph evolution along frame')
        plt.title('Temporal evolution of photon count')
        plt.plot( ufr, meannph_frame )
        plt.xlabel('frame number')
        plt.ylabel('Median photon count')
        plt.ylim(0,nph_vmaxhist)
        plt.savefig( main + 'FrameEvol_Nph_' + str_submethod + 'results.png')
        plt.savefig( main + 'FrameEvol_Nph_' + str_submethod + 'results.eps')  
        
if plot_nanor:
#    n_nanorulers =len(IMGS)
#    n_nano_col = int( np.sqrt(n_nanorulers)*2)
#    n_nano_lines = n_nanorulers//n_nano_col + 1
    
    print('ploting best 100 nanorulers')
    n_nanorulers = 100
    n_nanorulers = min( n_nanorulers, len(IMGS) )
    takebest=1
    
#    n_nanorulers = min( len(IMGS), 500 )
    
    n_nano_col=10

    n_nano_lines= n_nanorulers//n_nano_col + 1
    
    fig_nanor = plt.figure()
#    plt.subplots()
#    plt.subplots_adjust( wspace=0.1, hspace=0.1)
    fig_nanor.suptitle( ' Best NanoRulers - total= ' + str(len(IMGS)))
    if takebest:
        args = mSIGMAS.argsort()
    for i in range(0, n_nanorulers):
        if takebest:
            img = IMGS[ args[i] ]
        else:
            img = IMGS[i]
            
        ax=fig_nanor.add_subplot(n_nano_lines,n_nano_col, i+1)        
        ax.imshow(img , cmap = 'hot')
    
#        plt.subplot( n_nano_lines,n_nano_col, i+1)
#        plt.imshow( img , cmap = 'hot')
        
#        plt.scatter(i,i)
        
        plt.xticks([])
        plt.yticks([])
    plt.savefig( main + str_submethod + '500NanoRulers_.png')
        
if save_nanor:
    if not os.path.isdir( main + str_submethod + 'Nanor/'):
        os.mkdir( main + str_submethod + 'Nanor/' )
    mm = main + str_submethod + 'Nanor/'
    for i in range(0, n_nanorulers):
        img = IMGS[i]
        img = np.array( img/np.max(img)*256,dtype='uint8')
        imageio.imwrite( mm + str(i) + '.png' , img )
        
    
    
"""
# =============================================================================
#       End of plots- SAVING
# =============================================================================
"""
        
if save_measurements:
    print('Saving measurements and parameters')
    
    #ALL
    filterlonesome_parameters = ['area_for_valid_friends','kill_everyone_with_less_than_this_number_of_friends']
    filterresult_parameters = ['MIN_DSUB','MAX_DSUB','MIN_DSUBTOT','MAX_DSUBTOT','MAX_EXC','minsubloc']
    dbscan_parameters = ['method','dmax','minloc', 'NANOR_DIST']
    dbscan2_parameters = ['method_subid','gmm_covtype','N_SUB_SPOT','NANOR_SUB_DIST']
    allparams = filterlonesome_parameters + filterresult_parameters + dbscan_parameters + dbscan2_parameters
        
    with open(main + 'Measurements.txt','w') as f:
        f.write('\n')
        f.write(main)
        f.write('\n')
    
        f.write('\n Mean Sigma : ')
        f.write( str( np.mean(mSIGMAS)))
        f.write('\n Std Sigma : ')
        f.write( str( np.std(mSIGMAS)))
        
        f.write('\n Mean Ecart : ')
        f.write( str( np.mean(INTERD[:,[0,1]]) ))
        f.write('\n Std Ecart : ')
        f.write( str( np.std(INTERD[:,[0,1]]) ))
        
        f.write('\n Mean Size : ')
        f.write( str( np.mean(INTERD[:,2]) ))
        f.write('\n  Std Size : ')
        f.write( str( np.std(INTERD[:,2]) ))
        
        f.write('\n Mean Exc : ')
        f.write( str( np.mean(EXC) ))
        f.write('\n Std Exc : ')
        f.write( str( np.std(EXC) ))
        
        f.write('\n')
        f.write('\n Median Sigma : ')
        f.write( str( np.median(mSIGMAS)))
        f.write('\n Median Ecart : ')
        f.write( str( np.median(INTERD[:,[0,1]]) ))
        f.write('\n Median Size : ')
        f.write( str( np.median(INTERD[:,2]) ))
        f.write('\n Median Exc : ')
        f.write( str( np.median(EXC) ))
        
        f.write('\n')
        f.write('PARAMETERS ----')
        for p in allparams:
            exec( "f.write( \' "+p+" : \'" + "+str("+ p +"))" )


if save_resultdata:
    if doload==2:
        print('Not saving data as it had been loaded !')
    else:
        print('Saving data in a folder')
#        data_string = ['MX0','MY0','SIGMAS0','EXC0','IMGS0','INTERD0','ANGLE0','NPH0']
        data_string = ['MX0','MY0','SIGMAS0','EXC0','IMGS0','INTERD0','ANGLE0','NPH0','x','y','nph','frame']
        data_string = ['MX0','MY0','SIGMAS0','EXC0','IMGS0','INTERD0','ANGLE0','NPH0','x','y','nph','frame','res','res2']
        
        for ds in data_string:
            print('\t saving ' + ds)
            exec( "np.save( save_result_folder + ds + '.npy'," + ds + ") " )
            
        

    
    

print('\n THIS ALL TOOK ',time.perf_counter()-t00,' seconds')

if 'dt_load' in locals():
    print('\t Data Loader took ' + str(round(dt_load,4)) + 'seconds')
    print('\t Filter Lone took ' + str(round(dt_filter_lonesome,4)) + 'seconds')
    print('\t Association took ' + str(round(dt_association,4)) + 'seconds')
    
    print('\t Sub Processes took ' + str(round(dt_sub_process,4)) + 'seconds')
    print('\t\t Sub-GMM & ID_asso took ' + str(round(tgmm,4)) + 'seconds')
    print('\t\t\t Sub-GMM cluster took ' + str(round(tgmm_tcluster,4)) + 'seconds')
    print('\t\t\t Sub-GMM ID asso took' + str(round(tgmm_tdist,4)) + 'seconds')
    print('\t\t Sub-GMM prob-filt took ' + str(round(tgmmfilter,4)) + 'seconds')
    print('\t\t Sub-GMM UID Count took ' + str(round(tgmmcount,4)) + 'seconds')
    print('\t\t Sub-GMM data-line took ' + str(round(tgmm_dataline,4)) + 'seconds')
    print('\t\t Sub-GMM plot_sub3 took ' + str(round(tgmm_psub3,4)) + 'seconds')