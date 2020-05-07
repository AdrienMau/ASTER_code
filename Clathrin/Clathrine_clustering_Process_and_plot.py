# -*- coding: utf-8 -*-
"""
Created on 16 Oct 2019/
Last modified -  - Apil 2020.
   
    Opens X Y data of Clathrin (in data0, drift). Made to open csv files from ThunderSTORM or Abbelight.
    - Either:
        Use Clustering (typical: sklearn dbscan ) to get each clathrin x,y, points_id
        Opens Clustering result from Nemo software to get each clathrin x,y
    

     + Throw away unvalid clusters ( ..., noise, bad clathrin ...) => see criteria to use... TODO
     - Calculate stuff (Fenster diameter, sizex, sizey, circularity, hollow...)

    - Save Results and Related Parameters (if CROP => save in a specific folder )

+ You may also open already analyzed data results (doload==2)

Results are stored in RES, which is a dict(). RES['sx'] = > np array of sx for each cluster.
Units by default are in nm (as loaded through csv files.)

    PLot of results: use plotfilter_keys to select specific population or none.



Simple use of this program consist of modifying True (1) or False (0) conditions in the 'control board'.
For example : setting plotfilter to True will allow filtering based on keys: "plotfilter_keys]" and bounds "plotfilter_ranges"


Once data is treated you may change doload to a value of 2, to load and plot result.


Note: image of clathrin is done via this code then saved for a clusterAnalysis object..
@author: Adrien MAU / ISMO & Abbelight


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import cm
from mpl_toolkits.mplot3d import Axes3D


import sys, os, time
import pandas, scipy
from scipy.interpolate import griddata
from scipy import signal

import sklearn
from sklearn import cluster, mixture
import imageio
import Caliper_Feret
import ClusterAnalysis
import tifffile as tf

""" Global parameters """

npzfolder = "'EPISpy_res/"  #my notation for Python localized data.
npzfile = 'save.npz'    #my notation for Python localized data.
csvfile = "CoordTable2D.csv"
csvdrift_file = "DriftTable.csv"

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

ALLOW_PICKLE = True #since new versions need to be true for np load ?

PIX = 108 #pixel size (to convert drift to x,y,sx,sy same unit (nm) )
PIXCOR = 60/64   #applied to x and y at Opening  (pour Pierre mettre à 1)


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

    
#Charge les data csv , + éventuellement applique le drift.
# Note: Pour les csv le drift est déjà appliqué. (soft 180 Live)
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

#plt.close('all')

doload = 2   #2 : load result data. 1:load data - use 1 to process data.
if doload==2:
    DO_NOT_PROCESS = True    #  just plot. - comment this if needed.


#data path:
main = ""
datapath = main + 'Coordinates/'

#Result path:
save_resultdata = 1
save_result_folder = main + "results/"
if not os.path.isdir( save_result_folder ):
    os.mkdir( save_result_folder )
save_measurements = 1  #save median resolution and its deviation, same for angle, excentricity, mX, mY ...




docrop= 0  #crop (when doload=1) loaded data0 with CROP -and- CROP MX MY result before plot.
CROP = None    #x0 y0 dx dy
# CROP = np.array([ 52730,64770,7500,7500])  # Iterative finding. Zone 1
#CROP = np.array([ 56730,64770,5500,3500])  # Iterative finding. Zone 1 - area of interest.
shift_nearzero = 1  #shift loaded data after crop so that the minimum is at (PIX,PIX)

plot_load = 0  # Scatter plot of data after load, crop, and shift (Heavy !)


"""
# =============================================================================
#  CONTROL BOARD - PROCESSES
# =============================================================================
"""

""" FIRST FILTER - remove lonesome points - DBscan """
filter_lonesome_guys_with_dbscan_before_anything = True
kill_everyone_with_less_than_this_number_of_friends = 10 #we will only keep data with this number of friend (-1?) in around area
area_for_valid_friends = 30   # define area of friendliness. People farther away don't like you.
plt_filterlonesome = 0  #HEAVY: scatter plot of filter lonesome before and after. (use CROP to not f*ck up your memory)

""" Separate different clathrin - DBscan """
do_association = True
method=12    #1 or 12 for sklearn dbscan   (( 2 for sklearn meanshift.   3 ELKI 4 self dbscan ))
npts = 25  #minimum number of loc for dbscan - typ for clathrine : 30
epsilon_dmax = 35    #typical distance for cluster asso. (epsilon)
show_idassoc = 0  #scatter plot with id number annotation. (~heavy.)
show_idassoc_annotate = False #add text id in scatter plot of show_idassoc. (heavy+)

""" Sort and keep clusters only """
checkid = True  #sort along id order, and throw away id=-1   (This sort step is needed for further calculus as it expects sorted data to speed up processes.)
method_id = 1

""" Calculus on clusters """
do_calculus = 1
maxloc = 750 # filter: cluster with more than this number of loc are thrown out. (500 was use dbefore but throw away some good hollow clathrin.)
maxferet = 600 # filter: cluster with feret higher than 1µm are thrown out.
minferet = 20

show_clusters = 0  #show sub cluster scatter  ( !heavy, you may set specific N values to look at specific clusters)
show_clusters_images = 0     #show for each sub cluster the loc 2D image. (cumuled histogram)
print_invalid = 0 # when a cluster is invalid: print invalidity condition.
show_idfinal = False  #scatter of found valid cluster. Comparaison with clusters before and raw data.


""" PLOT """
hist_bin = 400
hist_keys = ['radius','se','radiusratio','feret']
#hist_keys = ['mnph','radius','stdradius','feret']
map_bin = 200
sr_max = 100
feret_max = 400   #vmax
nph_vmax = 20000  #for map
map_ticks = True  #add µm ticks to res and photon map.-to link

nloc_max = 600 # for nloc range.




#map_keyx, map_keyy, map_keyz = ( 'feret','nloc','se')
map_keyx, map_keyy, map_keyz = ( 'mx','my','radiusratio')
dens_keyx, dens_keyy = ( 'feret','radiusratio')


plotfinal = 1   #plot final data in general. --- detail below

plot_hist = 1   #histogram of results. - and save
plot_map = 0  #map of resolution and scatter map
plot_map_any = 0 #map of spatial map_keyx, map_keyy and color for map_keyz
plot_graph = 0 #plot res on scatter along axis
plot_density = 0 #plot Global density map (of localisation) and of validated clusters
plot_density_any = 0
plot_nph = 0
plot_dataframe = 0 #plot evolution of different datas along frame (global)
plot_nloc = 0   #plot number of loc (per cluster) depending on other parameter (ex: feret)


probability_map = 0 #for fit_dist or fit_map: create map of population. saves it.
fit_dist = 1    #fit a distribution by n gaussians. - good with feret and radius.

#key to fit gaussian populations:
fit_key = 'radius'
#fit_key = 'feret'
fit_bin = 400

fit_map = 0   #fit a 2D histogram by n gaussians.  (x: feret , y:radiusratio, z:count. )
fit_gmm = 0  #fit a 2D scatter points by n gaussians prob dist. GMM
fit_map_keys = ( 'feret','radiusratio')
fit_map_bin = 200



plot_imgs = 1  #plot all found cluster images
plot_imgs_n = 200  #0: plot all, else: plot Image of cluster n° 0 to 'plot_imgs_n' 
plot_img_randomize = True #if True: plot *plot_imgs_n* random elements taken in results. (instead of first ones)
np.random.seed( 1337 )
plot_imgs_gflou = 0.9  #0 to do nothing. Else is sigma of gaussian to blur-convolve image
save_imgs = 0 ##need plot_imgs=1: #save plot_img image
save_allimgs = 1   #need plot_imgs=1 : save img one by one 


#FILTERING FOR PLOT :
plotcrop=0   # note :in not processing ,is set to docrop value - crop for histogram-result plot.
plotfilter = False     #filter for plot; with keys plotfilter_key on ranges plotfilter_ranges: ( low,high )

#plotfilter_keys=['feret','sr']
#plotfilter_ranges=[(60,72),(15,21)]  #first population - sub1
#plotfilter_ranges=[(60,72),(21.5,27)]

#for publication : Select population based on radius.
plotfilter_keys=['radius']
# plotfilter_ranges=[(6,23)] #  #xint is 25.8   34.7    60.9
# plotfilter_ranges=[(28,33)] #
plotfilter_ranges=[(36,58)] #
# plotfilter_ranges=[(63,90)] #
# plotfilter_ranges=[(63,120)] #  



#LIST FOR PLOT :
make_croplist=0  #for plot: make list with different crop from saved data.
CROP0 = [0,6e4,6e4,6e4]
CROP1 = [6e4, 0 , 5e4, 9e4 ]
CROP2 = [9e4,9e4,4e4,4e4]
CROP_list = [CROP0, CROP1, CROP2 ]



#for calculus process: you may precise specific id to plot and analyse.
N = None # if None: take all cluster for sub associations. You may chose specific cluster number yourself.
#N = np.array([10,11,12,13,14])
# N = np.arange(0,20)


if DO_NOT_PROCESS:
    print('NOT PROCESSING ANYTHING !')
    if doload==2:
        newdoload = 2
    else:
        newdoload = 0
    doload, filter_lonesome_guys_with_dbscan_before_anything, do_association, checkid, do_calculus = newdoload,0,0,0,0
#    plotcrop = docrop
    docrop = -1  # => not renewing data with data0.
    
else:
    print('PROCESSING')
    
"""
# =============================================================================
#  PROCESSES
# =============================================================================
"""

#Load
# t00 =  time.clock()
if doload==1:
    # t0 = time.clock()   
    
    if main.endswith( npzfolder ):
        print('Loading npz data, and drift from csv')
        frame,x,y,sigma, nph = load_npz( datapath + npzfile )
        
        (datafun, drift) = load_csv( datapath+csvfile, datapath+csvdrift_file )
        del(datafun)
        fcount = drift[:,0]
        arr = np.arange(0,len(frame))
        data0 = np.vstack(( arr,frame,x,y,sigma,nph )).T  #according to FCOL XCOL.. etc
    
    else:
        print('Loading in' + datapath + csvfile)
        (data0, drift) = load_csv( datapath+csvfile, datapath+csvdrift_file )
        #fcount = drift[:,0]   #nmol per frame, (note: is not modified while docrop)
        driftx = drift[:,1]*PIX #drift in nm
        drifty = drift[:,2]*PIX #drift in nm
        
    fstart = int( np.min( data0[:,FCOL]) )  #first frame number

    #crop
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
        
        main = datapath + 'CROP-' + str(CROP) + '/'
        if not os.path.isdir( main ):
            print('Creating CROP res folder :' + main )
            os.mkdir( main )
    elif docrop!=-1:
        data=data0.copy()
    
    #Shift et correction de taille pixel.
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
#End of loading, cropping, pixel size correction, optional shift.


#Pre-Filter : deleting lonesome points.
if filter_lonesome_guys_with_dbscan_before_anything:
    # t0 = time.clock() 
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
    # dt_filter_lonesome = time.clock() - t0 
    
    
#detecting clusters
if do_association:
    print('\nAssociating ID')
    # t0 = time.clock()
    if method==1:
        print('\t Using sklearn Dbscan with dist=',epsilon_dmax,' and min loc=',npts)
        C = cluster.DBSCAN(eps=epsilon_dmax, min_samples=npts, n_jobs=1)
        C.fit( np.transpose( np.vstack((x,y)) ) )
        res = C.labels_
    if method==12:
        print('\t Using sklearn Dbscan-ball_tree, dist=',epsilon_dmax,' and min loc=',npts)
        C = cluster.DBSCAN(eps=epsilon_dmax, min_samples=npts, algorithm='ball_tree')
        C.fit( np.transpose( np.vstack((x,y)) ) )
        res = C.labels_

    print( '\t ' + str(res.max()+1) + ' clusters found' )
    # dt_association = time.clock() - t0 
    
    if show_idassoc:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Scatter- cluster ID association')
        ax.scatter(x,y, s=0.5, alpha=0.6, c=res%12,cmap='gist_rainbow')
        ax.set_facecolor('black')
        if show_idassoc_annotate:
            for i, txt in enumerate(res):
                ax.annotate(str(txt), (x[i], y[i]))
    
        
#resorting cluster results.
if checkid:
    # t0 = time.clock() 
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
    if idcount.min()<npts:
        invalid = idcount<npts
        nincorrect = str(np.sum( invalid ))
        print('\t Cluster fucked up at some point, ' + nincorrect+' clusters have nelem<' + str(npts) )
        print('\t Corresponding to ids ' +  str(uid[:-1][invalid]) )
    # dt_checkid = time.clock() - t0 


#calculus on clusters
if 'uid' in locals():
    if N is None:
        N = uid[:-1]


if do_calculus: 
    print('\nAssociation for each cluster')
    # t0 = time.clock() #we will measure how long some sub processes take.
    
    imgsshape = (250,250)  #warning: will crop saved image if not high enough. - area taken for img , in x y unit. (here nm)
    img_pixel = 5   #saved img is of shape imgsshape/binning. => pixel is of size img_pixel
    NLOC = [] # final cluster data will be: ['nloc','sx','sy','feret','mx','my','img','mnph']
    SX = []
    SY = []
    FERET = []
    MX = []
    MY = []
    IMGS = [] #we create image here
    NPH = []
    ANGLE = []
    RAD = [] #mean distance from center.
    SRAD = []
    MID = []
    
    global_id_valid = np.ones(len(res),dtype='bool')  #for each loc: true if the associated loc belong to validated cluster.
    id_valid = np.ones(len(N),dtype='bool')  #For each cluster : true if cluster is validated in calculus.
    print('\n')
    for idpos, idn in enumerate(N):
        #data of actual cluster (c prefix) :
        cx = x[cumcount[idn]:cumcount[idn+1]]
        cy = y[cumcount[idn]:cumcount[idn+1]]
        cnph = nph[cumcount[idn]:cumcount[idn+1]]
        valid=1
        
        #Calculus
        cnloc = len(cx)

        cmx, cmy = Caliper_Feret.center_fit(cx, cy)  # find center where it minimizes radius std.
        cmx0 = np.mean(cx)
        cmy0 = np.mean(cy)
        if abs(cmx-cmx0)>35:
            cmx = cmx0
        if abs(cmy-cmy0)>35:
            cmy = cmy0
            
        radius = np.hypot(cx-cmx,cy-cmy)
        mradius = np.mean( radius )
        stdradius = np.std( radius )
        
#        csigmax = np.std(cx)             #personal note: beware of names here ! (don't use loaded data names)
#        csigmay = np.std(cy)
        feret = Caliper_Feret.diameter( [ [cx[i],cy[i]] for i in range(0,len(cx))] ) #minx miny maxx maxy,=longest line that can be plot from data
        dxfer = feret[0][0] - feret[1][0]
        dyfer = feret[0][1] - feret[1][1]
        drfer = np.hypot(dxfer,dyfer)
        
        cangle = Caliper_Feret.get_angle( cx, cy, 0.01)
        rcx, rcy = Caliper_Feret.rotate(cx,cy,cangle)
        csigmax = np.std(rcx)             #personal note: beware of names here ! (don't use loaded data names)
        csigmay = np.std(rcy)
        
        mcnph = np.mean(cnph)

        
#        cimg = loc2imgC( cx-cx.min()+30, cy-cy.min()+30, imgsshape, binning=img_pixel).astype('uint8')  #max 255
        cimg = loc2imgC( cx-cmx+imgsshape[1]/2, cy-cmy++imgsshape[1]/2, imgsshape, binning=img_pixel).astype('uint8')  #max 255

        nloc_valid = (cnloc<maxloc)
        feret_valid = (drfer<maxferet)*(drfer>minferet)
        valid = valid * nloc_valid * feret_valid
        if not(valid):
            id_valid[idpos] = False
            global_id_valid[cumcount[idn]:cumcount[idn+1]] = False
        else:
            NLOC.append( cnloc ) # final cluster data will be: ['nloc','sx','sy','feret','mx','my','img','mnph']
            SX.append( csigmax )
            SY.append( csigmay )
            FERET.append( drfer )
            MX.append( cmx )
            MY.append( cmy )
            IMGS.append( cimg )
            NPH.append( mcnph )
            ANGLE.append( cangle )
            RAD.append( mradius )
            SRAD.append( stdradius )
            MID.append( idn )
            
            
        if print_invalid:
            if not(nloc_valid):
                print('id n°',idn,' has invalid nloc')
            if not(feret_valid):
                print('id n°',idn,' has invalid feret diameter')
        
        if show_clusters:
            plt.figure( 'Id n°'+str(idn) )
            plt.scatter( cx-cx.min(), cy-cy.min(), s=100, alpha=0.7 )
            plt.scatter( cmx-cx.min(), cmy-cy.min(), s=80, alpha=0.7 , marker='*')
            plt.scatter( cmx0-cx.min(), cmy0-cy.min(), s=80, alpha=0.7 , marker='*')
            plt.title('id n°'+str(idn) + 'valid:'+str(valid)+ ['_nloc='+str(cnloc)+'-'+str(nloc_valid)+'_feret'+str(feret_valid), ''][valid] +'_r='+str(mradius) )
        
        if show_clusters_images:
            plt.figure( 'Image of Id n°'+str(idn) )
            plt.imshow( cimg )
            
            
    
    gdata_names = ['frame','x','y','cluster_id','nph','sigma'] #global loc data (after filtering)
    gid = global_id_valid
    gdata_values = [frame[gid], x[gid], y[gid], res[gid], nph[gid], sigma[gid] ]
    gRES_raw = dict( zip( gdata_names, gdata_values )) #result dictionary for all fluorophore from valid cluster.

    data_names = ['nloc','sx','sy','feret','mx','my','img','mnph','angle','radius','stdradius','c_id'] #data for each cluster
    data_values = [np.array(NLOC), np.array(SX),np.array(SY),np.array(FERET),np.array(MX),np.array(MY),np.array(IMGS),np.array(NPH), np.array(ANGLE), np.array(RAD), np.array(SRAD), np.array(MID) ]
#    data_values = [NLOC,SX,SY,FERET,MX,MY,IMGS,NPH ]
    RES_raw = dict( zip( data_names, data_values ) ) # result dictionary for individual validated clusters.
    
    # dt_sub_process = time.clock() - t0 
    print('\t Kept '+str( len(N[id_valid])) +' Clusters on ' + str(len(N)))
    
    if show_idfinal:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Scatter- cluster ID final association')
        ax.scatter(x[gid],-y[gid], s=0.01, alpha=0.9, c=res[gid]%12,cmap='gist_rainbow')
#        ax.scatter(MX,MY, s=1, alpha=0.6, c=np.array(MID)%12,cmap='gist_rainbow')
        ax.set_facecolor('black')
        
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Scatter- cluster ID Initial')
        ax.scatter(x,-y, s=0.01, alpha=0.9, c=res%12,cmap='gist_rainbow')
        ax.set_facecolor('black')
        
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Scatter- Data Initial')
        ax.scatter(data[:,XCOL]-data[:,XCOL].min()+x.min(),-(data[:,YCOL]-data[:,YCOL].min()+y.min()), s=0.01, alpha=0.9,cmap='gray')
        ax.set_facecolor('black')
        
        
        
#"""    End of loop on clusters -- """
    
"""
# =============================================================================
#   End of processes - Now Saving (if data was not loaded) :
# =============================================================================
"""

#Extracting data - loading( if doload=2) , reading dictionnary
if doload==2:
    print('Loading previously saved data')
    RES_raw = np.load( save_result_folder + 'RES.npy', allow_pickle=ALLOW_PICKLE).item()  #clathrin-cluster data
    gRES_raw = np.load( save_result_folder + 'gRES.npy', allow_pickle=ALLOW_PICKLE).item()   #clathrin-cluster_asso localisations data
        

if 'RES_raw' in locals():
    if not( RES_raw['nloc'] is None):
        # remember:     data_names = ['nloc','sx','sy','feret','mx','my','img','mnph'] #data for each cluster
        #Data for plot : (any modification to it will not affect raw RES :
        RES = RES_raw.copy()
        gRES = gRES_raw.copy()  
        MX = RES['mx']
        MY = RES['my']
        NLOC = RES['nloc']
        SIGMAX = RES['sx']
        SIGMAY = RES['sy']
        FERET = RES['feret']
        IMGS = RES['img']
        MNPH = RES['mnph']
        ANGLE = RES['angle']
        RAD = RES['radius']
        SRAD = RES['stdradius']
else:
    print('\nNO CORRECT CLATHRIN FOUND - Check your parameters, plot stuff...')
    print('\t Aborting plot and save')
    plotcrop, plotfinal, plot_nanor, save_allimgs, save_resultdata, save_measurements = 0, 0, 0, 0, 0, 0



if save_measurements and doload==1:
    print('Saving measurements and parameters (doload=1)')
    
    #ALL
    filterlonesome_parameters = ['filter_lonesome_guys_with_dbscan_before_anything','area_for_valid_friends','kill_everyone_with_less_than_this_number_of_friends']
    filterresult_parameters = ['maxferet','minferet','maxloc','npts']
    dbscan_parameters = ['method','epsilon_dmax','npts']
    calculus_parameters = ['maxloc','maxferet','minferet']
    allparams = filterlonesome_parameters + filterresult_parameters + dbscan_parameters + calculus_parameters
        
    with open(main + 'Measurements.txt','w') as f:
        f.write('\n')
        f.write(main)
        f.write('\n')

        f.write('\n Mean Nloc : ')
        f.write( str( np.mean(NLOC) ))
        f.write('\n  Std Nloc : ')
        f.write( str( np.std(NLOC) ))
        
        f.write('\n Mean MX : ')
        f.write( str( np.mean(MX) ))
        f.write('\n Std MX : ')
        f.write( str( np.std(MX) ))
        
        f.write('\n Mean MY : ')
        f.write( str( np.mean(MY) ))
        f.write('\n Std MY : ')
        f.write( str( np.std(MY) ))

        f.write('\n Mean SigmaX : ')
        f.write( str( np.mean(SIGMAX)))
        f.write('\n Std SigmaX : ')
        f.write( str( np.std(SIGMAX)))
        
        f.write('\n Mean SigmaY : ')
        f.write( str( np.mean(SIGMAY)))
        f.write('\n Std SigmaY : ')
        f.write( str( np.std(SIGMAY)))
        
        f.write('\n Mean Feret : ')
        f.write( str( np.mean(FERET) ))
        f.write('\n Std Feret : ')
        f.write( str( np.std(FERET) ))
        
        f.write('\n Mean Nph : ')
        f.write( str( np.mean( NPH ) ))
        f.write('\n Std Nph : ')
        f.write( str( np.std( NPH ) ))
        
        f.write('\n Mean Radius : ')
        f.write( str( np.mean( RAD ) ))
        f.write('\n Std Rad : ')
        f.write( str( np.std( RAD ) ))
        
        f.write('\n')
        f.write('PARAMETERS ----')
        for p in allparams:
            exec( "f.write( \' "+p+" : \'" + "+str("+ p +"))" )


if save_resultdata:
    if doload==2:
        print('Not saving data as it had been loaded !')
    else:
        print('Saving data in a folder')
        np.save( save_result_folder + 'RES.npy', RES_raw)
        np.save( save_result_folder + 'gRES.npy', gRES_raw)
        
        
"""
# =============================================================================
#               PLOTTING STUFF
# =============================================================================
"""

#Data containers for plot:
if make_croplist: #make list of results based on different crop regions
    print('Making ClusterResult List with CROP_list ')
    CRlist = []
    for somecrop in CROP_list:
        print('somecrop is'),somecrop
        CR = ClusterAnalysis.ClusterResults()
        CR.load_dict( save_result_folder )
        CR.crop_all( somecrop )
        CRlist.append( CR )
        del(CR)
else: #or: use results directly (no crop)
    CR = ClusterAnalysis.ClusterResults()
    CR.load_dict( save_result_folder )
    CRlist = [CR]


for cr in CRlist: #range for plot:
    cr.set_rangekeys('sr',(0,sr_max))
    cr.set_rangekeys('feret',(40,feret_max))
    cr.set_rangekeys('mnph',(0,nph_vmax))
    cr.set_rangekeys('nloc',(0,nloc_max))
    cr.set_rangekeys('angle',(-90,90))
    

# Here : crop for all data in CRlist:
if plotcrop and not(CROP is None):
    print('Cropping for plot')
    for pos in range(0,len(CRlist)):
        CRlist[pos].crop_cluster( CROP, keys=['mx','my']) #to test yet
        if not os.path.isdir( main ):
            print('Creating CROP res folder :' + main )
            os.mkdir( main )

#Here: filter on cluster keys (similar to plotcrop if keys are mx, my )
if plotfilter:
    for kpos, k in enumerate(plotfilter_keys):
        rng = plotfilter_ranges[kpos]
        for cr in CRlist:
            cr.filter_cluster( rng=rng, key=k , printnum=True)


if plotfinal:
    
    if plot_hist:
        ClusterAnalysis.res_hist( CRlist , hist_bin=hist_bin, savepath=save_result_folder, keylist=hist_keys )
#        plt.savefig(save_result_folder + 'hist_' + "_".join(hist_keys) + '.png')
#        plt.savefig(save_result_folder + 'hist_' + "_".join(hist_keys) + '.eps')
        
    if plot_map:
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx ='mx', keyy='my', keyz='feret', map_ticks=True, cmap='nipy_spectral', doscatter=True )
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx ='mx', keyy='my', keyz='feret', map_ticks=True, cmap='nipy_spectral', doscatter=False )
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx ='mx', keyy='my', keyz='sr', map_ticks=True, cmap='nipy_spectral', doscatter=False )
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx ='mx', keyy='my', keyz='mnph', map_ticks=True, cmap='nipy_spectral', doscatter=False )

    if plot_graph:
        ClusterAnalysis.res_axis( CRlist, daxis=1000, savepath=None, keyaxis ='mx', keyz='feret', dcut=500, mcut=None, keycut='my', clist=None, doscatter=False)
        ClusterAnalysis.res_axis( CRlist, daxis=1000, savepath=None, keyaxis ='my', keyz='feret', dcut=500, mcut=None, keycut='mx', clist=None, doscatter=False)
        ClusterAnalysis.res_axis( CRlist, daxis=1000, savepath=None, keyaxis ='mr', keyz='feret', dcut=0, mcut=None, keycut=None, clist=None, doscatter=False)
    
    if plot_density:
        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='mx', keyy='my', binx=1000, biny=1000, cmap='nipy_spectral', vmax=50 )
        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='x', keyy='y', binx=1000, biny=1000, cmap='nipy_spectral', vmax=1000 , getglobal=True )

    if plot_nph:
        ClusterAnalysis.res_map( CRlist , map_bin=100, savepath=save_result_folder, keyx ='mx', keyy='my', keyz='mnph', map_ticks=True, cmap='nipy_spectral', doscatter=False )

    if plot_dataframe:
        ClusterAnalysis.res_axis( CRlist, daxis=1, savepath=None, keyaxis ='frame', keyz='nph', dcut=0, mcut=None, keycut=None, clist=None, doscatter=False, getglobal=True)
        ClusterAnalysis.res_axis( CRlist, daxis=1, savepath=None, keyaxis ='frame', keyz='sigma', dcut=0, mcut=None, keycut=None, clist=None, doscatter=False, getglobal=True)
        
    if plot_nloc:
        ClusterAnalysis.res_axis( CRlist, daxis=1, savepath=None, keyaxis ='feret', keyz='nloc', dcut=0, mcut=None, keycut=None, clist=None, doscatter=True, getglobal=False)
        
    if plot_map_any:
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx=map_keyx, keyy=map_keyy, keyz=map_keyz, map_ticks=True, cmap='nipy_spectral', doscatter=False )
        ClusterAnalysis.res_map( CRlist , map_bin=map_bin, savepath=save_result_folder, keyx=map_keyx, keyy=map_keyy, keyz=map_keyz, map_ticks=True, cmap='nipy_spectral', doscatter=True )
    
    if plot_density_any:
        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx =dens_keyx, keyy=dens_keyy, cmap='nipy_spectral',vmax=None)
#        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='feret', keyy='nloc', cmap='nipy_spectral',vmax=None)
#        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='feret', keyy='angle', cmap='nipy_spectral',vmax=None)
#        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='feret', keyy='se', cmap='nipy_spectral',vmax=None)
#        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='radius', keyy='stdradius', cmap='nipy_spectral',vmax=None)
#        ClusterAnalysis.res_densitymap( CRlist, savepath=None, keyx ='radius', keyy='feret', cmap='nipy_spectral',vmax=None, trueticks=True)

    
    if fit_dist:
        
        y, x = CR.get_hist(fit_key,bins=fit_bin)
        
        if fit_key is 'feret':
#            p0list = [   250, 2, 67,     220, 18, 85,   125, 60, 172,   0]
            p0list = p0list = [ 250, 5, 65,   230, 15, 85,   80, 40, 170,   35, 30, 250 , 0]
            
            
        elif fit_key is 'radius':
#            p0list = [   300, 2, 22,     80, 10, 40,   45, 40, 82,   0]
            p0list = [   400, 3, 22,   150, 5, 28,   130, 10, 48,  50, 10, 70,   0]
        
        if len(p0list)//3==3:
            print('Using three gaussian fit')
    #        ClusterAnalysis.three_gaussian1d_checkparam(x,y,p0list)
            pfitlist = ClusterAnalysis.three_gaussian1d_fit(x, y, p0list)
            ClusterAnalysis.three_gaussian1d_checkparam(x, y , pfitlist )
        else:
            ClusterAnalysis.n_gaussian1d_checkparam(x,y,p0list); plt.title('first estimation on ' + fit_key);
            ngaussian = len(p0list)//3
            print('Using ' , ngaussian , ' gaussian fit')
            pfitlist = ClusterAnalysis.n_gaussian1d_fit(x, y, p0list)
            ClusterAnalysis.n_gaussian1d_checkparam(x, y , pfitlist )
        
        plt.title(fit_key + 'fit of population')
                
        pixsize = 10
        param = pfitlist.copy()
#                CID = CRlist[0].get_reskeys('cluster_id',True)
        X , Y =CRlist[0].get_reskeys(['x','y'],True)
        f = CRlist[0].get_reskeys( fit_key )
        local_cid = CRlist[0].get_reskeys('c_id',False)
        nloc = CRlist[0].get_reskeys('nloc')
        if len(p0list)//3==3:
            p1 = np.ravel(ClusterAnalysis.gaussian1d( f, *param[0:3]) ) #param is amp sigma xc
            p2 = np.ravel(ClusterAnalysis.gaussian1d( f, *param[3:6]) )
            p3 = np.ravel(ClusterAnalysis.gaussian1d( f, *param[6:9]) )
            s = p1 + p2 + p3
#            p1, p2, p3 = p1/s, p2/s, p3/s #normalisez probability for each -maybe not needed.
            pop= np.argmax([p1,p2,p3],axis=0) # classification for each f value.
            x12, x23 = ClusterAnalysis.threeg_intersections( param , in_between=1) #intersection of prob. 
            pop2 = 0*(f<=x12) + 1*(f>x12)*(f<=x23) + 2*(f>x23)
        else:
            pn = []
            for ng in range(0,len(param)//3):
                pn.append( np.ravel(ClusterAnalysis.gaussian1d( f, *param[3*ng:3*ng+3]) ))
            s = np.sum(pn ,axis=0)
            pop = np.argmax(pn, axis=0)
            xint = ClusterAnalysis.n_intersections( param , in_between=1)
            pop2 = np.sum(f>xint, axis=0)
            
        #Two choice can be made for population assignment: either use probability (pop) . Or set only two boundary between the 3 gaussians dist. (pop2).
        


        print('prob map with ' + fit_key)
        frng = CRlist[0].get_rangekeys(fit_key)

        plt.figure()
        
        #Adapted bin to border between gaussians : (this will change a slight bit bin size for each pop region)
        dbin = np.diff(frng) / hist_bin
        new_bins = np.array([])
        frontiere = np.append( np.insert(xint,0,0), frng[-1] )
        for xpos, axint in enumerate( frontiere[:-1] ):
            nelem = int( (frontiere[xpos+1]-frontiere[xpos])/dbin )
            new_bins = np.concatenate((new_bins,np.linspace(frontiere[xpos], frontiere[xpos+1], nelem, endpoint=True)))
#        hist_binf = hist_bin  #uncomment to chose classical bin option.
        hist_binf = new_bins
        
        plt.subplot(221)
        if len(p0list)//3==3:
            plt.hist(f[pop==0],range=frng,bins=hist_binf,color='blue',alpha=0.6,label='pop1')
            plt.hist(f[pop==1],range=frng,bins=hist_binf,color='green',alpha=0.6,label='pop2')
            plt.hist(f[pop==2],range=frng,bins=hist_binf,color='red',alpha=0.6,label='pop3')
        else:
            clist = ['blue','green','red','purple','orange']
            for n in range(0,pop.max()+1):
                plt.hist(f[pop==n],range=frng,bins=hist_binf,color=clist[n],alpha=0.6,label='pop'+str(n))
        plt.legend()
        
        plt.subplot(222)
        if len(p0list)//3==3:
            plt.scatter(f,p1/s,s=1,label='pop1') #/s => normalized prob
            plt.scatter(f,p2/s,s=1,label='pop2')
            plt.scatter(f,p3/s,s=1,label='pop3')
        else:
            for ng in range(0,len(pn)):
                plt.scatter(f,pn[ng]/s,s=1,label='pop'+str(ng)) #/s => normalized prob

        plt.xlabel( CRlist[0].get_namekeys(fit_key) ) #or something else...
        plt.xlim(frng)
        plt.ylabel('probability')
        plt.legend()
        plt.title('Population repartition')
        
        plt.subplot(223)
#        farr = np.arange(0,400,0.01)
        farr = np.arange(*frng,0.5)
        if len(p0list)//3==3:
            plt.plot(farr, ClusterAnalysis.gaussian1d( farr, *param[0:3]) )
            plt.plot(farr, ClusterAnalysis.gaussian1d( farr, *param[3:6]) )
            plt.plot(farr, ClusterAnalysis.gaussian1d( farr, *param[6:9]) )
            plt.scatter( x12, ClusterAnalysis.gaussian1d( x12, *param[0:3]) )
            plt.scatter( x23, ClusterAnalysis.gaussian1d( x23, *param[3:6]) )
        else:
            for ng in range(0,len(param)//3):
                plt.plot(farr, ClusterAnalysis.gaussian1d( farr, *param[3*ng:3*ng+3]) )
            for intermediary in xint: 
                plt.scatter(intermediary, ClusterAnalysis.gaussian1d( intermediary, *param[3*ng:3*ng+3]))
        
        
        plt.subplot(224)
        if len(p0list)//3==3:
            plt.hist(f[pop2==0],range=frng,bins=hist_binf,color='blue',alpha=0.6,label='pop1')
            plt.hist(f[pop2==1],range=frng,bins=hist_binf,color='green',alpha=0.6,label='pop2')
            plt.hist(f[pop2==2],range=frng,bins=hist_binf,color='red',alpha=0.6,label='pop3')
        else:
            for pval in range(0,pop2.max()+1):
                plt.hist(f[pop2==pval],range=frng,bins=hist_binf,color=clist[pval],alpha=0.6,label='pop'+str(pval))

        if probability_map:

            #Here for each localID we have feret, nloc and assigned population. (we'll take pop2)
            #In global (all points): we have X Y and all IDs.
            #We want to create the global vector of population.
            
            #SLOW WAY;
            Gpop = np.zeros((len(X)),dtype='uint8')
            Gfer = np.zeros((len(X)),dtype='uint16')
            count = 0
            print('creating Population Image - pixel of' + str(pixsize) + 'nm')
            for pos, aid in enumerate(local_cid):
                localpop = pop2[pos]
                local_nloc = nloc[pos]
                Gpop[count:count+local_nloc] = localpop
                Gfer[count:count+local_nloc] = int(f[pos]+0.5)
                count = count + local_nloc
                
    #                plt.figure()
    #                plt.scatter(X,Y,c=Gpop,cmap='hsv',s=1)
            
            
            img = vhist2D(X, Y, Gpop+1, 'mean',nb=(X.max()/pixsize , Y.max()/pixsize) ) #todo: check x y bins not inverted..
            img = img.astype('uint8')
            img[np.isnan(img)] = 0
            plt.figure()
            plt.imshow(img, cmap ='nipy_spectral',vmin=0,vmax=3.5)
            plt.colorbar()
            tf.imsave( save_result_folder+'ImgPop_'+fit_key+str(pixsize)+'nm.png' , img, dtype='uint8')
            
            
            imgfer = vhist2D(X, Y, Gfer, 'mean',nb=(X.max()/pixsize , Y.max()/pixsize) ) #todo: check x y bins not inverted..
            imgfer = imgfer.astype('uint16')
            imgfer[np.isnan(imgfer)] = 0
            plt.figure()
            plt.imshow(imgfer, cmap ='nipy_spectral',vmin=0)
            plt.colorbar()
            tf.imsave( save_result_folder+'Img'+fit_key+'_'+str(pixsize)+'nm.png' , imgfer, dtype='uint16')
            plt.savefig( save_result_folder+'Img'+fit_key+'_'+str(pixsize)+'nm.png')
            plt.savefig( save_result_folder+'Img'+fit_key+'_'+str(pixsize)+'nm.eps')
            



    if fit_map: #fit on image.  need default plist.
        #or do GMM clustering ... GMM seems worse. (not logical but hey)
        ngaussian = 4
        nparam = 6 #length of a p parameter.
        if 'pfit' in locals(): del(pfit)  #comment to keep previous pfit.
        
        fit_map_bin = 80
        xb, yb, zimg = CR.get_map( fit_map_keys[0], fit_map_keys[1], getglobal=False, binx=fit_map_bin,biny=fit_map_bin,density=None, plot_img = 0)
        xb = (xb[1:] + xb[:-1])/2
        yb = (yb[1:] + yb[:-1])/2
        show_3D = 1
        if show_3D:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            XB, YB = np.meshgrid(xb,yb)
            surf = ax.plot_surface(XB, YB, zimg, cmap='jet',linewidth=0, antialiased=False)
        
        #OK - hand made
#        p0 = [ 30, 0.4, 6, 0.0 , 2.64, 66]  #amp sx sy theta xc yc
#        p1 = [ 30, 0.34, 10, 0.0 , 2.4, 80]  #amp sx sy theta xc yc
#        p2 = [ 20, 0.25, 30, 0.0 , 2.3, 108]
#        p3 = [ 17, 0.25, 38, 0.0 , 2.45, 190]  #amp sx sy theta xc yc
#        p4 = [ 7, 0.25, 38, 0.0 , 2.6, 270]
        
        
        #hand and some GMM
        p0 = [ 35, 0.40,  12, 0, 2.59, 70]  #amp sx sy theta xc yc #45, 0.35,  10, 0, 2.53, 70 #p2
        p1 = [ 32, 0.3,  17, 0, 2.26, 90]   #p1
        p2 = [15,  0.26, 25, -0., 2.25, 142.8 ]  #p4
        p3 = [ 15, 0.25, 30, 0, 2.45, 190]   #p0
        p4 = [8 , 0.3, 43.7, -0. , 2.7,  263.5]  #amp sx sy theta xc yc
        
        
        plist = p0 + p1 + p2 + p3 + p4
        plist = plist[:nparam*ngaussian] + [0]
        
        Gfit = ClusterAnalysis.n_gaussian_2de(xb,yb,plist)
        
        
        
        wx,wy = None,None #ticks for next plot.
        if fit_map_keys[1] is 'radiusratio': wx = np.array([1,2,3,4])
        if fit_map_keys[0] is 'feret': wy = np.array([50,100,150,200,250,300,350],dtype='int')
        
        go_further = 1
        if go_further==-1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            XB, YB = np.meshgrid(xb,yb)
            surf = ax.plot_surface(XB, YB, zimg-Gfit, cmap='jet',linewidth=0, antialiased=False)
        
        if go_further==0: #test initial plist
            ClusterAnalysis.n_gaussian_2de_checkparam(xb,yb,zimg,plist, 1, wanted_tickx=wx, wanted_ticky=wy)
        elif go_further>0:
            if not 'pfit' in locals():
                print('fitting...')
                pfit = ClusterAnalysis.n_gaussian_2de_fit(xb,yb,zimg,plist, 0)
                print('\tdone')
            else:
                print('NOT FITTING AS PFIT ALREADY EXISTS !')
                
            pfit2 = pfit.copy()  #we apply correction for population selection, this avoids some overlaps.
            print('reducing pfit2 last gaussian sigma !')
            pfit2[ (ngaussian-1)*nparam + 2] = pfit2[ (ngaussian-1)*nparam + 2] * 0.7 #reduced sigma
            pfit2[ (ngaussian-2-1)*nparam + 2] = pfit2[ (ngaussian-2-1)*nparam + 2] * 0.85 #reduced sigma
            
#            
            ClusterAnalysis.n_gaussian_2de_checkparam(xb,yb,zimg,pfit2 , 1, wanted_tickx=wx, wanted_ticky=wy)
            
            #Exp data: 'probability' to exist on each gaussian :
            (cy,cx) = CR.get_reskeys(fit_map_keys, getglobal=False)
#            cvalid = (cx< np.mean(cx) + np.std(cx)*12 )
#            cx,cy = cx[cvalid], cy[cvalid] #
            

            
            pn = []           
            for ng in range(0,len(pfit)//nparam):
    #            print('param here is',pfit[nparam*ng:nparam*ng+nparam] )
                pn.append( np.ravel(ClusterAnalysis.gaussian_2de_list( cx,cy, *pfit2[nparam*ng:nparam*ng+nparam]) ))
            pop = np.argmax(pn, axis=0)
            plt.figure()
            
            
            my_cmap = matplotlib.colors.ListedColormap(['#4ac5ff', '#00b247','#fff700', '#ff371e', '#9e00c8', 'pink', 'brown'][:ngaussian])
#            my_cmap = cm.get_cmap('jet', ngaussian)
            plt.scatter( cx, cy, c=pop, s=0.1, alpha=1, cmap=my_cmap, vmax=ngaussian-0.5,vmin=-0.5)
            plt.gca().invert_yaxis()
            plt.colorbar(ticks = np.arange(0,ngaussian,1))
            plt.title('Population repartition')
            plt.ylim( CR.get_rangekeys(fit_map_keys[0]))
            plt.xlim( CR.get_rangekeys(fit_map_keys[1]))
            
#            for n in range(0,len(pn)):  #Assess individual gauss prob.
#                plt.figure()
#                plt.scatter( cy, cx, c=pn[n], s=1, alpha=0.7, cmap='jet', label='pop'+str(n))
#                plt.gca().invert_yaxis()
#                plt.colorbar()
#            plt.legend()
        
        
        if probability_map:
            pixsize = 10 #nm
            X , Y = CRlist[0].get_reskeys(['x','y'],True)
            local_cid = CRlist[0].get_reskeys('c_id',False)
            nloc = CRlist[0].get_reskeys('nloc')
            #Here for each localID we have feret, nloc and assigned population. (we'll take pop2)
            #In global (all points): we have X Y and all IDs.
            #We want to create the global vector of population.
            
            Gpop = np.zeros((len(X)),dtype='uint8')
            count = 0
            print('creating Population Image - pixel of' + str(pixsize) + 'nm')
            for pos, aid in enumerate(local_cid):
                localpop = pop[pos]
                local_nloc = nloc[pos]
                Gpop[count:count+local_nloc] = localpop
                count = count + local_nloc
            
            img = vhist2D(X,Y, Gpop+1, 'mean',nb=(X.max()/pixsize , Y.max()/pixsize) ) #todo: check x y bins not inverted..
            img = img.astype('uint8')
            img[np.isnan(img)] = 0
            plt.figure()
            plt.imshow(img, cmap ='nipy_spectral',vmin=0,vmax=ngaussian+0.5)
            plt.colorbar()
            
            tf.imsave( save_result_folder+'FitMAP_NanoPop_'+fit_map_keys[0]+'_'+fit_map_keys[1]+'ng='+str(ngaussian)+'_'+str(pixsize)+'nm.png' , img, dtype='uint8')
            
            
            
            
            

    if fit_gmm: #fit on xy points, no need of default param.
        (cx,cy) = CR.get_reskeys(fit_map_keys, getglobal=False)
        gmm_bins=160
#        xb = np.linspace( cx.min(),cx.max(), 50)
#        yb = np.linspace( cy.min(),cy.max(), 50)
        yb, xb, zimg = CR.get_map( fit_map_keys[0], fit_map_keys[1], getglobal=False, binx=gmm_bins,biny=gmm_bins,density=None, plot_img = 0)
        xb = (xb[1:] + xb[:-1])/2
        yb = (yb[1:] + yb[:-1])/2
        zimg = zimg.T
        
        #for 5 finds: array([  7,  36,   0,  -3, 270,   2,  14,  25,   0,   0, 117,   2,  11,  26,   0,   0, 191,   2,   5, 103,   0,  -3, 277,   2,  22,  12, 0,  -3,  76,   2,   0])
        #for 4 finds : array([  5.5,  15. ,   0.4,  -3.1,  80.2,   2.4,   2. ,  43.7,   0.3, -0. , 248.5,   2.6,   3.9,  39.7,   0.3,  -0. , 152.8,   2.4, 1.4,  98.5,   0.5,  -3.1, 273.2,   2.8,   0. ])
        ngaussian = 4
        gmm_covtype = 'full' #full tied diag... of covariance for GMM clustering.
        
        C2 = mixture.GaussianMixture(n_components=ngaussian , covariance_type=gmm_covtype, weights_init= np.repeat(1/ngaussian, ngaussian) )
        subdata = np.transpose( np.vstack((cx,cy)) )
        C2.fit( subdata )
            
        gmm_c = C2.means_
#        gmm_amps = C2.weights_ * np.sum(zimg) #sum weights is 1. 
        gmm_amps = C2.weights_
        
        cov = C2.covariances_
#        gmm_sigmas = ClusterAnalysis.get_sigmas( cov, ngaussian ) #general sigma.
        res = np.array( ClusterAnalysis.cov_to_params( cov ) )
        gmm_sxs = res[:,0]
        gmm_sys = res[:,1]
        gmm_angles =  (res[:,2]+np.pi) % np.pi
        gmm_angles = -gmm_angles
        xcol=0
        ycol=1
        plist = []
        for n in range(0,len(gmm_amps)):
            plist = plist + [ gmm_amps[n], gmm_sxs[n], gmm_sys[n], gmm_angles[n], gmm_c[n,xcol], gmm_c[n,ycol] ]  #amp sx sy theta xc yc
        plist = plist + [0]

        amp0 = np.sum( ClusterAnalysis.n_gaussian_2de(xb,yb,plist )) #prop a amp, sx et sy
        gmm_amps = C2.weights_ / np.sum(amp0) * np.sum(zimg) #sum weights is 1. - but there should be no cropping of gaussians for this...
        
        plist = []
        for n in range(0,len(gmm_amps)):
            plist =plist + [ gmm_amps[n], gmm_sxs[n], gmm_sys[n], gmm_angles[n], gmm_c[n,xcol], gmm_c[n,ycol] ]  #amp sx sy theta xc yc
        plist = plist + [0]
        
        ampnorm = np.sum( ClusterAnalysis.n_gaussian_2de(xb,yb,plist ))
        ClusterAnalysis.n_gaussian_2de_checkparam(xb,yb,zimg,plist, individual=1)



    if plot_imgs:
        nimg =  plot_imgs_n
        if nimg==0:
            print('plotting all clusters images  !')
        else:
            print('plotting first ' + str(nimg) + 'clusters images')
        if plot_imgs_gflou !=0:
            print('\t gaussian blur will be added, sigma:'+str(plot_imgs_gflou) )
            X,Y = np.arange(0,cr.imgshape[1]), np.arange(0,cr.imgshape[0])
            Gflou = ClusterAnalysis.gaussian_2d(X,Y, 1, plot_imgs_gflou, int(cr.imgshape[0]/2), int(cr.imgshape[1]/2), 0)
        
        if save_allimgs:
            if not os.path.isdir( save_result_folder + 'AllImg/'):
                os.mkdir( save_result_folder + 'AllImg/' )
            fstr = ['','filter_'][plotfilter] + ['','_'.join(plotfilter_keys)][plotfilter] + ['',str(plotfilter_ranges)][plotfilter]
            mm = save_result_folder + 'AllImg/' + fstr
            print('Preparing to save individual imgs in' + mm)
            
        for crpos, cr in enumerate(CRlist):
            IMGS = cr.get_reskeys('img')
            
            if nimg!=0: #select only a few images:
                if plot_img_randomize:
                    arr = np.arange(0,len(IMGS))
                    np.random.shuffle( arr ) #act directly on arr. (note: or shuffle directly IMGS)
                    IMGS = IMGS[ arr[:nimg] ] #ou random ?
                else:
                    IMGS = cr.get_reskeys('img')[:nimg]
            li = len(IMGS)
            n_cols= int( np.sqrt(li))
            n_lines= li//n_cols + 1    
            
            fig_nanor = plt.figure() #this 1/ and next line
            fig_nanor.suptitle( ' Clusters of Resultn°' + str(crpos) + '- total= ' + str(len(IMGS)))            
            for ipos, img in enumerate(IMGS):
                if plot_imgs_gflou !=0:
                    img = scipy.signal.convolve2d( img, Gflou , mode='same')
                ax=fig_nanor.add_subplot(n_lines,n_cols, ipos+1)     #this /1
                ax.imshow(img , cmap = 'hot', vmax = 0.9*img.max()  ) #note:Modif 2020.
                plt.xticks([])
                plt.yticks([])
                if save_allimgs:
                    img = np.array( img/np.max(img)*255,dtype='uint8')
                    imageio.imwrite( mm + str(ipos) + '.png' , img )
        
                
        if save_imgs:
            if not os.path.isdir( save_result_folder + 'Imgs/img'):
                os.mkdir( save_result_folder + 'Imgs/img' )
            fstr = ['','filter_'][plotfilter] + ['','_'.join(plotfilter_keys)][plotfilter] + ['',str(plotfilter_ranges)][plotfilter]
            savef = save_result_folder + 'Imgs/img' + fstr + ''
            print('Saving imgs of ' + str(nimg) + 'clusters in ' + savef)
            plt.savefig(savef+'.png')
            plt.savefig(savef+'.eps')
                
#if save_allimgs:
#    if not os.path.isdir( save_result_folder + 'AllImg/'):
#        os.mkdir( save_result_folder + 'AllImg/' )
#    mm = save_result_folder + 'AllImg/'
#    IMGS = cr.get_reskeys('img')[:nimg]
#    for i in range(0, len(IMGS)):
#        img = IMGS[i]
#        img = np.array( img/np.max(img)*256,dtype='uint8')
#        imageio.imwrite( mm + str(i) + '.png' , img )
#        
    
    
"""
# =============================================================================
#       End of plots
# =============================================================================
"""



    
    

# print('\n THIS ALL TOOK ',time.clock()-t00,' seconds')

# if 'dt_load' in locals():
#     print('\t Data Loader took ' + str(round(dt_load,4)) + 'seconds')
#     print('\t Filter Lone took ' + str(round(dt_filter_lonesome,4)) + 'seconds')
#     print('\t Association took ' + str(round(dt_association,4)) + 'seconds')
    