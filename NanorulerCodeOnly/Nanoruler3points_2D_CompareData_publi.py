# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:35:58 2019
Modified on 7 oct 2019 to be able to adapt to nspots = 3 or 2

- Opens Saved result data (from Nanoruler3points_2D )
- Plot and compare results. 
    
@author: Adrien MAU / ISMO & Abbelight


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid

import sys, os, time
import pandas, scipy
from scipy.interpolate import griddata
import scipy.ndimage
from scipy.ndimage import gaussian_filter as gf
from scipy import signal

import sklearn
from sklearn import cluster, mixture
import imageio

""" Global parameters """

#colorsList = ['c','r','y']
colorsList = ["C0","C2","C1","C3","C4","C5","C6"]
CustomCmap = matplotlib.colors.ListedColormap(colorsList)
                
#
#PIX = 108 #pixel size (to convert drift to x,y,sx,sy same unit (nm) )
#PIXCOR = 60/64   #applied to x and y at Opening  (pour Pierre mettre à 1)

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
def vhist2D(x,y,values, method='median', nb=(50,50), rng=None, nanzero = 0):
    img = scipy.stats.binned_statistic_2d(x, y, values, statistic=method, bins=nb, range=rng, expand_binnumbers=False)[0]
    if nanzero:
        img[ np.isnan(img)] = 0 
    return img

#pxy is pixel size in (x,y) directions. shape_ is final image shape (sy,sx)
    #recenter: image will be centered according to xy centroid. (median)
         #if a final shape is given, img is centered at final, according to median x and y
         
def myvhist2D(x,y,values, method='median', pxy=(50,50), shape_=None, nanzero = 0, recenter=False):
    x = x-x.min()
    y = y-y.min()
    bx = pxy[0]
    by = pxy[1]
    spx=int(x.max()//bx)+1; #number of box along x dir (0 1 2 3) => spx=4
    spy=int(y.max()//by)+1;
    IMG = np.zeros((spy,spx))
    
    nbox = (x//bx) + (y//by)*spx
    nboxs = nbox.argsort()
    values = values[nboxs]
    nbox = nbox[nboxs]
    ubox, countbox = np.unique( nbox, return_counts = True)
    ubox = ubox.astype('int')
    cumcount = 0
    for pos, b in enumerate(ubox):
        countb = countbox[pos]
#        print('box n°'+str(b))
#        print('y box is ', b//spx, ' x box is ', b%spx)
        IMG[ b//spx , b%spx ] = np.median( values[cumcount:cumcount+countb])
        cumcount = cumcount + countb
        
    if nanzero:
        IMG[ np.isnan(IMG) ] = 0
    if (shape_ is None):
        print('\t returning image with default shape - myvhist2D')
        return IMG
    else:
        print('\t Adapting to asked shape - myvhist2d')
        wanted_sx = shape_[1]
        wanted_sy = shape_[0]
        print('\t reshaping in sxsy : ' + str(wanted_sx) + ' ' + str(wanted_sy) )
        offx, offy = 0, 0
        if recenter and(wanted_sx>spx)and(wanted_sy)>spy:
            print('\t adding offset to recenter...') #maybe work only if final image is bigger...
            offx = int( wanted_sx/2 - (np.median(x)//bx) )
            offy = int( wanted_sy/2 - (np.median(y)//by) )
        IMGf = np.zeros((wanted_sy, wanted_sx))
        sxeff = min(wanted_sx,spx)
        syeff = min(wanted_sy,spy)
        print('\t sxeff and syeff',sxeff,syeff)
        IMGf[offy:offy+syeff, offx:offx+sxeff] = IMG[0:syeff, 0:sxeff]
        return IMGf


#slow but easy to implement myvhist2D. - no recenter here. Should give same result as myvhist2D !
    
#pxy is pixel size in (x,y) directions. shape_ is final image shape 
    #recenter: image will be centered according to xy centroid. (median)
def myvhist2D_baka(x,y,values, method='median', pxy=(50,50), shape_=None, nanzero = 0):
    x = x-x.min()
    y = y-y.min()
    bx = pxy[0]
    by = pxy[1]
    spx=int(x.max()//bx)+1; #number of box along x dir (0 1 2 3) => spx=4
    spy=int(y.max()//by)+1;
    IMG = np.zeros((spy,spx))
    nbox = (x//bx) + (y//by)*spx
    for px in range(0,IMG.shape[1]):
        for py in range(0,IMG.shape[0]):
            related_box = px + spx*py
            IMG[py,px] = np.median(values[nbox==related_box])
    if nanzero:
        IMG[ np.isnan(IMG) ] = 0
    return IMG

#fill img holes (value=numtofill), except those at border. 
  #  it does median or mean of 8 surrounding pixels + itself. (when dxy=1),
    #you can increase area by increasing dxy (area border = 2*dxy+1 )
# in order not to include 'holes' in median or mean, use ignoreholes = True
    #else holes (example: value 0) will be taken in calculus.
# you can input maxhole: if surrounding holes are superior that maxhole, we keep the hole value. (no median/mean)
def fill_img_holes(img, method='median',numtofill=0, dxy=1, ignoreholes=True, maxhole=4):
    img_temp = img.copy()
    for px in range(dxy,img.shape[1]-dxy):
        for py in range(dxy,img.shape[0]-dxy):
            if img[py,px]==numtofill:
                data = img_temp[py-dxy:py+dxy+1 , px-dxy: px+dxy+1]
                holes = (data==numtofill)
                if np.sum( holes )< maxhole:
                    if ignoreholes:
                        data = data[data!=numtofill]
                        
                    if method is 'median':
                        value = np.mean(data)
                    elif method is 'mean':
                        value = np.median(data)
                    img[py,px] = value
        img[ np.isnan(img)] = numtofill
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



#Create a gaussian image. amp*exp(-x²/2sigma²)
def gaussian_2d(X,Y, amp, sigma, xc, yc, offset=0):
    X = (X-xc)/(sigma**2)
    Y = (Y-yc)/(sigma**2)
    eX= np.exp(-0.5*(X**2))
    eY= np.exp(-0.5*(Y**2))
    eY=eY.reshape(len(eY),1)
    
    return offset + amp*eY*eX




"""
# =============================================================================
#  CONTROL BOARD - LOADING
# =============================================================================
"""

doload = 1
result_folder = "results/"

file_list = []  #Will contain list of files to analyze.


"""
# =============================================================================
#  For Publication :
# =============================================================================
"""

#PAINT data of Nanoruler. Comment and uncomment titles value depending on what you want to see.

### Multiple plot :
M = ""
titles = np.flip( ["gaussian/","ASTER-small/", "ASTER-wide/"] )
plot_titles = np.flip( ["Gaussian","70*70µm²","120*120µm²"] )
save_figs_folder = "Multiple/"



### Only gaussian (useful for cropping parts):
# titles = ["gaussian/"]
# plot_titles = ["Gaussian"]
# save_figs_folder = "Gaussian/"

"""
# =============================================================================
#  ---
# =============================================================================
"""


for pos, t in enumerate(titles):  #some data has result in a 180 folder , other don't..
    print(pos,t)
    if os.path.isdir( M + t + "/180/"):
        file_list.append( M + t +"/180/" ) 
    else:
        file_list.append( M + t +"/" ) 


docrop=0     #crop loaded result with CROP according to MX MY . 1 for gaussian crop on areas.
CROP = []   #x0 y0 dx dy - note: in original PIX value

if len(titles)==1:
    if titles == ["gaussian/"]:
        #note: multiple crop append doesnt work for file name. ..
        
        # CROP.append( np.array([ 40000,40000,15000,15000]) )  #Center
        CROP.append( np.array([ 24000,40000,15000,15000]) ) #left middle
        # CROP.append( np.array([12000,40000,15000,15000]) ) #left left middle
#        CROP.append( np.array([40000,24000,15000,15000]) ) # down middle
 
shift_nearzero = 1  #shift loaded data MX MY so that the minimum is at (PIX,PIX)

nanomax = None  #Limit final plot to this number of nanoruler (plot_nanor = 1)


"""
# =============================================================================
#  CONTROL BOARD - PLOT characteristics
# =============================================================================
"""
    
if not os.path.isdir( save_figs_folder ):
    os.mkdir( save_figs_folder ) #will fail if '/' is in titles.
    

resmin = 6 #200200: typ 6-12? else typ 5-11
resmax = 12

globaldensity_min = 0
globaldensity_max = 2800
nanordensity_min = 0
nanordensity_max = 20

nphmin = 1000
nphmax = 30000 
    
hist_rmin=5
hist_rmax=25
hist_bin = 40
hist_alpha = 0.4
hist_norm = True
mean_interd = True #True = > merge interd0 and 1 data together for hist.
hist_ccycles = ["C0","C1","C2","C3","C4","C5","C6"][:len(plot_titles)] #color cycle
#hist_ccycles = ["g","r","b"][:len(plot_titles)] #color cycle
hist_cecolors = ["C0","C1","C2","C3","C4","C5","C6"][:len(plot_titles)]
#hist_cecolors = ["g","r","b"][:len(plot_titles)]
hist_cmarker = 0
hist_type = 'stepfilled'  #'bar', 'barstacked', 'step', 'stepfilled'
hist_linewidth = 1
#hist_addcontour = 1  #plot a second histogram with contour only (good for eps file.) - TODO
#hist_addcontour_alpha = 0.9   # alpha of added contour (if added)


map_bin = 16   #for resol and nph maps.
map_bin2 = 30  #for size maps
map_binsame = 2  #for resol same map (same pixel, same size)
size_binsame = int(140/map_binsame) #size for resol same map in pixel (same pixel, same size)
recenter_binsame = True #for binsame: recenter each image on xy median.
fill_holes_binsame = True  #for binsame: fill holes (0) by local median
gfilter = 0   #for binsame: ass a gaussian blur to better mean data.

scatsize = 8 #for scatter plot (resolution)
graph_dx = 2
graph_dy = 2
graph_dr = 0.7
graph_ds = 20 # ok .. but high var at the end.
graph_dn = 50 # ok .. but high var at the end.


xcut = 11 #for plot along y, only take data in the range of median(x) +- xcut . If 0 takes all
ycut = 11 #for plot along x, only take data in the range of median(y) +- ycut . If 0 takes all. In µm (xdata/1000)

density_bin = 3  #for density map.

n_nanorulers = 100

map_ticks = True  #put real ticks in µm for res and nph maps.



plot_results = 0           #histogram of results. 1

print_results = 0           #print mean and std of some data (resolution, distance...)
plot_resmaps = 0            # Map and scatter of resolution
plot_resmapssame = 1        # Map --- of resolution, same pixel, same size. - useful to compare
plot_sizemaps =  0           # Map and scatter of size (interd) - ...

plot_resgraph = 0         # Resolution along axis
plot_resgraph_showcut = 0   #Show how data is cut in a scatter plot.
plot_resgraph2 = 0             #Scatter of resolution along axis
plot_res_rollmedian = 1         #Resolution along axis: rolling median and var. for X Y R. 1
plot_res_rollmedian_rsym = True   #for the r plot: do symetric graph or not. True
plot_res_rollmedian_rlim = 40  #limit for gaussian profile to stay in single molecule regime area.


plot_sizegraph2 = 0           #Scatter along axis of nanoruler size (total size)
plot_size_rollmedian_xy = 0         # Size along axis: rolling median and var
plot_size_rollmedian_r = 0        # Size along radial: rolling median and var. Along dr slices. 1
plot_size_rollmedian_s = 0        # Size along rings: rolling median and var
plot_size_rollmedian_n = 1       #size along radial, so that each part has same ndetection.

relative_error = True  #plot relative error on size (expected; 80nm or 50nm)
nside_median = 5
save_rollmedian_data = 1 #save data used for rollmedian graph - res and size.. (used for calculus of contrast in resolution.)

show_crop_scat = 1 #show crop effect: before and after.

plot_density = 0  # ok
plot_nph = 0   #ok - to complete for global.

plot_nanor = 0   #plot nanoruler images. 1 
plot_nanor_gflou = 0.9   # 0: nothing happen, else perform gaussian blur with this sigma value.


#NOT IMPLEMENTED YET ... unique(res) does not have expected shape ? because global is not exactly linked to local. (filter on local..)
# plot_nanor_makeimg = 1  #plot nanoruler images and rebuild images. with below parameters:
# plot_nanor_makeimg_gflou = 2  # 0: nothing happen, else perform gaussian blur with this sigma value.

#res_cmap = "hsv"
res_cmap = "nipy_spectral"
#res_cmap = None

"""
# =============================================================================
#  PROCESS
# =============================================================================
"""



if doload:
    globaldata_loaded = False
    globalresdata_loaded = False
    nph0_loaded = False
    MX0list = [] #nanoruler data
    MY0list = []
    SIGMAS0list = []
    EXC0list = []
    IMGS0list = []
    INTERD0list = []
    ANGLE0list = []
    NPH0list = []
    x0list = [] #global data used for nanoruler clustering
    y0list = []
    nph0list = []
    frame0list = []
    reslist = []   #global data of id association. (res: 3spots clusters , res2: individual spot cluster)
    res2list = []
    for somepath in file_list:
        someresult_folder = somepath + result_folder
        print('Loading previously saved data')
        MX0list.append( np.load( someresult_folder + 'MX0.npy') )
        MY0list.append( np.load( someresult_folder + 'MY0.npy') )
        SIGMAS0list.append( np.load( someresult_folder + 'SIGMAS0.npy') )
        EXC0list.append( np.load( someresult_folder + 'EXC0.npy') )
        IMGS0list.append( np.load( someresult_folder + 'IMGS0.npy') )
        INTERD0list.append( np.load( someresult_folder + 'INTERD0.npy') )
        ANGLE0list.append( np.load( someresult_folder + 'ANGLE0.npy') )
        
        if os.path.isfile( someresult_folder + 'NPH0.npy'):    
            NPH0list.append( np.load( someresult_folder + 'NPH0.npy') )
            nph0_loaded = True
        else: print('\t No NPH0 data found in ' + someresult_folder)
        
        if os.path.isfile( someresult_folder + 'x.npy'):
            print('\tAlso loading saved global data x y nph...')
            x0list.append( np.load( someresult_folder + 'x.npy') )
            y0list.append( np.load( someresult_folder + 'y.npy') )
            nph0list.append( np.load( someresult_folder + 'nph.npy') )
            frame0list.append( np.load( someresult_folder + 'frame.npy') )
            globaldata_loaded = True
        if os.path.isfile( someresult_folder + 'res.npy'):
            reslist.append( np.load( someresult_folder + 'res.npy') )
            res2list.append( np.load( someresult_folder + 'res2.npy') )
            globalresdata_loaded = True
            
        else: print('\t No global data found in ' + someresult_folder)

if len(NPH0list)!=len(MX0list):
    print('NOT ALL DATA CONTAINED NPH0 ! WE WILL NOW IGNORE NPH0 DATA.') #best to implement: only plot for loaded . But need to keep track on who belongs to who.
    nph0_loaded = False
if len(NPH0list)!=len(MX0list):
    print('NOT ALL DATA CONTAINED GLOBAL DATA !  WE WILL NOW IGNORE GLOBAL DATA.') #best to implement: only plot for loaded . But need to keep track on who belongs to who.
    globaldata_loaded = False
    

print('\n Creating data list , with eventual crop')
MXlist = []
MYlist = []
SIGMASlist = []
mSIGMASlist = []
EXClist = []
IMGSlist = []
INTERDlist = []
ANGLElist = []
NPHlist = []
xlist = []
ylist= []
nphlist = []
framelist = []
for pos, somepath in enumerate( file_list ):
    print('\tHandling results from' + somepath )
    #Data for plot : (any modification to it will not affect raw results_0) :
    SIGMAS = np.array( SIGMAS0list[pos] )  #Global data 
    mSIGMAS = np.mean(SIGMAS0list[pos],axis=1)
    EXC = np.array( EXC0list[pos] )
    INTERD = np.array( INTERD0list[pos] )
    ANGLE = np.array(ANGLE0list[pos])*180/np.pi
    MY = np.array(MY0list[pos])
    MX = np.array(MX0list[pos])
    IMGS = IMGS0list[pos].copy()
    if nph0_loaded:
        NPH = np.array(NPH0list[pos])
    if globaldata_loaded:
        x = np.array(x0list[pos])
        y = np.array(y0list[pos])
        nph = np.array(nph0list[pos])
        frame = np.array(frame0list[pos])

    if docrop:
        localcrop = CROP[pos]
#        save_figs_folder = "CROP_" + save_figs_folder + "last_title_is" + titles[-1]  # "220220_tint250_p76_tirfOS"         #"220220_tint100p68_tirfOS"
        save_figs_folder = save_figs_folder +'CROP_/'  # "220220_tint250_p76_tirfOS"         #"220220_tint100p68_tirfOS"
        if not os.path.isdir( save_figs_folder):
            os.mkdir( save_figs_folder)
        save_figs_folder = save_figs_folder + str(*CROP) + '/'
        if not os.path.isdir( save_figs_folder):
            os.mkdir( save_figs_folder)
            
        if not ( (localcrop is []) or (localcrop is None)):
            cvalid = (MX>localcrop[0])*(MX<localcrop[0]+localcrop[2])*(MY>localcrop[1])*(MY<localcrop[1]+localcrop[3])
            
            if show_crop_scat:
                plt.figure('Crop on'+str(localcrop))
                plt.scatter( MX, MY, c = cvalid)
                plt.savefig(save_figs_folder + 'CropScat.png')
                    
            if np.sum(cvalid)!=len(cvalid):
                print('Cropping nanoruler''s results with',CROP)
                SIGMAS = SIGMAS[cvalid]
                mSIGMAS = mSIGMAS[cvalid]
                EXC = EXC[cvalid]
                INTERD = INTERD[cvalid]
                ANGLE = ANGLE[cvalid]
                MY = MY[cvalid]
                MX = MX[cvalid]
                IMGS = [IMGS[i] for i in range(0,len(IMGS)) if cvalid[i] ]
                if nph0_loaded:
                    NPH = NPH[cvalid]
                    

                
#                if globaldata_loaded: 
#                    x = x[cvalid]
#                    y = y[cvalid]
#                    nph = nph[cvalid]
#                    frame = frame[cvalid]
                    
            else:
                print('Crop not needed')
        else:
            print('\t No crop here')
    MXlist.append( MX )
    MYlist.append( MY )
    SIGMASlist.append( SIGMAS )
    mSIGMASlist.append( mSIGMAS )
    EXClist.append( EXC )
    IMGSlist.append( IMGS )
    INTERDlist.append( INTERD )
    ANGLElist.append( ANGLE )
    if nph0_loaded:
        NPHlist.append( NPH )
    if globaldata_loaded:
        xlist.append( x )
        ylist.append( y )
        nphlist.append( nph )
        framelist.append( frame )
        
del( SIGMAS, mSIGMAS, MX, MY, EXC, INTERD, ANGLE, IMGS )
if nph0_loaded: del( NPH ) 
if globaldata_loaded: del( x, y, nph, frame )


print('\nStarting plot' )  
if not(( nanomax is None) or (nanomax==0)):
    print('SELECTING ONLY FIRST ' + str(nanomax) + ' Nanorulers for plot')

    SIGMASlist = [ l[:nanomax] for l in SIGMASlist ]
    mSIGMASlist = [ l[:nanomax] for l in mSIGMASlist ]
    EXClist = [ l[:nanomax] for l in EXClist ]
    INTERDlist = [ l[:nanomax] for l in INTERDlist ]
    ANGLElist = [ l[:nanomax] for l in ANGLElist ]
    MYlist = [ l[:nanomax] for l in MYlist ]
    MXlist = [ l[:nanomax] for l in MXlist ]
    IMGSlist = [ l[:nanomax] for l in IMGSlist ]
    #TODO

nspots = 3
if np.min(EXClist[0])==0:
    print('EXClist minimum is 0: considering 2 spots nanorulers')
    nspots = 2


        
        
if plot_results:
    print('\t Plotting histograms' )
    
    if nspots==3:
        figf, axf = plt.subplots(2,2,figsize=(12,9))
        figf.canvas.set_window_title(' Some final Datas ')
        plt.subplot(221)
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            plt.hist(mSIGMASlist[pos],bins=hist_bin, range=(hist_rmin,hist_rmax), color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type, linewidth = hist_linewidth, edgecolor=ec )

        plt.title('Histogram of localization precisions ')
        plt.xlabel('Localization precision - nm')
        plt.ylabel('Occurence')
        plt.legend( plot_titles )
        
        
        plt.subplot(222)
        plt.title('Histogram of excentrism ')
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            if np.min(EXClist[pos]) >= 0:
                rmin = 0
            else:
                rmin = -18
            plt.hist(EXClist[pos],bins=hist_bin, range=(rmin,18) ,color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type, linewidth = hist_linewidth, edgecolor=ec )
        plt.xlabel('Excentrism - nm')
        plt.ylabel('Occurence')
        plt.legend( plot_titles )
        
        plt.subplot(223)
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            if mean_interd:
                plt.hist(np.ravel(INTERDlist[pos][:,[0,1]]),bins=hist_bin, range=(20,60), color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type, linewidth = hist_linewidth, edgecolor=ec )
            else:
                #todo: differentiate color..
                plt.hist(INTERDlist[pos][:,0],bins=hist_bin, range=(20,60), color=color, alpha=hist_alpha, density=hist_norm )
                plt.hist(INTERDlist[pos][:,1],bins=hist_bin, range=(20,60), color=color, alpha=hist_alpha, density=hist_norm )
        plt.title('Histogram of short distances ')
        plt.xlabel('Distance between cluster - nm')
        plt.ylabel('Occurence')
        # plt.legend( plot_titles )
        
        plt.subplot(224)
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            plt.hist(INTERDlist[pos][:,2],bins=hist_bin, range=(60,100), color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type, linewidth = hist_linewidth, edgecolor=ec )
        plt.title('Histogram of nanoruler size ')
        plt.xlabel('Size - nm')
        plt.ylabel('Occurence')
        plt.legend( plot_titles )
        plt.savefig( save_figs_folder  + 'results.png')
        plt.savefig( save_figs_folder  + 'results.eps')
        plt.savefig( save_figs_folder  + 'results.pdf')
        
    else:  #for nspot = 2. 

        figf, axf = plt.subplots(1,2,figsize=(7,3))
        figf.canvas.set_window_title(' Some final Datas ')
        plt.subplot(121)
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            plt.hist(mSIGMASlist[pos],bins=hist_bin, range=(hist_rmin,hist_rmax), color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type , linewidth = hist_linewidth, edgecolor=ec)
        plt.title('Histogram of localization precisions ')
        plt.xlabel('Localization precision - nm')
        plt.ylabel('Occurence')
        plt.legend( plot_titles )
        
        plt.subplot(122)
        for pos in range(0,len(MXlist)):
            hist_cmarker = ((hist_cmarker+1)%len(hist_ccycles))
            color = hist_ccycles[ hist_cmarker ]
            ec = hist_cecolors[ hist_cmarker]
            plt.hist(INTERDlist[pos][:,0],bins=hist_bin, range=(35,65), color=color, alpha=hist_alpha, density=hist_norm, histtype=hist_type, linewidth = hist_linewidth, edgecolor=ec )
        plt.title('Histogram of nanoruler size ')
        plt.xlabel('Size - nm')
        plt.ylabel('Occurence')
        plt.legend( plot_titles )
        plt.savefig( save_figs_folder  + 'results.png')
        plt.savefig( save_figs_folder  + 'results.eps')
        plt.savefig( save_figs_folder  + 'results.pdf')
        
if print_results:
    print('RESULTS ---')
    for pos in range(0,len(MXlist)):
        d = INTERDlist[pos]
        sig = mSIGMASlist[pos]
        print(plot_titles[pos])
        print('\t Sigma ---')
        print('\t\t mean:' , np.nanmean(sig) )
        print('\t\t std:' , np.nanstd(sig) )
        if nspots ==2:
            print('\t Dist nanorulers ---')
            print('\t\t mean:' , np.nanmean(d[:,0]) )
            print('\t\t std:' , np.nanstd(d[:,0]) )
        elif nspots ==3:
            print('\t Dist interd nanorulers ---')
            print('\t\t mean:' , np.nanmean(d[:,[0,1]]) )
            print('\t\t std:' , np.nanstd(d[:,[0,1]]) )
            print('\t Dist total nanorulers ---')
            print('\t\t mean:' , np.nanmean(d[:,2]) )
            print('\t\t std:' , np.nanstd(d[:,2]) )
            exc = EXClist[pos]
            print('\t Exc ---')
            print('\t\t mean:' , np.nanmean(exc) )
            print('\t\t std:' , np.nanstd(exc) )
            
    
        
    
if plot_resmaps:
    
    print('\t Plotting Resolution maps' ) 
    
    fig = plt.figure('Res image',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        img_sigma = vhist2D(localx, localy, mSIGMASlist[pos], method='median', nb=(map_bin,map_bin), rng=None, nanzero = 0)
        im = axs[pos].imshow(img_sigma, cmap =res_cmap, vmin=resmin, vmax=resmax)
        dtick = 20
        tickvaluey = (localy.max()//map_bin)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
        tickvaluex = (localx.max()//map_bin)/1000
        tickX = np.arange(0, map_bin-1, dtick/tickvaluex )
        tickY = np.arange(0, map_bin-1, dtick/tickvaluey )
        axs[pos].set_xticks( tickX )
        axs[pos].set_yticks( tickY )
        axs[pos].set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
        axs[pos].set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
        axs[pos].set_xlabel('field - µm')
        
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    plt.savefig( save_figs_folder  + 'resmax='+str(resmax)+ 'results_MapRes.png')
    plt.savefig( save_figs_folder  + 'resmax='+str(resmax)+ 'results_MapRes.eps')
        
    
    
    fig = plt.figure('Res image Scatter',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        im = axs[pos].scatter(localx/1000,localy/1000, s=scatsize, c= mSIGMASlist[pos], cmap=res_cmap, vmin=resmin, vmax=resmax)
#        axs[pos].set_aspect('equal', adjustable='box')
        axs[pos].set_xlabel('field - µm')
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Resolution -nm', rotation=270)
    plt.savefig( save_figs_folder  + 'results_MapScat-s='+str(scatsize)+'_resmax='+str(resmax)+'.png')
    plt.savefig( save_figs_folder  + 'results_MapScat-s='+str(scatsize)+'_resmax='+str(resmax)+'.eps')
    
    
if plot_resmapssame:
    fig = plt.figure('Res image Same size',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    ps = map_binsame #pixel size in µm for incoming maps.
    ms = size_binsame #map size in pixel for incoming maps.
    mxmax, mymax = size_binsame*map_binsame, size_binsame*map_binsame
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        localv = mSIGMASlist[pos]
        nbs = (mxmax/1000/map_binsame, mxmax/1000/map_binsame)
        img_sigma = myvhist2D(localx/1000,localy/1000,localv, method='median', pxy=(ps,ps), shape_=(ms,ms), nanzero = 1, recenter=recenter_binsame)
        if fill_holes_binsame:
            img_sigma = fill_img_holes( img_sigma, 'median', 0, 1)
        if gfilter:
            img_sigma = gf( img_sigma, gfilter)
        im = axs[pos].imshow(img_sigma, cmap =res_cmap, vmin=resmin, vmax=resmax)
        dtick = 20
        tickvaluey = ms # size in pixel
        tickvaluex = ms
        tickX = np.arange(0, ms, dtick/ps )
        tickY = np.arange(0, ms, dtick/ps )
        axs[pos].set_xticks( tickX )
        axs[pos].set_yticks( tickY )
        axs[pos].set_xticklabels( [str(int(round(n,0))) for n in tickX*ps] )
        axs[pos].set_yticklabels( [str(int(round(n,0))) for n in tickY*ps] )
        axs[pos].set_xlabel('field - µm')
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Resolution -nm', rotation=270)
    
    plt.savefig( save_figs_folder  + 'results_MapResSame='+'_mapbinsame='+str(map_binsame)+'_gf-'+str(gfilter)+'resmax='+str(resmax)+'.png')
    plt.savefig( save_figs_folder  + 'results_MapResSame='+'_mapbinsame='+str(map_binsame)+'_gf-'+str(gfilter)+'resmax='+str(resmax)+'.eps')

    
    fig = plt.figure('Res image Scatter Same',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
#    mxmax = np.max([ l.max() for l in MXlist])
#    mymax = np.max([ l.max() for l in MYlist])
    mxmax, mymax = size_binsame*map_binsame, size_binsame*map_binsame
    if recenter_binsame:
#        listmax = np.argmax([ l.max() for l in MXlist])
#        center = ( np.median(MXlist[listmax]) , np.median(MYlist[listmax]))
        center = ( int(size_binsame*map_binsame*1000)/2, int(size_binsame*map_binsame*1000)/2 )
        
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        if recenter_binsame:
            localc = ( np.median(localx) , np.median(localy))
            localx = localx - localc[0] + center[0]
            localy = localy - localc[1] + center[1]
        im = axs[pos].scatter(localx/1000,localy/1000, s=scatsize, c= mSIGMASlist[pos], cmap=res_cmap, vmin=resmin, vmax=resmax, alpha=0.9)
#        axs[pos].set_aspect('equal', adjustable='box')
        axs[pos].set_xlabel('field - µm')
        axs[pos].set_xlim(0,mxmax)
        axs[pos].set_ylim(0,mymax)
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Resolution -nm', rotation=270)
    plt.savefig( save_figs_folder  + 'results_MapScatSame-s='+str(scatsize)+'_mapbinsame='+str(map_binsame)+'_gf-'+str(gfilter)+'.png')
    plt.savefig( save_figs_folder  + 'results_MapScatSame-s='+str(scatsize)+'_mapbinsame='+str(map_binsame)+'_gf-'+str(gfilter)+'.eps')



if plot_resgraph:
    print('Plotting resolution Graphs')

    plt.figure('field resolution graph - X ycut=' + str(ycut) )
    dx = graph_dx
    plt.title('Field dependance of resolution')
    for pos in range(0,len(MXlist)):
        mSIGMAS = mSIGMASlist[pos]
        MX = MXlist[pos] / 1000   #in µm
        if ycut!=0:
            MY = MYlist[pos]/1000
            ym = np.median( MY )
            valid = np.abs(MY-ym)<=ycut

            if plot_resgraph_showcut:
                plt.figure('Cut verification Ycut' + str(pos))
                plt.title( plot_titles[pos] )
                plt.scatter(MX,MY, alpha=0.5)
                plt.scatter(MX[valid],MY[valid],alpha=0.5)
                plt.figure('field resolution graph - X ycut=' + str(ycut) )
            
            MX=MX[valid]
            mSIGMAS = mSIGMAS[valid]

        MX = MX - np.median(MX) #we center to compare one data to another.
        xsort = np.argsort( MX )
        MXsort = MX[xsort]
        SIGMAsort = (mSIGMAS)[xsort]
        
        Xcoord, Nnanoruler_in_dx = np.unique( MXsort//dx, return_counts=True)
        Xcoord = Xcoord*dx
        count = 0
        Xvalue = []
        for n_nano in Nnanoruler_in_dx:
            Xvalue.append( np.median( SIGMAsort[count:count+n_nano] ) )
            count = count + n_nano
        Xcoord = np.array( Xcoord )
        Xvalue = np.array( Xvalue )
        plt.plot( Xcoord, Xvalue, label=plot_titles[pos])
    plt.xlabel('position - µm')
    plt.ylabel('Median resolution - nm')
    plt.legend()
    plt.savefig( save_figs_folder  + 'results_Graphdx='+str(dx)+'ycut='+str(ycut)+'.png')
    plt.savefig( save_figs_folder  + 'results_Graphdx='+str(dx)+'ycut='+str(ycut)+'.eps')


    plt.figure('field resolution graph - Y xcut=' + str(xcut) )
    dy = graph_dy
    plt.title('Field dependance of resolution')
    for pos in range(0,len(MXlist)):
        mSIGMAS = mSIGMASlist[pos]
        MY = MYlist[pos] / 1000   #in µm
        if ycut!=0:
            MX = MXlist[pos]/1000
            xm = np.median( MX )
            valid = np.abs(MX-xm)<=xcut
            
            if plot_resgraph_showcut:
                plt.figure('Cut verification Xcut' + str(pos))
                plt.title( plot_titles[pos] )
                plt.scatter(MX,MY, alpha=0.5)
                plt.scatter(MX[valid],MY[valid],alpha=0.5)
                plt.figure('field resolution graph - Y xcut=' + str(xcut) )
            
            MY=MY[valid]
            mSIGMAS = mSIGMAS[valid]
            
        MY = MY - np.median(MY) #we center to compare one data to another.
        ysort = np.argsort( MY )
        MYsort = MY[ysort]
        SIGMAsort = (mSIGMAS)[ysort]
        
        Ycoord, Nnanoruler_in_dy = np.unique( MYsort//dy, return_counts=True)
        Ycoord = Ycoord*dy
        count = 0
        Yvalue = []
        for n_nano in Nnanoruler_in_dy:
            Yvalue.append( np.median( SIGMAsort[count:count+n_nano] ) )
            count = count + n_nano
        Ycoord = np.array( Ycoord )
        Yvalue = np.array( Yvalue )
        plt.plot( Ycoord, Yvalue, label=plot_titles[pos])
    plt.xlabel('position - µm')
    plt.ylabel('Median resolution - nm')
    plt.legend()
    plt.savefig( save_figs_folder  + 'results_Graphdy='+str(dy)+'ycut='+str(xcut)+'.png')
    plt.savefig( save_figs_folder  + 'results_Graphdy='+str(dy)+'ycut='+str(xcut)+'.eps')



if plot_resgraph2:
    
    plt.figure('Scatter X - field res graph',figsize=(10,5))
    plt.ylabel(' resolution - nm')
    plt.xlabel('axial position - µm')
    for pos in range(0,len(MXlist)):
        MX = MXlist[pos]
        mSIGMAS = mSIGMASlist[pos]
        if ycut!=0:
            MY = MYlist[pos]/1000
            ym = np.median( MY )
            valid = np.abs(MY-ym)<=ycut
            MX=MX[valid]
            mSIGMAS = mSIGMAS[valid]
        MX = (MX - np.median(MX))/1000
        plt.scatter(MX, mSIGMAS,alpha=0.8,s=4, label= plot_titles[pos])
    plt.legend()
    plt.ylim(resmin,resmax)
    plt.title('Scatter plot of resolution')
    plt.savefig( save_figs_folder  + 'GraphResolScat_X_resmax='+str(resmax) + 'results.png')
    plt.savefig( save_figs_folder  + 'GraphResolScat_X_resmax='+str(resmax) + 'results.eps')
    
    plt.figure('Scatter Y - field res graph',figsize=(10,5))
    plt.ylabel(' resolution - nm')
    plt.xlabel('axial position - µm')
    for pos in range(0,len(MXlist)):
        MY = MYlist[pos]
        mSIGMAS = mSIGMASlist[pos]
        if xcut!=0:
            MX = MXlist[pos]/1000
            xm = np.median( MX )
            valid = np.abs(MX-xm)<=xcut
            MY=MY[valid]
            mSIGMAS = mSIGMAS[valid]
        MY = (MY - np.median(MY))/1000
        plt.scatter(MY, mSIGMAS,alpha=0.8,s=4, label= plot_titles[pos])
    plt.legend()
    plt.ylim(resmin,resmax)
    plt.title('Scatter plot of resolution')
    plt.savefig( save_figs_folder  + 'GraphResolScat_Y_resmax='+str(resmax) + 'results.png')
    plt.savefig( save_figs_folder  + 'GraphResolScat_Y_resmax='+str(resmax) + 'results.eps')


if plot_res_rollmedian:
    print('Plotting rolling median + var resolution along axis')
    
    dx = graph_dx*1000
    prefix = save_figs_folder + 'results_GraphRolldx='+str(dx)+'ycut='+str(ycut)
    plt.figure('Scatter X - field roll res graph',figsize=(7,4))
    plt.ylabel(' resolution - nm')
    plt.xlabel('axial position - µm')
    xlim = [0,0]
    #note: rolling median 
    for pos in range(0,len(MXlist)):
        MX = MXlist[pos]
        mSIGMAS = mSIGMASlist[pos]
        if ycut!=0:
            MY = MYlist[pos]/1000
            ym = np.median( MY )
            valid = np.abs(MY-ym)<=ycut
            MX=MX[valid]
            mSIGMAS = mSIGMAS[valid]
        
        mxs = MX.argsort()
        MXsort = MX[mxs]
        mSIGMASsort = mSIGMAS[mxs]
        mxstep, mxstepcount = np.unique( MXsort//dx, return_counts = True )
        mxstep_cumcount = np.concatenate(([0], np.cumsum( mxstepcount) ))

        SIGmed = []
        SIGstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, mxvalue in enumerate(mxstepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mxstepcount)-1, cpos + nside_median)
            pos_init = mxstep_cumcount[pos0]
            pos_final = mxstep_cumcount[posf]
            
            localsigma = mSIGMASsort[ pos_init : pos_final ]
            SIGmed.append( np.nanmedian(localsigma) )
#            SIGstd.append( np.std(localsigma) / np.sqrt(2*nside_median+1) ) #not exactly accurate (we measure median of data... not obvious how to take this std..) + not accurate at borders.
            SIGstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        
        SIGmed = np.array( SIGmed )
        SIGstd = np.array( SIGstd )
        xarr = ( mxstep*dx )/1000 #µm unit, recentered
        xarr = xarr - np.median(xarr)
        xlim = [ min(xlim[0],xarr.min())  , max(xlim[1],xarr.max()) ]
        plt.plot( xarr, SIGmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(xarr, SIGmed-SIGstd, SIGmed+SIGstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s 
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/' + here + '.data_xarr.npy',xarr )
            np.save(prefix + 'rollmed_data/' + here + '.data_med.npy',SIGmed )
            np.save(prefix + 'rollmed_data/' + here + '.data_std.npy',SIGstd )
    
    plt.legend()
    plt.ylim(resmin,resmax)
    plt.title('Scatter plot of resolution')
    plt.xlim( xlim )
    plt.savefig( save_figs_folder + 'results_GraphRolldx='+str(dx)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder + 'results_GraphRolldx='+str(dx)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.eps')
    
    
    
    dy = graph_dy*1000
    prefix = save_figs_folder  + 'results_GraphRolldy='+str(dy)+'ycut='+str(xcut)
    plt.figure('Scatter Y - field roll res graph',figsize=(7,4))
    plt.ylabel(' resolution - nm')
    plt.xlabel('axial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        MY = MYlist[pos]
        mSIGMAS = mSIGMASlist[pos]
        if xcut!=0:
            MX = MXlist[pos]/1000
            xm = np.median( MX )
            valid = np.abs(MX-xm)<=xcut
            MY=MY[valid]
            mSIGMAS = mSIGMAS[valid]
            
        mys = MY.argsort()
        mysort = MY[mys]
        mSIGMASsort = mSIGMAS[mys]
        mystep, mystepcount = np.unique( mysort//dy, return_counts = True )
        mystep_cumcount = np.concatenate(([0], np.cumsum( mystepcount) ))
        
        SIGmed = []
        SIGstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, MYvalue in enumerate(mystepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mystepcount)-1, cpos + nside_median)
            pos_init = mystep_cumcount[pos0]
            pos_final = mystep_cumcount[posf]

            localsigma = mSIGMASsort[ pos_init : pos_final ]
            SIGmed.append( np.nanmedian(localsigma) )
            SIGstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        SIGmed = np.array( SIGmed )
        SIGstd = np.array( SIGstd )
        
        yarr = ( mystep*dy )/1000 #µm unit, recentered
        yarr = yarr - np.median(yarr)
        xlim = [ min(xlim[0],yarr.min())  , max(xlim[1],yarr.max()) ]
        plt.plot( yarr, SIGmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(yarr, SIGmed-SIGstd, SIGmed+SIGstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/' + here + '.data_xarr.npy',yarr )
            np.save(prefix + 'rollmed_data/' + here + '.data_med.npy',SIGmed )
            np.save(prefix + 'rollmed_data/' + here + '.data_std.npy',SIGstd )
        
    plt.legend()
    plt.ylim(resmin,resmax)
    plt.title('Y Rollmed plot of resolution')
    plt.xlim( xlim )
    plt.savefig( save_figs_folder + 'results_GraphRolldy='+str(dy)+'xcut='+str(xcut)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder + 'results_GraphRolldy='+str(dy)+'xcut='+str(xcut)+'nside='+str(nside_median)+'.eps')
    
        
        
        #TOCHECK 
    dr = graph_dr*1000
    prefix = save_figs_folder  + 'results_GraphRolldr='+str(dr)
    plt.figure(' R - field roll res graph',figsize=(7,4))
    plt.ylabel(' resolution - nm')
    plt.xlabel(' radial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        
        MR = np.hypot( MYlist[pos] - np.median(MYlist[pos]) , MXlist[pos] - np.median(MXlist[pos]) )
        mSIGMAS = mSIGMASlist[pos]
            
        mrs = MR.argsort()
        mrsort = MR[mrs]
        mSIGMASsort = mSIGMAS[mrs]
        mstep, mstepcount = np.unique( mrsort//dr, return_counts = True )
        mstep_cumcount = np.concatenate(([0], np.cumsum( mstepcount) ))
        
        SIGmed = []
        SIGstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, MYvalue in enumerate(mstepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mstepcount)-1, cpos + nside_median)
            pos_init = mstep_cumcount[pos0]
            pos_final = mstep_cumcount[posf]

            localsigma = mSIGMASsort[ pos_init : pos_final ]
            SIGmed.append( np.nanmedian(localsigma) )
            SIGstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        SIGmed = np.array( SIGmed )
        SIGstd = np.array( SIGstd )
        
        Rarr = ( mstep*dr )/1000 #µm unit, recentered
        Rarr = Rarr - 0 
        
        if plot_res_rollmedian_rsym:
            Rarr2 = np.concatenate((-np.flip(Rarr),Rarr))
            SIGmed2 = np.concatenate((np.flip(SIGmed),SIGmed))
            SIGstd2 = np.concatenate((np.flip(SIGstd),SIGstd))
            xlim = [ min(-xlim[1],Rarr2.min())  , max(xlim[1],Rarr2.max()) ]
            if plot_res_rollmedian_rlim>1:
                if titles[pos] == "Gauss4_det1.2":
                    print('Keeping gaussian rmed roll in SMLM area.')
                    valid = np.abs(Rarr2) < plot_res_rollmedian_rlim
                    Rarr2 = Rarr2[ valid ]
                    SIGmed2 = SIGmed2[ valid ]
                    SIGstd2 = SIGstd2[ valid ]

        else:
            Rarr2, SIGmed2, SIGstd2 = Rarr, SIGmed, SIGstd
            xlim = [ min(xlim[0],Rarr2.min())  , max(xlim[1],Rarr2.max()) ]
        
        plt.plot( Rarr2, SIGmed2, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(Rarr2, SIGmed2-SIGstd2, SIGmed2+SIGstd2, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/R' + here + '.data_xarr.npy',Rarr )
            np.save(prefix + 'rollmed_data/R' + here + '.data_med.npy',SIGmed )
            np.save(prefix + 'rollmed_data/R' + here + '.data_std.npy',SIGstd )
        
    plt.legend()
    plt.ylim(resmin,resmax)
    plt.xlim( xlim )
    plt.title('R rollmed plot of resolution')
    plt.savefig( save_figs_folder  + 'results_GraphRolldr='+str(dr)+'.png')
    plt.savefig( save_figs_folder  + 'results_GraphRolldr='+str(dr)+'.eps')



# =============================================================================
#   SIZE 
# =============================================================================

if nspots==3:
    column = 2
    sizemin = 65
    sizemax = 95
    expected_size = 80.0
elif nspots==2:
    column = 0 #where to take in INTERDlist the total size of nanorulers.
    sizemin = 40
    sizemax = 50
    expected_size = 50
else:
    print('Unknown number of spot ???')


if plot_sizemaps:
    print('\t Plotting Size maps' )         
    fig = plt.figure('Size image',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        img_size = vhist2D(localx, localy, INTERDlist[pos][:,column], method='median', nb=(map_bin2,map_bin2), rng=None, nanzero = 0)
        im = axs[pos].imshow(img_size, cmap =res_cmap, vmin=sizemin, vmax=sizemax)
        dtick = 20
        tickvaluey = (localy.max()//map_bin2)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
        tickvaluex = (localx.max()//map_bin2)/1000
        tickX = np.arange(0, map_bin-1, dtick/tickvaluex )
        tickY = np.arange(0, map_bin-1, dtick/tickvaluey )
        axs[pos].set_xticks( tickX )
        axs[pos].set_yticks( tickY )
        axs[pos].set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
        axs[pos].set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
        axs[pos].set_xlabel('field - µm')
        
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    plt.savefig( save_figs_folder  + 'sizemax='+str(sizemax)+ 'results_MapSize.png')
    plt.savefig( save_figs_folder  + 'sizemax='+str(sizemax)+ 'results_MapSize.eps')
        
    
    
    fig = plt.figure('Size image Scatter',figsize=(4*len(MXlist),4))
    axs = fig.subplots(1,len(MXlist), sharex=False , sharey=False )
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    for pos in range(0,len(MXlist)):
        localx = MXlist[pos]
        localy = MYlist[pos]
        im = axs[pos].scatter(localx/1000,localy/1000, s=scatsize, c= INTERDlist[pos][:,column], cmap=res_cmap, vmin=sizemin, vmax=sizemax)
#        axs[pos].set_aspect('equal', adjustable='box')
        axs[pos].set_xlabel('field - µm')
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Size -nm', rotation=270)
    plt.savefig( save_figs_folder  + 'results_MapScatSize-s='+str(scatsize)+'_sizemax='+str(sizemax)+'.png')
    plt.savefig( save_figs_folder  + 'results_MapScatSize-s='+str(scatsize)+'_sizemax='+str(sizemax)+'.eps')
    


if plot_sizegraph2:
        
    plt.figure('Scatter X - field size graph',figsize=(10,5))
    plt.ylabel(' size - nm')
    plt.xlabel('axial position - µm')
    for pos in range(0,len(MXlist)):
        MX = MXlist[pos]
        mSIZE = INTERDlist[pos][:,column]
        if ycut!=0:
            MY = MYlist[pos]/1000
            ym = np.median( MY )
            valid = np.abs(MY-ym)<=ycut
            MX=MX[valid]
            mSIZE = mSIZE[valid]
        MX = (MX - np.median(MX))/1000
        plt.scatter(MX, mSIZE,alpha=0.8,s=4, label= plot_titles[pos])
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.title('Scatter plot of size')
    plt.savefig( save_figs_folder  + 'GraphSizeScat_X_sizemax='+str(sizemax) + 'results.png')
    plt.savefig( save_figs_folder  + 'GraphSizeScat_X_sizemax='+str(sizemax) + 'results.eps')
    
    plt.figure('Scatter Y - field size graph',figsize=(10,5))
    plt.ylabel(' size - nm')
    plt.xlabel('axial position - µm')
    for pos in range(0,len(MXlist)):
        MY = MYlist[pos]
        mSIZE = INTERDlist[pos][:,column]
        if xcut!=0:
            MX = MXlist[pos]/1000
            xm = np.median( MX )
            valid = np.abs(MX-xm)<=xcut
            MY=MY[valid]
            mSIZE = mSIZE[valid]
        MY = (MY - np.median(MY))/1000
        plt.scatter(MY, mSIZE,alpha=0.8,s=4, label= plot_titles[pos])
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.title('Scatter plot of size')
    plt.savefig( save_figs_folder  + 'GraphSizeScat_Y_resmax='+str(resmax) + 'results.png')
    plt.savefig( save_figs_folder  + 'GraphSizeScat_Y_resmax='+str(resmax) + 'results.eps')




if plot_size_rollmedian_xy:

    print('Plotting rolling median + var size along axis')
    
    dx = graph_dx*1000
    prefix = save_figs_folder + 'results_GraphRollSizedx='+str(dx)+'ycut='+str(ycut)
    plt.figure(' X - field roll size graph',figsize=(5,4))
    if relative_error:
        plt.ylabel(' Relative size error - %')
        plt.title('Rolling median plot of size error')
    else:
        plt.ylabel(' size - nm')
        plt.title('Rolling median plot of size')
    plt.xlabel('axial position - µm')
    xlim = [0,0]
    #note: rolling median 
    for pos in range(0,len(MXlist)):
        MX = MXlist[pos]
        mSIZE = INTERDlist[pos][:,column]
        mSIGMAS = mSIGMASlist[pos] #for std calculus..
        if ycut!=0:
            MY = MYlist[pos]/1000
            ym = np.median( MY )
            valid = np.abs(MY-ym)<=ycut
            MX=MX[valid]
            mSIZE = mSIZE[valid]
            mSIGMAS = mSIGMAS[valid]
        
        mxs = MX.argsort()
        MXsort = MX[mxs]
        mSIZEsort = mSIZE[mxs]
        
        mxstep, mxstepcount = np.unique( MXsort//dx, return_counts = True )
        mxstep_cumcount = np.concatenate(([0], np.cumsum( mxstepcount) ))
#        nside_median = 5
        SIZEmed = []
        SIZEstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, mxvalue in enumerate(mxstepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mxstepcount)-1, cpos + nside_median)
            pos_init = mxstep_cumcount[pos0]
            pos_final = mxstep_cumcount[posf]
            
            localsigma = mSIZEsort[ pos_init : pos_final+1 ]
            SIZEmed.append( np.nanmedian(localsigma) )
            SIZEstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )

        
        SIZEmed = np.array( SIZEmed )
        SIZEstd = np.array( SIZEstd )
        xarr = ( mxstep*dx )/1000 #µm unit, recentered
        xarr = xarr - np.median(xarr)
        xlim = [ min(xlim[0],xarr.min())  , max(xlim[1],xarr.max()) ]
        if relative_error:
            SIZEmed = (expected_size - SIZEmed)/expected_size * 100 # in %
            SIZEstd = SIZEstd/expected_size *100
            sizemax, sizemin = 10, -3
        plt.plot( xarr, SIZEmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(xarr, SIZEmed-SIZEstd, SIZEmed+SIZEstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s 
            if not os.path.isdir( prefix + 'rollsizemed_data/'):
                os.mkdir( prefix + 'rollsizemed_data/' )
            np.save(prefix + 'rollsizemed_data/X' + here + '.data_xarr.npy',xarr )
            np.save(prefix + 'rollsizemed_data/X' + here + '.data_med.npy',SIZEmed )
            np.save(prefix + 'rollsizemed_data/X' + here + '.data_std.npy',SIZEstd )
    
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.xlim( xlim )
    plt.savefig( save_figs_folder + 'results_GraphRollSizedx='+str(dx)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder + 'results_GraphRollSizedx='+str(dx)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.eps')
    
    
    
    dy = graph_dy*1000
    prefix = save_figs_folder  + 'results_GraphRolldy='+str(dy)+'ycut='+str(xcut)
    plt.figure(' Y - field roll res graph',figsize=(5,4))
    if relative_error:
        plt.ylabel(' Relative size error - %')
        plt.title('Rolling median plot of size error')
    else:
        plt.ylabel(' size - nm')
        plt.title('Rolling median plot of size')
    plt.xlabel('axial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        MY = MYlist[pos]
        mSIZE = INTERDlist[pos][:,column]
        if xcut!=0:
            MX = MXlist[pos]/1000
            xm = np.median( MX )
            valid = np.abs(MX-xm)<=xcut
            MY=MY[valid]
            mSIZE = mSIZE[valid]
            
        mys = MY.argsort()
        mysort = MY[mys]
        mSIZEsort = mSIZE[mys]
        mystep, mystepcount = np.unique( mysort//dy, turn_counts = True )
        mystep_cumcount = np.concatenate(([0], np.cumsum( mystepcount) ))
#        nside_median = 5
        SIZEmed = []
        SIZEstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, MYvalue in enumerate(mystepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mystepcount)-1, cpos + nside_median)
            pos_init = mystep_cumcount[pos0]
            pos_final = mystep_cumcount[posf]

            localsigma = mSIZEsort[ pos_init : pos_final+1 ]
            SIZEmed.append( np.nanmedian(localsigma) )
            SIZEstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )

        SIZEmed = np.array( SIZEmed )
        SIZEstd = np.array( SIZEstd )
        
        yarr = ( mystep*dy )/1000 #µm unit, recentered
        yarr = yarr - np.median(yarr)
        xlim = [ min(xlim[0],yarr.min())  , max(xlim[1],yarr.max()) ]
        if relative_error:
            SIZEmed = (expected_size - SIZEmed)/expected_size * 100 # in %
            SIZEstd = SIZEstd/expected_size * 100
            sizemax, sizemin = 10, -3
        plt.plot( yarr, SIZEmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(yarr, SIZEmed-SIZEstd, SIZEmed+SIZEstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian size data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollsizemed_data/'):
                os.mkdir( prefix + 'rollsizemed_data/' )
            np.save(prefix + 'rollsizemed_data/Y' + here + '.data_xarr.npy',yarr )
            np.save(prefix + 'rollsizemed_data/Y' + here + '.data_med.npy',SIZEmed )
            np.save(prefix + 'rollsizemed_data/Y' + here + '.data_std.npy',SIZEstd )
        
        
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.xlim( xlim )
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedy='+str(dy)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedy='+str(dy)+'ycut='+str(ycut)+'nside='+str(nside_median)+'.eps')


if plot_size_rollmedian_r:

    dr = graph_dr*1000
    prefix = save_figs_folder  + 'results_GraphRolldr='+str(dr)
    plt.figure(' R - field roll size graph',figsize=(5,4))
    plt.ylabel(' size - nm')
    plt.xlabel('axial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        
        MR = np.hypot( MYlist[pos] - np.median(MYlist[pos]) , MXlist[pos] - np.median(MXlist[pos]) )
        mSIZE = INTERDlist[pos][:,column]
            
        mrs = MR.argsort()
        mrsort = MR[mrs]
        mSIZEsort = mSIZE[mrs]
        mstep, mstepcount = np.unique( mrsort//dr, return_counts = True )
        mstep_cumcount = np.concatenate(([0], np.cumsum( mstepcount) ))
        
        SIZEmed = []
        SIZEstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, Mvalue in enumerate(mstepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mstepcount)-1, cpos + nside_median)
            pos_init = mstep_cumcount[pos0]
            pos_final = mstep_cumcount[posf]

            localsigma = mSIZEsort[ pos_init : pos_final ]
            SIZEmed.append( np.nanmedian(localsigma) )
            SIZEstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        SIZEmed = np.array( SIZEmed )
        SIZEstd = np.array( SIZEstd )
        if relative_error:
            SIZEmed = (expected_size - SIZEmed)/expected_size * 100 # in %
            SIZEstd = SIZEstd/expected_size * 100
            sizemax, sizemin = 10, -3
            plt.ylabel(' relative size error - %')
            
        Rarr = ( mstep*dr )/1000 #µm unit, recentered
        Rarr = Rarr - 0 
        xlim = [ min(xlim[0],Rarr.min())  , max(xlim[1],Rarr.max()) ]
        plt.plot( Rarr, SIZEmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(Rarr, SIZEmed-SIZEstd, SIZEmed+SIZEstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/R' + here + '.data_xarr.npy',Rarr )
            np.save(prefix + 'rollmed_data/R' + here + '.data_med.npy',SIZEmed )
            np.save(prefix + 'rollmed_data/R' + here + '.data_std.npy',SIZEstd )
        
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.xlim( xlim )
    plt.title('R rollmed plot of resolution')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedr='+str(dr)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedr='+str(dr)+'nside='+str(nside_median)+'.eps')


if plot_size_rollmedian_s:

    ds = graph_ds*(1000)**2 #nm²
    prefix = save_figs_folder  + 'results_GraphRollds='+str(ds)
    plt.figure(' S - field roll size graph',figsize=(5,4))
    plt.ylabel(' size - nm')
    plt.xlabel('axial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        
        MR = np.hypot( MYlist[pos] - np.median(MYlist[pos]) , MXlist[pos] - np.median(MXlist[pos]) )
        mSIZE = INTERDlist[pos][:,column]
        
        MS = np.pi * MR**2
            
        mss = MS.argsort()
        mssort = MS[mss]
        mSIZEsort = mSIZE[mss]
        
        mstep, mstepcount = np.unique( mssort//ds, return_counts = True )
        mstep_cumcount = np.concatenate(([0], np.cumsum( mstepcount) ))
        
        SIZEmed = []
        SIZEstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        for cpos, Mvalue in enumerate(mstepcount):
            pos0 = max(0, cpos - nside_median)
            posf = min(len(mstepcount)-1, cpos + nside_median)
            pos_init = mstep_cumcount[pos0]
            pos_final = mstep_cumcount[posf]

            localsigma = mSIZEsort[ pos_init : pos_final ]
            SIZEmed.append( np.nanmedian(localsigma) )
            SIZEstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        SIZEmed = np.array( SIZEmed )
        SIZEstd = np.array( SIZEstd )
        if relative_error:
            SIZEmed = (expected_size - SIZEmed)/expected_size * 100 # in %
            SIZEstd = SIZEstd/expected_size * 100
            sizemax, sizemin = 10, -3
            plt.ylabel(' relative size error - %')
        
        RSarr = np.sqrt( mstep*ds /np.pi) #in nm ?
        RSarr = RSarr/1000 #µm unit, recentered
        RSarr = RSarr - 0 
        
        xlim = [ min(xlim[0],RSarr.min())  , max(xlim[1],RSarr.max()) ]
        plt.plot( RSarr, SIZEmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(RSarr, SIZEmed-SIZEstd, SIZEmed+SIZEstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/S' + here + '.data_xarr.npy',RSarr )
            np.save(prefix + 'rollmed_data/S' + here + '.data_med.npy',SIZEmed )
            np.save(prefix + 'rollmed_data/S' + here + '.data_std.npy',SIZEstd )
        
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.xlim( xlim )
    plt.title('S rollmed plot of resolution')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizeds='+str(ds)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizeds='+str(ds)+'nside='+str(nside_median)+'.eps')



if plot_size_rollmedian_n:

    dn = graph_dn
    prefix = save_figs_folder  + 'results_GraphRolldn='+str(dn)
    plt.figure(' N - field roll size graph',figsize=(5,4))
    plt.ylabel(' size - nm')
    plt.xlabel('radial position - µm')
    xlim = [0,0] #for final graph limit.
    for pos in range(0,len(MXlist)):
        
        MR = np.hypot( MYlist[pos] - np.median(MYlist[pos]) , MXlist[pos] - np.median(MXlist[pos]) )
        mSIZE = INTERDlist[pos][:,column]
        
            
        mrs = MR.argsort()
        mrsort = MR[mrs]
        mSIZEsort = mSIZE[mrs]
                
        SIZEmed = []
        SIZEstd = []
        rollmedian = [] #contain median in each dx step - used for std calculation.
        count = 0
        for npos in np.arange(0, len(MR), dn):
            pos_init = max(0, npos - nside_median*dn)
            pos_final = min(len(MR)-1, npos + nside_median*dn )

            localsigma = mSIZEsort[ pos_init : pos_final ]
            SIZEmed.append( np.nanmedian(localsigma) )
            SIZEstd.append( np.std(localsigma) / np.sqrt( len(localsigma) ) )
        SIZEmed = np.array( SIZEmed )
        SIZEstd = np.array( SIZEstd )
        if relative_error:
            SIZEmed = (expected_size - SIZEmed)/expected_size * 100 # in %
            SIZEstd = SIZEstd/expected_size * 100
            sizemax, sizemin = 6, -1
            plt.ylabel(' relative size error - %')
        
        RNarr = mrsort[ np.arange(0, len(MR), dn) ]
        RNarr = RNarr/1000 #µm unit, recentered
        RNarr = RNarr - 0 
        
        xlim = [ min(xlim[0],RNarr.min())  , max(xlim[1],RNarr.max()) ]
        plt.plot( RNarr, SIZEmed, color = colorsList[pos], label=plot_titles[pos])
        plt.fill_between(RNarr, SIZEmed-SIZEstd, SIZEmed+SIZEstd, alpha=0.2, edgecolor=colorsList[pos], facecolor=colorsList[pos])
        if save_rollmedian_data:
            print('\t Saving rollmedian data for ' + plot_titles[pos] )
            here = '_'
            for s in plot_titles[pos]:
                if s!='*':
                    here = here + s
            if not os.path.isdir( prefix + 'rollmed_data/'):
                os.mkdir( prefix + 'rollmed_data/' )
            np.save(prefix + 'rollmed_data/N' + here + '.data_xarr.npy',RNarr )
            np.save(prefix + 'rollmed_data/N' + here + '.data_med.npy',SIZEmed )
            np.save(prefix + 'rollmed_data/N' + here + '.data_std.npy',SIZEstd )
        
    plt.legend()
    plt.ylim(sizemin,sizemax)
    plt.xlim( xlim )
    plt.title('N rollmed plot of resolution')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedn='+str(dn)+'nside='+str(nside_median)+'.png')
    plt.savefig( save_figs_folder  + 'results_GraphRollSizedn='+str(dn)+'nside='+str(nside_median)+'.eps')




if plot_density:
    print('Plotting Density Maps')
    dbin = density_bin
    gdvmin = globaldensity_min
    gdvmax = globaldensity_max
    dvmin = nanordensity_min
    dvmax = nanordensity_max
    
    dfig = plt.figure('Density - Nanoruler', figsize=(4*len(MXlist),4))
    axs = dfig.subplots(1,len(MXlist), sharex=False , sharey=False )
    plt.title('Nanoruler Density Map')
    if len(MXlist)==1:
        axs = [ axs ]
    axs[0].set_ylabel('field - µm')
    for pos in range(0,len(MXlist)):
        ndens = loc2imgC(MXlist[pos]/1000,MYlist[pos]/1000, binning=dbin)
        dim = axs[pos].imshow(ndens, cmap='nipy_spectral',vmin=dvmin,vmax=dvmax, label= plot_titles[pos])
    
    cb_ax = dfig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
    cbar = dfig.colorbar(dim, cax=cb_ax)
    plt.savefig( save_figs_folder  + 'DensityNanoRMap_' +'dbin='+str(dbin) + 'results.png')
    plt.savefig( save_figs_folder  + 'DensityNanoRMap_' +'dbin='+str(dbin) + 'results.eps')

    if globaldata_loaded:
        print('\t plotting global density map')
        gdfig = plt.figure('Density - Global', figsize=(4*len(MXlist),4))
        axs2 = gdfig.subplots(1,len(MXlist), sharex=False , sharey=False )
        if len(xlist)==1:
            axs2 = [ axs2 ]
        for pos in range(0,len(xlist)):
            print('\t \t img '+str(pos+1) + ' / ' + str(len(xlist)) )
            gdens = loc2imgC( xlist[pos]/1000, ylist[pos]/1000, binning=dbin)
            gdim = axs2[pos].imshow(gdens, cmap='nipy_spectral', vmin=gdvmin, vmax=gdvmax, label= plot_titles[pos])
            if map_ticks:
                dtick = 20 # wanted difference in um
                umx_max = ((xlist[pos].max()/1000)//dbin)*dbin
                umy_max = ((ylist[pos].max()/1000)//dbin)*dbin #this is max µm value.
                tickX = np.arange(0, umx_max, dtick )/dbin
                tickY = np.arange(0, umy_max, dtick )/dbin
                axs2[pos].set_xticks( tickX )
                axs2[pos].set_yticks( tickY )
                axs2[pos].set_xticklabels( [str(int(round(n,0))) for n in tickX*dbin] )
                axs2[pos].set_yticklabels( [str(int(round(n,0))) for n in tickY*dbin] )
                axs2[pos].set_xlabel('field - µm')
            else:
                axs2[pos].set_xticks( [] )
                axs2[pos].set_yticks( [] )

        cb_ax = gdfig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
        cbar = gdfig.colorbar(gdim, cax=cb_ax)
        cbar.set_label('relative density of fluorophore', rotation=270)
        print('\t saving global density map')
        
        plt.savefig( save_figs_folder  + 'DensityGlobalMap_' +'dbin='+str(dbin) + 'results.png')
        plt.savefig( save_figs_folder  + 'DensityGlobalMap_' +'dbin='+str(dbin) + 'results.eps')

            
    else:
        print('\t Not showing global density as no global data was loaded')


if plot_nph:
    print('Plotting Photon Maps')
    
    if nph0_loaded:
        nphfig = plt.figure('Photon - Nanoruler', figsize=(4*len(MXlist),4))
        axs = nphfig.subplots(1,len(MXlist), sharex=False , sharey=False )
        if len(MXlist)==1:
            axs = [ axs ]
        axs[0].set_ylabel('field - µm')
        for pos in range(0,len(MXlist)):
            local_mNPH0 = np.mean(NPH0list[pos], axis=1)
            img_nph = vhist2D(MXlist[pos], MYlist[pos], local_mNPH0, method='median', nb=(map_bin,map_bin), rng=None, nanzero = 0)
            nphim = axs[pos].imshow(img_nph, cmap='nipy_spectral',vmin=nphmin,vmax=nphmax, label= plot_titles[pos])
            if map_ticks:
                dtick=20
                tickvaluey = (MYlist[pos].max()//map_bin)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
                tickvaluex = (MXlist[pos].max()//map_bin)/1000
                tickX = np.arange(0, map_bin-1, dtick/tickvaluex )
                tickY = np.arange(0, map_bin-1, dtick/tickvaluey )
                axs[pos].set_xticks( tickX )
                axs[pos].set_yticks( tickY )
                axs[pos].set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
                axs[pos].set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
                axs[pos].set_xlabel('field - µm')
            else:
                axs[pos].set_xticks( [] )
                axs[pos].set_yticks( [] )
        
        cb_ax = nphfig.add_axes([0.92, 0.1, 0.02, 0.75]) # - xoffyoff cbarwidth heigt
        cbar = nphfig.colorbar(nphim, cax=cb_ax)
        cbar.set_label('Camera count', rotation=270)
        plt.savefig( save_figs_folder  + 'PhotonMap_' +'bin='+str(map_bin) + 'results.png')
        plt.savefig( save_figs_folder  + 'PhotonMap_' +'bin='+str(map_bin) + 'results.eps')
        
    #todo : for global data
    
if plot_nanor:
    if nspots == 3:
        xysize = 100  #nm
    elif nspots == 2:
        xysize = 70

#Show images of nanoruler. (already created in the analysis => IMGSlist)
if plot_nanor:  # Note: not common plot

    n_nanorulers0 = n_nanorulers
    print('plotting best ' + str(n_nanorulers0) + 'nanorulers')
    takebest=1
    
    
    if plot_nanor_gflou !=0: #gaussian blur . Creating gaussian.
        print('\t gaussian blur will be added, sigma:'+str(plot_nanor_gflou) )
        imgsshape = IMGSlist[0][0].shape
        X,Y = np.arange(0,imgsshape[1]), np.arange(0,imgsshape[0])
        Gflou = gaussian_2d(X,Y, 1, plot_nanor_gflou, int(imgsshape[0]/2), int(imgsshape[1]/2), 0)



    for pos in range(0,len(IMGSlist)):
        
        IMGS = IMGSlist[pos]
        mSIGMAS = mSIGMASlist[pos]
        n_nanorulers = min( n_nanorulers, len(IMGS) )
        n_nano_col= int( np.sqrt(n_nanorulers))
        n_nano_lines= n_nanorulers//n_nano_col + 1
        
        fig_nanor = plt.figure()
        fig_nanor.suptitle( ' Best NanoRulers of ' + plot_titles[pos] + '- total= ' + str(len(IMGS)))
        if takebest: #should be shifted in the next (if takebest) ?
            args = mSIGMAS.argsort()
        for i in range(0, n_nanorulers): 
            if takebest:
                img = IMGS[ args[i] ]
            else:
                img = IMGS[i]
                
            if plot_nanor_gflou !=0:
                img = scipy.signal.convolve2d( img, Gflou , mode='same')
                #note: may be possible to directly convolve all IMGS ?
                
                
            ax=fig_nanor.add_subplot(n_nano_lines,n_nano_col, i+1)        
            ax.imshow(img , cmap = 'hot')
        
            plt.xticks([])
            plt.yticks([])
        plt.savefig( save_figs_folder  + 'pos-'+str(pos) + '_' + str(n_nanorulers)+ 'NanoRulers_.png')
        plt.savefig( save_figs_folder  + 'pos-'+str(pos) + '_' + str(n_nanorulers)+ 'NanoRulers_.eps')
        
        
        
        
        
# if plot_nanor_makeimg:
#     print('Building image and plotting best ' + str(n_nanorulers0) + 'nanorulers')
#     n_nanorulers0 = n_nanorulers
#     print('plotting best ' + str(n_nanorulers0) + 'nanorulers')
#     takebest=1
    
#     for pos in range(0,len(IMGSlist)):
#         nanoruler_id = reslist[pos]
#         unique_id = np.unique( nanoruler_id )
        
#         mSIGMAS = mSIGMASlist[pos]
#         n_nanorulers = min( n_nanorulers, len(IMGS) )
#         n_nano_col= int( np.sqrt(n_nanorulers))
#         n_nano_lines= n_nanorulers//n_nano_col + 1
        
#         fig_nanor = plt.figure()
#         fig_nanor.suptitle( ' Best NanoRulers of ' + plot_titles[pos] + '- total= ' + str(len(unique_id)))
#         if takebest:
#             args = mSIGMAS.argsort()
#         for i in range(0, n_nanorulers):
#                 img = IMGS[ args[i] ]
#             else:
#                 img = IMGS[i]
                
#             ax=fig_nanor.add_subplot(n_nano_lines,n_nano_col, i+1)        
#             ax.imshow(img , cmap = 'hot')
        
#             plt.xticks([])
#             plt.yticks([])
#         plt.savefig( save_figs_folder  + 'pos-'+str(pos) + '_' + str(n_nanorulers)+ 'NanoRulers_.png')
#         plt.savefig( save_figs_folder  + 'pos-'+str(pos) + '_' + str(n_nanorulers)+ 'NanoRulers_.eps')
        
    
    
    
print('\nDone And Saved In ' + save_figs_folder )