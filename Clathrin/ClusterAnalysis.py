# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:31:45 2019

A container for cluster results. (version Clathrine for ex. Results are saved as dict in npy files)

@author: Adrien MAU / ISMO & Abbelight

"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy  #for plot
from scipy.interpolate import griddata
from scipy.optimize import least_squares

RES_NAMES = ['nloc','sx','sy','feret','mx','my','img','mnph','angle','radius','stdradius','c_id'] #keys   + make: mr gr sr se radiusratio
GRES_NAMES = ['frame','x','y','cluster_id','nph','sigma']

RES_RANGES = [ (0,1e5), (0,80), (0,80), (50,200),(0,2000), (0,2000), (0,0), (0,3e4), (-90,0) , (0,110) , (0,40), (0,None) ]
GRES_RANGES = [ (0,5e4), (0,2000), (0,2000), (0,3000),(0,3e4), (0,250) ]

ALLOW_PICKLE = True #since new versions need to be true for np load ?

class ClusterResults():
    
    expname = ''
    
    #Result of cluster analysis are in dictionnaries :
    RES_raw = None   #cluster results - keys: 'nloc','sx','sy','feret','mx','my','img','mnph'
    gRES_raw = None  #global results- keys : 'frame','x','y','cluster_id','nph','sigma'
    RES_names = dict( zip( RES_NAMES, ['#Localisation','size x - nm','size y - nm','feret diameter - nm','mean x position','mean y position','image','photon count','angle','radius - nm','Std radius - nm','cluster id'] ))
    gRES_names = dict( zip( GRES_NAMES, ['frame','x position - nm','y position - nm','cluster id','photon count','PSF size - nm']))
    RES_ranges = dict( zip( RES_NAMES, RES_RANGES ))
    gRES_ranges = dict( zip( GRES_NAMES, GRES_RANGES ))
    keys = None
    gkeys = None 
    
    imgshape = None #shape of nano imgs.
#    MX, MY, MLOC, SIGMAX, SIGMAY, FERET, IMGS, NPH =  [None]*8
#    x, y, frame, sigma, nph = [None]*5
    
    def __init__(self, path=None):
        print('Creating ClusterAnalysis object')
        if not(path is None):
            self.load_dict( path )
    
    def load_dict(self, path ,resname = 'RES.npy', gresname = 'gRES.npy'):
        if path.endswith('/'):
            if os.path.isfile( path + resname):
                self.RES_raw = np.load( path + resname, allow_pickle=ALLOW_PICKLE ).item()
            else:
                print('\t'+ resname + ' file not found' )
            if os.path.isfile( path + gresname):
                self.gRES_raw = np.load( path + gresname, allow_pickle=ALLOW_PICKLE ).item()
            else:
                print('\t'+ gresname + ' file not found' )
        self.init_afterload()
    
    def init_afterload(self):
        self.keys = self.RES_raw.keys()
        self.gkeys = self.gRES_raw.keys()
        self.make_radius() #add a mr (radius) key and values.s
        self.make_gradius() # add a r (radius) key and values.
        self.make_msr()
        self.make_se()  # anisotropy
        self.make_rratio()   #ratio holowness: radius / std_radius
        self.RES_ranges[-1] = (0,len(self.RES_raw['c_id']))
        self.imgshape = self.RES_raw['img'][0].shape
    
    #Add keys : radius (with mx,my or with x,y)  sr (with sx,sy)
    def make_radius(self, keyx='mx',keyy='my',cmx=None,cmy=None, radiuskey='mr'):  #cluster radius (position)
        x,y = self.get_reskeys([keyx,keyy], getglobal=False)
        if cmx is None:
            cmx = np.mean(x)
        if cmy is None:
            cmy = np.mean(y)
        mr = np.hypot(x-cmx,y-cmy)
        self.RES_raw[radiuskey] = mr
        self.RES_names[radiuskey] = 'radius'
        self.RES_ranges[radiuskey] = (0,mr.max())
        
    def make_gradius(self, keyx='x', keyy='y', cx=None, cy=None, gradiuskey='r'): #global radius.
        x,y = self.get_reskeys([keyx,keyy], getglobal=True)
        if cx is None:
            cx = np.mean(x)
        if cy is None:
            cy = np.mean(y)
        r = np.hypot(x-cx,y-cy)
        self.gRES_raw[gradiuskey] = r
        self.gRES_names[gradiuskey] = 'radius'
        self.gRES_ranges[gradiuskey] = (0,r.max())
        
    def make_msr(self, keyx='sx',keyy='sy', srkey='sr'):  #sr² = sx² + sy²
        sx,sy = self.get_reskeys([keyx,keyy], getglobal=False)
        sr = np.hypot(sx,sy)
        self.RES_raw[srkey] = sr
        self.RES_names[srkey] = 'std sigma_r'
        self.RES_ranges[srkey] = (0,sr.mean()+3*sr.std())
    
    #se : eccentricty = sy/sx  or  sx/sy ...
    def make_se(self, keyx='sx',keyy='sy',srkey='se'):
        sx,sy = self.get_reskeys([keyx,keyy], getglobal=False)
#        se = sy/sx
        se = np.max([sx,sy],axis=0) / np.min([sx,sy],axis=0)  #force ratio >1
        
        if se.min()>=1:
            serng = (1,se.mean()+3*se.std())
        else:
            serng = (0,se.mean()+3*se.std())

#        se = sy/sx*(sy>sx) + sx/sy*(sy<=sx)  #force ratio >1
#        serng = (1,se.mean()+3*se.std())  #force ratio >1
        self.RES_raw[srkey] = se
        self.RES_names[srkey] = 'std anisotropy'
        self.RES_ranges[srkey] = serng
        
    def make_rratio(self, keyx='stdradius',keyy='radius', srkey='radiusratio'):  #radiusratio such as r/sigmar.the higher it is, the more probable it is that we have a hollowness.
        stdmr,mr = self.get_reskeys([keyx,keyy], getglobal=False)
#        rr = mr / stdmr * np.sqrt(mr)
        rr = mr / stdmr   #good ratio : pretty independant of nloc. Some kind of clustering with feret.
        if rr.min()>=1:
            rr_rng = (1,rr.mean()+5*rr.std())
        else:
            rr_rng = (0,rr.mean()+3*rr.std())
        self.RES_raw[srkey] = rr
        self.RES_names[srkey] = 'r ratio'
        self.RES_ranges[srkey] = rr_rng
        
    #FILTERING ---    
    #filter resdict data by bool array
    def filter_dict(self, boolarr , getglobal=False):
        if getglobal:
            resdict = self.gRES_raw
        else:
            resdict = self.RES_raw
        
        for k in resdict.keys():
            resdict[k] = resdict[k][boolarr]
#        IMGS = [IMGS[i] for i in range(0,len(IMGS)) if cvalid[i] ]
        if getglobal:
            self.gRES_raw = resdict
        else:
            self.RES_raw = resdict
    
    
    #Crop data (dictionnaries) either with cluster or global data (or both):
    
    def crop_all(self,CROP,ckeys=['mx','my'], gkeys=['x','y'] ):
        self.crop_cluster(CROP, ckeys)
        self.crop_global(CROP, gkeys)
        
    #CROP is [x0,y0,dx,dy]
    def crop_cluster(self, CROP, keys=['mx','my']):
        MX, MY = self.get_reskeys( keys, getglobal=False)
        cvalid = (MX>CROP[0])*(MX<CROP[0]+CROP[2])*(MY>CROP[1])*(MY<CROP[1]+CROP[3])
        if np.sum(cvalid)!=len(cvalid): #modified.
            print('Cropping cluster''s results with',CROP)
            self.filter_dict(cvalid,getglobal=False)
        else:
            print('Crop cluster has no effect here.')

    def crop_global(self, CROP, keys=['x','y']):
        x, y = self.get_reskeys( keys, getglobal=True)
        cvalid = (x>CROP[0])*(x<CROP[0]+CROP[2])*(y>CROP[1])*(y<CROP[1]+CROP[3])
        if np.sum(cvalid)!=len(cvalid): #modified.
            print('Cropping global results with',CROP)
            self.filter_dict(cvalid,getglobal=True)
        else:
            print('Crop global has no effect here.')
    #Filter the object : keep data where key is in range rng.
    def filter_cluster(self, rng, key, printnum=False):
        data = self.get_reskeys( key, getglobal=False)
        cvalid = (data>=rng[0])*(data<=rng[1])
        if printnum:
            print('\tFiltering',rng,' on',key,' ',np.sum(cvalid),'/',len(cvalid))
        self.filter_dict(cvalid, getglobal=False)
        
    def filter_global(self, rng, key):
        data = self.get_reskeys( key, getglobal=True)
        cvalid = (data>=rng[0])*(data<=rng[1])
        self.filter_dict(cvalid, getglobal=True)


    # =============================================================================
    #         Get and set: result, range, and names dictionnaries.
    # =============================================================================
                
    #return tuple of result-dict according to provided keys.
    def get_reskeys(self, keys, getglobal=False):
        if getglobal:
            resdict = self.gRES_raw
        else:
            resdict = self.RES_raw
        if type(keys) is type('a'):   #only one key asked.
            return resdict.get(keys,None)
        else:
            return tuple( [resdict.get(k,None) for k in keys] )   #or [resdict[k] for k in keys]

    #return tuple of result-ranges according to provided keys.
    def get_rangekeys(self, keys, getglobal=False):
        if getglobal:
            rngdict = self.gRES_ranges
        else:
            rngdict = self.RES_ranges
        if type(keys) is type('a'):   #only one key asked.
            return rngdict[keys]
        else:
            return tuple( [rngdict[k] for k in keys] )
    #todo: function to set ranges.

    #return tuple of result-ranges according to provided keys.
    def get_namekeys(self, keys, getglobal=False):
        if getglobal:
            rngdict = self.gRES_names
        else:
            rngdict = self.RES_names
        if type(keys) is type('a'):   #only one key asked.
            return rngdict[keys]
        else:
            return tuple( [rngdict[k] for k in keys] )


    #return tuple of result-dict according to provided keys.
    # warning: if one key only, results should not be provided as a list (but np array.)
    # if multiple key: input list for each key.
    def set_reskeys(self, keys, results, getglobal=False):
        if not type(keys) is list:
            keys = [keys]
        if not type(results) is list:
            results = [results]
        if getglobal:
            for kpos, k in enumerate(keys):
                self.gRES_raw[k] = results[kpos]
        else:
            for kpos, k in enumerate(keys):
                self.RES_raw[k] = results[kpos]
        return 1

    #return tuple of result-ranges according to provided keys.
    # keys and rangetuple can be list, for each keys. Or one element (string, tuple.)
    def set_rangekeys(self, keys, rangetuple, getglobal=False):
        if not type(keys) is list:
            keys = [keys]
        if not type(rangetuple) is list:
            rangetuple = [rangetuple]
        if getglobal:
            for kpos, k in enumerate(keys):
                self.gRES_ranges[k] = rangetuple[kpos]
        else:
            for kpos, k in enumerate(keys):
                self.RES_ranges[k] = rangetuple[kpos]
        return 1
    #todo: function to set ranges.

    #return tuple of result-ranges according to provided keys.
    def set_namekeys(self, keys, names, getglobal=False):
        if not type(keys) is list:
            keys = [keys]
        if not type(names) is list:
            names = [names]
        if getglobal:
            for kpos, k in enumerate(keys):
                self.gRES_names[k] = names[kpos]
        else:
            for kpos, k in enumerate(keys):
                self.RES_names[k] = names[kpos]
        return 1
    
    
    # =============================================================================
    #         Get some data for further fit or stuffs
    # =============================================================================

    def get_hist(self, key, getglobal=False, bins='auto',hrange=None,density=None, meanbin=1):
        data = self.get_reskeys(key, getglobal=getglobal)
        Count, Bin = np.histogram(data, bins=bins, range=hrange, density=density)
        if meanbin:
            Bin = 0.5*(Bin[:-1] + Bin[1:])
        return Count, Bin
    
    #create 2D map of density along axis keyx and keyy. Map is restrained to keyx and keyy ranges.
    def get_map(self, keyx, keyy, getglobal=False, binx=None,biny=None,density=None, plot_img = False):
        
        X = self.get_reskeys(keyx, getglobal=getglobal)
        Y = self.get_reskeys(keyy, getglobal=getglobal)
        xrng = self.get_rangekeys(keyx, getglobal=getglobal)
        yrng = self.get_rangekeys(keyy, getglobal=getglobal)
#        img,xb,yb = np.histogram2d(X,Y,bins=(binx,biny),range=(xrng,yrng))
        img,yb,xb = np.histogram2d(X,Y,bins=(binx,biny),range=(xrng,yrng))
        if density:
            img = img/np.sum(img)
            
        if plot_img:
            plt.figure('get_map utilisation - ' + str(plot_img) )
            plt.imshow(img, cmap='nipy_spectral')
            if plot_img>2:
                xrng = (xb.min(), xb.max())
                yrng = (yb.min(), yb.max())
                actual_tx = plt.gca().get_xticks()
                actual_ty = plt.gca().get_xticks()
                fx = (actual_tx[-1]-0) / (xrng[1] - xrng[0]) #yes
                fy = (actual_ty[-1]-0) / (yrng[1] - yrng[0])
                
                plt.gca().set_xticklabels(  [str((round(n,1))) for n in actual_tx/fx+xrng[0]]   )
                plt.gca().set_yticklabels(  [str((round(n,1))) for n in actual_ty/fy+yrng[0]]   )
                if plot_img==4:
                    dtx = actual_tx[1]-actual_tx[0]
                    dty = actual_ty[1]-actual_ty[0]
                    fdx = 2*5**( round(np.log(dtx/fx)/np.log(5),0)) #new round d
                    fdy = 2*5**( round(np.log(dty/fy)/np.log(5),0))
                    print('fdx fdy',fdx,fdy)
                    wanted_x = np.arange( xrng[0],xrng[1], fdx )
                    wanted_y = np.arange( yrng[0],yrng[1], fdy )
                    wanted_x = wanted_x[ (wanted_x >= xrng[0])*(wanted_x <= xrng[1]) ] #or will extend image.
                    wanted_y = wanted_y[ (wanted_y >= yrng[0])*(wanted_y <= yrng[1])  ] #or will extend image.
                    
                    plt.gca().set_xticks(  (wanted_x-xrng[0])*fx )
                    plt.gca().set_yticks(  (wanted_y-yrng[0])*fy )
#                    plt.gca().set_xticks(  (wanted_x-xrngw[0])*fx )
#                    plt.gca().set_yticks(  (wanted_y-yrngw[0])*fy )
                    
                    plt.gca().set_xticklabels(  [str((round(n,1))) for n in wanted_x]   )
                    plt.gca().set_yticklabels(  [str((round(n,1))) for n in wanted_y]   )
        
            elif plot_img==2:
                plt.xticks([])
                plt.yticks([])
        return xb,yb,img
    
    

# =============================================================================
#         Plot stuff using a list of ClusterResults, and keys.
# =============================================================================
    
#from a list of ClusterResults, plot histogram.
    #clist: color list for each cluster.
def res_hist( cres_list , hist_bin=15, hist_type='step', savepath=None, keylist=['sx','sy','feret','mnph'] , clist = None, alpha = 0.8, getglobal=False):
    if not ( type( cres_list) is  list ):
        cres_list = [ cres_list ]
    
    print('\t Plotting Histogram Result images')
    ncol = int( round(np.sqrt(len(keylist))+0.5) )
    nline = int(len(keylist)/ncol + 0.5)
    figf, axf = plt.subplots( nline, 1, figsize=(6.5*nline,5*ncol) )
    figf.canvas.set_window_title(' Some final Datas ')

    for pos in range(0,len(cres_list)):
        if not clist is None:
            color = clist[pos]
        else:
            color = None
        cr = cres_list[pos]
        for kpos, k in enumerate(keylist):
            data = cr.get_reskeys( k , getglobal=getglobal)
            datar = cr.get_rangekeys( k , getglobal=getglobal) #range
            datan = cr.get_namekeys( k , getglobal=getglobal)  #name
            plt.subplot(100*nline + 10*ncol + kpos+1 )
            plt.hist(data,bins=hist_bin, range=datar, color=color, alpha=alpha , histtype=hist_type)
            plt.title('Histogram of ' + k )
            plt.xlabel(datan)
            plt.ylabel('Occurence')
            
    if not savepath is None:
        if savepath.endswith('/'):
            savepath = savepath + 'results'
        plt.savefig( savepath + '.png')
        plt.savefig( savepath + '.eps')


#from a list of ClusterResults, plot map of some data
    #clist: color list for each cluster.
def res_map( cres_list , map_bin=10, savepath=None, keyx ='mx', keyy='my', keyz='feret', map_ticks=False, cmap='nipy_spectral', doscatter=False, getglobal=False):
    if not ( type( cres_list) is  type([]) ):
        cres_list = [ cres_list ]
    
    print('\t Plotting Map/Scatter Result images')
    lcr = len(cres_list)
    figf, axf = plt.subplots(1,lcr,figsize=(5*lcr,5))
    figf.suptitle('Map of '+keyz)
    
    for pos in range(0,len(cres_list)):
        cr = cres_list[pos]
        x,y,z = cr.get_reskeys( [keyx, keyy, keyz] , getglobal=getglobal )
        zrange = cr.get_rangekeys( keyz , getglobal=getglobal )  #range
        zname = cr.get_namekeys( keyz , getglobal=getglobal ) #name
        plt.subplot(1,lcr,pos+1)
        if doscatter:
            plt.scatter(y,x, c=z, s=1, alpha=0.7, cmap = cmap, vmin=zrange[0], vmax=zrange[-1])
        else:
            img_sigma = vhist2D(x, y, z, method='median', nb=(map_bin,map_bin), rng=None, nanzero = 0)
            plt.imshow(img_sigma, cmap = cmap, vmin=zrange[0], vmax=zrange[-1])
#        Iinterp = interpolate_img( MX, MY, mSIGMAS, binning=100, method='linear') #or method nearest.
            
        if map_ticks==2 and not(doscatter):
            tickvaluey = (y.max()//map_bin)/1000 # one img pixel is x or ylocal.max()/map_bin -nm
            tickvaluex = (x.max()//map_bin)/1000
            tickX = np.arange(0, map_bin-1, 10/tickvaluex )
            tickY = np.arange(0, map_bin-1, 10/tickvaluey )
            plt.gca().set_xticks( tickX )
            plt.gca().set_yticks( tickY )
            plt.gca().set_xticklabels( [str(int(round(n,0))) for n in tickX*tickvaluex] )
            plt.gca().set_yticklabels( [str(int(round(n,0))) for n in tickY*tickvaluey] )
            plt.gca().set_xlabel('field - µm')
        elif not(map_ticks):
            plt.xticks([])
            plt.yticks([])
        if pos==0:
            plt.ylabel( cr.get_namekeys(keyy) )
    plt.xlabel( cr.get_namekeys(keyx) )
    cbar = plt.colorbar()
    cbar.set_label(zname, rotation=270) #note; name for last CR of list (but should all be the same)
    
    if not savepath is None:
        if savepath.endswith('/'):
            savepath = savepath + 'Map_'+keyz
        plt.savefig( savepath + '.png')
        plt.savefig( savepath + '.eps')



#from a list of ClusterResults, plot data along one axis. 'cut' can be provided to select more specifically with respect to another value.
        #cut: will only select data keyz where the data of keycut is comprised in mcut +- dcut
        #   by default mcut is mean of datacut (data of keycut)
        #   if dcut is 0 or keycut is None: no cut happens.
    #if nside median is provided (and doscatter=False): do a rolling median of side (total stack: 2*nside +1) : nside_median
    #clist: color list for each cluster.
def res_axis( cres_list, nside_median=0, daxis=1000, savepath=None, keyaxis ='mx', keyz='feret', dcut=0, mcut=None, keycut='my', clist=None, doscatter=False, plot_std=False, getglobal=False):
    if not ( type( cres_list) is  type([]) ):
        cres_list = [ cres_list ]
    
    print('\t Plotting Result Graph along axis ' + keyaxis)
    plt.figure( figsize=(10,6))
    
    for pos in range(0,len(cres_list)):
        if not clist is None:
            color = clist[pos]
        else:
            color = None

        cr = cres_list[pos]
        x,z = cr.get_reskeys( [keyaxis, keyz], getglobal=getglobal)
        zrange = cr.get_rangekeys( keyz , getglobal=getglobal)  #range
        zname = cr.get_namekeys( keyz , getglobal=getglobal)  #name
        if not (keycut is None):
            if dcut!=0:
                datacut = cr.get_reskeys( keycut , getglobal=getglobal)
                if mcut is None:
                    mcut = np.mean( datacut )
                print('\t res_axis: keeping data '+keycut+' between ' + str(mcut-dcut) + '-' + str(mcut+dcut) )
                cutvalid = np.abs( datacut - mcut ) < dcut
                x = x[cutvalid]
                z = z[cutvalid]
        
        if doscatter:
            plt.scatter(x, z, color = color)
        else:
            if nside_median==0:
                if plot_std:
                    xcoord, zvalue, zstd = data_slice( x, z, daxis, 'median', return_std=True)
                else:
                    xcoord, zvalue = data_slice( x, z, daxis, 'median')
            else:
                if plot_std:
                    xcoord, zvalue, zstd = data_slice_rollmed( x, z, daxis, nside_median, method='median', return_std=True )
                else:
                    xcoord, zvalue = data_slice_rollmed( x, z, daxis, nside_median, method='median' )
                    
            plt.plot( xcoord, zvalue, color=color )
            if plot_std:
                plt.fill_between(xcoord,zvalue-zstd, zvalue+zstd, alpha=0.2, edgecolor=color, facecolor=color)

            
        plt.ylim( zrange )
    plt.ylabel('Mean of ' + zname)
    plt.xlabel( cr.get_namekeys( keyaxis , getglobal=getglobal) ) #note: name for last chosen cr
    
    if nside_median==0:
        plt.title('Slice evolution of ' + keyz + ' along ' + keyaxis )
    else:
        plt.title('Slice Roll median evolution of' + keyz + ' along ' + keyaxis + ' nside='+str(nside_median) )
        
    if not savepath is None:
        if savepath.endswith('/'):
            method = ['','scat'][doscatter]
            if nside_median==0:
                savepath = savepath + 'Axis_'+keyz+'daxis='+str(daxis)+'_'+str(keyaxis)+'dcut='+str(dcut)+'_'+str(keycut)+'-'+method
            else:
                savepath = savepath + 'Axis_'+keyz+'daxis='+str(daxis)+'_'+str(keyaxis)+'dcut='+str(dcut)+'_'+str(keycut)+'-'+method+'Roll-nside_'+str(nside_median)
        plt.savefig( savepath + '.png')
        plt.savefig( savepath + '.eps')


#todo: add ticks.
def res_densitymap( cres_list, savepath=None, keyx ='mx', keyy='my', binx=None, biny=None, cmap='nipy_spectral', vmax=None , getglobal=False, trueticks=False):
    if not ( type( cres_list) is  type([]) ):
        cres_list = [ cres_list ]
    if binx is None:
        x = cres_list[0].get_reskeys(keyx , getglobal=getglobal)
        binx = (x.max()-x.min())/200
        print('binx is',binx)
    if biny is None:
        y = cres_list[0].get_reskeys(keyy , getglobal=getglobal)
        biny = (y.max()-y.min())/200
        print('biny is',biny)
    
    print('\t Plotting Density Map')
    lcr = len(cres_list)
    figf, axf = plt.subplots(1,lcr,figsize=(5*lcr,5))
    figf.suptitle(' Density Maps')
    
    for pos in range(0,len(cres_list)):
        cr = cres_list[pos]
        x,y = cr.get_reskeys( [keyx, keyy] , getglobal=getglobal)
#        print('x and y are',x,y)
        if x.min()<0:
            print('resdensitymap: shifting x to upper value.')
            x = x - x.min()
        if y.min()<0:
            print('resdensitymap: shifting x to upper value.')
            y = y - y.min()
#            
        density_img = loc2imgC(x, y, binning=binx, binningy=biny)
        plt.subplot(1,lcr,pos+1)
        if vmax is None:
            d0 = density_img[density_img!=0]
#            vmax = np.mean(density_img[density_img!=0]) + 5*np.std(density_img[density_img!=0])
            vmax = 0.6*(np.max(d0)  + np.mean(d0)) + np.std(d0)
            
            print('vmax is',vmax)
        plt.imshow( density_img, cmap=cmap, vmax=vmax, vmin=1)
        if trueticks:
            xrng, yrng = cr.get_rangekeys( [keyx, keyy] , getglobal=getglobal)
            xfactor = x.max()/density_img.shape[1]
            new_xticks = np.arange( xrng[0] , xrng[1], 10) / xfactor
            plt.gca().set_xticklabels( [str(round(n*xfactor,1)) for n in new_xticks] )
            
            yfactor = y.max()/density_img.shape[0]
            new_yticks = np.arange( yrng[0] , yrng[1], 10) / yfactor
            plt.gca().set_yticklabels( [str(round(n*yfactor,1)) for n in new_yticks] )
            
        if pos==0:
            plt.ylabel( cr.get_namekeys(keyy, getglobal= getglobal) )
        plt.xlabel( cr.get_namekeys(keyx, getglobal= getglobal) )
    plt.colorbar()
        
        
    if not savepath is None:  
        if savepath.endswith('/'):
            savepath = savepath + 'Map_Density'+keyx+'_'+keyy
        plt.savefig( savepath + '.png')
        plt.savefig( savepath + '.eps')
    

    
    
#Return Xcoord and Values: mean or median of value on 'dx' long ranges.
def data_slice(x, value, dx, method = 'median', return_std=False):
    xs = x.argsort()
    x = x[xs]
    value = value[xs]
    Xcoord, Count_in_dx = np.unique( x//dx, return_counts=True)
    Xcoord = Xcoord*dx
    count = 0
    Values = []
    Std = []
    for xc in Count_in_dx:
        local_values = value[count:count+xc] 
        if method is 'median':
            Values.append( np.nanmedian( local_values) )
        elif method is 'mean':
            Values.append( np.nanmean( local_values ) )
        if return_std:
            Std.append( np.std(local_values) / np.sqrt( len(local_values) ) )
        count = count + xc
        
    if return_std:
        return np.array(Xcoord), np.array(Values), np.array(Std)
    else:
        return np.array(Xcoord), np.array(Values)

#Return Xcoord and Values: mean or median of value on 'dx' long ranges. Rolling.
def data_slice_rollmed(x, value, dx, nside_median=2, method = 'median', return_std=False):
    xs = x.argsort()
    x = x[xs]
    value = value[xs]
    Xcoord, Count_in_dx = np.unique( x//dx, return_counts=True)
    Xcoord = Xcoord*dx
    mxstep_cumcount = np.concatenate(([0], np.cumsum( Count_in_dx ) ))
    Valuemed = []
    Valuestd = []
    for cpos, mxvalue in enumerate(Count_in_dx):
        pos0 = max(0, cpos - nside_median)
        posf = min(len(Count_in_dx)-1, cpos + nside_median)
        pos_init = mxstep_cumcount[pos0]
        pos_final = mxstep_cumcount[posf]
        
        localval = value[ pos_init : pos_final ]
        if method is 'median':
            Valuemed.append( np.nanmedian(localval) )
        if method is 'mean':
            Valuemed.append( np.nanmean(localval) )
        if return_std:
            Valuestd.append( np.std(localval) / np.sqrt( len(localval) ) )
        
    if return_std:
        return np.array(Xcoord), np.array(Valuemed), np.array(Valuestd)
    else:
        return np.array(Xcoord), np.array(Valuemed)
        
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
    print('spx and spy are',spx,spy)
    print('sx sy is',sx, sy)
    print('bin is',binning,binningy)
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


#create a gaussian : amp*exp(-x²/2s²)
def gaussian1d( X, amp=1, sigma=1, xc=0 ,offset=0):
    return amp*np.exp( - 0.5*(X-xc)**2/sigma**2)+offset

#FIT BY 3 GAUSSIANS. by default amp >0.
def three_gaussian1d(X, amp0=1, s0=1, xc0=0 , amp1=1, s1=1, xc1=0 , amp2=1, s2=1, xc2=0 ,offset=0):
    g0 = gaussian1d( X, max(0,amp0), s0, xc0 ,0)
    g1 = gaussian1d( X, max(0,amp1), s1, xc1 ,0)
    g2 = gaussian1d( X, max(0,amp2), s2, xc2 ,0)
    return g0+g1+g2+offset
    
def three_gaussian1d_fixedoffset(X, fixed_offset, amp0=1, s0=1, xc0=0 , amp1=1, s1=1, xc1=0 , amp2=1, s2=1, xc2=0):
    g0 = gaussian1d( X, max(0,amp0), s0, xc0 ,0)
    g1 = gaussian1d( X, max(0,amp1), s1, xc1 ,0)
    g2 = gaussian1d( X, max(0,amp2), s2, xc2 ,0)
    return g0+g1+g2+fixed_offset

def three_gaussian1d_diff(p, x, y):
    return three_gaussian1d(x, *p) - y

def three_gaussian1d_diff_fixedoffset(p, x, y, offset):
    return three_gaussian1d_fixedoffset(x,offset, *p) - y


def three_gaussian1d_fit(x,y,p0list, fixed_offset=None):
    p0list = np.ravel(p0list)
    if fixed_offset is None:
        res = least_squares(three_gaussian1d_diff, p0list, args=(x,y), method='lm')
    else:
        p0list = p0list[:6]
        res = least_squares(three_gaussian1d_diff_fixedoffset, p0list, args=(x,y,fixed_offset), method='lm')
    pfit = res['x']
    return pfit  #for 3 gauss, offset seems to fuck up ? (too high values...)

def three_gaussian1d_checkparam(x,y,param):
    plt.figure()
    plt.plot(x,y,label='data')
    plt.plot(x,three_gaussian1d(x,*param),label='fit',alpha=0.7)
    
    offset = param[-1]
    g0 = gaussian1d( x, *param[0:3]) + offset
    g1 = gaussian1d( x, *param[3:6]) + offset
    g2 = gaussian1d( x, *param[6:9]) + offset
    plt.plot(x,g0,label='gauss0',linestyle='--')
    plt.plot(x,g1,label='gauss1',linestyle='--')
    plt.plot(x,g2,label='gauss1',linestyle='--')
    plt.legend()
    
def three_gaussian1D_returngaussians(x,param):
    offset = param[-1]
    g0 = gaussian1d( x, *param[0:3]) + offset
    g1 = gaussian1d( x, *param[3:6]) + offset
    g2 = gaussian1d( x, *param[6:9]) + offset
    return g0,g1,g2

#create a map from X,Y, color is encoded as probability to get mvalue.
    #note : we suppose global and local data are still linked.
def threeg_probability_map(param, X,Y,CID, mvalue, pixsize = 10):
    offset = param[-1]
    cids = CID.argsort()
    X,Y,CID = X[cids], Y[cids], CID[cids]
    X = (X//pixsize)
    Y = (Y//pixsize)
    idcount , uid = np.histogram(CID,bins=np.arange(0,CID.max()+ 2)) #ok
    cumcount = np.cumsum(np.concatenate(([0],idcount)))
    
    g0 = gaussian1d( mvalue, *param[0:3]) + offset
    g1 = gaussian1d( mvalue, *param[3:6]) + offset
    g2 = gaussian1d( mvalue, *param[6:9]) + offset

    G0 = np.zeros((len(X)))
    G1 = np.zeros((len(X)))
    G2 = np.zeros((len(X)))
    for idpos, idn in enumerate( uid[:-1] ):
        G0[cumcount[idn]:cumcount[idn+1]] = g0[idpos]
        G1[cumcount[idn]:cumcount[idn+1]] = g1[idpos]
        G2[cumcount[idn]:cumcount[idn+1]] = g2[idpos]
        
#    map_bin=50
    I0 = vhist2D(X, Y, G0, method='median', rng=(0,X.max()), nanzero = 0)
    I1 = vhist2D(X, Y, G0, method='median', rng=(0,X.max()), nanzero = 0)
    I2 = vhist2D(X, Y, G0, method='median', rng=(0,X.max()), nanzero = 0)
    return I0, I1, I2  


#return intersection points between two gaussians. in between: solutions in between centers.
def intersection_gaussian( amp0, s0, x0, amp1, s1, x1, in_between=False):
    a = (s1**2 - s0**2)
    b = 2*(s0**2*x1 - s1**2*x0)
    c = s1**2*x0**2 - s0**2*x1**2 - 2*s0**2*s1**2*np.log(amp0/amp1)
    if a==0:
        if b==0:
            return [] #gaussian are only different in amplitude: the higher one always stays higher: no intersection
        else:
            sols = np.array([ -c/b]) #ok...
    else:
        delta = b**2-4*a*c
        sols = np.array( [ (-b+np.sqrt(delta))/2/a , (-b-np.sqrt(delta))/2/a ] )
    if in_between:
        sols = sols[(sols>min(x0,x1))*(sols<max(x1,x0))]
    return sols


#return inters point. param should be in center order.
def threeg_intersections(param, in_between=False):
    p0 = param[0:3]
    p1 = param[3:6]
    p2 = param[6:9]
    sol01 = intersection_gaussian(*p0,*p1)
    sol12 = intersection_gaussian(*p1,*p2)
    if in_between:
        sol01 = sol01[ (sol01>=p0[2])*(sol01<=p1[2])]
        sol12 = sol12[ (sol12>=p1[2])*(sol12<=p2[2])]
    
    return sol01, sol12


#for n gaussian: plist is of size 3*ngaussian +1 (offset) 
    # each 3 are: amp, sigma, xc
def n_gaussian1d(x, plist):
    g = np.zeros(len(x))
    for ppos in range(0,int(len(plist)/3)):
        ppos = ppos*3
#            p = plist[ppos:ppos+3]
        g = g + gaussian1d(x, plist[ppos], plist[ppos+1], plist[ppos+2], 0)
    if len(plist)%3==1:
        g = g + plist[-1]  #adding offset.
    return g

def n_gaussian1d_diff(plist,x,y): #here plist is of length: n*3 +1
    return y - n_gaussian1d(x,plist)

def n_gaussian1d_diff_fixedoffset(plist,x,y,offset):  #here plist is of length: n*3
    plist.append( offset )
    return y - n_gaussian1d(x,plist)

#note: in any case p0list is of len  3*k+1. 
def n_gaussian1d_fit(x,y,p0list, fixed_offset=None):
    p0list = np.ravel(p0list)
    if fixed_offset is None:
        res = least_squares(n_gaussian1d_diff, p0list, args=(x,y), method='lm')
        pfit = res['x']
    else:
        p0list = p0list[:-1]
        res = least_squares(n_gaussian1d_diff_fixedoffset, p0list[:-1], args=(x,y,fixed_offset), method='lm')
        pfit = res['x']
        pfit.append( fixed_offset )

    return pfit  #for 3 gauss, offset seems to fuck up ? (too high values...)

def n_gaussian1d_checkparam(x,y,param):
    plt.figure()
    plt.plot(x,y,label='data')
    plt.plot(x,n_gaussian1d(x,param),label='fit',alpha=0.7)
    
    for ng in range(0,len(param)//3):
        g = gaussian1d( x, *param[3*ng:3*(ng+1)])
        plt.plot(x,g,label='gauss'+str(ng),linestyle='--')
    plt.legend()

#return inters point. param should be in center order.  in between: keep sols in between gaussian centers.
def n_intersections(param, in_between=False):
    centers = param[ np.arange(2, len(param) ,3) ] 
    sol_list = []
    for ng in range(0,len(param)//3- 1):
        sol_list.append( intersection_gaussian(*param[3*ng:3*ng+3], *param[3*ng+3:3*ng+6]) )  
    if in_between:
        new_sol_list = []
        for spos, s in enumerate(sol_list):
#            print('s is ',s)
#            print('centers are', centers[spos],centers[spos+1])
            new_sol_list.append( s[(s>centers[spos]) * (s<centers[spos+1]) ]  )
        sol_list = new_sol_list
    return sol_list






#Note: fit of gaussian2Dnot yet tested, gaussian2de tested.
#Create a gaussian image. amp*exp(-x²/2s²)
def gaussian_2d(X,Y, amp, sigma, xc, yc, offset=0):
    X = (X-xc)/(sigma**2)
    Y = (Y-yc)/(sigma**2)
    eX= np.exp(-0.5*(X**2))
    eY= np.exp(-0.5*(Y**2))
    eY=eY.reshape(len(eY),1)
    
    return offset + amp*eY*eX

#Parameter for n_gaussian2D is consecutive p of len(4), and a possible additionnal offset.
#Create a gaussian image. amp*exp(-x²/2s²)
def n_gaussian_2d(X,Y,plist):
    nparam = 4 
    G = np.zeros((len(Y),len(X)))
    for ng in range(0,int(len(plist)/nparam)):
        start = ng*nparam
        G = G + gaussian_2d(X,Y, plist[start], plist[start+1], plist[start+2], plist[start+3], 0)
    if len(plist)%nparam==1:
        G = G + plist[-1]  #adding offset.
    return G


#Elliptic gaussian.
#Return a np array 1D.
def gaussian_2de_list(xl, yl, amp, sx,sy, theta, xc, yc, offset=0):
    xl = xl - xc
    yl = yl - yc
    theta = theta 
    c = np.cos(theta)
    s = np.sin(theta)
    A = 0.5*c**2/sx**2 + 0.5*s**2/sy**2
    B = 0.25*( np.sin(2*theta)/sy**2 - np.sin(2*theta)/sx**2)
    C = 0.5*s**2/sx**2 + 0.5*c**2/sy**2
    return offset + amp*np.exp(-(A*xl**2 + 2*B*xl*yl + C*yl**2))
    
    
#Create an elliptic gaussian image.
def gaussian_2de(X,Y, amp, sx,sy, theta, xc, yc, offset=0):
    c = np.cos(theta)
    s = np.sin(theta)
    A = 0.5*c**2/sx**2 + 0.5*s**2/sy**2
    B = 0.25*( np.sin(2*theta)/sy**2 - np.sin(2*theta)/sx**2)
    C = 0.5*s**2/sx**2 + 0.5*c**2/sy**2
    
    XX,YY = np.meshgrid(X-xc,Y-yc)
    
    return offset + amp*np.exp(-(A*XX**2 + 2*B*XX*YY + C*YY**2))

#Parameter for n_gaussian2D is consecutive p of len(5), and a possible additionnal offset.
#Create a gaussian image. amp*exp(-x²/2s²)
def n_gaussian_2de(X,Y,plist):
    nparam = 6
    G = np.zeros(( len(Y),len(X) ))
    for ng in range(0,int(len(plist)/nparam)):
        start = ng*nparam
        localp = plist[start:start+nparam]
        localp = np.append(localp,0)
        G = G + gaussian_2de(X,Y, *localp)
    if len(plist)%nparam==1:
        G = G + plist[-1]  #adding offset.
    return G

def n_gaussian_2d_diff(plist,X,Y,Z):
    return Z - n_gaussian_2d(X,Y,plist)  #maybe ravel?

def n_gaussian_2d_diff_fixedoffset(plist,X,Y,Z,offset):
    plist.append( offset )
    return Z - n_gaussian_2d(X,Y,plist)

def n_gaussian_2de_diff(plist,X,Y,Z):
    return np.ravel(Z - n_gaussian_2de(X,Y,plist))  #maybe ravel?

def n_gaussian_2de_diff_fixedoffset(plist,X,Y,Z,offset):
#    plist.append( offset )
    return np.ravel(Z - n_gaussian_2de(X,Y,np.append(plist,offset)))



#note: in any case p0list is of len  4*k+1.
def n_gaussian_2d_fit(x,y,z,p0list, fixed_offset=None):
#    p0list = np.ravel(p0list)
    if fixed_offset is None:
        res = least_squares(n_gaussian_2d_diff, p0list, args=(x,y,z), method='lm')
        pfit = res['x']
    else:
        p0list = p0list[:-1]
        res = least_squares(n_gaussian_2d_diff_fixedoffset, p0list[:-1], args=(x,y,z,fixed_offset), method='lm')
        pfit = res['x']
        pfit.append( fixed_offset )

    return pfit  #for 3 gauss, offset seems to fuck up ? (too high values...)

def n_gaussian_2d_checkparam(x,y,z,plist):
    nparam = 4
    Gfit = n_gaussian_2d(x,y,plist)
    plt.figure()
    plt.subplot(131)
    plt.imshow(z)
    plt.subplot(132)
    plt.imshow(Gfit)
    plt.subplot(133)
    plt.imshow(z-Gfit)
    
    plt.figure()
    ngaussian = len(plist)//nparam
    for ng in range(0,ngaussian):
        plt.subplot(100 + ngaussian*10 + ng)
        plt.imshow( n_gaussian_2d(x,y,plist) )
    

#note: in any case p0list is of len  4*k+1.
def n_gaussian_2de_fit(x,y,z,p0list, fixed_offset=None):
#    p0list = np.ravel(p0list)
    if fixed_offset is None:
        res = least_squares(n_gaussian_2de_diff, p0list, args=(x,y,z), method='lm')
        pfit = res['x']
    else:
        p0list = p0list[:-1]
        res = least_squares(n_gaussian_2de_diff_fixedoffset, p0list, args=(x,y,z,fixed_offset), method='lm')
        pfit = res['x']
        print('lp',len(pfit))
        pfit = np.append(pfit, fixed_offset)

    return pfit  #for 3 gauss, offset seems to fuck up ? (too high values...)

#From a z image, and a plist describng multiple gaussian. (nparam=6). Plot the differences. Compare images.
#note: for yticks we use int.
def n_gaussian_2de_checkparam(x,y,z,plist, individual=0, wanted_tickx = None, wanted_ticky = None):
    nparam = 6
    xf = (x.max() - x.min())/z.shape[1]
    yf = (y.max() - y.min())/z.shape[0]
    Gfit = n_gaussian_2de(x,y,plist)
    
    
        
    fig = plt.figure( figsize=(12,3))
    plt.subplot(131)
    plt.imshow(z, cmap='nipy_spectral',vmax=z.max(),vmin=-2)
    if not(wanted_tickx is None):
        plt.gca().set_xticks( (wanted_tickx-x.min())/xf )
    if not(wanted_ticky is None):
        plt.gca().set_yticks( (wanted_ticky-y.min())/yf )
    plt.gca().set_xticklabels( [str(round(n*xf+x.min(),1)) for n in plt.gca().get_xticks()] )
    plt.gca().set_yticklabels( [str(int(round(n*yf+y.min(),1))) for n in plt.gca().get_yticks()] )
    plt.subplot(132)
    im = plt.imshow(Gfit, cmap='nipy_spectral',vmax=z.max(),vmin=-2)
    if not(wanted_tickx is None):
        plt.gca().set_xticks( (wanted_tickx-x.min())/xf )
    if not(wanted_ticky is None):
        plt.gca().set_yticks( (wanted_ticky-y.min())/yf )
    plt.gca().set_xticklabels( [str(round(n*xf+x.min(),1)) for n in plt.gca().get_xticks()] )
    plt.gca().set_yticklabels( [str(int(round(n*yf+y.min(),1))) for n in plt.gca().get_yticks()] )
    plt.title('sum =' + str(np.round(np.sum(Gfit),1)) )
#    plt.colorbar(im,fraction=0.046, pad=0.04)
    
    plt.subplot(133)
    imf = plt.imshow(z-Gfit, cmap='nipy_spectral',vmax=z.max(),vmin=-2)
    if not(wanted_tickx is None):
        plt.gca().set_xticks( (wanted_tickx-x.min())/xf )
    if not(wanted_ticky is None):
        plt.gca().set_yticks( (wanted_ticky-y.min())/yf )
    plt.gca().set_xticklabels( [str(round(n*xf+x.min(),1)) for n in plt.gca().get_xticks()] )
    plt.gca().set_yticklabels( [str(int(round(n*yf+y.min(),1))) for n in plt.gca().get_yticks()] )
    plt.title('RMSE error =' + str(np.round( np.sqrt(np.sum( (z-Gfit)**2 )),2)) )
    
    plt.colorbar(imf,fraction=0.046, pad=0.04)

    
    
    if individual: #also plot each gaussian separately
        plt.figure( figsize=(18,3))
        ngaussian = len(plist)//nparam
        for ng in range(0,ngaussian):
#            print( "subplot is",100 + ngaussian*10 + ng )
            plt.subplot(100 + ngaussian*10 + ng + 1)
            plt.title( np.round(plist[ng*nparam:ng*nparam+nparam],2) , fontsize=8) 
            plt.imshow( n_gaussian_2de(x,y,plist[ng*nparam:ng*nparam+nparam]) )
            if not(wanted_tickx is None):
                plt.gca().set_xticks( (wanted_tickx-x.min())/xf )
                plt.gca().set_xticklabels( [str(n*xf+x.min()) for n in plt.gca().get_xticks()] )
            else:
                plt.gca().set_xticks([])
            if not(wanted_ticky is None):
                plt.gca().set_yticks( (wanted_ticky-y.min())/yf )
                plt.gca().set_yticklabels( [str(int(n*yf+y.min())) for n in plt.gca().get_yticks()] )
            else:
                plt.gca().set_yticks([])
    
    
    #from GMM cluster res cov: get sigma of gaussians. (on diagonal of cov matrix.)
def get_sigmas(cov, ngaussian):
    if cov.ndim==2:
        return np.tile( np.sqrt(np.trace(cov/3)) , 3 )
    else:
        return [ np.sqrt(  np.trace(cov[i])/ngaussian) for i in range(0,ngaussian) ]
    
#from cov matrix (from GMM cluster by ex.) return sx,sy and theta. Return between (-pi,pi)
def cov_to_params(covlist):
    res = []
    for c in covlist:
        eVa, eVe = np.linalg.eig( c )
        R, S = eVe, np.diag(np.sqrt(eVa))
        theta = (np.arccos( R[0][0] ) ) * -np.sign( np.arcsin(R[0][1]))

        sx = S[0][0]
        sy = S[1][1]
        
        res.append([sx,sy,theta])
    return res
    