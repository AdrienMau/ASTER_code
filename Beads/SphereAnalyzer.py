# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:28:18 2019

English ///
Object Class for Sphere analysis
Use 3 images: Radial (focus on median sphere plan), Epi (focus near coverslip), TIRF (focus near coverslip)
Result data are : centers (X,Y) and radiuses (R) for each localized spheres.
Process: vertical profile fit by two gaussians.

Francais ///
Objet pour traiter les spheres.
Utilise image: rayon, epi et Tirf.
On stock les données: centre (X Y) et rayons (R) pour chaque sphere localisée.
+ Fonctions de fit par 2 gaussiennes. (utilisé sur le profil vertical)

SA = SphereAnalyzer.


Units are usually in pixel as comes from detection on image.


@author: Adrien MAU / ISMO & Abbelight

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy import optimize
from skimage.filters import threshold_local
import os
from PIL import Image
import warnings




class SphereAnalysis:
    
    #Path to image files :
    main = ""
    rfile= ""
    epifile= ""
    tirffile= ""
    crop = None #x y dx dy - crop applied at load.
    pixel = 108 #pixel size in nm.
    
    #Radius (Median), Epi and Tirf img:
    rimg = None
    epiimg = None
    tirfimg = None
    
    #Detection of sphere and center :
    detimg = "tirf" #img to use for X Y detection
    detcolor = 'red' #color for detection assessment
    detf_smin = 0  #min donut sigma for fourier/Laplacian detection  
    detf_smax = 100  #max --
    detf_thres = 1e4  #threshold on laplacian fourier-filetered image for labeling.
    recentered = 0 # True : a second estimation of XY has been done after detection.
    
    #Data of spheres: position xy and radius, and fit parameter (vertical and horizontal) :
    X = None
    Y = None
    R = None
    Repi = None
    Rtirf = None
    Zepi = None
    Ztirf = None
    
    #Extracted image and profiles - for now only py is implemented.  (py <=> vertical fit) 
    pwidth = 1 #A Profile is the horizontal mean around a column, meaning horizontally 0+2*sum_area elements
    plength = None #cool value: approx 3*R i guess
    r_px = None
    r_py = None
    r_py_params = []
    epi_px = None
    epi_py = None
    epy_py_params = []
    tirf_px = None
    tirf_py = None
    tirf_py_params = []
    rfit_method = None
    epifit_method = None
    tirffit_method = None
    fit_validated = None #bool array => 0 if sphere (from X,Y) was unfit for fit. 1 if was fitted.

    #for whatimg recognition:
    tirflike = ["tirf","Tirf","t",'t']
    epilike = ["EPI","epi","e","Epi"]
    rlike = ["R","r","radius","rad","rayon"]
    
    """
# =============================================================================
#    LOADING OF IMAGES
# =============================================================================
    """
    
    #Load image from folder main.
    #Automatically look for img with name epi, tirf and radius/rayon
    # OR if epifile, rfile or tirffile is != "": look directly for these files.
    def load(self, main=None, filename=None, whatimg=None):
        if main is None:
            main = self.main #by default local dir.
        
        #If any file path is already in object attribute we load it:
        loading_done = 0
        if not (self.tirffile is ""):
            print('Loading tirffile in ' + main + self.tirffile)
            self.loadimg( os.path.join(main, self.tirffile) , "tirf")
            loading_done = 1 
        if not (self.epifile is ""):
            print('Loading epifile in ' + main + self.epifile)
            self.loadimg( os.path.join(main, self.epifile) , "epi")
            loading_done=1
        if not (self.rfile is ""):
            print('Loading radiusfile in ' + main + self.rfile)
            self.loadimg( os.path.join(main, self.rfile) , "radius")
            loading_done=1
            
        if loading_done: #we loaded files as indicated by objet attribute. We end function.
            print('Loading Done ! \n')
            return
            
        if filename is None: #Automatically finds filename with keyword epi tirf or radius.
            for f in os.listdir(main):
                if f.endswith('.tif') or f.endswith('.tiff'):
                    whatimg = self.guessimage( f )
                    if not whatimg is "":
                        print('Loading ' + f + ' as ' + whatimg + ' image')
                        self.loadimg( os.path.join(main, f) , whatimg)
        else:
            if whatimg is None:
                whatimg = self.guessimage( filename )
            print('Loading ',filename, ' as ', whatimg, ' image')
            self.main = main
            self.loadimg( os.path.join(main, filename) , whatimg)
        print('Loading Done ! \n')

    
    def loadimg(self, path, whatimg):
        c = self.crop
        print('c is ',c)
        img = np.array( Image.open( path ) )
        if not (c is None):
            img = img[ c[0]:(c[0]+c[2]), c[1]:(c[1]+c[3])]
            
        if (whatimg is "tirf") or (whatimg is "t"):
            self.tirfimg = img
            self.tirffile = path
        elif (whatimg is "epi") or (whatimg is "e"):
            self.epiimg = img
            self.epifile = path
        elif (whatimg is "radius") or (whatimg is "r"):
            self.rimg = img
            self.rfile = path
            
    
    def showimgs(self):
        plt.figure('Loaded images')
        plt.subplot(131)
        plt.imshow( self.rimg )
        plt.title('radius image')
        plt.subplot(132)
        plt.imshow( self.epiimg )
        plt.title('EPI image')
        plt.subplot(133)
        plt.imshow( self.tirfimg )
        plt.title('TIRF image')
            
    #from filename/string guess if it is tirf, epi or radius/rayon
    def guessimage(self, filename):
        if filename.find("tirf")!=-1:
            whatimg="tirf"
        elif filename.find("epi")!=-1:
            whatimg="epi"
        elif (filename.find("radius")!=-1)or(filename.find("rayon")!=-1):
            whatimg="radius"
        else:
            whatimg = ""
        return whatimg
    
        
    #Return asked image of object (already loaded) between epi, tirf and radius/rayon
    def getimage(self,whatimg):
        if (whatimg is "tirf") or (whatimg is "t"):
            img =self.tirfimg
        elif (whatimg is "epi") or (whatimg is "e"):
            img = self.epiimg
        elif (whatimg is "radius") or (whatimg is "rayon")or (whatimg is "r"):
            img = self.rimg
        return img
    

    """
# =============================================================================
#    POSITION OF CENTER
# =============================================================================
    """
    
    """ Fonctions de detection : """
    
    #Depuis une des images, utilise un seuil local, binarise; ferme les trou, et détecte les zones indépendantes.
#pb ici: détecte souvent entre 2 spheres.
    def detect_classical(self, whatimg=None, kern_size=59, coef_thresh=None):
        print('Classical Segmentation-Detection of individual spheres')
        if whatimg is None:
            whatimg = self.detimg   
        img = self.getimage(whatimg)
        
        if coef_thresh is None:
            coef_thresh = np.median(img) + 0.6*np.std(img)
            print('\t using coef_thres of ' + str(coef_thresh))

        local_thresh = threshold_local(img, kern_size, offset=0)
        binary_local = img > local_thresh + coef_thresh   #detection par seuil...
        
        #Fermeture des trous: #mias peut être remerge les sphere proches aussi !
        binary_local = ndimage.morphology.binary_closing(binary_local, structure=np.ones((10,10)))
        
#        plt.figure();  plt.imshow(binary_local*img)   #assess detection efficacity
        
        labels,nb = ndimage.label(binary_local);
        out = ndimage.measurements.center_of_mass(binary_local, labels, np.arange(1,np.max(labels)+1));
        out = np.array(out)
        if len(out):
            self.X = out[:,0]
            self.Y = out[:,1]
            print('Segmentation Done ! - found ' + str(len(self.X)) + ' spheres \n')
        else:
            print('No Sphere found, check detection parameters')
            

        
    def detect_laplacian(self,whatimg=None, gauss_blur=1, doplot=0):
        if whatimg is None:
            whatimg = self.detimg
        img = self.getimage(whatimg)
        
        print('Laplacian + Fourier Segmentation-Detection of individual spheres')
        lap = scipy.ndimage.filters.laplace( img ) # Laplacian
        filteredimg = Fourier_filter(lap, self.detf_smin, self.detf_smax, gauss_blur, doplot)
        blap = (filteredimg > self.detf_thres )
        
        #blap = ndimage.morphology.binary_opening(blap, structure=np.ones((3,3))) #or 10 10
        labels,nb = ndimage.label( blap);
        out = ndimage.measurements.center_of_mass( filteredimg, labels, np.arange(1,np.max(labels)+1));
        out = np.array(out)
        if len(out):
            self.X = out[:,0]
            self.Y = out[:,1]
            print('Segmentation Done ! - found ' + str(len(self.X)) + ' spheres \n')
        else:
            print('No Sphere found, check detection parameters')
            
    def detect_laplacian_setparameters(self,whatimg=None, thres=None, smax=None, smin=None):
        if whatimg is None:
            whatimg = self.detimg
            
        print('Setting fixed parameters for Laplacian-Fourier detection')
        # Note that this works with crop of dx dy = 500 500
        if whatimg in self.rlike:
            t, smin, smax = 1.2e4, 40, 70
            self.detimg = "radius"
        elif whatimg in self.epilike:
            t, smin, smax = 1e4, 25, 40
            self.detimg = "epi"
        elif whatimg in self.tirflike:
            t, smin, smax = 7e3, 1, 24
            self.detimg = "tirf"
        else:
            warnings.warn('whatimg not recognized:' + whatimg + '\n')
            return
        if self.crop is None:
            length = self.getimage(whatimg)
        else:
            length = self.crop[2]
        ratio = length/500
        self.detf_thres = t
        self.detf_smin = smin*ratio
        self.detf_smax = smax*ratio
        
    
    
    
    def check_detect(self, whatimg=None):
        if whatimg is None:
            whatimg = self.detimg
        img = self.getimage(whatimg)
        plt.figure('Assessment of Detection on ' + whatimg + ' img')
        plt.imshow(img)
        plt.scatter( self.Y, self.X, s=2, marker='+', color=self.detcolor )
    
    # From XY center of detect, recaculte center via center of mass and image of +-pwidth.
    #note: this does not check if image is too close to border..
    def recalculate_centers(self,plength, whatimg='all'):
        if (self.X is None):
            print('No X Y position found ! Please perform detection first')
            return
        r = plength
        if whatimg is 'all':
            print('Recalculating center via all images')
            rimg = self.getimage("radius")
            eimg = self.getimage("epi")
            timg = self.getimage("tirf")
            for pos, x in enumerate(self.X):
                x = int(x)
                y = int(self.Y[pos])
                cr = ndimage.measurements.center_of_mass( rimg[x-r:x+r+1, y-r:y+r+1] ) #note: cr[1] seems to be +1 than normal. or even +2
                ce = ndimage.measurements.center_of_mass( eimg[x-r:x+r+1, y-r:y+r+1] )
                ct = ndimage.measurements.center_of_mass( timg[x-r:x+r+1, y-r:y+r+1] )
                self.X[pos] = np.nanmean([cr[1],ce[1],ct[1]])
                self.Y[pos] = np.nanmean([cr[0],ce[0],ct[0]])
        else:
            print('Recalculating center via ' + whatimg + ' images')
            img = self.getimage(whatimg)
            for pos, x in enumerate(self.X):
                x = int(x)
                y = int(self.Y[pos])
                c = ndimage.measurements.center_of_mass( img[x-r:x+r+1, y-r:y+r+1] )
                if not(np.isnan(c[0])): #may happen is image near border.
                    self.X[pos] = c[1] + self.X[pos] - r
                    self.Y[pos] = c[0] + self.Y[pos] - r
        self.recentered = True
        
    #plot number associated with each detected sphere.
    def show_detect_num(self, whatimg = None):
        if whatimg is None:
            whatimg = self.detimg
        img = self.getimage(whatimg)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter( self.Y, self.X, s=2, marker='+', color=self.detcolor )
        fig.canvas.set_window_title('Scatter- detection number association')
        num = np.arange(0, len(self.X))
        for i, txt in enumerate(num):
            ax.annotate(str(txt), (self.Y[i], self.X[i]))
                    
    """
# =============================================================================
#    PROFILES
# =============================================================================
    """
    
#Extract profile from found position in detect;
    #Also filter : - images whose center of mass is too far from center.
    #              - X Y Position too close to image border
    def get_profiles(self,plength=None, doplot=0):
        print('\nExtracting Profiles')
        if (self.X is None):
            print('No X Y position found ! Please perform detection first')
            return
        if plength is None: #length of a profile extraction.
            if self.plength is None:
                print('please give an approximate value of plength (typ:3*R)')
                return
            else:
                plength = self.plength
        pwidth = self.pwidth
        
        rimg = self.getimage("radius")
        eimg = self.getimage("epi")
        timg = self.getimage("tirf")
        
        rps = []
        eps = []
        tps = []
        valid = []
        r = plength
        for pos, x in enumerate(self.X):
#            print('pos number' , pos)
            x = int(x)
            y = int(self.Y[pos])
            if (x-r<0) or (x+r>rimg.shape[1])or(y-plength<0) or (y+plength>rimg.shape[0]):
                valid.append(0)  
                print('\t pos n°' + str(pos) + ': too close to border sphere')
                continue
            #Verification on center of mass :
            
            exc = self.filter_exc
            local_rimg = rimg[x-r:x+r+1, y-r:y+r+1]
            local_eimg = eimg[x-r:x+r+1, y-r:y+r+1]
            local_timg = timg[x-r:x+r+1, y-r:y+r+1]
            cr = ndimage.measurements.center_of_mass( local_rimg ) #note: cr[1] seems to be +1 than normal. or even +2
            ce = ndimage.measurements.center_of_mass( local_eimg )
            ct = ndimage.measurements.center_of_mass( local_timg )
            if np.abs(cr[0]-r)>exc or np.abs(cr[1]-r)>exc:
                print('\t pos n°' + str(pos) + ': too far from sphere initial center')
                #we could also compare in between r epi tirf ...
                valid.append(0)
                continue #or continue , plot , title unvalid, and... rps not added.
            
            local_rp = rimg[x-plength:x+plength+1, y-pwidth:y+pwidth+1]
            local_ep = eimg[x-plength:x+plength+1, y-pwidth:y+pwidth+1]
            local_tp = timg[x-plength:x+plength+1, y-pwidth:y+pwidth+1]
            rps.append( np.mean( local_rp,axis=1) )
            eps.append( np.mean( local_ep,axis=1) )
            tps.append( np.mean( local_tp,axis=1) )
            
            valid.append(1)
            
            if doplot:
                
                pos = pos - np.sum(np.array(valid)==0)
                plt.figure('Sphere n°' + str(pos))
                plt.subplot(231)
                plt.imshow( local_rimg )
                plt.scatter( cr[1], cr[0])
                plt.title('radial')
                plt.subplot(232)
                plt.imshow( local_eimg )
                plt.scatter( ce[1], ce[0])
                plt.title('epi')
                plt.subplot(233)
                plt.imshow( local_timg )
                plt.scatter( ct[1], ct[0])
                plt.title('tirf')
                plt.subplot(212)
                plt.plot(rps[pos]/np.max(rps[pos]),label='radius')
                plt.plot(eps[pos]/np.max(eps[pos]),label='epi')
                plt.plot(tps[pos]/np.max(tps[pos]),label='tirf')
                plt.legend()
        
        valid = np.array(valid,dtype='bool')
        print('\t Keeping ' , np.sum(valid) , ' spheres on ',len(valid) )    
        self.X = self.X[valid]
        self.Y = self.Y[valid]
        
        self.r_py = np.array( rps )
        self.epi_py = np.array( eps )
        self.tirf_py = np.array( tps )
        self.fit_validated = valid
            
        print('\t Profiles - Done! \n')
            
    #Return asked profile of object (already loaded) between epi, tirf and radius/rayon
    def getprofiles(self,whatimg):
        if whatimg in self.tirflike:
            return self.tirf_py
        elif whatimg in self.epilike:
            return self.epi_py
        elif whatimg in self.rlike:
            return self.r_py
        return 0
    
    def reset_profiles_results(self,whatimg='all'):
        if whatimg is 'all':
            self.r_py_params = []
            self.epi_py_params = []
            self.tirf_py_params = []
        else:
            if whatimg in self.rlike:
                self.r_py_params = []
            elif whatimg in self.epilike:
                self.epi_py_params = []
            elif whatimg in self.tirflike:
                self.tirf_py_params = []
    
    def append_param(self,whatimg,param):
        if whatimg in self.rlike:
            self.r_py_params.append( param )
        elif whatimg in self.epilike:
            self.epi_py_params.append( param )
        elif whatimg in self.tirflike:
            self.tirf_py_params.append( param )
    
    
    #fit py profile (already loaded) via different method:
    # - fit_2gaussian :fit by 2 independant gaussian.
    # - fit_2gaussian_same : fit by 2 gaussian , position indep.      (same sigma, amp..)      
    # - fit_peaks : find maximum peaks. TODO
    
    #Whatimg: all to fit for r_py epi_py and tirf_py, else user should precise (t e or r)
    def fit_profiles(self, whatimg='all', method='fit_2gaussian' ,doplot=0 ):
        if whatimg is 'all':
            W = ['radius','epi','tirf']
        else:
            W = [whatimg]
        print('\nFitting profiles')
        self.reset_profiles_results( whatimg ) #reset to empty list.
        for w in W:
            print('\t fitting ' + w + ' profiles with method:' + method)
            prfs = self.getprofiles( w )
            for prf in prfs:
                if method is 'fit_2gaussian_same': #param is xc,rx,sigma,amp,offset
                    param = fit_2g_profile(prf, showres=doplot, typ_title=w)
                elif method is 'fit_2gaussian':  #param is : x0 sigma0 amp0 , x1 sigma1 amp1, offset
                    param = fit_2g_profileA(prf, showres=doplot, typ_title=w)
                elif method is 'fit_peaks':  #param is center, x2-x1, 1, max, min
                    param = guess_peaks(prf, doplot=doplot, typ_title=w)
                elif method is 'fit_peaks2':  #param is center, x2-x1, 1, max, min
                    param = guess_peaks2(prf, doplot=doplot, typ_title=w)
                else:
                    print('fit_profiles: UNKWOWN METHOD: ' + method)
                self.append_param( w, param )
        
        if whatimg in self.rlike:
            self.rfit_method = method
        elif whatimg in self.epilike:
            self.epifit_method = method
        elif whatimg in self.tirflike:
            self.tirffit_method = method
        
        
    #from fitted parameter of each profile , give effective radius.
    def param2radius(self):
        if len(self.r_py_params):
            self.r_py_params = np.array( self.r_py_params )
            m = self.rfit_method
            if m is 'fit_2gaussian_same':
                self.R = self.r_py_params[:,1]
            elif m is 'fit_2gaussian':
                self.R = np.abs(self.r_py_params[:,3] - self.r_py_params[:,0])
            elif (m is 'fit_peaks') or (m is 'fit_peaks2'):
                self.R = self.r_py_params[:,1]
                
        if len(self.epi_py_params):
            self.epi_py_params = np.array( self.epi_py_params )
            m = self.epifit_method
            if m is 'fit_2gaussian_same':
                self.Repi = self.epi_py_params[:,1]
            elif m is 'fit_2gaussian':
                self.Repi = np.abs(self.epi_py_params[:,3] - self.epi_py_params[:,0])
            elif (m is 'fit_peaks') or (m is 'fit_peaks2'):
                self.Repi = self.epi_py_params[:,1]
                
        if len(self.tirf_py_params):
            self.tirf_py_params = np.array( self.tirf_py_params )
            m = self.tirffit_method
            if m is 'fit_2gaussian_same':
                self.Rtirf = self.tirf_py_params[:,1]
            elif m is 'fit_2gaussian':
                self.Rtirf = np.abs(self.tirf_py_params[:,3] - self.tirf_py_params[:,0])
            elif (m is 'fit_peaks') or (m is 'fit_peaks2'):
                self.Rtirf = self.tirf_py_params[:,1]

# Radius-Epi-Tirf Line plot of profiles number N
    def show_profiles(self, whatimg=None, N=None ):
        if N is None:
            N = np.arange(0,len(self.epi_py))
        if whatimg is None:
            whatimg = 'all'
        
        if whatimg in self.rlike:
            data = [self.r_py]
            dname = ["radius"]
        elif whatimg in self.epilike:
            data = [self.epi_py]
            dname = ["epi"]
        elif whatimg in self.tirflike:
            data = [ self.tirf_py ]
            dname = [ "tirf" ]
        elif whatimg == 'all':
            data = [self.r_py, self.epi_py, self.tirf_py]
            dname = ['radius','epi','tirf']
        colorcycle = ['blue','orange','green']
        alp = 0.8
        for pos in N:
            print()
            plt.figure('Show profiles plot' )
            arr = np.arange(0, len(data[0][pos])) * self.pixel/1000 #in um
            if len(data)==1:
                centroid = np.sum( arr* data[0][pos] )/ np.sum(data[0][pos]) #centroid calculated on only given imgtype
            else:
                centroid = np.sum( arr* data[0][pos] )/ np.sum(data[0][pos]) #centroid calculated on tirf imgtype.
            for datapos, d in enumerate(data): #for radius, epi , tirf (or only one of them) :
                dp = d[pos]
                dp = (dp - dp.min())/(dp.max() - dp.min()) #normed
                plt.plot(arr-centroid,dp,label=dname[datapos], color=colorcycle[datapos%3], alpha=alp)
#                plt.plot(dp,label=dname[pos])
                
            plt.legend()
            plt.title('Data profile')
            plt.xlabel('pixel')

    """
# =============================================================================
#    Handling Results & calculating Depth from R
# =============================================================================
    """
    
    #dr is for histogram; in pix unit.
    def show_rs(self, titlebonus = '', dr = 1, trueunit=False):
        if trueunit:
            coef = self.pixel/1000 #µm unit conversion.
        else:
            coef = 1
        plt.figure('Results' + titlebonus)
        plt.subplot(231)
        plt.title('Radius')
        plt.hist( self.R*coef , bins=coef*np.arange(self.R.min(), self.R.max(), dr) )
        plt.subplot(232)
        plt.title('EPI')
        plt.hist( self.Repi*coef , bins=coef*np.arange(self.Repi.min(), self.Repi.max(), dr) )
        print('epimax is', self.Repi.max() , 'pix' )
        plt.subplot(233)
        plt.title('TIRF')
        plt.hist( self.Rtirf*coef , bins=coef*np.arange(self.Rtirf.min(), self.Rtirf.max(), dr) )
        
        plt.subplot(212)
        plt.title('Field repartition of radius ')
        plt.scatter( self.X*coef, self.R*coef, color='cyan' , marker='X',label='R_x')
        plt.scatter( self.Y*coef, self.R*coef, color='cyan', marker='1',label='R_y')
        plt.scatter( self.X*coef, self.Repi*coef, color='green' , marker='X',label='Repi_x')
        plt.scatter( self.Y*coef, self.Repi*coef, color='green', marker='1',label='Repi_y')
        plt.scatter( self.X*coef, self.Rtirf*coef, color='red' , marker='X',label='Rtirf_x')
        plt.scatter( self.Y*coef, self.Rtirf*coef, color='red', marker='1',label='Rtirf_y')
        plt.legend()
        plt.xlabel('field')

    def filter_r(self, whatimg, rmin, rmax):
        if whatimg in self.rlike:
            r = self.R
        elif whatimg in self.epilike:
            r = self.Repi
        elif whatimg in self.tirflike:
            r = self.Rtirf
        valid = (r>=rmin)*(r<=rmax)
        
        print('Filtering '+whatimg+' radius between '+str(rmin)+' and '+str(rmax))
        self.filter_(valid)
        print('\tKept ' , np.sum(valid), ' spheres on' , len(valid) )
        
    
    #from a valid vector, filter out spheres on a False.
    #here i hesitate with creating a new objects, or keeping a basis of found vector
    # so that you could undo your filter..
    # balancing solution: we do not change epi/r/tirf_py so you can just refit.
    def filter_(self,valid):
        V = ['X','Y','R','Repi','Rtirf','r_py_params','epi_py_params','tirf_py_params']
#        V = ['X','Y','R','Repi','Rtirf','r_py','epi_py','tirf_py','r_py_params','epi_py_params','tirf_py_params']
        for var in V:
            exec( "self." + var + " = self." + var + "[valid]") # i heard its ugly to do this.
    
    
    
    #from 2D radius on sphere, and known RADIUS , give depth: 
    def rxy2z(self,rxy,R):
        return R - np.sqrt(np.abs(R**2-rxy**2)) #+ for other side       
        
    #from Repi and Rtirf calculate effective Z, using each sphere R or a global R (Rglobal)
    def radius2z(self, Rglobal=None):
        print('Creating Z Zepi Ztirf variables..')
        
        if not (Rglobal is None):
            R = Rglobal
        elif len(self.R):
            R = self.R
        else:
            print('radius2z : NO R FOUND ! Fit profile for calculation or provide a general R!')
            return
                    
        if len(self.Repi):
            self.Zepi = self.rxy2z( self.Repi, R)
                
        if len(self.Rtirf):
            self.Ztirf = self.rxy2z( self.Rtirf, R)
        print('\t Z calculated!')
    
     #multiply space variable results by a factor = factor.
     #will fail if any of these variable is None!
    def change_unit(self, factor = 108):
        V = ['X','Y','R','Repi','Rtirf','Zepi','Ztirf']
        for var in V:
            exec( "self." + var + " = self." + var + "*factor") # i heard its ugly to do this.
    
        
    def show_zs(self, titlebonus = '', dr=50):
        plt.figure('Z Results' + titlebonus, figsize=(12,8))
        plt.subplot(231)
        plt.title('R Radius')
        plt.hist( self.R , bins=np.arange(self.R.min(), self.R.max(),dr) )
        plt.subplot(232)
        plt.title('Z EPI')
        plt.hist( self.Zepi , bins=np.arange(self.Zepi.min(), self.Zepi.max(),dr) )
        # print('\t epimax is', self.Zepi.max() )
        plt.subplot(233)
        plt.title('Z TIRF')
        plt.hist( self.Ztirf , bins=np.arange(self.Ztirf.min(), self.Ztirf.max(),dr) )
        plt.subplot(212)
        plt.title('Field repartition of penetration depth ')
        plt.scatter( self.X, self.Zepi, color='green' , marker='1',label='Zepi_x')
        plt.scatter( self.Y, self.Zepi, color='green', marker='X',label='Zepi_y')
        plt.scatter( self.X, self.Ztirf, color='red' , marker='1',label='Ztirf_x')
        plt.scatter( self.Y, self.Ztirf, color='red', marker='X',label='Ztirf_y')
        plt.legend()

    def scat_zs(self, titlebonus = ''):
        ratio = 1000 #to go in µm
        plt.figure('Z Results Scattering' + titlebonus, figsize=(6,3.5))
        plt.title('Field repartition of penetration depth ')
        plt.scatter( self.X/ratio, self.Zepi, color='green' , marker='1',label='Zepi_x')
        plt.scatter( self.Y/ratio, self.Zepi, color='green', marker='*',label='Zepi_y')
        plt.scatter( self.X/ratio, self.Ztirf, color='red' , marker='1',label='Ztirf_x')
        plt.scatter( self.Y/ratio, self.Ztirf, color='red', marker='*',label='Ztirf_y')
        plt.legend()
        plt.xlabel('field - µm')
        plt.ylabel('penetration depth - nm')
        plt.tight_layout()

""" Fonctions de fit par 2 gaussiennes identiques : """


#Fonction two gaussian: Fit par deux gaussiennes qui different seulement par leur position.
#p is xc rx sigma amplitude offset, gaussians are in xc +- rx
def two_gaussian(p,X):
    e1 = np.exp( -0.5*((X-p[0]+p[1])/p[2])**2)
    e2 = np.exp( -0.5*((X-p[0]-p[1])/p[2])**2)
    return p[4] + p[3]*(e1+e2)

def two_gaussianfit(p,X,Y):
    return two_gaussian(p,X) - Y

def two_gaussianrmse(p,X,Y):
    return np.sqrt( np.sum( (two_gaussianfit(p,X,Y))**2 ) )/ X.shape[0]

#guess initial param
def guessp_twog(X,Y):    
    offset = Y.min() #offset (or take median)
    argm = Y.argmax()
    maxi = Y[argm]
    amp = (maxi - offset)/2 #amplitude
    xc = np.sum(X*(Y-offset))/np.sum(Y-offset)   
    rx = abs(xc - X[argm])
    if X[argm]>xc:
        fwhm = int(xc) + np.argmin( np.abs( Y[int(xc):] - maxi/2))
    else:
        fwhm = np.argmin( np.abs( Y[:int(xc)] - maxi/2))
    sigma = abs(fwhm - argm)/2.35
#    print( 'p0 is',[ xc,rx,sigma,amp,offset])
    return np.array( [ xc,rx,sigma,amp,offset])
    
def fit_2g_profile(profile, showres=0,typ_title=''):
    X = np.arange(0,profile.shape[-1])
    
    if profile.ndim==1:
        p0 = guessp_twog(X,profile)
        res = optimize.least_squares( two_gaussianfit , p0 , args=(X , profile), max_nfev=500)
        pfit = res["x"]
    else:
        print('fitting multiple profiles')
        pfit = np.zeros((profile.shape[0],5))
        for n,pf in enumerate(profile):
            p0 = guessp_twog(X,pf)
            res = optimize.least_squares( two_gaussianfit , p0 , args=(X , pf) , max_nfev=500)
            pfit[n] = res["x"]
    
    if showres==1:
        plt.figure()
        plt.title( typ_title )
        plt.plot(X,profile,label='data')
        plt.plot(X, two_gaussian(p0,X),label='guess')
        plt.plot(X, two_gaussian(pfit,X),label='fit')
        plt.legend()
    elif showres==2:
        print('RMSE :')
        print(two_gaussianrmse(p0,X,profile))
        print(two_gaussianrmse(pfit,X,profile))
 
    return pfit


""" Autre : """



#return center, x2-x1, 1, max, min
#Note: guess peak on exp data may chose the center as the two peaks have different height:
 #the center sometime has higher value than the right peak.
def guess_peaks(Y, doplot=0, typ_title=''):
    offset = Y.min() #offset (or take median)
    X = np.arange(0,len(Y))
    xc = int( np.sum(X*(Y-offset))/np.sum(Y-offset) )    #this should separate the two gaussian in half.
    yleft = Y[:int(xc)]
    yright = Y[int(xc):]
    aml = yleft.argmax()
    amr = yright.argmax()
    xl = aml
    xr = amr + int(xc)
    if doplot:
        plt.figure()
        plt.title( typ_title )
        plt.plot( X, Y )
        plt.scatter( xl,Y[aml])
        plt.scatter( xr,Y[amr+int(xc)])
        plt.scatter( xc, Y[xc], marker='+')
        
    return np.array( [ xc, abs(xl-xr),1, max(Y[aml],Y[xr]), offset] )

#guess peaks based on derivative.
    # basically: take centroid of data. take max derivative left and right
    # and return point in the middle : devmax-center center-devmax2
    #note: we could try to go from edge to center and... check condition on derivative ?
def guess_peaks2(Y, doplot=0, typ_title='' ):
    offset = Y.min() #offset (or take median)
    X = np.arange(0,len(Y))
    xc = int( np.sum(X*(Y-offset))/np.sum(Y-offset) )    #this should separate the two gaussian in half.
    yleft = Y[:int(xc)]
    yright = Y[int(xc):]
    
    dlm = np.argmax( np.diff(yleft)[:-2]) #pente la plus montante
    drm = np.argmax( -np.diff(yright)[2:] ) + 2  #pente la plus descendante
    drm =drm + xc
    xl = (dlm+xc)/2
    xr = (drm+xc)/2
    if doplot:
        plt.figure()
        plt.title( typ_title )
        plt.plot( X, Y )
        plt.scatter(dlm,Y[dlm], marker='+')
        plt.scatter(drm,Y[drm], marker='+')
        plt.scatter( xc, Y[xc], marker='+')
        plt.scatter( int(xl) , Y[int(xl)])
        plt.scatter( int(xr) , Y[int(xr)])
        
    return np.array( [ xc, abs(xl-xr),1, max(Y[int(xr)],Y[int(xl)]), offset] )


""" Fonctions de fit par 2 gaussiennes  : """

#Fonction two gaussian: Fit par deux gaussiennes qui different par pos, sigma, amp
#p est x0 sigma0 amp0 , x1 sigma1 amp1, offset
def two_gaussianA(p,X):
    e1 = np.exp( -0.5*((X-p[0])/p[1])**2)
    e2 = np.exp( -0.5*((X-p[3])/p[4])**2)
    return p[-1] + p[2]*e1 + p[5]*e2

def two_gaussianfitA(p,X,Y):
    return two_gaussianA(p,X) - Y

def two_gaussianrmseA(p,X,Y):
    return np.sqrt( np.sum( (two_gaussianfitA(p,X,Y))**2 ) )/ X.shape[0]

#guess initial param
def guessp_twogA(X,Y):    
    offset = Y.min() #offset (or take median)
    xc = np.sum(X*(Y-offset))/np.sum(Y-offset)    #this should separate the two gaussian in half.
    yleft = Y[:int(xc)]
    yright = Y[int(xc):]
    
    aml = yleft.argmax()
    amr = yright.argmax()
    ampl = yleft[aml] - offset #note: we ignore contribution of other gaussian in the local max.
    ampr = yright[amr] - offset
    xl = aml
    xr = amr + int(xc)
    
    fwhml = abs(aml - np.argmin( np.abs( yleft - yleft[aml]/2)))
    fwhmr = abs( amr - np.argmin( np.abs( yright - yright[amr]/2)))
    sigmal = fwhml/2.35
    sigmar = fwhmr/2.35
    return np.array( [ xl,sigmal,ampl,xr,sigmar,ampr,offset] )
    
def fit_2g_profileA(profile, showres=0, typ_title=''):
    X = np.arange(0,profile.shape[-1])
    
    if profile.ndim==1:
        p0 = guessp_twogA(X,profile)
#        print(p0)
        res = optimize.least_squares( two_gaussianfitA , p0 , args=(X , profile), max_nfev=500)
        pfit = res["x"]
    else:
        print('fitting multiple profiles')
        pfit = np.zeros((profile.shape[0],7))
        for n,pf in enumerate(profile):
            print(n)
            p0 = guessp_twogA(X,pf)
            res = optimize.least_squares( two_gaussianfitA , p0 , args=(X , pf) , max_nfev=500)
            pfit[n] = res["x"]
    
    if showres==1:
        plt.figure()
        plt.plot(X,profile,label='data')
        plt.plot(X, two_gaussianA(p0,X),label='guess')
        plt.plot(X, two_gaussianA(pfit,X),label='fit')
        plt.legend()
        plt.title(typ_title)
    elif showres==2:
        print('RMSE :')
        print(two_gaussianrmseA(p0,X,profile))
        print(two_gaussianrmseA(pfit,X,profile))
 
    return pfit


""" Fin - Fonctions de fit par 2 gaussiennes identiques """

    
""" HyperGaussiennes """
#p is sx sy x y height offset Power

#Creation of HyperGaussian2D Image user provides image and/or shape
    #p : widthx,widthy,centerx,centery,height,offset, power (likely even)
def HGaussianImage(p,I):
    if not (len(I)==2):
        sizex=I.shape[1]
        sizey=I.shape[0]
    else:
        sizex = I[1]
        sizey = I[0]

    x= (np.arange(0,sizex) - p[2])/p[0]
    y= (np.arange(0,sizey) - p[3])/p[1]
    X,Y = np.meshgrid(x,y)
    R=np.hypot(X,Y)
    eR= np.exp(-0.5*(R**p[6]))

    return p[5] + p[4]*eR
    

""" Fourier filtrage par Hypergaussienne """

def Fourier_filter(img, sigmain, sigmaout, gauss_flou=0, doplot=0):
    if gauss_flou!=0:
        img = ndimage.gaussian_filter(img, gauss_flou)

    sh = img.shape
    xc = sh[1]/2 - 0.5
    yc = sh[0]/2 - 0.5
    
    fimg= np.fft.fft2(img)
    fimgshift=np.fft.fftshift(fimg)
    
    p=[ sigmaout , sigmaout , xc, yc , 1 , 0 , 10 ] #xc and yc seems inverted to me but this works.. / note
    HG = HGaussianImage(p,sh)
    
    p=[ sigmain , sigmain , xc, yc , 1 , 0 , 10 ]
    HG = HG - HGaussianImage(p,sh)
    
    fimghg= np.fft.fftshift( HG*fimgshift )
    result = np.real(np.fft.ifft2(fimghg) )
    
    if doplot:
        plt.figure()
        plt.subplot(221)
        plt.imshow( img )
        plt.title('initial image (+ possible blur)')
        plt.subplot(222)
        f =  np.real( fimgshift )
        plt.imshow( f ,vmax= np.median( f ) + 2*np.std(f) )
        plt.title('Fourier img')
        plt.colorbar()
        plt.subplot(223)
        plt.imshow( HG )
        plt.title('Fourier Donut Filter')
        plt.subplot(224)
        plt.imshow( result , vmax= np.median( result ) + 4*np.std(result))
        plt.colorbar()
        plt.title('final image')
        
    return result

