# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:28:18 2019

ENG//
Treatment of sphere with different effective radius:
    Detection of individual spheres
    Fit of their radius.
    => plot
    
Call the SphereAnalyzer Object
Use image: either radius, epi or tirf
We get x,y, and radiuses R for each individual sphere.
+ Fit by 2 gaussians


FR //
Appel Objet pour traiter les spheres.
Utilise image: rayon, epi et Tirf.
On stock les données: centre (X Y) et rayons (R) pour chaque sphere localisée.
+ Fonctions de fit par 2 gaussiennes.



SA = SphereAnalyzer.


@author: Adrien MAU / ISMO & Abbelight

"""

import numpy as np
import matplotlib.pyplot as plt
import SphereAnalyzer

SA = SphereAnalyzer.SphereAnalysis()

SA.main = "" #local or global directory to find images.
SA.tirffile = "tirf_img.tif"; #125 126 ou 128 - 126 bof - 128 ok - 125 
SA.epifile =  "epi_img.tif"
SA.rfile = "radius_img.tif"



SA.crop = [350,350,1500,1500]
#SA.crop = None

SA.load()
#SA.showimgs()

SA.detimg = "t"
SA.detect_laplacian_setparameters(whatimg=None)
SA.detect_laplacian()
#SA.check_detect()

SA.recalculate_centers(plength = 25, whatimg='r' )
#SA.check_detect()

#SA.show_detect_num()


SA.filter_exc = 6
SA.get_profiles( plength = 25 , doplot = 0)
SA.fit_profiles( 'tirf' , method='fit_peaks2', doplot=0)
SA.fit_profiles( 'epi' , method='fit_peaks', doplot=0)
SA.fit_profiles( 'radius' , method='fit_peaks', doplot=0)
SA.param2radius()
SA.show_detect_num()



SA.show_profiles('all',N=[30,32]) #show some profiles.

SA.filter_r('r',27,35)
SA.filter_r('e',15,25)
SA.filter_r('t',2.5,25)

SA.radius2z()


SA.change_unit(factor = 108) #change pixel to nm.

#Show results in radius : 
SA.show_rs(titlebonus='filteredR-Repi-Rtirf' , dr=100)

#Show results in z :
SA.scat_zs()
#Save img:
plt.savefig('TirfEpi_vert_depth2.png')
plt.savefig('TirfEpi_vert_depth2.eps')
