import kanapy as knpy
from math import pi

if not knpy.MTEX_AVAIL:
    raise ModuleNotFoundError('Anaysis of EBSD maps is only possible with an'+\
                              ' existing MTEX installation in Matlab.')
    
fname = 'ebsd_316L_500x500.ang'  # name of ang file to be imported
#fname = 'ebsd_316L_1000x1000.ang'
matname = 'Iron fcc'  # material name for MTEX
matnumber = 4         # material number of austenite in CP UMAT

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname, matname)
#ebsd.showIPF()
print('==== Grain size ====')
print(f'mean equivalent diameter: {ebsd.gs_param[2].round(3)}, ' +
      f'std. deviation: {ebsd.gs_param[0].round(3)}, ' +
      f'offset: {ebsd.gs_param[1].round(3)}')
print('==== Aspect ratio ====')
print(f'mean value: {ebsd.ar_param[2].round(3)}, ' +
      f'std. deviation: {ebsd.ar_param[0].round(3)}, ' +
      f'offset: {ebsd.ar_param[1].round(3)}')
print('==== Tilt angle ====')
print(f'mean value: {(ebsd.om_param[1]*180/pi).round(3)}, ' +
      f'std. deviation: {(ebsd.om_param[0]*180/pi).round(3)}')

# create dictionary with statistical information obtained from EBSD map
# gs_param : [std deviation, offset of lognorm distrib., mean grain size]
# ar_param : [std deviation, offset of gamma distrib., mean aspect ration]
# om_param : [std deviation, mean tilt angle]
ms_stats = knpy.set_stats(ebsd.gs_param, ebsd.ar_param, ebsd.om_param,
                          deq_min=8., deq_max=16., asp_min=0.95, asp_max=3.5,
                          omega_min=0., omega_max=2*pi, voxels=30, size=50,
                          periodicity=True, VF = 1.0, phasename = "XXXX", phasenum = 0)

# create and visualize synthetic RVE
ms = knpy.Microstructure(descriptor=ms_stats, name=fname+'_RVE')
ms.init_stats(gs_data=ebsd.gs_data, ar_data=ebsd.ar_data)
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.plot_voxels(sliced=False)
ms.generate_grains()
ms.plot_grains()
ms.plot_stats(gs_param=ebsd.gs_param, ar_param=ebsd.ar_param)

# compare microstructure on three surfaces 
# for voxelated and polygonalized grains
ms.plot_slice(cut='xz', pos='top', data='poly')
ms.plot_slice(cut='yz', pos='top', data='poly')
ms.plot_slice(cut='xy', pos='top', data='poly')

# get list of orientations for grains in RVE matching the ODF of the EBSD map
ori_rve = ebsd.calcORI(ms.Ngr, ms.shared_area)

# export virtual EBSD map as ang file and analyse with MTEX
ang_rve = ms.output_ang(ori=ori_rve, cut='xy', pos='top', cs=ebsd.cs,
                        matname=matname, plot=False)
ebsd_rve = knpy.EBSDmap(ang_rve, matname)

# write Abaqus input file for voxelated structure
ms.output_abq('v')
# write Euler angles of grains into Abaqus input file
knpy.writeAbaqusMat(matnumber, ori_rve)

# smoothen voxelated structure and write Abaqus .inp file
ms.smoothen()
ms.output_abq('s')
