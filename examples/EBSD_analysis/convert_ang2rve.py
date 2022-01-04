import kanapy as knpy
from math import pi

if not knpy.MTEX_AVAIL:
    raise ModuleNotFoundError('Anaysis of EBSD maps is only possible with an existing MTEX installation in Matlab.')
    
fname = 'ebsd_316L_500x500.ang'
#fname = 'ebsd_316L_1000x1000.ang'
matname = 'Iron fcc'
matnumber = 4  # 'material number of austenite in CP UMAT

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname, matname)

# create dictionary with statistical information obtained from EBSD map
# gs_param : [std deviation, mean grain size, offset of lognorm distrib.]
# ar_param : [std deviation, mean aspect ration, offset of gamma distrib.]
# om_param : [std deviation, mean tilt angle]
ms_stats = knpy.set_stats(ebsd.gs_param, ebsd.ar_param, ebsd.om_param,
                          deq_min=8., deq_max=30., asp_min=1., asp_max=3.,
                          omega_min=0., omega_max=2*pi, voxels=60, size=100,
                          periodicity=True)

# create and visualize synthetic RVE
ms = knpy.Microstructure(descriptor=ms_stats, name=fname+'_RVE')
ms.init_stats(gs_data=ebsd.gs_data, ar_data=ebsd.ar_data)
ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.analyze_RVE()
ms.plot_voxels(sliced=False)
ms.plot_stats(gs_param=ebsd.gs_param, ar_param=ebsd.ar_param)

# get list of orientations for grains in RVE matching the ODF of the EBSD map
ori_rve = ebsd.calcORI(ms.Ngr, ms.shared_area)
knpy.writeAbaqusMat(matnumber, ori_rve)

#write Abaqus input file for voxelated structure
ms.output_abq('v')

#smoothen grain boundaries and write Abaqus input file for smoothened structure
ms.smoothen()
ms.output_abq('s')

