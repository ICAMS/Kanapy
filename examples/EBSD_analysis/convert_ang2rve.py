import kanapy as knpy

fname = 'ebsd_316L.ang'
matname = 'Iron fcc'
matnumber = 4  # 'material number of austenite in CP UMAT

# read EBSD map and evaluate statistics of microstructural features
ebsd = knpy.EBSDmap(fname, matname)

# create dictionary with statistical information obtained from EBSD map
# gs_param = [std deviation, mean grain size, offset of lognorm distrib.]
# ar_param = [std deviation, mean aspect ration, offset of gamma distrib.]
# om_param = [std deviation, mean tilt angle]
ms_stats = knpy.set_stats(ebsd.gs_param, ebsd.ar_param, ebsd.om_param)

# create RVE
ms = knpy.Microstructure(descriptor=ms_stats, name=fname+'_RVE')
ms.create_stats(gs_data=ebsd.gs_data, ar_data=ebsd.ar_data)
ms.create_RVE()
ms.pack()
ms.plot_ellipsoids()
ms.voxelize()
ms.output_stats()
ms.plot_3D(sliced=False)
ms.plot_stats(gs_param=ebsd.gs_param, ar_param=ebsd.ar_param)

# get list of orientations matching the ODF of the EBSD map
ori_rve = ebsd.calcORI(ms.particle_data['Number'], ms.shared_area)
knpy.writeAbaqusMat(matnumber, ori_rve)

#write Abaqus input file for voxelated structure
ms.output_abq('v')

#smoothen grain boundaries and write Abaqus input file for smoothened structure
ms.smoothen()
ms.output_abq('s')

