%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', [3.7 3.7 3.7], 'mineral', 'Iron fcc', 'color', [0.53 0.81 0.98])};

% plotting convention
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% which files to be imported
fname = 'ebsd_316L.ang';

%% Import the Data

% create an EBSD variable containing the data
ebsd_r = EBSD.load(fname,CS,'interface','ang',...
  'convertSpatial2EulerReferenceFrame');


ebsd = ebsd_filter(ebsd_r,'indexed');


grains_r = calcGrains(ebsd,'boundary','tight','angle',5*degree);
grains = grains_r(grains_r.grainSize > 5);

plot(ebsd,ebsd.orientations,'micronbar','off')
hold on

plot(grains.boundary,'linewidth',2.0,'micronbar','off')


[omega,a,b] = principalComponents(grains);
plotEllipse(grains.centroid,a,b,omega,'lineColor','r','linewidth',2.0)
