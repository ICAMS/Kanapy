%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the entire script or parts of it in order to import your data. There
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
fname = '/Users/alexander/Documents/Projekte/SFB-TR-103/EBSD-Data-AM_316L/ebsd_316L_500x500.ang';

%% Import the Data

% create an EBSD variable containing the data
ebsd_full = EBSD.load(fname,CS,'interface','ang',...
  'convertSpatial2EulerReferenceFrame', 'setting 2');

ebsd = ebsd_full('indexed');

grains_full = calcGrains(ebsd,'boundary','tight','angle',5*degree);
grains = grains_full(grains_full.grainSize > 5);

plot(ebsd,ebsd.orientations,'micronbar','on');
hold on;
plot(grains.boundary,'linewidth',2.0,'micronbar','off');

[omega,a,b] = principalComponents(grains);
plotEllipse(grains.centroid,a,b,omega,'lineColor','r','linewidth',2.0);
hold off;

%% Plot ODF map
psi = deLaValleePoussinKernel('halfwidth', 5*pi/180.);
ori = getfield(ebsd, 'orientations');
cs = getfield(ebsd, 'CS');
h = [Miller(1, 0, 0, cs), Miller(1, 1, 0, cs), Miller(1, 1, 1, cs)];
odf = calcKernelODF(ori, 'kernel', psi);
figure;
hold on;
plotPDF(odf,h,'contourf');
mtexColorbar;
