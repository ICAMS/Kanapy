%% Import Script for EBSD Data
%
% Read EBSD map and analyse microstructure w.r.t. grain shapes.
%
% Author: Alexander Hartmaier
% ICAMS, Ruhr University Bochum, Germany
% February 2024

% define plotting convention
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');

% which files to be imported
fname = 'ebsd_316L_500x500.ang';
fname = '/Users/alexander/Documents/Projekte/HybridWelding-GhazalMoeini/EBSD/AlSi10Mg_SLM_cast_SLM_100X.ctf';

%% Import the Data
% create an EBSD variable containing the data
ebsd_full = EBSD.load(fname, ...
  'convertSpatial2EulerReferenceFrame', 'setting 2');
% remove non-indexed areas
ebsd = ebsd_full('indexed');
% select majority phase
ebsd_h = ebsd('1');
% get crystal symmetry from EBSD map
cs = getfield(ebsd_h, 'CS');
% calculate grain boundaries
grains_h = calcGrains(ebsd_h,'boundary','tight','angle',5*degree);
grains = grains_h(grains_h.grainSize > 5);
% plot EBSD map with grain boundaries
plot(ebsd_h,ebsd_h.orientations,'micronbar','on');
hold on;
plot(grains.boundary,'linewidth',2.0,'micronbar','off');
% plot ellipses fited to grains
[omega,a,b] = principalComponents(grains);
plotEllipse(grains.centroid,a,b,omega,'lineColor','r','linewidth',2.0);
hold off;

%% Plot ODF map
psi = deLaValleePoussinKernel('halfwidth', 5*pi/180.);
ori = getfield(ebsd_h, 'orientations');
h = [Miller(1, 0, 0, cs), Miller(1, 1, 0, cs), Miller(1, 1, 1, cs)];
odf = calcKernelODF(ori, 'kernel', psi);
figure;
hold on;
plotPDF(ori,h,'all');
%plotPDF(odf,h,'contourf');  % requires NFFT
mtexColorbar;
