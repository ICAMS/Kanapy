function [orired_f,odfred_f,ero,varargout]=...
    ODF_reduction_algo(p_mtex,ns,varargin)
%% 
% This function sysytematically recontructs the ODF by a smaller number of
% orientations (refer .....)  
% also the misorientation distribution is   
%
%
% Syntax:
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'ebsdMatfile',ebsdfile)
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'ebsd',ebsd) 
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'ebsdMatfile',ebsdfile,  
%           'grainsMatfile',grainsfile)
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'ebsd',ebsd,'grains',...
%           'grains')
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'orientations',ori)  
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'orientations',ori,...  
%           'kernel',psi)
% [ori,odf,e]=ODF_reduction_algo('mtexfolderpath',n,'orientations',ori,...  
%           'kernelShape','kappa') 
%
% Inputs:
% 
%  1) p_mtex: MTEX folder path
%  2) n: number of reduced orientations/grains in RVE
%  3) Either path+filename of ebsd data saved as *.mat file (it should 
%     contain only one phase/mineral) or ebsd(single phase)/orientations
%  4) Either path+filename of the estiamted grains from above 
%     EBSD saved as *.mat file (it should contain only one phase/mineral)
%     or kernel(only deLaValeePoussinKernel)/kernelshape, if nothing       
%     mentioned then default value kappa = 5 (degree) is assumed.
%
% Output: reduced orientation set, ODF and L1 error 
% 
%% input fields and checks
if p_mtex(end)=='/'
    try
        run([p_mtex 'install_mtex.m'])
    catch 
        error('ODF_reduction_algo:e1',...
            'Could not find MTEX installation file')
    end    
else
    try
        run([p_mtex '/install_mtex.m'])
    catch
        error('ODF_reduction_algo:e1',...
            'Could not find MTEX installation file')
    end    
end     

assert(length(varargin)>=2,'ODF_reduction_algo:e2','Insufficient options')
 
options = {'ebsdMatFile','ebsd','orientation'};
flag = 1; 
grains = [];
for i = 1:length(options)

    loc = find(strcmp(varargin,options{i}));
    
    if length(loc)>1
        error('ODF_reduction_algo:e4','Multiple options for same input')       
    end
    if flag>=0
        if ~isempty(loc)    
            switch varargin{loc}
                case 'ebsdMatFile'

                    ebsd = load(varargin{loc+1});
                    ebsd_var = who(matfile(varargin{loc+1}));
                    ebsd = ebsd.(ebsd_var{1});
                    assert(length(unique(ebsd.phaseId))==1,...
                    'ODF_reduction_algo:e3', 'EBSD has multiple phases')
                    ori = ebsd.orientations;
                    flag = flag-1;

                case 'ebsd'
                    ebsd = varargin{loc+1};
                    assert(length(unique(ebsd.phaseId))==1,...
                        'ODF_reduction_algo:e3','EBSD has multiple phases')
                    ori = ebsd.orientations;
                    flag = flag-1;

                case 'orientation'
                    ori = varargin{loc+1};      
                    flag = flag-1;
            end
            
        end
        
    end
    
end

assert(flag>=0,'ODF_reduction_algo:e4','Multiple options for same input')
assert(flag==0,'ODF_reduction_algo:e5',...
      'Insufficient options atleast EBSD or orientation set required')

flag=1;  
  
if length(varargin)>2
    options = {'grainsMatFile','grains','kernel','kernelShape'};
        
    for i = 1: length(options)
        
        loc = find(strcmp(varargin,options{i}));
        
        if length(loc)>1
            error('ODF_reduction_algo:e4',...
                'Multiple options for same input')       
        end
        
        if flag>=0           
            if ~isempty(loc)
                switch varargin{loc}
                    case 'grainsMatFile'
                        grains = load(varargin{loc+1});
                        grains_var = who(matfile(varargin{loc+1}));
                        grains = grains.(grains_var{1});
                        assert(length(unique(grains.phaseId))==1,...
                     'ODF_reduction_algo:e6','Grains has multiple phases')
                        disp(...
              'Optimum kernel estimated from mean orientations of grains')
                        psi = calcKernel(grains.meanOrientation);
                        flag = flag-1;

                    case 'grains'
                        grains = varargin{loc+1};
                        assert(length(unique(grains.phaseId))==1,...
                      'ODF_reduction_algo:e6','Grains has multiple phases')
                        disp(...
               'Optimum kernel estimated from mean orientations of grains')
                        psi = calcKernel(grains.meanOrientation);
                        flag = flag-1;
                    case 'kernel'
                        assert(isa(varargin{loc+1},...
                            'deLaValeePoussinKernel'),...
                            'ODF_reduction_algo:e7',...
                            'Invalid kernel use deLaValeePoussinKernel')
                        psi = varargin{loc+1};
                        
                        flag = flag-1;
                    case 'kernelShape'
                        psi = deLaValeePoussinKernel('halfwidth',...
                            varargin{loc+1});
                        flag = flag-1;
                        
                end
                
            end
          
        end
    
    end
  
end

assert(flag>=0,'ODF_reduction_algo:e4','Multiple options for same input')
if flag==1 
    psi = deLaValeePoussinKernel('halfwidth',5*degree);
    disp(['Default initial kernel shape factor: ',num2str(5),' degree'])
end

if ~isempty (find(strcmp(varargin,'MisAngDist'), 1))
    
    assert(~isempty(grains),'ODF_reduction_algo:e8',...
        'Grain information missing')    
        
    loc = find(strcmp(varargin,'sharedArea'));
    if ~isempty(loc)
        fl = importdata(varargin{loc+1});
        fl = fl.data;
        assert(ns==length(unique([fl(:,1); fl(:,2)])),...
            'ODF_reduction_algo:e9',...
     'Shared surface data and number of reduced orienation do not match')
        
    else    
        error('ODF_reduction_algo:e10',...
            'Grain boundary shared area missing')    
    end    
      
    loc = find(strcmp(varargin,'nbins'));
    if isempty(loc)
        nbin=13;
    else
        nbin = varargin{loc+1};
        assert(mod(nbin,1)==0,'ODF_reduction_algo:e11',...
            'Please enter integer value for number of bins')        
    end                 
    
end


%% ODF calculation and L1 minimization setup
odf = calcKernelODF(ori,'kernel',psi);

if size(ori,2)>size(ori,1)
   ori = reshape(ori,[length(ori),1]); 
end

ll=10;
hl = 55;

step = 1;

ero=100;

e_mod = [];

lim = 10;

hh=[];

kappa = psi.halfwidth*180/pi; % initial kernel shape
%%
tic
for hw = ll:step:hl
% hw
%% part of code from mtex  
S3G = equispacedSO3Grid(ori.CS,ori.SS,'resolution',(hw/2)*degree);

weights = ones(1,length(ori));
M = sparse(1:length(ori),find(S3G,ori),weights,length(ori),length(S3G));

weights = full(sum(M,1));
weights = weights ./ sum(weights);

S3G = subGrid(S3G,weights~=0);
weights = weights(weights~=0);

%% Integer approximation 

lval = 0;
hval = ns;
ifc = 1.0;
ihval =  sum(round(hval.*weights));

while (hval-lval > hval*1e-15 || ihval < ns) && ihval ~= ns
    
    if ihval < ns
      hvalOld = hval;
      hval =hval+ifc*(hval-lval)/2.0;
      lval = hvalOld;      
      ifc=ifc * 2.0;      
      ihval = sum(round(hval.*weights));
      
    else
      hval = (lval+hval)/2.0;
      ifc = 1.0;
      ihval = sum(round(hval.*weights));       
            
    end 
    
end  

screen = round(weights.*hval);
diff = sum(screen) - ns;

[~,weights_loc] = sort(weights);
co = 1;

while diff>0
    if screen(weights_loc(co))>0 
        screen(weights_loc(co)) = screen(weights_loc(co)) - 1;
        diff = sum(screen) - ns;
    end
    co=co+1;
end

fval = screen(screen>0);

%% mean orientation estimation and kernel optimization 

[xt,yt]=find(M==1);
ytun = unique(yt);
ytfreq = histc(yt,ytun);
oriseq = mat2cell(ori(xt),ytfreq);

oriseq = oriseq(screen>0);

ori_f = S3G(screen==1);

oriseq_p = oriseq(fval>1);

ind = num2cell(fval(fval>1)');
pendList = cellfun(@splitMean, oriseq_p, ind, 'UniformOutput', false);

ori_f = [ori_f [pendList{:}]];
  
[odfred,h] = odfEst(ori_f,ones(ns,1),kappa,odf);
hh = [hh h];
er = calcError(odf,odfred);

er = calcError(odf,odfred);
if er<ero
    orired_f=ori_f;
    odfred_f = odfred;
    ohw = h;
    ero=er;
end
e_mod = [e_mod er];

[~,dd] = min(e_mod);
if length(e_mod)-dd>lim
    break
end

end

time = toc

loc = find(strcmp(varargin,'path'));
fpath = '';
if ~isempty(loc)
    fpath = varargin{loc+1};
    if ~exist([fpath '/mat_files/'], 'dir')
        mkdir([fpath '/mat_files/'])
    end    
    
    
end 

if isempty (find(strcmp(varargin,'MisAngDist'), 1))
    if ~isempty(fpath)
        save([fpath '/mat_files/' 'red_ori_' num2str(ns)],'orired_f')
        save([fpath '/mat_files/' 'red_odf_' num2str(ns)],'odfred_f')
        
        grainOri = round(Euler(orired_f).*180/pi);
        fileID = fopen(...
            [fpath '/mat_files/' '/OutFile_' num2str(ns) '.txt'],'w');
        fprintf(fileID,'L1 error ODF reconstruction error = %4.5f\n',ero);
        fprintf(fileID,'Initial Kernel shape factor = %4.5f\n',kappa);
        fprintf(fileID,'Final Kernel shape factor = %4.5f\n',ohw);
        fprintf(fileID,'****************************************\n');
        fprintf(fileID,'EulerAngles(phi1 Phi phi2) \n');
        fprintf(fileID,'%4.1i  %4.1i  %4.1i\n',grainOri(:,1:3)');
        
    end
else
    
    [orired_f,varargout{1},varargout{2},varargout{3}] = ...
        mdf_Anglefitting_algo_MC(grains,orired_f,fl,nbin);
    
    loc = find(strcmp(varargin,'GrainVolume'));
    if ~isempty(loc)
        gw = importdata(varargin{loc+1});
        disp('Reduced ODF corrected considering grain volume as weights')
        odfred_f=calcKernelODF(orired_f,'halfwidth',ohw*degree,...
            'weights',gw);
        ero = calcError(odf,odfred_f);
    else
        disp('Reduced ODF estimated considering equally weighted grains')
    end
    
    if ~isempty(fpath)
        
        save([fpath '/mat_files/' 'red_ori_' num2str(ns)],'orired_f')
        save([fpath '/mat_files/' 'red_odf_' num2str(ns)],'odfred_f')
        mdf_err = sum(abs(varargout{1} - varargout{2}))/2;
        grainOri = round(Euler(orired_f).*180/pi);
        fileID = fopen(...
            [fpath '/mat_files/' '/OutFile_' num2str(ns) '.txt'],'w');
        
        fprintf(fileID,...
            'L1 error ODF reconstruction error = %4.5f\n',ero);
        fprintf(fileID,...
         'L1 error miasorientation angle distribution = %4.5f\n',mdf_err);
        fprintf(fileID,'Initial Kernel shape factor = %4.5f\n',kappa);
        fprintf(fileID,'Final Kernel shape factor = %4.5f\n',ohw);
        fprintf(fileID,'****************************************\n');
        fprintf(fileID,'Angle distribution: \n');
        fprintf(fileID,...
            'Misorientation angle    Input data    Output data\n');
        fprintf(fileID,'%1.1f                   %1.3f         %1.3f\n',...
            [varargout{3}' varargout{1}.*100 varargout{2}.*100]');
        fprintf(fileID,'****************************************\n');
        fprintf(fileID,'Grain ID   EulerAngles(phi1 Phi phi2) \n');
        fprintf(fileID,'%4.1i       %4.1i  %4.1i  %4.1i\n',...
            [(1:ns);grainOri(:,1:3)']);
    
    
    end
    
end








