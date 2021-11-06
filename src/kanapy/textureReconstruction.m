function [orired_f,odfred_f,ero,varargout]=...
    textureReconstruction(ns,varargin)
%% 
% This function sysytematically recontructs the ODF by a smaller number of
% orientations (refer .....)  
% also the misorientation distribution is   
%
%
% Syntax:
% [ori,odf,e]=ODF_reduction_algo(n,'ebsdMatfile',ebsdfile)
% [ori,odf,e]=ODF_reduction_algo(n,'ebsd',ebsd) 
% [ori,odf,e]=ODF_reduction_algo(n,'ebsdMatfile',ebsdfile,  
%           'grainsMatfile',grainsfile)
% [ori,odf,e]=ODF_reduction_algo(n,'ebsd',ebsd,'grains',...
%           'grains')
% [ori,odf,e]=ODF_reduction_algo(n,'orientations',ori)  
% [ori,odf,e]=ODF_reduction_algo(n,'orientations',ori,...  
%           'kernel',psi)
% [ori,odf,e]=ODF_reduction_algo(n,'orientations',ori,...  
%           'kernelShape','kappa') 
%
% Inputs:
% 
%  1) n: number of reduced orientations/grains in RVE
%  2) Either path+filename of ebsd data saved as *.mat file (it should 
%     contain only one phase/mineral) or ebsd(single phase)/orientations
%  3) Either path+filename of the estiamted grains from above 
%     EBSD saved as *.mat file (it should contain only one phase/mineral)
%     or kernel(only deLaValeePoussinKernel)/kernelshape, if nothing       
%     mentioned then default value kappa = 5 (degree) is assumed.
%
% Output: reduced orientation set, ODF and L1 error 
% 
%% input fields and checks
    
%run('../../libs/mtex/install_mtex.m')
 
options = {'ebsdMatFile','ebsd','orientation'};
flag = 1; 
grains = [];
for i = 1:length(options)

    loc = find(strcmp(varargin,options{i}));

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

flag=1;  
  
if length(varargin)>2
    options = {'grainsMatFile','grains','kernel','kernelShape'};
        
    for i = 1: length(options)
        
        loc = find(strcmp(varargin,options{i}));
                
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
                            'deLaValleePoussinKernel'),...
                            'ODF_reduction_algo:e7',...
                            'Invalid kernel use deLaValeePoussinKernel')
                        psi = varargin{loc+1};
                        
                        flag = flag-1;
                    case 'kernelShape'
                        psi = deLaValleePoussinKernel('halfwidth',...
                            varargin{loc+1});
                        flag = flag-1;                        
                end                
            end          
        end    
    end  
end

assert(flag>=0,'ODF_reduction_algo:e4','Multiple options for same input')
if flag==1 
    psi = deLaValleePoussinKernel('halfwidth',0.08726646259971647);
    disp(['Default initial kernel shape factor: ',num2str(5),' degree'])
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
S3G = equispacedSO3Grid(ori.CS,ori.SS,'resolution',(hw/2)*(pi/180));

weights = ones(1,length(ori));
M = sparse(1:length(ori),find(S3G,ori),weights,length(ori),length(S3G));

weights = full(sum(M,1));
weights = weights ./ sum(weights);

S3G = subGrid(S3G,weights~=0);
weights = weights(weights~=0);

%% Integer approximation 

lval = 0;
hval = double(ns);
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












