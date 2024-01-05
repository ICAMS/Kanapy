function [orilist,ein,eout,mbin] = ...
    gb_textureReconstruction(inp,orilist,fl,nbin)

tic
RVEgrains = length(unique([fl(:,1); fl(:,2)])); 
bin_score_area = sparse(fl(:,1),fl(:,2),fl(:,3));  % RVEgrains,RVEgrains had been given as dimensions
area = bin_score_area(bin_score_area>0);

index = 1:RVEgrains;

oR = fundamentalRegion(orilist.CS,orilist.CS);
maxAng = oR.maxAngle;
bins = linspace(0,maxAng,nbin+1);
mbin = 0.5.*(bins(1:end-1)+bins(2:end)).*180/pi;

if strcmp(class(inp),'grain2d')
grains = inp;
gb = grains.boundary('indexed');
seglen = gb.segLength;
misori = gb.misorientation;
a_in = angle(misori);
[~,loc] = histc(a_in,bins);
et = sparse(loc,1:length(loc),seglen,nbin,length(loc));
ein=sum(et,2);
ein = full(ein);
ein = ein./sum(ein); %weights of each bin based on segment length
else
ein = inp(:);
if nbin~=length(ein)
nbin=length(ein);
end
end

%% first create bins of disorientation angle and apply weights to bins

e_old = 100;
flag=1; 
tol=0.05;
rep_count_lim = 200;

%% Monto Carlo algorithm

while flag==1

i1 = randi([1 max(index)],1);
i2 = randi([1 max(index)],1);

% switch index
index([i1 i2]) = index([i2 i1]);
orilist([i1 i2]) = orilist([i2 i1]);
an = angle(orilist'*orilist);
a_out = an(bin_score_area~=0);
[~,loc] = histc(a_out,bins);
et = sparse(loc,1:length(loc),area,nbin,length(loc));
eout = sum(et,2);
eout = full(eout);
eout = eout./sum(eout);

e = sum(abs(ein - eout))/2;

if e<e_old
    e
    e_old=e;
    rep_count=0;
        
else
    % index reset
    index([i2 i1]) = index([i1 i2]);
    orilist([i2 i1]) = orilist([i1 i2]);
    rep_count=rep_count+1;
end

if rep_count>rep_count_lim
    tol = 1.2*tol;
end

if e<tol
    flag=0;
    break
end

end
time = toc
