function col = get_ipf_col(ang)
%% 
% This function creates a list of RGB colors
% representing the given angles in an IPF color key
% currently cubic crystal symmetry and cubic speciment geometry are assumed

% Input:
%  ang: list of Euler angles
%
% Output:
%  col: list of RGB colors

    cs = crystalSymmetry('cubic');
    ss = specimenSymmetry('O');
    ipfKey = ipfHSVKey(cs);
    ori = orientation.byEuler(ang, cs, ss);
    col = ipfKey.orientation2color(ori);

end