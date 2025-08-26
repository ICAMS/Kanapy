function col = get_ipf_col(ang, ckey)
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
    switch ckey
        case 1
            colorKey = BungeColorKey(cs);
        case 2
            colorKey = ipfHKLKey(cs);
        otherwise
            colorKey = ipfHSVKey(cs);
    end
    col = colorKey.orientation2color(ori);
end