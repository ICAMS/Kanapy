%%
%
%%


%% Fitting a rotation to vectors
%

rot = rotation.rand

left = vector3d.rand(3);
right = rot * left;


rotation.fit(left,right)

