
addpath(genpath('./matlab/PESQ/MATLAB/'));
addpath(genpath('./matlab/SRMR/MATLAB/'));
addpath(genpath('./matlab/STOI/MATLAB/'));

disp(files)
disp(mKinds)

pesq_main(models, dirs, files, mKinds);
srmr_main(models, dirs, files, mKinds);
stoi_main(models, dirs, files, mKinds);


