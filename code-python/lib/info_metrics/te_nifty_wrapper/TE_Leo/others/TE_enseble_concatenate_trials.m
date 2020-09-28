%% add NIfTy toolbox to the current MATLAB path, plus scripts by Yaro

restoredefaultpath;
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\mtp\');
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\plots\');
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\NIfTy_Ver1\');


%%

my_data(1,:,1)=reshape(squeeze(data(iGO,41:70,1))',1,[]);
my_data(1,:,2)=reshape(squeeze(data(iGO,41:70,3))',1,[]);


%%

figure(1)
plot(1:length(my_data(:,:,1)),my_data(:,:,1))

figure(2)
plot(1:length(my_data(:,:,2)),my_data(:,:,2))


%%

% select data for the current channel pair, and swap tensor dimensions in order to suit the requirements of the NIfTy toolbox
DataRaster = permute(my_data(:,:,1:2),[3 2 1]);

% state the data using the method and bins specified in the parameters
MethodAssign = {1,1,'UniCB',{4};1,2,'UniCB',{4}};
StatesRaster = data2states(DataRaster, MethodAssign);

% loop over different values of delay u and choose the one the
% maximizes the average TE

Method = 'TE';
nT = size(StatesRaster,2);


% perform the information calculation for all time bins


VariableIDs = {1,2,2;... % Receiving variable in the future
    1,2,1;... % Receiving variable in the past
    1,1,1}; % Transmitting variable in the past
instinfo(StatesRaster, Method, VariableIDs, 'MCOpt', 'off')







