addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\mtp\');

[pathname] = uigetfile_n_dir;

data = [];
iGO = [];
iNOGO = [];
iMISS = [];
iFA = [];

%loop through all the selected folders
for count_file = 1:length(pathname)
    
    temp_path_behaviorvar_file = fullfile(pathname{count_file},'behaviorvar.mat');
    temp_path_data_file = fullfile(pathname{count_file},'data.mat');
    
    temp_behaviorvar = load(temp_path_behaviorvar_file,'iGO','iNOGO','iMISS','iFA');
    temp_data = load(temp_path_data_file,'data');
            
    iGO = cat(2,iGO,size(data,1) + temp_behaviorvar.iGO);
    iNOGO = cat(2,iNOGO,size(data,1) + temp_behaviorvar.iNOGO);
    iMISS = cat(2,iMISS,size(data,1) + temp_behaviorvar.iMISS);
    iFA = cat(2,iFA,size(data,1) + temp_behaviorvar.iFA);
    
    % normalize
    for count_channel = 1:size(temp_data.data,3)
        temp_data.data(:,:,count_channel) = temp_data.data(:,:,count_channel) / max(max(abs(temp_data.data(:,:,count_channel))));
    end
    
    data = cat(1,data,temp_data.data);
    
end

%% save merged data.mat and behaviorvar.mat files

save('E:\mtp_7\mtp_7_all\data.mat','data')
save('E:\mtp_7\mtp_7_all\behaviorvar.mat','iGO','iNOGO','iMISS','iFA')

%%



