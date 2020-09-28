% select session folders
fprintf('Select session folders \n')
[path_names] = uigetfile_n_dir;

sessions = length(path_names);

% check that every folder contains desired data
for count = 1:sessions
    path_session_id = path_names{1,count};
    if ~exist(fullfile(path_session_id,'behaviorvar.mat'),'file')
        error('ERROR: behaviorvar.mat file missing in folder: %s',fullfile(path_session_id,'Transfer Entropy'))
    end
end

% loop through selected session folders

for count = 1:sessions
    
    % load results file
    path_session_id = path_names{1,count};
    load(fullfile(path_session_id,'behaviorvar.mat'));
    
    % perform changes/computation.....
    performance = (length(iGO) + length(iNOGO))/(length(iGO) + length(iNOGO) + length(iMISS) + length(iFA));
    
    % save
    path_save_results_folder = fullfile(path_session_id,'Transfer Entropy');
    path_save_results_file = fullfile(path_save_results_folder,'performance.mat');
    save(path_save_results_file,'performance');
    
end
