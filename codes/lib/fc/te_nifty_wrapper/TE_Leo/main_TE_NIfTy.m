%% add NIfTy toolbox and all the required scripts to the current MATLAB path

restoredefaultpath;
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\mtp\');
%addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\plots\');
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\NIfTy_Ver1\');
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\TE_Leo\');

%% define constants

samples_CUE = 21:30;    % 1-1.5 seconds trial time
samples_TEX = 41:70;    % 2-3.5 seconds trial time
samples_LIK = 71:120;   % 3.5-6 seconds trial time
samples_ALL = 1:200;    % 0-10  seconds

% set parameters

% select session folders
fprintf('Select session folders')
[path_names] = uigetfile_n_dir;

% check that every folder contains data and behavior files
for count = 1:length(path_names)
    
    path_session_id = path_names{1,count};
    
    if ~exist(fullfile(path_session_id,'data.mat'),'file')
        error('ERROR: data file missing in folder: %s',path_session_id)
    end
    
    if ~exist(fullfile(path_session_id,'behaviorvar.mat'),'file')
        error('ERROR: behavior file missing in folder: %s',path_session_id)
    end
    
end

% ask user whether to save results files
save_dialog = questdlg('Save results files? Press "Cancel" or close dialog to abort.');
if strcmp(save_dialog,'Yes')
    save_results = true;
elseif strcmp(save_dialog,'No')
    save_results = false;
else
    error('Program aborted by user')
end

 
%---- loop through selected sessions folders and set parameters ------------------------------------

clear parameters
clear my_data
clear loaded_vard
clear results

for count = 1:length(path_names)
    
    %---- set parameters for TE estimation ---------------------------------------------------------
    
    parameters(count).samples_window = 'ALL';
    parameters(count).trials_type = 'GO';
    
    parameters(count).subsampling = true;
    subsampling_windows_lenghts = 4*ones(1,50); %uniform subsampling, 50 bins
    %subsampling_windows_lenghts = [20,20,30,50,80]; %sum up to 200
           
    % choose which channels to analyze
    parameters(count).channels = 1:12;
    
    % define total number of channels, independent of how many are currently being analyzed
    parameters(count).channels_total = 12;
    % define labels for all brain areas
    parameters(count).channel_labels = {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12'};
    
    % set plausible range for delay among areas, in terms of number of bins(after subsampling)
    parameters(count).delay_min = 1;
    parameters(count).delay_max = 6;
    parameters(count).delay_stepsize = 1;
    parameters(count).receiver_embedding_tau = 1;
    
    parameters(count).sample_rate = 20;
    
       
    %---- add automatically generated parameters ---------------------------------------------------
    
    % compute all channel pairs
    parameters(count).channel_pairs = [nchoosek(parameters(count).channels,2);nchoosek(flip(parameters(count).channels),2)];
        
    % load data and behaviour files
    path_session_id = path_names{1,count};
    loaded_vars(count).datavar = load(fullfile(path_session_id,'data.mat'),'data');
    loaded_vars(count).behaviorvar = load(fullfile(path_session_id,'behaviorvar.mat'),'iGO','iNOGO','iMISS','iFA');
    
    % add session id to parameters
    [~,parameters(count).session_id,~] = fileparts(path_session_id);
     
    switch parameters(count).samples_window
        case 'CUE'
            parameters(count).samples_original = samples_CUE;
        case 'TEX'
            parameters(count).samples_original = samples_TEX;
        case 'LIK'
            parameters(count).samples_original = samples_LIK;
        case 'ALL'
            parameters(count).samples_original = samples_ALL;
        otherwise
            error('ERROR: Unknown samples window. Valid options: CUE, TEX, LIK, ALL')
    end
    
    switch parameters(count).trials_type
        case 'GO'
            parameters(count).trials_numbers = loaded_vars(count).behaviorvar.iGO;
        case 'NOGO'
            parameters(count).trials_numbers = loaded_vars(count).behaviorvar.iNOGO;
        case 'MISS'
            parameters(count).trials_numbers = loaded_vars(count).behaviorvar.iMISS;
        case 'FA'
            parameters(count).trials_numbers = loaded_vars(count).behaviorvar.iFA;
        case 'ALL'
            parameters(count).trials_numbers = cat(2,loaded_vars(count).behaviorvar.iGO,loaded_vars(count).behaviorvar.iNOGO,loaded_vars(count).behaviorvar.iMISS,loaded_vars(count).behaviorvar.iFA);
        otherwise
            error('ERROR: Unknown trials type. Valid options: GO, NOGO, MISS, FA, ALL')
    end
    
    if parameters(count).subsampling
        
        parameters(count).subsampling_windows_lenghts = subsampling_windows_lenghts;
        temp_windows_n = length(parameters(count).subsampling_windows_lenghts);
        
        % check that the sum of the subsampling windows lengths is equal to the entire data length
        if sum(parameters(count).subsampling_windows_lenghts) ~= length(parameters(count).samples_original)
            error('ERROR: invalid choice of subsampling windows')
        end
        
        my_data(count).data = NaN([length(parameters(count).trials_numbers),temp_windows_n,parameters(count).channels_total]);
        parameters(count).samples_timesteps = NaN([1,temp_windows_n]);
        
        temp_window_start = 1;
        for i = 1:temp_windows_n
            temp_window_end = temp_window_start + parameters(count).subsampling_windows_lenghts(i) - 1;
            my_data(count).data(:,i,:) = mean(loaded_vars(count).datavar.data(parameters(count).trials_numbers,temp_window_start:temp_window_end,:),2);
            parameters(count).samples_timesteps(i) = (parameters(count).samples_original(1) + temp_window_start - 1) / parameters(count).sample_rate;
            temp_window_start = temp_window_end + 1;
        end
        
    else
        
        my_data(count).data = loaded_vars(count).datavar.data(parameters(count).trials_numbers,parameters(count).samples_original,:);
        parameters(count).samples_timesteps = parameters(count).samples_original / parameters(count).sample_rate;
    end
    
    parameters(count).states_method = 'UniCB';
    %parameters(count).states_bins_count = 4 * ones(1,parameters(count).channels_total);
    parameters(count).states_bins_count = estimate_TE_bins(my_data(count).data);
    
    % sort parameter fields alphabetically
    parameters = orderfields(parameters);
        
end

    

% TE estimation 

% record starting time
temp_t_start = tic;

% parallel for loop
parfor count = 1:length(path_names)        
       
    % estimate TE
    results(count) = TE_NIfTy(my_data(count).data,parameters(count));    
        
end

% compute elapsed time
temp_t_elapsed = toc(temp_t_start);

fprintf('\nTransfer Entropy calculation ended: %s \nCalculation took %.0f minutes (%.0f seconds)\n', datestr(now), temp_t_elapsed/60, temp_t_elapsed);


% save results (if user decided to save)

temp_results = results;
clear results

for count = 1:length(path_names)
    
    path_session_id = path_names{1,count};
        
    if save_results
        
        path_save_results_folder = fullfile(path_session_id,'Transfer Entropy');
        if ~exist(path_save_results_folder,'dir')
            mkdir(path_save_results_folder)
        end
        path_save_results_file = fullfile(path_save_results_folder,sprintf('results_%s',parameters(count).trials_type));
        clear results
        results = temp_results(count);
        save(path_save_results_file,'results');
        
        fprintf('\nResults saved\n');
                       
    end
    
end

