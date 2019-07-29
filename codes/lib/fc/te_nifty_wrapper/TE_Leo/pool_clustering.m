% Fig. pool selected mice and calculate clustering

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m
cd 'E:\data\texture_discrimination\mtp_12\mtp_12_TE'
path_raw = pwd;

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
pagerank_constant = 0.85;
network_frequency_threshold = 0.5;
% define as 0 for naive animals i.e. pool sessions below the performance threshold
% define as 1 for expert animals i.e. pool sessions above the performance threshold
perf_def = 1;

% select mice folders
[paths_mice] = uigetfile_n_dir('','Select mice folders');
mice = length(paths_mice);

% get mouse ID
mouse_IDs = cell([1,mice]);
for count_mice = 1:mice
    [~,mouse_IDs{count_mice},~] = fileparts(paths_mice{count_mice});
end

% import all channel labels for comparison
channel_labels_all = cell([1,mice]);
for count_mice = 1:mice
    temp_file_name = fullfile(paths_mice{count_mice},'channel_labels.mat');
    if exist(temp_file_name,'file')
        temp = load(temp_file_name);
        channel_labels_all{count_mice} = temp.channel_labels;
    else
        error('ERROR: channel_labels.mat file missing in folder: %s',paths_mice{count_mice})
    end
end

% find shared channel labels among all mice
channel_labels_shared = channel_labels_all{1};
for count_mice = 1:mice
    channel_labels_shared = intersect(channel_labels_shared,channel_labels_all{count_mice},'stable');
end

% find the indices(positions) of the shared labels in each mouse channel labels
channel_labels_all_indices = cell([1,mice]);
for count_mice = 1:mice
    [~,~,channel_labels_all_indices{count_mice}] = intersect(channel_labels_shared,channel_labels_all{count_mice},'stable');
end

clustering_Hit=cell(3,1);
clustering_CR=cell(3,1);

for count_mice = 1:mice
        
        % Get a list of all files and folders in the mouse folder
        all_files = dir(paths_mice{count_mice});
        % Remove . and .. folders
        all_files(1:2) = [];
        % Get a logical vector that tells which is a directory
        dirFlags = [all_files.isdir];
        % Extract only those that are directories.
        sessions_IDs = {all_files(dirFlags).name};
        sessions = length(sessions_IDs);
        
        mouse_ID= mouse_IDs{count_mice};
        % import performance files and create performance vector
        performance = NaN([1,sessions]);
        
        clustering_Hit{1,count_mice}= mouse_ID;
        clustering_CR{1,count_mice}= mouse_ID;
        
        for count_sessions = 1:sessions
            % load results file
            path_session_id = fullfile(paths_mice{count_mice},sessions_IDs{1,count_sessions});
            temp_file_name = fullfile(path_session_id,'Transfer Entropy','performance.mat');
            if exist(temp_file_name,'file')
                temp_loaded_data = load(temp_file_name);
                performance(count_sessions) = temp_loaded_data.performance;
            else
                error('ERROR: performance.mat file missing in folder: %s',fullfile(path_session_id,'Transfer Entropy'))
            end
        end
        
        % find high performance sessions
        high_perf_sessions_indices = find(performance > network_performance_threshold);
        
        % ------------- Pool Clustering without Session Name ------------- 
%         [outdegree{count_mice},indegree{count_mice},clustering{count_mice},eigencentrality{count_mice},...
%             clustering_Hit{count_mice},clustering_CR{count_mice}]=do_clustering_analysis(perf_def,path_raw,mouse_ID,sessions_IDs);
%         
        %  ------------- Pool Clustering with Session Name ------------- 
        clustering_Hit{2,count_mice}= sessions_IDs(high_perf_sessions_indices);
        clustering_CR{2,count_mice}= sessions_IDs(high_perf_sessions_indices);

        [outdegree{count_mice},indegree{count_mice},clustering{count_mice},eigencentrality{count_mice},...
            clustering_Hit{3,count_mice},clustering_CR{3,count_mice}]=do_clustering_analysis(perf_def,path_raw,mouse_ID,sessions_IDs);
        
        
%         % get adjacency matrices for GO and NOGO
%         network_adj_matrices_GO = NaN([length(channel_labels_shared),length(channel_labels_shared),length(high_perf_sessions_indices)]);
%         network_adj_matrices_NOGO = NaN([length(channel_labels_shared),length(channel_labels_shared),length(high_perf_sessions_indices)]);

        
        
%         % select shared links among all high-performance sessions (shared subnetwork)
%         cluster_mean_GO = mean(network_cluster_GO,2);
%         cluster_var_GO = var(network_cluster_GO,0,2);
%         
%         cluster_mean_NOGO = mean(network_cluster_NOGO,2);
%         cluster_var_NOGO = var(network_cluster_NOGO,0,2);
%         
%         % select shared links among all high-performance sessions (shared subnetwork)
%         shared_subnetwork_high_perf_GO(:,:,count_mice) = all(network_adj_matrices_GO,3);
%         high_freq_GO(:,:,count_mice) = mean(network_adj_matrices_GO,3)>= network_frequency_threshold;
%         high_freq_NOGO(:,:,count_mice) = mean(network_adj_matrices_NOGO,3)>= network_frequency_threshold;
        
        
end
    
  