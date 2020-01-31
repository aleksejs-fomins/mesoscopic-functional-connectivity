function [network_total_links,network_shared_links,network_shared_links_p_value,network_unique_links,performance]= do_links_analysis(path_raw,mouse_ID,sessions_IDs)

% channel labels should be imported

% select session folders
%fprintf('Select session folders\n');[path_names] = uigetfile_n_dir;

% path_names are read from the 
%[path_names] = uigetfile_n_dir('','Select session folders');
 
sessions = length(sessions_IDs);
path_names =fullfile(path_raw,mouse_ID,sessions_IDs);

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m

% define thresholds
network_p_threshold = 0.01;

% check that every folder contains performance file
for count = 1:sessions
    path_session_id = path_names{1,count};
    if ~exist(fullfile(path_session_id,'Transfer Entropy','performance.mat'),'file')
        error('ERROR: performance.mat file missing in folder: %s',fullfile(path_session_id,'Transfer Entropy'))
    end
end

% loop through selected session folders and import performance files
performance = NaN([1,sessions]);
for count = 1:sessions
    % load results file
    path_session_id = path_names{1,count};
    temp = load(fullfile(path_session_id,'Transfer Entropy','performance.mat'));
    performance(count) = temp.performance;
    clear temp
end

network_time_interval_count = 0;
% repeat plot for CUE, TEX and LIK
% time intervals (in seconds)
% CUE=[1,1.5], TEX=[2,3.5], LIK=[3.5,6]
for network_time_interval = {[1,1.5],[2,3.5],[3.5,6]}
    
    
    % generate adjacency matrices for GO and NOGO
    
    clear network_adj_matrices_GO
    clear network_adj_matrices_NOGO
    
    for trials_type = {'GO','NOGO'};
        
        % check that every folder contains results file
        
        for count = 1:sessions
            path_session_id = path_names{1,count};
            if ~exist(fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1},'.mat']),'file')
                error('ERROR: results_%s.mat file missing in folder: %s',trials_type{1,1},fullfile(path_session_id,'Transfer Entropy'))
            else
                
                % load results file
                path_session_id = path_names{1,count};
                load(fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1}]),'results')
                
                % select timesteps in the desired time interval
                network_time_steps = find(results.parameters.samples_timesteps >= min(network_time_interval{1,1}) & results.parameters.samples_timesteps < max(network_time_interval{1,1}));
                if length(network_time_steps) < 1
                    error('ERROR: no data points found in the specified time interval.')
                end
                
                % store the new adjacency matrix (assumption: all training sessions have the same number of channels)
                if strcmp(trials_type{1,1},'GO')
                    network_adj_matrices_GO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                elseif strcmp(trials_type{1,1},'NOGO')
                    network_adj_matrices_NOGO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                end
            end
        end
    end
        
    network_time_interval_count = network_time_interval_count + 1;
    
    %---------------------------------
    %--- update links total/shared ---
    %---------------------------------
    
    %GO
    
    % find number of nodes and links
    network_size = size(network_adj_matrices_GO,1);
    network_links_total = network_size^2 - network_size;
    % find indices of weights above threshold
    links_indices = network_adj_matrices_GO > 0;
    % compute number of links
    network_links = squeeze(sum(sum(links_indices,1),2));
    network_total_links.GO(:,network_time_interval_count) = network_links;
    
    % find indices of weights above threshold
    links_indices_go = network_adj_matrices_GO > 0;

    % compute number of shared links between consecutive sessions
    indices_sum = links_indices(:,:,1:end-1) + links_indices(:,:,2:end);
    network_links_shared = squeeze(sum(sum(indices_sum == 2,1),2));
    network_shared_links.GO(:,network_time_interval_count) = network_links_shared;

    % compute p-value of shared links
    network_links_shared_p_value = NaN([length(network_links_shared),1]);
    warning('off','MATLAB:nchoosek:LargeCoefficient');
    for count = 1:sessions-1
        network_links1 = network_links(count);
        network_links2 = network_links(count+1);
        network_links_shared_p_value(count) = nchoosek(network_links_total,network_links_shared(count))*nchoosek(network_links_total-network_links_shared(count),network_links1-network_links_shared(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_shared(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
    end
    
    network_shared_links_p_value.GO(:,network_time_interval_count)= network_links_shared_p_value;

    
    % NOGO
    % find number of nodes and links
    network_size = size(network_adj_matrices_NOGO,1);
    network_links_total = network_size^2 - network_size;
    % find indices of weights above threshold
    links_indices = network_adj_matrices_NOGO > 0;
    % compute number of links
    network_links = squeeze(sum(sum(links_indices,1),2));
    network_total_links.NOGO(:,network_time_interval_count) = network_links;
    
    % find indices of weights above threshold
    links_indices_nogo = network_adj_matrices_NOGO > 0;
    
    % compute number of shared links between consecutive sessions
    indices_sum = links_indices(:,:,1:end-1) + links_indices(:,:,2:end);
    network_links_shared = squeeze(sum(sum(indices_sum == 2,1),2));
    network_shared_links.NOGO(:,network_time_interval_count) = network_links_shared;
    
    % compute p-value of shared links
    network_links_shared_p_value = NaN([length(network_links_shared),1]);
    warning('off','MATLAB:nchoosek:LargeCoefficient');
    for count = 1:sessions-1
        network_links1 = network_links(count);
        network_links2 = network_links(count+1);
        network_links_shared_p_value(count) = nchoosek(network_links_total,network_links_shared(count))*nchoosek(network_links_total-network_links_shared(count),network_links1-network_links_shared(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_shared(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
    end
    
    network_shared_links_p_value.NOGO(:,network_time_interval_count)= network_links_shared_p_value;
    
    %---------------------------  
    %--- update unique links ---
    %---------------------------
    
    % compute number of links present in Hit trials
    indices_sum = links_indices_go(:,:,1:end) - links_indices_nogo(:,:,1:end);
    % --- exclusive links for Hit ---
    network_unique_links.GO(:,network_time_interval_count) = squeeze(sum(sum(indices_sum == 1,1),2));
    
    % --- exclusive links for CR ---
    network_unique_links.NOGO(:,network_time_interval_count) = squeeze(sum(sum(indices_sum == -1,1),2));

%     for i = 1:length(high_perf_sessions_indices)
%         G_NOGO{i} = NaN;
%     end
    
    
    %--- Intersect GO and NOGO figure ---
%     
%     % compute for different links
% 
%     network_intersect = NaN([size(network_adj_matrices_GO,1),length(high_perf_sessions_indices)]);
%     
%     for i = 1:length(high_perf_sessions_indices)
%         network_cluster{i} = NaN;
%     end
%     
%     for count_sessions = 1:length(high_perf_sessions_indices)  
%         % --- Octave find cluster ---
%         network_intersect(:,:,count_sessions) = network_adj_matrices_GO(:,:,high_perf_sessions_indices(count_sessions)) -network_adj_matrices_NOGO(:,:,high_perf_sessions_indices(count_sessions))>0;
% 
%     end
%     
%     shared_intersect_subnetwork_high_perf = all(network_intersect,3);
%     
%     %subplot_count = subplot_count + 1;
%     %subplot(3,1,subplot_count)
%     figure
%     hold on
%     
%     % create plot object for GO
%     network_plot = plot(digraph(shared_intersect_subnetwork_high_perf>= network_frequency_threshold,channel_labels));
%     
%     % set node coordinates
%     network_plot.XData = layout_spring_coordinates.X;
%     network_plot.YData = layout_spring_coordinates.Y;

end

end
