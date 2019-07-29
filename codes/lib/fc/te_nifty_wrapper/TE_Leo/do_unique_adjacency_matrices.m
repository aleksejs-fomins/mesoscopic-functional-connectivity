function network_adj_matrices= do_unique_adjacency_matrices(perf_def,path_raw,mouse_ID,sessions_IDs)

% if perf_def = 1; than high performance sessions above 80% are pooled

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
network_performance_threshold = 0.8;
pagerank_constant = 0.85;

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

% preallocate space for adjacency matrices
for i=1:2 % Hit and CR trials
    for ii=1:3 % Network time interval
        for count = 1:sessions % Number of sessions in a mouse folder
            
                % find high-performance sessions
    
    if perf_def == 1
        
        high_perf_sessions = performance > network_performance_threshold;
        high_perf_sessions_indices = find(performance > network_performance_threshold);

    elseif perf_def == 2
        
        high_perf_sessions1= performance < network_performance_threshold;
        high_perf_sessions2= performance < 0.6;
        high_perf_sessions = high_perf_sessions1 - high_perf_sessions2;
        
        high_perf_sessions_indices= find(high_perf_sessions)
        
    elseif perf_def == 3
        
        high_perf_sessions = performance > network_performance_threshold;
        high_perf_indices = find(performance > network_performance_threshold);
        
        if length(high_perf_indices)>2
            high_perf_sessions = high_perf_sessions(1:2);
            high_perf_sessions_indices = high_perf_indices(1:2);
        elseif length(high_perf_indices)==1
            high_perf_sessions = high_perf_sessions(1);
            high_perf_sessions_indices = high_perf_indices(1)
        else
            high_perf_sessions = NaN;
            high_perf_sessions_indices = NaN;
        end
    end
        
            network_adj_matrices{1,i}{1,ii}= cell(1,length(high_perf_sessions_indices));
        end
    end
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
    
    
    % -----------------------------------------------------------------
    % --- redefine adjacency matrices for a trial unique links only ---
    % -----------------------------------------------------------------

%     network_adj_matrices_GO = network_difference ==1
%     network_adj_matrices_NOGO = network_difference ==-1;
        
    network_time_interval_count = network_time_interval_count + 1;
    
    % --- Unique Links ---
    links_indices_go = network_adj_matrices_GO>0; 
    links_indices_nogo= network_adj_matrices_NOGO>0;
    
    % compute number of links present in Hit trials
    indices_sum = links_indices_go(:,:,1:end) - links_indices_nogo(:,:,1:end);

    % test for the repeated links in 2 sessions
    network_adj_matrices_GO = (indices_sum == 1);
    % test_all_go intersects along all sessions and keeps only shared links
    %test_all_go= all(test_go(:,:,sessions),3);

    network_adj_matrices_NOGO = (indices_sum == -1);
    % test_all_nogo intersects along all sessions and keeps only shared links
    %test_all_nogo= all(test_nogo(:,:,sessions),3);
    
    for count_sessions = 1:length(high_perf_sessions_indices)
        
        network_adj_matrices{1,1}{1,network_time_interval_count}{1,count_sessions} = ...
            network_adj_matrices_GO(:,:,high_perf_sessions_indices(count_sessions));
        network_adj_matrices{1,2}{1,network_time_interval_count}{1,count_sessions} = ...
            network_adj_matrices_NOGO(:,:,high_perf_sessions_indices(count_sessions));

    end
    
    
end

end
