% Fig.?? Large component formation

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
network_frequency_threshold = 0.5;

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

% find the indices(positions) of the shared labels for each mouse
channel_labels_all_indices = cell([1,mice]);
for count_mice = 1:mice
    [~,~,channel_labels_all_indices{count_mice}] = intersect(channel_labels_shared,channel_labels_all{count_mice},'stable');
end

clear network_adj_matrices_GO;
clear network_adj_matrices_NOGO;

% time intervals (in seconds)
timesteps = (0:0.2:10);
network_time_intervals = mat2cell([timesteps(1:end-1)',timesteps(2:end)'],ones([1,50]),[2]);
for count_time_interval = 1:length(network_time_intervals)
    
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
        
        % import performance files and create performance vector
        performance = NaN([1,sessions]);
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
            
        % generate adjacency matrices for GO and NOGO
                
        for trials_type = {'GO','NOGO'};
            
            for count_sessions = 1:length(high_perf_sessions_indices)
                
                % check if folder contains results file
                path_session_id = fullfile(paths_mice{count_mice},sessions_IDs{1,high_perf_sessions_indices(count_sessions)});
                temp_file_name = fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1},'.mat']);
                if ~exist(temp_file_name,'file')
                    error('ERROR: results_%s.mat file missing in folder: %s',trials_type{1,1},fullfile(path_session_id,'Transfer Entropy'))
                else
                    % load results file
                    load(temp_file_name,'results')
                    
                    % select timesteps in the desired time interval
                    network_time_steps = find(results.parameters.samples_timesteps >= min(network_time_intervals{count_time_interval}) & results.parameters.samples_timesteps < max(network_time_intervals{count_time_interval}));
                    if length(network_time_steps) < 1
                        error('ERROR: no data points found in the specified time interval.')
                    end
                    
                    % add the adjacency matrix, only taking the shared channels
                    % only high performance sessions
                    if strcmp(trials_type{1,1},'GO')
                        temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                        network_adj_matrices_GO(:,:,count_sessions,count_mice,count_time_interval) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                    elseif strcmp(trials_type{1,1},'NOGO')
                        temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                        network_adj_matrices_NOGO(:,:,count_sessions,count_mice,count_time_interval) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                    end
                end
                
            end
        end
        
    end
    
end

%%

test_n = 100;

lcc_GO_size = NaN([1,length(network_time_intervals)]);
lcc_NOGO_size = NaN([1,length(network_time_intervals)]);

lcc_GO_pval = NaN([1,length(network_time_intervals)]);
lcc_NOGO_pval = NaN([1,length(network_time_intervals)]);

mean_GO = squeeze(mean(mean(network_adj_matrices_GO,3),4));
mean_NOGO = squeeze(mean(mean(network_adj_matrices_NOGO,3),4));

network_size = size(network_adj_matrices_GO,1);

for count_time_interval = 1:length(network_time_intervals)
    
    % GO
    components_GO = conncomp(digraph(mean_GO(:,:,count_time_interval) > network_frequency_threshold),'Type','weak','OutputForm','cell');
    max_size = 0;
    for count_component = 1:length(components_GO)
        if length(components_GO{count_component}) > max_size
           max_size =  length(components_GO{count_component});
        end
    end    
    lcc_GO_size(count_time_interval) = max_size;
    
    % p-value
    links_count = sum(sum(mean_GO(:,:,count_time_interval) > network_frequency_threshold));
    test_array = NaN(1,test_n);
    for count_test = 1:test_n
        rnd_adj_mat = zeros(network_size);
        rnd_adj_mat(randi([1,network_size^2],1,links_count)) = 1;
        components_test = conncomp(digraph(rnd_adj_mat),'Type','weak','OutputForm','cell');
        max_size = 0;
        for count_component = 1:length(components_test)
            if length(components_test{count_component}) > max_size
                max_size =  length(components_test{count_component});
            end
        end
        test_array(count_test) = max_size;
    end
    lcc_GO_pval(count_time_interval) = sum(test_array > lcc_GO_size(count_time_interval))/test_n;
    
    % NOGO
    components_NOGO = conncomp(digraph(mean_NOGO(:,:,count_time_interval) > network_frequency_threshold),'Type','weak','OutputForm','cell');
    max_size = 0;
    for count_component = 1:length(components_NOGO)
        if length(components_NOGO{count_component}) > max_size
           max_size =  length(components_NOGO{count_component});
        end
    end    
    lcc_NOGO_size(count_time_interval) = max_size;
    
    links_count = sum(sum(mean_NOGO(:,:,count_time_interval) > network_frequency_threshold));
    test_array = NaN(1,test_n);
    for count_test = 1:test_n
        rnd_adj_mat = zeros(network_size);
        rnd_adj_mat(randi([1,network_size^2],1,links_count)) = 1;
        components_test = conncomp(digraph(rnd_adj_mat),'Type','weak','OutputForm','cell');
        max_size = 0;
        for count_component = 1:length(components_test)
            if length(components_test{count_component}) > max_size
                max_size =  length(components_test{count_component});
            end
        end
        test_array(count_test) = max_size;
    end
    lcc_NOGO_pval(count_time_interval) = sum(test_array > lcc_NOGO_size(count_time_interval))/test_n;
    
end


%% plot

% create figures
figure_lcc = figure;
figure_lcc.Name = 'Largest connected component';

plot(timesteps(2:end),lcc_GO_size,timesteps(2:end),lcc_NOGO_size)
xlabel('Time')
ylabel('Size of largest connected component')
ylim([0,length(channel_labels_shared)])
legend('GO','NOGO')

%% overlay p-value color coding

% GO

% overlay markers with color coding for p-values
marker_color = NaN([length(lcc_GO_pval),1]);
marker_color(lcc_GO_pval >= 0.05) = 3;
marker_color(lcc_GO_pval < 0.05) = 1;
hold on
scatter_links = scatter(timesteps(2:end),lcc_GO_size,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])

% NOGO

% overlay markers with color coding for p-values
marker_color = NaN([length(lcc_NOGO_pval),1]);
marker_color(lcc_NOGO_pval >= 0.05) = 3;
marker_color(lcc_NOGO_pval < 0.05) = 1;
hold on
scatter_links = scatter(timesteps(2:end),lcc_NOGO_size,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])