cd 'E:\data\texture_discrimination\mtp_12\mtp_12_TE'

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

% time intervals (in seconds)
timesteps = (0:0.2:10);
network_time_intervals = mat2cell([timesteps(1:end-1)',timesteps(2:end)'],ones([1,50]),[2]);

lcc_GO_size = NaN([100,mice,length(network_time_intervals)]);
lcc_NOGO_size = NaN([100,mice,length(network_time_intervals)]);

network_outdegree_GO = cell(1,mice);
network_outdegree_NOGO = cell(1,mice);

network_indegree_GO = cell(1,mice);
network_indegree_NOGO = cell(1,mice);

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
                    
                    % compute LCC size (frequent links only)
                    temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                    temp_graph_frequent = digraph(temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice}) > network_frequency_threshold);
                    components = conncomp(temp_graph_frequent,'Type','weak','OutputForm','cell');
                    max_size = 0;
                    for count_component = 1:length(components)
                        if length(components{count_component}) > max_size
                            max_size =  length(components{count_component});
                        end
                    end
                    
                    if strcmp(trials_type{1,1},'GO')
                        lcc_GO_size(count_sessions,count_mice,count_time_interval) = max_size;
                        
                        network_outdegree_GO{1,count_mice}{1,count_sessions}{1,count_time_interval} = squeeze(sum(temp_matrix>0,2));
                        network_indegree_GO{1,count_mice}{1,count_sessions}{1,count_time_interval} = squeeze(sum(temp_matrix>0,1))';

                    elseif strcmp(trials_type{1,1},'NOGO')
                        lcc_NOGO_size(count_sessions,count_mice,count_time_interval) = max_size;
                        
                        network_outdegree_NOGO{1,count_mice}{1,count_sessions}{1,count_time_interval} = squeeze(sum(temp_matrix>0,2));
                        network_indegree_NOGO{1,count_mice}{1,count_sessions}{1,count_time_interval} = squeeze(sum(temp_matrix>0,1))';

                    end
                                        
                end
                
            end
        end
        
    end
    
end

%% average across sessions and mice, excluding the NaN values
% ------- network_ out/in _degree is taken as -------
% ------- {1:end,1:end,count_time_interval} not as -------
% ------- {1,count_mice}{1,count_sessions}{1,count_time_interval}-------
mean_outdegree_GO=[];
mean_outdegree_NOGO=[];

std_outdegree_GO=[];
std_outdegree_NOGO=[];

mean_indegree_GO=[];
mean_indegree_NOGO=[];

std_indegree_GO=[];
std_indegree_NOGO=[];

for count_time_interval = 1:length(network_time_intervals)

mean_outdegree_GO = [mean_outdegree_GO mean([network_outdegree_GO{1:end,1:end,count_time_interval}],2,'omitnan')];
mean_outdegree_NOGO = [mean_outdegree_NOGO mean([network_outdegree_NOGO{1:end,1:end,count_time_interval}],2,'omitnan')];

std_outdegree_GO = [std_outdegree_GO std([network_outdegree_GO{1:end,1:end,count_time_interval}],[],2,'omitnan')];
std_outdegree_NOGO = [std_outdegree_NOGO std([network_outdegree_NOGO{1:end,1:end,count_time_interval}],[],2,'omitnan')];

mean_indegree_GO = [mean_indegree_GO mean([network_indegree_GO{1:end,1:end,count_time_interval}],2,'omitnan')];
mean_indegree_NOGO = [mean_indegree_NOGO mean([network_indegree_NOGO{1:end,1:end,count_time_interval}],2,'omitnan')];

std_indegree_GO = [std_indegree_GO std([network_indegree_GO{1:end,1:end,count_time_interval}],[],2,'omitnan')];
std_indegree_NOGO = [std_indegree_NOGO std([network_indegree_NOGO{1:end,1:end,count_time_interval}],[],2,'omitnan')];

end

%%
ch1=2;
ch2=10;
t=0.2:0.2:10;

subplot(4,1,1)
plot(t,mean_outdegree_GO(ch1,:),t,mean_outdegree_GO(ch2,:))
title('Hit')

subplot(4,1,2)
plot(t,mean_outdegree_NOGO(ch1,:),t,mean_outdegree_NOGO(ch2,:))
title('CR')

subplot(4,1,3)
plot(t,mean_indegree_GO(ch1,:),t,mean_indegree_GO(ch2,:))
title('Hit')

subplot(4,1,4)
plot(t,mean_indegree_NOGO(ch1,:),t,mean_indegree_NOGO(ch2,:))
title('CR')
%%
h1=figure
 shadedErrorBar(t,mean_outdegree_GO(ch1,:),std_outdegree_GO(ch1,:),'b')
h2=figure
 shadedErrorBar(t,mean_outdegree_GO(ch2,:),std_outdegree_GO(ch2,:),'r')

%% average across sessions and mice, excluding the NaN values

mean_GO = squeeze(mean(mean(lcc_GO_size,1,'omitnan'),2,'omitnan'));
mean_NOGO = squeeze(mean(mean(lcc_NOGO_size,1,'omitnan'),2,'omitnan'));

std_GO = squeeze(std(std(lcc_GO_size,0,1,'omitnan'),0,2,'omitnan'));
std_NOGO = squeeze(std(std(lcc_NOGO_size,0,1,'omitnan'),0,2,'omitnan'));


%% plot shaded error bar

a.FaceAlpha = 0.5;

dt = timesteps(2:end)';
fill([dt;flipud(dt)],[mean_GO-std_GO; flipud(mean_GO + std_GO)],[.0 .0 .8],'FaceAlpha','0.3','linestyle','none');
line(timesteps(2:end),mean_GO,'LineWidth',2,'Color',[0 0 1])
xlabel('Time')
ylabel('Size of largest connected component GO trials')
ylim([0,length(channel_labels_shared)])
hold on

fill([dt;flipud(dt)],[mean_NOGO-std_NOGO; flipud(mean_NOGO + std_NOGO)],[.8 .0 .0],'FaceAlpha','0.3','linestyle','none');
line(timesteps(2:end),mean_NOGO,'LineWidth',2,'Color',[1 0 0])
xlabel('Time')
ylabel('Size of largest connected component')
ylim([0,length(channel_labels_shared)])
legend('GO','NOGO')
hold off


%% plot

figure_lcc = figure;
figure_lcc.Name = 'Largest connected component';

plot(timesteps(2:end),mean_GO,timesteps(2:end),mean_NOGO)
xlabel('Time')
ylabel('Size of largest connected component')
ylim([0,length(channel_labels_shared)])
legend('GO','NOGO')
