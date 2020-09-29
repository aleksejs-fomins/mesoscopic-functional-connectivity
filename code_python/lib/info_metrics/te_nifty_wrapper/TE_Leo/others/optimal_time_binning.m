%% add NIfTy toolbox
restoredefaultpath;
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\NIfTy_Ver1\');

%% define constants

path_mtp_parent = 'E:'; %without separator

samples_CUE = 21:30;    % 1-1.5 seconds trial time
samples_TEX = 41:70;    % 2-3.5 seconds trial time
samples_LIK = 71:120;   % 3.5-6 seconds trial time
samples_ALL = 1:200;    % 0-10  seconds


%% set parameters for comparison

% define binning options to compare
bin_comparison.temp_binnings = [10,20,50,100,200];

% define p-value threshold (only TE values below threshold will be
% used for the average TE)
bin_comparison.p_threshold = 0.01;


%% set parameters for TE estimation

save_results = false;

parameters.sample_rate = 20;

parameters.mouse_id = '7';
parameters.session_id = '2016_03_18_a';

parameters.samples_window = 'ALL';
parameters.trials_type = 'GO';

parameters.channels = 1:2;
% compute all channel pairs
parameters.channel_pairs = [nchoosek(parameters.channels,2);nchoosek(flip(parameters.channels),2)];

parameters.channels_total = 12;
parameters.channel_labels = {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12'};    %replace with names of brain areas

delay_min = 0;
delay_max = 5;
delay_stepsize = 1;
parameters.receiver_embedding_tau = 1;

% load data and behaviour files
path_folder_id = strcat('mtp_',parameters.mouse_id,'_',parameters.session_id);
path_data_file = fullfile(path_mtp_parent,path_folder_id,'data.mat');
path_behaviorvar_file = fullfile(path_mtp_parent,path_folder_id,'behaviorvar.mat');
load(path_data_file)
load(path_behaviorvar_file,'iGO','iNOGO','iMISS','iFA')

switch parameters.samples_window
    case 'CUE'
        parameters.samples_timesteps = samples_CUE;
    case 'TEX'
        parameters.samples_timesteps = samples_TEX;
    case 'LIK'
        parameters.samples_timesteps = samples_LIK;
    case 'ALL'
        parameters.samples_timesteps = samples_ALL;
    otherwise
        fprintf('\n')
        error('ERROR: Unknown samples window. Valid options: CUE, TEX, LIK, ALL')
end

switch parameters.trials_type
    case 'GO'
        parameters.trials_numbers = iGO;
    case 'NOGO'
        parameters.trials_numbers = iNOGO;
    case 'MISS'
        parameters.trials_numbers = iMISS;
    case 'FA'
        parameters.trials_numbers = iFA;
    case 'ALL'
        parameters.trials_numbers = cat(2,iGO,iNOGO,iMISS,iFA);
    otherwise
        fprintf('\n')
        error('ERROR: Unknown trials type. Valid options: GO, NOGO, MISS, FA, ALL')
end


%% perform comparison


bin_comparison.temp_subsampling_windows = NaN([1, length(bin_comparison.temp_binnings)]);
bin_comparison.TE_total_average = NaN([1, length(bin_comparison.temp_binnings)]);
bin_comparison.t = NaN([1, length(bin_comparison.temp_binnings)]);
bin_comparison.MI_total_average = NaN([1, length(bin_comparison.temp_binnings)]);
bin_comparison.storage_total_average = NaN([1, length(bin_comparison.temp_binnings)]);

temp_count_outcome = 1;

for temp_subsampling_windows = bin_comparison.temp_binnings
    
    parameters.subsampling_windows_lenghts = (200/temp_subsampling_windows)*ones(1,temp_subsampling_windows);
    
    windows_n = length(parameters.subsampling_windows_lenghts);
    
    % check that the sum of the subsampling windows lengths is equal to the
    % entire data length
    if sum(parameters.subsampling_windows_lenghts) ~= length(parameters.samples_timesteps)
        error('ERROR: invalid choice of subsampling windows')
    end
    
    % record starting time
    t_elapsed = tic;
    
    my_data = NaN([length(parameters.trials_numbers),windows_n,parameters.channels_total]);
    parameters.plot_samples_timesteps = NaN([1,windows_n]);
    
    window_start = 1;
    for i = 1:windows_n
        window_end = window_start + parameters.subsampling_windows_lenghts(i) - 1;
        my_data(:,i,:) = mean(data(parameters.trials_numbers,window_start:window_end,:),2);
        parameters.plot_samples_timesteps(i) = (parameters.samples_timesteps(1) + window_start - 1) / parameters.sample_rate;
        window_start = window_end + 1;
    end
    
    parameters.states_method = 'UniCB';
    parameters.states_bins = estimate_TE_bins(my_data);
    
    
    % perform TE
    
    % initialize TE table and p-value table
    temp_results.TE_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
    temp_results.p_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
    temp_results.delay_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
    temp_results.MI_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
    temp_results.storage_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
    
    % swap tensor dimensions in order to suit the requirements of the NIfTy toolbox
    my_data = permute(my_data,[3 2 1]);
    
    TE_max_total_average = 0;
    window_length_optimal = NaN;
    for i = 1:length(parameters.channel_pairs)
        
        % select data for the current channel pair
        DataRaster = my_data(parameters.channel_pairs(i,:),:,:);
        
        % state the data using the method and bins specified in the parameters
        MethodAssign = {1,1,parameters.states_method,{parameters.states_bins(parameters.channel_pairs(i,1))};1,2,parameters.states_method,{parameters.states_bins(parameters.channel_pairs(i,2))}};
        StatesRaster = data2states(DataRaster, MethodAssign);
        
        % loop over different values of delay u and choose the one the
        % maximizes the average TE
        
        nT = size(StatesRaster,2);
        
        TE_optimal = NaN([nT,1]);
        p_optimal = NaN([nT,1]);
        MI_optimal = NaN([nT,1]);
        storage_optimal = NaN([nT,1]);
        
        TE_max_mean = 0;
        delay_optimal = delay_min;
        for temp_delay = delay_min:delay_stepsize:delay_max
            
            % perform the information calculation for all time bins
            temp_TE = NaN([nT,1]);
            temp_p = NaN([nT,1]);
            temp_MI = NaN([nT,1]);
            temp_storage = NaN([nT,1]);
            for iT = (max(temp_delay,parameters.receiver_embedding_tau) + 2):nT
                VariableIDs = {1,2,iT;... % Receiving variable in the future
                    1,2,iT - parameters.receiver_embedding_tau;... % Receiving variable in the past
                    1,1,iT - 1 - temp_delay}; % Transmitting variable in the past
                [temp_TE(iT),temp_p(iT)] = instinfo(StatesRaster, 'TE', VariableIDs, 'MCOpt', 'on');
                temp_MI(iT) = instinfo(StatesRaster, 'PairMI', {1,2,iT;1,1,iT - 1 - temp_delay});
                temp_storage = instinfo(StatesRaster, 'PairMI', {1,2,iT;1,2,iT - parameters.receiver_embedding_tau});
            end
            
            %TE_mean = mean(temp_TE(~isnan(temp_TE)));
            TE_mean = mean(temp_TE(temp_p < bin_comparison.p_threshold));
            if TE_mean > TE_max_mean
                delay_optimal = temp_delay;
                TE_optimal = temp_TE;
                p_optimal = temp_p;
                MI_optimal = temp_MI;
                storage_optimal = temp_storage;
                
                TE_max_mean = TE_mean;
            end
            
        end
        
        % add TE and p-value to the temp_results
        temp_results.TE_table{parameters.channel_pairs(i,1),parameters.channel_pairs(i,2)}{1,1} = TE_optimal';
        temp_results.p_table{parameters.channel_pairs(i,1),parameters.channel_pairs(i,2)}{1,1} = p_optimal';
        temp_results.delay_table{parameters.channel_pairs(i,1),parameters.channel_pairs(i,2)}{1,1} = delay_optimal;
        temp_results.MI_table{parameters.channel_pairs(i,1),parameters.channel_pairs(i,2)}{1,1} = MI_optimal;
        temp_results.storage_table{parameters.channel_pairs(i,1),parameters.channel_pairs(i,2)}{1,1} = storage_optimal;
        
        fprintf('.')
    end
    
    % compute elapsed time
    t = toc(t_elapsed);
    
    % compute total average
    temp_TE_sum = 0;
    temp_TE_count = 0;
    temp_MI_sum = 0;
    temp_MI_count = 0;
    temp_storage_sum = 0;
    temp_storage_count = 0;
    
    for temp_pair = parameters.channel_pairs'
        temp_TE_series = temp_results.TE_table{temp_pair(1),temp_pair(2)}{1}(temp_results.p_table{temp_pair(1),temp_pair(2)}{1} < bin_comparison.p_threshold);
        temp_MI_series = temp_results.MI_table{temp_pair(1),temp_pair(2)}{1};
        temp_storage_series = temp_results.storage_table{temp_pair(1),temp_pair(2)}{1};
        % remove NaN values
        temp_TE_series = temp_TE_series(~isnan(temp_TE_series));
        temp_MI_series = temp_MI_series(~isnan(temp_MI_series));
        temp_storage_series = temp_storage_series(~isnan(temp_storage_series));
        % update sum and length
        temp_TE_sum = temp_TE_sum + sum(temp_TE_series);
        temp_TE_count = temp_TE_count + length(temp_TE_series);
        temp_MI_sum = temp_MI_sum + sum(temp_MI_series);
        temp_MI_count = temp_MI_count + length(temp_MI_series);
        temp_storage_sum = temp_storage_sum + sum(temp_storage_series);
        temp_storage_count = temp_storage_count + length(temp_storage_series);
    end
    
       
    % return average TE
    TE_total_average = temp_TE_sum/temp_TE_count;
    MI_total_average = temp_MI_sum/temp_MI_count;
    storage_total_average = temp_storage_sum/temp_storage_count;
    if TE_total_average > TE_max_total_average
        window_length_optimal = temp_subsampling_windows;
        TE_max_total_average = TE_total_average;
    end
    
    bin_comparison.temp_subsampling_windows(temp_count_outcome) = temp_subsampling_windows;
    bin_comparison.TE_total_average(temp_count_outcome) = TE_total_average;
    bin_comparison.t(temp_count_outcome) = t;
    bin_comparison.MI_total_average(temp_count_outcome) = MI_total_average;
    bin_comparison.storage_total_average(temp_count_outcome) = storage_total_average;
    
    temp_count_outcome = temp_count_outcome + 1;
    
end


%% save

if save_results
    save_results_path = fullfile(path_mtp_parent,path_folder_id,'Leonardo')
    if ~exist(save_results_path)
        mkdir(save_results_path)
    end
    save(fullfile(save_results_path,'bin_comparison.mat'),'bin_comparison');
end

%% plot

figure()
%subplot(2,1,1)
plot(bin_comparison.temp_subsampling_windows,bin_comparison.TE_total_average,'--o',bin_comparison.temp_subsampling_windows,bin_comparison.MI_total_average,'--o',bin_comparison.temp_subsampling_windows,bin_comparison.storage_total_average,'--o')
legend('TE(source, target)','MI(source,target)','Information Storage(target)')
% subplot(2,1,2)
% plot(bin_comparison.temp_subsampling_windows,bin_comparison.t)

