function [ results ] = TE_NIfTy( my_data , parameters )
%TE_NIFTY_LEO Summary of this function goes here
%   Detailed explanation goes here

% MCpThresh is a parameter of the NIfTy toolox. If the algorithm finds that the p-value will be
% above the threshold, the calculation ceases. This speeds up the computation of p-values, but the
% p-values above the threshold will not be computed acurately.
MCpThresh = 0.01;

% append simulation parameters to the results
results.parameters = parameters;

% append data to results
results.data = my_data;

% initialize TE table and p-value table
results.TE_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
results.p_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
results.delay_table = cell2table(cell(parameters.channels_total), 'VariableNames', parameters.channel_labels, 'RowNames', parameters.channel_labels );
results.entropy = cell2table(cell([1,parameters.channels_total]), 'VariableNames', parameters.channel_labels);
results.states_bins_edges = cell2table(cell([1,parameters.channels_total]), 'VariableNames', parameters.channel_labels);

% prepare data for analysis

% swap tensor dimensions in order to suit the requirements of the NIfTy toolbox
DataRaster = permute(my_data,[3 2 1]);

for channel = parameters.channels
    
    % convert continuous data into discrete states, using the method and bin numbers specified in the parameters
    stating_method = parameters.states_method;
    bin_number = parameters.states_bins_count(channel);
    MethodAssign = {1,1,stating_method,{bin_number}};
    [StatesRaster(channel,:,:),MethodResults] = data2states(DataRaster(channel,:,:), MethodAssign);
    
    % append bins edges used for defining the states of the channel to results
    results.states_bins_edges{1,channel} = {MethodResults{1}};
    
end

% compute TE for all channel pairs
Method = 'TE';
nT = size(StatesRaster,2);
for i = 1:length(parameters.channel_pairs)
    
    channel1 = parameters.channel_pairs(i,1);
    channel2 = parameters.channel_pairs(i,2);
    
    % loop over different values of delay u and choose the one that maximizes the average TE
    TE_optimal = NaN([nT,1]);
    p_optimal = NaN([nT,1]);
    TE_max_mean = 0;
    delay_optimal = parameters.delay_min;
    
    for temp_delay = parameters.delay_min:parameters.delay_stepsize:parameters.delay_max
             
        % perform the TE calculation for all time bins
        temp_TE = NaN([nT,1]);
        temp_p = NaN([nT,1]);
        for iT = (max(temp_delay,parameters.receiver_embedding_tau) + 1):nT
            VariableIDs = {1,2,iT;... % Receiving variable in the future
                1,2,iT - parameters.receiver_embedding_tau;... % Receiving variable in the past
                1,1,iT - temp_delay}; % Transmitting variable in the past
            [temp_TE(iT),temp_p(iT)] = instinfo(StatesRaster([channel1,channel2],:,:), Method, VariableIDs, 'MCOpt', 'on', 'MCpThresh', MCpThresh);
        end
        
        % check if current TE is the optimal so far
        TE_mean = sum(temp_p < MCpThresh); % number of TE values with p-values below threshold
        %TE_mean = mean(temp_TE(temp_p < MCpThresh)); % mean computed only over TE values with p-values below threshold
        %TE_mean = mean(temp_TE(~isnan(temp_TE))); % mean computed over all TE values
        if TE_mean > TE_max_mean || temp_delay == parameters.delay_min
            delay_optimal = temp_delay;
            TE_optimal = temp_TE;
            p_optimal = temp_p;
            TE_max_mean = TE_mean;
        end
        
    end
    
    % append optimal values to the results
    results.TE_table{channel1,channel2}{1,1} = TE_optimal';
    results.p_table{channel1,channel2}{1,1} = p_optimal';
    results.delay_table{channel1,channel2}{1,1} = delay_optimal;
    
    % display progress
    %fprintf('%.0f%%,',i*100/size(parameters.channel_pairs,1))
end

% additional: compute Entropy of single channels
Method = 'Ent';
nT = size(StatesRaster,2);
for channel = parameters.channels
    % Compute channel entropy for all time bins
    temp_entropy = NaN([nT,1]);
    for iT = 1:nT
        VariableIDs = {1,1,iT};
        temp_entropy(iT) = instinfo(StatesRaster(channel,:,:), Method, VariableIDs);
    end
    % append entropy to results
    results.entropy{1,channel} = {temp_entropy};
end

end

