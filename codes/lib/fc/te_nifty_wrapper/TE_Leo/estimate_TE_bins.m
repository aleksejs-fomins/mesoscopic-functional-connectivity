function [ suggested_TE_bins ] = estimate_TE_bins( data )
% dimension of data tensor are: (trials, timesteps, channels)

temp_channels_n = size(data,3);

suggested_TE_bins = NaN([1,temp_channels_n]);

%figure()
for count_channel = 1:temp_channels_n
   
    %subplot(temp_channels_n,1,count_channel)
    
    temp_std = std(data(:,:,count_channel),0,1);
    temp_range_spanned = max(data(:,:,count_channel),[],1)-min(data(:,:,count_channel),[],1);
    temp_bins_number_series = temp_range_spanned./(3*temp_std);
    
    suggested_TE_bins(count_channel) = ceil(max(temp_bins_number_series));
    
    %plot(samples_one_second,temp_bins_number_series)
    
end

end

