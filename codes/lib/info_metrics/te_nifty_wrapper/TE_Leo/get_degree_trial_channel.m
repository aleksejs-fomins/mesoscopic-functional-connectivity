% --- get channel adjacency matrix

function degree = correlate_degree_trial_channel(network_degree,pool_mice,label,links)

ind=[];
degree=[];
    
mice = length(pool_mice);

% --- count across mice ---
for i = 1:mice
    
    count_mice=pool_mice(i);
    ind = find(label(:,count_mice),1);
    
    if isempty(ind)
        
        % do nothing
        % degree= [degree; NaN(1,50)];
        
    else
        
        % --- count across sessions ---
        [~,sessions] = size(network_degree{1,count_mice});
        
        for count_session=1:sessions

                if strcmp(links,'in')

                % create a degree vector for all time intervals during a
                % trial- gets 12 by 50 matrix (nodes by time intervals)
                tmp=cell2mat(network_degree{1,count_mice}{1,count_session});

                degree=[degree; tmp(ind,:)];

                elseif strcmp(links,'out')

                tmp=cell2mat(network_degree{1,count_mice}{1,count_session});

                degree=[degree; tmp(ind,:)];
                
                end

            end
    end

% end count mice
end

end