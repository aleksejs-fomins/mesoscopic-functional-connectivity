% --- get channel adjacency matrix

function [adj_Hit, adj_CR] = get_adj_channel(network_adj_matrices,pool_mice,label,links)

TEX=2;
ind=[];

adj_Hit=[];
adj_CR=[];
    
mice = length(pool_mice);

% --- count across mice ---
for i = 1:mice
    
    count_mice=pool_mice(i);
    ind = find(label(:,count_mice),1);
    
    if isempty(ind)
        
        adj_Hit= [adj_Hit; NaN(1,12)];
        adj_CR= [adj_CR; NaN(1,12)];
        
    else
        
        % --- count across sessions ---
        sessions = length(network_adj_matrices{1, count_mice}{1, 1}{1, TEX});
        
        for count_session=1:sessions
            
            tmp_1=double(network_adj_matrices{1, count_mice}{1, 1}{1, TEX}{1,count_session});
            tmp_2=double(network_adj_matrices{1, count_mice}{1, 2}{1, TEX}{1,count_session});
            
            if strcmp(links,'in')

            adj_Hit=[adj_Hit; tmp_1(ind,:)];            
            adj_CR= [adj_CR; tmp_2(ind,:)];
            
            elseif strcmp(links,'out')
                
            adj_Hit=[adj_Hit; tmp_1(:,ind)'];            
            adj_CR= [adj_CR; tmp_2(:,ind)'];
            
            end

        end
        
        % introduce cell to add across mice
%         adj_Hit{1,i}= [adj_Hit; NaN(1,12)];
%         adj_CR{1,i}= [adj_CR; NaN(1,12)];
    
    end


end

end