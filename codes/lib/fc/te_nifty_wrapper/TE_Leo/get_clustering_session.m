function [c_Hit,c_CR]= get_clustering_session(clustering_Hit,clustering_CR,label,pool_mice)

TEX=2;
ind=[];

mice = length(pool_mice);
c_Hit=cell(1,mice);
c_CR=cell(1,mice);

for i = 1:mice
    
    count_mice=pool_mice(i);
    ind = find(label(:,count_mice),3);
    
    if isempty(ind)
        c_Hit{1, i}=NaN;        
        c_CR{1, i}=NaN;
    else
        
        c_Hit{1, i}= clustering_Hit{1, count_mice}{1,TEX}(ind,:);            
        c_CR{1, i}= clustering_CR{1, count_mice}{1,TEX}(ind,:);

    end
    
end


end
