function [c1,cv1,c2,cv2]= get_clustering_value(clustering,label,pool_mice)

TEX=2;
ind=[];
c1=[];
cv1=[];
mice = length(pool_mice);

for i = 1:mice
    
    count_mice=pool_mice(i);
    ind = find(label(:,count_mice),3);
    
    if isempty(ind)
        c1(count_mice)=NaN;
        cv1(count_mice)=NaN;
        
        c2(count_mice)=NaN;
        cv2(count_mice)=NaN;
    else
        
        c1(count_mice)= mean(clustering{1, count_mice}.mean.GO(ind,TEX));
        cv1(count_mice)= mean(clustering{1, count_mice}.var.GO(ind,TEX));
            
        c2(count_mice)= mean(clustering{1, count_mice}.mean.NOGO(ind,TEX));
        cv2(count_mice)= mean(clustering{1, count_mice}.var.NOGO(ind,TEX));

    end
    
end


end
