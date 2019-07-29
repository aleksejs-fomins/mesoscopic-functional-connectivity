function [c1,cv1,c2,cv2]= find_clusters_labels(clustering,label,mice)

TEX=2;
ind=[];
c1=[];
cv1=[];

for count_mice = 1:mice
    
    ind = find(label.one(:,count_mice),3);
    if isempty(ind)
        c1(count_mice)=NaN;
        cv1(count_mice)=NaN;
    else
        c1(count_mice)= mean(clustering{1, count_mice}.mean.GO(ind,TEX));
        cv1(count_mice)= mean(clustering{1, count_mice}.var.GO(ind,TEX));
    end
    
end

ind=[];
c2=[];
cv2=[];

for count_mice = 1:mice
    
    ind = find(label.two(:,count_mice),3);
    if isempty(ind)
        c2(count_mice)=NaN;
        cv2(count_mice)=NaN;
    else
        c2(count_mice)= mean(clustering{1, count_mice}.mean.GO(ind,TEX));
        cv2(count_mice)= mean(clustering{1, count_mice}.var.GO(ind,TEX));
    end
    
end

end
