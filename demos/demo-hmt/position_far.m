% Position Far
load('~/Desktop/generated.mat');
whichbands=1;
distance=16;
numlevels = 4;
whichlevels=numlevels-1;
structel = [0 distance];
D=[];
CC = datass_graph;
LL = [20 20; 40 40; 80 80; 160 160];
num_scaling=0;
indcount=0;

for level=1:numlevels
    bandsize=LL(level,:);
    if any(level==whichlevels)g
        inds=[];
        for i=1:bandsize(1)
            for j=1:bandsize(2)
                testpos=repmat([i j],size(structel,1),1);
                testpos=testpos+structel;
                removepos=find(testpos(:,1)<1 | testpos(:,1)>bandsize(1)  |  testpos(:,2)<1 | testpos(:,2)>bandsize(2)) ;
                testpos(removepos,:)=[];
                if ~isempty(testpos)
                    testinds=sub2ind(bandsize,testpos(:,1),testpos(:,2));
                    testinds=[ones(length(testinds),1)*sub2ind(bandsize,i,j) testinds(:)];
                    for k=1:size(testinds,1)
                        if isempty(inds)
                            inds=testinds(k,:);
                        else
                            if ~any((inds(:,1)==testinds(k,1)&inds(:,2)==testinds(k,2)) | (inds(:,1)==testinds(k,2)&inds(:,2)==testinds(k,1)))
                                inds=[inds;testinds(k,:)];
                            end;
                        end;
                    end;
                end;
            end;
        end;
        if ~isempty(inds)
            A=CC(whichbands,inds(:,1)+num_scaling+indcount);
            A=A(:);
            B=CC(whichbands,inds(:,2)+num_scaling+indcount);
            B=B(:);
            D=[D;[A B; B A]];
        end
    end;
    indcount=indcount+prod(bandsize);
end;

save('~/Desktop/gen_far128.mat', 'D');