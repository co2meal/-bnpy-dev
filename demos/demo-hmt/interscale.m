%Interscale
load('~/Desktop/generated.mat');
CC = datass_graph;
LL = [20 20; 40 40; 80 80; 160 160];
whichbands = 1;
numlevels=size(LL,1);
whichlevels=diag(ones(1,numlevels-1),1)+diag(ones(1,numlevels-1),-1);
whichlevels(1,:)=0;
whichlevels(:,1)=0;
num_scaling=0;
levelbounds=[0;cumsum(prod(LL,2))];
numnodes=levelbounds(end);
leveldirectionnodes=cell(numlevels,1);
for level=1:numlevels
    leveldirectionnodes{level,1}=levelbounds(level)+1:levelbounds(level+1);
end;

children = children_matrix;
D = [];
numchild=size(children,2);
for level=1:numlevels-1
    if whichlevels(level,level+1)==1 | whichlevels(level+1,level)==1
        indices=[leveldirectionnodes{level,:}];
        for node=1:length(indices)
            data_par=CC(whichbands,indices(node)+num_scaling);
            data_par=data_par(:);
            data_par=repmat(data_par,numchild,1);
            children_ind=children(indices(node),:);
            data_children=CC(whichbands,children_ind+num_scaling);
            data_children=data_children(:);
            if whichlevels(level,level+1)==1
                D=[D;[data_par data_children]];
            end;
            if whichlevels(level+1,level)==1
                D=[D;[data_children data_par]];
            end;
        end;
    end;
end;

save('~/Desktop/gen_scale128.mat', 'D');