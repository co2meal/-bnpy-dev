%postProcess.m

%Creates estZ.mat, which has a estZ field that is a cell array that has one
%field per job name.  Each element of estZ.estZ is a (nLap+1) x nObs array,
%with each row being the estimated sequence of z's at that iteration.  The
%final row is the true sequence.

function postProcess()
    
    fpBase = '/home/will/bnpy/outdir/MoCap/HDPHMM/Gauss/VB/';
    trueZPath = '/home/will/bnpy/bnpy-dev/demodata/mocap6/trueZ.mat';
    
    amods = {'FiniteHMM', 'HDPMM'};
    omods = {'Gauss'};
    lalgs = {'moVB', 'soVB', 'VB'};
    nTask = 20;
    
    jobnames = getJobnames();
    
    for jname = 1 : length(jobnames)
        for task = 1 : nTask

            estZ = [];
            jobname = sprintf('%s/%d', jobnames{jname}, task);
            path = sprintf('%s/%s', fpBase, jobname);

            %Run the python script that comptues the optimal mapping of
            %estimated to true hidden state labels via the munkres algorithm
            pythonCmd = sprintf('python calcTrueParams.py %s/BestAllocModel.mat', path);
            system(pythonCmd);
            munkresOut = load('munkresOut.mat');
            inds = munkresOut.inds;

            list = ls(strcat(path, '/Lap*AllocModel.mat'));
            list = strsplit(list, '\n');
            for lap = 1 : length(list) - 1
               est = load((list{lap}));
               est = est.estZ;

               %Concatenate the multiple sequences of est into one big array
               tmp = est;
               est = [];
               for seq = 1 : length(tmp)
                  est = [est tmp{seq}];
               end
               est = est + 1; 

               %Permute est according to the inds found by munkres
               for i = 1 : length(est)
                   est(i) = inds(est(i), 2) + 1;
               end
               
                estZ(end+1, :) = est;

            end

            trueZ = load(trueZPath);
            trueZ = trueZ.trueZ;
            estZ(end+1, :) = trueZ;
            save(sprintf('%s/estZ.mat', path), 'estZ');
            imagesc(estZ);


        end
    end

end


function [jobnames] = getJobnames()

    K = [2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28 30];
    alphas = [.1];
    gammas = [.1];
    

    jobnames = {};
    
    jobnames{1} = 'defaultjob';
    
    
end