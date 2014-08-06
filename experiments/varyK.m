function varyK()

    %lalgs = {'soVB', 'moVB', 'VB'};
    lalgs = {'soVB'};
    amods = {'FiniteHMM'};
    dsets = {'MoCap'};
    K = [2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28];
    omods = {'Gauss'};
    
    nLap = 70;
    nTask = 30;
    nObsBatch = 2;
    nBatch = 3;
    
    for lalg = 1 : length(lalgs)
        for omod = 1 : length(omods)
            for amod = 1 : length(amods)
                for k = 1 : length(K)
                    cmd = sprintf('python -m bnpy.Run %s %s %s %s --nLap %d --K %d --nObsBatch %d --nBatch %d --nTask %d --jobname K=%d',...
                        'MoCap', amods{amod}, omods{omod}, lalgs{lalg}, nLap, K(k), nObsBatch, nBatch, nTask, K(k));
                    system(cmd);
                end
            end
        end
    end
    

end