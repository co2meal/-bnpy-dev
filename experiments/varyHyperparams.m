function varyHyperparams() 

    lalgs = {'soVB', 'moVB', 'VB'};
    dsets = {'MoCap'};
    K = 12;
    omods = {'Gauss'};
    
    gammas = linspace(.05, 5, 8);
    alphas = linspace(.05, 5, 8);
    
    nLap = 65;
    nTask = 8;
    nObsBatch = 2;
    nBatch = 3;
    
    for lalg = 1 : length(lalgs)
        for omod = 1 : length(omods)
            for gam = 1 : length(gammas)
                for alph = 1 : length(alphas)
                      cmd = sprintf('python -m bnpy.Run %s %s %s %s --nLap %d --K %d --nObsBatch %d --nBatch %d --nTask %d --jobname gam=%f_alpha=%f --alpha %f --tau %f --gamma %f ',...
                        'MoCap', 'HDPHMM', omods{omod}, lalgs{lalg}, nLap, K, nObsBatch, nBatch, nTask, gammas(gam), alphas(alph), ...
                        alphas(alph), alphas(alph), gammas(gam));
                    system(cmd);
                end
            end
        end
    end
    
end