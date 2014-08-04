function analyze_varyHyperparams()

    lalgs = {'soVB', 'moVB', 'VB'};
    dsets = {'MoCap'};
    K = 12;
    omods = {'Gauss'};
    
    gammas = linspace(.05, .99, 8);
    alphas = linspace(.05, .99, 8);
    
    nTask = 8;
    
    fpBase = '../outdir/MoCap/HDPHMM/Gauss';
    
    elbo = []
    
    for lalg = 1 : length(lalgs)
        for omod = 1 : length(omods)
            for gam = 1 : length(gammas)
                for alph = 1 : length(alphas)
                    jobname = sprintf('gam=%f_alpha=%f', gammas(gam), alphas(alph));
                    
                    avg = zeros(1, nTask);
                    for task = 1 : nTask
                       
                       path = sprintf('%s/%s/%s/%d/evidence.txt', fpBase, lalgs{lalg}, jobname, task);
                       %fprintf('%s\n', path);
                       if ~exist(path, 'file')
                          ;
                       else
                          ev = importdata(path);
                          avg(task) = ev(end-1);
                          if(ev(end-1) < -20000000)
                             
                             ev 
                             gammas(gam)
                             alphas(alph)
                             task
                             lalgs{lalg}
                          end
                       end

                    end
%                     if isnan(mean(avg))
%                        jobname
%                        avg
%                     end
%                     if mean(avg) < -20000000
%                        fprintf('alpha=%f gamma = %f\n', alphas(alph), gammas(gam)); 
%                        mean(avg)
%                        avg
%                     end
                    elbo(gam, alph) = mean(avg);
                    
                end
            end
        end
    end
    elbo = max(elbo, -200000);
    [xx, yy] = meshgrid(gammas, alphas);
    surf(xx,yy,elbo);
end