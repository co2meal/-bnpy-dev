function analyze_varyK()
    
    readInData();
    
    elbo = load('varyKData.mat');
   
        
end


function readInData()

    lalgs = {'moVB', 'VB', 'soVB'};
    amods = {'HDPHMM', 'FiniteHMM'};
    dsets = {'MoCap'};
    K = [2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24];
    omods = {'Gauss'};
    
    nLap = 80;
    nTask = 15;
    
    fpBase = '../outdir/MoCap';
    elbo = zeros(length(K), nTask/15);
    
   for lalg = 1 : length(lalgs)
      for omod = 1 : length(omods)
          for amod = 1 : length(amods)
              if strcmp('soVB', lalgs{lalg})
                 nTask = 30;
                 K =[2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28];
              end
              for k = 1 : length(K)
                  jobname = sprintf('K=%d', K(k));
                  
                  for task = 1 : nTask
                      if ~strcmp('soVB', lalgs{lalg}) && (K(k) > 24 || task > 15)
                        break;
                      end
%                       task
%                       K(k)
%                       lalgs{lalg}
                      path = sprintf('%s/%s/%s/%s/%s/%d/evidence.txt', fpBase, ...
                          amods{amod}, omods{omod}, lalgs{lalg}, jobname, task);
                      ev = load(path);
                      elbo(k, task) = max(ev(3:end));
                      if max(ev(3:end)) > -3000
                         lalgs{lalg}
                         amods{amod}
                         K(k)
                         task
                      end
                    
                  end
              end
              
              figure;
              scatter(repmat(K, 1, nTask), reshape(elbo, 1, length(K) * nTask));
              
              title(sprintf('K vs Final ELBO value for %s-%s with %s Likelihoods on the %s Dataset', ...
                  lalgs{lalg}, amods{amod}, omods{omod}, 'MoCap'));
              xlabel('K');
              ylabel('Final ELBO Value');
              %axis([K(1) K(end) min(min(elbo)) max(max(elbo))]);
              
              p = polyfit(repmat(K, 1, nTask), reshape(elbo, 1, length(K) * nTask), 2);
              hold on; plot(K, polyval(p, K)); hold off;
          end
      end
   end
   
   save('varyKData.mat', 'elbo');
end