%postProcess.m

%Creates estZ.mat, which has a estZ field that is a cell array that has one
%field per job name.  Each element of estZ.estZ is a (nLap+1) x nObs array,
%with each row being the estimated sequence of z's at that iteration.  The
%final row is the true sequence.

function postProcess()
    
    fpBase = '../outdir';
    trueZPath = '../demodata/mocap6/trueZ.mat';
    
    amods = {'FiniteHMM', 'HDPMM'};
    omods = {'Gauss'};
    lalgs = {'moVB', 'soVB', 'VB'};
    nTask = 15;
    
    jobnames = getJobnames();
    
    for jname = 1 : length(jobnames)
        for task = 1 : nTask

            estZ = [];
            jobname = sprintf('%s/%d', jobnames{jname}, task);
            path = sprintf('%s/%s', fpBase, jobname);
            path

            %Run the python script that comptues the optimal mapping of
            %estimated to true hidden state labels via the munkres algorithm
            pythonCmd = sprintf('python calcTrueParams.py %s/BestAllocModel.mat', path);
            system(pythonCmd);
            munkresOut = load('munkresOut.mat');
            inds = munkresOut.inds;

            list = ls(strcat(path, '/Lap*AllocModel.mat'))
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

    K = [2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24];
    alphas = [.1];
    gammas = [.1];
    lalgs = {'moVB', 'VB'};
    amods = {'HDPHMM', 'FiniteHMM'};
    dsets = {'MoCap'};
    K = [2 4 6 8 9 10 11 12 13 14 15 16 18 20 22 24];
    omods = {'Gauss'};
    

    jobnames = {};
    
    
    %jobnames{1} = 'MoCap/HDPHMM/Gauss/VB/defaultjob';
    
    for lalg = 1 : length(lalgs)
      for omod = 1 : length(omods)
         for amod = 1 : length(amods)
             for k = 1 : length(K)
                 ind = lalg*omod*amod*k;
                 jobnames{ind} = sprintf('%s/%s/%s/%s/K=%d', 'MoCap', ...
                     amods{amod}, omods{omod}, lalgs{lalg}, K(k));
             end
         end
      end
    end
    
    
end






function terms = strsplit(s, delimiter)
%STRSPLIT Splits a string into multiple terms
%
%   terms = strsplit(s)
%       splits the string s into multiple terms that are separated by
%       white spaces (white spaces also include tab and newline).
%
%       The extracted terms are returned in form of a cell array of
%       strings.
%
%   terms = strsplit(s, delimiter)
%       splits the string s into multiple terms that are separated by
%       the specified delimiter. 
%   
%   Remarks
%   -------
%       - Note that the spaces surrounding the delimiter are considered
%         part of the delimiter, and thus removed from the extracted
%         terms.
%
%       - If there are two consecutive non-whitespace delimiters, it is
%         regarded that there is an empty-string term between them.         
%
%   Examples
%   --------
%       % extract the words delimited by white spaces
%       ts = strsplit('I am using MATLAB');
%       ts <- {'I', 'am', 'using', 'MATLAB'}
%
%       % split operands delimited by '+'
%       ts = strsplit('1+2+3+4', '+');
%       ts <- {'1', '2', '3', '4'}
%
%       % It still works if there are spaces surrounding the delimiter
%       ts = strsplit('1 + 2 + 3 + 4', '+');
%       ts <- {'1', '2', '3', '4'}
%
%       % Consecutive delimiters results in empty terms
%       ts = strsplit('C,Java, C++ ,, Python, MATLAB', ',');
%       ts <- {'C', 'Java', 'C++', '', 'Python', 'MATLAB'}
%
%       % When no delimiter is presented, the entire string is considered
%       % as a single term
%       ts = strsplit('YouAndMe');
%       ts <- {'YouAndMe'}
%

%   History
%   -------
%       - Created by Dahua Lin, on Oct 9, 2008
%

%% parse and verify input arguments

assert(ischar(s) && ndims(s) == 2 && size(s,1) <= 1, ...
    'strsplit:invalidarg', ...
    'The first input argument should be a char string.');

if nargin < 2
    by_space = true;
else
    d = delimiter;
    assert(ischar(d) && ndims(d) == 2 && size(d,1) == 1 && ~isempty(d), ...
        'strsplit:invalidarg', ...
        'The delimiter should be a non-empty char string.');
    
    d = strtrim(d);
    by_space = isempty(d);
end


    
%% main

s = strtrim(s);

if by_space
    w = isspace(s);            
    if any(w)
        % decide the positions of terms        
        dw = diff(w);
        sp = [1, find(dw == -1) + 1];     % start positions of terms
        ep = [find(dw == 1), length(s)];  % end positions of terms
        
        % extract the terms        
        nt = numel(sp);
        terms = cell(1, nt);
        for i = 1 : nt
            terms{i} = s(sp(i):ep(i));
        end                
    else
        terms = {s};
    end
    
else    
    p = strfind(s, d);
    if ~isempty(p)        
        % extract the terms        
        nt = numel(p) + 1;
        terms = cell(1, nt);
        sp = 1;
        dl = length(delimiter);
        for i = 1 : nt-1
            terms{i} = strtrim(s(sp:p(i)-1));
            sp = p(i) + dl;
        end         
        terms{nt} = strtrim(s(sp:end));
    else
        terms = {s};
    end        
end
end
