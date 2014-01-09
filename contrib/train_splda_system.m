function [system,obj] = train_splda_system(nvoices, S, F, N, mu, niters, opts)
% Builds an SPLDA verifier, by using the given SPLDA model. This verifier
% scores all train inputs against all test inputs, to give a matrix of
% llr scores.
%
% Inputs:
%  nvoices: size of speaker subspace
%  S: dim-by-dim scatter matrix: sum_t (data(:,t)-mu)*(data(:,t)-mu)', where t
%     ranges over all the data of all the speakers.
%  F: dim-by-numspkrs, first order stats, i.e. sum of data vectors for every speaker
%     F(:,i) = sum_t data(:,t) - mu, where t ranges over data for speaker i 
%  N: 1-by-ns, zero-order stat, i.e. number of vectors per speaker
%  mu: dim-by-1, the global mean (over all speakers)
%   
% Output:
%   system: a struct with two function handles:
%
%   To produce a matrix of scores do:
%
%   train_stats = system.prepStats(Ntrain,Ftrain,ACCUMtrain)
%   test_stats = system.prepStats(Ntest,Ftest,ACCUMtest)
%   scores = system.score(train_stats,test_stats)
%
%   Here:
%     Ntrain, Ftrain are zero and first-order stats for m training inputs:
%
%     Ntrain: 1-by-M zero order stats, one for each train input. If every
%             train input is represented by just one training input vector, 
%             then this is just a row of ones.
%
%     Ftrain: dim-by-M first order stats, one for each train input.
%             Ftrain(:,i) is the sum over all training vectors for training 
%             input i. Note the mean should not be removed from F. If every 
%             train input is represented by just one training input vector, 
%             then this Ftrain is just the original training data.
%
%     Ntest and Ftest is the same for N test inputs.
% 
%     ACCUMtrain: can be [], in which case the identity matrix is assumed.
%                 An M-by-P matrix of 0's or 1', to accumulate training
%                 stats to form P =< M speaker 'models'. The 1's in column j
%                 indicate the stats to be accumulated for model j. 
%     
%     ACCUMtest:  can be [], in which case the identity matrix is assumed.
%                 An N-by-Q matrix of 0's or 1', to accumulate test
%                 stats to form Q =< N speaker 'models'. The 1's in column j
%                 indicate the stats to be accumulated for model j. 
%
%     scores: P-by-Q matrix of llr scores     
%
%  Notes: The 'ACCUM' matrices are for convenience only. The same effect may be
%         obtained by just summing the original N and F input stats.
%
%         The score function is symmetric in the inputs. This means that we 
%         can do 8conv-1conv type scoring, but also 1conv-8conv, or even 
%         8conv-8conv. This is why the interface also allows test inputs to 
%         be accumulated into test 'models'.
%
%         If you are doing 1conv-1conv, then N will be all ones, F will just 
%         be the train or test data and ACCUM = []. 
%
%         The F inputs to train_splda_system() must have the data mean (mu) 
%         removed. The F inputs to the resultant system.prepStats() should 
%         NOT have the data mean removed.
%
%         A technical note: There really are no explicit speaker model 
%         point-estimates. This system integrates over all possible speaker 
%         models. Statistics are stored, rather than models, but we call them
%         'models' above to give the look and feel of a more traditional 
%         system based on speaker model point-estimates.
%

if ~exist('niters','var')
    niters = 100;
end
if ~exist('options','var')
    opts = [];
end

[model,obj] = train_splda(nvoices, S, F, N, mu, niters, opts);
system = prepSPLDAsystem(model);



