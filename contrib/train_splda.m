function [model,obj] = train_splda(nvoices, S, F, N, mu, niters, opts)
%
%Inputs: 
%  nvoices: size of speaker subspace
%  S: dim-by-dim scatter matrix: sum_t (data(:,t)-mu)*(data(:,t)-mu)', where t
%     ranges over all the data of all the speakers.
%  F: dim-by-numspkrs, first order stats, i.e. sum of data vectors for every speaker
%     F(:,i) = sum_t data(:,t) - mu, where t ranges over data for speaker i 
%  N: 1-by-ns, zero-order stat, i.e. number of vectors per speaker
%  mu: dim-by-1, the mean of the data over all speakers
%  niters: number of EM iterations to do
%  opts: options for EM algorithm, omit for default, see em_splda for details

model.mu = mu;
model.D = inv(S/sum(N));
if ~exist('opts','var')
    opts = [];
end
dim = length(mu);
model.V = randn(dim,nvoices);

obj = zeros(1,niters);
for i=1:niters
    [model,obj(i)] = em_splda(model, S, F, N, opts);
    fprintf('%i: %g\n',i,obj(i));
end

