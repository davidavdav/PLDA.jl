function system = prepSPLDAsystem(model)
% Builds an SPLDA verifier, by using the given SPLDA model. This verifier
% scores all train inputs against all test inputs, to give a matrix of
% llr scores.
%
% Input:
%   model: splda model as trained by train_splda()
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
%     Ntrain, Ftrain are zero and first-order stats for M training inputs.
%     Ntrain: 1-by-M zero order stats, one for each train input. If every
%     train input is represented by just one training input vector, then
%     this is just a row of ones.
%
%     Ftrain: dim-by-M first order stats, one for each train input.
%     Ftrain(:,i) is the sum over all training vectors for training input
%     i. If every train input is represented by just one training input vector, then
%     this is just the original training data.
%
%     Ntest and Ftest is the same for N test inputs.
% 
%     ACCUMtrain: can be [], in which case the identity matrix is assumed.
%                 An M-by-P matrix of 0's or 1's, to accumulate training
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
%  Note: the 'ACCUM' matrices are for convenience only. The same effect may be
%        obtained by just summing the original N and F input stats.
%        The scores are symmetric. This means that we can do 8conv-1conv
%        type scoring, but also 1conv-8conv, or even 8conv-8conv. This is
%        why the interface also allows test inputs to be accumulated into
%        test 'models'. But there really are no explicit speaker model
%        point estimates. This system integrates over all possible speaker
%        models.

if nargin==0
    test_this();
    return
end



D = model.D;
V = model.V;
mu = model.mu;

DV = D*V;

Ftrans = DV'; % transforms input first-order stat to the first-order stat for y.
P0 = Ftrans*V;   % V'*DV 

system.prepStats = @(N,F,accum) prepStats(N,F,accum,Ftrans,mu);
system.score = @(left,right) scoreAll(left,right,P0);


end


function S = scoreAll(left,right,P0)
    m = length(left.N);
    n = length(right.N);
    S = zeros(m,n);
    uR = unique(right.N);
    for nleft = unique(left.N)
        ii = left.N==nleft;
        Fleft = left.F(:,ii);
        for nright = uR;
            jj = right.N==nright;
            Fright = right.F(:,jj);
            S(ii,jj) = score(nleft,Fleft,nright,Fright,P0);
        end
    end
end






function stats = prepStats(N,F,accum,Ftrans,mu)
% N: 1-by-k are zero-order stats
% F: dim-by-k, full-size first-order stats
stats.N = N;
stats.F = Ftrans*(F-bsxfun(@times,mu(:),N(:)')); % first-order stats for y
if ~isempty(accum)
    stats.N = stats.N*accum;
    stats.F = stats.F*accum;
end
end


function S = score(nleft,Fleft,nright,Fright,P0)

Iy = eye(size(P0));
Pleft = nleft*P0+Iy;
Pright = nright*P0+Iy;
P = (nleft+nright)*P0+Iy;

[Cleft,logdetLeft] = invchol(Pleft);
[Cright,logdetRight] = invchol(Pright);
[C,logdet] = invchol(P);
column = 0.5*(logdetLeft+sum(Fleft.*C(Fleft),1)-sum(Fleft.*Cleft(Fleft),1))';
row = 0.5*(logdetRight+sum(Fright.*C(Fright),1)-sum(Fright.*Cright(Fright),1));

S = Fleft'*C(Fright);
S = bsxfun(@plus,S,row-0.5*logdet);
S = bsxfun(@plus,S,column);
    
end


function [inv_map,logdet] = invchol(A)

R = chol(A);
inv_map = @(X) R\(R'\X); 

if nargout>1
    logdet = 2*sum(log(diag(R)));
end


end



function test_this()

close all;

dim = 10;
ydim = 2;
D = randn(dim,dim*2);D = D*D'/(dim*2);
Trans = chol(inv(D))';
V = randn(dim,ydim);
model.V = V;
model.D = D;
model.mu = 50*ones(dim,1);

system =  prepSPLDAsystem(model);

fprintf('synthesizing\n');
nspeakers = 500;
[tar11,non11] = synth(1,1);
[tar12,non12] = synth(1,2);
[tar22,non22] = synth(2,2);
[tar10_1,non10_1] = synth(1,10);
[tar10_10,non10_10] = synth(10,10);


fprintf('evaluating\n');
ape_plot({'1-1',{tar11,non11}},{'1-2',{tar12,non12}},{'2-2',{tar22,non22}},{'10-1',{tar10_1,non10_1}},...
    {'10-10',{tar10_10,non10_10}});



    function [tar,non] = synth(n1,n2)
    speakers = V*randn(ydim,nspeakers);
    train.F = repmat(speakers,1,n1) + Trans*randn(dim,nspeakers*n1);
    train.F = bsxfun(@plus,train.F,model.mu);
    test.F = repmat(speakers,1,n2) + Trans*randn(dim,nspeakers*n2);
    test.F = bsxfun(@plus,test.F,model.mu);
    train.N = ones(1,nspeakers*n1);
    test.N = ones(1,nspeakers*n2);
    train.accum = repmat(eye(nspeakers),n1,1); 
    test.accum = repmat(eye(nspeakers),n2,1); 
    
    fprintf('scoring\n');
    left = system.prepStats(train.N,train.F,train.accum);
    right = system.prepStats(test.N,test.F,test.accum);
    llr = system.score(left,right);
    tar = diag(llr)';
    non = llr(~eye(nspeakers))';
    end


end
