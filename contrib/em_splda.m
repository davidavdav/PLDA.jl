function [model,obj] = em_splda(model, S, F, N, opts)

%Inputs: (Assume mean has been subtracted from data)
%  S: data*data', i.e. dim-by-dim scatter matrix
%  F: dim-by-ns, first order stats, i.e. sum of data vectors for every speaker
%  N: 1-by-ns, zero-order stat, i.e. number of vectors per speaker
%  model.D: dim-by-dim within-speaker precision matrix, where dim is the 
%           input vector dimension.
%  model.V: dim-by-nvoices speaker factor loading matrix (initialize using
%           randn)


if nargin==0
    test_this();
    return
end


if ~exist('opts','var') 
    opts  = [];
end
if ~isfield(opts,'doMinDiv'), opts.doMinDiv = true; end
if ~isfield(opts,'doD'), opts.doD = true; end
if ~isfield(opts,'doV'), opts.doV = true; end
if ~(opts.doD || opts.doV), error('nothing to do'); end

D = model.D;
V = model.V;
logdetD = 2*sum(log(diag(chol(D))));



ydim = size(V,2);
[dim,nsp] = size(F); 


sumN = sum(N);

Iy = eye(ydim);
DV = D*V;


Ftrans = DV'; % transforms input first-order stat to the first-order stat for y.
P0 = Ftrans*V;   % V'*DV;

Pyhat = Ftrans*F;
Ryy = zeros(ydim);
Syy = zeros(ydim);
Ty = zeros(ydim,dim);
obj = 0.5*sumN*logdetD - 0.5*D(:)'*S(:); 
for n = unique(N)
    ii = (N==n);
    m = sum(ii);
    P = n*P0+Iy;      %posterior precision
    cholP = chol(P);
    yhat = cholP\(cholP'\Pyhat(:,ii));  %posterior mean
    
    Ty = Ty + yhat*F(:,ii)';
    
    Pi = cholP\(cholP'\Iy);     %posterior covariance
    Syyn =  m*Pi+yhat*yhat';
    Syy = Syy + Syyn;
    Ryy = Ryy + n*Syyn;
    
    logdetP = 2*sum(log(diag(cholP)));
    obj = obj - 0.5*(m*logdetP - sum(sum(yhat.*Pyhat(:,ii),1))); %for y-posterior
end


if opts.doV
    V = (Ryy\Ty)';
end

if opts.doD
    VTy = V*Ty;  
    if opts.doV %faster, but breaks unless updating V
	    D = inv((S - 0.5*(VTy+VTy'))/sumN);
    else %if not updating V
	    D = inv((S + V*Ryy*V' - VTy - VTy')/sumN);
    end
end


%min div
C = Syy/nsp;
CC = chol(C)';
fprintf('  cov(y): trace = %g, logdet = %g\n',trace(C),2*sum(log(diag(CC))));
if opts.doMinDiv && opts.doV
    V = V*CC;
end


model.V = V;
model.D = D;

end



function test_this()

close all;

%synthesize data
dim = 30;
ydim = 20;
D = randn(dim,dim*2); D = D*D'/(dim*2);
V = randn(dim,ydim);
model1.V = V;
model1.D = D;

ns = 100;
F = zeros(dim,ns);
N = zeros(1,ns);
S = zeros(dim);
VY = V*randn(ydim,ns);
h = waitbar(0,'simulating');
data = cell(1,ns);
Trans = chol(inv(D))';
for i=1:ns
    N(i) = 5+floor(rand*10);
    X = Trans*randn(dim,N(i));
    data{i} = bsxfun(@plus,VY(:,i),X);
    F(:,i) = sum(data{i},2);
    S = S + data{i}*data{i}';
    waitbar(i/ns,h);
end
close(h);

%init model
%D = inv(S/sum(N));
D = eye(dim);
V = randn(dim,ydim);
model.V = V;
model.D = D;


model0 = model;

niters = 50;
obj = zeros(1,niters);

model = model0;
opts.doD = false;
opts.doV = true;
%leg = {'mindiv'};
leg = {'only V'};
for i=1:niters
    [model,obj(i)] = em_splda(model,S,F,N,opts);
    fprintf('%i: %g\n',i,obj(i));
end

plot(obj,'g');
hold;
%pause;

model = model0;
leg = {leg{:},'only D'};
%opts.doMinDiv = false;
opts.doD = true;
opts.doV = false;
for i=1:niters
    [model,obj(i)] = em_splda(model,S,F,N,opts);
    fprintf('%i: %g\n',i,obj(i));
end
plot(obj,'b');

model = model0;
leg = {leg{:},'both'};
%opts.doMinDiv = false;
opts.doD = true;
opts.doV = true;
for i=1:niters
    [model,obj(i)] = em_splda(model,S,F,N,opts);
    fprintf('%i: %g\n',i,obj(i));
end
plot(obj,'r');

legend(leg);

end
