## splda.jl.  Simplified Probabillistic Linear Discriminant Analysis.  After Niko Brümmer. 

require("types.jl")

## ivector dimension d
## number of voices nv
## 
function em!(model::SPLDA, S::Matrix{Float64}, F::Matrix{Float64}, N::Vector{Int}; 
             doMinDiv=true, doD=true, doV=true)
    D = model.D                 # d x d
    V = model.V                 # d x nv
    logdetD = 2sum(log(diag(chol(D))))
    
    nv = size(V,2)
    (d, nsp) = size(F)
    
    sumN = sum(N)               # number of vectors in total
    
    Iy = eye(nv)                # nv x nv
    DV = D * V                  # d x nv
    
    Ftrans = DV' # nv x d, transforms input first-order stat to the first-order stat for y.
    P0 = Ftrans * V               # V'*DV, nv x nv

    Pyhat = Ftrans * F          # nv x ns
    Ryy = zeros(nv, nv)
    Syy = zeros(nv, nv);
    Ty = zeros(nv, d)
    obj = 0.5sumN*logdetD - 0.5*sum(D .* S)
    for n = unique(N)
        ii = N .== n
        m = sum(ii)
        P = n * P0 + Iy       # posterior precision
        cholP = chol(P)
        yhat = cholP \ (cholP' \ Pyhat[:,ii])   # posterior mean
    
        Ty += yhat * F[:,ii]'
    
        Pi = cholP \ (cholP' \ Iy);      # posterior covariance
        Syyn =  m*Pi + yhat*yhat';
        Syy += Syyn;
        Ryy += n*Syyn;
    
        logdetP = 2sum(log(diag(cholP)));
        obj = obj - 0.5(m*logdetP - sum(yhat .* Pyhat[:,ii]));  # for y-posterior
    end

    if doV
        V[:] = (Ryy \ Ty)'
    end
    
    if doD
        VTy = V * Ty  
        if doV  # faster, but breaks unless updating V
            D[:] = inv((S - 0.5*(VTy+VTy'))/sumN)
        else # if not updating V
            D[:] = inv((S + V*Ryy*V' - VTy - VTy')/sumN)
        end
    end
    
    ## min div
    C = Syy/nsp;
    CC = chol(C)';
    println(@sprintf("cov(y): trace = %f, logdet = %f", trace(C), 2*sum(log(diag(CC)))))
    if doMinDiv && doV
        V[:] = V*CC
    end
#    model.V = V
#    model.D = D
    obj
end
