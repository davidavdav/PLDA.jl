function test_em_splda()
    ## synthesize data
    dim = 30
    ydim = 20
    D = randn(dim, 2dim)
    D = D*D'/(2dim)
    V = randn(dim, ydim)
    model1 = SPLDA(D, V)
    
    ns = 100
    F = zeros(dim, ns)
    N = zeros(Int, ns)
    S = zeros(dim, dim)
    
    VY = V*randn(ydim,ns)
    print("Simulating...")
    data = cell(ns)
    Trans = chol(inv(D))'
    for i=1:ns
        N[i] = rand(5:14)
        X = Trans*randn(dim,N[i])
        data[i] = broadcast(+, VY[:,i], X);
        F[:,i] = sum(data[i], 2);
        S += data[i] * data[i]';
    end

    D = eye(dim)
    V = randn(dim, ydim)
    model = SPLDA(D, V)
    model0 = copy(model)
    
    niters = 50
    obj = zeros(niters)

    model = copy(model0)
    for i = 1:niters
        obj[i] = em!(model, S, F, N, doD=false)
    end
    print(obj)

    model = copy(model0)
    for i = 1:niters
        obj[i] = em!(model, S, F, N, doV=false)
    end
    print(obj)

    model = copy(model0)
    for i = 1:niters
        obj[i] = em!(model, S, F, N)
    end
    print(obj)
    

end
