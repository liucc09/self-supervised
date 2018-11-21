%x is N*M, N is the number of clusters, M is the number of samples
function y = ccSoftmax(x, dzdy )
    
E = exp(bsxfun(@minus, x, max(x,[],1))) ;
L = sum(E,1) ;
y = bsxfun(@rdivide, E, L) ;

if nargin <= 1 || isempty(dzdy)
    return ; 
end

% backward
y = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, 1)) ;

end

