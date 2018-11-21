%inds : same val
%indi : individual val
function [ x,inds,indi,val ] = ccDataSqueeze(x,inds,indi,val)

if nargin<=1 || isempty(inds)
    x_min = min(x,[],2);
    x_max = max(x,[],2);
    inds = (x_max == x_min);
    indi = (x_max ~= x_min);
    val = x_max(inds);
    x(inds,:) = [];
else
    y = zeros(length(inds)+length(indi),size(x,2));
    y(indi,:) = x;
    y(inds,:) = bsxfun(ones(length(inds),size(x,2)),val);
end


end

