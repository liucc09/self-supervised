% 归一化数据->真实数据
%one parameter: get w
%two parameter: process data
function [y, w, b] = ccPostprocess(x, w, b)
    
    if nargin<=1 || isempty(w)
        x_min = min(x,[],2);
        x_max = max(x,[],2);
        w = 2./(x_max - x_min);
        b = (x_min + x_max)./(x_min - x_max);
        y = [];
    else 
        y = bsxfun(@rdivide,bsxfun(@minus,x,b),w);
    end

end

