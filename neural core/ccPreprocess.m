% 真实数据->归一化数据
%one parameter: get w
%two parameter: process data
function [y, w, b] = ccPreprocess(x, w, b)
    
    if isempty(x)
        y = [];
        w = [];
        b = [];
        return;
    end

    if nargin<=1 || isempty(w)
%         x_min = min(x,[],2);
%         x_max = max(x,[],2);
%         w = 2./(x_max - x_min);
%         b = (x_min + x_max)./(x_min - x_max);
%         y = [];
        x_min = min(x(:));
        x_max = max(x(:));
        w = 2/(x_max-x_min)*ones(size(x,1),1);
        b = (x_min + x_max)/(x_min - x_max)*ones(size(x,1),1);
        y = [];
    else
        y = bsxfun(@plus,bsxfun(@times,w,x), b);
    end

end

