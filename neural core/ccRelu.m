%dzdy不存在时， 求值，x为函数输入值
%dzdy存在时,   求导，x为函数输入值
%x       B*M,  B is the dimension of clusters, M is the number of samples
%dzdy    B*M
%out     B*M 
function out = ccRelu(x, dzdy)


if nargin <= 1 || isempty(dzdy)
  out = max(x,0);
  
else
  dydx = max(sign(x),0);   %B*M
  out = dydx.*dzdy;  %B*M
end

end

