%什么都不做，用于替代激活函数的补丁
%dzdy不存在时， 求值，x为函数输入值
%dzdy存在时,   求导，x为函数输入值
%x       B*M,  B is the dimension of clusters, M is the number of samples
%dzdy    B*M
%out     B*M 
function out = ccNone(x, dzdy)

y = x;  %B*M
% y(x>20) = 1;
% y(x<-20) = -1;
if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy;  %B*M
end

end

