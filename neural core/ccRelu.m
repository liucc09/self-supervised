%dzdy������ʱ�� ��ֵ��xΪ��������ֵ
%dzdy����ʱ,   �󵼣�xΪ��������ֵ
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

