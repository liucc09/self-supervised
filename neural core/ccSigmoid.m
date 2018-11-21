%dzdy������ʱ�� ��ֵ��xΪ��������ֵ
%dzdy����ʱ,   �󵼣�xΪ��������ֵ
%x       B*M,  B is the dimension of clusters, M is the number of samples
%dzdy    B*M
%out     B*M 
function out = ccSigmoid(x, dzdy)

y = (exp(x)-exp(-x))./(exp(x)+exp(-x));  %B*M
% y(x>20) = 1;
% y(x<-20) = -1;
if nargin <= 1 || isempty(dzdy)
  out = 1.313*y ;
else
  dydx = 1.313*(1-y.^2);   %B*M
  out = dydx.*dzdy;  %B*M
end

end

