%ʲô���������������������Ĳ���
%dzdy������ʱ�� ��ֵ��xΪ��������ֵ
%dzdy����ʱ,   �󵼣�xΪ��������ֵ
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

