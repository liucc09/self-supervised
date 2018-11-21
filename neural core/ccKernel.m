%model: y = exp(-x)
%dzdy   A*M
%x      A*M
%out    A*M or A*M
function out = ccKernel(x,dzdy)
  
y = exp(-x); %A*M

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = -y.*dzdy; %A*M
end

end

