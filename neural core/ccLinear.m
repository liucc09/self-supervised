%flag = 0， 求值，x为函数输入值，y为输出值
%flag = 1,  对x求导
%flag = 2， 对w求导
%flag = 3， 对b求导
%x B*M
%W A*B
%B A*1
%dzdy A*M
%out A*M or B*M or A*B or A*1
function [out,W,B] = ccLinear(x,W,B,dzdy,flag)
    
    switch flag
        case 0
            out = bsxfun(@plus,W*x,B);
                              
        case 1
            %dzdy A*M
            out = W'*dzdy; %B*M
            
        case 2
            %dzdy A*M
            %x    B*M
            out = dzdy*x';        %A*B
        case 3
            out = sum(dzdy,2); %A*1
    end

end

