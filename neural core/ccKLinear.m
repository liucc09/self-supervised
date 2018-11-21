%flag = 0， 求值，x为函数输入值，y为输出值
%flag = 1,  对x求导
%flag = 2， 对w求导
%flag = 3， 对b求导
%x B*M
%W A*1
%B A*B
%dzdy A*M
%out A*M or B*M or A*1 or A*B
%model: |W|*(||x-B||)^2
function [out,W,B] = ccKLinear(x,W,B,dzdy,flag)
    
    switch flag
        case 0
            out = sum(x.*x,1);
            out = bsxfun(@plus,out,sum(B.*B,2));
            out = bsxfun(@minus,out,2*B*x);   %A*M x^2-2*x*b+b^2
            out = bsxfun(@times,abs(W),out);
                                               
        case 1
            x = permute(x,[3,1,2]); %1*B*M
            dzdy = permute(dzdy,[1,3,2]); %A*1*M
            
            out = bsxfun(@times,2*abs(W),bsxfun(@minus,x,B));  %A*B*M   
            out = sum(bsxfun(@times,dzdy,out),1); %1*B*M
            out = permute(out,[2,3,1]); %B*M
            
        case 2
            out = sum(x.*x,1);
            out = bsxfun(@plus,out,sum(B.*B,2));
            out = bsxfun(@minus,out, 2*B*x);   %A*M x^2-2*x*b+b^2
                        
            out = bsxfun(@times,sign(W),out);  %A*M
                       
            out = sum(dzdy.*out,2); %A*1

        case 3
            dzdy = permute(dzdy,[1,3,2]); %A*1*M
            x = permute(x,[3,1,2]); %1*B*M
            
            out = bsxfun(@times,2*abs(W),bsxfun(@minus,B,x)); %A*B*M
            out = sum(bsxfun(@times,dzdy,out),3); %A*B
            
    end

end

