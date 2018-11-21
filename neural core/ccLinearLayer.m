function layer = ccLinearLayer(npre, nnext, type, param)
    
    if nargin<4
        param.normalize = false;
    end
    
    switch type
        case 'linear'
%             nn1 = round(nnext/2);
%             nn2 = nnext - nn1;
%             maxv = 1;
%             if nnext>npre
%                 if npre == 1
%                     b = [linspace(-maxv,maxv,nn1) linspace(-maxv,maxv,nn2)]';
%                     w = [ones(nn1,npre);-ones(nn2,npre)]*maxv;
%                 else
%                     b = linspace(-maxv,maxv,nnext)';
%                     w = (rand(nnext,npre)*2-1)*maxv;
%                 end
%             else
%                 b = rand(nnext,1)*2-1;
%                 w = rand(nnext,npre)*2-1;
%             end
%             
%             if nnext == 1
%                 b = 0;
%             end

            b = zeros(nnext,1);
            w = zeros(nnext,npre);

            layer = struct('type', 'linear', ...
                           'w', single(w), ...
                           'b', single(b), ...
                           'normal', param.normalize) ; %是否标准化
        case 'klinear'
            k = exp(log(nnext)/npre);
            
            w = ones(nnext,1)/sqrt(npre);
            b = rand(nnext, npre)'*2-1;

%             k1 = floor(k);
%             k2 = nnext-k1^npre;
%             bb = linspace(-1,1,k1);
%             b1 = bb;
%             for i = 2:npre
%                 bb = repmat(bb,1,k1);
%                 b1 = [repelem(b1,1,k1);bb];
%             end
%             
%             b2 = rand(npre,k2)*2-1;
%             b = [b1 b2];
            
            layer = struct('type', 'klinear', ...
                       'w', single(w), ...
                       'b', single(b)') ; %是否标准化
    end
    

    
end

