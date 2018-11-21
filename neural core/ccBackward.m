function ccBackward(xs,net,layers,dzdy)

for i=length(layers):-1:1
    switch layers{i}.type
        case 'linear'
            [w,b] = net.getWB(layers{i});
            
            net.updateDW(layers{i}, ccLinear(xs{i},w,b,dzdy,2));
            
            net.updateDB(layers{i}, ccLinear(xs{i},w,b,dzdy,3));
            
            if i>1
                dzdy = ccLinear(xs{i},w,b,dzdy,1);
            end
            
       case 'klinear'
            [w,b] = net.getWB(layers{i});
            
            net.updateDW(layers{i}, ccKLinear(xs{i},w,b,dzdy,2));
            
            net.updateDB(layers{i}, ccKLinear(xs{i},w,b,dzdy,3));
                        
            if i>1
                dzdy = ccKLinear(xs{i},w,b,dzdy,1);
            end
            
        case 'sigmoid'
            dzdy = ccSigmoid(xs{i}, dzdy);
            
        case 'relu'
            dzdy = ccRelu(xs{i}, dzdy);
            
        case 'none'
            dzdy = ccNone(xs{i}, dzdy);
        
        case 'kernel'
            dzdy = ccKernel(xs{i}, dzdy);
        
    end
end


end

