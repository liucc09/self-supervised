function xs = ccForward(x,net,layers)

xs = cell(1,length(layers)+1);
xs{1} = x;
for i=1:length(layers)
    switch layers{i}.type
        case 'linear'
            [w,b] = net.getWB(layers{i});
            [xs{i+1},~,~] = ccLinear(xs{i},w,b,[],0);
        case 'sigmoid'
            xs{i+1} = ccSigmoid(xs{i});
        case 'relu'
            xs{i+1} = ccRelu(xs{i});
        case 'none'
            xs{i+1} = ccNone(xs{i});
        case 'kernel'
            xs{i+1} = ccKernel(xs{i});
        case 'klinear'
            [w,b] = net.getWB(layers{i});
            [xs{i+1},~,~] = ccKLinear(xs{i},w,b,[],0);
    end
end


end

