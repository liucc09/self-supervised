function net = ccUpdate( net, dTheta )

cur_ind = 1;
for i = 1:length(net.layers)
    if isfield(net.layers{i},'w')
        [wd1,wd2] = size(net.layers{i}.w);
        [bd1,bd2] = size(net.layers{i}.b);

        dw = reshape(dTheta(cur_ind:cur_ind+wd1*wd2-1),[wd1,wd2]);
        cur_ind = cur_ind + wd1*wd2;
        net.layers{i}.w = net.layers{i}.w + dw;

        db = reshape(dTheta(cur_ind:cur_ind+bd1*bd2-1),[bd1,bd2]);
        cur_ind = cur_ind + bd1*bd2;
        net.layers{i}.b = net.layers{i}.b + db;
    end
 end


end

