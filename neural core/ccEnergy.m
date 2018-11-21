%求能量函数，不带预处理和后处理
function [ener,ener_test] = ccEnergy(param, net, dataHeap)
    ener = 0;

    if param.trainLabel
        ener = param.labelPc * energyl(dataHeap.x_label, dataHeap.z_label, net, net.layers);
    end 

    if param.trainUnlabel
        ener = ener + param.unlabelPc * energyu(dataHeap.x_unlabel, dataHeap.s_unlabel, net, net.layers);
    end
       
    ener_test = energyl(dataHeap.x_test, dataHeap.z_test, net, net.layers);

function ener = energyl(datax, dataz, net, layers)
    
    if isempty(datax.x)
        ener = 0;
        return;
    end
    
    xs = ccForward(datax.x, net, layers);
        
    dis = (xs{end}-dataz.x).^2;
    ener = sum(dis(:));

%处理无标签数据
function ener = energyu(datax, datas, net, layers)
    if isempty(datax.x)
        ener = 0;
        return;
    end
    
    xs = ccForward(datax.x, net, layers);

    %处理编码值的理论值
    z = xs{end};
    zind = datas.x;
    for i = 1:size(zind,2)
        for j=1:size(z,1)
            z(j,zind(1,i):zind(2,i)) = linspace(z(j,zind(1,i)),z(j,zind(2,i)),zind(2,i)-zind(1,i)+1);
        end
    end

    dis = (xs{end}-z).^2;
    ener = sum(dis(:));
   
