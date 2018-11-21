function ccGradient( param, net, dataHeap )

    net.clearGradient();
    
%     if param.useGPU
%         processOnGPU(param,net,dataHeap);
%     elseif param.useParpool
%         processOnParpool(param,net,dataHeap);
%     else
      processOnDevice(param,net,dataHeap);
%     end

% function processOnParpool(param,net,dataHeap)
%     
%     poolobj = gcp('nocreate'); % If no pool, do not create new one.
%     if isempty(poolobj)
%         parpool('local',6);
%     end
%                 
%     spmd
%         net2 = net;
%         dataHeap2 = dataHeap.toSub(labindex,numlabs);
%         processOnDevice(param,net2,dataHeap2);
%         dW = net2.dW;
% %         delete(net2);
% %         delete(dataHeap2);
%     end
%     
%     for i=1:6
%         net.accGradient(dW{i});
%     end
% 
% function processOnGPU(param,net,dataHeap)
% %     gpuNum = gpuDeviceCount;
% %     poolobj = gcp('nocreate'); % If no pool, do not create new one.
% %     if isempty(poolobj)
% %         parpool('local',gpuNum);
% %     end
% %     
% %         gpuDevice([]);
% %             
% %     spmd
% %         gpuDevice(labindex);
% %         gpuDevice(1);
%         net2 = net.toGPU();
%         
% %         dataHeap2 = dataHeap.toGPU(labindex,gpuNum);
%         dataHeap2 = dataHeap.toGPU(1,1);
%         
%         processOnDevice(param,net2,dataHeap2);
%         dW = gather(net2.dW);
%         
% %         delete(net2);
% %         delete(dataHeap2);
% %     end
%     
% %     for i=1:gpuNum
% %         net.accGradient(dW{i});
% %     end
%         net.accGradient(dW);
        
function processOnDevice(param,net,dataHeap)
    
    avgN = 0;
    if (param.trainLabel)
        processl(dataHeap.x_label,dataHeap.z_label,net,net.layers,param.batchsize,param.labelPc);
        avgN = avgN + size(dataHeap.x_label.x,2);
    end

    if (param.trainUnlabel)
        processu(dataHeap.x_unlabel,dataHeap.s_unlabel,net,net.layers,param.batchsize,param.unlabelPc);
%         processl(dataHeap.x_unlabel,dataHeap.zz_unlabel,net,net.layers,param.batchsize,param.unlabelPc);
        avgN = avgN + size(dataHeap.x_unlabel.x,2);
    end
    
    net.dW = net.dW/avgN;
    
%处理带标签数据    
function processl(datax,dataz,net,layers,batchsize,percent)
    trainNum = size(datax.x,2);
    for  i=1:batchsize:trainNum
        t1 = i;
        t2 = min(trainNum,i+batchsize-1);

        xs = ccForward(datax.getBatch(t1:t2),net,layers);
        
        dzdy = percent*2*(xs{end} - dataz.getBatch(t1:t2));

        ccBackward(xs,net,layers,dzdy);
        
        clear xs;
    end

%处理无标签数据
function processu(datax,datas,net,layers,batchsize,percent)
    
    ser1 = 1;
    ser2 = 0;
    while ser1<=size(datas.x,2)
        curSize = 0;
        while ser2<size(datas.x,2) && curSize<batchsize
            ser2 = ser2+1;
            curSize = curSize + datas.x(2,ser2)-datas.x(1,ser2)+1;
        end
        
        %计算这个batch的梯度
        t1 = datas.x(1,ser1);
        t2 = datas.x(2,ser2);
        xs = ccForward(datax.getBatch(t1:t2),net,layers);
        
        %处理编码值的理论值
        z = xs{end};
        zind = datas.x(:,ser1:ser2)-(datas.x(1,ser1)-1);
        for i = 1:size(zind,2)
            for j=1:size(z,1)
                z(j,zind(1,i):zind(2,i)) = linspace(z(j,zind(1,i)),z(j,zind(2,i)),zind(2,i)-zind(1,i)+1);
            end
        end
                
        dzdy = percent*2*(xs{end} - z);

        ccBackward(xs,net,layers,dzdy);
        
        clear xs z;
                
        ser1 = ser2+1;
    end
    
    
    

