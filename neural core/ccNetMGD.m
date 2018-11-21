%带有动量的梯度下降算法
function rt =  ccNetMGD(encoder,rt)

    if isempty(encoder.net.dW)
        encoder.net.clearGradient();
        rt.r = encoder.net.dW*0;
    else
        rt.r = encoder.net.dW;
    end
                 
    [rt.energys,rt.energys_test] = ccEnergy(encoder.param,encoder.net,encoder.dataHeap);
    
    rt.fail = false;
    rt.stop = false;
    
    for i=1:encoder.param.iterations
                  
        %待修改
        rt.i = i;
        
        ccGradient(encoder.param,encoder.net,encoder.dataHeap); %初始梯度
        
        rt.gradient = norm(encoder.net.dW);
        rt.r = rt.mu*rt.r + (1-rt.mu)*encoder.net.dW;
        
        encoder.net.update(-rt.alpha*rt.r);
               
        [rt.energys(end+1),rt.energys_test(end+1)] = ccEnergy(encoder.param,encoder.net,encoder.dataHeap);
                
        if rt.energys(end)>rt.energys(end-1)
            rt.alpha = rt.alpha*0.5;
        end
        
        ener = rt.energys(end);
        ener_test = rt.energys_test(end);
        
        if (length(rt.energys)>20 && var(rt.energys(end-10:end))<encoder.param.min_energy_var)
            fprintf('Training frozen!\n');
            fprintf('iter:%d  error:%f  test error:%f  gra:%f\n',i, ener, ener_test, rt.gradient);
            break;
        end
                
        if (rt.gradient<=encoder.param.min_grad)
             fprintf('Training frozen!\n');
             fprintf('iter:%d  error:%f  test error:%f  gra:%f\n',i, ener, ener_test, rt.gradient);
%              rt.stop = true;
            break;
        end
        
        if (ener<encoder.param.min_energy)
            fprintf('Success!\n');
            rt.stop = true;
            break;
        end
                
        fprintf('iter:%d  error:%f  test error:%f  gra:%f\n',i, ener, ener_test, rt.gradient);
        
    end
   


 

