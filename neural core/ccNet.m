%train neural net
function rt =  ccNet(encoder)

    rt.sigma = 5e-5;
    rt.lambda1 = 5e-7;
    rt.lambda2 = 0;
    rt.k = 1;
    rt.sk = 0;
    
%     if (encoder.param.trainUnlabel)
%         [zz,~] = ccPredict(encoder.dataHeap.x_unlabel.x, encoder, false);
%         encoder.dataHeap.zz_unlabel = Data(zz);
%     end
    
    ccGradient(encoder.param,encoder.net,encoder.dataHeap); %³õÊ¼Ìİ¶È
    
    rt.p = -encoder.net.dW;
    rt.r = -encoder.net.dW;
    rt.p2 = rt.p'*rt.p;
    rt.gradient = norm(rt.r);
        
    [rt.energys,rt.energys_test] = ccEnergy(encoder.param,encoder.net,encoder.dataHeap);
    
    rt.fail = false;
    rt.stop = false;
    
    for i=1:encoder.param.iterations
                  
        %´ıĞŞ¸Ä
        rt.i = i;
        rt = ccOneStep(rt,encoder);
        
%         if (encoder.param.trainUnlabel)
%             [zz,~] = ccPredict(encoder.dataHeap.x_unlabel.x, encoder, false);
%             encoder.dataHeap.zz_unlabel = Data(zz);
%         end
                
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
%             rt.stop = true;
            break;
        end
                
        fprintf('iter:%d  error:%f  test error:%f  gra:%f\n',i, ener, ener_test, rt.gradient);
        
    end
   


 

