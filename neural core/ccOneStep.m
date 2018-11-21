%Err   N*M
%dzdy  A*N*M
function rt = ccOneStep(rt,encoder)

net2 = encoder.net.copy;
%共轭梯度法
if ~rt.fail
    sigmak = rt.sigma/sqrt(rt.p2);
    
    %求近似海森矩阵
    net2.update(sigmak*rt.p);
    ccGradient(encoder.param,net2,encoder.dataHeap);
    
    rt.sk = (net2.dW + rt.r)/sigmak; %Q*p
    rt.delta = rt.p'*rt.sk; %p'Qp
    
end

rt.sk = rt.sk + (rt.lambda1-rt.lambda2)*rt.p;
rt.delta = rt.delta + (rt.lambda1 - rt.lambda2)*rt.p2;

if rt.delta<=0   %非正定
    rt.sk = rt.sk + (rt.lambda1 - 2*rt.delta/rt.p2)*rt.p;
    rt.lambda2 = 2*(rt.lambda1 - rt.delta/rt.p2);
    rt.delta = -rt.delta + rt.lambda1*rt.p2;
    rt.lambda1 = rt.lambda2;
end

muk = rt.p'*rt.r;
alpha = muk/rt.delta;

net2.update(alpha*rt.p);

[ener,ener_test] = ccEnergy(encoder.param,net2,encoder.dataHeap);
difk = (2*rt.delta*(rt.energys(end) - ener))/muk^2;

difk = 0.5;

if difk>=0
    r_old = rt.r;
    encoder.net.calNet = net2.calNet;
    delete(net2);
    
    rt.energys(end+1) = ener;
    rt.energys_test(end+1) = ener_test;
    
    ccGradient( encoder.param, encoder.net, encoder.dataHeap ); %获得梯度
    
    rt.r = -encoder.net.dW;
    rt.gradient = norm(rt.r);
    
    rt.lambda2 = 0;
    rt.fail = false;
    
    if rem(rt.i, length(rt.r)) == 1
        rt.p = rt.r;
    else
        beta = (rt.r'*rt.r-rt.r'*r_old)/muk;
        rt.p = rt.r + beta*rt.p;
    end    
    
    rt.p2 = rt.p'*rt.p;
    
    if difk>=0.75
        rt.lambda1 = 0.25*rt.lambda1;
    end
    
else
    rt.lambda2 = rt.lambda1;
    rt.fail = true;
end

if difk<0.25 && difk>=0
%     worker.lambdak = worker.lambdak + worker.deltak*(1 - difk)/worker.nrmsqr_dWB;
      rt.lambda1 = rt.lambda1 + rt.delta*(1-difk)/rt.p2;
%     rt.lambda1 = 4*rt.lambda1;
end

