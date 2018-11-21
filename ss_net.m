% labeled distribution 15*15*15 125 samples
clear

layers = {} ;

% [ims,us] = generate_datas(1000);
load 'data\pumpkin_data_pca.mat';

%处理数据得到连续序列
series = {};

datas.inputs = ims_unlabel;
datas.targets = Y_unlabel;
datas.inputsl = ims_label;
datas.targetsl = Y_label;
datas.inputst = ims_test;
datas.targetst = Y_test;

datas.unlabel_ser = unlabel_ser;

[dimIn,sampleNum] = size(datas.inputs);
dimEncoder = size(datas.targetsl,1);

dataHeap  = DataHeap;
dataHeap.init(datas);

clear datas ims_label ims_unlabel ims_test

layers{end+1} = struct('type', 'linear','in',dimIn,'out',100); 
layers{end+1} = struct('type', 'relu');
% layers{end+1} = struct('type', 'linear','in',100,'out',100); 
% layers{end+1} = struct('type', 'sigmoid');
layers{end+1} = struct('type', 'linear','in',100,'out',dimEncoder); 


net = Net;
net.init(layers,dataHeap);

encoder = Encoder(net,dataHeap);

clear dataHeap;

%参数配置
encoder.param.batchsize = 200;
encoder.param.useGPU = false;
    
if encoder.param.useGPU
    encoder = encoder.toGPU(1,1);
end

eners = [];
err_u = [];
err_l = [];
err_t = [];

%%
n_u = size(encoder.dataHeap.x_unlabel.x,2);
n_l = size(encoder.dataHeap.x_label.x,2);
n_t = size(encoder.dataHeap.x_test.x,2);
ind_u = 1:n_u;
ind_l = (n_u+1):(n_u+n_l);
ind_t = (n_u+n_l+1):(n_u+n_l+n_t);
in_all = cat(2,encoder.dataHeap.x_unlabel.x,encoder.dataHeap.x_label.x,encoder.dataHeap.x_test.x);

en_all = cat(2,Y_unlabel,Y_label,Y_test);

rt.mu = 0.9;
rt.alpha = 10;

stopCtr = 0;
%%
for i=1:100

    %混合训练
    encoder.param.iterations = 46;
    encoder.param.trainLabel = true;
    if i<2
        encoder.param.trainUnlabel = false;  
    else
        encoder.param.trainUnlabel = true;  
    end
    encoder.param.labelPc = 1;
    encoder.param.unlabelPc = 0.01;
    
    tic;
    rt = ccNet(encoder);
    toc;
            
    
    outputs = ccPredict(in_all, encoder, false);
    
    if encoder.param.useGPU
        ener_cur = gather(rt.energys);
        outputs=gather(outputs);
    else
        ener_cur = rt.energys;
    end    
    
    eners = [eners, ener_cur];
    
    figure(1);
    clf;
    plot(log10(eners));
        
    figure(2);
    clf;
    hold on;
    scatter3(outputs(1,ind_l),outputs(2,ind_l),outputs(3,ind_l),[],'.r');
    scatter3(outputs(1,ind_u),outputs(2,ind_u),outputs(3,ind_u),[],'.k');
        
    out_encoder = ccPostprocess(outputs,encoder.dataHeap.wo,encoder.dataHeap.bo);
    err_encoder = sqrt(sum((out_encoder-en_all).^2,1));
    
    
    figure(3);
    clf;
    hold on;
    hist(err_encoder,50);
    
    err_u(i) = mean(err_encoder(ind_u));
    err_l(i) = mean(err_encoder(ind_l));
    err_t(i) = mean(err_encoder(ind_t));
    
    if i>1 && err_u(i)>=err_u(i-1)
        stopCtr = stopCtr+1;
    else
        stopCtr = 0;
    end
        
    figure(4);
    clf;
    hold on;
    plot(err_u,'g');
    plot(err_l,'r');
    plot(err_t,'k');
        
    pause(0.05);
    
    if  rt.stop || stopCtr>3
        break;
    end

end

%%
disp('Label Resutls:');
disp(['Max error:',num2str(max(err_encoder(ind_l))*180/pi)]);
disp(['Mean error:',num2str(err_l(end)*180/pi)]);
disp(' ');

disp('UnLabel Resutls:');
disp(['Max error:',num2str(max(err_encoder(ind_u))*180/pi)]);
disp(['Mean error:',num2str(err_u(end)*180/pi)]);
disp(' ');

disp('Test Resutls:');
disp(['Max error:',num2str(max(err_encoder(ind_t))*180/pi)]);
disp(['Mean error:',num2str(err_t(end)*180/pi)]);
disp(' ');


 