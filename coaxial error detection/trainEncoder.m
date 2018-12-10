% labeled distribution 20*20 samples
clear

layers = {} ;

load 'experiments/hole alignment/datas/data_hole_ser_pca.mat';

% %将图片转为PCA向量
% ims_label_raw = bsxfun(@minus,single(ims_label),ims_mean);
% ims_label = reshape(ims_label_raw,[size(ims_label,1)^2,size(ims_label,3)]);
% 
% ims_label = principle\ims_label;
% ims_label = ims_label(1:size(ims_test,1),:);
%处理数据得到连续序列

% unlabel_ser = unlabel_ser(:,1:100);
% ims_unlabel = ims_unlabel(:,1:unlabel_ser(2,end));
% Y_unlabel = Y_unlabel(:,1:unlabel_ser(2,end));

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
err_pos_u = [];
err_pos_l = [];
err_pos_t = [];

err_rot_u = [];
err_rot_l = [];
err_rot_t = [];

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
for i=1:30

    %混合训练
    encoder.param.iterations = 57;
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
    hold on;
    scatter(outputs(1,ind_l),outputs(2,ind_l),[],'.r');
    scatter(outputs(1,ind_u),outputs(2,ind_u),[],'.k');
    
%     figure(1);
%     clf;
%     hold on;
%     scatter3(outputs(1,ind_l),outputs(2,ind_l),outputs(3,ind_l),[],'.r');
%     scatter3(outputs(1,ind_u),outputs(2,ind_u),outputs(3,ind_u),[],'.k');
        
    out_encoder = ccPostprocess(outputs,encoder.dataHeap.wo,encoder.dataHeap.bo);
    
    err_pos = sqrt(sum((out_encoder(1:2,:)-en_all(1:2,:)).^2,1));
%     err_rot = abs(out_encoder(3,:)-en_all(3,:));
    
    err_pos_u(i) = mean(err_pos(ind_u));
    err_pos_l(i) = mean(err_pos(ind_l));
    err_pos_t(i) = mean(err_pos(ind_t));
    
%     err_rot_u(i) = mean(err_rot(ind_u));
%     err_rot_l(i) = mean(err_rot(ind_l));
%     err_rot_t(i) = mean(err_rot(ind_t));
    
    if i>1 && err_pos_u(i)>=err_pos_u(i-1)
        stopCtr = stopCtr+1;
    else
        stopCtr = 0;
    end
        
    figure(2);
    clf;
    hold on;
    plot(err_pos_u,'g');
    plot(err_pos_l,'r');
    plot(err_pos_t,'k');
    
%     figure(3);
%     clf;
%     hold on;
%     plot(err_rot_u,'g');
%     plot(err_rot_l,'r');
%     plot(err_rot_t,'k');
        
    pause(0.05);
    
    if  rt.stop || stopCtr>10
        break;
    end

end

%
disp('Label Pos Resutls:');
disp(['Mean error:',num2str(err_pos_l(end))]);
disp(' ');

disp('UnLabel Pos Resutls:');
disp(['Mean error:',num2str(err_pos_u(end))]);
disp(' ');

disp('Test Pos Resutls:');
disp(['Mean error:',num2str(err_pos_t(end))]);
disp(' ');

%保存网络
save 'experiments/hole alignment/network.mat' encoder;


 