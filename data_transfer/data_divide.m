if false
%% 3D
%将样本分为label unlabel 和 test
clear;

load 'datas/data_3D.mat';

%label
corner = mean(abs(ang),1);
ind_label = corner==max(corner) | corner==0;
ims_label = uint8(ims(:,:,ind_label));
Y_label = ang(:,ind_label);


[n1,n2,n3] = size(ims);

rind = randperm(n3);

num_unlabel = floor(n3*0.8);
num_test = n3-num_unlabel;

ind_unlabel = rind(1:num_unlabel);
ind_test = rind(end-num_test+1:end);

ims_unlabel = uint8(ims(:,:,ind_unlabel));
Y_unlabel = ang(:,ind_unlabel);

ims_test = uint8(ims(:,:,ind_test));
Y_test = ang(:,ind_test);


save datas/data_3D_all.mat ims_label ims_unlabel ims_test Y_label Y_unlabel Y_test -v7.3

end
%% 1D
%将样本分为label unlabel 和 test
clear;

load 'datas/data_1D.mat';

%label
corner = mean(abs(ang),1);
ind_label = corner==max(corner) | corner==0;
ims_label = uint8(ims(:,:,ind_label));
Y_label = ang(:,ind_label);


[n1,n2,n3] = size(ims);


%产生无标签序列
serNum =  8;    %产生几个序列
unlabel_ser = []; %2*serNum 每个序列的起始序号和结束序号

ind_unlabel = [];

while (size(unlabel_ser,2)<=serNum)
    ind1 = randperm(n3,1);
    step = randperm(5,1);
    num = randperm(5,1)+4;
    ind2 = ind1+step*(num-1);
    
    if (ind2<=n3)
        ser1 = length(ind_unlabel)+1;
        ind_unlabel = [ind_unlabel ind1:step:ind2];
        ser2 = length(ind_unlabel);
        unlabel_ser = [unlabel_ser [ser1 ser2]'];
    end
end

ims_unlabel = uint8(ims(:,:,ind_unlabel));
Y_unlabel = ang(:,ind_unlabel);


ind_unlabel = unique(ind_unlabel);
ims(:,:,ind_unlabel) = [];
ang(:,ind_unlabel) = [];

ims_test = uint8(ims);
Y_test = ang;

plot(Y_unlabel);

save datas/data_1D_all.mat ims_label ims_unlabel ims_test Y_label Y_unlabel Y_test unlabel_ser -v7.3