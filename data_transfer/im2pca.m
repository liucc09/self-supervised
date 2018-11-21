%将图像样本转为PCA样本
%translate images to pca
function im2pca(srcPath, desPath)

if nargin<2
    dimension = '3D';
    srcPath = ['datas/data_' dimension '_ser.mat'];
    desPath = ['data_' dimension '_ser.mat'];
end

load( srcPath, 'ims_label', 'ims_unlabel', 'ims_test', 'Y_label', 'Y_unlabel', 'Y_test','unlabel_ser');

n_l = size(ims_label,3);
n_u = size(ims_unlabel,3);
n_t = size(ims_test,3);

ims = single(cat(3,ims_label,ims_unlabel,ims_test));
ims_mean = mean(ims,3);

[ims,principle,importance] = toPCA(ims);

ims_label = ims(:,1:n_l);
ims_unlabel = ims(:,n_l+1:n_l+n_u);
ims_test = ims(:,n_l+n_u+1:end);

save( desPath, 'ims_label', 'ims_unlabel', 'ims_test', 'Y_label', 'Y_unlabel', 'Y_test', 'principle', 'ims_mean','unlabel_ser');

function [ims,principle,importance] = toPCA(ims)
    [n1,n2,n3] = size(ims);

    ims = reshape(ims, [n1*n2,n3]);

    [principle,ims,importance] = pca(ims');
    ims = ims';
    ims = ims(importance>0.01*importance(1),:);

    ims = single(ims);

    im_min = min(ims,[],2);
    im_max = max(ims,[],2);

    ind_del = im_min == im_max;
    ims(ind_del,:) = [];