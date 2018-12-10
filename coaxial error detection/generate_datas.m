clear;

%labeled samples
maxDis = 100;
gapDis = 50;

diss = -maxDis:gapDis:maxDis;
[dx,dy] = meshgrid(diss,diss);

dx=dx(:);
dy=dy(:);

Y_label = [dx';dy'];


%unlabeled series
Y_unlabel = [];
unlabel_ser = [];
for i=1:100
    %产生两个值之间以约两度为间隔的序列
    serLength = 0;
    while serLength<5
        d1 = rand(2,1)*200-100;
        d2 = rand(2,1)*200-100;

        dd = sqrt(sum((d1-d2).^2));
        serLength = floor(dd/20);
    end
    
    ds = [linspace(d1(1),d2(1),serLength);
            linspace(d1(2),d2(2),serLength);];
    
    ser1 = size(Y_unlabel,2)+1;
    ser2 = ser1 + size(ds,2) - 1;
    Y_unlabel = [Y_unlabel ds];
    unlabel_ser = [unlabel_ser [ser1 ser2]'];
end

%random samples
Y_test = rand(2,1000)*200-100;

fprintf('%d %d %d\n',size(Y_label,2),size(Y_unlabel,2),size(Y_test,2));

%%
ims_label = generate_hole_images(Y_label);
ims_unlabel = generate_hole_images(Y_unlabel);
ims_test = generate_hole_images(Y_test);

save 'experiments/hole alignment/datas/data_hole_ser.mat' ims_label ims_unlabel ims_test Y_label Y_unlabel Y_test unlabel_ser

%% im2pca
im2pca('experiments/hole alignment/datas/data_hole_ser.mat','experiments/hole alignment/datas/data_hole_ser_pca.mat')