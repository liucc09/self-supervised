load 'experiments/hole alignment/datas/data_hole_ser_pca.mat' ims_mean principle ims_label
load 'experiments/hole alignment/datas/data_hole_ser.mat' ims_test Y_test
load 'experiments/hole alignment/network.mat' encoder;


%将图片转为PCA向量
ims = bsxfun(@minus,single(ims_test),ims_mean);
ims = reshape(ims,[size(ims,1)^2,size(ims,3)]);

ims = principle\ims;
ims = ims(1:size(ims_label,1),:);

predict_u = ccPredict(ims, encoder, true);

%%
figure(1);
hold on;
for i=1:size(predict_u,2)
plot([Y_test(1,i),predict_u(1,i)],[Y_test(2,i),predict_u(2,i)],'r');
scatter(Y_test(1,i),Y_test(2,i),[],'g.');
end

%%
%hist
errors = predict_u - Y_test;
errors = sqrt(sum(errors.^2,1));
figure(2);
hist(errors,50);
xlabel('误差/(像素)');
ylabel('数量/(个)');

display(mean(errors));

%%
%images and origin
figure(3);
ha = tight_subplot(2,5,[.01 .01],[.1 .01],[.01 .01]);
for i=1:10
    axes(ha(i)); 
    im = generate_hole_image(Y_test(1,i),Y_test(2,i),[500,500]);
    hold on;
    imagesc(flipud(im));
    xlim([0 size(im,2)]);
    ylim([0 size(im,1)]);
    axis square
    axis off
    colormap gray
    scatter(Y_test(1,i)+size(im,2)/2,Y_test(2,i)+size(im,1)/2,[],'bx');
    scatter(predict_u(1,i)+size(im,2)/2,predict_u(2,i)+size(im,1)/2,[],'ro');
end
legend('Ground truth','Predicted');