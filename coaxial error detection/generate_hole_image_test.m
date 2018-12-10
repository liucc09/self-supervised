%%
width = 1000;

[im1,rank1,opa1] = generate_inner_ring();
[im2,rank2,opa2] = generate_middle_ring();
[im3,rank3,opa3] = generate_outter_ring();

save 'ring.mat' im1 im2 im3 rank1 rank2 rank3 opa1 opa2 opa3

%%
dp = [0,10];

im2 = imtranslate(im2,dp);
rank2 = imtranslate(rank2,dp);
opa2 = imtranslate(opa2,dp);

rks = cat(3,rank1,rank2,rank3);
ims = cat(3,im1,im2,im3);
opas = cat(3,opa1,opa2,opa3);

[~,ind] = sort(rks,3);

[ind2,ind1] = meshgrid(1:width,1:width);
ind1 = ind1(:);
ind2 = ind2(:);

%layer2
ind3 = ind(:,:,2);
ind3 = ind3(:);

im = zeros(width,width);
rkl2 = zeros(width,width);
im(:) = ims(sub2ind(size(ims),ind1,ind2,ind3));
rkl2(:) = rks(sub2ind(size(ims),ind1,ind2,ind3));

%layer3
ind3 = ind(:,:,3);
ind3 = ind3(:);

im_a = ims(sub2ind(size(ims),ind1,ind2,ind3));
opa_a = opas(sub2ind(size(ims),ind1,ind2,ind3));
opa_a = reshape(opa_a,[width,width]);

rkl3 = zeros(width,width);
rkl3(:) = rks(sub2ind(size(ims),ind1,ind2,ind3));

opa_a = max(opa_a,min(rkl3-rkl2,0.5)/0.5);

im(:) = im(:).*(1-opa_a(:))+im_a.*opa_a(:);

w = fspecial('gaussian',[6,6],3);
im = imfilter(im,w);

im = imcrop(im,[width/4 width/4 width/2 width/2]);
im = imresize(im,0.1);
imshow(im);