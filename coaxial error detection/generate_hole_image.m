function im = generate_hole_image(cx,cy,sz)

if nargin<3
    sz = [50,50];
end

width = 1000;

load 'ring.mat' im1 im2 im3 rank1 rank2 rank3 opa1 opa2 opa3;

dp = [cx,-cy];

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

im =im(251:750,251:750);

im = single(imresize(im,sz));

