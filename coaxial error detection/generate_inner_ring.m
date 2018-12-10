function [im,rank,opa] = generate_inner_ring(r1,r2,width)

if nargin<1
    r1 = 125;
    r2 = 145;
    width = 1000;
end

w0=200;

im = zeros(width,width);
rank = zeros(width,width);
opa = zeros(width,width);

[xs,ys] = meshgrid(1:width,1:width);

xs = xs(:);
ys = ys(:);

ps = [xs';ys'];

c0 = [floor(width/2);floor(width/2)];
d02 = sum((ps-c0).^2,1);

im(:) = 0.8*exp(-0.5*d02./w0.^2);

im_n = imnoise(im,'gaussian',0,0.01);
im_n(im_n>0.8) = 1;

im = min(im,im_n);


pos = ceil(rand(200,3).*[width width 5]);
im_dot = insertShape(zeros(width,width),'FilledCircle',pos);
im_dot = rgb2gray(im_dot);

im = im - im_dot/2;

d0 = sqrt(d02);
ind = d0>r2;
im(ind) = 0;

ind = d0<r1;
im(ind) = 1;

rank(:) = d0/10;
rank(ind) = r1/10;

ind = d0>r2;
rank(ind) = d0(ind)/20 + r2/10 - r2/20;

%不同区域透明度
ind = d0<r2;
opa(ind) = 1;

ind = d0>r2;
opa(ind) = 0;
opa = max(0,opa);
% figure(1);
% imshow(im);
% figure(2)
% imagesc(rank);

end

