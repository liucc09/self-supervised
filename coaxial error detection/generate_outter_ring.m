function [im,rank,opa] = generate_outter_ring(r3,r4,width)

if nargin<1
    r3 = 170;
    r4 = 200; 
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

im(:) = exp(-0.5*d02./w0.^2);

im_n = imnoise(im,'gaussian',0,0.01);
im_n(im_n>0.8) = 1;

im = min(im,im_n);


pos = ceil(rand(200,3).*[width width 5]);
im_dot = insertShape(zeros(width,width),'FilledCircle',pos);
im_dot = rgb2gray(im_dot);

im = im - im_dot/2;

d0 = sqrt(d02);
ind = d0<r3;
im(ind) = 0;
rank(ind) = 0;

ind = d0>r4;
im(ind) = 0;


rank(:) = d0/10;

ind = d0<r3;
rank(ind) = 0;

ind = d0>r3 & d0<r4;
opa(ind) = 1;

% figure(1);
% imshow(im);
% figure(2)
% imagesc(rank);

end

