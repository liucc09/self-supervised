function ims = generate_hole_images(labels,sz)
    if nargin<2
        sz = [50,50];
    end
    ims = single([]);
    for i =1:size(labels,2)
        ims(:,:,i) = generate_hole_image(labels(1,i),labels(2,i),sz);
        fprintf('%d/%d\n',i,size(labels,2));
    end

end

