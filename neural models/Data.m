classdef Data < handle
    %DATA 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        x;
    end
    
    methods
        function data = Data(x)
            data.x = x;
        end
        
        function batch = getBatch(obj, inds)
            batch = obj.x(:,inds);
        end
        
        function obj2 = toSub(obj,index,num)
            len_x = size(obj.x,2);
            
            batchNum = ceil(len_x/num); 
            
            ind1 = (index-1)*batchNum+1;
            ind2 = min(len_x, index*batchNum);
           
            x2 = obj.getBatch(ind1:ind2);
            obj2 = Data(x2);
        end
        
        function obj2 = toGPU(obj,index,num)
            len_x = size(obj.x,2);
            
            batchNum = ceil(len_x/num); 
            
            ind1 = (index-1)*batchNum+1;
            ind2 = min(len_x, index*batchNum);
           
            x2 = gpuArray(obj.getBatch(ind1:ind2));
            obj2 = Data(x2);
        end
    end
    
end

