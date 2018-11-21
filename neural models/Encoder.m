classdef Encoder < handle
   properties
      net;
      dataHeap;
      param;
   end
   events
      Event1
   end
   methods
      function encoder = Encoder(net,dataHeap)
        encoder.net = net;
        encoder.dataHeap = dataHeap;
          
        param0.min_grad = 1.0e-7;
        param0.min_energy = 0.00001;
        param0.min_energy_var = 1.0e-5;
        param0.batchsize = 100;
        param0.useGPU = false;
        param0.useParpool = false;

        param0.iterations = 1;
        
        param0.preprocess = true;
        param0.postprocess = true;
        
        param0.trainLabel = true;
        param0.trainUnlabel = true;
        
        param0.labelPc = 1;
        param0.unlabelPc = 1;
         
        encoder.param = param0;

      end
      
      %ind：第一层神经元密度排序
      function addElements(obj,ind)
          obj.net.addElements(ind);
      end
       
      function obj2 = toGPU(obj,index,num)
        obj2 = Encoder(obj.net.toGPU,obj.dataHeap.toGPU(index,num));  
        obj2.param = obj.param;
      end
            
      function delete(obj)
        delete(obj.net);
        delete(obj.dataHeap);
      end
      
      
   end
end
