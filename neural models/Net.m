classdef Net < handle
    %NET �˴���ʾ�йش����ժҪ
    %   �˴���ʾ��ϸ˵��
    
    properties
      numWeightElements;
      numLayers;
      encoderInd;
      dimInputs;
      dimEncoder;
      
      arrayType = 'single';
      
      calNet;
      dW;
      
      layers;
      
      isSubNet = false;
      
      l1w = 0.1;
      l1w2 = 0.3;
     
    end
    
    methods
        function init(obj, layers, dataHeap)
            obj.initLayers(layers);
            obj.initWB(dataHeap);
        end
        
        function initLayers(obj,layers)
                       
            obj.numLayers = length(layers);
            endInd = 0;
            for i=1:obj.numLayers
                if isfield(layers{i},'encoder')
                    obj.dimEncoder = layers{i-1}.out;
                    obj.encoderInd = i;
                end
                switch (layers{i}.type)
                    case 'linear'
                        curInd = endInd+1;
                        endInd = endInd + layers{i}.out;

                        layers{i}.b_ind1 = curInd;
                        layers{i}.b_ind2 = endInd;
                        layers{i}.b_dim1 = layers{i}.out;
                        layers{i}.b_dim2 = 1;

                        curInd = endInd+1;
                        endInd = endInd + layers{i}.in*layers{i}.out;

                        layers{i}.w_ind1 = curInd;
                        layers{i}.w_ind2 = endInd;
                        layers{i}.w_dim1 = layers{i}.out;
                        layers{i}.w_dim2 = layers{i}.in;
                    case 'klinear'
                        curInd = endInd+1;
                        endInd = endInd + layers{i}.in*layers{i}.out;

                        layers{i}.b_ind1 = curInd;
                        layers{i}.b_ind2 = endInd;
                        layers{i}.b_dim1 = layers{i}.out;
                        layers{i}.b_dim2 = layers{i}.in;

                        curInd = endInd+1;
                        endInd = endInd + layers{i}.out;

                        layers{i}.w_ind1 = curInd;
                        layers{i}.w_ind2 = endInd;
                        layers{i}.w_dim1 = layers{i}.out;
                        layers{i}.w_dim2 = 1;
                    case 'sigmoid'
                    case 'kernel'
                    case 'none'
                end

            end
            obj.numWeightElements = endInd;
            obj.dimInputs = layers{1}.in;
            obj.layers = layers;
            
            %��ʼ����kernel���w
            in_dim = layers{1}.in;
            
            obj.l1w = -log(0.001)/in_dim;
            obj.l1w2 = -log(0.001)/in_dim;
           
        end
        
        function initWB(obj,dataHeap)
            
            obj.initWB0;

            %init first kernel layer
            if strcmp(obj.layers{1}.type,'klinear')
                layer = obj.layers{1};
                xm = size(dataHeap.x_unlabel.x,1);
                if layer.b_dim2 ~= xm
                    error('ά�Ȳ�ƥ��');
                end

                dataAll = cat(2,dataHeap.x_unlabel.x,dataHeap.x_label.x);
                [~,b] = kmeans(dataAll',layer.b_dim1);

    %             b = rand(layer.b_dim1*layer.b_dim2,1,'single')*2-1;

                obj.calNet(layer.b_ind1:layer.b_ind2,1) = eval([obj.arrayType,'(b(:))']);
                obj.calNet(layer.w_ind1:layer.w_ind2,1) = eval([obj.arrayType,'(obj.l1w)']);
            end

        end
        
        %��ʼ������һ����ĸ���
        function initWB0(obj)
            %create weights
            obj.calNet = randn(obj.numWeightElements,1,obj.arrayType)*0.1;
        end
                
        function [w,b] = getWB(obj, layer)
            w = reshape(obj.calNet(layer.w_ind1:layer.w_ind2),[layer.w_dim1,layer.w_dim2]);
            b = reshape(obj.calNet(layer.b_ind1:layer.b_ind2),[layer.b_dim1,layer.b_dim2]);
        end

        %�ۼ�dW�е�W
        function updateDW(obj, layer, dWB)
            obj.dW(layer.w_ind1:layer.w_ind2) = obj.dW(layer.w_ind1:layer.w_ind2) + ...
                                                       reshape(dWB,[layer.w_dim1*layer.w_dim2,1]);
        end

        %�ۼ�dW�е�B
        function updateDB(obj, layer, dWB)
            obj.dW(layer.b_ind1:layer.b_ind2) = obj.dW(layer.b_ind1:layer.b_ind2) + ...
                                                        reshape(dWB,[layer.b_dim1*layer.b_dim2,1]);
            if obj.isSubNet
                obj.dW(obj.layers{3}.b_ind1:obj.layers{3}.b_ind2) = 0;
            end
        end

        function update(obj, dT)
            obj.calNet = obj.calNet + dT;
        end

        function accGradient(obj,dW2)
            obj.dW = obj.dW + dW2;
        end

        function clearGradient(obj)
            obj.dW = zeros(obj.numWeightElements,1,obj.arrayType);
        end
        
        function net2 = toGPU(obj)
            net2 = Net;
            net2.layers = obj.layers;
            
            net2.numWeightElements = obj.numWeightElements;
            net2.numLayers = obj.numLayers;
            net2.encoderInd = obj.encoderInd;
            net2.dimInputs = obj.dimInputs;
            net2.dimEncoder = obj.dimEncoder;
            
            net2.calNet = gpuArray(obj.calNet);
            net2.dW = gpuArray(obj.dW);
            
            net2.arrayType = 'gpuArray';
        end
        
        function net2 = copy(obj)
            net2 = Net;
            net2.layers = obj.layers;
            
            net2.numWeightElements = obj.numWeightElements;
            net2.numLayers = obj.numLayers;
            net2.encoderInd = obj.encoderInd;
            net2.dimInputs = obj.dimInputs;
            net2.dimEncoder = obj.dimEncoder;
            
            net2.calNet = obj.calNet;
            net2.dW = obj.dW;
            
            net2.arrayType = obj.arrayType;
        end
        
        %���������磬����ѵ������ӵ���Ԫ
        function netNew = toSubNew(net,eleNew)
            netNew = Net;
            
            netNew.isSubNet = true;
            netNew.arrayType = net.arrayType;
            
            layerN = net.layers;
            
            layerN{1}.out = size(eleNew,2);
            layerN{3}.in = layerN{1}.out;
            
            netNew.initLayers(layerN);
            netNew.initWB0;
            
            layer = netNew.layers{1};
            
            eleNew = eleNew';
            
            netNew.calNet(layer.b_ind1:layer.b_ind2,1) = eval([net.arrayType,'(eleNew(:))']);
            netNew.calNet(layer.w_ind1:layer.w_ind2,1) = eval([net.arrayType,'(net.l1w2)']);
            
            netNew.calNet(netNew.layers{3}.b_ind1:netNew.layers{3}.b_ind2,1) = net.calNet(net.layers{3}.b_ind1:net.layers{3}.b_ind2,1);
                       
        end
        
        %�ں����������һ��͵�����
        function merge(net,net2)
            layer_temp = net.layers;
            
            %ֻ�������Ĳ�����¼��������
            for i=1:net.numLayers
               if (strcmp(layer_temp{i}.type, 'linear') || strcmp(layer_temp{i}.type, 'klinear'))
                    
                  [layer_temp{i}.w,layer_temp{i}.b] = net.getWB(layer_temp{i});
                   
               end
            end
            
            %���������е���Ԫ��ӵ�ԭ������
            [w1n,b1n] = net2.getWB(net2.layers{1});
            layer_temp{1}.w = [layer_temp{1}.w; w1n];
            layer_temp{1}.b = [layer_temp{1}.b; b1n];

            [w3n,~] = net2.getWB(net2.layers{3});
            layer_temp{3}.w = [layer_temp{3}.w w3n];
            
            %��������������
            %init second kernel layer
            layer_temp{net.encoderInd+1}.b = rand(layer_temp{net.encoderInd+1}.b_dim1,layer_temp{net.encoderInd+1}.b_dim2,net.arrayType)*2-1;
            layer_temp{net.encoderInd+1}.w = net.l5w*ones(layer_temp{net.encoderInd+1}.w_dim1,layer_temp{net.encoderInd+1}.w_dim2,net.arrayType);
            
            layer_temp{end}.b = layer_temp{end}.b*0;
            layer_temp{end}.w = layer_temp{end}.w*0;
            
            %����������д��������
            net.calNet = [];
            for i=1:net.numLayers
               if (strcmp(layer_temp{i}.type, 'linear') || strcmp(layer_temp{i}.type, 'klinear'))
                    cur_ind = length(net.calNet);
%                     if i<=net.encoderInd+1 %ֻ�б����Ĳ�����Ҫ����
                    net.calNet = [net.calNet; layer_temp{i}.b(:); layer_temp{i}.w(:)];
                    [net.layers{i}.b_dim1,net.layers{i}.b_dim2] = size(layer_temp{i}.b);
                    [net.layers{i}.w_dim1,net.layers{i}.w_dim2] = size(layer_temp{i}.w);

                    
                    net.layers{i}.b_ind1 = cur_ind+1; cur_ind = cur_ind+net.layers{i}.b_dim1*net.layers{i}.b_dim2;
                    net.layers{i}.b_ind2 = cur_ind;

                    net.layers{i}.w_ind1 = cur_ind+1; cur_ind = cur_ind+net.layers{i}.w_dim1*net.layers{i}.w_dim2;
                    net.layers{i}.w_ind2 = cur_ind;
                end
            end
            
            net.numWeightElements = length(net.calNet);
            net.clearGradient();
        end
        
        %�滻�����һ�������ý�С����Ԫ,ratio,ɾ������Ԫ����
        function b = replaceElements(net,ratioDel,ratioAdd, err_sample)
            layer_temp = net.layers;
            
            %ֻ�������Ĳ�����¼��������
            for i=1:net.numLayers
               if (strcmp(layer_temp{i}.type, 'linear') || strcmp(layer_temp{i}.type, 'klinear'))
                    
                  [layer_temp{i}.w,layer_temp{i}.b] = net.getWB(layer_temp{i});
                   
               end
            end
            
            w3max = max(abs(layer_temp{3}.w),[],1);
            eleNum = length(w3max);
            eleAdd = fix(eleNum*ratioAdd);
            [~,indDel] = sort(w3max);
            indDel = indDel(1:fix(eleNum*ratioDel)); %��ɾ����Ԫ���
                        
            layer_temp{1}.b(indDel,:) = [];
            layer_temp{1}.w(indDel,:) = [];
            
            layer_temp{3}.w(:,indDel) = [];
            
            %�����µ���Ԫ
            eleAdd = min(size(err_sample,2),eleAdd);
            [~,b] = kmeans(err_sample',eleAdd);
            
            %���µ���Ԫ��ӵ�ԭ������
            layer_temp{1}.w = [layer_temp{1}.w; net.l1w2*ones(eleAdd,1,obj.arrayType)];
%                 layer_temp{1}.b = [layer_temp{1}.b; rand(eleAdd,layer_temp{1}.b_dim2,'gpuArray')*2-1];
            layer_temp{1}.b = [layer_temp{1}.b; gpuArray(b)];

            layer_temp{3}.w = [layer_temp{3}.w zeros(layer_temp{3}.w_dim1,eleAdd,obj.arrayType)];

            %���ñ�������
%                 layer_temp{net.encoderInd+1}.b = rand(layer_temp{net.encoderInd+1}.b_dim1,layer_temp{net.encoderInd+1}.b_dim2,'gpuArray')*2-1;
%                 layer_temp{net.encoderInd+1}.w = ones(layer_temp{net.encoderInd+1}.w_dim1,layer_temp{net.encoderInd+1}.w_dim2,'gpuArray')*gpuArray(net.l5w);


            
            net.calNet = [];
            for i=1:net.numLayers
               if (strcmp(layer_temp{i}.type, 'linear') || strcmp(layer_temp{i}.type, 'klinear'))
                   
                    cur_ind = length(net.calNet);
%                     if i<=net.encoderInd+1 %ֻ�б����Ĳ�����Ҫ����
                    net.calNet = [net.calNet; layer_temp{i}.b(:); layer_temp{i}.w(:)];
                    [net.layers{i}.b_dim1,net.layers{i}.b_dim2] = size(layer_temp{i}.b);
                    [net.layers{i}.w_dim1,net.layers{i}.w_dim2] = size(layer_temp{i}.w);
%                     else
%                         if isa( layer_temp{1}.w,'gpuArray')
%                             net.calNet = [net.calNet; zeros(net.layers{i}.b_dim1*net.layers{i}.b_dim2,1,'gpuArray'); zeros(net.layers{i}.w_dim1*net.layers{i}.w_dim2,1,'gpuArray')];
%                         else
%                             net.calNet = [net.calNet; zeros(net.layers{i}.b_dim1*net.layers{i}.b_dim2,1,'single'); zeros(net.layers{i}.w_dim1*net.layers{i}.w_dim2,1,'single')];
%                         end
%                     end
                    
                    net.layers{i}.b_ind1 = cur_ind+1; cur_ind = cur_ind+net.layers{i}.b_dim1*net.layers{i}.b_dim2;
                    net.layers{i}.b_ind2 = cur_ind;

                    net.layers{i}.w_ind1 = cur_ind+1; cur_ind = cur_ind+net.layers{i}.w_dim1*net.layers{i}.w_dim2;
                    net.layers{i}.w_ind2 = cur_ind;
                                 
               end
            end
            
            net.numWeightElements = length(net.calNet);
            net.clearGradient();
                   
        end
                
    end
        
end

