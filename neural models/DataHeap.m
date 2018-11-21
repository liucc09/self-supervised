classdef DataHeap < handle
    %DATAHEAP �˴���ʾ�йش����ժҪ
    %   �˴���ʾ��ϸ˵��
    
    properties
      x_unlabel; %�ޱ�ǩ����
      z_unlabel; 
      
      zz_unlabel;
      
      s_unlabel; %series
            
      x_label; %����ǩ����
      z_label;
      
      x_test;
      z_test;
      
      wi;
      bi;
      wo;
      bo;
      
      arrayType = 'single';
    end
    
    methods
        
        function init(obj, datas)
            obj.s_unlabel = Data(datas.unlabel_ser);
            
            in_all = cat(2,datas.inputs,datas.inputsl,datas.inputst);
            
            [~,obj.wi,obj.bi] = ccPreprocess(in_all);
            
            mi_all = cat(2,datas.targets,datas.targetsl,datas.targetst);
            
            [~,obj.wo,obj.bo] = ccPreprocess(mi_all);
            
            data_x = ccPreprocess(datas.inputs, obj.wi, obj.bi); %input
            data_z = ccPreprocess(datas.targets, obj.wo, obj.bo); %encode

            num = size(data_x,2);
                       
            if num~=size(data_z,2)
                error('ѵ�����ݺͱ�������������һ��');
            end

            obj.x_unlabel = Data(data_x);
            obj.z_unlabel = Data(data_z);
            
            obj.x_label = Data(ccPreprocess(datas.inputsl, obj.wi, obj.bi));  %input
            obj.z_label = Data(ccPreprocess(datas.targetsl, obj.wo, obj.bo)); %coding
            
            obj.x_test = Data(ccPreprocess(datas.inputst, obj.wi, obj.bi));  %input
            obj.z_test = Data(ccPreprocess(datas.targetst, obj.wo, obj.bo)); %coding
        end
        
        function obj2 = toSub(obj,index,num)
            obj2 = DataHeap;
            
            obj2.x_unlabel = obj.x_unlabel.toSub(index,num);
            obj2.z_unlabel = obj.z_unlabel.toSub(index,num);
            obj2.s_unlabel = obj.s_unlabel.toSub(index,num);
            
            obj2.x_test  = obj.x_test.toSub(index,num);
            obj2.z_test  = obj.z_test.toSub(index,num);
            
            obj2.x_label = obj.x_label.toSub(index,num);
            obj2.z_label = obj.z_label.toSub(index,num);
            
            
            obj2.arrayType = obj.arrayType;
        end
        
        %���޸�
        function obj2 = toGPU(obj,index,num)
            
            obj2 = DataHeap;
            
            obj2.x_unlabel = obj.x_unlabel.toGPU(index,num);
            obj2.z_unlabel = obj.z_unlabel.toGPU(index,num);
            obj2.s_unlabel = obj.s_unlabel.toGPU(index,num);
            
            obj2.x_test  = obj.x_test.toGPU(index,num);
            obj2.z_test  = obj.z_test.toGPU(index,num);
            
            obj2.x_label = obj.x_label.toGPU(index,num);
            obj2.z_label = obj.z_label.toGPU(index,num);
            
            obj2.wi = obj.wi;
            obj2.bi = obj.bi;
            obj2.wo = obj.wo;
            obj2.bo = obj.bo;
            
            obj2.arrayType = 'gpuArray';
            
        end
        
        function delete(obj)
            delete(obj.x_unlabel);
            delete(obj.z_unlabel);
            delete(obj.s_unlabel);
            delete(obj.x_test);
            delete(obj.z_test);
            delete(obj.x_label);
            delete(obj.z_label);
        end
        
        %Ϊ�����ӵ���Ԫ�ڵ�׼�����ݣ���Ԫ����������ʹ����ȷ����ֵ��Զ����Ԫ����������ֵΪ0������������������Ӱ�죩
        function new = toNearElementPatch(obj,eleNew,encoder_err,l1w,b3)
            
            if (size(obj.z_label.x,2)~=size(encoder_err,2))
                error('����ά�Ȳ�ƥ��');
            end
            
            new = DataHeap;
            
            new.wi = obj.wi;
            new.bi = obj.bi;
            new.wo = obj.wo;
            new.bo = obj.bo;
                        
            %�޼ල����
            dis_min = squeeze(min(sum(bsxfun(@minus,obj.x_unlabel.x,permute(eleNew,[1,3,2])).^2,1),[],3));
            dis_min = exp(-l1w*dis_min);
                        
            ind_sel = dis_min>0.9;
            new.x_unlabel = Data(obj.x_unlabel.x(:,ind_sel));
            new.z_unlabel = Data(obj.z_unlabel.x(:,ind_sel));
            
            %�ල���ݣ��������֣���Ԫ�����������룬Զ����Ԫ0���룩
            %����
            dis_min = squeeze(min(sum(bsxfun(@minus,obj.x_label.x,permute(eleNew,[1,3,2])).^2,1),[],3));
            dis_min = exp(-l1w*dis_min);
            ind_near = find(dis_min>0.9);
            num_near = length(ind_near);
            %Զ��
            ind_away = find(dis_min<0.8);
            ind_away = ind_away(randperm(length(ind_away)));
            num_away = min(length(ind_away),5*num_near);
            ind_away = ind_away(1:num_away);
            new.x_label = Data(obj.x_label.x(:,[ind_near,ind_away]));
            %��b��ֵ������ȥ����Ϊb�ǹ��õģ������ֻ��һ��b
            new.z_label = Data([bsxfun(@plus,encoder_err(:,ind_near),b3), bsxfun(@times,b3,ones(size(encoder_err(:,ind_away)),obj.arrayType))]);
            
            new.x_test = Data([]);
            new.z_test = Data([]);
        end
    end
    
end

