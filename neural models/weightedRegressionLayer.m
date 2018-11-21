classdef weightedRegressionLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties

        % Layer properties go here
    end
 
    methods
        function layer = weightedRegressionLayer()           
            % (Optional) Create a myRegressionLayer

            % Layer constructor function goes here
            
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T
            %
            % Inputs:
            %         layer - Output layer
            %         Y     每 Predictions made by network
            %         T     每 Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here
             % Calculate MAE
             N = size(Y,4);
             R = size(Y,3)-1;
             weightedSquareError = sum((Y(:,:,1:end-1,:)-T(:,:,1:end-1,:)).^2,3).*T(:,:,end,:)/R;
             loss = sum(weightedSquareError)/N;
            
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function
            %
            % Inputs:
            %         layer - Output layer
            %         Y     每 Predictions made by network
            %         T     每 Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

            % Layer backward loss function goes here
            N = size(Y,4);
            R = size(Y,3)-1;
            dLdY = bsxfun(@times,2*(Y(:,:,1:end-1,:)-T(:,:,1:end-1,:)),T(:,:,end,:))/N/R;
            dLdY = cat(3,dLdY,zeros(1,1,1,N,'like',dLdY));
            
        end
    end
end