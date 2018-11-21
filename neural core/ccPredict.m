%{1}���һ������
%{2}���������
%{3}��һ������
function [output,xs] = ccPredict(input, encoder, scale)
    
    if scale 
        x = ccPreprocess(input,encoder.dataHeap.wi,encoder.dataHeap.bi);
    else
        x = input;
    end
    
    xs = ccForward(x,encoder.net,encoder.net.layers);

    if scale
        output = ccPostprocess(xs{end},  encoder.dataHeap.wo, encoder.dataHeap.bo);
    else
        output = xs{end};
    end
  
end

