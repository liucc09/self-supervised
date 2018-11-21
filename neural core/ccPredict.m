%{1}最后一层的输出
%{2}编码层的输出
%{3}第一层的输出
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

