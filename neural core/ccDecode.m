function output = ccDecode(input, encoder)
    
    x = ccPreprocess(input, encoder.dataHeap.wm,encoder.dataHeap.bm);
    
    xs = ccForward(x,encoder.net,encoder.net.layers(encoder.net.encoderInd+1:end));

    output = ccPostprocess(xs{end},  encoder.dataHeap.wo, encoder.dataHeap.bo);    
    
end

