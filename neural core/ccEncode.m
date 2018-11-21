function output = ccEncode(input, encoder)
    
    x = ccPreprocess(input,encoder.dataHeap.wi,encoder.dataHeap.bi);
    
    xs = ccForward(x,encoder.net,encoder.net.layers(1:encoder.net.encoderInd));

    output = ccPostprocess(xs{end},  encoder.dataHeap.wm, encoder.dataHeap.bm);    
    
end

