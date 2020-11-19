function crude_align( ref_logVAD, ref_Nsamples, deg_logVAD, ...
    deg_Nsamples, Utt_id)

global Downsample
global Nutterances Largest_uttsize Nsurf_samples Crude_DelayEst
global Crude_DelayConf UttSearch_Start UttSearch_End Utt_DelayEst
global Utt_Delay Utt_DelayConf Utt_Start Utt_End
global MAXNUTTERANCES WHOLE_SIGNAL
global pesq_mos subj_mos cond_nr

if(Utt_id == WHOLE_SIGNAL)
    nr = floor( ref_Nsamples/ Downsample);
    nd = floor( deg_Nsamples / Downsample);
    startr =1;
    startd = 1;
elseif Utt_id == MAXNUTTERANCES
    startr = UttSearch_Start(MAXNUTTERANCES);
    startd = startr+ Utt_DelayEst(MAXNUTTERANCES) / Downsample;
    if( startd < 0)
        startr = 1 - Utt_DelayEst(MAXNUTTERANCES) / Downsample;
        startd = 1;
    end
    
    nr = UttSearch_End(MAXNUTTERANCES) - startr;
    nd=nr;
    
    if(startd + nd > floor(deg_Nsamples /Downsample))
        nd = floor(deg_Nsamples/ Downsample)- startd;
    end
    
    
else
    startr = UttSearch_Start(Utt_id);
    startd = startr + Crude_DelayEst / Downsample;
    
    if(startd < 0)
        startr = 1- Crude_DelayEst /Downsample;
        startd = 1;
    end
    
    nr = UttSearch_End(Utt_id) - startr;
    nd=nr;
    if(startd + nd > floor(deg_Nsamples/ Downsample) +1)
        nd = floor(deg_Nsamples / Downsample) - startd + 1;
    end
end

max_Y = 0.0;
I_max_Y = nr;
if((nr > 1)&&(nd > 1))
    Y = FFTNXCorr(ref_logVAD, startr, nr, deg_logVAD, startd, nd);
    [max_Y, I_max_Y] = max(Y);
    if( max_Y <=0)
        max_Y=0;
        I_max_Y = nr;
    end
end



if(Utt_id == WHOLE_SIGNAL)
    Crude_DelayEst = (I_max_Y - nr)* Downsample;
    Crude_DelayConf = 0.0;
    
    
elseif(Utt_id == MAXNUTTERANCES)
    Utt_Delay(MAXNUTTERANCES) = (I_max_Y - nr)* Downsample + ...
        Utt_DelayEst(MAXNUTTERANCES);
    
        
        
else
    
    Utt_DelayEst(Utt_id)= (I_max_Y - nr)* Downsample+ ...
        Crude_DelayEst;
end








