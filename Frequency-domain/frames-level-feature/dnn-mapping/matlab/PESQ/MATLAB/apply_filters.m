function mod_data = apply_filters( data, Nsamples)


global InIIR_Hsos InIIR_Nsos DATAPADDING_MSECS Fs



sosMatrix= zeros( InIIR_Nsos, 6);
sosMatrix( :, 4)= 1;

sosMatrix(:, 1:3) = InIIR_Hsos( :, 1:3);
sosMatrix(:, 5:6) = InIIR_Hsos( :, 4:5);



iirdf2= dfilt.df2sos( sosMatrix);

mod_data = filter( iirdf2, data);