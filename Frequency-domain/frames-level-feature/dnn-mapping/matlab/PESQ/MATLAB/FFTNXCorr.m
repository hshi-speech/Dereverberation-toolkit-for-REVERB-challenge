function Y = FFTNXCorr( ref_VAD, startr, nr, deg_VAD, startd, nd)
%this function has other simple implementations, current implementation is
%consistent with the C version

x1 = ref_VAD( startr: startr + nr -1);  

x2 = deg_VAD( startd: startd + nd -1);
x1 = fliplr( x1);
Y =conv(x2, x1);


