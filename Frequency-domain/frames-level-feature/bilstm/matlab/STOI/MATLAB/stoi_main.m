function stoi_main(models, dirs, files, mKinds)

	if ~exist(['./matlab/STOI/MODELS/'  models '/' mKinds '/'])
		mkdir(['./matlab/STOI/MODELS/'  models '/' mKinds '/']);
	end

	fid_whole_Name = ['./stoi_results_latest.txt'];
	fid_whole =fopen(fid_whole_Name,'a+');

	fprintf(fid_whole, '%s\n', char(mKinds))

	fileName = ['./matlab/STOI/MODELS/'  models '/'  char(mKinds)   '/' char(files) '.txt'];
	fid=fopen(fileName,'w');

	clnName = ['./matlab/STOI/MODELS/REVERB_et/' char(files) '_cln.txt'];
	lines = importdata(clnName);

	[m, n] = size(lines)

	sum = 0;

	for i = 1 : m;
	 
		cleanSplit = strsplit(char(lines{i}));
		revebPath = [dirs  '/'  char(files) '/' char(cleanSplit{1}) '.enh.wav'];
			
		[wav_clean, fs] = audioread(cleanSplit{2});
		[wav_enhan, fs] = audioread(revebPath);
			
		len = min(length(wav_clean), length(wav_enhan));
			
		wav_clean = wav_clean(1:len);
		wav_enhan = wav_enhan(1:len);

		stoiscore = stoi(wav_clean, wav_enhan, fs);
		sum = sum + stoiscore;
		fprintf(fid, '%s %f\n', cleanSplit{1}, stoiscore);
		  
	end

	fprintf(fid, 'The averange of whole %s is %f\n', char(files), sum/m);
	fclose(fid);
	fprintf(fid_whole, '%s : %f\n', char(files), sum/m);
	fclose(fid_whole);

end