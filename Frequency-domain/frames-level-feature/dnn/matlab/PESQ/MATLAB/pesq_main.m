function pesq_main(models, dirs, files, mKinds)

	if ~exist(['./matlab/PESQ/MODELS/'  models '/' mKinds '/'])
		mkdir(['./matlab/PESQ/MODELS/'  models '/' mKinds '/']);
	end

	fid_whole_Name = ['./pesq_results_latest.txt']
	fid_whole =fopen(fid_whole_Name,'a+');

	fprintf(fid_whole, '%s\n', char(mKinds));
   
	fileName = ['./matlab/PESQ/MODELS/'  models '/'  char(mKinds) '/' char(files) '.txt'];
	fid=fopen(fileName,'w');

	clnName = ['./matlab/PESQ/MODELS/REVERB_et/' char(files) '_cln.txt'];
	lines = importdata(clnName);

	[m, n] = size(lines);

	sum = 0;

	for i = 1 : m;
		 
		cleanSplit = strsplit(char(lines{i}));
		revebPath = [dirs '/' char(files) '/' char(cleanSplit{1}) '.enh.wav'];

		pesqscore = pesq(cleanSplit{2}, revebPath );
		sum = sum + pesqscore;
		fprintf(fid, '%s %f\n', cleanSplit{1}, pesqscore);
			  
	end

	fprintf(fid, 'The averange of whole %s is %f\n', char(files), sum/m);
	fclose(fid);
			
	fprintf(fid_whole, '%s : %f\n', char(files), sum/m);
	
	fclose(fid_whole);

end
