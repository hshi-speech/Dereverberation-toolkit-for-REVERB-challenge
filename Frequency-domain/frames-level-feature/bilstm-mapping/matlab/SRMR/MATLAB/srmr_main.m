function srmr_main(models, dirs, files, mKinds)

	if ~exist(['./matlab/SRMR/MODELS/'  models '/' mKinds '/'])
		mkdir(['./matlab/SRMR/MODELS/'  models '/' mKinds '/']);
	end

	fid_whole_Name = ['./srmr_results_latest.txt'];
	fid_whole =fopen(fid_whole_Name,'a+');


	fprintf(fid_whole, '%s\n', char(mKinds));


	   
	fileName = ['./matlab/SRMR/MODELS/'  models '/'  char(mKinds) '/' char(files) '.txt'];
	fid=fopen(fileName,'w');

	clnName = ['./matlab/SRMR/MODELS/REVERB_et/' char(files) '.txt'];
	lines = importdata(clnName);

	[m, n] = size(lines);

	sum = 0;

	for i = 1 : m;
	 
		cleanSplit = strsplit(char(lines{i}));
		revebPath = [dirs '/' char(files) '/' char(cleanSplit{1}) '.enh.wav'];

		srmrscore = SRMR(revebPath );
		sum = sum + srmrscore;
		fprintf(fid, '%s %f\n', cleanSplit{1}, srmrscore);
		  
	end

	fprintf(fid, 'The averange of whole %s is %f\n', char(files), sum/m);
	fclose(fid);
	fprintf(fid_whole, '%s : %f\n', char(files), sum/m);
	fclose(fid_whole);

end