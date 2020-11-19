function data_cut_tt(enh_dir, save_dir, mKinds)

	lst_wav = ['./data_cut/tt/' mKinds '.lst'];

	lines_lst_wav = importdata(lst_wav);
	[mr, nr] = size(lines_lst_wav);

        if ~exist([save_dir '/' mKinds '/'])
            mkdir([save_dir '/' mKinds '/'])
	end



	for j = 1 : 1 : mr

		pathSplit = strsplit(lines_lst_wav{j});

		name = pathSplit{1};
		pathRvb = pathSplit{2};

                nameChange = strrep(name, '_ch1', '')
 
		[wavEnh, fs]= audioread([enh_dir '/' mKinds '/' nameChange '.enh.wav']);
		[wavRvb, fs]= audioread(pathRvb);
		
		if(length(wavEnh)>length(wavRvb))
			wavEnh = wavEnh(1:length(wavRvb));
		elseif(length(wavEnh)<length(wavRvb))
			for i = length(wavEnh) : 1 : length(wavRvb);
				wavEnh(i,1)=0;
			end
		end

		audiowrite([save_dir '/' mKinds '/' name '.wav'], wavEnh, fs);

	end
end
