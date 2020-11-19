
files = {'REVERB_et_far_room1', 'REVERB_et_far_room2', 'REVERB_et_far_room3', 'REVERB_et_near_room1', 'REVERB_et_near_room2', 'REVERB_et_near_room3'}
models = 'original-reverb'
dirs = '/Work18/2018/shihao/Dereverberation/reverb2014/mapping-utterance-blstm/data/separated/StandPsmPIT_BLSTM_3_496_ReLU_def/'



for flines  = 1 : length(files)
   
    fileName = ['../MODELS/'  models '/' char(files{flines}) '.txt']
    fid=fopen(fileName,'w');

    clnName = ['../MODELS/REVERB_et/' char(files{flines}) '_cln.txt']
    lines = importdata(clnName);
    revName = ['../MODELS/REVERB_et/' char(files{flines}) '.txt']
    lines_reverb = importdata(revName);

    [m, n] = size(lines)

    sum = 0;

    for i = 1 : m
 
        cleanSplit = strsplit(char(lines{i}));
        revebSplit = strsplit(char(lines_reverb{i}));

        if(cleanSplit{1} == revebSplit{1})

           pesqscore = pesq(cleanSplit{2}, revebSplit{2});
           sum = sum + pesqscore;
           fprintf(fid, '%s %f\n', cleanSplit{1}, pesqscore)
           
        end       
    end

    fprintf(fid, 'The averange of whole %s is %f\n', char(files{flines}), sum/m)

    fclose(fid);

end
