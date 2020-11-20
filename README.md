Dereverberation toolkit for the REVERB challenge  
====

```
[-] This tool is for deep learning-based monaural speech dereverberation.
[-] We want to make an open source tool for the REVERB CHALLANGE dataset. 

[-] If you find any bugs in this tool, please contact me! Thank you!
    My e-mail: hshi.cca@gmail.com.
```

## Models included in this tool
Frequency-domain
- [x] DNN (Frames-level Loss, mapping)
- [x] LSTM (Frames-level Loss, mapping)

## Evaluation measures
#### PESQ Results
| System | Far room 1 | Far room 2 | Far room 3 | Near room 1 | Near room 2 | Near room 3 | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Noisy             |     2.59   |    1.99   |    1.87   |    3.11    |    2.39    |    2.27    |
| DNN (mapping)     |     2.80   |    2.40   |    2.29   |    3.09    |    2.72    |    2.63    |
| Bi-LSTM (mapping) |     2.89   |    2.47   |    2.36   |    3.17    |    2.82    |    2.70    |

#### SRMR Results
| System | Far room 1 | Far room 2 | Far room 3 | Near room 1 | Near room 2 | Near room 3 | real far | real near |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Noisy             |    4.59   |    2.97   |    2.73   |    4.50    |    3.74    |    3.57    |   3.19   |   3.17    |
| DNN (mapping)     |    5.62   |    4.72   |    4.03   |    5.88    |    5.57    |    5.34    |   5.58   |   5.35    |
| Bi-LSTM (mapping) |    5.50   |    4.82   |    4.18   |    5.58    |    5.32    |    5.15    |   5.89   |   5.75    |

#### STOI Results
| System | Far room 1 | Far room 2 | Far room 3 | Near room 1 | Near room 2 | Near room 3 | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Noisy             |   84.69%  |   78.20%  |   71.31%  |   95.18%   |   92.32%   |   89.38%   |
| DNN (mapping)     |   87.93%  |   84.72%  |   82.46%  |   92.23%   |   90.99%   |   89.48%   |
| Bi-LSTM (mapping) |   88.65%  |   85.00%  |   82.83%  |   93.07%   |   91.90%   |   90.14%   |



## Todo
Frequency-domain
- [ ] DNN (Frames-level Loss, masking)
- [ ] DNN (Frames-level Loss, multi-target learning)
- [ ] LSTM (Frames-level Loss, masking)
- [ ] LSTM (Frames-level Loss, multi-target learning)
- [ ] LSTM (Utterance-level Loss, mapping)
- [ ] LSTM (Utterance-level Loss, masking)
- [ ] LSTM (Utterance-level Loss, multi-target learning)
- [ ] CNN (Frames-level Loss)
- [ ] CNN (Utterance-level Loss)
- [ ] U-net (Frames-level Loss)
- [ ] U-net (Utterance-level Loss)  

Time-domain
- [ ] SEGAN
- [ ] FCN 
- [ ] Conv-Tasnet 

## Data description
The data provided consists of a training set, a development test set, and a (final) evaluation test set.  

+ SimData: utterances from the WSJCAM0 corpus [1], which are convolved by room impulse responses (RIRs) measured in different rooms. Recorded background noise is added to the reverberated test data at a fixed signal-to-noise ratio (SNR).  

+ RealData: utterances from the MC-WSJ-AV corpus [2], which consists of utterances recorded in a noisy and reverberant room.  

<b>References:</b>  
[1] T. Robinson, J. Fransen, D. Pye and J. Foote and S. Renals, "Wsjcam0: A British English Speech Corpus For Large Vocabulary Continuous Speech Recognition", In Proc. ICASSP 95, pp.81--84, 1995  
[2] M. Lincoln, I. McCowan, J. Vepa and H.K. Maganti, "The multi-channel Wall Street Journal audio visual corpus (MC-WSJ-AV): Specification and initial experiments", IEEE Workshop on Automatic Speech Recognition and Understanding, 2005  

## Acknowledge
I modified my codes based on below github:  
```
1. https://github.com/yongxuUSTC/sednn
2. https://github.com/snsun/pit-speech-separation
3. https://github.com/kaituoxu/Conv-TasNet
```

