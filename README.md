Speech-dereverberation-tools  
====

```
[-] This tool is for deep learning-based monaural speech dereverberation.
[-] We want to make an open source tool for the REVERB CHALLANGE dataset. 

[-] If you find any bugs in this tool, please contact me! Thank you!
    My e-mail: hshi.cca@gmail.com.
```

## Models included in this tool:
Frequency-domain
- [x] DNN (Frames-level Loss)
- [ ] LSTM (Frames-level Loss)
- [ ] LSTM (Utterance-level Loss)
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
---
https://github.com/yongxuUSTC/sednn  
---
https://github.com/snsun/pit-speech-separation  
---
https://github.com/kaituoxu/Conv-TasNet  
---


