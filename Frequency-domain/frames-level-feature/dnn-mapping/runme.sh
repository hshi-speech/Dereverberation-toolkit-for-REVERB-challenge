#!/bin/bash

step=1

WORKSPACE="workspace"
mkdir $WORKSPACE

TR_SPEECH_DIR="lists/tr_clean.lst"
TR_NOISE_DIR="lists/tr_reverb.lst"

VA_SPEECH_DIR="lists/va_clean.lst"
VA_NOISE_DIR="lists/va_reverb.lst"
# VA1_SPEECH_DIR="lists/REVERB_dt_far_room1_clean.lst"
# VA1_NOISE_DIR="lists/REVERB_dt_far_room1.lst"
# VA2_SPEECH_DIR="lists/REVERB_dt_far_room2_clean.lst"
# VA2_NOISE_DIR="lists/REVERB_dt_far_room2.lst"
# VA3_SPEECH_DIR="lists/REVERB_dt_far_room3_clean.lst"
# VA3_NOISE_DIR="lists/REVERB_dt_far_room3.lst"
# VA4_SPEECH_DIR="lists/REVERB_dt_near_room1_clean.lst"
# VA4_NOISE_DIR="lists/REVERB_dt_near_room1.lst"
# VA5_SPEECH_DIR="lists/REVERB_dt_near_room2_clean.lst"
# VA5_NOISE_DIR="lists/REVERB_dt_near_room2.lst"
# VA6_SPEECH_DIR="lists/REVERB_dt_near_room3_clean.lst"
# VA6_NOISE_DIR="lists/REVERB_dt_near_room3.lst"

TE1_SPEECH_DIR="lists/REVERB_et_far_room1_clean.lst"
TE1_NOISE_DIR="lists/REVERB_et_far_room1.lst"
TE2_SPEECH_DIR="lists/REVERB_et_far_room2_clean.lst"
TE2_NOISE_DIR="lists/REVERB_et_far_room2.lst"
TE3_SPEECH_DIR="lists/REVERB_et_far_room3_clean.lst"
TE3_NOISE_DIR="lists/REVERB_et_far_room3.lst"
TE4_SPEECH_DIR="lists/REVERB_et_near_room1_clean.lst"
TE4_NOISE_DIR="lists/REVERB_et_near_room1.lst"
TE5_SPEECH_DIR="lists/REVERB_et_near_room2_clean.lst"
TE5_NOISE_DIR="lists/REVERB_et_near_room2.lst"
TE6_SPEECH_DIR="lists/REVERB_et_near_room3_clean.lst"
TE6_NOISE_DIR="lists/REVERB_et_near_room3.lst"
TE7_SPEECH_DIR="lists/REVERB_Real_et_far.lst"
TE7_NOISE_DIR="lists/REVERB_Real_et_far.lst"
TE8_SPEECH_DIR="lists/REVERB_Real_et_near.lst"
TE8_NOISE_DIR="lists/REVERB_Real_et_near.lst"


N_CONCAT=7
N_HOP=3

ITERATION=1000000

model_name=dnn

#############################################################################################################
################################    Step 1: Training  Data Processing    ####################################
#############################################################################################################

if [ $step -le 1 ]; then

  # Calculate mixture features.
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --dir_name=REVERB_tr_cut
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$VA_SPEECH_DIR --noise_dir=$VA_NOISE_DIR --data_type=validation --dir_name=REVERB_dt

  # Pack features.
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_tr_cut
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=validation --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_dt

  # Compute scaler. 
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --dir_name=REVERB_tr_cut
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=validation --dir_name=REVERB_dt

fi

#############################################################################################################
######################################    Step 2: Training  Model    ########################################
#############################################################################################################

if [ $step -le 2 ]; then

  # Train. 
  LEARNING_RATE=1e-4
  dropout=0.1
  
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py train \
                         --model_name=$model_name \
                         --workspace=$WORKSPACE \
                         --lr=$LEARNING_RATE \
                         --tr_dir_name=REVERB_tr_cut \
                         --va_dir_name=REVERB_dt \
                         --iteration=$ITERATION \
                         --dropout=$dropout

fi

#############################################################################################################
#################################    Step 3: Testing Data Processing    #####################################
#############################################################################################################

if [ $step -le 3 ]; then

#   # Create mixture csv.
#   python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

  # Calculate mixture features.
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE1_SPEECH_DIR --noise_dir=$TE1_NOISE_DIR --data_type=test --dir_name=REVERB_et_far_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE2_SPEECH_DIR --noise_dir=$TE2_NOISE_DIR --data_type=test --dir_name=REVERB_et_far_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE3_SPEECH_DIR --noise_dir=$TE3_NOISE_DIR --data_type=test --dir_name=REVERB_et_far_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE4_SPEECH_DIR --noise_dir=$TE4_NOISE_DIR --data_type=test --dir_name=REVERB_et_near_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE5_SPEECH_DIR --noise_dir=$TE5_NOISE_DIR --data_type=test --dir_name=REVERB_et_near_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE6_SPEECH_DIR --noise_dir=$TE6_NOISE_DIR --data_type=test --dir_name=REVERB_et_near_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE7_SPEECH_DIR --noise_dir=$TE7_NOISE_DIR --data_type=test --dir_name=REVERB_Real_et_far
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE8_SPEECH_DIR --noise_dir=$TE8_NOISE_DIR --data_type=test --dir_name=REVERB_Real_et_near

  # Pack features. 
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_far_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_far_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_far_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_near_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_near_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_et_near_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_Real_et_far
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --n_concat=$N_CONCAT --n_hop=$N_HOP --dir_name=REVERB_Real_et_near
  
  # Compute scaler. 
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_far_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_far_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_far_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_near_room1
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_near_room2
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_et_near_room3
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_Real_et_far
  /Work19/2018/shihao/sednn-env/bin/python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=test --dir_name=REVERB_Real_et_near

fi

#############################################################################################################
#########################################    Step 4: Testing    #############################################
#############################################################################################################

if [ $step -le 4 ]; then

  # Inference, enhanced wavs will be created. 
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_far_room1 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_far_room2 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_far_room3 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_near_room1 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_near_room2 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_et_near_room3 --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_Real_et_far --model_name=$model_name
  CUDA_VISIBLE_DEVICES=0 /Work19/2018/shihao/sednn-env/bin/python main_dnn.py inference --workspace=$WORKSPACE --n_concat=$N_CONCAT --iteration=$ITERATION --dir_name=REVERB_Real_et_near --model_name=$model_name

fi

#############################################################################################################
#######################################    Step 5: Evaluation    ############################################
#############################################################################################################

if [ $step -le 5 ]; then

    # Calculate PESQ, SRMR and STOI of all enhanced speech. 
    # python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

    echo "Start Evaluation."
    # Should write the paths in evaluation files. 
    models=${name}
    dirs=workspace/enh_wavs/test/
    mKinds=mapping

    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_far_room1';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_far_room2';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_far_room3';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_near_room1';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_near_room2';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_et_near_room3';real_data='0';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_Real_et_far';real_data='1';mKinds='$mKinds';evaluation;quit"
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';files='REVERB_Real_et_near';real_data='1';mKinds='$mKinds';evaluation;quit"

    # Calculate overall stats. 
    # python evaluate.py get_stats

fi

