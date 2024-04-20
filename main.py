from glob import glob
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

datasets_dir = "TurEV-DB-master\\Sound Source\\"
input_folder_list = glob(datasets_dir + "*")

max_len = 250

audio_inputs = []
audio_targets = []
target_id = -1
for input_folder in input_folder_list:
    waw_files = glob(input_folder + "\\*")
    target_id += 1
    for waw_file in waw_files:
        sampling_freq, audio = wavfile.read(waw_file)
        mfcc_feat = mfcc(audio, sampling_freq, nfft=2048)
        number_of_pad = max_len - mfcc_feat.shape[0]
        paddings = np.zeros((number_of_pad, mfcc_feat.shape[1]))
        mfcc_feat = np.concatenate((mfcc_feat, paddings))
        audio_inputs.append(mfcc_feat)
        audio_targets.append(target_id)

audio_inputs = np.array(audio_inputs)
audio_targets = np.array(audio_targets)

audio_inputs = np.reshape(audio_inputs, (audio_inputs.shape[0], -1))

x_train, x_test, y_train, y_test = train_test_split(audio_inputs, audio_targets, test_size=0.25)

algorithm_array = ["SAMME", "SAMME.R"]
learning_rate_array = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

best_score = 0.0
best_algorithm = ""
best_learning_rate = 0.0

for algorithm in algorithm_array:
    for learning_rate in learning_rate_array:
        adaboost_clf = AdaBoostClassifier(n_estimators=100,
                                          algorithm=algorithm,
                                          learning_rate=learning_rate)
        adaboost_clf.fit(x_train, y_train)
        score = adaboost_clf.score(x_test, y_test)
        print("Score: ", score)

        if score > best_score:
            best_score = score
            best_algorithm = algorithm
            best_learning_rate = learning_rate

print("Best Validation Score: {:.2f}".format(best_score))
print("best algorithm", best_algorithm)
print("best learning_rate", best_learning_rate)

