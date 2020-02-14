# VBHMM x-vectors Diarization (aka *VBx*)

[Diarization recipe](https://speech.fit.vutbr.cz/software/vbhmm-x-vectors-diarization) for [The Second DIHARD Diarization Challenge](https://coml.lscp.ens.fr/dihard/index.html) by Brno University of Technology. \
The recipe consists of 
- computing fbank features
- computing x-vectors
- doing Agglomerative Hierarchical Clustering on x-vectors as a first step to produce an initialization
- apply Variational Bayes HMM over x-vectors to produce the diarization output
- score the diarization output

More details about the full recipe in\
F. Landini, S. Wang, M. Diez, L. Burget et al.: *BUT System for the Second DIHARD Speech Diarization Challenge*, ICASSP 2020\
or [*BUT System Description for DIHARD Speech Diarization Challenge 2019*](https://arxiv.org/abs/1910.08847)

A more thorough analysis of the diarization approach is presented in\
M. Diez, L. Burget, F. Landini, S. Wang, J. Černocký: *Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech diarization challenge*, ICASSP 2020



## Getting Started

To use this recipe, first clone the repository.
```
git clone https://github.com/BUTSpeechFIT/VBx.git
```

### Prerequisites
```
Python >= 3.7
Pytorch >= 1.3.1
soundfile >= 0.10.3
```

### Usage
To run the recipe, execute `run_recipe.sh` followed by `all` to run all steps or `features`, `xvectors`, `VBx`, `score` for only computing fbank features, computing xvectors, running VBx diarization or scoring, respectively.

The script is prepared to run on the development and evaluation sets of [The Second DIHARD Diarization Challenge](https://coml.lscp.ens.fr/dihard/index.html) [track 1](http://dihard.ldc.upenn.edu/competitions/73). You need to provide the directory with the recordings in flac format and the directory for the speech activity detection labels as provided by the organizers:
```
0.130	4.010	speech
4.790	5.750	speech
...
```



### Resources
This recipe makes use of an x-vector extractor model which was trained on data from the VoxCeleb corpora and using the Kaldi toolkit.\
A. Nagrani, J. S. Chung, A. Zisserman: *VoxCeleb: a large-scale speaker identification dataset*\
J. S. Chung, A. Nagrani, A. Zisserman: *VoxCeleb2: Deep Speaker Recognition*\
D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann, P. Motlicek, Y. Qian, P. Schwarz et al.: *The Kaldi speech recognition toolkit*


The x-vector extractor file has been compressed and separated into two files to be able to upload it. To recover it, first unsplit it:
`
zip -s 0 split_xvector_extractor.txt.zip --out unsplit_xvector_extractor.txt.zip
`
and then unzip it:
`
unzip unsplit_xvector_extractor.txt.zip
`

The recipe also uses two probabilistic linear discriminant analysis (PLDA) models, one trained on VoxCeleb data and another on the DIHARD development set. In case of using any of these PLDA models, also cite the corresponding publications.\
A. Nagrani, J. S. Chung, A. Zisserman: *VoxCeleb: a large-scale speaker identification dataset*\
J. S. Chung, A. Nagrani, A. Zisserman: *VoxCeleb2: Deep Speaker Recognition*\
N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy, M. Liberman: 
*The Second DIHARD Diarization Challenge: Dataset, task, and baselines*


### Citations
In case of using the software please cite:\
F. Landini, S. Wang, M. Diez, L. Burget et al.: *BUT System for the Second DIHARD Speech Diarization Challenge*, ICASSP 2020

M. Diez, L. Burget, F. Landini, S. Wang, J. Černocký: *Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech diarization challenge*, ICASSP 2020

A. Nagrani, J. S. Chung, A. Zisserman: *VoxCeleb: a large-scale speaker identification dataset*

J. S. Chung, A. Nagrani, A. Zisserman: *VoxCeleb2: Deep Speaker Recognition*

N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy, M. Liberman: *The Second DIHARD Diarization Challenge: Dataset, task, and baselines*


### Results
The diarization error rates (DER) obtained with this recipe for the development and evaluation are:\
Development 17.87\
Evaluation 18.31

In our submission to the challenge we used the weighted prediction error method (see papers below). Processing the recordings with this method and this recipe we obtained:\
Development 17.64\
Evaluation 18.09

All scores were obtained using the [scoring tool provided by the organizers](https://github.com/nryant/dscore). Due to some non-deterministic parts of the recipe, the obtained diarization outputs can slightly change from run to run.

T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, and B. H. Juang: *Speech dereverberation based on variance-normalized delayed linear prediction*\
and\
L. Drude, J. Heymann, C. Boeddeker, and R. Haeb-Umbach: *NARA-WPE: A Python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing*



## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


## Contact
If you have any comment or question, please contact landini@fit.vutbr.cz or mireia@fit.vutbr.cz