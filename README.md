# VBHMM x-vectors Diarization (aka VBx)

Diarization recipe for VoxConverse by Brno University of Technology. \
It contains the main modules of the system [ranked second](https://competitions.codalab.org/competitions/26357#results) in [VoxSRC Challenge](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2020.html) Track 4.\
The shared recipe consists of 
- energy-based VAD
- computing x-vectors
- doing Agglomerative Hierarchical Clustering (AHC) on x-vectors as a first step to produce an initialization
- applying Variational Bayes HMM over x-vectors to produce the diarization output (VBx)
- extracting one x-vector per speaker given VBx's output
- applying AHC to recluster speakers
- applying an heuristic to handle overlapped speech
- using the second most likely speaker given by VBx to handle overlapped speech
- scoring the diarization output of the different intermediate and final steps

Note that for overlap handling the heuristic performs better than using the second most likely speaker from VBx and this was the approach used in our best submitted system. Since both approaches make use of an external tool to obtain the overlapped speech labels, we simply share such labels for both development and evaluation sets.
Similarly, we do not include all the modules that were used for our final voice activity detection (VAD) system but we do share the final labels used for the challenge both for development and evaluation sets. However, we do share the energy-based VAD module as it is simple and provides reasonable results.


More details about the full recipe in\
F. Landini, O. Glembek, P. Matějka, J. Rohdin, L. Burget, M. Diez, A. Silnova: [Analysis of the BUT Diarization System for VoxConverse Challenge](https://arxiv.org/abs/2010.11718)



## Usage
To run the recipe, execute the run script with the corresponding parameters, namely which set to use (dev/eval), the instruction to apply, the output directory, waveforms directory and RTTM directory (where data were downloaded). Please refer to the scripts for more details.


## Getting started
We recommend to create [anaconda](https://www.anaconda.com/) environment
```bash
conda create -n VBx python=3.6
conda activate VBx
```
Clone the repository
```bash
git clone --branch v1.1_VoxConverse2020 https://github.com/BUTSpeechFIT/VBx.git
```
Install the package
```bash
pip install -e .
```
Initialize submodule `dscore`:
```bash
git submodule init
git submodule update
```
Run the example
```bash
./run_example.sh
```
The output (last few lines) should look like this
```
File               DER    JER    B3-Precision    B3-Recall    B3-F1    GKT(ref, sys)    GKT(sys, ref)    H(ref|sys)    H(sys|ref)    MI    NMI
---------------  -----  -----  --------------  -----------  -------  ---------------  ---------------  ------------  ------------  ----  -----
ES2005a           7.06  29.99            0.65         0.78     0.71             0.71             0.56          1.14          0.59  1.72   0.67
*** OVERALL ***   7.06  29.99            0.65         0.78     0.71             0.71             0.56          1.14          0.59  1.72   0.67
```


## Citations
In case of using the software please cite:\
F. Landini, O. Glembek, P. Matějka, J. Rohdin, L. Burget, M. Diez, A. Silnova: [Analysis of the BUT Diarization System for VoxConverse Challenge](https://arxiv.org/abs/2010.11718)



## Results
We present here the results of our system for the two VADs (energy-based one and the final one described in the paper) and the two overlap handling approaches. Note that small differences with respect to the numbers reported in the paper are due to randomness in the pipeline which in this recipe were fixed for the sake of replicating results.
We also show the comparison between using the original recordings and the enhanced ones (as used in the paper).

```
Original recordings

Energy VAD
--------------------------------------------------
                  DER   Miss  FA    SpkE    JER
VBx               6.88  3.27  2.28  1.33    21.50
+ Reclustering    6.70  3.27  2.28  1.14    21.57
  + OV heuristic  6.40  2.52  2.57  1.31    21.43
  + OV label2nd   6.44  2.53  2.57  1.34    21.65

Final VAD
--------------------------------------------------
                  DER   Miss  FA    SpkE    JER
VBx               4.41  3.10  0.47  0.84    19.61
+ Reclustering    4.34  3.10  0.47  0.77    19.65
  + OV heuristic  4.04  2.34  0.76  0.94    19.57
  + OV label2nd   4.08  2.36  0.76  0.96    19.84

```

```
Enhanced recordings

Energy VAD
--------------------------------------------------
                  DER   Miss  FA    SpkE    JER
VBx               6.22  3.21  1.69  1.32    20.65
+ Reclustering    5.83  3.21  1.69  0.92    20.67
  + OV heuristic  5.53  2.46  1.98  1.08    20.60
  + OV label2nd   5.57  2.49  1.97  1.11    20.85

Final VAD
--------------------------------------------------
                  DER   Miss  FA    SpkE    JER
VBx               4.33  3.10  0.47  0.77    19.74
+ Reclustering    4.30  3.10  0.47  0.74    19.78
  + OV heuristic  4.00  2.34  0.75  0.91    19.77
  + OV label2nd   4.01  2.36  0.74  0.92    19.96

```


## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



## Contact
If you have any comment or question, please contact landini@fit.vutbr.cz
