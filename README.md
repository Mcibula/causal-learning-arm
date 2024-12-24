# Learning Low-Level Causal Relations using a Simulated Robotic Arm

This is the code for the paper [_Learning Low-Level Causal Relations Using a Simulated Robotic Arm_](https://arxiv.org/abs/2410.07751)
presented at the International Conference on Artificial Neural Networks 2024.

## Structure
The contents of this repository are organized as follows:
```
.
├── configs                                     // Config files for experiments in myGym environments
│   ├── cfg_kuka_kinematics.json                // Experiment 1
│   └── cfg_kukaM_push.json                     // Experiment 2
├── scripts
│   ├── data_gen                                // Data generation and post-processing
│   │   ├── kinematics_data_kuka.ipynb          // Post-processing of generated data for Exp. 1
│   │   ├── magnetic_data.ipynb                 // Post-processing of generated data for Exp. 2
│   │   └── run_mygym.py                        // Runs experiments in the simulation
│   ├── explain
│   │   └── explain_shap_magnetic.ipynb         // Knowledge extraction implementation for Exp. 2
│   └── models                                  // Training and experimentation with the selected models
│       ├── kinematics_inv_kuka.ipynb           // Experiments with inverse models for Exp. 1
│       ├── kinematics_models_kuka.ipynb        // Models for Exp. 1
│       ├── magnetic_models.ipynb               // Models for Exp. 2
│       └── test_fk_kuka.ipynb                  // Mental simulation testing for Exp. 1
├── LICENSE
├── README.md
└── requirements.txt
```

## Dependencies
The project is implemented mainly as a series of Jupyter Notebooks. 
Running the simulation experiments requires the [myGym toolbox](https://github.com/incognite-lab/myGym/tree/master) of version 3.7. 
The neural models are implemented in Keras 2 with Tensorflow back-end.

The myGym toolbox must be built from the source as per the [instructions](https://github.com/incognite-lab/myGym/tree/master?tab=readme-ov-file#installation).
Other dependencies can be installed by running:
```bash
pip install -r requirements.txt
```

## Citation
If you find this work helpful in your research, please consider citing:
```bibtex
@InProceedings{causal_learning_icann,
  author    = {Cibula, Miroslav and Kerzel, Matthias and Farka{\v{s}}, Igor},
  booktitle = {Artificial Neural Networks and Machine Learning -- ICANN 2024},
  title     = {Learning Low-Level Causal Relations Using a Simulated Robotic Arm},
  year      = {2024},
  editor    = {Wand, Michael and Malinovsk{\'a}, Krist{\'i}na and Schmidhuber, J{\"u}rgen and Tetko, Igor V.},
  pages     = {285--298},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-72359-9_21},
}
```
