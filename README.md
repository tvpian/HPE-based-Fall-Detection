# Fall Detection 
Striking the Balance: Human Pose Estimation based Optimal Fall Recognition
For more details, please refer to the paper pre-print: [Striking the Balance: Human Pose Estimation based Optimal Fall Recognition](./preprint.pdf).

# Contributors
- [Abhijay Singh](abhijay@umd.edu)
- [Tharun V. Puthanveettil](tvpian@umd.edu)


# Dependencies
The dependencies are listed in requirements.txt. You can install them with the following command:
```bash
pip install -r requirements.txt
```

# Dataset Preparation
 - Create a folder called dataset in the root folder of the project. The folder structure should be as follows:
    ```bash
    Fall Detection
    ├── data
    │   ├── adl-01-cam0-urfdd-1-no_keypoints.npy
    │   ├── ...
    ├── remaining files
    ```

# Training
- The training code uses the [PyTorch](https://pytorch.org/) framework.
- To start training, run the following command:
    ```bash
    python main.py --dataset ./data
    ```
- Other arguments can be found in the [main.py](./main.py) file.
- The trained models are stored in the `./output` folder.
- The hyperparameters of the model can be changed in the [./config/model.json](./config/model.json) file.