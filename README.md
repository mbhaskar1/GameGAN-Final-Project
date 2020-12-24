# GameGAN-Final-Project

This was my final project for my Deep Learning for Computer Graphics course taken at the University of Florida. I implement a heavily simplified version of the approach used in
[GameGAN](https://nv-tlabs.github.io/gameGAN/) for the game of Pong. I use a generative adversarial network based architecture in order train a model to simulate Pong gameplay, 
taking screen images and keyboard actions as input, and outputting the following frame in the game. For a more precise description of the project, please consult the
[Technical Report](https://github.com/mbhaskar1/GameGAN-Final-Project/blob/main/Technical%20Report.pdf).

Because of GitHub data size restrictions, the Data and Testing files are not included in this repository. These files can be generated using the `data_generation.py` file. To use
the code exactly as it is written, the Data folder should have 60 generated data files, and the Testing folder should have a single generated data file named `Data-1.npy`.

Instructions for how to use each file are also included in the Technical Report.
