# Self-Driving-Car-Simulation

Using the Udacity self driving simulator and the technique of **Behavioral Cloning** trained a car to drive itself on a track in Unity3D simulation.

## Usage

### To run the pretrained model

Start the [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose **lake track** and press the Autonomous Mode button.
Then Run:

```
python Run_Simulation.py
```
### To train your own model

Start the Simulator and record data in training mode for any track.Then Run:

```
python model.py
```

This will generate files `model-<val_loss>.h5`. Choose the file with minimum validation loss as the model file.

Then Run:

```
python Run_Simulation.py --path=[PATH TO MODEL]
```

## Demo

![ Demo ]( ./Demo.gif )
