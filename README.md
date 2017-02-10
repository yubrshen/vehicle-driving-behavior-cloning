#**Autonomous Car Steering by Behaviro Cloning of Neural Network**

This project studys how to use deep neural network and convolutional neural network to clone human steering behavior. 
Human steering behavior is recorded in terms of sequence of photo captures of the road sceen, and recording of the corresponding steering angles applied by the human driver. 
A convolutional and deep neural network is trained with such collected samples to emulate the desired steering angles. 
The autonomus car steering is achieved by feeding live photo captures to trained neural network, and use the neural network to prodict the steering angles. 
In a game simulator, the steering angles are fed to the simulated car to actuate the movement of the car in the simulator. 

The convolutional and deep neural network is implemented with Keras. 

I used the samples of human driving records collected by Udacity through the simulator. No manual samples were used, due to limited amount of time, and my focus on understanding existing problems. 

The project finds that it's feasible to use trained neural network to autonomously steer the car in the simulator.

It turned out that it's most crucial to properly crop front view images and augment sample data to train the neural network to generalize well, and to have well behaved autonomous steering. 

In the project, I also proposed a way to visualize the predictions of steering angles, 
to address the challenges that the neural network training objective function, the loss function, not able to reflect fully the performance of training.

This is a journey of understanding. The project explores the problem space, discover more questions than solutions found.

For more details, please read [writeup_report](./writeup_report.md)

