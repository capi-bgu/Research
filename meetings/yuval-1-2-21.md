# Meeting Notes

> 2.2.2021 meeting with Yuval Shahar discussing the state of the project and improvements

## Data Collection

* It was suggested that in addition to saving the processed data at the end of each session, we would save the raw data that the collectors outputs.
This will enable us to later process the data however we please, for different experiments.

* We discussed the timing of the user's labeling. As it is now, the system will ask the user for a label at set intervals.
It might be worth exploring more complex labeling strategies. For example, in addition to the set time label, there will be k random labeling calls
for the user throughout the session or group sessions.

* Lastly, we discussed adding an option for the user to prompt the system himself and submit a label for the current session while it's running.

## Experiments

* The experiments that we had in mind are a short 1-2 hour experiment, where the user is exposed to emotion-inducing content then asked
to perform some tasks while being recorded. The second experiment will be us using the system for a few weeks, daily, in an uncontrolled manner.
It was suggested we should add a third experiment where we will allow users to use the system in an uncontrolled manner for a day or two. This will likely not be able
to expose every emotion of the user but will provide valuable data for the longer running experiments.

* We also need to perform some pilot runs of the experiments to adjust some parameters like the emotion-inducing content, the tasks, the session length, and other UX factors.
