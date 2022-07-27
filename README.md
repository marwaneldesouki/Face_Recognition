# Project Title
- Face recognition

## Description
- Recognize and manipulate faces from Python

## DataSet used
-	There is a script(data_set_maker) to make dataset By taking 10 images to the person from the camera and name folder to name of the person to make it unique.
 ![data](https://user-images.githubusercontent.com/37198610/181280778-a3ed9bdd-184a-4ebd-b92c-00173ffb78f3.png)

## Libraries used
•	cv2
•	face_recognition
•	os
•	numpy 
•	PIL

## Requirements
•	Python 3.3+ or Python 2.7

## How it work ?
•	at first the (data_set_maker) script open the camera and take 10 images to the person and automatic crop the image to the face only
* ![image](https://user-images.githubusercontent.com/37198610/181282326-a96a8d76-e758-4e26-bb1c-d09b494cd223.png)

•	Then the (main) script Find and manipulate facial features in pictures Get the locations and outlines of each person’s eyes, nose, mouth and chin.
* ![image](https://user-images.githubusercontent.com/37198610/181284463-d038d828-b857-40d7-bed9-bccb966cd4ba.png)

•	Then eyes, nose, mouth and chin it`s called futures so the futures are taken to compare it with other futures to every person in dataset ,and when it match then it can recognize the face in image.
* ![image](https://user-images.githubusercontent.com/37198610/181284617-dae03a9e-1fd7-41a3-81b6-fcceb4018db6.png)



## HOW TO USE
 • make your own dataset by data_set_maker.py
 make sure to add more than 10 models to make a perfect train
 •run main.py and wait until the training end then the camera or the screen capture will open

## Developer
marwan eldesouki
