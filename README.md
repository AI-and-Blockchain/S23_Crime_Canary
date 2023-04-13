# Crime Canary

What is Crime Canary?
---------------------

Crime Canary is a framework that involves the Algorand block chain technology for the **incentivization** and **automation** of law enforcement reporting. Specifically, a user (civilian) can utilize the framework to report an incident, a crime, etc. to a local law enforcement via their submission of an image. Additionally, if the submission is proven to be valid, the user is then rewarded with governmental crypto-coins, which can be traded for approved services. 

Hence, the overall goal is
- To create an automated, transparent, and incentivized platform for fast and accurate incident reporting
- To reduce law enforcement response times, which may help save lives, apprehend crime perpetrators, and generally improve the safety of citizens
- To increase the volunteer participation of citizens in law enforcement
![alt text](https://github.com/AI-and-Blockchain/S23_Crime_Canary/blob/main/images/story.png)


This is a research project and not an official product of any governmental entities. Expect bugs and sharp edges.

## Installation
```
  pip3 install -r requirements.txt
```
and follow the installation of Algorand Sandbox (https://github.com/algorand/sandbox)

## Usage
![alt text](https://github.com/AI-and-Blockchain/S23_Crime_Canary/blob/main/images/flow_chart.png)

Launching Web App.
```
  python web_app.py
```
Creating Accounts
```
  python create_accounts.py
```

## Story Chart
![alt text](https://github.com/AI-and-Blockchain/S23_Crime_Canary/blob/main/images/components.png)

## Model
![alt text](https://github.com/AI-and-Blockchain/S23_Crime_Canary/blob/main/images/Siamese_Model_diagram.png)

## Update (3.23.23)
1. Initial code for web app.
    - Users can submit image and their public key
2. Code for creating accounts on Algorand

## Update (4.6.23)
1. Web app can load a model and predict an image
    - Still need extra functionality on ensuring the size of image is correct to the model

2. Web app can check if a user did put a stake on the blockchain

3. Siamese Network is added along with experimental results following the work on [Convolution neural network joint with mixture of extreme learning machines for feature extraction and classification of accident images](https://link.springer.com/article/10.1007/s11554-019-00852-3)

## Contributors
Chibuikem Ezemaduka (ezemac@rpi.edu)

Bao Pham (phamb@rpi.edu)

Maruf A. Mridul (mridum@rpi.edu)
