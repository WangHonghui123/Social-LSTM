# Social-LSTM
The Social-LSTM code for complete trajectory prediction (20 frames). In this repository, the normalized trajectory and non-normalized trajectory data
are used respectively.

## The aim of this repository
The aim of creating this repository is to research the effect of trajectory data processing on the pedestrian trajectory prediction.
Data processing part will be added to my master's thesis.

## Introduction
The data processing in pedestrian trajectory prediction generally uses signal processing since these data contain timeporal information. 
I rewrite Social-LSTM algorithm for normalized complete trajectory prediction and non-normalized complete trajectory prediction. 
The code regarding Social-LSTM algorithm for normalized complete trajectory prediction has been stored in the **Normalized** directory; 
The code regarding Social-LSTM algorithm for non-normalized complete trajectory prediction has been stored in the **Non_Normalized** directory.

## Result
The result of Social-LSTM algorithm on normalized complete trajectory prediction and non-normalized complete trajectory prediction are shown as follows:

**ADE:**
Average Displacement Error (ADE) is the mean square error (MSE) over all estimated points of a trajectory and true points.

| **Dataset**                           | **Social-LSTM (Non normalized)** | **Social-LSTM (Non normalized)**  |
| --------------------------------- | --------- |-----------------|
| **ETH**                     | 4.0781  |**1.9927**|
| **HOTEL**                     | 6.8003   |**1.4552**|
| **ZARA1**                     | 2.0111   |**1.7175**|
| **ZARA2**                     |  2.0794  |**1.3038**|
| **UNIV**                     | 3.9830   |**1.9072**|
| **Average**                     | 3.7904   |**1.6753**|

**FDE:**
FInal Displacement Error (FDE) is the distance between predicted final destination and true final destination at the end of prediction period (20 frames)
| **Dataset**                           | **Social-LSTM (Non normalized)** | **Social-LSTM (Non normalized)**  |
| --------------------------------- | --------- |-----------------|
| **ETH**                     | 5.2299  |**3.5463**|
| **HOTEL**                     | 8.3780   |**2.4048**|
| **ZARA1**                     | 3.1692   |**2.9853**|
| **ZARA2**                     |  3.2690  |**2.2892**|
| **UNIV**                     | 5.5075   |**3.2343**|
| **Average**                     | 5.1107   |**2.8920**|
