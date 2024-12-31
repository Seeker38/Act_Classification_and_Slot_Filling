# Act Classification and Slot Filling

This project is for Act Classification/Detection and Slot Filling on [MultiWOZ version 2.4](https://github.com/smartyfh/MultiWOZ2.4) dataset.

<<<<<<< HEAD
The model i use is bert-base-uncased
=======
The model i use bert-base-uncased
>>>>>>> cfac348a97d77f9f4723dff191a40d911b64a792

## Preprocess data

From the MultiWOZ2.4, use create-data.py to create 3 data: 
- train: train_dials.json
- test: test_dials.json 
- validation: dev_dials.json

## Details

There are 4 main ipynb files: <br>
<<<<<<< HEAD
- [dialogue act](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/dialogueAct.ipynb): This focus on the dialogue act detection on the test dataset but does not consider mutlilabel<br>
- [dialogue act improving version](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/dialogueActVer2.ipynb): A better version of the previous work on dialogue act, by by mutlilabel classification <br>
- [Slot Filling](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/slotFilling.ipynb): Slot Filling Detection on the test dataset <br>
- [joint Act Classification and Slot Filling](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/joint.ipynb): A joint version of model decteting both dialogue act and slot-filling on the test dataset <br>
=======
- [dialogue act](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/dialogueAct.ipynb): this focus on the dialogue act detection on the test dataset but does not consider mutlilabel<br>
- [dialogue act improving version](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/dialogueActVer2.ipynb): A better version of the previous work on dialogue act, by by mutlilabel classification <br>
- [Slot Filling](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/slotFilling.ipynb): Slot Filling Detection on the test dataset <br>
- [joint Act Classification and Slot Filling](https://github.com/Seeker38/Act_Classification_and_Slot_Filling/blob/master/joint.ipynb): a joint version of model decteting both dialogue act and slot-filling on the test dataset <br>
>>>>>>> cfac348a97d77f9f4723dff191a40d911b64a792

They all contain previous history run (inlcuding the training process and the evaluating results)