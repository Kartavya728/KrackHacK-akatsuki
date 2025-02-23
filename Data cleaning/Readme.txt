datatset_link = https://datasetninja.com/bdd100k#download

Steps:

1. After downloading arranging data in the folder.

2. Now run convert_json_to_yolo.py to convert the json annotaions of the images to yolo Data Form.

3. Now as the converted file endswith .jpg.txt but we need only .txt to train the model, So run removejpg.py

4. Now, Run training_model.py to train the fine tune the pretrained YOLO model

5. Now, Run tes.py to test for different videos and get ouput video.