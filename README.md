# Minerals-identification
Aim: To train a machine learning model using different images of minerals and to predict the name for a new mineral. Flask is used for UI.

Requirements:
Python, Flask, sklearn, tensorflow.

Folder structure for Machine learning:
- have a folder: minerals_dataset
- under minerals_dataset folder have following folders with following names:
          -train_set: 
          under this folder have 7 folders with mineral names and each folder has images of minerals:
           Calcite, Diamond, Sulfur, Dioptase, Ruby, Magnetite, Quartz
          -test_set: 
           under this folder have 7 folders with mineral names and each folder has images of minerals:
           Calcite, Diamond, Sulfur, Dioptase, Ruby, Magnetite, Quartz
- Download here: https://drive.google.com/drive/folders/1pQQXF0eyZJPgNtIq3szKYOSOEaZuCNKm

Folder structure for flask:
-Create main folder: main_folder
-inside main_folder have following:
   1. static (folder)
   2. templates (folder)
   3. app.py
- static should have following files and folders:
js(folder): idx.js
styles(folder): style6.css
bg.gif
cloud.png

- templates should have following:
index.html, success.html
