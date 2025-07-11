# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 18:42:37 2025

@author: polas
"""

import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')
object_names = list(model.names.values())

st.title("Webcam Live Feed")
run = st.checkbox('เปิดกล้อง')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(frame, verbose=True)
    for detection in result[0].boxes.data:
       x0, y0 = (int(detection[0]), int(detection[1]))
       x1, y1 = (int(detection[2]), int(detection[3]))
       score = round(float(detection[4]), 2)
       cls = int(detection[5])
       object_name =  model.names[cls]
       label = f'{object_name} {score}'  
      
       if  object_name != '' :
           cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
           cv2.putText(frame, label, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
           
       else :
           pass
                    
    
    FRAME_WINDOW.image(frame)
else:
    st.write('หยุด')