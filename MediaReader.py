# -*- coding: utf-8 -*-
"""
Author: Jesus L Garcia
Course: Udacity Intel(R) Edge AI for IoT Developers Nanodegree
Project: Deploy a People Counter App at the Edge
"""
import cv2
import os
from glob import glob

class MediaReader():
    
    _init_called = False
    
    def open(self,media):
        self.source_type = self.input_type(media)
        ret = False
        if self.source_type == "camera":
            media = int(media)
            
        if self.source_type == "video" or self.source_type == "camera":
            self.videosource = cv2.VideoCapture(media)
            ret = self.videosource.isOpened()
        elif self.source_type == "dir" or self.source_type == "image":
            self.image_iter = iter(self.all_images(media))
            ret = True
       
        return ret
    
    def __init__(self, media):
        
        
        self.source_type = None
        self.videosource = None
        self.image_iter = None
        self.current = None
         
        MediaReader._init_called = self.open(media)
    
    def input_type(self,input):
        if input.isdigit():
            itype = "camera"
        elif os.path.exists(input):
            extension = os.path.splitext(input)[1]
            if  extension == ".mp4":
                itype = "video"
            elif extension in [".jpg",".gif",".png",".tga",".bmp"]:
                itype = "image"
            else:
                itype = "dir"
        else:
            raise Exception("The input type is not valid or path does not exist.")
        
        return itype
        
    def read(self):
        try:
            if self.source_type == "video" or self.source_type == "camera":
                if self.videosource.isOpened():
                    flag, image = self.videosource.read()
            else:    
                imagename = next(self.image_iter)
                image = cv2.imread(imagename)
                flag = True
        except Exception as e:
            print("Exception: ", e)
            flag = False
            image = None
       
        return flag, image
    
    def isOpened(self):
        return MediaReader._init_called
        
    def all_images(self, path):
        ptype = self.source_type
        image_list = []
        if ptype == "dir":
            if path[-1] != "/":
                path = path + "/"
                pattern = path+'*.%s'
                ## The following helper line was adapted from 
                ## a comment in https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python
                ## by user user1269942
                image_list = [item for i in [glob(pattern % ext) for ext in ["jpg","gif","png","tga",".bmp"]] for item in i]          
        elif ptype == "image":
            image_list.append(path)
        
        print("type {} imagelist length {}", ptype, len(image_list))
        return image_list
        
    def release(self):
        if self.source_type == "video" or self.source_type == "camera":
            self.videosource.release()



