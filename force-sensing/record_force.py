##################################################################################################################################
# Robotic Dough Shaping - Force sensing: record force
#   Project for Robot Manipulation (CS 6751), Cornell University, Spring 2022
#   Group members: Di Ni, Xi Deng, Zeqi Gu, Henry Zheng, Jan (Janko) Ondras
##################################################################################################################################
# Author: 
#   Di Ni (dn273@cornell.edu)
##################################################################################################################################
# Upload force_sensor.ino to arduino before running this
##################################################################################################################################

import serial
import time
import keyboard
import csv
# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('COM4', 9800, timeout=1)

time.sleep(2)
# ser.flush()
start = time.time()
save_list = []
while True:
    line = ser.readline()   # read a byte

    if line:
        string = line.decode()  # convert the byte string to a unicode string
        # print('the force is',string)
        num = float(string) # convert the unicode string to an int
        save_list.append([time.time() - start, num])
        print('the force is',num, 'g')

    if keyboard.is_pressed("q"):
        print("q pressed, ending loop")
        break

with open(f'purple.csv','w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Force'])
    csv_writer.writerows(save_list)

ser.close()
