ADVANCED PROGRAMMING CSC_521
=============================

This project is done by:
TEAM:
=====
o	VISHNUGUPTHAA RAMIDI  [212238054]
o	TEJA VENKATA SAI PAVAN KOPPISETTY  [212231827]
o	YASWANTH KATUKOTA  [212240199]

PACKAGES REQUIRED TO RUN THE CODE:
=================================
"numpy" using "pip install numpy"
"matplotlib" using "pip install matplotlib"
"networkx" using "pip install networkx"

STEPS TO RUN THE CODE:
=====================
1. EXTRACT THE .ZIP 
2. COMPILE AND RUN THE PYTHON CODE "Ramidi_Katukota.py" USING THE COMMAND "python Ramidi_Katukota_Koppishetty.py"

Project Details:
===============
When you run the code
    it will prompt the user to enter the details of the network model and sensor model

Output:
=======
Enter the width x of the sensor network (default: 2000): 2000
Enter the length y of the sensor network (default: 2000): 2000
Enter the number of sensor nodes (default: 100): 100
Enter the transmission range in meters (default: 400): 400
Enter the number of DNs (default: 50): 50
Enter the maximum number of data packets each DN has (default: 1000): 1000
The Graph is Connected

Minimum Spanning Tree
//We get a graph here

DNs and their data packets:
DN 80: 634 data packets
DN 1: 614 data packets
DN 36: 331 data packets
DN 83: 874 data packets
DN 16: 543 data packets
DN 15: 440 data packets
DN 46: 599 data packets
DN 43: 945 data packets
DN 74: 61 data packets
DN 2: 827 data packets
DN 30: 684 data packets
DN 6: 293 data packets
DN 86: 221 data packets
DN 56: 38 data packets
DN 0: 657 data packets
DN 26: 497 data packets
DN 33: 895 data packets
DN 40: 600 data packets
DN 69: 131 data packets
DN 78: 236 data packets
DN 10: 149 data packets
DN 5: 209 data packets
DN 51: 964 data packets
DN 53: 21 data packets
DN 82: 289 data packets
DN 48: 582 data packets
DN 84: 531 data packets
DN 58: 280 data packets
DN 75: 176 data packets
DN 35: 52 data packets
DN 21: 999 data packets
DN 65: 452 data packets
DN 28: 77 data packets
DN 24: 525 data packets
DN 13: 334 data packets
DN 8: 266 data packets
DN 97: 600 data packets
DN 55: 512 data packets
DN 57: 221 data packets
DN 62: 574 data packets
DN 91: 650 data packets
DN 45: 731 data packets
DN 9: 206 data packets
DN 17: 958 data packets
DN 64: 46 data packets
DN 93: 629 data packets
DN 39: 972 data packets
DN 60: 512 data packets
DN 73: 134 data packets
DN 12: 836 data packets

Details for Greedy 1 Algorithm
Route:  [0, 19, 70, 0]
Cost:  438.1231360868138
Total Prizes:  2365.0538226296335
Remaining Budget:  168.73630110693276
Running time: 0.0000000 seconds

Details for Greedy 2 Algorithm
Route:  [0, 19, 70, 0]
Cost:  438.1231360868138
Total Prizes:  2365.0538226296335
Remaining Budget:  168.73630110693276
Running time: 0.0000000 seconds

Details for MARL Algorithm
Route:  [19, 61, 0]
Cost:  546.4221733651611
Total Prizes:  1686.2899476913158
Remaining Budget:  74.52660385545414
Running time: 163.0856209 seconds

NOTE : Detailed outputs are in Output.docx

Procedure:
===========

1. Initially the code takes the values user enters and generates the graph according to the model.

2. Gives the graph and data packets and the results of each algorithm as output.

CONTRIBUTION:
=============
VISHNUGUPTHAA_RAMIDI ---> Coding part
YASWANTH_KATUKOTA ---> Research paper analysing and help with code