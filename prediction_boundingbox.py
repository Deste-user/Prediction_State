import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import main as m

#ho 8 punti che passo alla funzione
def calculate_bounding_box_center(points):
    x = float(points["P1x"]) + float(points["P2x"]) + float(points["P3x"]) + float(points["P4x"]) + float(points["P5x"]) + float(points["P6x"]) + float(points["P7x"]) + float(points["P8x"])
    y = float(points["P1y"]) + float(points["P2y"]) + float(points["P3y"]) + float(points["P4y"]) + float(points["P5y"]) + float(points["P6y"]) + float(points["P7y"]) + float(points["P8y"])
    return [x/8, y/8]

def recognize_label(data, ground_truth):
    label_list = []
    first_data_vehicle = []
    first_gt_vehicle = []
    first_data_vehicle.append(data[0])
    first_gt_vehicle.append(ground_truth[0])
    j=0
    k=0
    #prendo i primi istanti di ogni veicolo
    for i in range(0, len(data)):
        if data[i]["box_label"] != first_data_vehicle[j]["box_label"]:
            first_data_vehicle.append(data[i])
            j+=1
    #print("First data vehicle: ", first_data_vehicle)

    for i in range(0, len(ground_truth)):
        if ground_truth[i]["label"] != first_gt_vehicle[k]["label"]:
            first_gt_vehicle.append(ground_truth[i])
            k+=1

    #devo capire quale label associare a quale veicolo

    for i in range(0, len(first_data_vehicle)):
        vectpos = calculate_bounding_box_center(first_data_vehicle[i])
        for j in range(0, len(first_gt_vehicle)):
            print("--------------------")
            if j==0:
                #min_error = np.sqrt((vectpos[0]-vectpos_gt[0])**2 + (vectpos[1]-vectpos_gt[1])**2)
                min_error = [abs(vectpos[0]-float(first_gt_vehicle[j]["x"])), abs(vectpos[1]-float(first_gt_vehicle[j]["y"]))]
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("Error: ", min_error)
                print("Min error: ", min_error)
            else:
                error = [abs(vectpos[0]-float(first_gt_vehicle[j]["x"])), abs(vectpos[1]-float(first_gt_vehicle[j]["y"]))]
                #error = np.sqrt((vectpos[0]-vectpos_gt[0])**2 + (vectpos[1]-vectpos_gt[1])**2)
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("Error: ", error)
                print("Min error: ", min_error)
                if error[0] < min_error[0] and error[1] < min_error[1]:
                    min_error = error
                    iter_label = j
        label_list.append([first_data_vehicle[i]["box_label"], first_gt_vehicle[iter_label]["label"]])
        print("Label list: ", label_list)
        print("Next iteration")
    return label_list

def main():
    ground_truth = m.read_csv_in_folder("pitt_trajectories.csv")
    data = m.read_csv_in_folder("data_ordinato.csv")
    posBB = calculate_bounding_box_center(data[0])
    #Stato Iniziale
    state= m.State(posBB[0], posBB[1])
    #Preparazione matrici
    P0,Q,R,C = m.setUp()
    posBB_array = []
    velBB_array = []
    pos_gt_array = []
    vel_gt_array = []
    time_array = []
    posBB_array.append([state.posX, state.posY])
    label_list=recognize_label(data, ground_truth)

    #TODO implementare il filtro di Kalman





main()