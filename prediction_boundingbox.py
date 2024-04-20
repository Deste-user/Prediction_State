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
                min_error = np.sqrt((vectpos[0]-float(first_gt_vehicle[j]["x"]))**2 + (vectpos[1]-float(first_gt_vehicle[j]["y"]))**2)
                print("BB pos: ", vectpos)
                print("GT pos: ", [float(first_gt_vehicle[j]["x"]), float(first_gt_vehicle[j]["y"])])
                #min_error = [abs(vectpos[0]-float(first_gt_vehicle[j]["x"])), abs(vectpos[1]-float(first_gt_vehicle[j]["y"]))]
                #min_error = [float(first_gt_vehicle[j]["x"])-vectpos[0] ,
                #             float(first_gt_vehicle[j]["y"])-vectpos[1]]
                #min_error = [abs(float(first_gt_vehicle[j]["x"]) - vectpos[0]),
                #         abs(float(first_gt_vehicle[j]["y"]) - vectpos[1])]
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("Error: ", min_error)
                print("Min error: ", min_error)
            else:
                print("BB pos: ", vectpos)
                print("GT pos: ", [float(first_gt_vehicle[j]["x"]), float(first_gt_vehicle[j]["y"])])
                #error = [abs(vectpos[0]-float(first_gt_vehicle[j]["x"])), abs(vectpos[1]-float(first_gt_vehicle[j]["y"]))]
                error = np.sqrt((vectpos[0]-float(first_gt_vehicle[j]["x"]))**2 + (vectpos[1]-float(first_gt_vehicle[j]["y"]))**2)
                #error= [abs(float(first_gt_vehicle[j]["x"]) - vectpos[0]), abs(float(first_gt_vehicle[j]["y"]) - vectpos[1])]
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("Error: ", error)
                print("Min error: ", min_error)
                #if error[0] < min_error[0] and error[1] < min_error[1]:
                if error < min_error:
                    min_error = error
                    iter_label = j
        label_list.append([first_data_vehicle[i]["box_label"], first_gt_vehicle[iter_label]["label"]])
        print("Label list: ", label_list)
        print("Next iteration")
    return label_list

def replace_label(data, label_list):
    for i in range(0,len(data)):
        for j in range(0,len(label_list)):
            if data[i]["box_label"] == label_list[j][0]:
                # sostituisco la BB_label con le label del pitt_trajectories
                data[i]["box_label"] = label_list[j][1]
    return data


def plot_dataBB(time_array, posBB_array, predicted_posBB_array, predicted_velBB_array, current_label):
    posBB_array = np.array(posBB_array)
    predicted_posBB_array = np.array(predicted_posBB_array)
    predicted_velBB_array = np.array(predicted_velBB_array)

    # Plot Predicted position and Realistic position
    # Draw Trajectory
    fig1 = plt.subplot()
    fig1.plot(posBB_array[:, 0], posBB_array[:, 1], label='Realistic position')
    fig1.plot(predicted_posBB_array[:, 0], predicted_posBB_array[:, 1], label='Predicted position')
    fig1.set_title(current_label)
    fig1.legend()
    plt.show()




def main():
    ground_truth = m.read_csv_in_folder("pitt_trajectories.csv")
    data = m.read_csv_in_folder("data_ordinato.csv")
    posBB = calculate_bounding_box_center(data[0])
    #Stato Iniziale
    state= m.State(posBB[0], posBB[1])
    #Preparazione matrici
    P0,Q,R,C = m.setUp()
    predicted_posBB_array = []
    predicted_velBB_array = []
    posBB_array = []
    time_array = []
    posBB_array.append([state.posX, state.posY])
    label_list=recognize_label(data, ground_truth)
    ready_data = replace_label(data, label_list)
    current_label = ready_data[0]["box_label"]
    #print("Ready data: ", ready_data)
    Pkminus1_kminus1 = P0
    for i in range(0, len(ready_data)):
        if ready_data[i]["box_label"] == current_label:
            if i == 0:
                A, B = m.calculate_matrix(0, 1)
            else:
                A, B = m.calculate_matrix(float(ready_data[i-1]["time"]), float(ready_data[i]["time"]))
            center_pos = calculate_bounding_box_center(ready_data[i])
            posBB_array.append(center_pos)
            time_array.append(float(ready_data[i]["time"])-20)

            #Predizione
            xk_kminus1 = np.dot(A, state.get_state())
            Pk_kminus1 = np.dot(np.dot(A, Pkminus1_kminus1), np.transpose(A)) + np.dot(np.dot(B,Q), np.transpose(B))

            #Correzione
            K = np.dot(np.dot(Pk_kminus1, np.transpose(C)), np.linalg.inv(np.dot(np.dot(C, Pk_kminus1), np.transpose(C)) + R))
            z = np.array(center_pos) + m.noisy_generator()

            #Calcolo del nuovo Stato
            xk_k = xk_kminus1 + np.dot(K, (z - np.dot(C, xk_kminus1)))

            #Calcolo della nuova matrice di covarianza
            Pk_k = np.dot(np.dot((np.identity(4)-np.dot(K,C)),Pk_kminus1),np.transpose(np.identity(4)-np.dot(K,C)))+np.dot(np.dot(K,R),np.transpose(K))
            #Aggiorna Stato
            state.update_state(xk_k)
            #Aggiorna matrice probabilità
            Pkminus1_kminus1 = Pk_k
            predicted_posBB_array.append([state.posX, state.posY])
            predicted_velBB_array.append([state.velX, state.velY])
        else:
            plot_dataBB(time_array, posBB_array, predicted_posBB_array, predicted_velBB_array, current_label)
            current_label = ready_data[i]["box_label"]
            nextBB= calculate_bounding_box_center(ready_data[i])
            state = m.State(nextBB[0], nextBB[1])
            predicted_posBB_array = []
            predicted_velBB_array = []
            posBB_array = []
            time_array = []
            Pkminus1_kminus1 = P0

    #TODO debuggare valutazione Bounding Box poichè fa sbarellare i grafici


exit = False
while not exit:
    choice = input("Enter choice (BB/GT): ")
    if choice == "BB":
        main()
        plt.close('all')
    elif choice == "GT":
        m.main()
        plt.close('all')
    else:
        print("Invalid choice")
    print("Do you want to exit? (yes/no)")
    exit_choice = input()
    if exit_choice == "yes":
        exit = True












