import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import main as m

def order_csv():
    data= pd.read_csv("data_ordinato_corretto.csv")
    data= data.sort_values(by=['box_label','time'])
    filter_condition = (data['time'] <= 50) | (data['time'] > 69)
    data = data[filter_condition]
    data_ordinato_corretto_corto = data.to_csv('dataBB_definitivo_corto.csv',index=False)
#ho 8 punti che passo alla funzione
def calculate_bounding_box_center(points):
    x = float(points["P1x"]) + float(points["P2x"]) + float(points["P3x"]) + float(points["P4x"]) + float(points["P5x"]) + float(points["P6x"]) + float(points["P7x"]) + float(points["P8x"])
    y = float(points["P1y"]) + float(points["P2y"]) + float(points["P3y"]) + float(points["P4y"]) + float(points["P5y"]) + float(points["P6y"]) + float(points["P7y"]) + float(points["P8y"])
    return [x/8, y/8]
def recognize_label(data, ground_truth):
    label_list = []
    first_data_vehicle = []
    first_gt_vehicle = []
    # creazione di una copia della lista ground_truth
    # per evitare di modificare la lista originale
    ground_truth_copy = []
    #nella copia del GT shifto e confronto dal 20esimo mmsecondo di campionamento
    for i in range(len(ground_truth)):
        if int(ground_truth[i]["time"])>=20:
            ground_truth_copy.append(ground_truth[i])


    first_data_vehicle.append(data[0])
    first_gt_vehicle.append(ground_truth_copy[0])
    j=0
    k=0
    #prendo i primi istanti di ogni veicolo
    for i in range(0, len(data)):
      if data[i]["box_label"] != "end":
        if data[i]["box_label"] != first_data_vehicle[j]["box_label"]:
            first_data_vehicle.append(data[i])
            j+=1

    print("-------------------")

    for i in range(0, len(ground_truth_copy)):
      if ground_truth_copy[i]["label"] != "end":
        if ground_truth_copy[i]["label"] != first_gt_vehicle[k]["label"]:
            first_gt_vehicle.append(ground_truth_copy[i])
            k+=1

    for i in range(0, len(first_data_vehicle)):
      if first_data_vehicle[i]["box_label"] != "end":
        vectpos = calculate_bounding_box_center(first_data_vehicle[i])
        for j in range(0, len(first_gt_vehicle)):
            print("--------------------")
            #la prima iterazione la indico sempre come minima distanza altrimenti confronto la min_distance
            # con la distanza calcolata
            if j==0:
                min_distance = np.sqrt((vectpos[0] - float(first_gt_vehicle[j]["x"])) ** 2 + (vectpos[1] - float(first_gt_vehicle[j]["y"])) ** 2)
                iter_label=j
                print("BB pos: ", vectpos)
                print("GT pos: ", [float(first_gt_vehicle[j]["x"]), float(first_gt_vehicle[j]["y"])])
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("Error: ", min_distance)
                print("Min distance: ", min_distance)
            else:
                print("BB pos: ", vectpos)
                print("GT pos: ", [float(first_gt_vehicle[j]["x"]), float(first_gt_vehicle[j]["y"])])
                distance = np.sqrt((vectpos[0] - float(first_gt_vehicle[j]["x"])) ** 2 + (vectpos[1] - float(first_gt_vehicle[j]["y"])) ** 2)
                print("Label: ", first_data_vehicle[i]["box_label"], " GT label: ", first_gt_vehicle[j]["label"])
                print("distance: ", distance)
                print("Min distance: ", min_distance)
                if distance <= min_distance:
                    min_distance = distance
                    #salvo l'indice del veicolo con la distanza minore
                    iter_label = j
                    print("Label_minima:",first_gt_vehicle[iter_label]["label"])

        label_list.append([first_data_vehicle[i]["box_label"], first_gt_vehicle[iter_label]["label"], min_distance])
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

    #si disegna la traiettoria del BB e la previsione della sua traiettoria
    fig1 = plt.subplot()
    fig1.plot(posBB_array[:, 0], posBB_array[:, 1], label='Realistic position')
    fig1.plot(predicted_posBB_array[:, 0], predicted_posBB_array[:, 1], label='Predicted position')
    fig1.set_title(current_label)
    fig1.legend()
    plt.show()

    #stampo il grafico della velocità predetta rispetto al tempo
    fig_velocity=plt.subplot()
    fig_velocity.plot(time_array,predicted_velBB_array[:,0],label="Predicted Velocity x")
    fig_velocity.plot(time_array,predicted_velBB_array[:,1],label="Predicted Velocity y")
    fig_velocity.set_title(current_label)
    plt.show()

    #stampo errore rispetto al tempo
    #err_pos= []
    #for i in range(len(predicted_posBB_array)):
    #    err_pos.append(np.sqrt((posBB_array[i][0] - predicted_posBB_array[i][0]) ** 2 + (posBB_array[i][1] - predicted_posBB_array[i][1]) ** 2))
    #fig_err=plt.subplot()
    #fig_err.plot(time_array,err_pos)
    #lbl=current_label + " error"
    #print(lbl)
    #fig_velocity.set_title(lbl)
    #plt.show()


def draw_all_trajectories(ground_truth, data):
   GT_label = ground_truth[0]["label"]
   colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
   GT_array = []
   GT_array_mod = []
   BB_label = data[0]["box_label"]
   BB_array = []
   ground_truth.append({"label": "end", "x": 0, "y": 0,"time": 88000,"heading":0,"object_type":0})
   data.append({"box_label": "end", "P1x": 0, "P1y": 0, "P2x": 0, "P2y": 0, "P3x": 0, "P3y": 0, "P4x": 0, "P4y": 0, "P5x": 0, "P5y": 0, "P6x": 0, "P6y": 0, "P7x": 0, "P7y": 0, "P8x": 0, "P8y": 0, "time": 0})


   ground_truth_copy=[]
   BB_label = data[0]["box_label"]
   GT_copy_label= ground_truth[0]["label"]
   BB_array= []
   GT_copy_array=[]
   fig_copy= plt.subplot()
   color_idx=0

   for i in range(len(ground_truth)):
       if int(ground_truth[i]["time"]) >= 20:
           ground_truth_copy.append(ground_truth[i])

   for i in range(0,len(data)):
       if data[i]["box_label"] != BB_label:
           BB_array = np.array(BB_array)
           fig_copy.plot(BB_array[0, 0], BB_array[0, 1], marker="x")
           fig_copy.plot(BB_array[:, 0], BB_array[:, 1],color=colors[color_idx],linestyle="--", label=GT_label, )
           BB_label = data[i]["box_label"]
           color_idx= color_idx+1 % len(colors)
           BB_array = []
       else:
           BB_array.append(calculate_bounding_box_center(data[i]))

   color_idx=0

   for i in range(len(ground_truth_copy)):
       if(ground_truth_copy[i]["label"] != GT_copy_label):
           GT_copy_array =np.array(GT_copy_array)
           fig_copy.plot(GT_copy_array[0,0],GT_copy_array[0,1],marker="X")
           fig_copy.plot(GT_copy_array[:, 0], GT_copy_array[:, 1],color=colors[color_idx], label=GT_label, )
           GT_copy_label= ground_truth_copy[i]["label"]
           color_idx = color_idx + 1 % len(colors)
           GT_copy_array=[]
       else:
           GT_copy_array.append([float(ground_truth_copy[i]["x"]),float(ground_truth_copy[i]["y"])])

   plt.show()





   #Stampo grafici senza SHIFT in un unico grafico
   BB_array = []
   GT_array = []
   GT_label = ground_truth[0]["label"]
   BB_label = data[0]["box_label"]
   fig1 = plt.subplot()
   color_idx=0

   for i in range(0, len(ground_truth)):
       if ground_truth[i]["label"] != GT_label:
           GT_array = np.array(GT_array)
           fig1.plot(GT_array[0,0], GT_array[0,1],color=colors[color_idx],marker="x")
           fig1.plot(GT_array[:,0], GT_array[:,1], label=GT_label,)
           GT_label = ground_truth[i]["label"]
           color_idx = color_idx + 1 % len(colors)
           GT_array = []
       else:
            GT_array.append([float(ground_truth[i]["x"]), float(ground_truth[i]["y"])])

   color_idx=0
   for i in range(0, len(data)):
       if data[i]["box_label"] != BB_label:
           BB_array = np.array(BB_array)
           fig1.plot(BB_array[0,0], BB_array[0,1],marker="x")
           fig1.plot(BB_array[:,0], BB_array[:,1],color=colors[color_idx],linestyle="--", label=BB_label)
           BB_label = data[i]["box_label"]
           color_idx = color_idx + 1 % len(colors)
           BB_array = []
       else:
           BB_array.append(calculate_bounding_box_center(data[i]))

   plt.legend()
   plt.show()

   #Stampo tutti i grafici insieme e adesso li stampo separatamente
   GT_array = []
   BB_array = []
   GT_label = ground_truth[0]["label"]
   BB_label = data[0]["box_label"]
   color_idx = 0
   fig_gt = plt.subplot()


   for i in range(0, len(ground_truth)):
       if ground_truth[i]["label"] != GT_label:
           GT_array = np.array(GT_array)
           fig_gt.plot(GT_array[0,0], GT_array[0,1],marker="x")
           fig_gt.plot(GT_array[:,0], GT_array[:,1],color=colors[color_idx], label=GT_label,)
           GT_label = ground_truth[i]["label"]
           GT_array = []
           color_idx = (color_idx + 1) % len(colors)
       else:
            GT_array.append([float(ground_truth[i]["x"]), float(ground_truth[i]["y"])])

   plt.legend()
   plt.show()

   fig_bb = plt.subplot()
   color_idx = 0
   for i in range(0, len(data)):
       if data[i]["box_label"] != BB_label:
           BB_array = np.array(BB_array)
           fig_bb.plot(BB_array[0,0], BB_array[0,1],marker="x")
           fig_bb.plot(BB_array[:,0], BB_array[:,1],color=colors[color_idx],label=BB_label)
           BB_label = data[i]["box_label"]
           BB_array = []
           color_idx=color_idx+1 % len(colors)
       else:
           BB_array.append(calculate_bounding_box_center(data[i]))

   plt.legend()
   plt.show()

def main():
    ground_truth = m.read_csv_in_folder("pitt_trajectories.csv")
    #il data preso seguentemente è un data dove è stato ordinato, e levato doppioni
    data = m.read_csv_in_folder("dataBB_definitivo_corto.csv")
    draw_all_trajectories(ground_truth, data)
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


exit = False
while not exit:
    choice = input("Enter choice (BB/GT): ")
    if choice == "BB":
        main()
        #plt.close('all')
    elif choice == "GT":
        m.main()
        #plt.close('all')
    else:
        print("Invalid choice")
    print("Do you want to exit? (yes/no)")
    exit_choice = input()
    if exit_choice == "yes":
        exit = True












