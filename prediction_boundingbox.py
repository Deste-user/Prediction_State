import os

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import prediction_ground_truth as m
import prediction_dim_vehicle as pdv

def plot_all_trajectoriesBB(vect):
    fig = plt.subplot()
    for i in range(0, len(vect)):
            vect_BB= np.array(vect[i][1])
            fig.plot(vect_BB[:,0],vect_BB[:,1],label=vect[i][0])
    plt.legend()
    plt.savefig(os.path.join("plots_boundingbox/","All_BB"))

#Ne faccio uno nuovo senza usare quello dello script prima così posso modificare parametri singolarmente
def setUp():
    #Create matrix A,B,C
    #A,B = calculate_matrix(0,1)
    C= [[1,0,0,0],[0,1,0,0]]
    #Create covariance matrix P0, 30m/s is the deviance standard of velocity
    P0 = [[2.5**2,0,0,0],[0,2.5**2,0,0],[0,0,30**2,0],[0,0,0,30**2]]
    #Create covariance matrix Q (process noise) it represents the noise in the system (accelleration)
    Q = [[(9.81/8)**2,0],[0,(9.81/8)**2]]
    #Create covariance matrix R (measurement noise) it represents the noise in the measurements
    #the error in the position has a deviance of  0.2 meter
    R = [[0.2**2,0],[0,0.2**2]] #diminuire la deviazione standard
    return P0,Q,R,C

def order_csv():
    data= pd.read_csv("dataBB.csv")
    data= data.sort_values(by=['box_label','time'])
    filter_condition = (data['time'] < 50)
    data = data[filter_condition]
    data_ordinato_corretto_corto = data.to_csv('dataBB_corto.csv',index=False)
#ho 8 punti che passo alla funzione
def calculate_bounding_box_center(points):
    x = float(points["P1x"]) + float(points["P2x"]) + float(points["P3x"]) + float(points["P4x"]) + float(points["P5x"]) + float(points["P6x"]) + float(points["P7x"]) + float(points["P8x"])
    y = float(points["P1y"]) + float(points["P2y"]) + float(points["P3y"]) + float(points["P4y"]) + float(points["P5y"]) + float(points["P6y"]) + float(points["P7y"]) + float(points["P8y"])
    return [x/8, y/8]

#Ho aggiunto una funzione per calcolare il guadagno prospettico sapendo il vero centro dei veicoli, lo si fa solo per
#la prima posizione
def calculate_gains_for_center(first_ready_data,first_gt_data,label_list):
    vectgain = []
    vectpos = []
    first_ready_data=replace_label(first_ready_data, label_list)
    #print("First ready data: ", first_ready_data)
    #print("First GT data: ", first_gt_data)
    for i in range(0,len(first_ready_data)):
        pos= calculate_bounding_box_center(first_ready_data[i])
        vectpos.append([first_ready_data[i]["box_label"],pos[0],pos[1]])
    #print("Vectpos: ", vectpos)
    for i in range(0,len(first_gt_data)):
        for j in range(0,len(vectpos)):
            if first_gt_data[i]["label"] == vectpos[j][0]:
                vectgain.append([first_gt_data[i]["label"],float(first_gt_data[i]["x"])/float(vectpos[j][1]),float(first_gt_data[i]["y"])/float(vectpos[j][2])])

    return vectgain

def prod_gain(vectgain,pos,current_label):
    for i in range(0,len(vectgain)):
        if vectgain[i][0] == current_label:
            x = pos[0]*vectgain[i][1]
            y = pos[1]*vectgain[i][2]
            return [x,y]


#oltre a buttare fuuri la lista di label, butto fuori anche i primi istanti di ogni veicolo sia GT che BB
def recognize_label(data, ground_truth):
    label_list = []
    first_data_vehicle = []
    first_gt_vehicle = []
    # creazione di una copia della lista ground_truth
    # per evitare di modificare la lista originale
    ground_truth_copy = []
    #nella copia del GT shifto e confronto dal 20esimo mmsecondo di campionamento
    #for i in range(len(ground_truth)):
    #    if int(ground_truth[i]["time"])>=20:
    #        ground_truth_copy.append(ground_truth[i])

    tmp = 0
    label = ground_truth[0]["label"]
    for i in range(len(ground_truth)):
        if ground_truth[i]["label"] == label:
            if tmp > 18:
                ground_truth_copy.append(ground_truth[i])
            else:
                tmp += 1
        else:
            label = ground_truth[i]["label"]
            tmp = 0

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

    return label_list, first_data_vehicle, first_gt_vehicle


def replace_label(data, label_list):
    for i in range(0,len(data)):
        for j in range(0,len(label_list)):
            etichetta1 = data[i]["box_label"]
            etichetta2 = label_list[j][0]
            if etichetta1 == etichetta2:
                # sostituisco la BB_label con le label del pitt_trajectories
                data[i]["box_label"] = label_list[j][1]
    return data

def prepare_ground_truth(ground_truth, current_label, skip=20):
    filtered_gt = []
    tmp = 0
    label = ground_truth[0]["label"]

    for item in ground_truth:
        if item["label"] == label:
            if tmp >= skip:
                filtered_gt.append(item)
            else:
                tmp += 1
        else:
            label = item["label"]
            tmp = 0
    return np.array([item for item in filtered_gt if item["label"] == current_label])

def plot_dataBB(predicted_posBB_array, predicted_velBB_array, ground_truth, current_label):
    predicted_posBB_array = np.array(predicted_posBB_array)
    predicted_velBB_array = np.array(predicted_velBB_array)
    ground_truth_copy = prepare_ground_truth(ground_truth, current_label)

    if not os.path.exists(f"plots_boundingbox/{current_label}"):
        os.makedirs(f"plots_boundingbox/{current_label}", exist_ok=True)

    #Prendo la velocità dal ground truth
    velGT_array = np.array([[float(entry["vx"]), float(entry["vy"])] for entry in ground_truth_copy])

    #prendo il minimo delle lunghezze
    min_length = min(len(predicted_velBB_array), len(velGT_array))
    predicted_velBB_array = predicted_velBB_array[:min_length]
    velGT_array = velGT_array[:min_length]
    #la funzione arange permette di creare un array di valori da 0 a min_length
    time_array = np.arange(min_length)

    #Confronto velocità per x e y
    m.save_plot([time_array],
              [predicted_velBB_array[:, 0], predicted_velBB_array[:, 1], velGT_array[:, 0], velGT_array[:, 1]],
              current_label,
              ["Predicted Velocity x", "Predicted Velocity y", "Real Velocity x", "Real Velocity y"],
              f"plots_boundingbox/{current_label}/Velocità Prevista.png")

    #Confronto errore velocità
    err_vel = m.calculate_error(predicted_velBB_array, velGT_array)

    m.save_plot([time_array], [err_vel], current_label, ["Error in Velocity"],
              f"plots_boundingbox/{current_label}/Errore Velocità.png")

    #prendo pos x e y dal ground truth
    posGT_array = np.array([[float(entry["x"]), float(entry["y"])] for entry in ground_truth_copy])


    min_length = min(len(predicted_posBB_array), len(posGT_array))
    predicted_posBB_array = predicted_posBB_array[:min_length]
    posGT_array = posGT_array[:min_length]

    time_array = np.arange(min_length)

    #Confronto posizione per x
    m.save_plot([time_array],
              [predicted_posBB_array[:, 0], posGT_array[:, 0]],
              current_label,
              ["Predicted Position x", "Real Position x"],
              f"plots_boundingbox/{current_label}/Posizione Prevista-Reale-X.png")

    #Confronto posizione per y
    m.save_plot([time_array],
              [predicted_posBB_array[:, 1], posGT_array[:, 1]],
              current_label,
              ["Predicted Position y", "Real Position y"],
              f"plots_boundingbox/{current_label}/Posizione Prevista-Reale-Y.png")

    #Traiettoria Reale e Predetta
    m.save_plot([posGT_array[:, 0],predicted_posBB_array[:,0]],
              [posGT_array[:, 1], predicted_posBB_array[:, 1]],
              current_label,
              ["Real Trajectory", "Predicted Trajectory"],
              f"plots_boundingbox/{current_label}/Traiettoria.png")

    #Confronto errore posizione
    err_pos = m.calculate_error(predicted_posBB_array, posGT_array)
    m.save_plot([time_array], [err_pos], current_label, ["Error in position"],
              f"plots_boundingbox/{current_label}/Errore Posizione.png")

    m.save_plot([time_array], [err_vel, err_pos], current_label, ["Error in velocity", "Error in position"],
              f"plots_boundingbox/{current_label}/Errore_Velocità_Posizione.png")

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

   tmp=0
   label=ground_truth[0]["label"]
   for i in range(len(ground_truth)):
       if ground_truth[i]["label"] == label:
           if tmp > 20:
               ground_truth_copy.append(ground_truth[i])
           else:
               tmp+=1
       else:
              label=ground_truth[i]["label"]
              tmp=0


   for i in range(0,len(data)):
       if data[i]["box_label"] != BB_label:
           BB_array = np.array(BB_array)
           fig_copy.plot(BB_array[0, 0], BB_array[0, 1], marker="x")
           fig_copy.plot(BB_array[:, 0], BB_array[:, 1],color=colors[color_idx],linestyle="--", label=BB_label, )
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
           fig_copy.plot(GT_copy_array[:, 0], GT_copy_array[:, 1],color=colors[color_idx], label=GT_copy_label, )
           GT_copy_label= ground_truth_copy[i]["label"]
           color_idx = color_idx + 1 % len(colors)
           GT_copy_array=[]
       else:
           GT_copy_array.append([float(ground_truth_copy[i]["x"]),float(ground_truth_copy[i]["y"])])

   plt.legend()
   plt.savefig(os.path.join("plots_boundingbox/","BB_GT_shiftate"))
   plt.close()

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
   plt.savefig(os.path.join("plots_boundingbox/","BB_GT_no_shiftate"))
   plt.close()


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
   plt.savefig(os.path.join("plots_boundingbox/","GT"))
   plt.close()

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
   plt.savefig(os.path.join("plots_boundingbox/","BB"))
   plt.close()

def main():
    ground_truth = m.read_csv_in_folder("pitt_trajectories.csv")
    #il data preso seguentemente è un data dove è stato ordinato, e levato doppioni
    data = m.read_csv_in_folder("dataBB_definitivo_corto.csv")
    draw_all_trajectories(ground_truth, data)
    posBB = calculate_bounding_box_center(data[0])
    #Stato Iniziale
    state= m.State(posBB[0], posBB[1])
    #Preparazione matrici
    P0,Q,R,C = setUp()
    predicted_posBB_array = []
    array_BB = []
    predicted_velBB_array = []
    posBB_array = []
    time_array = []
    posBB_array.append([state.posX, state.posY])
    #butto fuori la lista di label e il primo istante di ogni veicolo
    label_list,First_BB_place,First_GT_place=recognize_label(data, ground_truth)
    #calcolo i guadagni sull'istante iniziale di ogni veicolo.
    gains=calculate_gains_for_center(First_BB_place, First_GT_place,label_list)
    ready_data = replace_label(data, label_list)
    current_label = ready_data[0]["box_label"]
    #print("Ready data: ", ready_data)
    Pkminus1_kminus1 = P0
    for i in range(0, len(ready_data)):
        if ready_data[i]["box_label"] == current_label:
            if i == 0:
                A, B = m.calculate_matrix(0, 1)
                time_array.append(0)
            else:
                A, B = m.calculate_matrix(float(ready_data[i-1]["time"]), float(ready_data[i]["time"]))
                time_array.append(float(ready_data[i]["time"]) - 20)

            center_pos = calculate_bounding_box_center(ready_data[i])
            #faccio il prodotto dei guadagni per le posizioni
            #center_pos = prod_gain(gains, center_pos, current_label)
            #posBB_array.append(center_pos)


            #Predizione
            xk_kminus1 = np.dot(A, state.get_state())
            Pk_kminus1 = np.dot(np.dot(A, Pkminus1_kminus1), np.transpose(A)) + np.dot(np.dot(B,Q), np.transpose(B))

            #Correzione
            K = np.dot(np.dot(Pk_kminus1, np.transpose(C)), np.linalg.inv(np.dot(np.dot(C, Pk_kminus1), np.transpose(C)) + R))
            z = np.array(center_pos) + m.noisy_generator(0.2)


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
            plot_dataBB( predicted_posBB_array, predicted_velBB_array,ground_truth,current_label)
            array_BB.append([current_label, predicted_posBB_array])
            current_label = ready_data[i]["box_label"]
            nextBB= calculate_bounding_box_center(ready_data[i])
            state = m.State(nextBB[0], nextBB[1])
            predicted_posBB_array = []
            predicted_velBB_array = []
            posBB_array = []
            time_array = []
            time_array.append(0)
            Pkminus1_kminus1 = P0

    plot_all_trajectoriesBB(array_BB)













