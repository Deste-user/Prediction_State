import os

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import ground_truth_trajectories as m
import prediction_dim_vehicle as pdv

#Ne faccio uno nuovo senza usare quello dello script prima così posso modificare parametri singolarmente
def setUp():
    #Create matrix A,B,C
    #A,B = calculate_matrix(0,1)
    C= [[1,0,0,0],[0,1,0,0]]
    #Create covariance matrix P0, 30m/s is the deviance standard of velocity
    P0 = [[1**2,0,0,0],[0,1**2,0,0],[0,0,30**2,0],[0,0,0,30**2]]
    #Create covariance matrix Q (process noise) it represents the noise in the system (accelleration)
    #TODO: aumentare varianza distrubo processo Q
    Q = [[(9.81/9.2)**2,0],[0,(9.81/9.2)**2]]
    #Create covariance matrix R (measurement noise) it represents the noise in the measurements
    #the error in the position has a deviance of  0.2 meter
    R = [[0.2**2,0],[0,0.2**2]] #diminuire la deviazione standard
    return P0,Q,R,C

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
    #for i in range(len(ground_truth)):
    #    if int(ground_truth[i]["time"])>=20:
    #        ground_truth_copy.append(ground_truth[i])

    tmp = 0
    label = ground_truth[0]["label"]
    for i in range(len(ground_truth)):
        if ground_truth[i]["label"] == label:
            if tmp >= 20:
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

    return label_list


def replace_label(data, label_list):
    for i in range(0,len(data)):
        for j in range(0,len(label_list)):
            if data[i]["box_label"] == label_list[j][0]:
                # sostituisco la BB_label con le label del pitt_trajectories
                data[i]["box_label"] = label_list[j][1]
    return data


def plot_dataBB(predicted_posBB_array, predicted_velBB_array,ground_truth ,current_label):
    predicted_posBB_array = np.array(predicted_posBB_array)
    predicted_velBB_array = np.array(predicted_velBB_array)
    ground_truth_copy = []
    tmp= 0
    label = ground_truth[0]["label"]

    for i in range(len(ground_truth)):
        if ground_truth[i]["label"] == label:
            if tmp >= 19:
                ground_truth_copy.append(ground_truth[i])
            else:
                tmp += 1
        else:
            label = ground_truth[i]["label"]
            tmp = 0

    velGT_array = []
    #creo vettore velGT che ha componenti x e y
    for i in range(0, len(ground_truth_copy)):
        if ground_truth_copy[i]["label"] == current_label:
            velGT_array.append([float(ground_truth_copy[i]["vx"]), float(ground_truth_copy[i]["vy"])])

    if not os.path.exists("plots_boundingbox/" + current_label):
        os.mkdir("plots_boundingbox/" + current_label)


    velGT_array = np.array(velGT_array)
    if len(predicted_velBB_array) > len(velGT_array):
        predicted_velBB_array = predicted_velBB_array[:len(velGT_array)]
        time_array1=[]
        for i in range(len(predicted_velBB_array)):
            time_array1.append(i)

        #stampo grafico velocità nel tempo confronto tra predizione e realtà
        fig_velocity = plt.subplot()
        fig_velocity.plot(time_array1, predicted_velBB_array[:, 0], label="Predicted Velocity x")
        fig_velocity.plot(time_array1, predicted_velBB_array[:, 1], label="Predicted Velocity y")
        fig_velocity.plot(time_array1, velGT_array[:, 0], label="Realistic Velocity x")
        fig_velocity.plot(time_array1, velGT_array[:, 1], label="Realistic Velocity y")
        fig_velocity.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Velocità Prevista.png'))
        plt.close()

        #stampo grafico errore velocità nel tempo
        err_vel=[]
        for i in range(len(predicted_velBB_array)):
            err_vel.append(np.sqrt((predicted_velBB_array[i,0]-velGT_array[i][0])**2+(predicted_velBB_array[i,1]-velGT_array[i][1])**2))
        fig_err_vel = plt.subplot()
        fig_err_vel.plot(time_array1, err_vel, label="Error Velocity")
        fig_err_vel.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Errore Velocità.png'))
        plt.close()
    else:
        velGT_array = velGT_array[:len(predicted_velBB_array)]
        time_array1 = []
        for i in range(len(velGT_array)):
            time_array1.append(i)

        # stampo grafico velocità nel tempo confronto tra predizione e realtà
        fig_velocity = plt.subplot()
        fig_velocity.plot(time_array1, predicted_velBB_array[:, 0], label="Predicted Velocity x")
        fig_velocity.plot(time_array1, predicted_velBB_array[:, 1], label="Predicted Velocity y")
        fig_velocity.plot(time_array1, velGT_array[:, 0], label="Realistic Velocity x")
        fig_velocity.plot(time_array1, velGT_array[:, 1], label="Realistic Velocity y")
        fig_velocity.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Velocità Prevista.png'))
        plt.close()

        # stampo grafico errore velocità nel tempo
        err_vel = []
        for i in range(len(predicted_velBB_array)):
            err_vel.append(np.sqrt((predicted_velBB_array[i, 0] - velGT_array[i][0]) ** 2 + (predicted_velBB_array[i, 1] - velGT_array[i][1]) ** 2))
        fig_err_vel = plt.subplot()
        fig_err_vel.plot(time_array1, err_vel, label="Error Velocity")
        fig_err_vel.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Errore Velocità.png'))
        plt.close()

    #stampo le posizioni nel tempo e l'errore
    #creo vettore posGT che ha componenti x e y

    posGT_array = []
    for i in range(0, len(ground_truth_copy)):
        if ground_truth_copy[i]["label"] == current_label:
            posGT_array.append([float(ground_truth_copy[i]["x"]), float(ground_truth_copy[i]["y"])])

    posGT_array = np.array(posGT_array)



    if len(predicted_posBB_array) > len(posGT_array):
        predicted_posBB_array = predicted_posBB_array[:len(posGT_array)]
        time_array2 = []
        for i in range(len(predicted_posBB_array)):
            time_array2.append(i)

        #stampo grafico coordinata x posizione nel tempo confronto tra predizione e realtà
        fig_position = plt.subplot()
        fig_position.plot(time_array2, predicted_posBB_array[:, 0], label="Predicted Position x")
        fig_position.plot(time_array2, posGT_array[:, 0], label="Realistic Position x")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Posizione Prevista-Reale-X.png'))
        plt.close()

        #stampo grafico coordinata y posizione nel tempo confronto tra predizione e realtà
        fig_position = plt.subplot()
        fig_position.plot(time_array2, predicted_posBB_array[:, 1], label="Predicted Position y")
        fig_position.plot(time_array2, posGT_array[:, 1], label="Realistic Position y")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Posizione Prevista-Reale-Y.png'))
        plt.close()

        #stampo traiettoria reale e predetta:
        fig_position = plt.subplot()
        fig_position.plot(posGT_array[:, 0], posGT_array[:, 1], label="Realistic Position")
        fig_position.plot(predicted_posBB_array[:, 0], predicted_posBB_array[:, 1], label="Predicted Position")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Traiettoria.png'))
        plt.close()

        #stampo grafico errore posizione nel tempo
        err_pos = []

        for i in range(len(predicted_posBB_array)):
            err_pos.append(np.sqrt((predicted_posBB_array[i, 0] - posGT_array[i][0]) ** 2 + (predicted_posBB_array[i, 1] - posGT_array[i][1]) ** 2))
        fig_err_pos = plt.subplot()
        fig_err_pos.plot(time_array2, err_pos, label="Error Position")
        fig_err_pos.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Errore Posizione.png'))
        plt.close()
    else:
        posGT_array = posGT_array[:len(predicted_posBB_array)]
        time_array2 = []
        for i in range(len(posGT_array)):
            time_array2.append(i)

        # stampo grafico coordinata posizione x nel tempo confronto tra predizione e realtà
        fig_position = plt.subplot()
        fig_position.plot(time_array2, predicted_posBB_array[:, 0], label="Predicted Position x")
        fig_position.plot(time_array2, posGT_array[:, 0], label="Realistic Position x")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Posizione Prevista-Reale-X.png'))
        plt.close()

        # stampo grafico coordinata posizione y nel tempo confronto tra predizione e realtà
        fig_position = plt.subplot()
        fig_position.plot(time_array2, predicted_posBB_array[:, 1], label="Predicted Position y")
        fig_position.plot(time_array2, posGT_array[:, 1], label="Realistic Position y")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Posizione Prevista-Reale-Y.png'))
        plt.close()

        #stampo traiettoria reale e predetta:
        fig_position=plt.subplot()
        fig_position.plot(posGT_array[:,0],posGT_array[:,1],label="Realistic Position")
        fig_position.plot(predicted_posBB_array[:,0],predicted_posBB_array[:,1],label="Predicted Position")
        fig_position.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Traiettoria.png'))
        plt.close()

        # stampo grafico errore posizione nel tempo
        err_pos = []

        for i in range(len(predicted_posBB_array)):
            err_pos.append(np.sqrt((predicted_posBB_array[i, 0] - posGT_array[i][0]) ** 2 + (predicted_posBB_array[i, 1] - posGT_array[i][1]) ** 2))
        fig_err_pos = plt.subplot()
        fig_err_pos.plot(time_array2, err_pos, label="Error Position")
        fig_err_pos.set_title(current_label)
        plt.legend()
        plt.savefig(os.path.join("plots_boundingbox/" + current_label, 'Errore Posizione.png'))
        plt.close()


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
           if tmp >= 20:
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
           print("Ground truth copy: ", ground_truth_copy[i]["label"])
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
                time_array.append(0)
            else:
                A, B = m.calculate_matrix(float(ready_data[i-1]["time"]), float(ready_data[i]["time"]))
                time_array.append(float(ready_data[i]["time"]) - 20)

            center_pos = calculate_bounding_box_center(ready_data[i])
            posBB_array.append(center_pos)

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
            if (current_label == "4375"):
                print("Realistic position: ", posBB_array)
            #    print("Predicted position: ", predicted_posBB_array)
            plot_dataBB( predicted_posBB_array, predicted_velBB_array,ground_truth,current_label)
            current_label = ready_data[i]["box_label"]
            nextBB= calculate_bounding_box_center(ready_data[i])
            state = m.State(nextBB[0], nextBB[1])
            predicted_posBB_array = []
            predicted_velBB_array = []
            posBB_array = []
            time_array = []
            time_array.append(0)
            Pkminus1_kminus1 = P0















