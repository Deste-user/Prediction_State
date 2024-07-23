import numpy as np
import ground_truth_trajectories as m
import prediction_boundingbox as pbb
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.pyplot as plt
import math

#it takes the data of the csv file and calculate the length and the width of the vehicle
def calculate_length_width(data):
    points = data

    # Arrotondare le coordinate X e Y a due cifre decimali
    x_coords = [round(float(points[f'P{ix}x']), 2) for ix in range(1, 9)]
    y_coords = [round(float(points[f'P{ix}y']), 2) for ix in range(1, 9)]
    fig=plt.subplots()

    for i in range(0, len(x_coords)):
        plt.plot(x_coords[i], y_coords[i], 'ro')
        plt.text(x_coords[i], y_coords[i], f'P{i+1}', fontsize=9)
    #plt.show()
    plt.close()

    length = math.sqrt((float(data['P2x']) - float(data['P1x']))**2 + (float(data['P2y']) - float(data['P1y']))**2)
    width = math.sqrt((float(data['P3x'])-float(data['P1x']))**2 + (float(data['P3y'])-float(data['P1y']))**2)
    print("Length: ", length)
    print("Width: ", width)
    return length, width



def variant_set_up():
   # Measurement matrix, in this matrix the first two rows are the position of the center of the vehicle
   # and the last two rows are the dimension of the vehicle
   #they are the part of the state that we can measure
   C =[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]

   #covariance matrix with the covariance of stime dimension of the vehicle (0.1 m)
   #più è grande il valore meno si è sicuri della stima
   P0 =[[1**2,0,0,0,0,0],[0,1**2,0,0,0,0],[0,0,30**2,0,0,0],[0,0,0,30**2,0,0],[0,0,0,0,0.1**2,0],[0,0,0,0,0,0.1**2]]

   #covariance matrix Q (process noise) it represents the noise in the system (accelleration) i.e it represents
   # the systematic or random uncertainties that are not explicitly modeled in the system.
   Q = [[(9.81/8)**2,0],[0,(9.81/8)**2]]

   # Create covariance matrix R (measurement noise) it represents the noise in the measurements
   # the error in the measure of dimension of the veicole has a deviance of  0.1 meter
   R=[[0.1**2,0,0,0],[0,0.1**2,0,0],[0,0,0.1**2,0],[0,0,0,0.1**2]]

   return P0,Q,R,C

   
def plot_dimention_vehicle(pred_dimention_vehicle_array,gt_dimention_vehicle_array,time_array,label):
    #TODO: NON TORNANO MOLTO I GRAFICI RIGUARDARE!!!
    #To see if there is a variable dimention (it's not possible)
    pred_dimention_vehicle_array = np.array(pred_dimention_vehicle_array)
    fig1 = plt.subplot()
    fig1.plot(time_array,pred_dimention_vehicle_array[:,0],label='Length')
    fig1.plot(time_array,pred_dimention_vehicle_array[:,1],label='Width')
    fig1.set_title(label)
    fig1.legend()

    if not os.path.exists("dimension_vehicles/"+label):
        os.mkdir("dimension_vehicles/"+label)

    plt.savefig(os.path.join("dimension_vehicles/"+label, 'Dimensione veicolo.png'))
    plt.close()

    #Plot the predicted dimension and the ground truth dimension
    gt_dimention_vehicle_array = np.array(gt_dimention_vehicle_array)
    fig2 = plt.subplot()
    fig2.plot(time_array,gt_dimention_vehicle_array[:,0],label='GT Length')
    fig2.plot(time_array,pred_dimention_vehicle_array[:,0],label='Predicted Length')
    fig2.set_title(label)
    fig2.legend()
    plt.savefig(os.path.join("dimension_vehicles/"+label, 'Predicted&GT Length.png'))
    plt.close()

    fig3 = plt.subplot()
    fig3.plot(time_array,gt_dimention_vehicle_array[:,1],label='GT Width')
    fig3.plot(time_array,pred_dimention_vehicle_array[:,1],label='Predicted Width')
    fig3.set_title(label)
    fig3.legend()
    plt.savefig(os.path.join("dimension_vehicles/"+label, 'Predicted&GT Width.png'))
    plt.close()

    #Plot the difference between the predicted dimension and the ground truth dimension
    diff_length = np.abs(gt_dimention_vehicle_array[:,0]-pred_dimention_vehicle_array[:,0])
    diff_width = np.abs(gt_dimention_vehicle_array[:,1]-pred_dimention_vehicle_array[:,1])
    fig4 = plt.subplot()
    fig4.plot(time_array,diff_length,label='Difference Length')
    fig4.plot(time_array,diff_width,label='Difference Width')
    fig4.set_title(label)
    fig4.legend()
    plt.savefig(os.path.join("dimension_vehicles/"+label, 'Error.png'))
    plt.close()


def main():
    if not os.path.exists("dimension_vehicles"):
        os.mkdir("dimension_vehicles")

    csv_data = m.read_csv_in_folder("dataBB_definitivo_corto.csv")
    #print(csv_data[0])
    pos0 = pbb.calculate_bounding_box_center(csv_data[0])
    state = m.State(float(pos0[0]), float(pos0[1]))
    length,width =calculate_length_width(csv_data[0])
    P0,Q,R,C = variant_set_up()
    Q = np.array(Q)
    R = np.array(R)
    state.update_state_dim(length,width)
    current_label= csv_data[0]['box_label']
    time_array = []
    pred_dimention_vehicle_array = []
    gt_dimention_vehicle_array = []
    Pkminus1_kminus1 = P0

    for i in range(1,len(csv_data)):
        if csv_data[i]['box_label']== current_label:
            if i==0:
                A,B = m.calculate_matrix(0,1)
                A=np.array(A)
                B=np.array(B)
            else:
                A,B = m.calculate_matrix(float(csv_data[i-1]['time']),float(csv_data[i]['time']))
                time_array.append(float(csv_data[i]['time'])-20)
                A=np.array(A)
                B=np.array(B)
            center_pos = pbb.calculate_bounding_box_center(csv_data[i])
            #TODO:PROVO

            length,width=calculate_length_width(csv_data[i])
            #posBB_array.append(center_pos)
            #Predict the state

            #Calculate A_tilde
            A_tilde = np.block([[A, np.zeros((4, 2))], [np.zeros((2, 4)), np.eye(2)]])

            #Calculate Q_tilde
            #in the previous version of the code the Q_tilde wasn't calculated
            # because the process rumor in this case is moduled by the matrix B #TODO:(?)
            B_Q_B_t = np.dot(B,np.dot(Q,np.transpose(B)))
            W=np.eye(2,2)*(0.001)**2
            Q_tilde= np.block([[B_Q_B_t,np.zeros((4,2))],[np.zeros((2,4)), W]])

            xk_kminus1 = np.dot(A_tilde,state.get_state_dim())
            Pk_kminus1 = np.dot(np.dot(A_tilde,Pkminus1_kminus1),np.transpose(A_tilde))+Q_tilde
            #Correction step:
            #Calculate the Kalman gain
            K = np.dot(np.dot(Pk_kminus1,np.transpose(C)),np.linalg.inv(np.dot(np.dot(C,Pk_kminus1),np.transpose(C))+R))
            #In questo caso z è il vettore di misura
            #Si ha due misure: la posizione del centro del veicolo e la dimensione del veicolo
            #Quindi si hanno due errori
            dim= [length,width]
            z= np.array(center_pos) + m.noisy_generator()
            dim= np.array(dim)+m.noisy_generator()
            measure= np.array([z[0],z[1],dim[0],dim[1]])
            #Il commento sottostante non è corretto, poichè io devo avere nel mio vettore di misura
            #la posizione del centro del veicolo e la dimensione del veicolo
            #metto a zero poichè non ho la misura della velocità e della dimensione del veicolo
            #z= np.array([z[0],z[1],0,0])
            z=np.array(measure)

            #Calculate the new state
            xk_k = xk_kminus1 + np.dot(K,(z - np.dot(C,xk_kminus1)))

            #Calculate the new covariance matrix
            Pk_k = np.dot(np.dot((np.identity(6)-np.dot(K,C)),Pk_kminus1),np.transpose(np.identity(6)-np.dot(K,C)))+np.dot(np.dot(K,R),np.transpose(K))
            #update state
            state.update_all_state(xk_k)
            #Update the covariance matrix
            Pkminus1_kminus1 = Pk_k
            dim = state.get_state_dim()
            gt_dimention_vehicle_array.append(calculate_length_width(csv_data[i]))
            pred_dimention_vehicle_array.append([dim[4],dim[5]])
        else:
            #print(pred_dimention_vehicle_array)
            #print(gt_dimention_vehicle_array)
            plot_dimention_vehicle(pred_dimention_vehicle_array,gt_dimention_vehicle_array,time_array,current_label)
            current_label = csv_data[i]['box_label']
            nextBB=calculate_length_width(csv_data[i])
            state = m.State(nextBB[0],nextBB[1])
            length,width= calculate_length_width(csv_data[i])
            state.update_state_dim(length,width)
            #TODO: vediamo se fare length0 e width0 a parte o no(per adesso sembra di no)
            pred_dimention_vehicle_array = []
            gt_dimention_vehicle_array = []
            time_array = []
            Pkminus1_kminus1 = P0