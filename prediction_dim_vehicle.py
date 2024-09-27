import numpy as np
import prediction_ground_truth as m
import prediction_boundingbox as pbb
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.pyplot as plt
import math


#stampo una funzione che mi permette di fare grafici raggruppati in due cartelle a seconda dei veicoli
#Serve per il ricevimento e basta poi la cancello
def print_graphs(ground_truth, prev_data,label,time_array):
    if not os.path.exists("dimension_vehicles/type_vehicles_error"):
        os.mkdir("dimension_vehicles/type_vehicles_error")
    gt_dimentions = []
    for i in range(0,len(ground_truth)):
        if ground_truth[i][0]== label:
            for j in range(0,len(prev_data)):
                gt_dimentions.append([ground_truth[i][1],ground_truth[i][2]])

    gt_dimentions = np.array(gt_dimentions)
    prev_data = np.array(prev_data)
    if gt_dimentions[0][0] > 6.5:
        if not os.path.exists("dimension_vehicles/type_vehicles_error/bus"):
            os.mkdir("dimension_vehicles/type_vehicles_error/bus")
    else:
        if not os.path.exists("dimension_vehicles/type_vehicles_error/car"):
            os.mkdir("dimension_vehicles/type_vehicles_error/car")
    print(gt_dimentions)
    print(prev_data)
    length_diff = np.abs(gt_dimentions[:,0]-prev_data[:,0])
    width_diff = np.abs(gt_dimentions[:,1]-prev_data[:,1])
    fig1 = plt.subplot()
    fig1.plot(time_array,length_diff,label='Length')
    fig1.plot(time_array,width_diff,label='Width')
    fig1.set_title(label)
    fig1.legend()
    if gt_dimentions[0][0] > 6.5:
        plt.savefig(os.path.join("dimension_vehicles/type_vehicles_error/bus",label+'.png'))
    else:
        plt.savefig(os.path.join("dimension_vehicles/type_vehicles_error/car",label+'.png'))
    plt.close()

    fig2 = plt.subplot()
    err = np.sqrt(length_diff**2+width_diff**2)
    fig2.plot(time_array,err,label='Error')
    fig2.set_title(label)
    fig2.legend()
    if gt_dimentions[0][0] > 6.5:
        plt.savefig(os.path.join("dimension_vehicles/type_vehicles_error/bus",label+'_err.png'))
    else:
        plt.savefig(os.path.join("dimension_vehicles/type_vehicles_error/car",label+'_err.png'))
    plt.close()

# it takes the data of the csv file and calculate the length and the width of the vehicle
def calculate_length_width(data):
    # points = data

    # Arrotondare le coordinate X e Y a due cifre decimali
    # x_coords = [round(float(points[f'P{ix}x']), 2) for ix in range(1, 9)]
    # y_coords = [round(float(points[f'P{ix}y']), 2) for ix in range(1, 9)]
    # fig=plt.subplots()

    # for i in range(0, len(x_coords)):
    #    plt.plot(x_coords[i], y_coords[i], 'ro')
    #    plt.text(x_coords[i], y_coords[i], f'P{i+1}', fontsize=9)
    # plt.savefig(os.path.join("to_trash", data['time']+data['box_label'] + '.png'))
    # plt.close()

    length = math.sqrt((float(data['P2x']) - float(data['P1x'])) ** 2 + (float(data['P2y']) - float(data['P1y'])) ** 2)
    width = math.sqrt((float(data['P6x']) - float(data['P1x'])) ** 2 + (float(data['P6y']) - float(data['P1y'])) ** 2)
    return length, width

# We know the dimension of the vehicle in the ground truth, so we can calculate a prospective gain to improve the prediction
def calculate_gain_length_width(length, width, ground_truth_dim, label):
    for i in range(0, len(ground_truth_dim)):
        if ground_truth_dim[i][0] == label:
            gain_length = ground_truth_dim[i][1] / length
            gain_width = ground_truth_dim[i][2] / width
            return gain_length, gain_width

def calculate_max_length_width(data,label):
    length_max = 0
    width_max = 0
    for i in range(0, len(data)):
        if data[i]['box_label'] == label:
            length, width = calculate_length_width(data[i])
            if length > length_max:
                length_max = length
            if width > width_max:
                width_max = width
    return [length_max, width_max]

def gain_max_length_width(length_max, width_max,length,width):
    gain_length = length_max / length
    gain_width = width_max / width
    return [gain_length, gain_width]


def variant_set_up():
    # Measurement matrix, in this matrix the first two rows are the position of the center of the vehicle
    # and the last two rows are the dimension of the vehicle
    # they are the part of the state that we can measure
    C = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]

    # covariance matrix with the covariance of stime dimension of the vehicle (0.1 m)
    # più è grande il valore meno si è sicuri della stima
    P0 = [[1 ** 2, 0, 0, 0, 0, 0], [0, 1 ** 2, 0, 0, 0, 0], [0, 0, 30 ** 2, 0, 0, 0], [0, 0, 0, 30 ** 2, 0, 0],
          [0, 0, 0, 0, 0.2 ** 2, 0], [0, 0, 0, 0, 0, 0.2 ** 2]]

    # covariance matrix Q (process noise) it represents the noise in the system (accelleration) i.e it represents
    # the systematic or random uncertainties that are not explicitly modeled in the system.
    Q = [[(9.81 / 8) ** 2, 0], [0, (9.81 / 8) ** 2]]

    # Create covariance matrix R (measurement noise) it represents the noise in the measurements
    # the error in the measure of dimension of the veicole has a deviance of  0.1 meter
    R = [[0.1 ** 2, 0, 0, 0], [0, 0.1 ** 2, 0, 0], [0, 0, 0.2 ** 2, 0], [0, 0, 0, 0.2 ** 2]]

    return P0, Q, R, C


def plot_dimention_vehicle(pred_dimention_vehicle_array,pred_dimention_vehicle_array1,pred_dimention_vehicle_array2, ground_truth_dim, time_array, label,title):
    if not os.path.exists(f"dimension_vehicles/{title}"):
        os.makedirs(f"dimension_vehicles/{title}", exist_ok=True)
    if not os.path.exists(f"dimension_vehicles/{title}/{label}"):
        os.makedirs(f"dimension_vehicles/{title}/{label}", exist_ok=True)
    # To see if there is a variable dimention (it's not possible)
    pred_dimention_vehicle_array = np.array(pred_dimention_vehicle_array)
    pred_dimention_vehicle_array1 = np.array(pred_dimention_vehicle_array1)
    pred_dimention_vehicle_array2 = np.array(pred_dimention_vehicle_array2)
    #ground_truth_dim = np.array(ground_truth_dim)
    #fig1 = plt.subplot()
    #fig1.plot(time_array, pred_dimention_vehicle_array[:, 0], label='Length')
    #fig1.plot(time_array, pred_dimention_vehicle_array[:, 1], label='Width')
    #fig1.set_title(label)
    #fig1.legend()

    #if not os.path.exists("dimension_vehicles/" + label):
    #    os.mkdir("dimension_vehicles/" + label)

    #plt.savefig(os.path.join("dimension_vehicles/" + label, 'Dimensione veicolo.png'))
    #plt.close()

    # Plot the predicted dimension and the ground truth dimension

    gt_dimention = []
    for i in range(0, len(ground_truth_dim)):
        if ground_truth_dim[i][0] == label:
            for j in range(0, len(pred_dimention_vehicle_array)):
                gt_dimention.append([ground_truth_dim[i][1], ground_truth_dim[i][2]])
            break

    gt_dimention = np.array(gt_dimention)
    m.save_plot([time_array], [gt_dimention[:,0],gt_dimention[:,1],pred_dimention_vehicle_array[:, 0], pred_dimention_vehicle_array[:, 1],
                               pred_dimention_vehicle_array1[:, 0],
                               pred_dimention_vehicle_array1[:, 1], pred_dimention_vehicle_array2[:, 0],
                               pred_dimention_vehicle_array2[:, 1]],
                label,
                ['Lung. realistica','Larg. realistica','Lung. predetta senza Guadagno', 'Larg. predetta senza Guadagno', 'Lung. predetta Guadagno su GT',
                 'Larg predetta Guadagno su GT', 'Lung. predetta Guadagno su MaxDim',
                 'Larg. predetta Guadagno su MaxDim'], f"dimension_vehicles/{title}/{label}/Dimensione veicolo.png")

    #m.save_plot([time_array],[gt_dimention[:,0],pred_dimention_vehicle_array[:,0]],label,['GT Length','Predicted Length'],f"dimension_vehicles/{title}/{label}/Predicted&GT Length.png")
    m.save_plot([time_array],[gt_dimention[:,0],pred_dimention_vehicle_array[:,0],pred_dimention_vehicle_array1[:,0],pred_dimention_vehicle_array2[:,0]],label,
                ['Lung. realistica','Lung. predetta senza Guadagno','Lung. predetta con Guadagno su GT','Lung. predetta con Guadagno su DimMax'],f"dimension_vehicles/{title}/{label}/Predicted&GT Length.png")
    #fig2 = plt.subplot()
    #fig2.plot(time_array, gt_dimention[:, 0], label='GT Length')
    #fig2.plot(time_array, pred_dimention_vehicle_array[:, 0], label='Predicted Length')
    #fig2.set_title(label)
    #fig2.legend()
    #plt.savefig(os.path.join("dimension_vehicles/" + label, 'Predicted&GT Length.png'))
    #plt.close()

    #m.save_plot([time_array],[gt_dimention[:,1],pred_dimention_vehicle_array[:,1]],label,['GT Width','Predicted Width'],f"dimension_vehicles/{title}/{label}/Predicted&GT Width.png")
    m.save_plot([time_array],[gt_dimention[:,1],pred_dimention_vehicle_array[:,1],pred_dimention_vehicle_array1[:,1],pred_dimention_vehicle_array2[:,1]],label,
                ['Larg. realistica','Larg. predetta senza Guadagno','Larg. predetta Guadagno su GT','Larg. predetta Guadagno su DimMax'],f"dimension_vehicles/{title}/{label}/Predicted&GT Width.png")
    #fig3 = plt.subplot()
    #fig3.plot(time_array, gt_dimention[:, 1], label='GT Width')
    #fig3.plot(time_array, pred_dimention_vehicle_array[:, 1], label='Predicted Width')
    #fig3.set_title(label)
    #fig3.legend()
    #plt.savefig(os.path.join("dimension_vehicles/" + label, 'Predicted&GT Width.png'))
    #plt.close()

    # Plot the difference between the predicted dimension and the ground truth dimension
    gt_dimention = gt_dimention.astype(float)
    pred_dimention_vehicle_array = pred_dimention_vehicle_array.astype(float)
    diff_length = np.abs(gt_dimention[:, 0] - pred_dimention_vehicle_array[:, 0])
    diff_width = np.abs(gt_dimention[:, 1] - pred_dimention_vehicle_array[:, 1])
    diff_length1 = np.abs(gt_dimention[:, 0] - pred_dimention_vehicle_array1[:, 0])
    diff_width1 = np.abs(gt_dimention[:, 1] - pred_dimention_vehicle_array1[:, 1])
    diff_length2 = np.abs(gt_dimention[:, 0] - pred_dimention_vehicle_array2[:, 0])
    diff_width2 = np.abs(gt_dimention[:, 1] - pred_dimention_vehicle_array2[:, 1])
    m.save_plot([time_array],[diff_length,diff_width,diff_length1,diff_width1,diff_length2,diff_width2],label,
                ['Errore Lung. senza Guadagno','Errore Larg. senza Guadagno','Errore Lung. Guadagno su GT','Errore Larg. Guadagno su GT','Errore Lung. Guadagno su MaxDim','Errore Larg. Guadagno su MaxDim'],f"dimension_vehicles/{title}/{label}/Error.png")
    #m.save_plot([time_array],[diff_length,diff_width],label,['Difference Length','Difference Width'],f"dimension_vehicles/{title}/{label}/Error.png")
    #fig4 = plt.subplot()
    #fig4.plot(time_array, diff_length, label='Difference Length')
    #fig4.plot(time_array, diff_width, label='Difference Width')
    #fig4.set_title(label)
    #fig4.legend()
    #plt.savefig(os.path.join("dimension_vehicles/" + label, 'Error.png'))
    #plt.close()


def main():
    if not os.path.exists("dimension_vehicles"):
        os.mkdir("dimension_vehicles")

    csv_data = m.read_csv_in_folder("dataBB_definitivo_corto.csv")
    gt_data = m.read_csv_in_folder("pitt_trajectories.csv")
    label_list,First_Data_v,GT = pbb.recognize_label(csv_data, gt_data)

    csv_data = pbb.replace_label(csv_data, label_list)
    # print(csv_data[0])
    pos0 = pbb.calculate_bounding_box_center(csv_data[0])
    state = m.State(float(pos0[0]), float(pos0[1]))
    state1= m.State(float(pos0[0]), float(pos0[1]))
    state2= m.State(float(pos0[0]), float(pos0[1]))
    current_label = csv_data[0]['box_label']
    ground_truth_dim = [["3886", 6.5, 1.66], ["4287", 6.5, 1.66], ["4257", 10.89, 2.94], ["3135", 6.5, 1.66],
                        ["4375", 6.5, 1.66], ["AV", 6.5, 1.66]]
    length, width = calculate_length_width(csv_data[0])
    gain = calculate_gain_length_width(length, width, ground_truth_dim, current_label)
    dim_max = calculate_max_length_width(csv_data, current_label)

    gain_max = gain_max_length_width(dim_max[0],dim_max[1],length,width)


    P0, Q, R, C = variant_set_up()
    Q = np.array(Q)
    R = np.array(R)
    state.update_state_dim(length, width)
    state1.update_state_dim(length * gain[0], width * gain[1])
    state2.update_state_dim(length * gain_max[0], width * gain_max[1])

    time_array = []
    pred_dimention_vehicle_array = []
    pred_dimention_vehicle_array1 = []
    pred_dimention_vehicle_array2 = []

    Pkminus1_kminus1 = P0

    for i in range(0, len(csv_data)):
        if csv_data[i]['box_label'] == current_label and i != len(csv_data) - 1:
            if i == 0:
                A, B = m.calculate_matrix(0, 1)
                A = np.array(A)
                B = np.array(B)
                time_array.append(0)
            else:
                A, B = m.calculate_matrix(float(csv_data[i - 1]['time']), float(csv_data[i]['time']))
                time_array.append(float(csv_data[i]['time']) - 20)
                A = np.array(A)
                B = np.array(B)
            center_pos = pbb.calculate_bounding_box_center(csv_data[i])

            # we calculate length and width of the vehicle
            length, width = calculate_length_width(csv_data[i])
            # we calculate the gain associated to the dimentions of the vehicle
            gain = calculate_gain_length_width(length, width, ground_truth_dim, current_label)
            gain_max = gain_max_length_width(dim_max[0],dim_max[1],length,width)
            # we multiply the length and the width of the vehicle for the gain
            length1 = length * gain[0]
            width1 = width * gain[1]
            length2 = length * gain_max[0]
            width2 = width * gain_max[1]

            # Predict the state

            # Calculate A_tilde
            A_tilde = np.block([[A, np.zeros((4, 2))], [np.zeros((2, 4)), np.eye(2)]])

            # Calculate Q_tilde
            # in the previous version of the code the Q_tilde wasn't calculated
            # because the process rumor in this case is moduled by the matrix B
            B_Q_B_t = np.dot(B, np.dot(Q, np.transpose(B)))
            W = np.eye(2, 2) * (0.1) ** 2
            Q_tilde = np.block([[B_Q_B_t, np.zeros((4, 2))], [np.zeros((2, 4)), W]])

            xk_kminus1 = np.dot(A_tilde, state.get_state_dim())
            xk_kminus1_1 = np.dot(A_tilde, state1.get_state_dim())
            xk_kminus1_2 = np.dot(A_tilde, state2.get_state_dim())
            Pk_kminus1 = np.dot(np.dot(A_tilde, Pkminus1_kminus1), np.transpose(A_tilde)) + Q_tilde
            # Correction step:
            # Calculate the Kalman gain
            K = np.dot(np.dot(Pk_kminus1, np.transpose(C)),
                       np.linalg.inv(np.dot(np.dot(C, Pk_kminus1), np.transpose(C)) + R))
            # In questo caso z è il vettore di misura
            # Si ha due misure: la posizione del centro del veicolo e la dimensione del veicolo
            # Quindi si hanno due errori
            dim = [length, width]
            z = np.array(center_pos) + m.noisy_generator(0.2)
            dim = np.array(dim) + m.noisy_generator(0.2)
            measure = np.array([z[0], z[1], dim[0], dim[1]])
            dim1 = [length1, width1]
            z1 = np.array(center_pos) + m.noisy_generator(0.2)
            dim1 = np.array(dim1) + m.noisy_generator(0.2)
            measure1 = np.array([z1[0], z1[1], dim1[0], dim1[1]])
            dim2 = [length2, width2]
            z2 = np.array(center_pos) + m.noisy_generator(0.2)
            dim2 = np.array(dim2) + m.noisy_generator(0.2)
            measure2 = np.array([z2[0], z2[1], dim2[0], dim2[1]])

            # Il commento sottostante non è corretto, poichè io devo avere nel mio vettore di misura
            # la posizione del centro del veicolo e la dimensione del veicolo
            # metto a zero poichè non ho la misura della velocità e della dimensione del veicolo
            # z= np.array([z[0],z[1],0,0])
            z = np.array(measure)
            z1 = np.array(measure1)
            z2 = np.array(measure2)

            # Calculate the new state
            xk_k = xk_kminus1 + np.dot(K, (z - np.dot(C, xk_kminus1)))
            xk_k1= xk_kminus1_1 + np.dot(K, (z1 - np.dot(C, xk_kminus1_1)))
            xk_k2= xk_kminus1_2 + np.dot(K, (z2 - np.dot(C, xk_kminus1_2)))

            #print("Residuo: ", z - np.dot(C, xk_kminus1))
            #print("Predicted state: ", xk_k[4], xk_k[5])

            # Calculate the new covariance matrix
            Pk_k = np.dot(np.dot((np.identity(6) - np.dot(K, C)), Pk_kminus1),
                          np.transpose(np.identity(6) - np.dot(K, C))) + np.dot(np.dot(K, R), np.transpose(K))
            # update state
            state.update_all_state(xk_k)
            state1.update_all_state(xk_k1)
            state2.update_all_state(xk_k2)
            # Update the covariance matrix
            Pkminus1_kminus1 = Pk_k
            dim = state.get_state_dim()
            dim1 = state1.get_state_dim()
            dim2 = state2.get_state_dim()
            #gt_dimention_vehicle_array.append(calculate_length_width(csv_data[i]))
            pred_dimention_vehicle_array.append([dim[4], dim[5]])
            pred_dimention_vehicle_array1.append([dim1[4], dim1[5]])
            pred_dimention_vehicle_array2.append([dim2[4], dim2[5]])
        else:
            print("Label: ", current_label)
            print_graphs(ground_truth_dim, pred_dimention_vehicle_array, current_label, time_array)
            plot_dimention_vehicle(pred_dimention_vehicle_array,pred_dimention_vehicle_array1,pred_dimention_vehicle_array2, ground_truth_dim, time_array, current_label,"ConfrontoGuadagni")
            #plot_dimention_vehicle(pred_dimention_vehicle_array1, ground_truth_dim, time_array, current_label,"ConGuadagno")
            #plot_dimention_vehicle(pred_dimention_vehicle_array2, ground_truth_dim, time_array, current_label,"MaxGuadagno")
            current_label = csv_data[i]['box_label']
            nextBB = pbb.calculate_bounding_box_center(csv_data[i])
            #IMPOSTO COME STATO INIZIALE IL CENTRO DELLA BOUNDING BOX
            state = m.State(nextBB[0], nextBB[1])
            state1 = m.State(nextBB[0], nextBB[1])
            state2 = m.State(nextBB[0], nextBB[1])
            length, width = calculate_length_width(csv_data[i])

            #RICALCOLO DEI GUADAGNI A NUOVO VEICOLO
            dim_max = calculate_max_length_width(csv_data, current_label)
            gain_max = gain_max_length_width(dim_max[0],dim_max[1],length,width)
            gain = calculate_gain_length_width(length, width, ground_truth_dim, current_label)
            state.update_state_dim(length, width)
            state1.update_state_dim(length * gain[0], width * gain[1])
            state2.update_state_dim(length * gain_max[0], width * gain_max[1])
            #RESET PREDIZIONI
            pred_dimention_vehicle_array = []
            pred_dimention_vehicle_array1 = []
            pred_dimention_vehicle_array2 = []
            #RESET TEMPO
            time_array = []
            Pkminus1_kminus1 = P0
