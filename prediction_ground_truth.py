import csv
import os
import numpy as np
import matplotlib.pyplot as plt
def read_csv_in_folder(dir):
    with open(dir, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = []
            for row in csv_reader:
                data.append(row)
    return data

def print_velocity(data):
    velocity = []
    current_label = data[0]['label']
    len_min=0
    lung=0
    for i in range(0,len(data)):
        print(data[i]['label'], "lung:", lung)
        if data[i]['label'] == current_label and i!=len(data)-1:
            lung+=1
        else:
            if lung < len_min or len_min==0:
                len_min=lung
            current_label = data[i]['label']
            lung=0
        print("len min",len_min)
    tmp=0
    for i in range(0,len(data)):
        if data[i]['label'] == current_label :
            if tmp<len_min:
                velocity.append([current_label,data[i]['vx'],data[i]['vy']])
                tmp+=1
        else:
            current_label = data[i]['label']
            tmp=0
    time=[]
    for i in range(0,len_min):
        time.append(i)

    fig = plt.figure()
    lbl = velocity[0][0]
    vel_array = []
    for i in range(0,len(velocity)):
        if velocity[i][0] == lbl and i!=len(velocity):
            vel_array.append([float(velocity[i][1]),float(velocity[i][2])])
        else:
            if len(vel_array)==31:
                vel_array.append([float(velocity[i][1]), float(velocity[i][2])])
            elif len(vel_array)==30:
                vel_array.append([float(velocity[i][1]), float(velocity[i][2])])
                #vel_array.append([float(velocity[i+1][1]), float(velocity[i+1][2])])


            vel_array = np.array(vel_array)
            for j in range(0,len(vel_array)):
                vel_array[j]=np.sqrt(vel_array[j][0]**2+vel_array[j][1]**2)
            plt.plot(time,vel_array,label=lbl)
            lbl = velocity[i][0]
            vel_array = []

    if len(vel_array) == 31:
        vel_array.append([float(velocity[i][1]), float(velocity[i][2])])
    if lbl == 'AV':
        vel_array = np.array(vel_array)
        for j in range(0, len(vel_array)):
            vel_array[j] = np.sqrt(vel_array[j][0] ** 2 + vel_array[j][1] ** 2)
        plt.plot(time, vel_array, label=lbl)

    plt.legend()
    plt.show()


def calculate_error(predicted_array, gt_array):
    return np.sqrt(np.sum((predicted_array - gt_array) ** 2, axis=1))

def save_plot(x, y, title, labels, save_path):
    plt.figure()
    if len(x) == len(y):
        for i in range(0, len(x)):
            plt.plot(x[i], y[i], label=labels[i])
    elif len(x) == 1:

        for i in range(0, len(y)):
            plt.plot(x[0],y[i], label=labels[i])

    plt.title(title)
    plt.legend(fontsize=8)
    plt.savefig(save_path)
    plt.close()




def setUp():
    #Create matrix A,B,C
    #A,B = calculate_matrix(0,1)
    C= [[1,0,0,0],[0,1,0,0]]
    #Create covariance matrix P0, 30m/s is the deviance standard of velocity
    P0 = [[1^2,0,0,0],[0,1^2,0,0],[0,0,30^2,0],[0,0,0,30^2]]
    #Create covariance matrix Q (process noise) it represents the noise in the system (accelleration)
    Q = [[(9.81/7)**2,0],[0,(9.81/7)**2]]
    #Create covariance matrix R (measurement noise) it represents the noise in the measurements
    #the error in the position has a deviance of  0.2 meter
    R = [[0.2**2,0],[0,0.2**2]] #diminuire la deviazione standard
    return P0,Q,R,C

def module(vector):
    module_vector = []
    for i in range(0,len(vector)):
        module_vector.append(np.sqrt(vector[i][0]**2+vector[i][1]**2))
    return module_vector

#we need it when time in the csv file is not continuos
def calculate_matrix(t_in,t_fin):
    t_camp=t_fin-t_in
    t_camp = 0.1*t_camp #ogni decimo di secondo
    A = [[1,0,t_camp,0],[0,1,0,t_camp],[0,0,1,0],[0,0,0,1]]
    B = [[0.5*t_camp**2,0],[0,0.5*t_camp**2],[t_camp,0],[0,t_camp]]
    return A,B

def plot_data(pos_array,vel_array,realistic_pos_array,realistic_vel_array,label,time_array):
    pos_array=np.array(pos_array)
    vel_array=np.array(vel_array)
    realistic_pos_array=np.array(realistic_pos_array)
    realistic_vel_array=np.array(realistic_vel_array)

    if not os.path.exists("plots_pitt_trajectories/"+label):
        os.mkdir("plots_pitt_trajectories/"+label)

    #Plot Predicted position and Realistic position
    save_plot([pos_array[:,0], realistic_pos_array[:,0]], [pos_array[:,1], realistic_pos_array[:,1]], label,
              ["Predicted Trajectory", "Real Trajectory"], os.path.join("plots_pitt_trajectories/"+label, 'Posizione Reale e Prevista.png'))

    #Plot Predicted velocity and Realistic velocity
    save_plot([time_array], [module(vel_array), module(realistic_vel_array)], label, ["Predicted ", "Real"],
              os.path.join("plots_pitt_trajectories/"+label, 'Velocità Reale e Prevista.png'))

    #Plot position x and his realistic position in time
    save_plot([time_array], [pos_array[:,0], realistic_pos_array[:,0]], label, ["Predicted Position x", "Real Position x"],
              os.path.join("plots_pitt_trajectories/"+label, 'Posizione x Reale e Prevista.png'))

    #Plot position y and his realistic position in time
    save_plot([time_array], [pos_array[:,1], realistic_pos_array[:,1]], label, ["Predicted Position y", "Real Position y"],
              os.path.join("plots_pitt_trajectories/"+label, 'Posizione y Reale e Prevista.png'))

    #Plot velocity x and his realistic velocity in time
    save_plot([time_array], [vel_array[:,0], realistic_vel_array[:,0],vel_array[:,1],realistic_vel_array[:,1]], label,
              ["Predicted vel. x", "Real vel. x","Predicted vel. y", "Real vel. y"],
              os.path.join("plots_pitt_trajectories/"+label, 'Velocità x e y Reale e Prevista.png'))

    #Plot error in position and velocity
    errpos=calculate_error(pos_array, realistic_pos_array)
    errvel=calculate_error(vel_array, realistic_vel_array)
    save_plot([time_array], [errpos, errvel], label, ["Error in position", "Error in velocity"],
              os.path.join("plots_pitt_trajectories/"+label, 'Errore in posizione e velocità.png'))
    return

def noisy_generator(dev):
    media = 0
    deviazione_standard = dev
    dimensione_di_campionamenti = 2
    r= np.random.normal(media, deviazione_standard, dimensione_di_campionamenti)
    rumore=[r[0],r[1]]
    return rumore

class State:
    def __init__(self,posX,posY):
        self.posX=posX
        self.posY=posY
        self.velX=0
        self.velY=0
        self.length=0
        self.width=0
    def update_all_state(self,state):
        self.posX=state[0]
        self.posY=state[1]
        self.velX=state[2]
        self.velY=state[3]
        self.length=state[4]
        self.width=state[5]
    def get_state(self):
        return [self.posX,self.posY,self.velX,self.velY]
    def update_state(self,state):
        self.posX=state[0]
        self.posY=state[1]
        self.velX=state[2]
        self.velY=state[3]

    def update_state_dim(self,length,width):
        self.length=length
        self.width=width

    def get_state_dim(self):
        return [self.posX,self.posY,self.velX,self.velY,self.length,self.width]

def main():
    csv_data= read_csv_in_folder("pitt_trajectories.csv")
    state = State(float((csv_data[0]['x'])), float(csv_data[0]['y']))
    P0,Q,R,C = setUp()
    pos_array = []
    vel_array = []
    realistic_pos_array = []
    realistic_vel_array = []
    time_array = []
    print_velocity(csv_data)
    #pos_array.append([state.posX,state.posY])
    #vel_array.append([state.velX,state.velY])
    label=csv_data[0]['label']
    Pkminus1_kminus1=P0
    for i in range(0,len(csv_data)):
        if(csv_data[i]['label']==label):
            if(i==0):
                A,B=calculate_matrix(0,float(csv_data[i]['time']))
            else:
                A,B=calculate_matrix(float(csv_data[i-1]['time']),float(csv_data[i]['time']))

            realistic_pos_array.append([float(csv_data[i]['x']),float(csv_data[i]['y'])])
            realistic_vel_array.append([float(csv_data[i]['vx']),float(csv_data[i]['vy'])])
            time_array.append(float(csv_data[i]['time']))

            #Prediction step:
            #we dirty the position with gaussian noise
            xk_kminus1 = np.dot(A,state.get_state())
            Pk_kminus1 = np.dot(np.dot(A,Pkminus1_kminus1),np.transpose(A))+ np.dot(np.dot(B,Q),np.transpose(B))

            #Correction step:
            #Calculate the Kalman gain
            K = np.dot(np.dot(Pk_kminus1,np.transpose(C)),np.linalg.inv(np.dot(np.dot(C,Pk_kminus1),np.transpose(C))+R))
            z = np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])]) + noisy_generator(0.2)
            #Calculate the new state
            xk_k = xk_kminus1 + np.dot(K,(z - np.dot(C,xk_kminus1)))

            #Calculate the new covariance matrix
            Pk_k = np.dot(np.dot((np.identity(4)-np.dot(K,C)),Pk_kminus1),np.transpose(np.identity(4)-np.dot(K,C)))\
                   +np.dot(np.dot(K,R),np.transpose(K))
            #Update all the variables for the next iteration
            state.update_state(xk_k)
            Pkminus1_kminus1 = Pk_k
            #save position and velocity:
            pos_array.append([state.posX,state.posY])
            vel_array.append([state.velX,state.velY])
        else:
           #Plot data of previous label and reset all variabiles
           if (label=='AV'):
               print('AV')
               print(pos_array)
           plot_data(pos_array,vel_array,realistic_pos_array,realistic_vel_array,label,time_array)
           label=csv_data[i]['label']
           state = State(float((csv_data[i]['x'])), float(csv_data[i]['y']))
           pos_array=[]
           vel_array=[]
           time_array=[]
           realistic_pos_array=[]
           realistic_vel_array=[]
           Pkminus1_kminus1=P0
           A, B = calculate_matrix(0, 1)

    #Il server che gestisce la visualizzazione dei grafici non riesce a gestire tutte le richieste di plot in contemporanea
    plot_data(pos_array, vel_array, realistic_pos_array, realistic_vel_array, label, time_array)



