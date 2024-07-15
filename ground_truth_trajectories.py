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

def setUp():
    #Create matrix A,B,C
    #A,B = calculate_matrix(0,1)
    C= [[1,0,0,0],[0,1,0,0]]
    #Create covariance matrix P0, 30m/s is the deviance standard of velocity
    P0 = [[1^2,0,0,0],[0,1^2,0,0],[0,0,30^2,0],[0,0,0,30^2]]
    #Create covariance matrix Q (process noise) it represents the noise in the system (accelleration)
    Q = [[(9.81/8)**2,0],[0,(9.81/8)**2]]
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
    # Draw Trajectory
    fig1= plt.subplot()
    fig1.plot(pos_array[:,0],pos_array[:,1],label='Predicted position')
    fig1.plot(realistic_pos_array[:,0],realistic_pos_array[:,1],label='Realistic position')
    fig1.set_title(label)
    fig1.legend()
    #plt.show()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Posizione Reale e Prevista.png'))
    plt.close()

    #Plot Predicted velocity and Realistic velocity
    fig2= plt.subplot()
    module_predicted_vel = module(vel_array)
    module_realistic_vel = module(realistic_vel_array)
    fig2.plot(time_array,module_predicted_vel,label='Predicted velocity')
    fig2.plot(time_array,module_realistic_vel,label='Realistic velocity')
    fig2.set_title(label)
    fig2.legend()
    #plt.show()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Velocità Reale e Prevista.png'))
    plt.close()

    #Plot position x and his realistic position in time
    fig3= plt.subplot()
    fig3.plot(time_array,pos_array[:,0],label='Position x')
    fig3.plot(time_array,realistic_pos_array[:,0],label='Real Position x')
    fig3.set_title(label)
    fig3.legend()
    #plt.show()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Posizione x Reale e Prevista.png'))
    plt.close()


    #Plot position y and his realistic position in time
    figy= plt.subplot()
    figy.plot(time_array,pos_array[:,1],label='Position y')
    figy.plot(time_array,realistic_pos_array[:,1],label='Real Position y')
    figy.legend()
    #plt.show()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Posizione y Reale e Prevista.png'))
    plt.close()

    #Plot velocity x and his realistic velocity in time
    fig6= plt.subplot()
    fig6.plot(time_array,vel_array[:,0],label='Velocity x')
    fig6.plot(time_array,realistic_vel_array[:,0],label='Real Velocity x')
    fig6.set_title(label)

    #Plot velocity y and his realistic velocity in time

    fig6.plot(time_array,vel_array[:,1],label='Velocity y')
    fig6.plot(time_array,realistic_vel_array[:,1],label='Real Velocity y')
    fig6.legend()
    #plt.show()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Velocità x e y Reale e Prevista.png'))
    plt.close()

    #L'ho stampato nello stesso grafico sia pos x che pos y ( stessa cosa per vel x e vel y) poichè altrimenti il
    #server non riesce a gestire tutte le richieste di plot in contemporanea

    #Plot error in position and velocity

    errpos = []
    errvel = []
    for i in range(0,len(pos_array)):
        errpos.append(np.sqrt((pos_array[i][0] - realistic_pos_array[i][0]) ** 2 + (pos_array[i][1] - realistic_pos_array[i][1]) ** 2))
        errvel.append(np.sqrt((vel_array[i][0] - realistic_vel_array[i][0]) ** 2 + (vel_array[i][1] - realistic_vel_array[i][1]) ** 2))

    fig5= plt.subplot()
    fig5.plot(time_array, errpos, label='Error in position')
    fig5.plot(time_array, errvel, label='Error in velocity')
    fig5.set_title(label)
    fig5.legend()
    plt.savefig(os.path.join("plots_pitt_trajectories/"+label, 'Errore in posizione e velocità.png'))
    plt.close()
    return

def noisy_generator():
    media = 0
    deviazione_standard = 0.2
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
    #pos_array.append([state.posX,state.posY])
    #vel_array.append([state.velX,state.velY])
    label=csv_data[0]['label']
    Pkminus1_kminus1=P0
    for i in range(0,len(csv_data)):
        lbl = csv_data[i]['label']
        if(lbl==label):
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
            #print("label:",label)
            #print("K:",K)
            #print("array:",np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])]))
            #print("array2:",np.dot(C,xk_kminus1))
            #print("differenza:",np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])])-np.dot(C,xk_kminus1))
            #print("Prodotto:", np.dot(K,(np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])])-np.dot(C,xk_kminus1))))
            z = np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])]) + noisy_generator()
            #Calculate the new state
            xk_k = xk_kminus1 + np.dot(K,(z - np.dot(C,xk_kminus1)))

            #Calculate the new covariance matrix
            Pk_k = np.dot(np.dot((np.identity(4)-np.dot(K,C)),Pk_kminus1),np.transpose(np.identity(4)-np.dot(K,C)))+np.dot(np.dot(K,R),np.transpose(K))
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



