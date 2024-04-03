import csv
import numpy as np
import matplotlib.pyplot as plt


#To-do: take the file .csv from a folder
def read_csv_in_folder():
    with open("pitt_trajectories.csv", newline='') as csv_file:
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
    Q = [[(9.81/8)**2,0,],[0,(9.81/8)**2]]
    #Create covariance matrix R (measurement noise) it represents the noise in the measurements
    #the error in the position has a deviance of  1 meter
    R = [[0.2**2,0],[0,0.2**2]] #diminuire la deviazione standard
    return P0,Q,R,C

def module(vector):
    module_vector = []
    for i in range(0,len(vector)):
        module_vector.append(np.sqrt(vector[i][0]**2+vector[i][1]**2))
    return module_vector

#we need it where the time in the csv file is not continuos
def calculate_matrix(t_in,t_fin):
    t_camp=t_fin-t_in
    t_camp = 0.1*t_camp #ogni decimo di secondo
    A = [[1,0,t_camp,0],[0,1,0,t_camp],[0,0,1,0],[0,0,0,1]]
    B = [[0.5*t_camp**2,0],[0,0.5*t_camp**2],[t_camp,0],[0,t_camp]]
    return A,B

def plot_data(pos_array,vel_array,realistic_pos_array,realistic_vel_array,label):
    pos_array=np.array(pos_array)
    vel_array=np.array(vel_array)
    realistic_pos_array=np.array(realistic_pos_array)
    realistic_vel_array=np.array(realistic_vel_array)
    iter = []
    for i in range(0,len(vel_array)):
        iter.append(i)

    fig1= plt.subplot()
    fig1.plot(pos_array[:,0],pos_array[:,1],label='Predicted position')
    fig1.plot(realistic_pos_array[:,0],realistic_pos_array[:,1],label='Realistic position')
    fig1.set_title(label)
    fig1.legend()
    plt.show()

    fig2= plt.subplot()
    module_predicted_vel = module(vel_array)
    module_realistic_vel = module(realistic_vel_array)
    fig2.plot(iter,module_predicted_vel,label='Predicted velocity')
    fig2.plot(iter,module_realistic_vel,label='Realistic velocity')
    fig2.legend()
    plt.show()

    #to-do: make the plot of error in velocity
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
    def get_state(self):
        return [self.posX,self.posY,self.velX,self.velY]
    def update_state(self,state):
        self.posX=state[0]
        self.posY=state[1]
        self.velX=state[2]
        self.velY=state[3]

def main():
    csv_data= read_csv_in_folder()
    state = State(float((csv_data[0]['x'])), float(csv_data[0]['y']))
    P0,Q,R,C = setUp()
    pos_array = []
    vel_array = []
    realistic_pos_array = []
    realistic_vel_array = []
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

            #Prediction step:
            #we dirty the position with gaussian noise
            xk_kminus1 = np.dot(A,state.get_state())
            Pk_kminus1 = np.dot(np.dot(A,Pkminus1_kminus1),np.transpose(A))+ np.dot(np.dot(B,Q),np.transpose(B))

            #Correction step:
            #Calculate the Kalman gain
            K = np.dot(np.dot(Pk_kminus1,np.transpose(C)),np.linalg.inv(np.dot(np.dot(C,Pk_kminus1),np.transpose(C))+R))
            print("label:",label)
            print("K:",K)
            print("array:",np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])]))
            print("array2:",np.dot(C,xk_kminus1))
            print("differenza:",np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])])-np.dot(C,xk_kminus1))
            print("Prodotto:", np.dot(K,(np.array([float(csv_data[i]['x']),float(csv_data[i]['y'])])-np.dot(C,xk_kminus1))))
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
           plot_data(pos_array,vel_array,realistic_pos_array,realistic_vel_array,label)
           label=csv_data[i]['label']
           state = State(float((csv_data[i]['x'])), float(csv_data[i]['y']))
           pos_array=[]
           vel_array=[]
           realistic_pos_array=[]
           realistic_vel_array=[]
           Pkminus1_kminus1=P0


main()
