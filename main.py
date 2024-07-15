import matplotlib.pyplot as plt
import prediction_dim_vehicle as pdv
import prediction_boundingbox as pbb
import ground_truth_trajectories as m

exit = False
while not exit:
    choice = input("Enter choice (BB/GT/DIM): ")
    if choice == "BB":
        pbb.main()
    elif choice == "GT":
        m.main()
        plt.close('all')
    elif choice == "DIM":
        pdv.main()
    else:
        print("Invalid choice")
    print("Do you want to exit? (yes/no)")
    exit_choice = input()
    if exit_choice == "yes":
        exit = True