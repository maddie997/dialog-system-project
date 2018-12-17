#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import sys

if (len(sys.argv) != 2):
    sys.exit("Expect 1 argument: Usage udc_process.py <log file>")
    
x, y1, y2, y3, y4, y5 = np.genfromtxt(sys.argv[1], delimiter=',', unpack=True)
#plt.plot(x,y1, label='Recall 1 in 10')

#approach1 - just plot one at a time
approach1 = False

if approach1:
    plt.xlabel('Number of Epochs')
    plt.ylabel('Recall@1')
    plt.title('Recall@1')
    plt.grid()
    plt.legend()
    plt.show()
else:
    fig, ax = plt.subplots()
    plt.grid()
    plt.xlabel('Number of Epochs')
    plt.ylabel('Recall Value')

    #Limit Range of x axis values
    ax.set_xlim(right=100000)

    #Plot the recall values for @1, @2, @5 and loss with epochs.
    ax.plot(x, y1, 'k--', color='green', label='Recall@1')
    ax.plot(x, y2, 'k:', color='blue', label='Recall@2')
    ax.plot(x, y3, 'k', color='orange', label='Recall@5')
    ax.plot(x, y5*0.25, 'k', color='black' , label='Loss')

    legend = ax.legend(loc='lower right', shadow=True, fontsize='large')

    # Put a nicer background color on the legend.
    #legend.get_frame().set_facecolor('C0')
    plt.show()

