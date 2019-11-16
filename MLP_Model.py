import numpy as np
import matplotlib.pyplot as plt
import math as math
from matplotlib.pyplot import figure
np.set_printoptions(suppress=True)

def find_output(X):
    Y= np.add(np.add(np.add(3*X[:,0:1],X[:,1:2],X[:,2:3]),np.add(X[:,3:4],np.cos(np.add(X[:,4:5],X[:,5:6])))),np.add(np.add(7*X[:,6:7],np.exp(X[:,7:8])),X[:,8:9]),4*X[:,9:10]*X[:,9:10])
    Y=activate_sigmoid(Y)
    return Y

def activate_sigmoid(sum):
    return (2/(1+np.exp(-sum))-1)

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)

def initialize_network():
    input_neurons=len(X[0])
    hidden_neurons=input_neurons+1
    output_neurons=1
    n_hidden_layers=1
    net=list()
    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])    
        hidden_layer = [ { 'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)
    output_layer = [ { 'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)
    return net

def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)           
            result=activate_sigmoid(sum)
            neuron['result']=result
            prev_input=np.append(prev_input,[result])
        row=prev_input
    return row

def sigmoidDerivative(output):
    return (0.5*(1.0-output*output))

def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results)
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])            
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])

def updateWeights(net,input,lrate):   
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]

def training(net, epochs,lrate,n_outputs):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs=forward_propagation(net,row)
            sum_error+=0.5*(math.pow((Y[i]-outputs),2))
            back_propagation(net,row,Y[i])
            updateWeights(net,row,0.05)
        errors.append(sum_error)
    return errors

def predict(network, row):
    outputs = forward_propagation(net, row)
    return outputs

X = np.random.uniform(-1,1,(1000,10))
Y = find_output(X)

net=initialize_network()

errors=training(net,100, 0.01,1)
epochs = []
for i in range(100):
    epochs.append(i)
plt.plot(epochs,errors)
plt.xlabel("epochs")
plt.ylabel('error')
textstr = 'Neural Network error = '+str(errors[99])
plt.text(-3, -3, textstr, fontsize=10)
plt.legend(['LMS Error'])
plt.show()

print_network(net)

Xpred = np.random.uniform(-1,1,(15,10))
Ypred = find_output(Xpred)
predarr=[]
for i,row in enumerate(Xpred):
    pred=predict(net,row)
    output=(pred)
    predarr.append(pred)

epochs = []
Y12 = []
X12 = []
for i in range(15):
    epochs.append(i)
    Y12.append(Ypred[i])
    X12.append(X[i])
plt.plot(epochs, Y12)
plt.plot(epochs, predarr)
plt.legend(['Actual', 'Predicted'])
plt.show()

plt.scatter(epochs, Y12)
plt.scatter(epochs, predarr)
plt.legend(['Actual', 'Predicted'])
plt.show()