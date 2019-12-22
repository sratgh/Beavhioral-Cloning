
import matplotlib.pyplot as plt
import pickle


history = pickle.load(open("history.p", "rb"))
print(history)

# Plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('training_history.png') 
