import datetime, os, sys
from tensorflow.keras.callbacks import TensorBoard
from model.cnn import model
from preprocessing.loaders.drawings import load_drawings_dataset
    
def train_model():
	epochs = 14

	logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

	(training_dataset, validation_dataset) = load_drawings_dataset()

	model.fit(
			training_dataset,
			validation_data=validation_dataset,
			epochs=epochs,
			verbose=1,
			callbacks=[tensorboard_callback]
	)

	model.save('./model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    globals()[sys.argv[1]]()