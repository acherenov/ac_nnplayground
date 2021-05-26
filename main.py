from torch.utils.data import DataLoader
from datetime import datetime
import os
from dataset import Data_generator
from net import SimpleMLP
from trainer import Trainer, get_data_from_datasets
from visualize_utils import make_meshgrid, plot_predictions, predict_proba_on_mesh_tensor
import glob
from PIL import Image

if __name__ == "__main__":
    runname = datetime.now().strftime("RUN_%y%m%d_%H%M%S")
    
    path = os.getcwd()
    path = path + "/runs"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    path = os.getcwd()
    path = path + "/runs/" + runname
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    path = path + "/pictures"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
            
    path = os.getcwd()
    runname = path + "/runs/" + runname
    
    layers = [(2, 4, 'relu'), (4, 6, 'sigmoid'), (6, 2, 'tanh')]
    model = SimpleMLP(layers)

    trainer = Trainer(model, lr=0.03, runname = runname)
    print(trainer.device)

    train_dataset = Data_generator(n_samples=5000, shuffle=True, random_state=0, data_type="blobs")
    test_dataset = Data_generator(n_samples=1000, shuffle=True, random_state=0, data_type="blobs")

    print(train_dataset)


    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    trainer.fit(train_dataloader, n_epochs=100)


    test_predicion_proba = trainer.predict_proba(test_dataloader)
    
    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset, test_dataset)

    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)

    Z = predict_proba_on_mesh_tensor(trainer, xx, yy)

    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    model.save_model(runname + "/saved_model")
    
    fp_in = runname + "/pictures/nn_predictions_*.png"
    fp_out = runname + "/training_process.gif"
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    

