import torch
import torchvision.transforms as transforms 
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import Dataset
from utils import (intersection_over_union, non_max_suppression, mean_average_precision, load_checkpoint, save_checkpoint, cellboxes_to_boxes, get_bboxes)
from loss import YoloLoss
from utils import plot_image
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
seed = 41
torch.manual_seed(seed)


LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
BATCH_SIZE = 26
WEIGHT_DECAY = 0.0005

EPOCHS = 31
NUM_WORKERS = 5
PIN_MEMOTY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "Yolov1.path.tar"
IMG_DIR = './images'
LABEL_DIR = './labels'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image), boxes
        return image, boxes

transforms = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train(train_loader, model, optimizer, loss_fn, scheduler):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())
    mean_loss = sum(mean_loss) / len(mean_loss)
    #scheduler.step(mean_loss)
    print(f"Mean loss was {mean_loss}")
    

def main():
    model = YoloV1(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = YoloLoss()
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
    train_dataset = Dataset("./train.csv", transform=transforms, image_dir = IMG_DIR, label_dir = LABEL_DIR)
    test_dataset = Dataset("./test.csv", transform=transforms, image_dir = IMG_DIR, label_dir = LABEL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, 
                              num_workers = NUM_WORKERS, pin_memory = PIN_MEMOTY, drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, 
                              num_workers = NUM_WORKERS, pin_memory = PIN_MEMOTY, drop_last=True)
    
    for epoch in range(EPOCHS):

        #checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict()
        #     }
        #save_checkpoint(checkpoint, filename = f"overfit.path.tar")
        
        for x, y in train_loader:
           x = x.to(DEVICE)
           for idx in range(len(train_loader)):
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, probability_threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
           import sys
           sys.exit()
        
        pred_bboxes, target_bboxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_average_p = mean_average_precision(pred_bboxes, target_bboxes, iou_threshold = 0.5, box_format = 'midpoint')
        if epoch % 5 == 0 and epoch != 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename = LOAD_MODEL_FILE)
        print(f"Train mAP: {mean_average_p}, Epoch: {epoch}") 
        train(train_loader, model, optimizer, loss_fn, scheduler)
        scheduler.step()

if __name__ == "__main__":
    main()