import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Trainer():
    def __init__(self, model, optimizer, scheduler, load_path=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.current_epoch = 0
        self.num_epochs = 30
        if load_path is not None:
            check_point = torch.load(load_path)
            self.model.load_state_dict(check_point['model_state_dict'])
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            self.scheduler.load_state_dict(check_point['scheduler_state_dict'])
            self.current_epoch = check_point['epoch']

    def train(self, train_data_loader, val_data_loader=None, train_bbox=True):
        for epoch in range(self.current_epoch, self.num_epochs):
            epoch = epoch + 1
            total_loss = 0
            total_accuracy = 0

            self.model.train()  # set model to train mode
            pbar = tqdm(train_data_loader)
            for step, data in enumerate(pbar):
                # data
                images = data['image'].to(self.device)
                class_name = data['class_name']
                class_id = data['class_id'].to(self.device)  # [0 ~ 20]
                est_bbox = data['est_bbox'].to(self.device)  # [x, y, w, h]
                gt_bbox = data['gt_bbox'].to(self.device)    # [x, y, w, h]

                # inference
                class_pred, bbox_pred = self.model(images, bbox=train_bbox)

                # backprop
                clf_loss = F.cross_entropy(class_pred, class_id)
                loss = clf_loss

                # bbox regression loss
                if train_bbox:
                    bbox_est = bbox_pred

                    # regression targets are described in Appendix C.
                    p_x, p_y, p_w, p_h = est_bbox[:, 0], est_bbox[:, 1], est_bbox[:, 2], est_bbox[:, 3]
                    g_x, g_y, g_w, g_h = gt_bbox[:, 0], gt_bbox[:, 1], gt_bbox[:, 2], gt_bbox[:, 3]

                    t_x = (g_x - p_x) / p_w
                    t_y = (g_y - p_y) / p_h
                    t_w = torch.log(g_w / p_w)
                    t_h = torch.log(g_h / p_h)

                    bbox_ans = torch.stack([t_x, t_y, t_w, t_h], dim=1)
                    bbox_ans = bbox_ans.float().to(self.device)

                    # count only images that are not background
                    not_bg = (class_id > 0).unsqueeze(1).to(self.device)  # mask about whether each image is a background
                    bbox_est = bbox_est * not_bg
                    bbox_ans = bbox_ans * not_bg

                    # add to loss
                    bbox_loss = F.mse_loss(bbox_est, bbox_ans)
                    loss += bbox_loss

                # loss
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)

                # accuracy
                class_pred = class_pred.cpu().detach()
                class_pred = class_pred.argmax(dim=1)

                accuracy = accuracy_score(class_id.cpu().numpy(), class_pred.numpy())
                total_accuracy += accuracy
                avg_accuracy = total_accuracy / (step + 1)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logging ------------------------------------------
                pbar.set_description(f"[*] Training epoch {epoch} / {self.num_epochs} | Loss: %.3f  Accuracy: %.3f" % (avg_loss, avg_accuracy))

            # update lr
            self.scheduler.step()

            # # save checkpoints and log
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
            }, 'RCNN_checkpoint.pt')

            if val_data_loader is not None:
                self.validator.validate()