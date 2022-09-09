import torch
from models.model import UNet
from dataset import prepare
from config import *
from utils import *
from tqdm import tqdm
import numpy as np
import time


def run_train_batch(train_loader, model, opt, progress, e, writer, n_iter, inv_transform):
    '''
        train_loader: data loader
        model: net
        opt: optimizer
        progress: tqdm progress bar
        e: current epoch
        writer: tensorboard writer
        n_iter: global index iteration
    '''
    running_loss = 0
    running_score = 0

    model.train()

    for i, data in enumerate(train_loader):
        # data = {image, mask, original}
        image, mask = data['image'].to(DEVICE), data['mask'].to(DEVICE)
        out = model(image).squeeze(1)
        loss = dice_loss(mask, out)
        running_loss += loss.item()
        score = binary_crossentropy(mask.to(torch.float32), out).item()
        running_score += score

        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_description = f" Train-Iter {BATCH_SIZE*(i+1)}/{len(train_loader)*BATCH_SIZE}"
        progress.set_description(f"Epoch: {e}/{EPOCHS}" + batch_description)
        progress.set_postfix_str(f"AVG: DICE--{-running_loss / (i+1):.4f} BCE--{running_score / (i+1):.4f}")

        writer.add_scalar('DICE', -loss.item(), n_iter + i*BATCH_SIZE)
        writer.add_scalar('BCE', score, n_iter + i*BATCH_SIZE)

        if PLOT and e%PLOT_EVERY==0 and i == 0:
            losses = -dice_loss(mask, out, dim=(1, 2))
            plt_imgs, plt_outs, plt_masks = format_tensors_plt(image, out, inv_transform, mask=mask)
            plot_images(writer, plt_imgs, plt_masks, None, n_iter, e, name="Train Original")
            plot_images(writer, plt_imgs, plt_outs, losses, n_iter, e, name="Train Predicted")

def run_val_batch(val_loader, model, e, writer, n_iter, inv_transform):
    '''
        train_loader: data loader
        model: net
        opt: optimizer
        e: current epoch
        writer: tensorboard writer
        n_iter: global index iteration
    '''
    running_loss = 0
    running_score = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask = data['image'].to(DEVICE), data['mask'].to(DEVICE)
            out = model(image).squeeze(1)
            running_loss += dice_loss(mask, out).item()
            running_score += binary_crossentropy(mask.to(torch.float32), out).item()

            if PLOT and e%PLOT_EVERY==0 and i == 0:
                losses = -dice_loss(mask, out, dim=(1, 2))
                plt_imgs, plt_outs, plt_masks = format_tensors_plt(image, out, inv_transform, mask=mask)
                plot_images(writer, plt_imgs, plt_masks, None, n_iter, e, name="Val Original")
                plot_images(writer, plt_imgs, plt_outs, losses, n_iter, e, name="Val Predicted")

        running_score /= len(val_loader)
        dice = -running_loss / len(val_loader)

        writer.add_scalar('DICE', dice, n_iter)
        writer.add_scalar('BCE', running_score, n_iter)
        print(f"Epoch: {e}/{EPOCHS} Validation AVG: SCORE--{dice:.4f} BCE--{running_score:.4f}")

def eval_network(test_loader, model, inv_normalize, e, test_writer, n_iter):
    '''
        Plot Test Data.
    '''
    model.eval()
    with torch.no_grad():
        # Dev-Test
        for i, data in enumerate(test_loader):
            image = data["image"].to(DEVICE)
            out = model(image).squeeze(1)
            plt_imgs, plt_outs = format_tensors_plt(image, out, inv_normalize)
            plot_images(test_writer, plt_imgs, plt_outs, None, n_iter, e, name="Eval Predicted")
            break

def test_submission(submission_writer, model, test_transform, inv_normalize, e, n_iter):
    test_path = os.path.join("data/test_images/"+ '10078.tiff')
    submission_img = plt.imread(test_path)
    model.eval()

    image = test_transform(image=submission_img)['image'].to(DEVICE).unsqueeze(0)
    out = model(image).squeeze(1)
    plt_img = inv_normalize(image=image[0].permute(1, 2, 0).cpu().numpy())['image']
    plt_out = threshold_tensor(out).detach().cpu().numpy().astype(np.uint8)[0]
    plt_mask = get_mask_highlight(plt_img, plt_out)

    plt_img[plt_out == 1] = 1

    submission_writer.add_image('Submisssion Highlight', plt_mask, n_iter, dataformats="HW3")
    submission_writer.add_image('Submisssion Mask', plt_img, n_iter, dataformats="HW3")

def train_network(model, opt):
    # Get stats and transforms
    # full_dataset = HPADataset(CSV_FILE, ROOT_DIR, transform=None)
    # means, stds = find_stats(full_dataset, 256)
    means = np.array([212.1089, 205.7680, 210.4203]) / 255
    stds = np.array([41.9276, 48.7806, 45.6515]) / 255
    transform, test_transform, inv_normalize = get_transforms(means, stds, IMG_SIZE)

    train_loader, train_writer, \
        val_loader, val_writer, \
        test_loader, test_writer, \
        submission_writer       =   prepare(transform, test_transform, TENSORBOARD_PATH)

    model.to(DEVICE)

    progress = tqdm(range(1, EPOCHS+1))
    for e in progress:
        progress.set_description(f"Epoch: {e}/{EPOCHS}")
        n_iter = (e-1) * BATCH_SIZE * len(train_loader)
        val_iter = e * BATCH_SIZE * len(train_loader)

        run_train_batch(train_loader, model, opt, progress, e, train_writer, n_iter, inv_normalize)
        run_val_batch(val_loader, model, e, val_writer, val_iter, inv_normalize)
        if PLOT and e%PLOT_EVERY == 1:
            eval_network(test_loader, model, inv_normalize, e, test_writer, val_iter)
            test_submission(submission_writer, model, test_transform, inv_normalize, e, val_iter)
    return

if __name__ == "__main__":
    # Log experiments
    args = get_args()
    TENSORBOARD_PATH = TENSORBOARD_PATH_DIR + "/runs_" + args.number
    start = time.time()
    save = None
    try:
        print(f"Run Model Number: {args.number}")
        model = UNet(3, 1, init_features=32)
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

        if OPTIMIZER == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # TODO: set up patience
        # TODO: Learning rate scheduler
        train_network(model, opt)

    except KeyboardInterrupt: 
        save = input('Save Model and Log? [y]/n: ')
    finally:
        if save != 'n':
            end = time.time()
            torch.save(model.state_dict(), f'{MODEL_PATH}/model_{args.number}')
            print('Saved model.')
            add_log = {
                # Total Training Epoch
                'TotalTime': time.strftime('%H:%M:%S', time.gmtime(end - start))
                # Best Dice Score
                # average time, etc...
            }
            log_experiment(args, add_log)
        else:
            print('Not saved model.')
