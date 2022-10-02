import torch
from model import UNet
from config import *
from utils import *
from tqdm import tqdm
from PrepareData import create_modified_dataset
import time



def run_train_batch(train_loader, model, opt, progress, e, writer, n_iter, inv_transform, args):
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
    running_f1 = 0

    model.train()
    torch.autograd.set_detect_anomaly(True)
    for i, data in enumerate(train_loader):
        opt.zero_grad()
        image, mask = data
        image, mask = image.to(DEVICE), mask.to(DEVICE)
        out = model(image).squeeze(1)

        loss = criterion(out, mask)
        running_loss += loss.item()
        score = score_fn(out, mask).item()
        running_score += score
        running_f1 += f1_score(out, mask).item()

        loss.backward()
        opt.step()

        batch_description = f" Train-Iter {args.BATCH_SIZE*(i+1)}/{len(train_loader)*args.BATCH_SIZE}"
        progress.set_description(f"Epoch: {e}/{EPOCHS}" + batch_description)
        progress.set_postfix_str(f"AVG: {criterion_name}--{-running_loss / (i+1):.4f} {score_fn_name}--{running_score / (i+1):.4f} F1--{running_f1 / (i+1):.4f}")

        writer.add_scalar(criterion_name, loss.item(), n_iter + i*args.BATCH_SIZE)
        writer.add_scalar(score_fn_name, score, n_iter + i*args.BATCH_SIZE)

        if PLOT and e%PLOT_EVERY==0 and i == 0:
            losses = [score_fn(out[i], mask[i]) for i in range(args.BATCH_SIZE)]
            plt_imgs, plt_outs, plt_masks = format_tensors_plt(image, out, inv_transform, mask=mask)
            plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, n_iter, e, name="Train Original")
            plot_images(writer, plt_imgs, plt_outs, losses, score_fn_name, n_iter, e, name="Train Predicted")

def run_val_batch(val_loader, model, e, writer, n_iter, inv_transform, args):
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
    running_f1 = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask = data
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            out = model(image).squeeze(1)
            running_loss += criterion(out, mask).item()
            running_score += score_fn(out, mask).item()
            running_f1 += f1_score(out, mask).item()

            if PLOT and e%PLOT_EVERY==0 and i == 0:
                losses = [score_fn(out[i], mask[i]) for i in range(args.BATCH_SIZE)]
                plt_imgs, plt_outs, plt_masks = format_tensors_plt(image, out, inv_transform, mask=mask)
                plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, n_iter, e, name="Val Original")
                plot_images(writer, plt_imgs, plt_outs, losses, score_fn_name, n_iter, e, name="Val Predicted")

        loss = running_loss / len(val_loader)
        score = running_score / len(val_loader)

        writer.add_scalar(criterion_name, loss, n_iter)
        writer.add_scalar(score_fn_name, score, n_iter)
        print(f"Epoch: {e}/{EPOCHS} Validation AVG: {criterion_name}--{loss:.4f} {score_fn_name}--{score:.4f} F1--{running_f1 / (i+1):.4f}")

def eval_network(model, loader, inv_normalize, writer, args):
    '''
        Plot Test Data.
    '''
    model.eval()

    with torch.no_grad():
        if args.image_tiling:
            for i, data in enumerate(loader):
                images, masks = data
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                # Predict each tile
                outputs = []
                for i in range(4):
                    out = model(images[:, i, :, :, :]).squeeze(1)
                    outputs.append(out)
                outputs = torch.stack(outputs).permute(1, 0, 2, 3).detach()
                
                plt_imgs = combine_tensors(images.cpu(), (args.IMG_SIZE, args.IMG_SIZE))
                plt_imgs = inv_normalize(image=plt_imgs)['image']
                plt_masks = combine_tensors(masks.cpu(), (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
                plt_outs = combine_tensors(threshold_tensor(outputs).cpu(), (args.IMG_SIZE, args.IMG_SIZE)).astype(np.uint8)

                losses = [score_fn(plt_outs[i], plt_masks[i]).item() for i in range(len(plt_masks))]
                plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, 1, "Final", name="Test Original")  
                plot_images(writer, plt_imgs, plt_outs, None, score_fn_name, 1, "Final", name="Test Predicted")
                print(f"Test F1--{f1_score(outputs, masks).item() / {i+1}:.4f}")
        else:
            for i, data in enumerate(loader):
                print(data)
                image, mask = data
                image, mask = image.to(DEVICE), mask.to(DEVICE)
                out = model(image).squeeze(1)
                
                losses = [score_fn(out[i], mask[i]) for i in range(args.BATCH_SIZE)]
                plt_imgs, plt_outs, plt_masks = format_tensors_plt(image, out, inv_normalize, mask=mask)
                plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, 1, "Final", name="Test Original")  
                plot_images(writer, plt_imgs, plt_outs, losses, score_fn_name, 1, "Final", name="Test Predicted")
                break

def train_network(model, opt, scheduler):

    train_loader, train_writer, \
        val_loader, val_writer, \
        test_loader, test_writer, \
        inv_normalize               =   hpa_prepare(args)

    model.to(DEVICE)

    with tqdm(range(1, EPOCHS+1)) as progress:
        for e in progress:
            progress.set_description(f"Epoch: {e}/{EPOCHS}")
            n_iter = (e-1) * BATCH_SIZE * len(train_loader)
            val_iter = e * BATCH_SIZE * len(train_loader)

            run_train_batch(train_loader, model, opt, progress, e, train_writer, n_iter, inv_normalize, args)
            run_val_batch(val_loader, model, e, val_writer, val_iter, inv_normalize, args)

            if scheduler is not None:
                scheduler.step()

        eval_network(model, test_loader, inv_normalize, test_writer, args)

if __name__ == "__main__":
    # create_modified_dataset(DATA_CSV, IMAGES_DIR, MOD_IMAGES_DIR, MOD_MASKS_DIR, TEST_IMAGES_DIR, TEST_MASKS_DIR, IMG_SIZE)
    args = get_args()
    args.HPA_TENSORBOARD_DIR = HPA_TENSORBOARD_DIR + "/runs_" + args.description
    start = time.time()
    save = None
    # try:
    print(f"Run Model: {args.description}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pretrained works better
    model = UNet(3, 1, 32)
    model.load_state_dict(torch.load(args.MRI_MODEL_PATH + "/model_0"))

    if args.OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.HPA_LEARNING_RATE)
    elif args.OPTIMIZER == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.HPA_LEARNING_RATE, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones = [100, 200, 300, 400],
        gamma = 0.5,
        verbose = False
    )
    
    train_network(model, opt, scheduler)

    # except KeyboardInterrupt: 
    #     save = input('Save Model and Log? [y]/n: ')
    # finally:
    #     if save != 'n':
    #         end = time.time()
    #         torch.save(model.state_dict(), f'{args.HPA_MODEL_PATH}/model_{args.description}')
    #     else:
    #         print('Not saved model.')
