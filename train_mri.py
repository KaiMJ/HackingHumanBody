import torch
from utils import *
from model import UNet
from tqdm import tqdm
import random

def train_step(model, loader, opt, progress, e, writer, n_iter, args):
    '''
    Train for an epoch.

    Parameters:
        model:
        train_loader:
        opt:
        progress:
        e:
        train_writer:
        n_iter:
    '''
    model.train()

    # Keep track of dice loss and BCE
    running_loss = 0
    running_score = 0
    
    for i, data in enumerate(loader):
        opt.zero_grad()
        image, mask, (mean, std) = data
        image, mask = image.to(DEVICE), mask.to(DEVICE)
        output = model(image).squeeze(1)

        loss = criterion(output, mask)
        running_loss += loss.item()
        score = score_fn(output, mask).item()
        running_score += score

        loss.backward()
        opt.step()

        batch_description = f" Train-Iter {args.BATCH_SIZE*(i+1)}/{len(loader)*args.BATCH_SIZE}"
        progress.set_description(f"Epoch: {e}/{args.EPOCHS}" + batch_description)
        progress.set_postfix_str(f"AVG: {criterion_name}--{-running_loss / (i+1):.4f} {score_fn_name}--{running_score / (i+1):.4f}")

        writer.add_scalar(criterion_name, loss.item(), n_iter + i*args.BATCH_SIZE)
        writer.add_scalar(score_fn_name, score, n_iter + i*args.BATCH_SIZE)

        if PLOT and e%PLOT_EVERY==1 and i == 0:
            mri_plot(image, output, mask, std, mean, score_fn, score_fn_name, \
                    writer, n_iter, e, args, "Train")

        break

def val_step(model, loader, e, writer, val_iter, args):
    model.eval()

    running_loss = 0
    running_score = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            image, mask, (mean, std) = data
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            output = model(image).squeeze(1)

            running_loss += criterion(output, mask)
            running_score += score_fn(output, mask).item()

            if PLOT and e%PLOT_EVERY==1 and i == 0:
                mri_plot(image, output, mask, std, mean, score_fn, score_fn_name, \
                    writer, n_iter, e, args, "Val")
        
        loss = running_loss / len(loader)
        score = running_score / len(loader)

        writer.add_scalar(criterion_name, loss, val_iter)
        writer.add_scalar(score_fn_name, score, val_iter)
        print(f"Epoch: {e}/{EPOCHS} Validation AVG: {criterion_name}--{loss:.4f} {score_fn_name}--{score:.4f}")


def test_step(model, loader, writer, args):
    model.eval()

    running_loss = 0
    running_score = 0    
    with torch.no_grad():
        for i, data in enumerate(loader):
            image, mask, (mean, std) = data
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            output = model(image).squeeze(1)

            running_loss += criterion(output, mask)
            running_score += score_fn(output, mask).item()
        
        loss = running_loss / len(loader)
        score = running_score / len(loader)

        mri_plot_test(image, output, mask, mean, std, score_fn, score_fn_name, writer)
        print(f"Test AVG: {criterion_name}--{loss:.4f} {score_fn_name}--{score:.4f}")

if __name__ == "__main__":

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    save = None
    try:
        args = get_args()
        args.MRI_TENSORBOARD_DIR += "/runs_" + args.description
        # args.HPA_TENSORBOARD_DIR +=  "/runs_" + args.description
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = UNet()
        model.to(DEVICE)
        print(f"DEVICE: {DEVICE}")
        opt = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

        milestones = [args.EPOCHS * 0.8, args.EPOCHS * 0.9]
        gamma = 0.1
        schduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma, verbose=False)

        train_loader, train_writer, val_loader, val_writer, \
                test_loader, test_writer = mri_prepare(args)

        current_lr = args.LEARNING_RATE
        with tqdm(range(1, args.EPOCHS+1)) as progress:
            for e in progress:
                progress.set_description(f"Epoch: {e}/{args.EPOCHS}")
                n_iter = (e-1) * args.BATCH_SIZE * len(train_loader)
                val_iter = e * args.BATCH_SIZE * len(train_loader)

                if e in milestones:
                    current_lr *= gamma
                    print(f"Adjusted learning rate at Epoch {e} to {current_lr}")

                train_step(model, train_loader, opt, progress, e, train_writer, n_iter, args)
                val_step(model, val_loader, e, val_writer, val_iter, args)

            test_step(model, test_loader, test_writer, args)

    except KeyboardInterrupt:
        save = input("Save Model? [y]/n: ")
    finally:
        if save != 'n':
            torch.save(model.state_dict(), f"{args.MRI_MODEL_PATH}/model_{args.description}")
