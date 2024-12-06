import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from utils.utils import GaussianSmoothing


def single_deletion_protocol(model, explainer, val_loader, args, device):
   
    intervention_scores = np.empty((len(val_loader.dataset),args.grid_rows_and_cols+1,args.grid_rows_and_cols+1)) # +1 because -1, -1 is by default the original output score

    rank_correlations = []
    iteration = 0
    if args.nr_images != -1: # if I use lower amount of images this could make problems with different batch sizes as then interventions for a few samples could miss, so simply make sure that I can process exactly as many images as specified
        assert args.nr_images % args.batch_size == 0
        nr_images = args.nr_images
    else:
        nr_images = len(val_loader.dataset)

    blur = GaussianSmoothing(3, 51, 41, device=device)

    for images, targets, indices in tqdm(val_loader, total=int(nr_images/args.batch_size) - 1):
        images = images.to(device, non_blocking=True)
        images.requires_grad = True
        B,C,H,W = images.shape
        targets = targets.to(device, non_blocking=True)

        image_identifiers = indices

        # get attribution
        attributions = explainer.explain(images, target=targets)

        attributions_per_cell_list = []
        output_changes_per_cell_list = []
        cell_H = int(H/args.grid_rows_and_cols)
        cell_W = int(W/args.grid_rows_and_cols)
        assert cell_H == cell_W

        
        # get attributions per cell
        for cell_row in range(args.grid_rows_and_cols):
            for cell_col in range(args.grid_rows_and_cols):
                attributions_per_cell = attributions[:,:,cell_row*cell_H:(cell_row+1)*cell_H, cell_col*cell_W:(cell_col+1)*cell_W].sum(dim=(1,2,3)).detach().cpu().numpy() # B
                attributions_per_cell_list.append(attributions_per_cell)

                
        # get output changes per cell
        
        outputs = model(images)
        outputs_original = torch.gather(outputs, 1, targets.unsqueeze(-1))
        for b in range(outputs_original.shape[0]):
            intervention_scores[image_identifiers[b], -1, -1] = outputs_original[b].item()


        for cell_row in range(args.grid_rows_and_cols):
            for cell_col in range(args.grid_rows_and_cols):
                
                if args.sd_baseline == 'zeros':
                    baselines = torch.zeros_like(images)
                elif args.sd_baseline == 'random':
                    baselines = torch.rand_like(images) * 2. - 1.
                elif args.sd_baseline == 'blur':
                    baselines = blur(images.clone())
                else:
                    print('baseline not implemented')
                images_intervened = images.clone()
                images_intervened[:,:,cell_row*cell_H:(cell_row+1)*cell_H, cell_col*cell_W:(cell_col+1)*cell_W] = baselines[:,:,cell_row*cell_H:(cell_row+1)*cell_H, cell_col*cell_W:(cell_col+1)*cell_W]

                outputs = model(images_intervened)
                outputs_without_cell = torch.gather(outputs, 1, targets.unsqueeze(-1))
                for b in range(outputs_without_cell.shape[0]):
                    intervention_scores[image_identifiers[b], cell_row, cell_col] = outputs_without_cell[b].item()

                output_changes_for_cell = outputs_original - outputs_without_cell # B
                output_changes_for_cell = output_changes_for_cell.detach().cpu().numpy()
                output_changes_per_cell_list.append(output_changes_for_cell)
                
        # compute rank correlation
        for b in range(len(output_changes_per_cell_list[0])):
            attribution_per_cell_list = list(map(lambda x: x[b], attributions_per_cell_list))
            output_change_per_cell_list = list(map(lambda x: x[b], output_changes_per_cell_list))
            correlation, p_value = stats.spearmanr(attribution_per_cell_list, output_change_per_cell_list)

            rank_correlations.append(correlation)

        if args.nr_images != -1 and (iteration+1)*args.batch_size >= args.nr_images:
            break
        iteration += 1

    rank_correlations =  [x for x in rank_correlations if type(x) == np.float64]

    # accumulate results
    mean_rank_correlation = sum(rank_correlations) / len(rank_correlations)
    return mean_rank_correlation