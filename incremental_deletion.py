
import torch
import numpy as np
from tqdm import tqdm
from torch import nn

from utils.utils import GaussianSmoothing


def incremental_deletion_protocol(model, explainer, val_loader, device, args):

    # image, baseline, attribution all have the shape (1,C,H,W) with C=3 for image and baseline and C=1 for attribution. N is the number of clusters.
    # alphas: list of values between 0 and 1 that indicate at what location we are doing the perturbations
    # order: 'ascending' or 'descending'
    # attribution_style: 'raw', 'pos_only', 'neg_only', 'abs'
    def compute_permutation_mask(image, attribution, alpha, clustering, order, permutation_mask_old=None):
        permutation_mask = torch.zeros_like(attribution)

        attribution_styled = attribution
        
        # if permutation_mask_old is given, the attribution for the incremental deletion is recomputed in every step
        # then we have to ensure that the interventions from the previous steps are not changed so we fix them with permutation_mask_old
        # set values to inf -inf to ensure they are included
        if permutation_mask_old != None:
            if order == 'ascending':
                attribution_styled[permutation_mask_old[:,:1,:,:]] = -float("inf") # ascending goes from small to large so set mask values to -inf
            elif order == 'descending':
                attribution_styled[permutation_mask_old[:,:1,:,:]] = float("inf") # descending goes from large to small so set mask values to inf

        B,C,H,W = image.shape
        attribution_for_cluster_sorted = attribution_styled.flatten(1) # not sorted yet, but later

        permutation_mask_flat = permutation_mask.flatten(1) # B, H*W

        if order == 'ascending':
            vals_sorted, indices_sorted = torch.sort(attribution_for_cluster_sorted, descending=False, dim=1)

        elif order == 'descending':
            vals_sorted, indices_sorted = torch.sort(attribution_for_cluster_sorted, descending=True, dim=1)
            
        nr_indices_for_alpha = indices_sorted.shape[-1]
        nr_indices_for_alpha = int(nr_indices_for_alpha * alpha)
        indices_selected = indices_sorted[:, :nr_indices_for_alpha] # B, N
        B,N = indices_selected.shape
        #add batch indices for correct indexing; shape will be B*N, 2 and look like [[0,13][0,552]...[0,3],[1,41],...,[1,5354]]
        batch_indices = torch.arange(B).to(device)
        batch_indices = batch_indices.repeat_interleave(N)
        indices_selected = indices_selected.flatten()

        permutation_mask_flat[batch_indices, indices_selected] = 1.
        permutation_mask = permutation_mask_flat.view(B,1,H,W)
        
        return permutation_mask.repeat(1,3,1,1).bool() # because image has 3 channels


    
    if args.id_baseline == 'zeros':
        tmp = 0
    elif args.id_baseline == 'average':
        tmp = 0
    elif args.id_baseline == 'random':
        tmp = 0
    elif args.id_baseline == 'blur': 
        tmp = 0
        blur = GaussianSmoothing(3, args.id_baseline_gaussian_kernel, args.id_baseline_gaussian_sigma, device=device)
    else:
        print('No baseline for faithfulness protocol specified')
        raise NotImplementedError
    
    def cluster_function(image):
        B,C,H,W = image.shape
        segs = torch.arange(start=0, end=(H*W))
        segs = segs.view(1,H,W)
        segs = segs.repeat(B,1,1)
        return segs

    metric_scores = []
    if args.nr_images != -1: # if I use lower amount of images this could make problems with different batch sizes as then interventions for a few samples could miss, so simply make sure that I can process exactly as many images as specified
        assert args.nr_images % args.batch_size == 0
        nr_images = args.nr_images
    else:
        nr_images = len(val_loader.dataset)
    iteration = 0
    for images, targets, _ in tqdm(val_loader, total=int(nr_images/args.batch_size) - 1):
        images = images.to(device)
        images.requires_grad = True
        targets = targets.to(device)
        images_perturbed = images.clone()

        B,C,H,W = images.shape


        clusterings = cluster_function(images)
        
        # create baseline
        if args.id_baseline == 'zeros':
            baselines = torch.zeros_like(images)
        elif args.id_baseline == 'random':
            baselines = torch.rand_like(images) * 2. - 1.
        elif args.id_baseline == 'average':
            tmp = 0
        elif args.id_baseline == 'blur': 
            baselines = blur(images)

        sample_scores = [] # scores for the different 'interpolation' steps of alpha for each batch
        attributions = explainer.explain(images, target=targets) # attribution has 1 channel only
            
        permutation_masks = torch.zeros_like(images).bool() # True where baseline shoul be active
        
        alphas_all = np.linspace(0, 1.0, num=args.id_steps, endpoint=False)

        for alpha_idx, alpha in enumerate(alphas_all):
            if args.id_update_attribution:
                attributions = explainer.explain(images_perturbed, target=targets)
                permutation_masks_old = permutation_masks.clone()
            else:
                permutation_masks_old = None

            permutation_masks = compute_permutation_mask(images, attributions, alpha, clusterings, args.id_order, permutation_mask_old=permutation_masks_old)
            images_perturbed[permutation_masks] = baselines[permutation_masks]
            
            image_perturbed_numpy = images_perturbed[0].permute(1,2,0).detach().cpu().numpy()
            image_perturbed_numpy = image_perturbed_numpy-image_perturbed_numpy.min()
            image_perturbed_numpy = image_perturbed_numpy / image_perturbed_numpy.max()
            image_perturbed_numpy = image_perturbed_numpy * 255.
            #cv2.imwrite('img_outputs/img_for_alpha' + str(alpha_idx) + '_' + str(alpha)+'.jpg', image_perturbed_numpy) 
            
            outputs_perturbed = model(images_perturbed)
            outputs_perturbed = outputs_perturbed.detach()


            softmax = nn.Softmax(dim=1)
            softmax_target_outputs_perturbed = softmax(outputs_perturbed)
            softmax_target_outputs_perturbed = torch.gather(softmax_target_outputs_perturbed, 1, targets.unsqueeze(-1)) # shape is B, 1

            
            #softmax_target_outputs_perturbed = softmax_target_outputs_perturbed.tolist()
            sample_scores.append(softmax_target_outputs_perturbed)
 
        sample_scores = torch.cat(sample_scores, dim=1) # B, args.id_steps
        # merge scores for alphas and batches for the final score for the batch (already normalized by number of batches)
        
            
            
        sample_scores = sample_scores.mean().item()
        
        metric_scores.append(sample_scores)

        if args.nr_images != -1 and (iteration+1)*args.batch_size >= args.nr_images:
            break
        iteration += 1
        
    
    return (sum(metric_scores)/len(metric_scores))