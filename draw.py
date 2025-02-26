import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import norm
from tqdm import tqdm

def get_color_from_score(score):
    """
    Returns a color based on score from a smooth gradient.
    Low scores (0.0) = Red (more OOD)
    Mid scores (0.5) = Orange
    High scores (1.0) = Green (more ID)
    """
    if score < 0.5:
        # Interpolate between red and orange
        r = 255
        g = int(128 * (score * 2))  # 0 to 128
        b = 0
    else:
        # Interpolate between orange and green
        r = int(255 * (1 - (score - 0.5) * 2))  # 255 to 0
        g = int(128 + 127 * (score - 0.5) * 2)  # 128 to 255
        b = 0
    
    return (r, g, b)

def create_annotated_ood_images(cfg, args, model, valid_loader_list, output_dir):
    """
    Create annotated images with OOD scores displayed as a panel above each image.
    Uses explicit mapping between images and scores for reliability.
    
    Args:
        cfg: Configuration object
        args: Command-line arguments
        model: The trained model
        valid_loader_list: List of data loaders (ID and OOD)
        output_dir: Directory to save the annotated images
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get parameters for normalization from ID dataset
    print("Computing normalization parameters from ID dataset...")
    prob_ID_list = []
    rec_0_ID_list = []
    rec_1_ID_list = []
    
    with torch.no_grad():
        for imgs, _ in tqdm(valid_loader_list[0]):
            imgs = imgs.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
            valid_logits, x_0, x_1, rec_0, rec_1 = model(imgs)
            
            probs = F.softmax(valid_logits/1000, dim=-1)
            
            norm_0 = torch.norm(x_0, p=2, dim=-1, keepdim=True)
            rec_0_vals = torch.sum(((x_0/norm_0) - (rec_0/norm_0))**2, dim=-1)
            
            norm_1 = torch.norm(x_1, p=2, dim=-1, keepdim=True)
            rec_1_vals = torch.sum(((x_1/norm_1) - (rec_1/norm_1))**2, dim=-1)
            
            prob_ID_list.append(np.max(probs.cpu().numpy(), -1))
            rec_0_ID_list.append(rec_0_vals.cpu().numpy())
            rec_1_ID_list.append(rec_1_vals.cpu().numpy())
    
    prob_ID = np.concatenate(prob_ID_list)
    rec_0_ID = np.concatenate(rec_0_ID_list)
    rec_1_ID = np.concatenate(rec_1_ID_list)
    
    # Fit normalization parameters
    loc_p, scale_p = norm.fit(prob_ID)
    loc_0, scale_0 = norm.fit(rec_0_ID)
    loc_1, scale_1 = norm.fit(rec_1_ID)
    
    # Process OOD dataset with explicit mapping
    print("Computing OOD scores...")
    ood_dataset = valid_loader_list[1].dataset
    
    # Keep track of max score for normalization
    max_score = float('-inf')
    image_scores = []  # Will hold (img_path, score) tuples
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(tqdm(valid_loader_list[1])):
            imgs = imgs.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
            valid_logits, x_0, x_1, rec_0, rec_1 = model(imgs)
            
            probs = F.softmax(valid_logits/1000, dim=-1)
            
            norm_0 = torch.norm(x_0, p=2, dim=-1, keepdim=True)
            rec_0_vals = torch.sum(((x_0/norm_0) - (rec_0/norm_0))**2, dim=-1)
            
            norm_1 = torch.norm(x_1, p=2, dim=-1, keepdim=True)
            rec_1_vals = torch.sum(((x_1/norm_1) - (rec_1/norm_1))**2, dim=-1)
            
            # Calculate OOD scores for this batch
            prob_OOD_norm = norm.cdf(np.max(probs.cpu().numpy(), -1), loc=loc_p, scale=scale_p*10)
            rec_0_OOD_norm = 1 - norm.cdf(rec_0_vals.cpu().numpy(), loc=loc_0, scale=scale_0*10)
            rec_1_OOD_norm = 1 - norm.cdf(rec_1_vals.cpu().numpy(), loc=loc_1, scale=scale_1*10)
            
            scores_batch = prob_OOD_norm * rec_0_OOD_norm * rec_1_OOD_norm
            max_score = max(max_score, np.max(scores_batch))
            
            # Map each score to its corresponding image path
            # Calculate the indices in the dataset for this batch
            start_idx = batch_idx * valid_loader_list[1].batch_size
            end_idx = min(start_idx + valid_loader_list[1].batch_size, len(ood_dataset))
            
            for batch_pos, idx in enumerate(range(start_idx, end_idx)):
                if idx < len(ood_dataset.img_id):
                    img_name = ood_dataset.img_id[idx]
                    img_path = os.path.join(args.test_datapath, img_name)
                    score = scores_batch[batch_pos]
                    image_scores.append((img_path, score, img_name))
    
    # Now process each image with its correctly mapped score
    print(f"Creating annotated images with OOD scores...")
    for img_path, score, img_name in tqdm(image_scores):
        # Load the original image for annotation
        try:
            orig_img = Image.open(img_path)
            
            # Normalize the score
            score_norm = score / max_score
            
            # Standard target size for all images
            target_size = 256
            
            # Resize the image to our standard viewing size
            # Use LANCZOS resampling for better quality upscaling
            orig_size = orig_img.size
            scale_factor = target_size / max(orig_size)
            new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
            resized_img = orig_img.resize(new_size, Image.LANCZOS)
            
            # Get dimensions of resized image
            img_width, img_height = resized_img.size
            
            # Panel height - fixed at 30px which works well with 256px images
            panel_height = 30
            
            # Create a new image with a panel for the score
            new_img = Image.new('RGB', (img_width, img_height + panel_height), color=(240, 240, 240))
            new_img.paste(resized_img, (0, panel_height))
            
            # Draw the score on the panel
            draw = ImageDraw.Draw(new_img)
            
            # Font size that works well with our standard image size
            font_size = 14
            
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Get color based on score using our gradient function
            color = get_color_from_score(score_norm)
            
            # Position text vertically centered in the panel
            text = f"OOD Score: {score_norm:.4f}"
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
            position = (10, (panel_height - text_height) // 2)
            
            # Draw the text
            draw.text(position, text, fill=color, font=font)
            
            # Add a color bar to visualize the score - proportional to image width
            bar_width = int(img_width * 0.25)  # 25% of image width
            bar_height = int(panel_height * 0.4)  # 40% of panel height
            bar_x = img_width - bar_width - 10  # Right-aligned with 10px margin
            bar_y = (panel_height - bar_height) // 2  # Centered vertically
            
            # Draw background (gray) for empty part of bar
            draw.rectangle(
                [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
                fill=(220, 220, 220), outline=(180, 180, 180)
            )
            
            # Draw filled part of bar with gradient color
            filled_width = int(bar_width * score_norm)
            if filled_width > 0:
                draw.rectangle(
                    [(bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height)],
                    fill=color
                )
            
            # Save the annotated image
            output_path = os.path.join(output_dir, f"annotated_{img_name}")
            new_img.save(output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Annotated images saved to {output_dir}")
