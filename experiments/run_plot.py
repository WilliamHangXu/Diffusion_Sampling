import numpy as np
import os
from diffusion.sample import sample_batch
from utils.compute_FID import FID_score_calculator, compute_and_save_real_statistics, disassemble_integrated_image

samplers = ['ddpm']
schedulers = ['linear', 'cosine']
weight_epochs = ['50', '100', '150']
sample_time = 10

# Path to directory where we save and load stats
stats_dir = 'stats'
real_mu_path = os.path.join(stats_dir, 'real_mu.npy')
real_sigma_path = os.path.join(stats_dir, 'real_sigma.npy')

# Initialize the calculator
fid_calculator = FID_score_calculator()

# Compute real statistics if not available
if not (os.path.exists(real_mu_path) and os.path.exists(real_sigma_path)):
    print("Real statistics not found. Computing now...")
    compute_and_save_real_statistics(fid_calculator, data_root='data', num_images=50000, save_dir=stats_dir)

# Load real statistics
real_mu = np.load(real_mu_path)
real_sigma = np.load(real_sigma_path)
fid_calculator.set_real_statistics(real_mu, real_sigma)

# Start sampling
for sam in samplers:
    for sch in schedulers:
        for w in weight_epochs:
            w_name = '{}_200_{}.pt'.format(sch, w)
            print(f"Sampling {sam} {sch} {w_name}")
            sample_config = {
                'weight': w_name,
                'sampler': sam,
                'noise_scheduler': sch,
                'samp_time': sample_time
            }
            sample_batch(sample_config)
            print("Samping finished. Calculating FID score...")
            job_name = '{}_{}'.format(sam, w_name[0:-3])
            image_dir = 'results/{}'.format(job_name)

            # Directory containing integrated images
            integrated_image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

            total_list = []
            # Compute FID for each integrated image
            for img_idx, img_file in enumerate(integrated_image_files):
                img_path = os.path.join(image_dir, img_file)
                generated_images = disassemble_integrated_image(img_idx, img_path, sub_img_size=32, grid_size=10, divider_size=2, 
                edge_border=2,
                output_dir='results_sliced/{}'.format(job_name))
                total_list.extend(generated_images)


            # Compute generated statistics
            gen_mu, gen_sigma = fid_calculator.compute_statistics(total_list)

            # Compute FID
            fid = fid_calculator.compute_fid(gen_mu, gen_sigma)
            print(f"FID for {job_name}: {fid}")
            with open(f"fid/{job_name}.txt", "w") as text_file:
                text_file.write(f"FID for {job_name}: {fid}")


