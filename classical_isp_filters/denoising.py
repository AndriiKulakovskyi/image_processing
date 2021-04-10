import numpy as np
import matplotlib.pyplot as plt


def gaussian_dn(im, sigma):
    n = np.int(np.sqrt(sigma) * 3)
    height, width = im.shape
    img_filtered = np.zeros([height, width])
    
    print('Gaussian support:', n)
    
    for p_y in range(0, height):
        for p_x in range(0, width):
            gp = 0
            w = 0
  
            # Iterate over kernel locations to define pixel q
            for i in range(-n, n):
                for j in range(-n, n):
                    
                    # Make sure no index goes out of bounds of the image
                    q_y = np.max([0, np.min([height - 1, p_y + i])])
                    q_x = np.max([0, np.min([width - 1, p_x + j])])
                    
                    # Computer Gaussian filter weight at this filter pixel
                    g = np.exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )

                    # Accumulate filtered output
                    gp += g * im[q_y, q_x]
                    # Accumulate filter weight for later normalization, to maintain image brightness
                    w += g

            img_filtered[p_y, p_x] = gp / (w + np.finfo(np.float32).eps)

    return img_filtered
                 
def bilateral_dn(im, sigma_s, sigma_i):
    n = np.int(np.sqrt(sigma_s) * 3)
    
    height, width = im.shape
    img_filtered = np.zeros([height, width])
    
    # Iterate over pixel locations p
    print('Gaussian support:', n)
    for p_y in range(0, height):
        for p_x in range(0, width):
            gp = 0
            w = 0
  
            # Iterate over kernel locations to define pixel q
            for i in range(-n, n):
                for j in range(-n, n):
                    
                    # Make sure no index goes out of bounds of the image
                    q_y = np.max([0, np.min([height - 1, p_y + i])])
                    q_x = np.max([0, np.min([width - 1, p_x + j])])
                    
                    # Computer Gaussian filter weight at this filter pixel
                    g_s = np.exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma_s**2) )
                    g_i = np.exp( -((im[q_y, q_x] - im[p_y, p_x])**2) / (2 * sigma_i**2) )
                    # Accumulate filtered output
                    g = g_s * g_i
                    
                    gp += g * im[q_y, q_x]
                    # Accumulate filter weight for later normalization, to maintain image brightness
                    w += g

            img_filtered[p_y, p_x] = gp / (w + np.finfo(np.float32).eps)

    return img_filtered   
                    

if __name__ == '__main__':
    # Generate image
    image = np.empty([21,21])
    image[:10,:] = 0.2
    image[10:,:] = 0.8
    
    # Add noise 
    noise = np.random.uniform(0.0, 0.1, [21,21])
    image_noisy = image + noise
    image_noisy = image_noisy.clip(0, 1)
    
    # Apply Gaussian denoiser
    imgGaussian = gaussian_dn(image_noisy, sigma=2)
    
    # Apply Bilateral denoiser
    imgBilateral = bilateral_dn(image_noisy, sigma_s=2, sigma_i=0.1)
    
    # Visualize
    x = np.linspace(0, 21, 21)
    y = np.linspace(0, 21, 21)
    xv, yv = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dem3d = ax.plot_surface(xv, yv, image_noisy, cmap='viridis')
    ax.set_title('Noisy image')
    plt.show()
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dem3d = ax.plot_surface(xv, yv, imgGaussian, cmap='viridis')
    ax.set_title('Gaussian DN')
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dem3d = ax.plot_surface(xv,yv, imgBilateral, cmap='viridis')
    ax.set_title('Bilateral DN')
    plt.show()
