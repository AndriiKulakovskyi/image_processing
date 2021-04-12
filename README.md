# Image processing repository

Contains classical and ai-based ISP filters. 
- Classical ISP filters:
    - Denoising:
        - Gaussian filter
        - Bilateral filter
- AI-based ISP filters
    - Denoising:
        - ***Noise-to-noise CNN.*** In contrast to reference-based training, where a denoising CNN maps noisy inputs to clean ground truth images, the Noise2Noise (N2N) method attempts to learn a mapping between pairs of independently degraded versions of the same training image, i.e. (s + n; s + n'), that incorporate the same signal s, but independently drawn noise n and n'. Naturally, the denoising CNN cannot learn to perfectly predict one noisy image from another one. However, networks trained on this impossible training task can produce results that converge to the same statistical predictions as traditionally trained networks that do have access to ground truth images. This method can be especially intersting when the ground truth data is physically unobtainable,i.e. astronomy or microscopy, but several noisy observations of the same signal are accesible.
          
