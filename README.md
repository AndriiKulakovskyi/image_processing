# Image processing repository

Contains classical and ai-based ISP filters. 
- Classical ISP filters:
    - Denoising:
        - Gaussian filter
        - Bilateral filter
- AI-based ISP filters
    - Denoising:
        - Noise-to-noise CNN.
          It is based on the paper Noise2Noise: Learning Image Restoration without Clean Data (https://arxiv.org/abs/1803.04189). We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only.
