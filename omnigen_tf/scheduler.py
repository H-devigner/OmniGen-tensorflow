import tensorflow as tf

class OmniGenScheduler:
    def __init__(self, num_steps: int = 50, time_shifting_factor: int = 1):
        """Initialize the scheduler.
        
        Args:
            num_steps: Number of diffusion steps
            time_shifting_factor: Factor for time shifting in the scheduler
        """
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor
        
        # Create timesteps
        t = tf.linspace(0.0, 1.0, num_steps + 1)
        self.sigma = t / (t + time_shifting_factor - time_shifting_factor * t)
        
    def get_timesteps(self, num_inference_steps: int) -> tf.Tensor:
        """Get the sequence of timesteps for the inference process.
        
        Args:
            num_inference_steps: Number of inference steps to take
            
        Returns:
            Tensor of timesteps
        """
        if num_inference_steps > self.num_steps:
            raise ValueError(f"Number of inference steps ({num_inference_steps}) cannot be greater than "
                           f"number of scheduler steps ({self.num_steps})")
        
        # Create evenly spaced timesteps
        step_ratio = self.num_steps // num_inference_steps
        timesteps = tf.range(0, num_inference_steps) * step_ratio
        return timesteps
        
    def step(self, latents: tf.Tensor, noise_pred: tf.Tensor, timestep: tf.Tensor) -> tf.Tensor:
        """Perform one denoising step.
        
        Args:
            latents: Current latent states
            noise_pred: Predicted noise
            timestep: Current timestep
            
        Returns:
            Updated latents
        """
        # Get sigma values for current and next timestep
        sigma = tf.gather(self.sigma, timestep)
        sigma_next = tf.gather(self.sigma, timestep + 1)
        
        # Update latents using the predicted noise
        latents = latents + (sigma_next - sigma) * noise_pred
        
        return latents
