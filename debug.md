################################################################################
                          Learning iteration 1998/2000                           

                            Total steps: 65503232 
                       Steps per second: 39945 
                        Collection time: 0.767s 
                          Learning time: 0.053s 
                        Mean value loss: 0.0103
                    Mean surrogate loss: -0.0128
                      Mean entropy loss: -10.5983
                            Mean reward: 8.44
                    Mean episode length: 500.00
                        Mean action std: 0.10
--------------------------------------------------------------------------------
                         Iteration time: 0.82s
                           Time elapsed: 00:27:13
                                    ETA: 00:00:00

################################################################################
                          Learning iteration 1999/2000                           

                            Total steps: 65536000 
                       Steps per second: 39843 
                        Collection time: 0.770s 
                          Learning time: 0.053s 
                        Mean value loss: 0.0101
                    Mean surrogate loss: -0.0122
                      Mean entropy loss: -10.6037
                            Mean reward: 7.15
                    Mean episode length: 500.00
                        Mean action std: 0.10
--------------------------------------------------------------------------------
                         Iteration time: 0.82s
                           Time elapsed: 00:27:14
                                    ETA: 00:00:00

Loading latest model: /home/ubuntu/locomani/UniLab/logs/rsl_rl_ppo/RangerBoxReach/2026-08-06_17-29-05_mujoco/model_1999.pt
/home/ubuntu/locomani/UniLab/.venv/lib/python3.10/site-packages/rsl_rl/utils/utils.py:243: UserWarning: The observation configuration dictionary 'obs_groups' does not contain the 'critic' key. As an observation group with the name 'critic' was found, this is assumed to be the appropriate observation. Consider adding the 'critic' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
  warnings.warn(
--------------------------------------------------------------------------------
Resolved observation sets: 
	 default :  ['policy']
	 actor :  ['actor']
	 critic :  ['critic']
--------------------------------------------------------------------------------
Actor Model: MLPModel(
  (obs_normalizer): EmpiricalNormalization()
  (distribution): GaussianDistribution()
  (mlp): MLP(
    (0): Linear(in_features=41, out_features=256, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=64, out_features=10, bias=True)
  )
)
Critic Model: MLPModel(
  (obs_normalizer): EmpiricalNormalization()
  (mlp): MLP(
    (0): Linear(in_features=41, out_features=256, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=64, out_features=1, bias=True)
  )
)
Rendering video to /home/ubuntu/locomani/UniLab/logs/rsl_rl_ppo/RangerBoxReach/2026-08-06_17-29-05_mujoco/play_video.mp4...
Rendering playback frames...
Rendering 200 frames for 16 envs with 8 processes...
Done.
