# OpenAIGym-SAC
Course Project at UMN (EE5271)

### Code files
* **ReplayBuffer.py:** Implements sample buffer for off policy learning
* **VAE.py:** Implements Variational Autoencoder network using pytorch framework 
* **VAE_main.py:** Implements training and testing VAE along with tensorboard visualizations
* **Models.py:** Implements Actor and Critic networks using pytorch framework
* **SAC.py:** Implements the Soft Actor Critic(v2) RL Agent 
* **CarRacingSAC_main.py:** Implements learning and testing SAC agent for CarRacing-v0 gym environment 

### Running the code
**VAE:**
* To collect data using random policy and train the VAE: <br />
=> python3 VAE_main.py --model_dir your_model_dir --vae_model_name your_vae_model_name --collect_data y --vae_pipeline train_vae

* To just train/re-train the VAE without data collection: <br />
=> python3 VAE_main.py --model_dir your_model_dir --vae_model_name your_vae_model_name --vae_pipeline train_vae

* To test the VAE after training: <br />
=> python3 VAE_main.py --model_dir your_model_dir --vae_model_name your_vae_model_name --vae_pipeline test_vae

**SAC:**
* To train the SAC agent on CarRacing-v0 environment: <br />
=> python3 CarRacingSAC_main.py --model_dir your_model_dir --vae_model_name your_vae_model_name --sac_model_name your_sac_agent_name --agent_pipeline train_agent
  
* To test the SAC agent on CarRacing-v0 environment: <br />
=> python3 CarRacingSAC_main.py --model_dir your_model_dir --vae_model_name your_vae_model_name --sac_model_name your_sac_agent_name --agent_pipeline test_agent

### Results
* For 100-random tracks SAC gave a score of 901 &pm; 16 <br />
* Best performing model on the same task reported a score of 906 &pm; 21 <br />
* Reference: https://worldmodels.github.io/ <br />

### VAE Perception
* Left: Observation
* Right: VAE Reconstructed Observation
<img src="https://user-images.githubusercontent.com/43849409/145502470-a5340fc0-98d7-4ece-bbd9-f161e16a21ce.gif" width="750" height="300" />

### Policy visualization
https://user-images.githubusercontent.com/43849409/145500006-53425b04-59c9-4595-bc2a-001a2930a822.mp4


### Tensorboard training and evaluation metrics
* VAE train loss:
<img src="https://user-images.githubusercontent.com/43849409/145497069-b1954023-88a9-4d79-aad7-1c327a8919bd.jpg" width="500" height="200" />

* VAE test loss:
<img src="https://user-images.githubusercontent.com/43849409/145497089-6174e3d9-fd34-493c-a093-2dd8849f6391.jpg" width="500" height="200" />

* SAC Actor loss:
<img src="https://user-images.githubusercontent.com/43849409/145497119-4558ded8-395c-4db7-a652-c5c6c422290a.jpg" width="500" height="200" />

* SAC Critic loss:
<img src="https://user-images.githubusercontent.com/43849409/145497138-7d40e656-af1a-470f-a34e-148082f03129.jpg" width="500" height="200" />

* SAC Episodic return:
<img src="https://user-images.githubusercontent.com/43849409/145497191-447f64e5-647d-455f-a14a-6136d626cdd3.jpg" width="500" height="200" />

* SAC Evaluated return:
<img src="https://user-images.githubusercontent.com/43849409/145497209-0177be7a-d7de-42a9-927f-c4847b11133f.jpg" width="500" height="200" />



