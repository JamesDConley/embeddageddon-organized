# Large Adam 8 bit bnb training info
root@5e7f11f3f0b9:/app# bash shell_scripts/train_embeddageddon_l.sh                                                                                                                                                                                                                                           ████████████████████
============================================
Embeddageddon L Model Training (3584D)                                                                                                                                                                                                                                                                        ████████████████████
============================================
                                                                                                                                                                                                                                                                                                              ████████████████████
=== Training Embeddageddon L Model ===
Arguments saved to: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/args_used.json
Using embeddageddon embeddings from: data/embeddageddon_embeddings/tanh_l2_onecycle_e30_bs8192_lr0.00001_w8/embeddageddon_embeddings_l_3584d.pkl
`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got 512 and max_position_embeddings=512                                                                                                                                                                    ████████████████████
Loading tokenizer...
Total number of parameters: 6057565696                                                                                                                                                                                                                                                                        ████████████████████
Loading tokenizer...
Loaded dataset with 450000 samples                                                                                                                                                                                                                                                                            ████████████████████
Sample text lengths - Mean: 5064.4, Max: 1028429
Loaded dataset with 50000 samples                                                                                                                                                                                                                                                                             ████████████████████
Sample text lengths - Mean: 5069.1, Max: 754039
batches_per_subnetwork : 28125                                                                                                                                                                                                                                                                                ████████████████████
batches_per_subnetwork : 28125
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28125/28125 [3:34:34<00:00,  2.18it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [12:40<00:00, 16.45it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [14:51<00:00, 14.03it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [19:31<00:00, 10.67it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [29:06<00:00,  7.16it/s]
Checkpoint saved at: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/checkpoints/checkpoint_epoch_0_step_28125_subnetwork_s.pt                                                                                                                                                ████████████████████
batches_per_subnetwork : 28125
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28125/28125 [3:34:23<00:00,  2.19it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                    ███████████████████████████████████████████████| 12500/12500 [12:39<00:00, 16.45it/s]                                                                                                                                                                                                                                             
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [14:51<00:00, 14.02it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [19:32<00:00, 10.66it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [29:08<00:00,  7.15it/s]
Checkpoint saved at: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/checkpoints/checkpoint_epoch_0_step_56250_subnetwork_m.pt
batches_per_subnetwork : 28125
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28125/28125 [3:34:33<00:00,  2.18it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [12:40<00:00, 16.43it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [14:52<00:00, 14.00it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [19:32<00:00, 10.66it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [29:05<00:00,  7.16it/s]
Checkpoint saved at: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/checkpoints/checkpoint_epoch_0_step_84375_subnetwork_l.pt                                                                                                                                                 /a" 20:22 25-Oct-25
batches_per_subnetwork : 28125
 50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                           | 13928/28125 [ 50%|████████████████████████████████████▏                                    | 13929/28125 [1:46:12<1:50:53,  2.13it/s]                                                                                                                                                                                                                                                100%|███████████████████████████████████████████████████████████████████████████| 28125/28125 [3:34:37<00:00,  2.18it/s]
100%|█████████████████████████████████████████████████████████████████████████████| 12500/12500 [12:40<00:00, 16.43it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [14:53<00:00, 13.99it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [19:34<00:00, 10.64it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 12500/12500 [29:07<00:00,  7.15it/s]
Checkpoint saved at: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/checkpoints/checkpoint_epoch_1_step_112500_subnetwork_xl.pt
Final model saved at: data/language_models/embeddageddon/8bitopt_tanh_embeddageddon_l_20251025_042843/final_model

============================================
Embeddageddon L training completed!
============================================

=== Training Equivalent L Model ===
Traceback (most recent call last):
  File "/app/src/llm_train.py", line 22, in <module>
    from accelerate import Accelerator
ModuleNotFoundError: No module named 'accelerate'
root@5e7f11f3f0b9:/app# 



# Accelerate FP8 Doesn't seem to be working
root@995348e89857:/app# accelerate test

Running:  accelerate-launch /usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py
stdout: **Initialization**
stdout: Testing, testing. 1, 2, 3.
stdout: Distributed environment: DistributedType.NO
stdout: Num processes: 1
stdout: Process index: 0
stdout: Local process index: 0
stdout: Device: cuda
stdout: 
stdout: Mixed precision type: fp8
stdout: 
stdout: 
stdout: **Test process execution**
stdout: 
stdout: **Test split between processes as a list**
stdout: 
stdout: **Test split between processes as a dict**
stdout: 
stdout: **Test split between processes as a tensor**
stdout: 
stdout: **Test split between processes evenly**
stdout: 
stdout: **Test split between processes as a datasets.Dataset**
stdout: Skipped because Hugging Face datasets is not available
stdout: 
stdout: **Test random number generator synchronization**
stdout: All rng are properly synched.
stdout: 
stdout: **DataLoader integration test**
stdout: Non-shuffled dataloader passing.
stdout: Shuffled dataloader passing.
stdout: Non-shuffled central dataloader passing.
stdout: Shuffled central dataloader passing.
stdout: 
stdout: **Training integration test**
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Training yielded the same results on one CPU or distributed setup with no batch split.
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Training yielded the same results on one CPU or distributed setup with batch split.
stdout: Keep fp32 wrapper check.
stderr: Traceback (most recent call last):
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 947, in <module>
stderr:     main()
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 931, in main
stderr:     training_check(use_seedable_sampler=False)
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 539, in training_check
stderr:     model = accelerator.prepare(model)
stderr:             ^^^^^^^^^^^^^^^^^^^^^^^^^^
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 1547, in prepare
stderr:     args = self._prepare_te(*args)
stderr:            ^^^^^^^^^^^^^^^^^^^^^^^
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 2046, in _prepare_te
stderr:     raise ValueError(
stderr: ValueError: You must pass a model and an optimizer together to `accelerate.prepare()` when using TransformerEngine.
stderr: Traceback (most recent call last):
stderr:   File "/usr/local/bin/accelerate-launch", line 7, in <module>
stderr:     sys.exit(main())
stderr:              ^^^^^^
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 1241, in main
stderr:     launch_command(args)
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 1235, in launch_command
stderr:     simple_launcher(args)
stderr:   File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 823, in simple_launcher
stderr:     raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
stderr: subprocess.CalledProcessError: Command '['/usr/bin/python', '/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/usr/local/bin/accelerate", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/accelerate_cli.py", line 50, in main
    args.func(args)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/test.py", line 53, in test_command
    result = execute_subprocess_async(cmd)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/testing.py", line 777, in execute_subprocess_async
    raise RuntimeError(
RuntimeError: 'accelerate-launch /usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py' failed with returncode 1

The combined stderr from workers follows:
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 947, in <module>
    main()
  File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 931, in main
    training_check(use_seedable_sampler=False)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py", line 539, in training_check
    model = accelerator.prepare(model)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 1547, in prepare
    args = self._prepare_te(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 2046, in _prepare_te
    raise ValueError(
ValueError: You must pass a model and an optimizer together to `accelerate.prepare()` when using TransformerEngine.
Traceback (most recent call last):
  File "/usr/local/bin/accelerate-launch", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 1241, in main
    launch_command(args)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 1235, in launch_command
    simple_launcher(args)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/commands/launch.py", line 823, in simple_launcher
    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
subprocess.CalledProcessError: Command '['/usr/bin/python', '/usr/local/lib/python3.12/dist-packages/accelerate/test_utils/scripts/test_script.py']' returned non-zero exit status 1.
root@995348e89857:/app# 