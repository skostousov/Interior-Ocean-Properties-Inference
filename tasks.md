| Job ID        | Name           | Date Sub.  | Duration | Status    | Specifics  | Notes |
|---------------|----------------|------------|----------|-----------|------------|-------|
| 620854-620861 | continue_train | 17/06/2025 | 20:00:00 | Pending   |            |       |
| 620662        | eval           | 17/06/2025 | 24:00:00 | Pending   |            | saved_BoB_monthly_models/MODEL:UNetRegressionSE>TRAINSTART:20250617_092253>DATAFILE:BoBMonthly_1993-2003.nc>STRAT:test_indices_BoBMonthly_small_01.pt>|
| 620499-620503 | continue_train | 17/06/2025 | 20:00:00 | Completed |            |       |
| 620504        | eval           | 17/06/2025 | 24:00:00 | Analyzed  | BoBMonthly | In need of hp tuning, saved_BoB_monthly_models/MODEL:UNetRegression>TRAINSTART:20250616_173058>DATAFILE:BoBMonthly_1993-2003.nc>STRAT:test_indices_BoBMonthly_small_01.pt> |
| 620853        | continue_train | 17/06/2025 | 20:00:00 | Completed |            |       |
| 620498        | continue_train | 17/06/2025 | 20:00:00 | Completed |            |       |
| 620850        | eval           | 17/06/2025 | 24:00:00 | Analyzed  | BoBDaily   | Grossly in need of hp tuning: saved_BoB_daily_models/MODEL:UNetRegressionSE>TRAINSTART:20250617_091651>DATAFILE:BoBDaily_1993-1993.nc>STRAT:test_indices_BoBDaily_small_01.pt>   |
| 620871        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 40 trials | Finished early due to time limit, hptuning_UNET_SE/train_model_2025-06-17_11-07-23|
| 620909        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 10 trials, BoBMonthly | hptuning_EBAM_CNN_BoBMonthly_1993-2003/train_model_2025-06-17_16-30-03|
| 620908        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 10 trials, BoBMonthly | hptuning_DA_CNN_BoBMonthly_1993-2003/train_model_2025-06-17_16-04-33|
| 620892        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 40 trials | hptuning_DA_CNN/train_model_2025-06-17_13-13-42|
| 620886        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 40 trials | hptuning_EBAM_CNN/train_model_2025-06-17_12-11-06|
| 620907        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 10 trials, BoBMonthly | hptuning_UNET_SE_BoBMonthly_1993-2003/train_model_2025-06-17_15-51-02|
| 620904        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 30 trials, BoBDaily| hptuning_EBAM_CNN_BoBDaily_1993-1993/train_model_2025-06-17_15-46-02|
| 620905        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 30 trials, BoBDaily| hptuning_DA_CNN_BoBDaily_1993-1993/train_model_2025-06-17_15-46-41|
| 620906        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 30 trails, BoBDaily| hptuning_UNET_SE_BoBDaily_1993-1993/train_model_2025-06-17_15-46-47|
| 620902        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 30 trials, BoBDaily | hptuning_DA_CNN_BoBDaily_1993-1993/train_model_2025-06-17_15-46-41|
| 620903        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 10 trials, Monthly | hptuning_EBAM_CNN_ten_sample_1993-2003/train_model_2025-06-17_15-45-02|
| 620899        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 10 trials, monthly | hptuning_UNET_SE_ten_sample_1993-2003/train_model_2025-06-17_14-58-02|
| 620900        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 30 trials, Monthly | hptuning_EBAM_CNN_ten_sample_1993-2003/train_model_2025-06-17_14-58-02
| 620901        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 30 trials, monthly  | hptuning_DA_CNN_ten_sample_1993-2003/train_model_2025-06-17_14-58-02
| 620897        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 30 trials, Daily| hptuning_UNET_SE_small_daily_alternative_sample_1993-1993/train_model_2025-06-17_14-56-04|
| 620896        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Completed | 30 trials, daily | hptuning_EBAM_CNN_small_daily_alternative_sample_1993-1993/train_model_2025-06-17_14-56-31|
| 620895        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Completed | 30 trials, daily | hptuning_DA_CNN_small_daily_alternative_sample_1993-1993/train_model_2025-06-17_14-56-33|
| 620911        | hp_DA_ray      | 17/06/2025 | 24:00:00 | Running   | 10 trials, BobMonthly| hptuning_DA_CNN_BoBMonthly_1993-2003/train_model_2025-06-17_20-11-06|
| 620910        | hp_EBAM_ray    | 17/06/2025 | 24:00:00 | Running   | 10 trials, BoBMonthly| hptuning_EBAM_CNN_BoBMonthly_1993-2003/train_model_2025-06-17_20-11-02|
| 620870        | hp_UNETSE_ray  | 17/06/2025 | 24:00:00 | Completed | 20 trials | hptuning_UNET_SE/train_model_2025-06-17_11-03-584
| 


### RUN HISTORY

- dailyalternativesmall

    - UNETSE

        1. 
            - Name: "saved_models/cluster/saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250610_173939>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_cluster.pt>"
            - hp: grid_size=21, batch=50, lr=e-4ore-5?, base_filters=32, reduction=8
            - total_test_loss: 0.19952962516564302
            - qualitative analysis: good

        2. 
            - Name: "saved_models/cluster/saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250611_104538>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_cluster.pt>"
            - hp: grid_size=17, batch=50, lr=e-4ore-5?, base_filters=32, reduction=8
            - total_test_loss: 0.3319137859682907
            - qualitative analysis: good

        3. 
            - Name: "saved_models/cluster/saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250613_113852>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_cluster.pt>"
            - hp: grid_size=17, batch=100, base_filters=64, reduction=4
            - total_test_loss: 8.93909485973229
            - qualitative analysis: uniform

    - EBAM

        1. 
            - Name: saved_models/cluster/saved_models/saved_daily_alternative_small_models/MODEL:EBAM_CNN>TRAINSTART:20250612_154451>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_cluster.pt>
            - hp: grid_size=17, batch=100, lr=0.0005. num_heads=3
            - qualitative analysis : uniform
            - total_test_loss: 0.6112745401537959
    
    - DA

        1. 
            - Name: saved_models/cluster/saved_models/saved_daily_alternative_small_models/MODEL:DA_CNN>TRAINSTART:20250613_113722>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_cluster.pt>
            - hp: grid_size=17, batch=50, lr=0.001, first_layer_filters=64, kernel_size=3
            - qualitative analysis : uniform
            - total_test_loss: 13.3065884008189

- monthly

    - UNETSE

        1. 
            - Name: saved_models/cluster/saved_models/saved_monthly_models/MODEL:UNetRegressionSE>TRAINSTART:20250611_105333>DATAFILE:ten_sample_1993-2003.nc>STRAT:test_indices_monthly_ten_01.pt>
            - hp: grid_size=17, batch=500, lr=e-4ore-5?, base_filters=32, reduction=8
            - total_test_loss: 2.824481842308436
            - qualtiative analysis: poor

        2.
            - Name: 


        