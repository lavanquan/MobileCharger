# MobileCharger

Chuong trinh su dung thu vien mo phong la pymote da duoc them cac module ve energy
Co the tai ve tai dia chi https://github.com/lavanquan/pymote \

Thu vien dung de giai quyet bai toan MILP la Pulp 

In order to predict the energy used  in the next period:__
step1: load model by load_model function in energy_predict/commons/energy_predictor.py__
        + input includes: 1. path to model(leave it default)__
                          2.gpu device: (leave it default__
step2: predict by predict function in energy_predict/commons/energy_predictor.py__
        + input includes: 1. model loads from step 1__
                          2. data: n * y matrix. n is the number of sensor nodes, y is the number of timesteps in the previous three periods__
