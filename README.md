# MobileCharger

Chuong trinh su dung thu vien mo phong la pymote da duoc them cac module ve energy
Co the tai ve tai dia chi https://github.com/lavanquan/pymote 

Thu vien dung de giai quyet bai toan MILP la Pulp 

# Predict energy used in the next period

In order to predict the energy used  in the next period:

step1: load model by load_model function in energy_predict/energy_predictor.py

        + input includes: 1. path to model(leave it default)
        
                          2.gpu device: (leave it default
                          
step2: predict by predict function in energy_predict/energy_predictor.py

        + input includes: 1. model loads from step 1
        
                          2. data: y * n matrix. y is the number of timesteps in the previous three periods,n is the number of sensor nodes 
