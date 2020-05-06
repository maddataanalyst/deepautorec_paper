Model tracker:

1. [Features + 30 x dim embed + ubias + ibias] + drop(0.2) + dense (32 x 16), drop(0.2)

    a. val mape -0.1822
    
    b. test mape -0.195
   
2. [Features(dense 32) + 30 x dim embed + ubias + ibias] + drop(0.2) + dense(32 x 16), drop (0.2)

    a. val mape -0.1675
    
    b. test mape: -0.1825
    
3. [Features(dense 16) + 30 x dim embed + ubias + ibias] + drop(0.2) + dense(32 x 16), drop (0.2)

    a. val mape: -0.131
    
    b. test mape: -0.144