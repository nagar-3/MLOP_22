# Bonus Question results

To solve this problem i have created to flask app server one is running on container and other is running on base machine(trained model exists on this machine). I have created login in python code where it flask app running inside container will execute http request towards base machine flask app and will received the trained model predicted results.


## Container running on [http://172.17.0.2:5000]
```
--------------------------------------------
root@00383c475b07:/exp#
root@00383c475b07:/exp#
root@00383c475b07:/exp#
root@00383c475b07:/exp# python Classification_quiz_4_bonus.py
 * Serving Flask app 'Classification_quiz_4_bonus'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://172.17.0.2:5000
Press CTRL+C to quit


172.17.0.1 - - [14/Nov/2022 06:17:18] "POST /svm HTTP/1.1" 201 -
^Croot@00383c475b07:/exp#
root@00383c475b07:/exp#
root@00383c475b07:/exp#
```


## model present on [http://172.17.0.1:5000]
```
-------------------------------------------
(my_conda_env) root@mlops:~/MLOP_22#
(my_conda_env) root@mlops:~/MLOP_22#
(my_conda_env) root@mlops:~/MLOP_22# python Bonus_Q2_Machine_2_code.py
 * Serving Flask app 'Bonus_Q2_Machine_2_code'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://172.17.0.1:5000
Press CTRL+C to quit
172.17.0.2 - - [14/Nov/2022 11:47:18] "POST /predict HTTP/1.1" 201 -
(my_conda_env) root@mlops:~/MLOP_22#
(my_conda_env) root@mlops:~/MLOP_22#


```


## Testing results
```
(base) root@mlops:~/MLOP_22#
(base) root@mlops:~/MLOP_22#
(base) root@mlops:~/MLOP_22# curl -X POST http://172.17.0.2:5000/svm  -H 'Content-Type: application/json' -d '{"img1": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"],"img2": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}'
{"Image#1":1,"Image#2":1,"Result ":"Both image are same"}
(base) root@mlops:~/MLOP_22#
(base) root@mlops:~/MLOP_22#
(base) root@mlops:~/MLOP_22#

![image](https://user-images.githubusercontent.com/89742374/201590133-08c12c94-99ae-464b-8293-8fec3b574f87.png)

```

