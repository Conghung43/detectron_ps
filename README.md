Project requirement:
1, Detect ps error based on number of ps
2, Detect ps error based on distance from two ps 
3, Predict ps error based on predictive analytics algorithms
4, Detect unusual ps based on width of ps

Requirement analisys:

req1: Detect ps error based on number of ps
From beginning, program will show predicted number of PS and save it to config file. => Predicted number of PS will be confirmed by operator.
During operation, if number of PS lower than normal, AOI will send warning sign.

req2: Detect ps error based on distance from two ps 
If the closest distance between any part of PS lower than config, AOI will send warning sign.
Currently, There are 3 cases: V, X and two PSs stick together

req3: Predict potential overlap in outside camera FOV

req4: Highlight unusual width of PS

Program logical:

req1: number_ps_analysis
Detect ps error based on number of ps
If detected PSs > Operator input: 
    Remove wrong ps detection by multiple filter
Elif detected PSs > Operator input: 
    => Save image and log => contact with admin                     
Else:
    Pass

req2: distance_analisys
Find closest distance between two continuos PSs. If that distance over config, AOI will send warning sign.

req3: prediction_analisys
Predict potential overlap in outside camera FOV
If overlap position outside of FOV, program will besed on setting range to predict potential overlap.

req4: width_analysis
Highlight unusual width of PS# detectron_ps


03/21: 
To do: 
1,Delete overlap result
2, (auto labeling)Auto find ps based on it detection result #frame[np.where(mask_layers[0].mask == True)]
find line based on hsv color
using hsv in teest function
3, Create polygon for training
