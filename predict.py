import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import datetime # time, calendar,
from sklearn import metrics

from sklearn.cluster import DBSCAN


data_src = './data/002/Trajectory/'
df_all = os.listdir(data_src)
url = [data_src + i for i in df_all]

# df_list = pd.read_csv(url[0])

columns=['lat','lng','zero','alt','days','date','time']
df = pd.concat([pd.read_csv(i, header=6, names=columns,index_col=False) for i in url])
df.drop(['zero', 'days'], axis=1, inplace=True)
df['day'] = df.date.apply(lambda x : x[0:10])
df['day']=df.day.apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').strftime('%A'))
print(np.unique(df.day))

df.dropna()
# print(df)


# total=0
# df['numberDays']=0
# for j in range(len(df)):
#     df.numberDays[total:total+len(df[j])]=j
#     total=total+len(df[j])
# df
# np.max(df.iloc[:,1])
# pd.isnull(df).sum() > 0

#quick clustering

kms_per_radian = 3371.0088
e = 1.5/ kms_per_radian

datamin=df.iloc[::8,:]
coords = datamin.as_matrix(columns=['lat', 'lng'])
rads = np.radians(coords)
db = DBSCAN(eps=e, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_

# get the number of clusters
n_clusters_ = len(set(cluster_labels))
print('Estimated number of clusters: %d' % n_clusters_)

datamin.reset_index(drop=True,inplace=True)
datamin.head()


# np.unique(datamin.cluster)
#calculate transition probservation (movements between states)
counter=0 

trans_probrob = np.zeros(shape=(len(clusters), len(clusters)))
for row in range(0,datamin.shape[0]-1):
    trans_probrob[datamin.iloc[row+1, 6]][datamin.iloc[row, 6]] += 1

for i in range(len(clusters)):
    for j in range(len(clusters)):
        trans_probrob[i][j] = trans_probrob[i][j]/len(clusters[i])

#calculate emission probservation (probservation of the observationervation given the hidden states) - shape #clustersx7
e_prob = np.zeros(shape=(len(clusters), 7))


for row in range(datamin.shape[0]):
    row_e_prob = datamin.iloc[row, 6]
    if datamin.iloc[row, 5] == 'Sunday': 
        e_prob[row_e_prob][6] += 1
    elif datamin.iloc[row, 5] == 'Saturday': 
        e_prob[row_e_prob][5] += 1
    elif datamin.iloc[row, 5] == 'Friday': 
        e_prob[row_e_prob][4] += 1
    elif datamin.iloc[row, 5] == 'Thursday': 
        e_prob[row_e_prob][3] += 1
    elif datamin.iloc[row, 5] == 'Wednesday': 
        e_prob[row_e_prob][2] += 1
    elif datamin.iloc[row, 5] == 'Tuesday': 
        e_prob[row_e_prob][1] += 1
    elif datamin.iloc[row, 5] == 'Monday': 
        e_prob[row_e_prob][0] += 1

for i in range(len(clusters)):
    for j in range(7):
        e_prob[i,j] = e_prob[i,j]/np.sum(e_prob[i]) 
        
print(trans_probrob, e_prob)


initial_prob = [len(clusters[i])/len(datamin) for i in range(len(clusters))]

count_observation = pd.value_counts(datamin.iloc[:, 5])
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

visible_states_prob = [count_observation.loc[d]/len(datamin) for d in days]
# datamin.iloc[:, 5].index

print(initial_prob)


def predict(observationervation, e_prob, initial_prob, states_prob, len_clusters):
    
    ps = []
    for i in range(len_clusters):

        p_observation_given_cluster = e_prob[i][observationervation]
        p_cluster = initial_prob[i]
#         p_observation = states_prob[observationervation]
        p_observation = np.sum([e_prob[i][observationervation]*initial_prob[i] for i in range(len_clusters)])
        ps.append(p_observation_given_cluster*p_cluster/p_observation)
#         print(p_observation_given_cluster, p_cluster, p_observation, ps[i])
        
    max1, argmax1 = max(ps), np.argmax(ps)
    ps[argmax1] = 0

    max2, argmax2 = max(ps), np.argmax(ps)
    ps[argmax2] = 0
    
    max3, argmax3 = max(ps), np.argmax(ps)

    return [max1, argmax1], [max2, argmax2], [max3, argmax3]

max_probservation = predict(1, e_prob, initial_prob, visible_states_prob, 11)

d=[]

for i in max_probservation:
    d.append(i)
df = pd.DataFrame(d, columns = ['Probability', 'Cluster'])

print(df)


def viterbi(observation, states, initial_prob, trans_prob, e_prob):


	table = []
	# []
	for each in states:
		table[0] = {}
		tmp = initial_prob[each] * e_prob[each][observation[0]]
		table[0][each] = {"prob": tmp, "prev": None}


	#t>0

    for t in range(1, len(observation)):

        table.append({})

        for each in states:
        	allp = []

        	for p in states:
        		allp.append(V[t-1][p]["prob"]*trans_prob[p][each])
            probmax = max(allp)

            for prev_state in states:

                if table[t-1][prev_state]["prob"] * trans_prob[prev_state][st] == probmax:
                	# print(probmax * e_prob[st][observation[t]])
)
                    finalprob = probmax * e_prob[st][observation[t]]
                    V[t][each] = {"prob": finalprob, "prev": prev_state}

                    break

    #check
    for t in table:

        print(t)

    
    

    # max probability
    for v in table[-1].values():
	    max_prob = max(v["prob"])

    pre = None

    #most probable state
	chain = []
    for k,v in table[-1].items():

        if v["prob"] == max_prob:

            chain.append(k)

            pre = k

            break

    # traceback
    n = len(table) - 2
    for t in range(n, -1, -1):

        chain.insert(0, V[t + 1][pre]["prev"])
        # print(chain)
        pre = V[t + 1][pre]["prev"]

    # print(chain)
    print ("Predicted: " + str(chain) + " with max prob: " + max_prob)

    return


# observation = [0,1,1,1,1,2,3]
# viterbi(observation, states, initial_prob, trans_prob, e_prob)

