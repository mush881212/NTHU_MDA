#!/usr/bin/env python
# coding: utf-8

# # 巨量資料分析 HW1 : Matrix Multiplication
# ### 107062117 李采蓉

# In[1]:


from pyspark import SparkConf, SparkContext


# #### mapper1:  
#  
# Ｍ矩陣的 [i][j] 在矩陣相乘之後，會出現在ans的 [i][0:500] 這些位置  
# N矩陣的 [j][k] 在矩陣相乘之後，會出現在ans的 [0:500][k] 這些位置  
# 並且MN擁有相同j的值會進行相乘  
# 因此先把資料map到 (( j, output_x, output_y), value)  

# In[2]:


def mapper1(l):
    l = l.split(',')
    maplist = []
    for i in range(500):
        if l[0] == 'M':
            maplist.append(((l[2], l[1], str(i)), int(l[3])))
        else:
            maplist.append(((l[1], str(i), l[2]), int(l[3])))
    
    return maplist


# #### reducer1:    
# 對有相同 j 且 ans 在同一個位置的值相乘  

# In[3]:


def reducer1(x, y): 
    return x*y


# #### mapper2:  
# 因為已經對相同j值的MN進行相乘，之後要對在ans同一個位置的值進行相加  
# 因此MN重新map到：  
# M: ((i, k), value)  
# N: ((i, k), value)  

# In[4]:


def mapper2(l):
    return ((l[0][1], l[0][2]), l[1])


# #### reducer2:  
# 加總所有ans在相同位置的值

# In[5]:


def reducer2(x, y):
    return x+y


# In[6]:


conf = SparkConf().setMaster("local").setAppName("matProduct")
sc = SparkContext(conf=conf)
lines = sc.textFile("input.txt")

lines = lines.flatMap(mapper1)
lines = lines.reduceByKey(reducer1)
lines = lines.map(mapper2)
lines = lines.reduceByKey(reducer2)


# In[7]:


with open('output.txt', 'w') as f:
    for item in lines.collect():
        f.writelines(item[0][0]+","+item[0][1]+","+str(item[1])+'\n')
sc.stop()

