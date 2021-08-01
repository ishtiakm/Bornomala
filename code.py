import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_dir='/kaggle/input/bornomala-all-data/Bornomala_datasets/bornomala_all/train'
val_dir= '/kaggle/input/bornomala-all-data/Bornomala_datasets/bornomala_all/validation'
test_dir='/kaggle/input/bornomala-all-data/Bornomala_datasets/bornomala_all/test'
no_class=84
train_sample=no_class*200
val_sample=no_class*70
test_sample=no_class*73

from keras.layers import Dense,Conv2D,Dropout,MaxPool2D,Flatten,BatchNormalization
from keras.models import Sequential
dim=100
input_shape=(dim,dim,3)

model=Sequential()
model.add(Conv2D(32,(5,5),activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())


model.add(Conv2D(128,(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.25))#0.1 for compl

model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))#0.25 for compl

model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(no_class,activation='softmax'))
model.summary()


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
rotation_range=10,
width_shift_range=0.15,
height_shift_range=0.15,
shear_range=0.15,
zoom_range=0.15,
)
val_datagen=ImageDataGenerator(rescale=1./255)
batch_size_train=50 #6,32,50,80,100#50 for coml
batch_size_val=40# 20,40,60,70,120 #40for coml
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(dim,dim),
    batch_size=batch_size_train,
    class_mode='categorical')

val_generator=val_datagen.flow_from_directory(
    val_dir,
    target_size=(dim,dim),
    batch_size=batch_size_val,
    class_mode='categorical')


from keras import optimizers,callbacks
call=callbacks.EarlyStopping(
    monitor="val_acc",
    min_delta=0,
    patience=7,#ideal 5
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
reduce=callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
epochs=48


import time
start=time.time()
history=model.fit_generator(
    train_generator,
    steps_per_epoch=train_sample//batch_size_train,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_sample//batch_size_val,
    callbacks=[call,reduce]
)
end=time.time()
print(f"{(end-start)//60} min, {(end-start)%60} sec")
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
from matplotlib import pyplot as plt
x=list(range(1,len(acc)+1))
plt.plot(x,acc,'r')
plt.plot(x,val_acc,'b')
plt.figure()
plt.plot(x,loss,'r')
plt.plot(x,val_loss,'b')


from keras.preprocessing.image import ImageDataGenerator
batch_size=test_sample//no_class
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(dim,dim),
batch_size=batch_size,
class_mode='categorical')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_sample//batch_size)
print('test acc:', test_acc*100,'%')


#Training and testing of the model ended
#Following section contains the formation of confusion matrix, 
#calculation of recall,precision and f1 score
#Identification of classes with most errors

import numpy as np
def conf_mat(y,pred,matrix):
    
    for i in range(len(y)):
        matrix[y[i]][pred[i]]+=1
    return matrix

steps=(73*no_class)//batch_size
matrix=np.zeros((no_class,no_class))
true_y=[]
pred_y=[]
for i in range(steps):
    temp_img=test_generator.__getitem__(i)[0]
    label=test_generator.__getitem__(i)[1]
    y=label.argmax(axis=1)
    true_y.append(y)
    pred=model.predict(temp_img)
    p=pred.argmax(axis=1)
    pred_y.append(p)
    mat=conf_mat(y,p,matrix)
    
    
from matplotlib import pyplot as plt
print(mat)

plt.imshow(mat)
m=np.array(mat)
print(np.sum(m,axis=1))



import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

letters="o a i I u U Ri A OI OO OU ka kha ga gha Uno cha Chha ja jha Ino Ta Thhha Da Dha Murdho-Na to tho do dho Donto-Na Po Pho Ba Bha Ma Za ra la sh sa SH ha R|d|a R|dh|a O|Z| Khanda-ta onusshar bishorgo chondrobindu zero 1 2 3 4 5 6 7 8 9 KSHio bda NGO Ksa stha sfa chcha kta mLo Sno spo Hmo pto mbo nDo mVo ththa SHTHA lpo SHPO ndo nko mmo nTHO"
#letters="0 1 2 3 4 5 6 7 8 9"
#letters="o a i I u U Ri A OI OO OU"
#letters="ka kha ga gha Uno cha Chha ja jha Ino Ta Thhha Da Dha Murdho-Na to tho do dho Donto-Na Po Pho Ba Bha Ma Za ra la sh sa SH ha R|d|a R|dh|a O|Z| Khanda-ta onusshar bishorgo chondrobindu"
#letters="KSHio bda NGO Ksa stha sfa chcha kta mLo Sno spo Hmo pto mbo nDo mVo ththa SHTHA lpo SHPO ndo nko mmo nTHO"
#letters="o a i I u U Ri A OI OO OU ka kha ga gha Uno cha Chha ja jha Ino Ta Thhha Da Dha Murdho-Na to tho do dho Donto-Na Po Pho Ba Bha Ma Za ra la sh sa SH ha R|d|a R|dh|a O|Z| Khanda-ta onusshar bishorgo chondrobindu"
letters=letters.split(' ')

df_cm = pd.DataFrame(mat, index = [i for i in letters],
                  columns = [i for i in letters])
plt.figure(figsize = (20,14))
sn.heatmap(df_cm, annot=True)


#f1 score, precision, recall from confusion matrix
def calc_f1(mat,cls):
    tp=mat[cls][cls]
    fn=sum(mat[cls,:])-tp
    fp=sum(mat[:,cls])-tp
    rec=tp/(fn+tp)
    prec=tp/(fp+tp)
    f1=2*(rec*prec)/(rec+prec)
    return rec,prec,f1
recall=[]
precision=[]
f1score=[]
for i in range(no_class):
    trec,tprec,tf1=calc_f1(mat,i)
    recall.append(trec)
    precision.append(tprec)
    f1score.append(tf1)
#print(recall)
#print(precision)
#print(f1score)
print(sum(recall)/len(recall),sum(precision)/len(precision),sum(f1score)/len(f1score))
plt.bar(list(range(no_class)),recall)
plt.show()
plt.figure()
plt.bar(list(range(no_class)),precision)
plt.show()
plt.figure()
plt.bar(list(range(no_class)),f1score)
plt.show()
for c in range(no_class):
    print(f"{letters[c]}--> {f1score[c]}")
    
    
    
print(f"overall recall: {sum(recall)*100/len(recall)}, precision:{sum(precision)*100/len(recall)} and f1score:{sum(f1score)*100/len(recall)}")


for i,letter in enumerate(letters):
    print(f"{i}--> {letter}")
    
def Sort_Tuple(tup):  
  
    # reverse = None (Sorts in Ascending order)  
    # key is set to sort using second element of  
    # sublist lambda has been used  
    tup.sort(key = lambda x: x[1])  
    return tup  
recallzip=list(zip(list(range(no_class)),recall))
recall_sort=Sort_Tuple(recallzip)
recall_indx=[recall_sort[i][0] for i in range(10)]
recall_score=[recall_sort[i][1] for i in range(10)]
print(recall_indx)

preczip=list(zip(list(range(no_class)),precision))
prec_sort=Sort_Tuple(preczip)
prec_indx=[prec_sort[i][0] for i in range(10)]
prec_score=[prec_sort[i][1] for i in range(10)]
print(prec_indx)

f1zip=list(zip(list(range(no_class)),f1score))
f1_sort=Sort_Tuple(f1zip)
f1_indx=[f1_sort[i][0] for i in range(10)]
f1_score=[f1_sort[i][1] for i in range(10)]
print(f1_indx)


def recallsecondmax(row,matrix):    
    temp=list(matrix[row,:])
    #print(temp)
    temp2=[i for i in temp]
    temp2.remove(max(temp))
    secmax=max(temp2)
    ind=temp.index(secmax)
    return ind,temp[ind]
for i in recall_indx:
    
    idx,n=recallsecondmax(i,matrix)
    print(f"{letters[i]} is mostly mistakenly prediced as {letters[idx]}--> for {n} times")
    
    
def precsecondmax(column,matrix):
    temp=list(matrix[:,column])
    temp2=[i for i in temp]
    temp2.remove(max(temp))
    secmax=max(temp2)
    ind=temp.index(secmax)
    return ind,temp[ind]
for i in prec_indx:
    idx,n=precsecondmax(i,matrix)
    print(f"{letters[i]} is predicted in place of {letters[idx]}--> for {n} times")
    
 
def f1(letter,matrix):
    tp=matrix[letter,letter]
    fn=sum(matrix[letter,:])-tp
    fp=sum(matrix[:,letter])-tp
    return fn,fp
for i in f1_indx:
    fn,fp=f1(i,matrix)
    print(f"{letters[i]} was not recognized {fn} times and was predicted wrongly in place of other {fp} times")
    
    
f1_sort_high=Sort_Tuple(f1zip)
f1_sort_high.reverse()
f1_indx_high=[f1_sort_high[i][0] for i in range(10)]
f1_score_high=[f1_sort_high[i][1] for i in range(10)]
print(f1_indx_high)


for i in (f1_indx_high):
    print(f"{letters[i]} was recognized correctyly {recall[i]*100:.3}% times and predicted correctly {precision[i]*100:.3}% times")
    
#If you cant to save this model, uncomment the following line    
#model.save("all2.h5")
