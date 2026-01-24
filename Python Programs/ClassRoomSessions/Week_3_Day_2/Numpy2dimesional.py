import numpy as num

twodimarray=num.array([[1,2,3,4],
                       [5,6,7,8],
                       [9,0,1,2]])
print(twodimarray)
print(twodimarray.shape)
print(twodimarray[0:2,1:3])

# for i in twodimarray:
#     print("Plane",i)
#     for j in i:
#         print("Inside Row", j)
#         print("")
#         for k in j:
#             print("Column", k)
#             print("")