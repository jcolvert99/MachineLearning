import os


clear = lambda: os.system("cls")  #lambda most useful for quick functions (another high-level function is calling it)
clear()

#def remainder(num):
#   return num % 2

remainder = lambda num: num % 2

print(remainder(5))
print(type(remainder)) #num is the argument, % 2 is the expression, remainder is the name
                       #returns a function

product = lambda x,y: x*y
print(product(2,3))


'''
def myfunc(x)
map(function, iterable)
filter(function, iterable)
'''


def myfunction(num):
    return lambda x: x * num

result10 = myfunction(10)   #result10 is a function that requires 2 arguments
                            #same thing as result10 = lambda x: x * 10
result100 = myfunction(100)

print(result10(9))
print(result100(9))


def myfunc(n):
    return lambda a: a * n

mydoubler = myfunc(2)  #n becomes 2
mytripler = myfunc(3)  #n becomes 3

print(mydoubler(11))  #a becomes 11
print(mytripler(11))


numbers = [2,4,6,8,10,3,18,14,21]

filtered_list = list(filter(lambda num: (num > 7),numbers))  #what lambda is useful for: functions within functions without having to define that function
print(filtered_list)

mapped_list = list(map(lambda num: num % 2,numbers))  #applies a function to every element in an iterable
print(mapped_list)


x = lambda a: a + 10
print(x(5))

x = lambda a,b,c: a + b + c
print(x(5,6,7))

def addition(n):
    return n + n

numbers = [1,2,3,4]
result = map(addition,numbers)
result = map(lambda x: x + x,numbers)  #returns the same thing
print(list(result))

#lambda is a single line function, does not have a name, can have any number of arguments, but only one expression
#most commonly used- passed as an argument to another function




