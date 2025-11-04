###1
name = "Ail"
age = 20
print("My name is", name, "and I am", age, "years old.")
####

###2
num1 = input("Enter the first number: ")
num2 = input("Enter the second number: ")
num1 = float(num1)
num2 = float(num2)
sum = num1 + num2
difference = num1 - num2
product = num1 * num2
quotient = num1 / num2
print("Sum is: ", sum)
print("Difference is: ", difference)
print("Product is: ", product)
print("Quotient is: ", quotient)
#####

###3
number = int(input('Enter a number: '))
if number %2==0:
   print ("even  number")
else:
   print(" odd number")
#####



###4
for i in range(1, 11):
    print(i)
######



###5
fruits=["apple","banana","mango","orange","grape"]
print(fruits[0])
print(fruits[-1])
fruits.append("watermelon")
print(fruits)



numbers = list(range(1, 11))
print(numbers)
