#Converts the dengue fever risk into a number
#This allows the risks to be plotted on a scatter graph
#The new risks are written to a new file alongside the photo number, latitude and longitude

#CSV data file of our experiment results
file = open("data.csv","r")
#File which new data is being written to
numbers = open("geomapdata.txt","w")

prefix = ""

#Iterating through results
for line in file:
  line = line.split(",")
  if line[1] == "High risk":
    risk = 1
  elif line[1] == "Low risk":
    risk = 0.5
  elif line[1] == "No risk":
    risk = 0
  
  latitude = abs(float(line[4]))
  longitude = abs(float(line[5]))
  outline = prefix + str(line[0]) + "," + str(latitude) + ","+ str(longitude) + "," + str(risk)
  numbers.write(outline)
  prefix = "\n"

file.close()
numbers.close()
