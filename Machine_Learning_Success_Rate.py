#Checks the dengue fever risk produced by our experiment against Centres for Disease Control and Prevention (CSC) data by region
#Outputs the % success rate of the machine learning model in predicting the dengue fever level

import geocoder

#CDC data on dengue fever levels
dengue_risks = open("dengue_risks.txt","r")
dengue_risk_list = []
for line in dengue_risks:
  dengue_risk_list.append(line.strip().split("\t"))
dengue_risks.close()

#File containing country names in that country's language
languages = open("Country_language.txt","r")
language_list = []
for line in languages:
  add = line.strip().split("\t")
  add = [add[1],add[-1]]
  language_list.append(add)
languages.close()

#CSV data file of our experiment results
csv_file = open("Astropidata.csv","r")
lats = []
longs = []
machine_risks = []
for line in csv_file:
  line = line.strip().split(",")
  lats.append(line[4])
  longs.append(line[5])
  machine_risks.append(line[1])
csv_file.close()


correct = 0
total_on_land = 0

#Iterate through the data from each location
for i in range(len(lats)):

  country = ""
  
  lat = lats[i]
  long = longs[i]

  #Determining what country the location is in
  g = geocoder.osm([lat, long], method='reverse')
  if g.json != None:
    country = g.json['country']
    country = country.title()


  split_country = country.split(" ")
  if country != "":
    #Converts country name produced by geocoder into the country name in English
    for item in language_list:
      if country in item[1] or split_country[0] in item[1]:
        country = item[0]
    country = country.title()
    
  
  found = False
  #Iterates through CDC data to find true dengue fever risk for the country
  for item in dengue_risk_list:
    if item[0] == country:
      found = True
      risk = item[1]

  if found == False:
    if country == "":
      risk = "Not in a country"
    else:
      risk = "No risk"

  if risk == "Sporadic/Uncertain":
    risk = "Low risk"
  elif risk == "Frequent/Continuous":
    risk = "High risk"

  #Compares generated risk and true denge fever risk
  if risk != "Not in a country":
    total_on_land += 1
    machine_risk = machine_risks[i]

    if risk == machine_risk:
      correct += 1

#Percentage success rate of machine learning model
print(f"{(correct/total_on_land * 100):.2f}%")
