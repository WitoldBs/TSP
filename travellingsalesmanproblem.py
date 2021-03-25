import numpy as np, matplotlib.pyplot as plt, pandas as pd, random
from math import *

data = pd.read_excel("/Users/witekbieganski/PycharmProjects/TSP/Data/airports.xlsx", engine = "openpyxl")

npdata = data.to_numpy()

cities = []
for i in range(len(npdata)):
    cities.append(npdata[i][0])
cities.remove("ROS")

### SIMULATION PARAMETERS ###
topparents = 4
popsize = 40
mutation_rate = 0.01
iterations = 500

def getdist(city1, city2):         ### THE FUNCTION TO GET DISTANCE ###
    R = 6371 #km
    i1 = int(np.where(npdata == city1)[0])
    i2 = int(np.where(npdata == city2)[0])
    lat1 = npdata[i1][1]
    long1 = npdata[i1][2]
    lat2 = npdata[i2][1]
    long2 = npdata[i2][2]
    p = pi/180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((long2 - long1) * p)) / 2
    dist = round(2*R * asin(sqrt(a)), 2)
    return dist

### ROUTE GENERATION ###
def getRoute(cities):
    route = random.sample(cities, len(cities))
    return route

### ROUTE LENGTH AND FITNESS ###
def routeLength(route):
    length = 0
    i = 0
    while i < len(route)-1:
        length += getdist(route[i],route[i+1])
        i += 1
    return length

def routeFitness(route):
    fitness = 1/routeLength(route)
    return fitness

### GENERATING THE INITIAL POPULATION ###
def getPopulation(poplength, cities):
    population = []
    for i in range((poplength)):
        population.append(getRoute(cities))
        population[i].insert(0,"ROS")
        population[i].append("ROS")
    return population

def getFittest(population):         ### FIND THE ENTITIES WITH THE BEST FITNESS ###
    fitnessScores = []
    for i in range(len(population)):
        fitnessScores.append([population[i],routeFitness(population[i])])
    def takeSecond(element):
        return element[1]
    fitnessScores.sort(key = takeSecond, reverse=True)
    return fitnessScores

def matingPool(topparents,fitscore):    ### GENERATE THE MATING POOL ###
    matingpool = []
    for i in range(topparents):
        matingpool.append(fitscore[i][0])
        if "ROS" in matingpool[i]:
            matingpool[i].remove("ROS")
            matingpool[i].remove("ROS")
    return matingpool

def crossover(route1, route2):          ### BREEDING : CROSSOVER AND MUTATION ###
    chr1 = []
    geneA = int(random.random() * len(route1))
    geneB = int(random.random() * len(route2))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        chr1.append(route1[i])
    chr2 = [item for item in route2 if item not in chr1]
    child = chr1 + chr2
    child.insert(0, "ROS")
    child.append("ROS")

    return child

def mutation(child, mutation_rate):
    if "ROS" in child:
        child.remove("ROS")
        child.remove("ROS")
    for gene in range(len(child)):
        if random.random() < mutation_rate:
            newgene = int(random.random()*len(child))
            cityA = child[gene]
            cityB = child[newgene]
            child[gene] = cityB
            child[newgene]= cityA
    child.insert(0,"ROS")
    child.append("ROS")
    return child

def mutatepopulation(population, mutation_rate):
    mutated_population = []
    for i in range(len(population)):
        mutated_population.append(mutation(population[i],mutation_rate))
    return mutated_population

### BREEDING IN POPULATION ###
def breed(population, topparents,matingpool):
    children = []
    length = len(population)-topparents
    for i in range(topparents):
        entry = matingPool(topparents,getFittest(population))[i]
        children.append(entry)
    for i in range((length)):
        children.append(crossover(matingpool[0],matingpool[1]))
    children[0].append("ROS")
    children[0].insert(0,"ROS")
    children[1].append("ROS")
    children[1].insert(0, "ROS")
    return children

### GENERATING A NEW POPULATION ###
def nextGeneration(current_gen,mutation_rate,topparents):
    fitscore = getFittest(current_gen)
    matingpool = matingPool(topparents,fitscore)
    children = breed(current_gen,topparents,matingpool)
    next_generation = mutatepopulation(children,mutation_rate)
    return next_generation

def geneticAlgorithm(cities, popsize, iterations, mutation_rate, topparents):
    population = getPopulation(popsize,cities)
    print("Initial distance: " + str(round(1/getFittest(population)[0][1],2)))
    for i in range(iterations):
        population = nextGeneration(population,mutation_rate,topparents)
    print("Final distance: " + str(round(1/getFittest(population)[0][1],2)))
    best_route = getFittest(population)[0][0]
    print("Best route: " + str(best_route) )
    return best_route

def geneticPlot(cities, popsize, iterations, mutation_rate, topparents):
    population = getPopulation(popsize, cities)
    distance_hist = []
    distance_hist.append(1/getFittest(population)[0][1])
    for i in range(iterations):
        population = nextGeneration(population,mutation_rate,topparents)
        distance_hist.append(1 / getFittest(population)[0][1])
    plt.plot(distance_hist)
    plt.show()

algorithm = geneticAlgorithm(cities,popsize,iterations,mutation_rate,topparents)
plot = geneticPlot(cities,popsize,iterations,mutation_rate,topparents)





boundingbox = (2.373,7.273,49.411,53.697)
background = plt.imread("/Users/witekbieganski/PycharmProjects/TSP/Data/map.png")

x = []
y = []
for i in algorithm:
    index = int(np.where(npdata == i)[0])
    x.append(npdata[index][2])
    y.append(npdata[index][1])
plt.plot(x[0],y[0], '+r', markersize = 12)
plt.plot(x,y,'--o',zorder = 1)
plt.imshow(background, zorder = 0, extent=boundingbox)
aspect=background.shape[0]/float(background.shape[1])*((boundingbox[1]-boundingbox[0])/(boundingbox[3]-boundingbox[2]))
plt.gca().set_aspect(aspect)

plt.show()
