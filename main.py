import random
import numpy as np
import csv
from colorama import Fore
import os
import matplotlib.pyplot as plt
#####################################  read from csv file  #########################
myFile = open('items.csv', 'r')
reader = csv.reader(myFile)
flag_List = [] 
for record in reader:
    flag_List.append(record)

weight=[float(i) for i in flag_List[0][1:] ]
price=[float(i) for i in flag_List[1][1:] ]

items = [(x, y) for x, y in zip(weight, price)] 

####################### create random first population ############################
# jamiat avalie ra besoorat tamaman random ijad mikonim 
#sepas dar matris ijad shode baraye har choromosome fitness ra hesab mikonim
#va yek tuple be in shekl ijad mikonim (343,[01000111011100])
def first_population(popsize):
    first_population_init=np.random.randint(2,size=(popsize,len(weight)))
    first_population=[]
    for i in first_population_init:
        first_population.append((fitness(i),i))
    return first_population
    
####################################  fitness #####################################

def fitness(chromosom):
    
    weight = sum(items[i][0] for i in range(len(chromosom)) if chromosom[i] == 1)
    if weight > max_capacity:
        return 0
    value = sum(items[i][1] for i in range(len(chromosom)) if chromosom[i] == 1)
    return value

####################################################################################
#                                   selection
####################################################################################

def select_selection_Method(population):
    if selection=="Roulette Wheel Selection":
        parent=Roulette_wheel(population)

    elif selection=="Stochastic Universal Sampling":
        parent=Stochastic_Universal_Sampling(population,k_number)

    elif selection=="Tournament_Selection":
        parent=Tournament_Selection(population,T_size)

    elif selection=="Rank Selection":
        parent=Rank_Selection(population,S_number)
    else:
        parent=Random_Selection(population)

    return parent

#_______________________________________________selection methode____________________
      
def Random_Selection(population):
    return random.choice(population)

def Rank_Selection(population,s):
    #formol mohasebe ehtemal = 2-s/mou + 2i(s-1)/mou*(mou-1)
    population = sorted(population, key=lambda x:x[0])   
    ranks = list(range(1, len(population) + 1))
    probeblity=[((2-s)/max(ranks))+((2*(i-1)*(s-1))/(max(ranks)*(max(ranks)-1)))for i in ranks]
     
    return random.choices(population, weights=probeblity)[0]
#_____________________________________________________________________________________

def Tournament_Selection(population,tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x[0])
#_____________________________________________________________________________________
                                                                                                                               
def Roulette_wheel(population):
    total_fitness=sum(population[i][0]for i in range(len(population)))
    probility_list=[]
    for i in population:
        probility_list.append(i[0]/total_fitness)
    
    return random.choices(population, weights=probility_list)[0]

def Stochastic_Universal_Sampling(population,k):
    total_fitness=sum(population[i][0]for i in range(len(population)))
    distance=total_fitness/k
    start = random.uniform(0, distance)
    pointers = [start + i*distance for i in range(k)]
    selected = []
    j = 0
    for pointer in pointers:
        while j < len(population) and pointer > population[j][0]:
            pointer -= population[j][0]
            j += 1
        if j < len(population):
            selected.append(population[j])
       
    return random.choice(selected)
    
######################################################################################
#                                   Crossover
######################################################################################
def select_crossover_Method(parent1,parent2):
    if crossover=="One-Point Crossover":
        child=one_point(parent1,parent2)
    elif  crossover=="N-Point Crossover":
        child=n_point(parent1,parent2,N_number)
    else:
        child=uniform_crossover(parent1,parent2)
    return child

#______________________________________crossover Methode ______________
def one_point(parent1, parent2):
    crossover_point=random.randint(1,16)
    child=np.append(parent1[1][:crossover_point],parent2[1][crossover_point:])
    return child

def n_point(parent1, parent2, n):
    
    points = sorted(random.sample(range(1, len(parent1[1])), n))
    points.insert(0, 0)
    
    points.append(len(parent1[1]))
    child = []
    for i in range(n+1):
        if i % 2 == 0:
            child.append(parent1[1][points[i]:points[i+1]])
        else:
            child.append(parent2[1][points[i]:points[i+1]])
    return np.concatenate(child)

def uniform_crossover(parent1,parent2):
    newgen1=[]
    for i in range(len(parent1)):
        if random.random() >=0.5:
            newgen1.append(parent1[i])
        else:
            newgen1.append(parent2[i])
    return newgen1[1]

############################################################################
#                                Mutation
############################################################################
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 if chromosome[i] == 0 else 0
    return chromosome

############################################################################
#                               elitism     
############################################################################
def elitism(population,elite_percent):
    elite_size=int(len(population)*elite_percent)
    population = sorted(population, key=lambda x:x[0], reverse=True)
    elite = population[:elite_size]
    return elite





def main():
    os.system("cls")

    print(f""" 
    |{Fore.BLUE}#######################################################{Fore.WHITE}|
    |                                                       |
    |                                                       |
    |    {Fore.GREEN}Knapsack Problem using Genetic algorithm:{Fore.WHITE}          |
    |                                                       |
    |          P:[play]                 A:[About]           |
    |                      {Fore.RED}E:[Exit]{Fore.WHITE}                         |
    |{Fore.BLUE}#######################################################{Fore.WHITE}|
    """) 
    flag=input(f"{Fore.BLUE}:{Fore.WHITE}")
    if flag=="p":
        os.system("cls")
        print(f""" 
        |{Fore.BLUE}##################################{Fore.WHITE}|
        |  Enter Max capacity of Backpack: | 
        |                                  |
        |{Fore.BLUE}##################################{Fore.WHITE}|
        """)
        global max_capacity
        max_capacity=int(input(f"{Fore.GREEN}MAX capacity{Fore.WHITE}:"))
        os.system("cls")
        print(f"""
        |##############################|
        |           {Fore.RED}Elitism {Fore.WHITE}           |
        |                              |
        |    y:[Yes]         N:[No]    |
        |##############################|
        """)
        Elitism_flag=input(":")
        if Elitism_flag=="y":
            Elitism=True
            Elitepercent=float(input("darsad chromosome haye elite ra vared namaeed:"))
        else:
            Elitism=False
        os.system('cls')
        print(f""" 
        |{Fore.BLUE}#############################################{Fore.WHITE}|
        |  Please select crossover Methode :          |
        |                                             |
        |    O:[One-Point Crossover]                  |
        |    N:[N-Point Crossover]                    |
        |    U:[Uniform Crossover]                    |
        |                                             |
        |{Fore.BLUE}#############################################{Fore.WHITE}|
        """)
        crossover_flag=input("Crossover Methode:")
        global crossover

        if crossover_flag=="o":
            crossover="One-Point Crossover"
        elif crossover_flag=="n":
            global N_number
            N_number=int(input("tedad point ha ra vared namaeed:"))
            crossover="N-Point Crossover"
        else:
            crossover= "Uniform Crossover"
        global mutation_rate
        mutation_rate=float(input(f"Please enter {Fore.GREEN}Mutation rate{Fore.WHITE}:"))
        os.system("cls")



        print(f""" 
        |{Fore.BLUE}############################################{Fore.WHITE}|
        |                                            |
        |    please select your Selection Methode:   |
        |                                            |
        |    R:[Roulette Wheel Selection]            |
        |    S:[Stochastic Universal Sampling]       |
        |    T:[Tournament_Selection]                |
        |    RA:[Rank Selection]                     |
        |    RO:[Random Selection]                   |
        |                                            |
        |{Fore.BLUE}############################################{Fore.WHITE}|
        """)
        global selection , k_number,T_size , S_number
        selection_flag=input(":")
        if selection_flag=="r":
            selection="Roulette Wheel Selection"
        elif selection_flag == "s":
            k_number=int(input("k ra vared namaeed :"))
            selection="Stochastic Universal Sampling"
        elif selection_flag=="t":
            T_size=int(input("tournament size ra vared namaeed:"))
            selection="Tournament_Selection"
        elif selection_flag=="ra":
            S_number=float(input("zarib s ra vared namaeed:"))
            selection="Rank Selection"
        else:
            selection="Random Selection"

        os.system("cls")        


        popsize=int(input("please Enter population size:"))
        population=first_population(popsize)
        os.system("cls")
        print(f"""
        {Fore.GREEN}Sharayet Masale{Fore.WHITE}:
        Elitism:{Fore.BLUE}{Elitism}{Fore.WHITE}
        {("Elite percent: ",Elitepercent) if Elitism else None}
        selection_Methode:{Fore.BLUE}{selection}{Fore.WHITE}
        Crossover Methode:{Fore.BLUE}{crossover}{Fore.WHITE}
        {("numbr of point :",N_number) if crossover=="N-Point Crossover" else None }
        MAX Capacity:{Fore.BLUE}{max_capacity}{Fore.WHITE}
        Mutation rate:{Fore.BLUE}{mutation_rate}{Fore.WHITE}
        population size:{Fore.BLUE}{popsize}{Fore.WHITE}
        """)
       
        input()
        print("""
        |########################################|
        |    do you want to continue?            |
        |    y:[yes]          b:[main]           |
        |########################################|
        """)
        flag2=input(':')
        if flag2=='y':
            max_fitness_list=[]
            avg_fitness_list = []
            max_num_generation = 100
            for gen in range(max_num_generation):
                population = sorted(population, key=lambda x:x[0], reverse=True)
                if Elitism:
                    population=elitism(population,Elitepercent)
                    while len(population) < popsize:
                        parent1 = select_selection_Method(population)
                        parent2 = select_selection_Method(population)
                        child = select_crossover_Method(parent1, parent2)
                        child = mutation(child)
                        population.append((fitness(child),child))
                    avg_fitness=sum(population[i][0]for i in range(len(population)))/len(population)
                    max_fitness=max(population, key=lambda x: x[0])[0]
                    max_fitness_list.append(max_fitness)
                    avg_fitness_list.append(avg_fitness)

                else:
                    new_population = []
                    while len(new_population) < popsize:
                        parent1 = select_selection_Method(population)
                        parent2 = select_selection_Method(population)
                        child = select_crossover_Method(parent1, parent2)
                        child = mutation(child)
                        new_population.append((fitness(child),child))
                    population = new_population
                    avg_fitness=sum(population[i][0]for i in range(len(population)))/len(population)
                    max_fitness=max(population, key=lambda x: x[0])[0]
                    max_fitness_list.append(max_fitness)
                    avg_fitness_list.append(avg_fitness)
                    population = sorted(population, key=lambda x:x[0], reverse=True)

                    
            plt.subplot(1, 2, 1)
            plt.plot(avg_fitness_list)
            plt.title("Average Fitness OF Population")
            plt.xlabel('Generation')
            plt.ylabel('Average Fitness')
            plt.subplot(1, 2, 2)
            plt.plot(max_fitness_list)
            plt.title("Maximum Fitness OF Population")
            plt.xlabel('Generation')
            plt.ylabel('Max Fitness')
            plt.show()        
            best_chromosome = population[0]
            print(f"Best solution:{Fore.GREEN} {best_chromosome[1]} {Fore.WHITE} Value: {best_chromosome[0]}" )
            print(f"""
            |#########################|
            |                         |
            | B:[main]       E:[exit] |
            |#########################|
            """)
            flag3=input(":")
            if flag3=="b":
                population.clear()
                main()
            else:
                exit()
        else:
            main()


    elif flag=="a":
        os.system("cls")

        print(f"""
        |{Fore.BLUE}##################################{Fore.WHITE}|
        |                                  |
        |  Arian Akhshabi                  |
        |  STID: {Fore.GREEN}990201110009{Fore.WHITE}              |
        |                                  |
        |{Fore.BLUE}##################################{Fore.WHITE}|        
        
        """)
    elif flag=="e":
        os.system("cls")
        exit()
main()