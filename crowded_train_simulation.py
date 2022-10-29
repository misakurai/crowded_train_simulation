



#値設定


#シミュレーションを繰り返す回数
times = 1

#個体数と世代数
size_pop = 50

num_generations = 100

#進捗報告をする頻度(世代数)
period = 10

#交叉と突然変異の起きる確率
probab_clossing = 0.7
probab_mutating = 0.5

#最良の個体の保存される確率
best_surv= 1




#一つの駅で乗ってくる客数を決定
num_passengers = 30

#電車の車両編成
len_of_train = 6

#駅の階段の位置を指定
stairs = (2,6,4,1,4,5,3,6)
num_station = len(stairs)



#乗客の行き先情報を与えるか、自動生成するか決定
give_passengers = False


#乗客の行き先情報を与える場合のデータ
#各駅で乗車する客の目的駅を指定
#制作当時、各駅の乗客数を一定値にしていたため、終点に近い駅ほど降車客が多くなってしまっていた。
all_passengers = [[4, 2, 7, 4, 6, 3, 6, 7, 1, 3, 2, 2, 4, 3, 1,
                   1, 7, 5, 7, 3, 2, 7, 4, 3, 3, 1, 1, 7, 2, 1],
                  [5, 6, 7, 7, 4, 6, 3, 5, 4, 2, 3, 5, 7, 2, 7,
                   5, 4, 7, 5, 3, 6, 7, 4, 6, 2, 2, 5, 2, 5, 7],
                  [5, 4, 4, 7, 6, 3, 6, 6, 3, 4, 7, 3, 7, 7, 3,
                   7, 3, 5, 5, 3, 4, 4, 6, 6, 5, 6, 3, 6, 7, 6],
                  [7, 7, 7, 5, 4, 5, 7, 5, 4, 7, 4, 5, 6, 4, 6,
                   5, 4, 6, 7, 6, 5, 7, 6, 5, 5, 5, 4, 7, 4, 6],
                  [5, 6, 7, 7, 5, 7, 5, 6, 6, 5, 7, 5, 6, 5, 7,
                   6, 6, 5, 5, 6, 5, 7, 6, 6, 7, 5, 5, 7, 7, 7],
                  [6, 7, 6, 7, 7, 6, 6, 6, 7, 7, 6, 6, 7, 7, 6,
                   7, 7, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 7, 6, 7],
                  [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                  []]


#ファイルインポート


#グラフ描画用ファイル
import numpy as np
import matplotlib.pyplot as plt

#乱数と進化的アルゴリズム実装用ファイル
import random
from deap import base, creator, tools




#関数の定義


#客の行き先情報を格納するリストを生成
def package():
    li = [[],[0]*len_of_train]
    return(li)

#乗客の行き先決定に使用
def pas_destination():
    x = random.randint(k+1,num_station-1)
    return(x)

#個体の作成と、乗客乗り降りのシミュレーション
def make_individual():
    #個体の型の作成
    creator.Individual = toolbox.make_package_of_Individual(n=num_station)
    #乗客の乗車車両を決定
    for now_station in range(0,num_station-1):
        for person in range(0,len(all_passengers[now_station])):
            destination = all_passengers[now_station][person] 
            if stairs[now_station] > stairs[destination]:
                x = random.randint(stairs[destination]-1,stairs[now_station]-1)
            elif stairs[now_station] == stairs[destination]:
                x = stairs[now_station]-1
            else:
                x = random.randint(stairs[now_station]-1,stairs[destination]-1)
            creator.Individual[now_station][0].append(x)
            for n in range(now_station,destination):
                creator.Individual[n][1][x] += 1
    return(creator.Individual)


#乗客の乗り降りをシミュレーション
#交叉や突然変異後の個体の再評価時に使用
def get_on(indiv):
    for i in range(0,num_station-1):
        indiv[i][1] = [0]*len_of_train
    for now_station in range(0,num_station-1):
        for person in range(0,len(all_passengers[now_station])):
            destination = all_passengers[now_station][person] 
            if stairs[now_station] > stairs[destination]:
                x = random.randint(stairs[destination]-1,stairs[now_station]-1)
            elif stairs[now_station] == stairs[destination]:
                x = stairs[now_station]-1
            else:
                x = random.randint(stairs[now_station]-1,stairs[destination]-1)
            for n in range(now_station,destination):
                indiv[n][1][x] += 1

#世代の生成
def make_population(size):
    pop = []
    for m in range(size):
        pop.append(make_individual())
    return(pop)


#個体の評価時に使用する平均値の二乗を計算
def caluculate_average_two_square(indi):
    av_tw_sq = 0
    for i in indi:
        av_tw_sq += (sum(i[1])/(len_of_train))
    av_tw_sq = av_tw_sq/num_station
    av_tw_sq = av_tw_sq**2
    return(av_tw_sq)


#目的関数　個体の評価に使用
def eval_func_one(individual):
    mean_square_sum = 0
    
    for i in individual:
        for k in i[1]:
            mean_square_sum += k**2
        ave = sum(i[1])/6
        
    mean_square_ave = mean_square_sum/(len_of_train*num_station)
    distributed = mean_square_ave - average_two_square
    
    #(全ての駅間での車両ごとの人数の分散,)
    return(distributed,)

    


#最良の値の発見に使用
def find_best_ind(p):
    best_ind = tools.selBest(p, 1)[0]
    if not bool(find_best):
        return(best_ind)
    
    if best_ind.fitness.values[0] > find_best[-1].fitness.values[0]:
        best_ind = find_best[-1]
    return(best_ind)




#必要なクラスと関数を作成
creator.create("FitnessMax",base.Fitness, weights =(-1.0,)) #適応値の定義
creator.create("Individual",list,fitness=creator.FitnessMax,)   #個体の定義

toolbox = base.Toolbox()

toolbox.register("passengers",tools.initRepeat, list)   #客情報リスト制作用関数

#個体の型を作成する関数
toolbox.register("package",package) 
toolbox.register("make_package_of_Individual",
                 tools.initRepeat,creator.Individual,toolbox.package)

#評価関数
toolbox.register("evaluate",eval_func_one)

#交叉方法：2点交叉
toolbox.register("mate",tools.cxTwoPoint)

#突然変異:突然変異が起きた時、遺伝子のうち変異する箇所5%
toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)

#選択　トーナメントサイズは3
toolbox.register("select",tools.selTournament, tournsize=3)




#以下実行



#決定したデータを出力
print("個体数:",size_pop,"\t世代数:",num_generations)
print("交叉:",probab_clossing,"\t突然変異:",probab_mutating)
print("最良の個体の生き残る確率:",best_surv)


#乗客の行き先情報リストの作成
if give_passengers == False:
    all_passengers = []
    for k in range(num_station-1):
        p = toolbox.passengers(n=num_passengers,func=pas_destination)
        all_passengers.append(p)
    all_passengers.append([])



print("\nThe Stage has prepared")
print("Start of evolution\n")

for _ in  range(times):

    distributed = []
    find_best = []

    population = make_population(size_pop)

    average_two_square = caluculate_average_two_square(population[0])

    
    fitnesses = list(map(toolbox.evaluate, population))

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    first_rand_ind = population[0]

    for g in range(num_generations):

        #最良の個体を見つける
        best_ind = find_best_ind(population)
        find_best.append(best_ind)
    
        distributed.append(best_ind.fitness.values[0])
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone,offspring))


        #最良の個体を残すか否か決定
        start_p = 0
        start_m = 0
        if random.random() < best_surv:
            offspring[0] = best_ind
            start_p = 2
            start_m = 1
        
        for child1, child2 in zip(offspring[start_p::2],offspring[start_p+1::2]):

            #交叉
            if random.random() < probab_clossing:
                update = False

                for n in range(len(child1)):
                    toolbox.mate(child1[n][1],child2[n][1])
                    #chil1とchild2で2点交差をする
                del child1.fitness.values
                del child2.fitness.values
                #適応度をリセット
                get_on(child1)
                get_on(child2)
                    
        

        for mutant in offspring[start_m:]:
            #突然変異
            if random.random() < probab_mutating :
                st=random.randint(0,len(mutant)-2)
                person = random.randint(0,len(all_passengers[st])-1)
                destination = all_passengers[st][person]
            
                if stairs[st] > stairs[destination]:
                    x = random.randint(stairs[destination],stairs[st])
                elif stairs[st] == stairs[destination]:
                    x = stairs[st]
                else:
                    x = random.randint(stairs[st],stairs[destination])
                mutant[st][0][person]=x
    
                del mutant.fitness.values
                #適応度リセット
                get_on(child1)
                get_on(child2)

        
        #適応度を再計算
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))

        fit_sum = 0
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if (g+1)%period==0:
                fit_sum += fit[0]

        #進捗報告
        if (g+1)%period==0:
            fit_average = fit_sum/size_pop
            print("Generation",g+1,"\nBest\t",distributed[-1],"\nAverage\t",fit_average,"\n")


        #次世代の作成
        population[:] = offspring
            
        fits = [ind.fitness.values[0] for ind in population]
     
        random.shuffle(population)



    print("\n==== End of evolution")
    #最初の状態と最良の結果を表示
    best_num = min(distributed)
    best_num = distributed.index(best_num)
    best_ind = find_best[best_num]
    
    print("最初の個体(無作為抽出):")
    for n in first_rand_ind:
        print(n[1])
    print("不適応度:",first_rand_ind.fitness.values[0])
    print()
    print("最良の個体:")
    verification = 0
    sumsum = 0
    for n in best_ind:
        for i in n[1]:
            print(i,end =",")
            verification += i*i
            sumsum += i
        print()
    verification = verification/(len_of_train*num_station)
    sumsum = sumsum/(len_of_train*num_station)
    sumsum = sumsum*sumsum
    print("不適応度:",best_ind.fitness.values[0])




    


#折れ線グラフを出力

left = list(range(1,num_generations+1))
left = np.array(left)
height = np.array(distributed)
plt.plot(left, height)
plt.show()

