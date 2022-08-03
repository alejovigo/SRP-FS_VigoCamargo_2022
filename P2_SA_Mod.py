import numpy
import pandas
import math
import time
from random import choice





def OnB_Pol(N, T, quantities,demand, CS, CT_Single, MVisits, Io, rate):
    # --------Calculate preliminary truck bounds for each period and seleect peak period-------------

    lbs_period = []
    del_period = []
    tbound_period = []
    H_bound = 0

    for t in range(T):
        lbs_period.append(sum(quantities[t]))
        occurence_period = quantities[t] > .01
        del_period.append(occurence_period.sum())
        tbound_period.append(math.ceil(max(lbs_period[t] / CT_Single, del_period[t] / MVisits)))
        if tbound_period[t] >= H_bound:
            H_bound = tbound_period[t]
            peak_period = t

    # --------Calculate Truck Lower Bound--------

    cumu_lbs = 0
    cumu_del = 0

    for t in range(0, peak_period + 1):
        cumu_lbs = cumu_lbs + lbs_period[t]
        cumu_del = cumu_del + del_period[t]

    lbs_bound = (cumu_lbs / CT_Single) / (peak_period + 1)
    del_bound = (cumu_del / MVisits) / (peak_period + 1)

    truck_bound = math.ceil(max(lbs_bound, del_bound))

    # -------------------------------------------

    # -------Check which deliveries are FIXED and adjust truck lower bound

    Initial_Pass = numpy.zeros((N, T), dtype=int)
    potential = numpy.zeros((N, T), dtype=int)
    AdjDemand = numpy.zeros((N, T), dtype=int)

    for t in range(T):
        if t == 0:  # Deliveries scheduled on the first period are FIXED
            for n in range(N):
                potential[n, t] = min(CS, CS - Io.iloc[
                    n][0] + rate.iloc[n]
                                      [t])  # Determines the available space at the store considering initial inventory
                AdjDemand[n, t] = max(0, demand.iloc[n]
                [t] - 0)  # Io[n])  # Adjusts the demand for the store considering initial inventory
                if quantities.iloc[n][t] > .01:
                    Initial_Pass[n, t] = 1  # Fixes the delivery

        else:
            for n in range(N):  # Fix deliveries that have a fixed delivery on the previous period
                potential[n, t] = CS - max(0, Io.iloc[n][0] - sum(rate.iloc[n][j] for j in range(0
                                                                                                 ,
                                                                                                 t)))  # sum(demand.iloc[n][j] for j in range(0, t)))  # Determine the available space at the store
                AdjDemand[n, t] = max(0, demand.iloc[n]
                [
                    t])  # - (max(0, Io[n] - sum(demand.iloc[n][j] for j in range(0, t)))))  # Adjust the demand for the store considering initial inventory
                if Initial_Pass[n, t - 1] == 1 and quantities.iloc[n][
                    t] > .01:  # If there's a fixed delivery in the previous period and a delivery in period t, FIX delivery on period t
                    Initial_Pass[n, t] = 1
                else:
                    if quantities.iloc[n][t] > .01:
                        if potential[n, t - 1] < AdjDemand[
                            n, t]:  # Fix deliveries where the Empty Space on previous turn is not enough to cover demand
                            Initial_Pass[n, t] = 1

        for t in range(T):  # Check if the Max number of fixed deliveries on a period is higher than the Avg-Based LB
            LB2_del = math.ceil(sum(Initial_Pass[n, t] for n in range(N)) / MVisits)
            LB2_lbs = math.ceil(sum(quantities.iloc[n][t] for n in range(N) if Initial_Pass[n, t] == 1) / CT_Single)

            LB2 = max(LB2_del, LB2_lbs)
            if LB2 >= truck_bound:
                if t >= peak_period:
                    truck_bound = LB2
                    peak_period = t

    # print("Initial Pass")
    # print("Truck Bound: ", truck_bound)

    #for n in range(N):
        #print(Initial_Pass[n, :])
    # --------------------------------------------------

    # --------Build Schedule Solution-------------------
    # Create arrays of zero to store the final schedule and delivery plan
    I = numpy.zeros((N, T), dtype=float)  # Store inventory
    Y = numpy.zeros((N, T), dtype=float)  # delivery quantities
    X = numpy.zeros((N, T), dtype=int)  # delivery indicator

    # If peak period is initial period, then no changes to schedule
    if peak_period == 0:
        for n in range(N):
            for t in range(T):
                if quantities.iloc[n][t] >= .01:
                    X[n, t] = 1
                    Y[n, t] = quantities.iloc[n][t]
    else:

        # Construct Schedule
        lastDel = 0
        for t in range(T):
            for n in range(N):
                # check when is the next delivery needed scheduled
                if AdjDemand[n, t] > 0.01:  # quantities.iloc[n][t]>.01:
                    if t == T - 1:  # if its the last period, then no future deliveries are scheduled
                        NextDel = T
                    for j in range(t + 1, T):  # Check future periods
                        if AdjDemand[n, t] > 0.01:  # if quantities.iloc[n][j] >.01:
                            NextDel = j
                            break
                        if j == T - 1:
                            NextDel = T
                            break
                    for j in range(T - 1, -1, -1):  # Save when the last delivery has been scheduled
                        if X[n, j] == 1:
                            lastDel = j
                            break
                        else:
                            lastDel = 0

                    select_period = lastDel  # Select the period on which to start scheduling/fixing deliveries

                    if Initial_Pass[n, t] == 1:  # FIXED deliveries stay the same for the final solution
                        X[n, t] = 1
                        Y[n, t] = quantities.iloc[n][t]

                        # Adjust the empty space calculation at the store for future deliveries
                        potential[n, t] = 0
                        AdjDemand[n, t] = max(0, AdjDemand[n, t] - Y[n, t])

                        for j in range(t + 1, T):
                            potential[n, j] = CS - max(0, Io.iloc[n][0] + sum(Y[n, m] for m in range(0, t + 1)) - sum
                            (rate.iloc[n][m] for m in range(0, j)))
                            # sum(demand.iloc[n][m] for m in range(0, j)))
                            AdjDemand[n, j] = max(0, demand.iloc[n][j] - (max(0, sum
                            (Y[n, m] for m in range(0, t + 1)) - sum(demand.iloc[n][m] for m in range(0, j)))))

                    else:
                        while select_period < t + 1:
                            if select_period == t:

                                X[n, select_period] = 1

                                # lbs_limit calculates the amount of lbs available before having to increase the truck fleet size
                                lbs_limit = truck_bound * CT_Single - sum(
                                    max(Y[m, select_period], AdjDemand[m, select_period]) for m in range(N))
                                Y_limit = min(CT_Single, potential[n, select_period], lbs_limit,
                                              sum(AdjDemand[n, m] for m in range(select_period, T)))

                                Y[n, select_period] = max(AdjDemand[n, t], Y_limit)
                                # Y[n,select_period] = AdjDemand[n,t] #quantities.iloc[n][t]
                                potential[n, select_period] = 0
                                AdjDemand[n, select_period] = max(0, AdjDemand[n, select_period] - Y[n, select_period])

                                for j in range(t + 1, T):
                                    potential[n, j] = CS - max(0, Io.iloc[n][0] + sum
                                    (Y[n, m] for m in range(0, t + 1)) - sum(rate.iloc[n][m] for m in range(0, j)))
                                    # sum(demand.iloc[n][m] for m in range(0, j)))
                                    AdjDemand[n, j] = max(0, demand.iloc[n][j] - (max(0, sum
                                    (Y[n, m] for m in range(0, t + 1)) - sum
                                                                                      (demand.iloc[n][m] for m in
                                                                                       range(0, j)))))

                                select_period = T
                                lastDel = 0

                            else:
                                # Check that adding a delivery won't exceed the Upper Bound
                                tcheck_del = math.ceil((sum(X[m, select_period] for m in range(N)) + 1) / MVisits)
                                tcheck_lbs = math.ceil((sum(Y[m, select_period] for m in range(N)) + AdjDemand[
                                    n, t]) / CT_Single)  # quantities.iloc[n][t]
                                tcheck = max(tcheck_del, tcheck_lbs)

                                if tcheck <= truck_bound:
                                    if X[n, select_period] == 0:
                                        # Is there enough space at the store on select_period to cover Adj Demand up to the next scheduled delivery?
                                        if potential[n, select_period] >= sum(
                                                AdjDemand[n, j] for j in range(select_period, NextDel)):
                                            if potential[n, select_period] >= AdjDemand[n, j]:  # quantities.iloc[n][t]:
                                                # print("changed delivery: ", n, ', ', select_period)
                                                X[n, select_period] = 1

                                                lbs_limit = truck_bound * CT_Single - sum(
                                                    max(Y[m, select_period], AdjDemand[m, select_period]) for m in
                                                    range(N))
                                                Y_limit = min(CT_Single, potential[n, select_period], lbs_limit,
                                                              sum(AdjDemand[n, m] for m in range(select_period, T)))

                                                Y[n, select_period] = max(AdjDemand[n, t], Y_limit)

                                                potential[n, select_period] = 0
                                                AdjDemand[n, select_period] = max(0, AdjDemand[n, select_period] - Y[
                                                    n, select_period])

                                                # Adjust the rest of the data structure

                                                for j in range(select_period + 1, T):
                                                    potential[n, j] = CS - max(0, Io.iloc[n][0] + sum(
                                                        Y[n, m] for m in range(0, select_period + 1)) - sum
                                                                               (rate.iloc[n][m] for m in range(0, j)))
                                                    # sum(demand.iloc[n][m] for m in range(0, j)))
                                                    AdjDemand[n, j] = max(0, demand.iloc[n][j] - (max(0, sum(
                                                        Y[n, m] for m in range(0, select_period + 1)) - sum(
                                                        demand.iloc[n][m] for m in range(0, j)))))

                                                select_period = T
                                                lastDel = 0
                                            else:
                                                select_period = select_period + 1
                                        else:
                                            select_period = select_period + 1
                                    else:
                                        select_period = select_period + 1
                                else:
                                    select_period = select_period + 1

    # Calculate Inventory for the solution
    for n in range(N):
        for t in range(T):
            #print(n,t)
            if t == 0:
                I[n, t] = max(0, Io.iloc[n][0] + Y[n, t] - rate.iloc[n][t])  # demand.iloc[n][t]
            else:
                I[n, t] = max(0, I[n, t - 1] + Y[n, t] - rate.iloc[n][t])  # demand.iloc[n][t]

    # Calculate the truck lower bound for the constructed schedule

    lbs_period2 = []
    del_period2 = []
    tbound_period2 = []
    H_bound2 = 0

    for t in range(T):
        lbs_period2.append(sum(Y[:, t]))
        del_period2.append(sum(X[:, t]))
        tbound_period2.append(math.ceil(max(lbs_period2[t] / CT_Single, del_period2[t] / MVisits)))
        if tbound_period2[t] >= H_bound2:
            H_bound2 = tbound_period2[t]
            peak_period2 = t

    # print("original deliveries: ", tbound_period)
    # print("initial solution: ", tbound_period2)

    # -----------------------------------------------------------------------------------------------------------------------
    #print("Y:", Y)
    #print("X: ", X)
    #print("I: ", I)
    return Y, X, I


def Savings_Algo(N, T, Y, X, Store_IDs, dist, MVisits, CT_Single,overage,single_miles, single_lbs, team_miles, team_lbs):
    # This algorithm needs to be run for each period to build initial routes based on the new schedule

    Routes_t = []
    R_lbs_t = []
    R_Dist_t = []
    team_ind_t = []
    R_Cost_t = []

    for t in range(T):

        # print(t)

        # ---------Calculate Savings---------

        Stores_t = []  # save stores that are visited on period t
        for n in range(N):
            if X[n, t] == 1:
                Stores_t.append(Store_IDs.iloc[n][0])

        S = numpy.zeros((len(Stores_t), len(Stores_t)), dtype=float)

        for i in range(len(Stores_t)):
            for j in range(len(Stores_t)):
                if i != j:
                    S[i, j] = dist.loc[Stores_t[i]][1000] + dist.loc[1000][Stores_t[j]] - dist.loc[Stores_t[i]][
                        Stores_t[j]]

        # -------Order the Savings

        # Crates list of the row, column indexes of S in ascending order
        r, c = numpy.unravel_index(numpy.argsort(S, axis=None), S.shape)

        # Flip the lists to descending order
        r = r[::-1]
        c = c[::-1]

        # -----Build routes
        Covered = []
        Routes = []
        R_lbs = []
        R_Dist = []
        team_ind = []
        R_Cost = []

        while len(Covered) < len(Stores_t):
            R_counter = 0
            Capm = 0
            Distm = 0
            Timem = 0

            if len(Stores_t) - len(Covered) == 1:
                # print("last store standing")
                for n in Stores_t:
                    if n in Covered:
                        continue
                    else:
                        Rm = [n]
                        n_indx = pandas.Index(Store_IDs['Store ID']).get_loc(n)
                        Capm = Y[n_indx, t]
                        Distm = dist.loc[1000][n] + dist.loc[n][1000]
                        Timem = Capm*.022 + Distm*1.5 + len(Rm)*10
                        Covered.append(n)

            if len(Covered) < len(Stores_t):
                # Start a new route
                # print("New Route")
                Rm = []
                for i in range(len(r)):
                    v = Stores_t[r[i]]
                    w = Stores_t[c[i]]
                    v_indx = pandas.Index(Store_IDs['Store ID']).get_loc(v)
                    w_indx = pandas.Index(Store_IDs['Store ID']).get_loc(w)
                    if v in Covered or w in Covered:
                        continue
                    elif v == w:
                        continue
                    else:
                        Rm = [v, w]
                        Capm = Y[v_indx, t] + Y[w_indx, t]
                        Distm = dist.loc[1000][v] + dist.loc[v][w] + dist.loc[w][1000]
                        Timem = Capm * .022 + Distm * 1.5 + len(Rm) * 10
                        Covered.append(v)
                        Covered.append(w)
                        break

                # Add more stores to the route
                if Rm:
                    full = 0
                    while full == 0:
                        # print(Rm)
                        for i in range(len(r)):
                            if len(Rm) == MVisits:
                                full = 1
                                break

                            v = Stores_t[r[i]]
                            w = Stores_t[c[i]]
                            if v == w:
                                continue
                            elif w == Rm[0] or v == Rm[len(Rm) - 1]:
                                v_indx = pandas.Index(Store_IDs['Store ID']).get_loc(v)
                                w_indx = pandas.Index(Store_IDs['Store ID']).get_loc(w)
                                if w == Rm[0]:
                                    if v in Covered:
                                        continue
                                    elif Y[v_indx, t] + Capm <= CT_Single:  # _Single:
                                        Dist_Timecalc = 1.5 * (-dist.loc[1000][w] + dist.loc[1000][v] + dist.loc[v][w])
                                        if Timem + Dist_Timecalc + Y[v_indx, t] * .022 + 10 <= 2 * 810:
                                            Rm.insert(0, v)
                                            Capm = Capm + Y[v_indx, t]
                                            Distm = Distm - dist.loc[1000][w] + dist.loc[1000][v] + dist.loc[v][w]
                                            Timem = Capm * .022 + Distm * 1.5 + len(Rm) * 10
                                            Covered.append(v)
                                            break
                                        else:
                                            full = 1

                                    else:
                                        full = 1

                                elif v == Rm[len(Rm) - 1]:
                                    if w in Covered:
                                        continue
                                    elif Y[w_indx, t] + Capm <= CT_Single:  # _Single:
                                        Dist_Timecalc = 1.5 * (-dist.loc[1000][w] + dist.loc[1000][v] + dist.loc[v][w])
                                        if Timem + Dist_Timecalc + Y[w_indx, t] * .022 + 10 <= 2 * 810:
                                            Rm.append(w)
                                            Capm = Capm + Y[w_indx, t]
                                            Distm = Distm - dist.loc[v][1000] + dist.loc[v][w] + dist.loc[w][1000]
                                            Timem = Capm*.022 +Distm * 1.5 + len(Rm) * 10
                                            Covered.append(w)
                                            break
                                        else:
                                            full = 1

                                    else:
                                        full = 1

                            if len(Covered) == len(Stores_t):
                                full = 1
                                break

                            if full == 1:
                                break

            Routes.append(Rm)
            R_lbs.append(Capm)
            R_Dist.append(Distm)

            if Capm > CT_Single:
                l_team = 1
            else:
                l_team = 0
            team_ind.append(l_team)
            # ----Calculate Route Cost

            # calculate route time
            Rtime = Distm * 1.5 + len(Rm) * 10 + Capm * .022
            if l_team == 1:
                if Rtime > 1080:
                    Rcost = Distm * team_miles + Capm * team_lbs + overage
                else:
                    Rcost = Distm * team_miles + Capm * team_lbs
            else:
                if Rtime > 810:
                    Rcost = Distm * single_miles + Capm * single_lbs + overage
                else:
                    Rcost = Distm * single_miles  + Capm * single_lbs

            R_Cost.append(Rcost)
            R_counter = R_counter + 1

        Routes_t.append(Routes)
        R_lbs_t.append(R_lbs)
        R_Dist_t.append(R_Dist)
        team_ind_t.append(team_ind)
        R_Cost_t.append(R_Cost)

    return Routes_t, R_lbs_t, R_Dist_t, team_ind_t, R_Cost_t

def Rand_AltP(choice_periods,choice_stores, mvt,select_store,period_opts,x_list):

    selected_p2 = 0
    while selected_p2 == 0:
        # print("Choice Period 2: ", choice_periods)
        select_period = choice(choice_periods)

        if (mvt, select_store, select_period) in x_list:
            selected_p2 = 0
            choice_periods.remove(select_period)
            if not choice_periods:
                selected_p2 = 1
                selected = 0
                choice_stores.remove(select_store)
                if not choice_stores:
                    selected = 1
                    period_opts.remove(mvt)
                    selected_p = 0
                    if not period_opts:
                        selected_p = 1
                        restart = 1
                        # print("No Exchanges left to explore")
                        # restart the search from best solution found so far
                        x_list = []
                        # curr_sol = temp_sol.copy() #best_sol.copy()
                        # current_cost = temp_cost #best_cost
        else:

            selected = 1
            selected_p = 1
            selected_p2 = 1
            restart = 0
            x_list.append((mvt, select_store, select_period))
        # print("selected store ",select_store, select_period)

    return select_period, selected_p2, selected, selected_p, restart, choice_stores, period_opts

def Truck_Util(choice_periods,R_lbs_tp,team_ind_tp,CT,CT_Single):
    lowest = 100
    select_period = 100

    for j in choice_periods:

        if R_lbs_tp[j]:
            # Avg Truck Utilization
            #util_cal =sum(R_lbs_tp[j])/(CT_Single * len(R_lbs_tp))

            # Actual Truck Utilization (Averaged)
            #'''
            util_list = []
            for lbs in range(len(R_lbs_tp[j])):
                if team_ind_tp[j][lbs] == 1:
                    util_list.append(R_lbs_tp[j][lbs] / CT)
                else:
                    util_list.append(R_lbs_tp[j][lbs] / CT_Single)
            util_cal = sum(util_list) / len(util_list)
            #'''

            if util_cal <= lowest:
                lowest = util_cal
                select_period = j

    if select_period == 100: #If the only choice periods left have no deliveries, pick one at random
        select_period = choice(choice_periods)

    selected = 1
    selected_p = 1
    restart = 0

    return select_period, selected,selected_p, restart

def Store_Util(choice_periods, mvt,CS,I_p,store_lbs,demand,store_indx,rate):
    for j in choice_periods:
        hspace = 0
        if j < mvt:
            hspace_calc = CS - I_p[store_indx, j]

        elif j > mvt:
            add_lbs = max(0, store_lbs - sum(demand.iloc[store_indx][m] for m in range(mvt, j)))
            if add_lbs > I_p[store_indx, j]:
                hspace_calc = CS - (I_p[store_indx, j] - (
                    max(add_lbs - sum(rate.iloc[store_indx][m] for m in range(mvt, j)), 0)))
            else:
                hspace_calc = CS - I_p[store_indx, j] + add_lbs

        if hspace_calc >= hspace:
            select_period = j

    selected = 1
    selected_p = 1
    restart = 0

    return select_period, selected, selected_p, restart

def Team_Savings_Algo(NewRoutes, t, N, X, Store_IDs, dist, Y, MVisits, CT_Single, CT,overage,truck_c,single_miles, single_lbs, team_miles, team_lbs):
    # Copy the values to start the loop
    Routes = NewRoutes[0].copy()
    R_lbs = NewRoutes[1].copy()
    R_Dist = NewRoutes[2].copy()
    R_team = NewRoutes[3].copy()
    R_Cost = NewRoutes[4].copy()

    # ---------Calculate Distance Savings for each store---------

    Stores_t = []  # save stores that are visited on period t
    for n in range(N):
        if Y[n, t] >= 0.1:
            Stores_t.append(Store_IDs.iloc[n][0])

    S = numpy.zeros((len(Stores_t), len(Stores_t)), dtype=float)

    for i in range(len(Stores_t)):
        for j in range(len(Stores_t)):
            if i != j:
                S[i, j] = dist.loc[Stores_t[i]][1000] + dist.loc[1000][Stores_t[j]] - dist.loc[Stores_t[i]][Stores_t[j]]

    # -------Order the Savings

    # Crates list of the row, column indexes of S in ascending order
    r_og, c_og = numpy.unravel_index(numpy.argsort(S, axis=None), S.shape)

    # ---- Calculate Team Savings for each route
    loop = 0
    fixed = 0
    while loop == 0:
        # Team Savings
        TS = []
        TS_Routes = []
        TS_Rlbs = []
        TS_Rdist = []
        TS_team = []
        TS_RCost = []

        Covered_loop = []
        for route in range(fixed):
            for stop in Routes[route]:
                Covered_loop.append(stop)
        # print("Loop: ", fixed)
        # print("No. Routes: ", len(Routes)-fixed)
        for route in range(len(Routes)):

            if R_team[route] == 0:
                # Flip the lists to descending order
                r = r_og[::-1].tolist()
                c = c_og[::-1].tolist()

                # -----Build routes
                Covered = Covered_loop.copy()
                for stop in Routes[route]:
                    Covered.append(stop)
                # Covered = Routes[route].copy()
                Rm = Routes[route].copy()
                Capm = R_lbs[route]
                Distm = R_Dist[route]
                Timem = Capm*.022 + Distm*1.5 + len(Rm)*10
                # R_Cost = []
                # R_counter = 0
                # OnB = 0

                full = 0

                while full == 0:
                    #Timem = 0

                    # add stores to route
                    for i in range(len(r)):
                        if len(Rm) == MVisits:
                            full = 1
                            break

                        v = Stores_t[r[i]]
                        w = Stores_t[c[i]]
                        if v == w:
                            continue
                        elif w == Rm[0] or v == Rm[len(Rm) - 1]:
                            v_indx = pandas.Index(Store_IDs['Store ID']).get_loc(v)
                            w_indx = pandas.Index(Store_IDs['Store ID']).get_loc(w)
                            if w == Rm[0]:
                                if v in Covered:
                                    # r.remove(r[i])
                                    # c.remove(c[i])
                                    continue
                                elif Y[v_indx, t] + Capm <= CT:
                                    Dist_Timecalc =  1.5*(-dist.loc[1000][w] + dist.loc[1000][v] + dist.loc[v][w])
                                    if Timem + Dist_Timecalc + Y[v_indx,t]*.022 +10 <= 2*1080:
                                        Rm.insert(0, v)
                                        Capm = Capm + Y[v_indx, t]
                                        Distm = Distm - dist.loc[1000][w] + dist.loc[1000][v] + dist.loc[v][w]
                                        Timem = Distm * 1.5 + Capm * .022 + len(Rm) * 10
                                        Covered.append(v)
                                        # r.remove(r[i])
                                        # c.remove(c[i])
                                        if sum(Y[:, t]) > 0:
                                            if Capm + min([x for x in Y[:, t] if x > 0]) > CT:
                                                full = 1

                                        break
                                else:
                                    # r.remove(r[i])
                                    # c.remove(c[i])
                                    continue
                            elif v == Rm[len(Rm) - 1]:
                                if w in Covered:
                                    # r.remove(r[i])
                                    # c.remove(c[i])
                                    continue
                                elif Y[w_indx, t] + Capm <= CT:  # _Single:
                                    Dist_Timecalc = 1.5 * (-dist.loc[v][1000] + dist.loc[v][w] + dist.loc[w][1000])
                                    if Timem + Dist_Timecalc + Y[w_indx, t] * .022 + 10 <= 2 * 1080:
                                        Rm.append(w)
                                        Capm = Capm + Y[w_indx, t]
                                        Distm = Distm - dist.loc[v][1000] + dist.loc[v][w] + dist.loc[w][1000]
                                        Timem = Distm * 1.5 + Capm * .022 + len(Rm) * 10
                                        # print('TimeCheck: ', TimeCheck, Timem)
                                        Covered.append(w)
                                        # r.remove(r[i])
                                        # c.remove(c[i])
                                        if sum(Y[:, t]) > 0:
                                            if Capm + min([x for x in Y[:, t] if x > 0]) > CT:
                                                full = 1


                                        break
                                else:
                                    # r.remove(r[i])
                                    # c.remove(c[i])
                                    continue
                        if full == 1:
                            break
                    if i == len(r) - 1:  # No additional store could be added to route
                        full = 1

                # calculate route time
                Rtime = Distm * 1.5 + len(Rm) * 10 + Capm * .022
                # print("  TS Route: ", Rm)
                # print("  Rtime: ", Rtime)

                if Rtime > 1080:
                    # print("Route: ", Rm)

                    Costm = Distm * team_miles + Capm * team_lbs + overage
                    # print("     penalty: ", Costm, Rtime, Timem)
                else:
                    Costm = Distm * team_miles + Capm * team_lbs
                    # print("     no penalty: ", Costm, Rtime, Timem)

                Costmadd = 0

                if len(Rm) > len(Routes[route]):
                    addtruck = 1
                else:
                    addtruck = 0

                if addtruck == 1:
                    # calculate the cost of visiting the additional stores on a separate route
                    add_stores = []
                    for st in Rm:
                        if st in Routes[route]:
                            continue
                        else:
                            add_stores.append(st)
                    dc_dist = [dist.loc[1000][st] for st in add_stores]
                    st1 = dc_dist.index(min(dc_dist))
                    st1_indx = pandas.Index(Store_IDs['Store ID']).get_loc(add_stores[st1])
                    Distmadd = min(dc_dist)
                    Capmadd = Y[st1_indx, t]

                    Rm_add = [add_stores[st1]]

                    for st in range(len(add_stores) - 1):
                        test_dist = [dist.loc[Rm_add[st]][p] for p in add_stores]
                        for ff in range(len(add_stores)):
                            min_dist = sorted(test_dist)[ff]
                            stnext = test_dist.index(min_dist)
                            if add_stores[stnext] in Rm_add:
                                continue
                            else:
                                break
                        Rm_add.append(add_stores[stnext])
                        stnext_indx = pandas.Index(Store_IDs['Store ID']).get_loc(add_stores[stnext])
                        Distmadd = Distmadd + min(test_dist)
                        Capmadd = Capmadd + Y[stnext_indx, t]
                    Distmadd = Distmadd + dist.loc[Rm_add[len(Rm_add) - 1]][1000]
                    Timemadd = Distmadd * 1.5 + len(Rm_add) * 10 + Capmadd * .022

                    if Capmadd > CT_Single:
                        if Timemadd > 1080:

                            Costmadd = Distmadd * team_miles + Capmadd * team_lbs + overage

                        else:
                            Costmadd = Distmadd * team_miles + Capmadd * team_lbs
                    else:
                        if Timemadd > 810:

                            Costmadd = Distmadd * single_miles + Capmadd * single_lbs + overage
                        else:
                            Costmadd = Distmadd * single_miles  + Capmadd * single_lbs

                TS_Routes.append(Rm)
                TS_Rlbs.append(Capm)
                TS_Rdist.append(Distm)
                TS_team.append(1)
                TS_RCost.append(Costm)
                # print("R_route:", Routes[route])
                # print("TS Route: ", Rm)
                # print('TS Costs: ', R_Cost[route], Costmadd, Costm)
                teamsav = R_Cost[route] + Costmadd - Costm + addtruck * truck_c
                TS.append(teamsav)

        # -------Evaluate Team Savings
        if TS:
            # print("TS: ", TS)
            # ------------------------Evaluate Team Savings
            maxTS = max(TS)
            # print("max: ", fixed, maxTS)
            if maxTS > 0:  # Check if highest Team Savings is a positive saving
                route_indx = TS.index(maxTS)
                if fixed == 0:
                    # Include team route as part of the final routes
                    Routes = [TS_Routes[route_indx].copy()]
                    R_lbs = [TS_Rlbs[route_indx]]
                    R_Dist = [TS_Rdist[route_indx]]
                    R_team = [TS_team[route_indx]]
                    R_Cost = [TS_RCost[route_indx]]

                    Covered = TS_Routes[route_indx].copy()

                else:
                    # Save previously routes built with Team Savings
                    Routes = Routes[0:fixed]
                    R_lbs = R_lbs[0:fixed]
                    R_Dist = R_Dist[0:fixed]
                    R_team = R_team[0:fixed]
                    R_Cost = R_Cost[0:fixed]

                    # Include new route built with Team Savings
                    Routes.append(TS_Routes[route_indx])
                    R_lbs.append(TS_Rlbs[route_indx])
                    R_Dist.append(TS_Rdist[route_indx])
                    R_team.append(TS_team[route_indx])
                    R_Cost.append(TS_RCost[route_indx])

                    Covered = []
                    for j in range(len(Routes)):
                        for l in Routes[j]:
                            Covered.append(l)

                # print("Covered: ", Covered)
                # -----------------------Re-Build routes with stores not on Saved route
                fixed = fixed + 1

                # --------Initialize the rest of the stores as OnB
                SA_routes = []
                SA_Cap = []
                SA_Dist = []
                # route_by_store =[]
                SA_stores = []
                routes_full = []  # keeps track of when routes become full

                for store in Stores_t:
                    if store in Covered:
                        continue
                    else:
                        s_indx = pandas.Index(Store_IDs['Store ID']).get_loc(store)
                        SA_routes.append([store])
                        SA_Cap.append(Y[s_indx, t])
                        SA_Dist.append(dist.loc[1000][store] + dist.loc[store][1000])
                        SA_stores.append(store)
                        routes_full.append(0)

                route_by_store = [i for i in range(len(SA_routes))]  # Identifies on which route each store is at

                r = r_og[::-1].tolist()
                c = c_og[::-1].tolist()

                for i in range(len(r)):

                    v = Stores_t[r[i]]
                    w = Stores_t[c[i]]

                    if v in Covered or w in Covered:
                        continue
                    else:

                        # print('Edge: {}'.format((v,w)))
                        # print('  - Current routes: {}'.format(SA_routes))
                        # print('  - Route ids by store: {}'.format(route_by_store))

                        route1_id = route_by_store[SA_stores.index(v)]

                        route2_id = route_by_store[SA_stores.index(w)]

                        route1 = SA_routes[route1_id]
                        route2 = SA_routes[route2_id]
                        # print('Relevant routes: {} and {}'.format(route1, route2))

                        if route1_id != route2_id and \
                                len(route1) + len(route2) <= MVisits and \
                                SA_Cap[route1_id] + SA_Cap[route2_id] <= CT_Single:
                            # print('Potential merge!')
                            if (v, w) == (route1[-1], route2[0]):
                                U_dist1 = SA_Dist[route1_id] - dist.loc[route1[-1]][1000]
                                U_dist2 = SA_Dist[route2_id] - dist.loc[1000][route2[0]]

                                U_timecalc = (U_dist1 + U_dist2 + dist.loc[v][w])*1.5 + \
                                             (SA_Cap[route1_id]+SA_Cap[route2_id])*.022 + (len(route1)+len(route2))*10
                                if U_timecalc <= 2*810:

                                    route1.extend(route2)
                                    SA_Cap[route1_id] += SA_Cap[route2_id]
                                    SA_Cap[route2_id] = 0
                                    SA_Dist[route1_id] = U_dist1 + U_dist2 + dist.loc[v][w]
                                    SA_Dist[route2_id] = 0

                                    route_by_store = [route1_id if rid == route2_id else rid for rid in route_by_store]
                                    SA_routes[route2_id] = None
                                    routes_full[route2_id] = 1
                                    if len(route1) == MVisits or \
                                            SA_Cap[route1_id] + min([x for x in Y[:, t] if x > 0]) > CT_Single:
                                        routes_full[route1_id] = 1

                            elif (v, w) == (route2[-1], route1[0]):
                                U_dist1 = SA_Dist[route2_id] - dist.loc[route2[-1]][1000]
                                U_dist2 = SA_Dist[route1_id] - dist.loc[1000][route1[0]]

                                U_timecalc = (U_dist1 + U_dist2 + dist.loc[v][w]) * 1.5 + \
                                             (SA_Cap[route1_id] + SA_Cap[route2_id]) * .022 + (
                                                         len(route1) + len(route2)) * 10
                                if U_timecalc <= 2 * 810:

                                    route2.extend(route1)
                                    SA_Cap[route2_id] += SA_Cap[route1_id]
                                    SA_Cap[route1_id] = 0
                                    SA_Dist[route2_id] = U_dist1 + U_dist2 + dist.loc[v][w]
                                    SA_Dist[route1_id] = 0

                                    route_by_store = [route2_id if rid == route1_id else rid for rid in route_by_store]
                                    SA_routes[route1_id] = None
                                    routes_full[route1_id] = 1
                                    if len(route1) == MVisits or \
                                            SA_Cap[route2_id] + min([x for x in Y[:, t] if x > 0]) > CT_Single:
                                        routes_full[route2_id] = 1

                            if sum(routes_full) == len(SA_routes):
                                # print('All routes are full')
                                break
                # ----Append resulting routes to final solution
                for i, route in enumerate(SA_routes):
                    if route != None:
                        Routes.append(route)
                        R_lbs.append(SA_Cap[i])
                        R_Dist.append(SA_Dist[i])
                        l_team = 0
                        R_team.append(l_team)
                        # calculate route time
                        Rtime = SA_Dist[i] * 1.5 + len(route) * 10 + SA_Cap[i] * .022

                        if Rtime > 810:
                            # print("penalty: ", R_counter, Rtime, Timem)
                            # print("Route: ", Rm)
                            Rcost = SA_Dist[i] * single_miles  + SA_Cap[i] * single_lbs + overage
                        else:
                            Rcost = SA_Dist[i] * single_miles + SA_Cap[i] * single_lbs

                        R_Cost.append(Rcost)

            else:
                loop = 1
        else:
            loop = 1

    NewRoutes = [Routes, R_lbs, R_Dist, R_team, R_Cost]

    return NewRoutes


def mst_routes(N, X, Y, t, Store_IDs, dist, MVisits, CT_Single,overage,single_miles, single_lbs):
    # ---------Calculate Savings---------

    Stores_t = []  # save stores that are visited on period t
    for n in range(N):
        if Y[n, t] >= 0.1:
            Stores_t.append(Store_IDs.iloc[n][0])

    S = numpy.zeros((len(Stores_t), len(Stores_t)), dtype=float)

    for i in range(len(Stores_t)):
        for j in range(len(Stores_t)):
            if i != j:
                S[i, j] = dist.loc[Stores_t[i]][1000] + dist.loc[1000][Stores_t[j]] - dist.loc[Stores_t[i]][Stores_t[j]]

    # -------Order the Savings

    # Crates list of the row, column indexes of S in ascending order
    r_og, c_og = numpy.unravel_index(numpy.argsort(S, axis=None), S.shape)

    # --------Initialize the rest of the stores as OnB
    SA_routes = []
    SA_Cap = []
    SA_Dist = []
    # route_by_store =[]
    SA_stores = []
    routes_full = []  # keeps track of when routes become full

    for store in Stores_t:
        s_indx = pandas.Index(Store_IDs['Store ID']).get_loc(store)
        SA_routes.append([store])
        SA_Cap.append(Y[s_indx, t])
        SA_Dist.append(dist.loc[1000][store] + dist.loc[store][1000])
        SA_stores.append(store)
        routes_full.append(0)

    route_by_store = [i for i in range(len(SA_routes))]  # Identifies on which route each store is at

    r = r_og[::-1].tolist()
    c = c_og[::-1].tolist()

    for i in range(len(r)):

        v = Stores_t[r[i]]
        w = Stores_t[c[i]]

        # print('Edge: {}'.format((v, w)))
        # print('  - Current routes: {}'.format(SA_routes))
        # print('  - Route ids by store: {}'.format(route_by_store))

        route1_id = route_by_store[SA_stores.index(v)]

        route2_id = route_by_store[SA_stores.index(w)]

        route1 = SA_routes[route1_id]
        route2 = SA_routes[route2_id]
        # print('Relevant routes: {} and {}'.format(route1, route2))

        if route1_id != route2_id and \
                len(route1) + len(route2) <= MVisits and \
                SA_Cap[route1_id] + SA_Cap[route2_id] <= CT_Single:
            # print('Potential merge!')
            if (v, w) == (route1[-1], route2[0]):

                U_dist1 = SA_Dist[route1_id] - dist.loc[route1[-1]][1000]
                U_dist2 = SA_Dist[route2_id] - dist.loc[1000][route2[0]]

                U_timecalc = (U_dist1 + U_dist2 + dist.loc[v][w]) * 1.5 + \
                             (SA_Cap[route1_id] + SA_Cap[route2_id]) * .022 + (len(route1) + len(route2)) * 10
                if U_timecalc <= 2 * 810:

                    route1.extend(route2)
                    SA_Cap[route1_id] += SA_Cap[route2_id]
                    SA_Cap[route2_id] = 0

                    SA_Dist[route1_id] = U_dist1 + U_dist2 + dist.loc[v][w]
                    SA_Dist[route2_id] = 0

                    route_by_store = [route1_id if rid == route2_id else rid for rid in route_by_store]
                    SA_routes[route2_id] = None
                    routes_full[route2_id] = 1
                    if len(route1) == MVisits or \
                            SA_Cap[route1_id] + min([x for x in Y[:, t] if x > 0]) > CT_Single:
                        routes_full[route1_id] = 1

            elif (v, w) == (route2[-1], route1[0]):
                U_dist1 = SA_Dist[route2_id] - dist.loc[route2[-1]][1000]
                U_dist2 = SA_Dist[route1_id] - dist.loc[1000][route1[0]]

                U_timecalc = (U_dist1 + U_dist2 + dist.loc[v][w]) * 1.5 + \
                             (SA_Cap[route1_id] + SA_Cap[route2_id]) * .022 + (len(route1) + len(route2)) * 10
                if U_timecalc <= 2 * 810:

                    route2.extend(route1)
                    SA_Cap[route2_id] += SA_Cap[route1_id]
                    SA_Cap[route1_id] = 0

                    SA_Dist[route2_id] = U_dist1 + U_dist2 + dist.loc[v][w]
                    SA_Dist[route1_id] = 0

                    route_by_store = [route2_id if rid == route1_id else rid for rid in route_by_store]
                    SA_routes[route1_id] = None
                    routes_full[route1_id] = 1
                    if len(route1) == MVisits or \
                            SA_Cap[route2_id] + min([x for x in Y[:, t] if x > 0]) > CT_Single:
                        routes_full[route2_id] = 1

            if sum(routes_full) == len(SA_routes):
                # print('All routes are full')
                break
    # ----Save resulting routes to final solution
    Routes = []
    R_lbs = []
    R_Dist = []
    R_team = []
    R_Cost = []

    for i, route in enumerate(SA_routes):
        if route != None:
            Routes.append(route)
            R_lbs.append(SA_Cap[i])
            R_Dist.append(SA_Dist[i])
            l_team = 0
            R_team.append(l_team)
            # calculate route time
            Rtime = SA_Dist[i] * 1.5 + len(route) * 10 + SA_Cap[i] * .022

            if Rtime > 810:
                # print("penalty: ", R_counter, Rtime, Timem)
                # print("Route: ", Rm)
                Rcost = SA_Dist[i] * single_miles + SA_Cap[i] * single_lbs  + overage
            else:
                Rcost = SA_Dist[i] * single_miles + SA_Cap[i] * single_lbs

            R_Cost.append(Rcost)

    NewRoutes = [Routes, R_lbs, R_Dist, R_team, R_Cost]

    return NewRoutes


def PA_SA_InvInt(paramfile, paraminst ,n,inst ,dlevel,rep):

#Loads Up and Initializes the main parameters (inst, dlevel)

    # start clock:
    start_time = time.time()

    # Read input data

    # Store Capacity
    CS = 7480

    # Truck Capacity
    CT = 15000
    CT_Single = 9000

    # Max Visits per route
    MVisits = 5

    #Cost parameters
    overage = 150
    truck_c = 150
    team_lbs = .02
    single_lbs = .015
    team_miles = 1.5
    single_miles = 1.2

    # Use solution file if all necessary data is in different sheets of excel file, otherwise you can open individual files
    solfile = 'Output_Industry{}.xlsx'.format(inst) #'Output_stress2.xlsx'#Output_stress-n10.xlsx'

    # Delivery quantities to stores by time period
    quantities = pandas.read_excel(solfile ,sheet_name='Quantity' ,index_col = 0) #pandas.read_excel('n{}_demand{}{}.xlsx'.format(n,inst, dlevel), sheet_name='Sheet1',index_col=0)

    # Demand information
    demand = quantities.copy()  # For this initial solution delivery quantities = demand

    # Store IDs
    Store_IDs = pandas.read_excel(solfile,sheet_name='Store_ID',index_col = 0)# pandas.read_excel('n{}_Store_ID.xlsx'.format(n), sheet_name='Store_ID',index_col=0)

    # Inventory space available at each store per period
    AInv = pandas.DataFrame(columns=quantities.columns, index=quantities.index)
    for i in range(len(quantities)):
        invappend = []
        for j in range(len(quantities.iloc[i, :])):
            invappend.append(CS - quantities.iloc[i, j] + demand.iloc[i, j])
        AInv.iloc[i] = invappend

    # Stores
    N = len(quantities)

    # Time horizon
    T = len(quantities.columns)

    # Deliveries to store by route and time period
    # q = pandas.read_excel(solfile, sheet_name='Little qs', index_col =0)

    # Distance Matrix between stores and DC
    distfile = 'Distance_Matrix.xlsx' #'n{}_dist{}.xlsx'.format(n,inst)
    dist = pandas.read_excel(distfile, index_col=0)

    #
    # Initial Inventory
    ratefile = 'Industry_rate{}-Average.xlsx'.format(inst)
    Io = pandas.read_excel(ratefile, sheet_name='Io', index_col=0)  # numpy.zeros(N, dtype=int) #n100_rate{}-Instant.xlsx'.format(inst)
    rate = pandas.read_excel(ratefile, sheet_name='rate', index_col=0)  # numpy.zeros(N,dtype=int)

    # -----------------------------------------Initial Schedule with Adjusted OnB Policy-------------------------------------

    Y, X, I = OnB_Pol(N, T, quantities,demand, CS, CT_Single, MVisits, Io, rate)


    # --------------------------Savings Algorithm (Routing)------------------------------------------------------------------
    Routes_t, R_lbs_t, R_Dist_t, team_ind_t, R_Cost_t = Savings_Algo(N, T, Y, X,Store_IDs,dist,MVisits, CT_Single,overage,single_miles, single_lbs, team_miles, team_lbs)

    #--------------------------Calculate Initial Solution Cost----------------------------------------------------

    Initial_Cost = 0
    trucks_used = 0
    for t in range(T):
        #print("Routes: ",Routes_t[t])
        #print("Lbs: ",R_lbs_t[t])
        #print("Dist: ", R_Dist_t[t])
        #print("Team: ", team_ind_t[t])
        #print("Cost: ", R_Cost_t)

        Initial_Cost = Initial_Cost + sum(R_Cost_t[t])
        if len(Routes_t[t]) > trucks_used:
            trucks_used = len(Routes_t[t])
        # print(t)
        # for cost in R_Cost_t[t]:
        # print(cost)


    Initial_Cost = Initial_Cost + (trucks_used * truck_c * (T - 1))
    #print(Initial_Cost, trucks_used, T - 1)
    # print("Initial Cost: ", Initial_Cost)

    # -----------------------------------------------------------------------------------------------------------------------

    # ------------------------SA Structure to Alter Schedule-----------------------------------------------------------------


    # Initialize all relevant parameters

    SA_params = pandas.read_csv(paramfile)

    SA_StartTime = time.time()
    # SA Input
    current_cost = Initial_Cost
    best_cost = current_cost
    best_sol = [Routes_t, R_lbs_t, R_Dist_t, team_ind_t, R_Cost_t, Y, I, X]
    curr_sol = best_sol.copy()

    # SA Parameters
    endcriteria = 0
    endcode = 0
    #input
    alpha = SA_params.iloc[0][0]
    tau = SA_params.iloc[1][0]
    e = SA_params.iloc[2][0]
    SA_N = SA_params.iloc[3][0]

    t0 = current_cost / tau

    I_temp = 1
    accept_i = 1
    Temp = [t0, alpha * t0]
    i_temp = 1

    epoch_counter = 0
    MaxEpochs = 350
    epsilon = .05

    e_counter = 0
    eN_counter = 0

    cost_sum = 0
    fe_bar = 0
    fe_bar_prime = 0
    fe_bar_list = []
    fe_bar_list_e = []

    MaxRT = 3600 # 86400 #3600  # 10800 #input
    SA_counter = 0
    bsol_counter = 0
    bsol_accepted = 0
    max_eN = 0

    temp_sol = curr_sol.copy()
    temp_cost = current_cost

    accepted_counter = 0


    x_list = []  # store exchanges done by SA until next accepted solution

    #--------------------------------

    #Start the SA loop

    print("SA Initial Loop")
    # for disturb in range(100):

    # Copy Solution into Prime solution
    Routes_tp = Routes_t.copy()
    R_lbs_tp = R_lbs_t.copy()
    R_Dist_tp = R_Dist_t.copy()
    team_ind_tp = team_ind_t.copy()
    R_Cost_tp = R_Cost_t.copy()
    X_p = X.copy()
    Y_p = Y.copy()
    I_p = I.copy()

    # Select period with highest cost
    '''
    h_cost = 0
    for t in range(T):
        cost = sum(R_Cost_tp[t])
        if cost >= h_cost:
            h_cost = cost
            mvt = t
    '''

    while endcriteria == 0:
        # for loop in range(100):
        SA_counter = SA_counter + 1

        Routes_tp = curr_sol[0].copy()
        R_lbs_tp = curr_sol[1].copy()
        R_Dist_tp = curr_sol[2].copy()
        team_ind_tp = curr_sol[3].copy()
        R_Cost_tp = curr_sol[4].copy()
        Y_p = curr_sol[5].copy()
        I_p = curr_sol[6].copy()
        X_p = curr_sol[7].copy()

        #Select an Initial Period

        selected_p = 0
        period_opts = [t for t in range(T - 1) if sum(Y_p[:, t]) > 0.1]  # [*range(T-1)]

        restart = 0
        while selected_p == 0:

            #print(Y_p)

            #----Randomly Select Initial Period
            mvt = choice(period_opts)


            #---Select Initial Period with lowest truck utilization
            #mvt, d_selected,d_selected_p, d_restart = Truck_Util(period_opts, R_lbs_tp, team_ind_tp, CT, CT_Single)
            #print("mvt = ",mvt)

            #----Select Initial Period with Highest number of visits
            '''
            highV = 0
            for j in period_opts:
                V_calc = sum(X_p[:, j])
                if V_calc >= highV:
                    highV = V_calc
                    mvt = j
            '''

            #-----Select Initial Period with Highest Route Costs
            '''
            highC = 0
            for j in period_opts:
                cost_calc = sum(R_Cost_tp[j])
                if cost_calc >= highC:
                    highC = cost_calc
                    mvt = j
            #'''

            # print("Choice Periods 1: ", period_opts)
            # print("random period ", mvt)


            # ----------Step 2a

            # Create a list of all the stores visited on this period
            stores_mvt = []
            for route in Routes_tp[mvt]:
                for store in route:
                    stores_mvt.append(store)

            choice_stores = stores_mvt.copy()
            # print(choice_stores)
            selected = 0

            if not choice_stores:
                selected = 1
                period_opts.remove(mvt)
                selected_p = 0

            # Randomly select a store to move its delivery
            while selected == 0:
                # print("Choice Store: ", choice_stores)

                #Select store randomly
                select_store = choice(choice_stores)
                #print("select_store = ", select_store)

                #Select store with smallest delivery
                '''
                small_del = CS
                for outlet in choice_stores:
                    out_indx = pandas.Index(Store_IDs['Store ID']).get_loc(outlet)
                    check_del = Y_p[out_indx, mvt]
                    if check_del <= small_del:
                        small_del = check_del
                        select_store = outlet
                #'''

                # print(select_store)
                store_indx = pandas.Index(Store_IDs['Store ID']).get_loc(select_store)
                store_lbs = Y_p[store_indx, mvt]

                # Evaluate feasible periods to exchange
                choice_periods = []

                for t in range(mvt, -1, -1):  # Check previous periods
                    if t == mvt:
                        continue
                    if CS - I_p[store_indx, t] > 0:  # store_lbs:
                        choice_periods.append(t)

                # remove this for loop to only look at moving deliveries to an earlier period
                for t in range(mvt + 1, T):  # Check future periods
                    add_lbs = store_lbs - sum(demand.iloc[store_indx][m] for m in range(mvt, t))
                    if add_lbs > 0:  # check if the excess inventory can be delivered later
                        #        if CS - I_p[store_indx,t] >= store_lbs - sum(demand.iloc[store_indx][m] for m in range(mvt, t)):
                        if CS - (I_p[store_indx, t] - (max(add_lbs - rate.iloc[store_indx][t], 0))) > 0:
                            choice_periods.append(t)

        #------------------------------------Select an Alt Period-----------------------------
                selected_p2 = 0
                while selected_p2 == 0:
                    if choice_periods:  # If the alt period list is not empty

                        #-----Select the alt period randomly:
                        #select_period = choice(choice_periods)

                        #select_period, selected_p2, selected, selected_p, restart, choice_stores, period_opts = \
                         #   Rand_AltP(choice_periods,choice_stores, mvt,select_store,period_opts,x_list)

                        #-----Select the alt period with lowest truck utilization (TRUCK 1 & 2)

                        select_period, selected, selected_p, restart = Truck_Util(choice_periods,R_lbs_tp,team_ind_tp,CT,CT_Single)

                        #print("select_period = ", select_period)

                        #-----Select the alt period with most store space available

                        #select_period, selected, selected_p, restart = Store_Util(choice_periods, mvt,CS,I_p,store_lbs,demand,store_indx,rate)

                        #-----Select the alt period with lowest # of Visits
                        '''
                        lowV = 100
                        for j in choice_periods:
                            V_calc = sum(X[:,j])
                            if V_calc <= lowV:
                                lowV = V_calc
                                select_period = j
                        #'''
                        #print(x_list)
                        if (mvt, select_store,select_period) in x_list:
                            selected_p2=0
                            choice_periods.remove(select_period)

                        else:
                            selected = 1
                            selected_p = 1
                            selected_p2 = 1
                            restart = 0
                            x_list.append((mvt, select_store, select_period))
                            #print(x_list)

                    else:  # If the alt period list is empty, select another store randomly
                        selected_p2 = 1
                        selected = 0
                        choice_stores.remove(select_store)
                        if not choice_stores:
                            selected = 1
                            period_opts.remove(mvt)
                            selected_p = 0
                            if not period_opts:
                                selected_p = 1
                                restart = 1
                                # print("No Exchanges left to explore")
                                # restart the search from best solution found so far
                                x_list = []
                                # curr_sol = temp_sol.copy() #best_sol.copy()
                                # current_cost = temp_cost #best_cost
        #---------------------------------------------------------------------------------------------

        if restart == 0:
            new_ts = [mvt, select_period]  # list of periods where changes to the delivery plan occur

            # X_p[store_indx,select_period] = 1

            # Decide on quantity to move
            space_lim = []
            if select_period < mvt:
                # Make sure the new delivery does not use up the space needed for other earlier deliveries scheduled
                for j in range(select_period, mvt):
                    if Y_p[store_indx, j] > .01:
                        # Store Cap - What's been delivered to the store - Inventory
                        if j == 0:
                            space_lim.append(CS - Y_p[store_indx, j] - Io.iloc[store_indx][0])
                        else:
                            space_lim.append(CS - Y_p[store_indx, j] - I_p[store_indx, j - 1])
                if not space_lim:
                    space_lim.append(CS * 2)

                # Determine the new quantities to move to new period
                # Min between: Space available at the store | The total original quantity | the min set in previous section to no disrupt other deliveries
                if select_period == 0:
                    new_lbs = min(max(CS - Y_p[store_indx, select_period] - Io.iloc[store_indx][0],0), store_lbs, min(space_lim))
                else:
                    new_lbs = min(max(CS - Y_p[store_indx, select_period] - I_p[store_indx, select_period - 1],0), store_lbs,
                                  min(space_lim))
            else:
                # determine how much can be moved forward without disrupting demand needs
                add_lbs = store_lbs - sum(demand.iloc[store_indx][m] for m in range(mvt, select_period))
                # determine new quantity for the selected period
                # min between: what can be moved forward | Space available at the store (considering that the add_lbs amount is no longer there) |
                new_lbs = min(add_lbs, max(CS - Y_p[store_indx, select_period] -
                                           (max(I_p[store_indx, select_period - 1] - add_lbs,0),0)))# -demand.iloc[store_indx][select_period],0)))

            # Update the prime solution delivery schedule and store inventory to reflect the change
            for t in new_ts:  # ange(T):
                if t == mvt:
                    Y_p[store_indx, t] = max(Y_p[store_indx, t] - new_lbs, 0)
                elif t == select_period:
                    Y_p[store_indx, t] = Y_p[store_indx, t] + new_lbs
                # elif t > select_period:
                #    if Y_p[store_indx, t] > 0:
                #        new_ts.append(t)
                #        Y_p[store_indx, t] = max((sum(Y[store_indx,m] for m in range(t+1)) - sum(Y_p[store_indx,m] for m in range(t))), 0)

            for t in range(T):

                if t == 0:
                    I_p[store_indx, t] = max(0 ,Io.iloc[store_indx][0] + Y_p[store_indx, t] - rate.iloc[store_indx][t])
                else:
                    I_p[store_indx, t] = max(0, I_p[store_indx, t - 1] + Y_p[store_indx, t] - rate.iloc[store_indx][t])

                if Y_p[store_indx, t] > 0:
                    X_p[store_indx, t] = 1
                else:
                    X_p[store_indx, t] = 0

            stores_sp = []  # build a list of stores visited on select_period
            for route in Routes_t[select_period]:
                for store in route:
                    stores_sp.append(store)

            # Identify Nearest Neighbor

            # NearN, NearR = NN(select_store,select_period, new_lbs, stores_sp,Routes_t,R_lbs_t, MVisits,dist)
            if new_lbs > 0:
                for t in new_ts:

                    # Build new routes with the updated schedule using the Savings Algorithm
                    # print(t)
                    BaseNewRoutes = mst_routes(N, X_p, Y_p, t, Store_IDs, dist, MVisits, CT_Single,overage,single_miles, single_lbs)
                    # print("BaseNewRoutes: ")

                    # for results in range(len(BaseNewRoutes[0])):
                    # print("   ", BaseNewRoutes[0][results])
                    # Savings_Algo(t,N,X_p,Store_IDs,dist,Y_p,MVisits, CT_Single)
                    BaseCost = sum(BaseNewRoutes[4]) + len(BaseNewRoutes[0]) * truck_c
                    PrevCost = 0
                    NewCost = 0
                    # print("BaseCost: ",BaseCost)

                    NewRoutes = Team_Savings_Algo(BaseNewRoutes, t, N, X_p, Store_IDs, dist, Y_p, MVisits, CT_Single, CT,overage,truck_c,single_miles, single_lbs, team_miles, team_lbs)
                    # Savings_AlgoV7(BaseNewRoutes,t,N,X_p,Store_IDs,dist,Y_p,MVisits, CT_Single,CT)
                    NewCost = sum(NewRoutes[4]) + len(NewRoutes[0]) * truck_c

                    if NewCost < BaseCost:
                        # print("NewCost: ", NewCost)
                        BaseNewRoutes = NewRoutes.copy()
                        BaseCost = NewCost
                        # print("New Routes: ")
                        # for results in range(len(NewRoutes[0])):
                        # print("   ", NewRoutes[0][results])


                    NewRoutes = BaseNewRoutes.copy()

                    Routes_tp[t] = NewRoutes[0].copy()
                    R_lbs_tp[t] = NewRoutes[1].copy()
                    R_Dist_tp[t] = NewRoutes[2].copy()
                    team_ind_tp[t] = NewRoutes[3].copy()
                    R_Cost_tp[t] = NewRoutes[4].copy()


                sol_prime = [Routes_tp, R_lbs_tp, R_Dist_tp, team_ind_tp, R_Cost_tp, Y_p, I_p, X_p]

                Updated_Cost = 0
                trucks_used = 0
                for t in range(T):
                    Updated_Cost = Updated_Cost + sum(R_Cost_tp[t])
                    if len(Routes_tp[t]) > trucks_used:
                        trucks_used = len(Routes_tp[t])
                    # print(t)
                    # for cost in R_Cost_tp[t]:
                    # print(cost)

                Updated_Cost = Updated_Cost + (trucks_used * truck_c * (T - 1))
                # print("Updated Cost: ", Updated_Cost, SA_counter)
                # skip = 0

                # if skip == 1:
                # Compute the change in total cost
                Cost_Delta = current_cost - Updated_Cost

                if Cost_Delta > 0:
                    sample_rand = 0
                    accept = 1
                else:
                    sample_rand = 1
                    accept = 0
            else:
                accept = 0
                sample_rand = 0
            # Sample a random variable to see if the solution is accepted

            if sample_rand == 1:
                rand_var = numpy.random.uniform(0, 1)
                # print("rand: ",rand_var)
                # print("treshold: ",math.exp(Cost_Delta/Temp[i_temp]))
                # print("Temp: ", Temp[i_temp])
                # print("Cost_Delta: ", Cost_Delta)
                if rand_var <= math.exp(Cost_Delta / Temp[i_temp]):
                    accept = 1
                else:
                    accept = 0
                    eN_counter = eN_counter + 1

                    '''
                    if eN_counter >= max_eN:
   
                        #restart the search from best solution found so far
                        x_list = []
                        curr_sol = best_sol.copy()
                        current_cost = best_cost
                        eN_counter = 0
                        for t in range(T):
                            num_dels = num_dels + len([x for x in curr_sol[5][:, t] if x > 0])
   
                        max_eN = num_dels * 6 * 2
   
                        #endcriteria = 1
                        #print("Max Consecutive Non-Improving Changes Exceeded")
                    '''

            # ----Accept Solution

            if accept == 1:
                accepted_counter = accepted_counter + 1
                x_list = []  # restart exchange list

                curr_sol = sol_prime.copy()
                current_cost = Updated_Cost
                #print("Accepted: ", accepted_counter, current_cost)
                num_dels = 0
                for t in range(T):
                    num_dels = num_dels + len([x for x in curr_sol[5][:, t] if x > 0])

                max_eN = num_dels * 6 * 2
                eN_counter = 0

                if current_cost < best_cost:
                    best_sol = curr_sol.copy()
                    best_cost = current_cost
                    I_temp = i_temp
                    bsol_counter = SA_counter
                    bsol_accepted = accepted_counter
                    # print("best sol: ", bsol_counter, best_cost)

                e_counter = e_counter + 1  # counter for accepted solutions
                fe_bar_list_e.append(current_cost)
                cost_sum = cost_sum + current_cost
                fe_bar = cost_sum / e
                # print("Updated Current Sol: ", e_counter, SA_counter)

            if e_counter >= e:  # if e candidate solutions have been accepted
                epoch_counter = epoch_counter + 1
                # check for equilibrium
                # print("epoch_counter: ", epoch_counter)
                # print("fe_bar_list: ", fe_bar_list)
                if fe_bar_list:
                    fe_bar_prime = sum(fe_bar_list) / len(fe_bar_list)

                    # print("fe_bar: ", fe_bar)
                    # print("fe_bar_prime: ", fe_bar_prime)
                    # print("equilibrium: ",abs(fe_bar - fe_bar_prime)/fe_bar_prime)
                    if abs(fe_bar - fe_bar_prime) / fe_bar_prime >= epsilon:
                        e_counter = 0
                        endcriteria = 0
                        cost_sum = 0
                        for en in fe_bar_list_e:
                            fe_bar_list.append(en)
                        fe_bar_list_e = []
                    else:  # Update temperature
                        # print("Update Temp: ", i_temp + 1)
                        i_temp = i_temp + 1
                        Temp.append(Temp[0] * alpha ** i_temp)
                        e_counter = 0
                        cost_sum = 0
                        fe_bar_list_e = []
                        fe_bar_list = []
                        temp_sol = curr_sol.copy()
                        temp_cost = current_cost

                        if i_temp - I_temp < SA_N:  # max number of successive temp changes with no improvements
                            if epoch_counter < MaxEpochs:
                                endcriteria = 0
                            else:
                                endcriteria = 1
                                endcode = 1
                                # print("Max Epochs exceeded")
                        else:
                            endcriteria = 1
                            endcode = 2
                            # print("Max Successive temp changes without improvement exceeded")

                else:  # next epoch
                    e_counter = 0
                    endcriteria = 0
                    cost_sum = 0
                    for en in fe_bar_list_e:
                        fe_bar_list.append(en)
                    fe_bar_list_e = []

            if time.time() - SA_StartTime >= MaxRT:
                # print("Max Runtime exceeded")
                endcriteria = 1
                endcode = 3

    endscript = ["Max Epochs exceeded", "Max Successive temp changes without improvement exceeded",
                 "Max Runtime exceeded"]

    #Output File Writer
    with open('Ouput_File_inst{}-{}.txt'.format(inst,rep), 'w') as f:
        f.write("Params: a ={}, tau ={}, e = {}, MaxTemp = {}".format(alpha, tau, e, SA_N))
        f.write('\n')
        if endcode == 1:
            f.write(endscript[0])
        elif endcode == 2:
            f.write(endscript[1])
        elif endcode == 3:
            f.write(endscript[2])

        f.write('\n')

        f.write("Iterations: {}".format(SA_counter))
        f.write('\n')
        f.write("Runtime: {}".format(time.time() - start_time))
        f.write('\n')
        f.write("Final Best Cost: {}".format(best_cost))
        f.write('\n')
        f.write("Found in Iteration: {}".format(bsol_counter))
        f.write('\n')
        f.write("Best Solution Counter: {}".format(bsol_accepted))
        f.write('\n')
        f.write("Accepted Solutions: {}".format(accepted_counter))
        f.write('\n')
        f.write("Epochs: {}".format(epoch_counter))
        f.write('\n')
        # sol_prime = [Routes_tp, R_lbs_tp, R_Dist_tp, team_ind_tp,R_Cost_tp, Y_p, I_p, X_p]

        f.write("Solution")
        f.write('\n')
        f.write("Y:")
        f.write('\n')
        for n in range(N):
            f.write(str(best_sol[5][n, :]))
            f.write('\n')
        f.write("X:")
        f.write('\n')
        for n in range(N):
            f.write(str(best_sol[7][n, :]))
            f.write('\n')
        f.write("I:")
        f.write('\n')
        for n in range(N):
            f.write(str(best_sol[6][n, :]))
            f.write('\n')
        f.write("Routes:")
        f.write('\n')
        for t in range(T):
            f.write(str(t))
            f.write('\n')
            for route in best_sol[0][t]:
                f.write(str(route))
                f.write('\n')
        f.write("Lbs:")
        f.write('\n')
        for t in range(T):
            f.write(str(best_sol[1][t]))
            f.write('\n')
        f.write("Miles:")
        f.write('\n')
        for t in range(T):
            f.write(str(best_sol[2][t]))
            f.write('\n')

        f.write('Teams:')
        f.write('\n')
        for t in range(T):
            f.write(str(best_sol[3][t]))
            f.write('\n')

        f.write("Cost:")
        f.write('\n')
        for t in range(T):
            f.write(str(t))
            f.write('\n')
            for cost in best_sol[4][t]:
                f.write(str(cost))
                f.write('\n')
