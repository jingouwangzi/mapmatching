# -*- coding: UTF-8 -*-
"""
-------------------------------------------------------------------------------
# Name:        mapMatcher
# Purpose:      This python script allows map matching (matching of track points to a network)
#               in arcpy using a Hidden Markov model with
#               probabilities parameterized based on spatial + network distances.
#               Follows the ideas in Newson, Krumm (2009):
#               "Hidden markov Map Matching through noise and sparseness"
#
#               Example usage under '__main__'
#
# Author:      Simon Scheider
#
# Created:     01/03/2017
# Copyright:   (c) simon 2017
# Licence:     <your licence>

The code is written in Python 2.7 and depends on:

* arcpy (ships with ArcGIS and its own Python 2.7)
* networkx (# python pip install networkx (https://networkx.github.io))
    (note: requires installing GDAL first, which can be obtained as a wheel from
    http://www.lfd.uci.edu/~gohlke/pythonlibs/ and then installed with pip locally:
    python pip install GDAL-2.1.3-cp27-cp27m-win32.whl
    )

#-------------------------------------------------------------------------------
"""

__author__      = "Simon Scheider"
__copyright__   = ""


import sys

try:
    from math import exp, sqrt
    import os
    import arcpy
    arcpy.env.overwriteOutput = True #管理工具在运行时是否自动覆盖任何现有输出。设置为 True 时，工具将执行并覆盖输出数据集。
    import networkx as nx
    import time

except ImportError:
    print "Error: missing one of the libraries (arcpy, networkx)"
    sys.exit()



def mapMatch(track, segments, decayconstantNet = 30, decayConstantEu = 10, maxDist = 50, addfullpath = True):
    #6个参数，第一个是点，第二个是路径，后面四个是可选参数，具体看下面解释：
    #这里用的是隐马尔科夫模型，维特比算法
    """
    The main method. Based on the Viterbi algorithm for Hidden Markov models,
    see https://en.wikipedia.org/wiki/Viterbi_algorithm.
    It gets trackpoints and segments, and returns the most probable segment path (a list of segments) for the list of points.
    Inputs:
        @param track = a shape file (filename) representing a track, can also be unprojected (WGS84)
        @param segments = a shape file of network segments, should be projected (in meter) to compute Euclidean distances properly (e.g. GCS Amersfoord)
        @param decayconstantNet (optional) = the network distance (in meter) after which the match probability falls under 0.34 (exponential decay). (note this is the inverse of lambda).
        This depends on the point frequency of the track (how far are track points separated?)
        网络衰减距离，应该是大于这个距离的匹配可能性低于0.34（指数衰减）,根据代码可知，这个是状态转移概率，是指前后两个点的候选路段（也就是前后两个状态）之间的转移概率，两个路段距离越远转移概率越小
        这个距离的计算是最近端点之间的距离，如果两个路段首尾相连，那么转移概率是1
        取决于轨迹点的频率（轨迹点分布距离有多远）

        @param decayConstantEu (optional) = the Euclidean distance (in meter) after which the match probability falls under 0.34 (exponential decay). (note this is the inverse of lambda).
        This depends on the positional error of the track points (how far can points deviate from their true position?)
        欧氏衰减距离，应该是大于这个距离的匹配可能性低于0.34（指数衰减），根据后面的代码,这个是发射概率（输出概率）,就是指轨迹点在这个候选路段上的情况下，这个路段被选中的概率
        这里是指一个路段距离某个点的距离大于这个距离，那么点在这个路段上的概率就小于0.34，根据我的计算实际上大于这个距离，概率小于0.3679，概率计算：1/exp(dist/decayconstant)，dist就是点距离路段距离
        取决于轨迹点的位置错误（一个点与它的真实位置距离多远）

        @param maxDist (optional) = the Euclidean distance threshold (in meter) for taking into account segments candidates.
        最大距离，是可以匹配到路段上的欧氏距离的阈值，大于这个距离就认为跟路段不匹配，小于这个距离就选取作为路段候选

        @param addfullpath (optional, True or False) = whether a contiguous full segment path should be outputted. If not, a 1-to-1 list of segments matching each track point is outputted.
        是否输出完整的距离，如果选否，那么就只输出每个轨迹点对应的路段

    note: depending on the type of movement, optional parameters need to be fine tuned to get optimal results.
    """
    #Make sure passed in parameters are floats
    #强制转换后面三个参数为浮点型
    decayconstantNet = float(decayconstantNet)
    decayConstantEu = float(decayConstantEu)
    maxDist= float(maxDist)

    #gest start time
    start_time = time.time()

    #this array stores, for each point in a track, probability distributions over segments, together with the (most probable) predecessor segment taking into account a network distance
    #这段话太长没完全理解
    V = [{}]
    #V这个列表存储的是：对每一个点，对应一个字典，每个字典的key是这个点对应一定距离之内路段（getSegmentCandidates获得的候选路段）的objectid，value是另带一个字典，包含四对key-value，分别是'path': [], 'prev': None, 'prob': 0.4463, 'pathnodes': []。其中path是？prev是？prob是？pathnodes是？

    #get track points, build network graph (graph, endpoints, lengths) and get segment info from arcpy
    #获取轨迹点，创建网络图形，从arcpy获取路段信息
    points = getTrackPoints(track, segments)
    #参数是传入的轨迹点和道路线，但是看这个函数的代码，没有使用到segments，因为使用segments的一句被注释掉了
    r = getSegmentInfo(segments)#r就是两个字典，路段路径和路段长度两个字典
    endpoints = r[0] #这个就是路段路径所经过的点
    lengths = r[1] #这个就是路段的长度
    graph = getNetworkGraph(segments,lengths)#获取输入网络的最大连通分量（networkx的无向图，并且边的属性有路段长度）
    #以上有点不理解，看这个函数，获取的原始图有735个边，但是实际原来的shp有745个边，为什么？
    #原始图有735个边，最大分量有681个边（看了一下原始图、导入networkx的图和最大分量，确实数量都没问题，但是这个取舍是什么原理就不知道了）
    
    pathnodes = [] #set of pathnodes to prevent loops
    #创建一个点序列，来防止循环

    #init first point
    #初始化第一个轨迹点
    sc = getSegmentCandidates(points[0], segments, decayConstantEu, maxDist)
    #sc就是第一个点50米范围内每个路段的概率，一个字典，key是路段objectid，value是概率值

    for s in sc:
        V[0][s] = {"prob": sc[s], "prev": None, "path": [], "pathnodes":[]}
    # Run Viterbi when t > 0   #这里开始维特比算法
    for t in range(1, len(points)):
        V.append({})
        #Store previous segment candidates
        lastsc = sc  #保存上一个点相关路段的信息，sc用作下一个点的信息保存
        #Get segment candidates and their a-priori probabilities (based on Euclidean distance for current point t)
        sc = getSegmentCandidates(points[t], segments, decayConstantEu, maxDist)
        for s in sc:
            #s是一个point对应的某个观测状态，也就是某个路段（用objectid表示）及其对应的概率
            max_tr_prob = 0
            prev_ss = None
            path = []
            for prev_s in lastsc:
                #determine the highest network transition probability from previous candidates to s and get the corresponding network path
                #确定前一个点对应候选路段到s（这个点对应的某个路段）的最高网络转移概率，并得到相应的网络路径
                pathnodes = V[t-1][prev_s]["pathnodes"][-10:]
                n = getNetworkTransP(prev_s, s, graph, endpoints, lengths, pathnodes, decayconstantNet)
                np = n[0] #This is the network transition probability
                tr_prob = V[t-1][prev_s]["prob"]*np
                #this selects the most probable predecessor candidate and the path to it
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_ss = prev_s
                    path = n[1]
                    if n[2] != None:
                        pathnodes.append(n[2])
            #The final probability of a candidate is the product of a-priori and network transitional probability
            max_prob =  sc[s] * max_tr_prob
            V[t][s] = {"prob": max_prob, "prev": prev_ss, "path": path, "pathnodes":pathnodes}

        #Now max standardize all p-values to prevent running out of digits
        maxv = max(value["prob"] for value in V[t].values())
        maxv = (1 if maxv == 0 else maxv)
        for s in V[t].keys():
            V[t][s]["prob"]=V[t][s]["prob"]/maxv


    intertime1 = time.time()
    print("--- Viterbi forward: %s seconds ---" % (intertime1 - start_time))
    #print V

    #opt is the result: a list of (matched) segments [s1, s2, s3,...] in the exact order of the point track: [p1, p2, p3,...]
    opt = []

    # get the highest probability at the end of the track
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    if max_prob == 0:
        print " probabilities fall to zero (network distances in data are too large, try increasing network decay parameter)"

    # Get most probable ending state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
##    print  " previous: "+str(previous)
##    print  " max_prob: "+str(max_prob)
##    print  " V -1: "+str(V[-1].items())

    # Follow the backtrack till the first observation to fish out most probable states and corresponding paths
    for t in range(len(V) - 2, -1, -1):
        #Get the subpath between last and most probable previous segment and add it to the resulting path
        path = V[t + 1][previous]["path"]
        opt[0:0] =(path if path !=None else [])
        #Insert the previous segment
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    intertime2 = time.time()
    print("--- Viterbi backtracking: %s seconds ---" % (intertime2 - intertime1))

    #Clean the path (remove double segments and crossings) (only in full path option)
    print "path length before cleaning :" +str(len(opt))
    opt = cleanPath(opt, endpoints)
    intertime3 = time.time()
    print("--- Path cleaning: %s seconds ---" % (intertime3 - intertime2))
    print "final length: "+str(len(opt))
    pointstr= [str(g.firstPoint.X)+' '+str(g.firstPoint.Y) for g in points]
    optstr= [str(i) for i in opt]
    print 'The path for points ['+' '.join(pointstr)+'] is: '
    print '[' + ' '.join(optstr) + '] with highest probability of %s' % max_prob

    #If only a single segment candidate should be returned for each point:
    if addfullpath == False:
        opt = getpointMatches(points,opt)
        optstr= [str(i) for i in opt]
        print "Individual point matches: "+'[' + ' '.join(optstr) + ']'
        intertime4 = time.time()
        print("--- Picking point matches: %s seconds ---" % (intertime4 - intertime3))

    return opt

#Fishes out a 1-to-1 list of path segments nearest to the list of points in the track (not contiguous, may contain repeated segments)
def getpointMatches(points, path):
    qr =  '"OBJECTID" IN ' +str(tuple(path))
    arcpy.SelectLayerByAttribute_management('segments_lyr',"NEW_SELECTION", qr)
    opta = []
    for point in points:
        sdist = 100000
        candidate = ''
        cursor = arcpy.da.SearchCursor('segments_lyr', ["OBJECTID", "SHAPE@"])
        for row in cursor:
            #compute the spatial distance
            dist = point.distanceTo(row[1])
            if dist <sdist:
                sdist=dist
                candidate = row[0]
        opta.append(candidate)
    del cursor
    #print str(candidates)
    return opta



def simplisticMatch(track, segments, maxDist = 50):
    maxDist= float(maxDist)
    #get track points, build network graph (graph, endpoints, lengths) and get segment info from arcpy
    points = getTrackPoints(track, segments)
    opt =[]
    for t in range(1, len(points)):
        point = points[t]
        s = getClosestSegment(point, segments,maxDist)
        opt.append(s)

    return opt



def cleanPath(opt, endpoints):
    # removes redundant segments and segments that are unnecessary to form a path (crossings) in an iterative manner
    last =()
    lastlast =()
    optout = []
    for s in opt:
        if s != last:
            match = False
            if last != () and lastlast != ():
                lastep = endpoints[last]
                lastlastep = endpoints[lastlast]
                sep = endpoints[s]
                for j in lastlastep:
                    if lastep[0]== j:
                        for k in sep:
                            if lastep[1] == k:
                                match = True
                    elif lastep[1]== j:
                        for k in sep:
                            if lastep[0] == k:
                                match = True
            elif last != ():
                sep = endpoints[s]
                lastep = endpoints[last]
                for k in sep:
                    if lastep[1] == k or lastep[0] == k:
                        match = True
            if match:
                optout.append(last)
            if s == opt[-1]:
                #print "add final segment:"+str(s)
                optout.append(s)
        lastlast = last
        last = s
    #print "final length: "+str(len(optout))
    return optout

def getUniqueList(my_list):
    from collections import OrderedDict
    from itertools import izip, repeat

    unique_list = list(OrderedDict(izip(my_list, repeat(None))))
    return unique_list


def exportPath(opt, trackname):
    """
    This exports the list of segments into a shapefile, a subset of the loaded segment file, including all attributes
    """
    start_time = time.time()
    opt=getUniqueList(opt)
    qr =  '"OBJECTID" IN ' +str(tuple(opt))
    outname = (os.path.splitext(os.path.basename(trackname))[0][:9])+'_pth'
    arcpy.SelectLayerByAttribute_management('segments_lyr',"NEW_SELECTION", qr)
    try:
        if arcpy.Exists(outname):
            arcpy.Delete_management(outname)
        arcpy.FeatureClassToFeatureClass_conversion('segments_lyr', arcpy.env.workspace, outname)
        print("--- export: %s seconds ---" % (time.time() - start_time))
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])

        # If using this code within a script tool, AddError can be used to return messages
        #   back to a script tool.  If not, AddError will have no effect.
        arcpy.AddError(e.args[0])
        arcpy.AddError(arcpy.env.workspace)
        arcpy.AddError(outname)
        #raise arcpy.ExecuteError
    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))

    # Return any other type of error
    except:
        # By default any other errors will be caught here
        #
        e = sys.exc_info()[1]
        print(e.args[0])
        arcpy.AddError(e.args[0])
        arcpy.AddError(arcpy.env.workspace)
        arcpy.AddError(outname)


def getPDProbability(dist, decayconstant = 10):  #就是返回一个概率，距离越远概率越小
    """
    The probability that given a certain distance between points and segments, the point is on the segment
    This needs to be parameterized
    Turn difference into a probability with exponential decay function
    """
    decayconstant= float(decayconstant)
    dist= float(dist)
    try:
        p = 1 if dist == 0 else round(1/exp(dist/decayconstant),4)  #点落在路段上的概率，指数的概率递减
    except OverflowError:
        p =  round(1/float('inf'),2)   #注意float('inf')是一个特定的常量，是指正无穷，float('-inf')指负无穷，所以这句的意思就是，如果计算结果超出最大限制就认为是无穷大
    return p

def getSegmentCandidates(point, segments, decayConstantEu, maxdist=50):
    #获取候选路段，并给出候选路段的概率
    #这一步就是获取这个点在maxdist距离之内的所有路段，然后根据decayConstantEu，给每个路段赋一个概率值，也就是可能是点经过路段的概率
    #返回一个字典，key是路段的objectid，value是路段的概率
    """
    Returns closest segment candidates with a-priori probabilities.
    Based on maximal spatial distance of segments from point.
    基于从轨迹点到路段的最大空间距离
    以先验概率返回最接近的候选路段
    参数有四个，一个轨迹点，所有的路段，网络衰减距离，路段匹配最大距离
    """
    p = point.firstPoint #get the coordinates of the point geometry
    #p是一个这东西：<Point (160934.301881, 386407.104818, #, #)>，也就是arcpy.arcobjects.arcobjects.Point
    #arcpy里面点和点几何不一样，可以从下面两行代码看出来
    #point = arcpy.Point(25282, 43770)
    #ptGeometry = arcpy.PointGeometry(point)

    #print "Neighbors of point "+str(p.X) +' '+ str(p.Y)+" : "
    #Select all segments within max distance
    arcpy.SelectLayerByLocation_management ("segments_lyr", "WITHIN_A_DISTANCE", point, maxdist)
    #注意：如果图层没有，要创建一个arcpy.MakeFeatureLayer_management(segments, "segments_lyr"),
    #在getSegmentInfo函数里面已经创建了这个layer，看起来在这里还可以继续使用
    #但是这句运行之后出来的是个啥也没搞清楚
    #好像运行出来之后不会生成一个新的东西，但是对这个层操作就只会对选中的产生影响

    #注意以上三行的第一行有问题，实际getSegmentInfo函数里面已经创建了这个layer

    candidates = {}
    #Go through these, compute distances, probabilities and store them as candidates
    cursor = arcpy.da.SearchCursor('segments_lyr', ["OBJECTID", "SHAPE@"])
    row =[]  #这个好像也没有用到，下面的循环不用定义这个应该也是可以的
    for row in cursor:
        feat = row[1]  #这个好像没有用到
        #compute the spatial distance
        dist = point.distanceTo(row[1])  #这个distanceTo的用法是：如果垂线在路段上，就是垂线长度，如果不是，就是到线的最近端点的长度，总的来说就是到路段的最近点的长度
        #compute the corresponding probability
        candidates[row[0]] = getPDProbability(dist, decayConstantEu)
    del row
    del cursor
    #print str(candidates)
    return candidates

def getClosestSegment(point, segments, maxdist):
    arcpy.Delete_management('segments_lyr')
    arcpy.MakeFeatureLayer_management(segments, 'segments_lyr')
    arcpy.SelectLayerByLocation_management ("segments_lyr", "WITHIN_A_DISTANCE", point, maxdist)

    #Go through these, compute distances, probabilities and store them as candidates
    cursor = arcpy.da.SearchCursor('segments_lyr', ["OBJECTID", "SHAPE@"])
    sdist = 100000
    candidate = ''
    for row in cursor:
        #compute the spatial distance
        dist = point.distanceTo(row[1])
        if dist <sdist:
            sdist=dist
            candidate = row[0]
    del row
    del cursor
    #print str(candidates)
    return candidate


def getNDProbability(dist,decayconstant = 30):
    """
    The probability that given a certain network distance between segments, one is the successor of the other in a track
    This needs to be parameterized
    Turn difference into a probability  with exponential decay function
    """
    decayconstant = float(decayconstant)
    dist = float(dist)
    try:
        p = 1 if dist == 0 else  round(1/exp(dist/decayconstant),2)
    except OverflowError:
        p =  round(1/float('inf'),2)
    return p

def getNetworkTransP(s1, s2, graph, endpoints, segmentlengths, pathnodes, decayconstantNet):
    #一大堆参数，s1是上一个点的某个候选路段；s2是这个点的某个候选路段；graph是道路的最大连通分量（networkx对象）；endpoints是路段路径点字典，具体看前面的获取结果；segmentlengths是每个路段的长度字典，具体看前面的获取结果；pathnodes是前面定义的一个列表，应该是存储已经经过的点，防止环路出现的，但是看这个函数的代码好像没用到；decayconstantNet计算状态转移概率用的一个阈值距离
    #最后返回三个值，一个是路段之间的状态转移概率，一个是状态转移所经过的路段（如果首尾相连，那么转移概率为1，经过路段为空），还有一个是第二个点候选路段距离第一个点候选路段最近的端点



    """
    Returns transition probability of going from segment s1 to s2, based on network distance of segments, as well as corresponding path
    返回转移概率，应该就是状态序列（状态序列是隐含的，也就是实际上gps点代表的路径经过的路段序列）之间的转移概率，也就是一个正确的路径，前后路段之间的转移概率
    """
    subpath = []
    s1_point = None
    s2_point = None

    if s1 == s2:
        dist = 0
    else:
        #Obtain edges (tuples of endpoints) for segment identifiers
        s1_edge = endpoints[s1]
        s2_edge = endpoints[s2]

        s1_l = segmentlengths[s1]
        s2_l = segmentlengths[s2]

        #This determines segment endpoints of the two segments that are closest to each other
        minpair = [0,0,100000]
        for i in range(0,2):
            for j in range(0,2):
                d = round(pointdistance(s1_edge[i],s2_edge[j]),2)  #这个就是两个路段每个端点之间的距离
                if d<minpair[2]:
                    minpair = [i,j,d]  #得到的就是s1和s2两个路段距离最近的端点编号，和距离，端点编号为0或者1，0 是起点，1是终点
        s1_point = s1_edge[minpair[0]]
        s2_point = s2_edge[minpair[1]]

##        if (s1_point in pathnodes or s2_point in pathnodes): # Avoid paths reusing an old pathnode (to prevent self-crossings)
##            dist = 100
##        else:
        if s1_point == s2_point:
                #If segments are touching, use a small network distance
                    dist = 5    #这个dist是干嘛的？
        else:
                try:
                    #Compute a shortest path (using segment length) on graph where segment endpoints are nodes and segments are (undirected) edges
                    if graph.has_node(s1_point) and graph.has_node(s2_point):
                        dist = nx.shortest_path_length(graph, s1_point, s2_point, weight='length')
                        path = nx.shortest_path(graph, s1_point, s2_point, weight='length')
                        #get path edges
                        path_edges = zip(path,path[1:])
                        #print "edges: "+str(path_edges)
                        subpath = []
                        # get object ids for path edges
                        for e in path_edges:
                            oid = graph[e[0]][e[1]]["OBJECTID"]
                            subpath.append(oid)
                        #print "oid path:"+str(subpath)
                    else:
                        #print "node not in segment graph!"
                        dist = 3*decayconstantNet #600
                except nx.NetworkXNoPath:
                    #print 'no path available, assume a large distance'
                    dist = 3*decayconstantNet #700
    #print "network distance between "+str(s1) + ' and '+ str(s2) + ' = '+str(dist)
    return (getNDProbability(dist,decayconstantNet),subpath,s2_point)

def pointdistance(p1, p2):
    #This Eucl distance can only be used for projected coordinate systems
    dist = sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
    return dist

def getTrackPoints(track, segments):
    """
    Turns track shapefile into a list of point geometries, reprojecting to the planar RS of the network file
    把轨迹点转换为一个点几何要素的列表，并且投影到路网坐标系（实际投影这一步被原作者注释掉了，所以segments参数也没用上）
    """
    trackpoints = []
    if arcpy.Exists(track):
        for row in arcpy.da.SearchCursor(track, ["SHAPE@"]):
            #SearchCursor 用于建立从要素类或表中返回的记录的只读访问权限。返回一组迭代的元组。元组中值的顺序与 field_names 参数指定的字段顺序相符。
            #Geometry 属性可通过在字段列表中指定令牌 SHAPE@ 进行访问。
            #row就是一个个的点要素
            #make sure track points are reprojected to network reference system (should be planar)
            geom = row[0]
            #geom = row[0].projectAs(arcpy.Describe(segments).spatialReference) #投影到路网坐标系，被原作者注释掉
            trackpoints.append(row[0])
        print 'track size:' + str(len(trackpoints))
        return trackpoints  #返回一个由点要素对象组成的列表
    else:
        print "Track file does not exist!"

def getNetworkGraph(segments,segmentlengths):
    """
    Builds a networkx graph from the network file, inluding segment length taken from arcpy.
    It selects the largest connected component of the network (to prevent errors from routing between unconnected parts)
    """
    #generate the full network path for GDAL to be able to read the file
    path =str(os.path.join(arcpy.env.workspace,segments))
    print path
    if arcpy.Exists(path):
        #这里下面g的路段比原shp少些，只有573个个路段，sg更少，只有520个
        g = nx.read_shp(path)   #g就是一个图，就是提供的shp形成的一个图，g是digraph对象，是有向图
        #networkx的用法，学习理解一下
        #This selects the largest connected component of the graph
        sg = list(nx.connected_component_subgraphs(g.to_undirected()))[0]
        #.to_undirected是转换成无向图
        #获取连通分量(nx.connected_component_subgraphs(G)，返回的是列表，但是元素是图，这些分量按照节点数目从大到小排列，所以第一个就是最大的连通分量)。
        #获取的是一个由图组成的数组，所以sg就是第0个元素，也就是最大连通分量的元素
        ##sg是graph对象，是无向图
        #原图不是一个连通图，有些游离于大部队之外，这里只提取其中的最大联通部分，也就是最大连通分量元素
        print "graph size (excluding unconnected parts): "+str(len(g))
        # Get the length for each road segment and append it as an attribute to the edges in the graph.
        #获取每段路的长度，添加到networkx图中边的属性
        for n0, n1 in sg.edges():
            oid = sg[n0][n1]["OBJECTID"]
            sg[n0][n1]['length'] = segmentlengths[oid]#把路段长度赋值给图sg的边的length属性
        return sg
        '''
        print len(g)
        print len(sg)  
        print g.number_of_nodes()
        print g.number_of_edges()
        print sg.number_of_nodes()
        print sg.number_of_edges()
        '''
    else:
        print "network file not found on path: "+path

def getSegmentInfo(segments):
    """
    Builds a dictionary for looking up endpoints of network segments (needed only because networkx graph identifies edges by nodes)
    创建一个查询网络路段的字典，实际是两个字典，第一个字典key是路段objectid，值是一个由点坐标值组成的元祖，也就是路段经过的所有点；第二个字典，key也是objectid，值就是这个路段的长度。
    这么做仅仅是因为networkx中的图使用点来表示路段
    """
    if arcpy.Exists(segments):
        cursor = arcpy.da.SearchCursor(segments, ["OBJECTID", "SHAPE@"])
        endpoints = {}
        segmentlengths = {}
        for row in cursor:
              endpoints[row[0]]=((row[1].firstPoint.X,row[1].firstPoint.Y), (row[1].lastPoint.X, row[1].lastPoint.Y))
              segmentlengths[row[0]]= row[1].length
        del row
        del cursor
        print "Number of segments: "+ str(len(endpoints))
        #prepare segment layer for fast search
        arcpy.Delete_management('segments_lyr')  #首先删除图层，如果没有就算了
        arcpy.MakeFeatureLayer_management(segments, 'segments_lyr')  #此处创建一个arcgis的图层
        return (endpoints,segmentlengths)   
        #返回两个字典，第一个字典key是路段objectid，值是一个由点坐标值组成的元祖，也就是路段经过的所有点
        #第二个字典，key也是objectid，值就是这个路段的长度
    else:
        print "segment file does not exist!"


if __name__ == '__main__':  #如果是主动运行这个程序，就要定义以下三个参数，如果是调用这个程序，那么会传过来相关参数

##    #Test using the shipped data example
    arcpy.env.workspace = 'C:\\Users\\schei008\\Documents\\Github\\mapmatching'
    opt = mapMatch('testTrack.shp', 'testSegments.shp', 25, 10, 50)
    #outputs testTrack_path.shp
    exportPath(opt, 'testTrack.shp')

##    arcpy.env.workspace = 'C:\\Temp\\Road_293162'
##    opt = mapMatch('Track293162.shp', 'Road_2931_corr.shp', 300, 10, 50)
##    #outputs testTrack_path.shp
##    exportPath(opt, 'Track293162.shp')


##    arcpy.env.workspace = 'C:/Users/simon/Documents/GitHub/mapmatching/'
##    trackname ='QT170212C.shp'
##    roadname ='Roads2.shp'
##    opt = mapMatch(trackname, roadname, 20, 10, 50)
##    exportPath(opt, trackname)
##
##    arcpy.env.workspace = 'C:\\Temp\\Simon.gdb\\test'
##
##    trackname ='Tom170218.shp'
##    roadname ='TrkRoads_Tom1.shp'
##    print trackname
##    print roadname
##    opt = mapMatch(trackname, roadname, 20, 10, 50)
##    exportPath(opt, trackname)
##
##    trackname ='Tom170218_2.shp'
##    roadname ='TrkRoads_Tom2.shp'
##    print trackname
##    print roadname
##    opt = mapMatch(trackname, roadname, 20, 10, 50)
##    exportPath(opt, trackname)
##
####    trackname ='Maarten150318.shp'
####    roadname ='TrkRoads_Maarten.shp'
####    print trackname
####    print roadname
####    opt = mapMatch(trackname, roadname, 20, 10, 50)
####    exportPath(opt, trackname)
##
##    trackname ='Nico160706_ss.shp'
##    roadname ='TrkRoads_Nico.shp'
##    print trackname
##    print roadname
##    opt = mapMatch(trackname, roadname, 60, 40, 50)
##    exportPath(opt, trackname)
##    opt = simplisticMatch(trackname, roadname,50)
##    exportPath(opt, 'Nico_s')

