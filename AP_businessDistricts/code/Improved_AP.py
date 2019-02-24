


class Node:
    def __init__(self, x=0.0, y=0.0, checkin=0.0):
        self.x = x
        self.y = y
        self.checkin = checkin


def make_vertex_list_from_file(dir_path, file_name):
    """
        返回包含一座城市里所有节点的list，
        每个list里包含的是Node对象
    """
    ll = []
    data = {}
    with open(dir_path + file_name, 'rb+') as f:
        for line in f.readlines():  
            if line.strip():
                line = line.encode('utf-8').strip().split(' ')
                # 相同经纬度去重，把签到值加到一块
                if str(line[:-1]) not in data:
                    data[','.join(line[:-1])] = float(line[-1])
                else:
                    data[','.join(line[:-1])] += float(line[-1])
                    
    # 创建Node实例并保存到ll中
    # 因为放入到dict，节点的顺序将不同于源文件的顺序
    for k in data.keys():
        node_data = map(lambda x: float(x), k.split(','))
        node_data.append(data[k])
        ll.append(Node(*node_data))

    return ll

def print_dict(data):
    for k in data.keys():
        print('{}\t=> '.format(k), '|'*data[k], data[k])
        
# 重启后需要再次运行
charlotte_vertices = make_vertex_list_from_file(dir_path, file_name_list[0])
phoenix_vertices = make_vertex_list_from_file(dir_path, file_name_list[1])
pittsburgh_vertices = make_vertex_list_from_file(dir_path, file_name_list[2])
print(len(charlotte_vertices))
print(len(phoenix_vertices))
print(len(pittsburgh_vertices))

import math

def euclid_distance(node1, node2):
    return math.sqrt(
        math.pow((node1.x - node2.x), 2) + math.pow(node1.y - node2.y, 2))


# 数据点太大不宜计算矩阵
# def make_distance_matrix(vertices_list):
#     """
#         返回一座城市的距离矩阵
#     """
#     len_of_list = len(vertices_list)
#     distance_matrix = np.zeros((len_of_list, len_of_list), dtype=np.float)
#     for i in range(len_of_list):
#         distance_matrix[i][i] = 0.0
#         for j in range(i + 1, len_of_list):
#             distance_matrix[i][j] = distance_matrix[j][i] = euclid_distance(
#                 vertices_list[i], vertices_list[j])

#     return distance_matrix


# In[14]:


def yield_distance(vertices_list, i):
    """调用这个方法和直接循环没啥区别"""
    for j in range(len(vertices_list)):
        if i != j:
            yield euclid_distance(vertices_list[i], vertices_list[j])


def calc_k_distance(k_distance_list, j):
    # 当列表末尾的距离大于新距离j时就进行替换
    if k_distance_list[-1] > j:
        k_distance_list[-1] = j
    # 替换完后重新排序以保证尾数是列表中最大的值
    k_distance_list.sort()


def k_distance(vertices_list, k=4):
    max_k = -1
    max_distance = -1
    min_distance = 1 << 30
    for i in range(len(vertices_list)):
        # 保存前k个最小距离
        k_distance_list = []
        #         for j in yield_distance(vertices_list, i):
        for j in range(len(vertices_list)):
            if i != j:
                d = euclid_distance(vertices_list[i], vertices_list[j])
                if max_distance < d:
                    max_distance = d
                if min_distance > d:
                    min_distance = d
                # 列表小于k往里面加值最后排序
                # 否则比较新来的距离d和列表中的距离
                if len(k_distance_list) < k:
                    k_distance_list.append(d)
                    if len(k_distance_list) == k:
                        k_distance_list.sort()
                else:                    
                    calc_k_distance(k_distance_list, d)
        print('=> max distance', max_distance, '=> min distance', min_distance)
        # 除去影响最大k-distance的离异值
        # 归一化后这个语句貌似没什么意义了
        if k_distance_list[-1] > 5:
            print(i)
            print(k_distance_list[-1])
        else:
            max_k = max(max_k, k_distance_list[-1])
            print('=> max {} distance: {}'.format(k, max_k))
        print('-'*75)

    return max_k, max_distance, min_distance


# In[18]:


charlotte_vertices_max_k_dist, charlotte_vertices_max_dist, charlotte_vertices_min_dist = k_distance(
    uniformed_filtered_charlotte_vertices, k=124
)
print(charlotte_vertices_max_k_dist)
# node = charlotte_vertices[125]
# node2 = charlotte_vertices[124]
# node3 = charlotte_vertices[123]
# print(node3.x, node3.y)
# print(node2.x, node2.y)
# print(node.x, node.y)


# In[16]:


phoenix_vertices_max_k_dist, phoenix_vertices_max_dist, phoenix_vertices_min_dist = k_distance(
    uniformed_filtered_phoenix_vertices
)


# In[19]:


pittsburgh_vertices_max_k_dist, pittsburgh_vertices_max_dist, pittsburgh_vertices_min_dist = k_distance(
    uniformed_filtered_pittsburgh_vertices,k=106
)


# In[20]:


def get_rho(vertices_list, max_k):
    max_rho = -1
    min_rho = 1 << 30
    for node in vertices_list:
        count = 0
        for diff_node in vertices_list:
            if node is not diff_node:
                if euclid_distance(node, diff_node) <= max_k:
                    count += 1  
        node.rho = count
        if max_rho < node.rho:
            max_rho = node.rho
        if min_rho > node.rho:
            min_rho = node.rho
            
        print('=> rho', node.rho)
        print('=> max rho', max_rho)
        print('=> min rho', min_rho)
        print('-'*75)
    return max_rho, min_rho

def get_uniformed_rho(vertices_list, max_rho, min_rho):
    for node in vertices_list:
        node.rho = (node.rho - min_rho)/(max_rho - min_rho)
        print('=> uniformed rho', node.rho)
        


# In[21]:


charlotte_max_rho, charlotte_min_rho = get_rho(
    uniformed_filtered_charlotte_vertices, 
    charlotte_vertices_max_k_dist
)
# get_uniformed_rho(
#     uniformed_filtered_charlotte_vertices, 
#     charlotte_max_rho, 
#     charlotte_min_rho
# )


# In[22]:


phoenix_max_rho, phoenix_min_rho = get_rho(
    uniformed_filtered_phoenix_vertices, 
    phoenix_vertices_max_k_dist
)
get_uniformed_rho(
    uniformed_filtered_phoenix_vertices, 
    phoenix_max_rho,
    phoenix_min_rho
)


# In[23]:


pittsburgh_max_rho, pittsburgh_min_rho = get_rho(
    uniformed_filtered_pittsburgh_vertices,
    pittsburgh_vertices_max_k_dist
)
get_uniformed_rho(
    uniformed_filtered_pittsburgh_vertices,
    pittsburgh_max_rho,
    pittsburgh_min_rho
)


# In[24]:


import traceback
ZERO_DIVISION_EPSILON = 0.00001
# 获取选取潜在中心点以及选取潜在异常点的参数
def get_parameters(vertices_list, max_distance, min_distance):
    global ZERO_DIVISION_EPSILON
    
    for node in vertices_list:
        rho_above_list = filter(lambda x: x.rho >= node.rho and x is not node,
                                vertices_list)
        min_center_distance = 1 << 28

        for v in rho_above_list:
            if min_center_distance > euclid_distance(node, v):
                min_center_distance = euclid_distance(node, v)
                min_v = v

        try:
#             min_center_distance = (min_center_distance - min_distance)/(max_distance-min_distance)
            #node.center_parameter = min_center_distance * node.rho * node.checkin
            node.center_parameter = min_center_distance * node.rho
#             node.outlier_paremeter = node.rho * node.checkin / (min_center_distance+ZERO_DIVISION_EPSILON)
            node.outlier_paremeter = node.rho / min_center_distance
            print('=> center parameter {}, => outlier parameter {}'
                  .format(node.center_parameter, node.outlier_paremeter))
        except ZeroDivisionError:
            print('-' * 75)
            print('[!]', node.x, node.y)
            print('[!]', min_v.x, min_v.y)
        except Exception:
            traceback.print_exc()


# In[25]:


get_parameters(uniformed_filtered_charlotte_vertices, charlotte_vertices_max_dist, charlotte_vertices_min_dist) 


# In[26]:


get_parameters(uniformed_filtered_phoenix_vertices, phoenix_vertices_max_dist, phoenix_vertices_min_dist)


# In[27]:


get_parameters(uniformed_filtered_pittsburgh_vertices, pittsburgh_vertices_max_dist, pittsburgh_vertices_min_dist) 


# In[28]:


def add_index(vertices_list):
    for i, v in enumerate(vertices_list):
        v.index = i
        
def normal_node_get_k_potentail_center(potential_center_list, rest_node_list, max_distance, min_distance, k):
    """
        potential_center_list:潜在中心点列表
        rest_node_list: 除去潜在中心点的节点列表
        k: 普通节点要获取k个潜在中心点
    """
    print(len(rest_node_list+potential_center_list))
    for node in rest_node_list+potential_center_list:
        node_center_distance = []
        for center in potential_center_list: 
            if center is not node:
                node_center_distance.append(
                    (center, (euclid_distance(node, center)-min_distance)/(max_distance-min_distance))
                )

        node.potential_center_list = sorted(
            node_center_distance, 
            lambda x, y:cmp(x[1], y[1]),
        )[:k]
        
    
               
def write_to_file(dir_path, file_name, rest_node_list):
    global ZERO_DIVISION_EPSILON
    with open(dir_path+file_name, 'w') as f:
        for v in rest_node_list:
            for c in v.potential_center_list:
                line = map(lambda x:str(x), [v.index, c[0].index, c[0].checkin/(c[1]+ZERO_DIVISION_EPSILON)])
                f.write(' '.join(line)+'\n')

# 给各个城市的节点增加索引
add_index(uniformed_filtered_charlotte_vertices)
# add_index(uniformed_filtered_phoenix_vertices)
# add_index(uniformed_filtered_pittsburgh_vertices)


# In[45]:


# draw charlotte
y_center = []
y_outlier = []
for v in uniformed_filtered_charlotte_vertices:
    y_center.append(v.center_parameter)
    y_outlier.append(v.outlier_paremeter)

center_sorted_uniformed_filtered_charlotte_vertices = sorted(
    uniformed_filtered_charlotte_vertices, 
    lambda x, y:cmp(x.center_parameter, y.center_parameter),
    reverse=True
)

y_center = sorted(y_center, reverse=True)
plt.plot(range(len(uniformed_filtered_charlotte_vertices)), y_center, linewidth=1.0)

plt.xlim(0, 6000)
#plt.ylim(0,0.000001)
plt.show()

y_outlier = sorted(y_outlier, reverse=True)
plt.plot(range(len(uniformed_filtered_charlotte_vertices)), y_outlier, color="blue", linewidth=1.0, linestyle="-")  
plt.show()


# In[31]:


# charlotte城市中划分潜在中心点个数
charlotte_number_of_potential_center_nodes = 4630
charlotte_potential_center_list = center_sorted_uniformed_filtered_charlotte_vertices[
    :charlotte_number_of_potential_center_nodes]
charlotte_rest_node_list = center_sorted_uniformed_filtered_charlotte_vertices[
    charlotte_number_of_potential_center_nodes:]
           
normal_node_get_k_potentail_center(
    charlotte_potential_center_list, 
    charlotte_rest_node_list, 
    charlotte_vertices_max_dist,
    charlotte_vertices_min_dist
)
write_to_file(result_dir_path, 'charlotte_result_8', charlotte_rest_node_list+charlotte_potential_center_list)


# In[ ]:


# draw phoenix
y_center = []
y_outlier = []
for v in uniformed_filtered_phoenix_vertices:
    y_center.append(v.center_parameter)
    y_outlier.append(v.outlier_paremeter)


center_sorted_uniformed_filtered_phoenix_vertices = sorted(
    uniformed_filtered_phoenix_vertices, 
    lambda x, y:cmp(x.center_parameter, y.center_parameter),
    reverse=True
)

y_center = sorted(y_center, reverse=True)
plt.plot(range(len(uniformed_filtered_phoenix_vertices)), y_center, linewidth=1.0, linestyle="-")  
plt.xlim(0, 6000)
plt.show()

y_outlier = sorted(y_outlier, reverse=True)
plt.plot(range(len(uniformed_filtered_phoenix_vertices)), y_outlier, color="blue", linewidth=1.0, linestyle="-")  
plt.show()


# In[33]:


# phoenix城市中划分潜在中心点个数
phoenix_number_of_potential_center_nodes = 100
phoenix_potential_center_list = center_sorted_uniformed_filtered_phoenix_vertices[
    :phoenix_number_of_potential_center_nodes]
phoenix_rest_node_list = center_sorted_uniformed_filtered_phoenix_vertices[
    phoenix_number_of_potential_center_nodes:]
           
normal_node_get_k_potentail_center(
    phoenix_potential_center_list, 
    phoenix_rest_node_list,
    phoenix_vertices_max_dist,
    phoenix_vertices_min_dist
)
write_to_file(result_dir_path, 'phoenix_result', phoenix_rest_node_list)


# In[46]:


# draw pittsburgh
y_center = []
y_outlier = []
for v in uniformed_filtered_pittsburgh_vertices:
    y_center.append(v.center_parameter)
    y_outlier.append(v.outlier_paremeter)

    
center_sorted_uniformed_filtered_pittsburgh_vertices = sorted(
    uniformed_filtered_pittsburgh_vertices, 
    lambda x, y:cmp(x.center_parameter, y.center_parameter),
    reverse=True
)

y_center = sorted(y_center, reverse=True)

plt.plot(range(len(uniformed_filtered_pittsburgh_vertices)), y_center, linewidth=1.0, linestyle="-")
plt.xlim(0, 5000)
plt.ylim(0,0.0000005)
plt.show()

y_outlier = sorted(y_outlier, reverse=True)
plt.plot(range(len(uniformed_filtered_pittsburgh_vertices)), y_outlier, color="blue", linewidth=1.0, linestyle="-")  
plt.show() 


# In[114]:


# pittsburgh城市中划分潜在中心点个数
pittsburgh_number_of_potential_center_nodes = 3500
pittsburgh_potential_center_list = center_sorted_uniformed_filtered_pittsburgh_vertices[
    : pittsburgh_number_of_potential_center_nodes]
pittsburgh_rest_node_list = center_sorted_uniformed_filtered_pittsburgh_vertices[
    pittsburgh_number_of_potential_center_nodes:]


normal_node_get_k_potentail_center(
    pittsburgh_potential_center_list, 
    pittsburgh_rest_node_list,
    pittsburgh_vertices_max_dist,
    pittsburgh_vertices_min_dist
)
write_to_file(dir_path, 'pittsburgh_result', pittsburgh_rest_node_list)

