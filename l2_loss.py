def l2_loss(arr_1 : In[Array[(Array[(Array[(float, 3)], 200)], 201)]], arr_2 : In[Array[(Array[(Array[(float, 3)], 200)], 201)]], height : In[int], width : In[int], color : In[int]) -> float:
    i : int = 0
    j : int = 0
    k : int = 0
    s : float = 0.0
    while (i < height, max_iter := 200):
        j = 0
        while (j < width, max_iter := 200):
            k = 0
            while (k < color, max_iter := 3):
                s = s + pow(arr_1[i][j][k] - arr_2[i][j][k], 2.0)
                k = k + 1
            j = j + 1
        i = i + 1
    return s / (height * width * 3)

def pyramid_l2_loss(arr_1 : In[Array[(Array[(Array[(float, 3)], 200)], 201)]], arr_2 : In[Array[(Array[(Array[(float, 3)], 200)], 201)]], height : In[int], width : In[int], color : In[int], pyramid_para: In[Array[(int, 10)]], pyramid_para_size: In[int]) -> float:
    i : int = 0
    j : int = 0
    k : int = 0
    t : int = 0
    x : int = 0
    y : int = 0
    count : int = 0
    s : float = 0.0
    arr_1_ : Array[(Array[(Array[(float, 3)], 200)], 201)]
    while (t < pyramid_para_size, max_iter := 1):
        k = 0
        while (k < color, max_iter := 3):
            i = 0
            while (i < height, max_iter := 201):
                j = 0
                while (j < width, max_iter := 200):

                    count = 0
                    arr_1_[i][j][k] = 0
                    x = i - pyramid_para[t]
                    while (x <= i + pyramid_para[t], max_iter := 10):
                        y = j - pyramid_para[t]
                        while (y <= j + pyramid_para[t], max_iter := 10):
                            if (x >= 0 and x < height) and (y >= 0 and y < width):
                                arr_1_[i][j][k] = arr_1_[i][j][k] + arr_1[x][y][k]
                                count = count + 1
                            y = y + 1
                        x = x + 1
                    arr_1_[i][j][k] = arr_1_[i][j][k] / count

                    j = j + 1
                i = i + 1
            k = k + 1
        s = s + l2_loss(arr_1_, arr_2, height, width, color)
        t = t + 1
    return s

d_l2_loss = rev_diff(l2_loss)
